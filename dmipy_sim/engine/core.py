"""The simulation entry points.

``simulate()`` walks the substrate to coefficients (:func:`simulate_trajectories`: per walker the exact endpoints and
the sine-band projections of its path, accumulated as the walk runs) and contracts them with the waveform
(:meth:`~dmipy_sim.replay.replay.ReplayPack.replay`). The producers return a
:class:`~dmipy_sim.coefficient_walk.CoefficientWalk`; ``walk_cost`` prices one before it runs. The single-pass
integrator of the phase along the walk is the oracle (:mod:`dmipy_sim.engine._fused`), not an entry point.
"""

import jax
import jax.numpy as jnp
import contextlib
import dataclasses
from dataclasses import dataclass
from typing import NamedTuple
import numpy as np

from .backends import JAX, WalkRequest, CoefficientBatch, event_capacity, resolve as resolve_backend
import logging

log = logging.getLogger(__name__)

from ..acquisition.waveforms import band_hz
from .accumulate import default_K
from .physics import (make_myelin_traj_step_fn, make_packed_myelin_traj_step_fn,
                      seed_walkers, pair_step)
from ..run import Run
from ..acquisition.rf import RFSchedule
from ..acquisition.scanner_sequence import Protocol


# ═══════════════════════════════════════════════════════════════════════════
# ROUTING
# ═══════════════════════════════════════════════════════════════════════════
# simulate() walks once (simulate_trajectories -> CoefficientWalk), then contracts the walk's bands with the
# waveform, the scalar / per-compartment T2, T1 and the wall's surface relaxivity (ReplayPack.replay), when the
# geometry declares `replay_parity` (the producer's walk IS the fused walk, validated in
# tests/test_replay_parity.py) and _replay_gap() finds nothing the producer cannot walk (a dedicated fused step
# kernel, per-compartment D, a missing diffusivity); otherwise it runs the fused oracle (engine/_fused.py). The
# capability is declared on the geometry, never read off its name.

_BATCH_CACHE_ATTR = "_batch_cache"   # per-geometry {key: jitted batch function}, on the object


@contextlib.contextmanager
def _bounce_budget_override(geometry, bounce_budget):
    """``geometry.bounce_budget`` set to ``bounce_budget`` for the walk and restored after it. Touches
    ``geometry.bounce_loop`` once here, on the host, so a geometry that builds a kernel per budget (the packs) builds
    it outside any trace."""
    if bounce_budget is None:
        yield
        return
    if geometry.bounce_loop is None:
        raise ValueError(f"{type(geometry).__name__} has no bounce loop to size: bounce_budget= does not apply")
    if int(bounce_budget) < 1:
        raise ValueError(f"bounce_budget must be at least one collision per step, got {bounce_budget!r}")
    had, prev = "bounce_budget" in vars(geometry), geometry.bounce_budget
    geometry.bounce_budget = int(bounce_budget)
    geometry.bounce_loop                                 # built now, on the host
    try:
        yield
    finally:
        if had:
            geometry.bounce_budget = prev
        else:
            del geometry.bounce_budget


def _geometry_state(geometry):
    """The scalar state of a geometry that a step function bakes in at trace time.

    Device arrays are fixed at construction; the knobs a caller may set afterwards
    (``reflect_mode``, ``_GRAZE``, ``adaptive_nudge``, ...) are Python scalars, strings, bools
    or 0-d arrays. Their values are part of the cache key, so changing one after a walk builds a
    new program rather than replaying the old one.
    """
    items = []
    for k, v in sorted(vars(geometry).items()):
        if k == _BATCH_CACHE_ATTR:
            continue
        if isinstance(v, (bool, int, float, str, type(None))):
            items.append((k, v))
        elif isinstance(v, (np.ndarray, jnp.ndarray)) and v.ndim == 0:
            items.append((k, float(v)))
        elif isinstance(v, tuple) and all(isinstance(x, (bool, int, float, str)) for x in v):
            items.append((k, v))
    return tuple(items)


def cached_batch(geometry, key, build):
    """One compiled program per (geometry, configuration, shape), across calls.

    ``simulate`` used to build its scan closure and dispatch it eagerly on every call, so a
    second call on the same geometry paid a full compile again (measured 4.8-6.4 s against a
    20 ms kernel run). The jitted batch function is kept ON the geometry object (the closure
    references the geometry, so a side table keyed by it would pin it forever; on the object the
    reference is a cycle the garbage collector frees with the geometry), keyed on its scalar
    state and ``key`` -- everything the closure bakes in that is not a traced argument. ScannerSequence
    samples, walker positions, keys and labels are arguments, so a sweep over b, seed or direction
    recompiles nothing; a new walker count retraces inside ``jax.jit`` by shape, once.
    """
    try:
        per_geom = vars(geometry).setdefault(_BATCH_CACHE_ATTR, {})
    except TypeError:                      # no instance dict: no caching, same result
        return build()
    full = (key, _geometry_state(geometry))
    fn = per_geom.get(full)
    if fn is None:
        fn = build()
        per_geom[full] = fn
    return fn


def _replay_gap(geometry, diffusivity):
    """Why the coefficient producer cannot walk this run, or ``None`` when it can: a dedicated fused step kernel
    (the myelinated cylinders), per-pool diffusivity (that changes the walk itself, so it is not a replay knob) or
    no diffusivity (the producer sizes its step from it). Per-pool ``T2`` / ``T1`` ARE replay knobs: they gate the
    occupancy the walk records. :func:`simulate` serves a geometry from its coefficient walk when this is ``None``
    and the geometry declares ``replay_parity`` (its producer's walk IS the fused walk, tests/test_replay_parity.py)."""
    if geometry._is_myelinated:
        return "MyelinatedCylinder uses a dedicated fused step kernel (no replay walk)"
    if geometry._is_packed_myelinated:
        return ("PackedMyelinatedCylinders fused single-reflection kernel is not "
                "position-parity with the multi-bounce replay walk")
    if geometry._D_comp_jax is not None:
        return "per-compartment diffusivity (Mesh intra/extra D) alters the walk (fused-only)"
    if diffusivity is None:
        return "replay needs an explicit diffusivity for the sub-step auto-tune"
    return None


def _walk_band(G, dt, T):
    """``(K, f_band)`` of a walk that serves the waveform ``G`` (``(n_meas, n_t, 3)`` on a grid of ``dt``) over ``T``:
    the band the waveform reads (:func:`~dmipy_sim.acquisition.waveforms.band_hz`) sets the step law's band and the
    bands ``ceil(2 f T)`` the walk accumulates, never fewer than the storage default's."""
    f = band_hz(G, dt)
    if f is None:
        return None, None
    return max(default_K(T), int(np.ceil(2.0 * f * T - 1e-9))), float(f)


def _walk_pack(n_walkers, diffusivity, waveform, geometry, *, seed, T2, T1, r0, require_gpu, walker_batch_size,
               sub_steps=None):
    """``(pack, tissue, pose)``: the coefficient walk of ``geometry`` over the span of ``waveform`` (one storage
    window, the band the waveform reads), assembled into an in-memory :class:`~dmipy_sim.replay.replay.ReplayPack`,
    with the :class:`~dmipy_sim.spec.tissue.Tissue` (scalar or per-pool ``T2`` / ``T1``, the wall's ``rho2``) and the
    pose (the geometry's placement in the bore) a replay of ``waveform`` reads. The walk depends only on (geometry,
    diffusivity, seed); the waveform and the relaxation are replay settings."""
    from ..replay.bank import build_replay_pack
    from ..spec.tissue import Tissue

    G = np.asarray(waveform.G_eff, dtype=np.float64)
    dt = float(waveform.dt)
    T_max = dt * (G.shape[1] - 1)
    K, f_band = _walk_band(G, dt, T_max)

    rho = geometry.surface_relaxivity_t2
    has_surf = rho is not None and float(rho) > 0.0 and geometry.has_walls
    T2_comp, T1_comp = geometry._T2_comp, geometry._T1_comp
    relaxes = has_surf or T2 is not None or T1 is not None or T2_comp is not None or T1_comp is not None

    kw = dict(seed=seed, require_gpu=require_gpu, tiers=("all" if relaxes else ()), K=K, f_band=f_band, window_T=T_max)
    if r0 is not None:
        kw['r0'] = r0
    if walker_batch_size is not None:
        kw['walker_batch_size'] = walker_batch_size
    if sub_steps:
        kw['sub_steps'] = sub_steps
    walk = simulate_trajectories(n_walkers, diffusivity, geometry, T_max, **kw)
    pack = build_replay_pack(walk, id="simulate", license="in-memory", citation="simulate")

    pools = sorted(pack.substrate.pools, key=lambda p: p.id)

    def per_pool(scalar, by_id):
        """The ``{pool name: seconds}`` of the geometry's per-pool times when it has them, else of the one number."""
        if by_id is not None:
            return {p.name: float(by_id[p.id]) for p in pools}
        return None if scalar is None else {p.name: float(scalar) for p in pools}

    tissue = Tissue(T2=per_pool(T2, T2_comp), T1=per_pool(T1, T1_comp), rho2=(float(rho) if has_surf else None))
    return pack, tissue, geometry._orient_R


def _replay_signal(pack, waveform, tissue, pose):
    """The real part of the ensemble signal of ``waveform`` on ``pack`` as the walk's mean ``Re<exp(i phi)>``,
    ``float32`` of shape ``(n_measurements,)``."""
    S = pack.replay(waveform, tissue=tissue, orientation=pose, complex_signal=True)
    return np.real(np.asarray(S)).astype(np.float32)


def simulate(
    n_walkers: int,
    diffusivity=None,
    waveform=None,
    geometry=None,
    seed: int = 123,
    T2: float = None,
    T1: float = None,
    r0=None,
    walker_batch_size: int = None,
    require_gpu=None,
    sub_steps: int = None,
):
    """The signal of ``waveform`` on ``geometry``: the substrate walked to coefficients, the waveform contracted
    with them.

    The walk is :func:`simulate_trajectories` over the waveform's span at the band the waveform reads
    (:func:`~dmipy_sim.acquisition.waveforms.band_hz`); the contraction is
    :meth:`~dmipy_sim.replay.replay.ReplayPack.replay` on the pack of that walk. The walk depends only on
    (geometry, diffusivity, seed), so the waveform, ``T2``, ``T1`` and the wall's surface relaxivity are replay
    settings of one walk. A run the coefficient walk is not validated to serve (a geometry that does not declare
    ``replay_parity``, or :func:`_replay_gap`: a dedicated fused step kernel, per-pool diffusivity, no diffusivity)
    runs on the fused oracle (:func:`~dmipy_sim.engine._fused.simulate_fused`), which integrates the phase along the walk.

    Parameters
    ----------
    n_walkers : int
        Number of random walkers.
    diffusivity : float, optional
        Diffusion coefficient in m²/s. Required for standard geometries.
        Omit for MyelinatedCylinder (D values are in the geometry).
    waveform : ScannerSequence
        The acquisition: a :class:`~dmipy_sim.acquisition.scanner_sequence.ScannerSequence`, a multi-TE
        :class:`~dmipy_sim.acquisition.scanner_sequence.Protocol` (one walk per echo time, the signals placed at
        their rows) or anything with a ``.waveform``. Its effective gradient has shape (n_measurements, n_t, 3).
    geometry : Geometry
        The substrate: a geometry, a :class:`~dmipy_sim.spec.SubstrateSpec`, its dict or a ``.sub.json`` path.
    seed : int
        Master PRNG seed (split into per-walker keys internally).
    T2 : float, optional
        Transverse relaxation time in seconds, in every pool, gated by the waveform's transverse coherence
        (1 transverse, 0 stored longitudinally). A plain spin echo has coherence 1 throughout. A per-pool
        ``T2`` on the geometry replaces it.
    T1 : float, optional
        Longitudinal relaxation time in seconds, acting through the coherence's complement: the stored
        (``chi_perp == 0``) block of a stimulated echo, e.g. the mixing time of a PGSTE. A spin echo never
        reaches it.
    r0 : array-like of shape (n_walkers, 3), optional
        Custom initial walker positions in metres (lab frame, float32), in place of ``geometry.init_positions()``:
        mixed initial conditions (f·N walkers inside cylinders, (1-f)·N outside, as the Kärger-model validation
        requires). Compartment ids follow one convention: 0 is the extra-cellular / free pool and enclosed
        pools are positive (``Cylinder``, ``Sphere``, ``Ellipsoid``, ``Mesh``: 1 = intra; ``Box1D``: 1 = the slab;
        ``PackedCylinders``, ``PackedSpheres``: 1..N the object the walker is in).
    walker_batch_size : int, optional
        If set and smaller than ``n_walkers``, the walk runs in walker chunks of this size and the coefficients
        are joined: peak device memory is one chunk — use this on a small GPU.
    require_gpu : {None, True, False}, optional
        GPU guard against a silent CPU fallback.  ``True`` raises if no GPU is
        visible; ``False`` opts out (e.g. a CPU float64 reference check);
        ``None`` (default) warns when a large run is about to use the CPU.
    sub_steps : int, optional
        Fine sub-steps per step; overrides the per-geometry auto-tune.

    Returns
    -------
    signals : np.ndarray of shape (n_measurements,), float32
        Normalised signal: Re(<exp(i·phi)>) averaged over walkers.
    """
    with Run("simulate", params=dict(n_walkers=n_walkers, diffusivity=diffusivity, geometry=type(geometry).__name__)):
        from ..spec.build import as_geometry
        geometry = as_geometry(geometry)               # a spec, a spec file or a dict is a substrate too
        _acq = waveform.waveform if hasattr(waveform, 'waveform') else waveform
        if isinstance(_acq, Protocol):
            # a multi-TE scheme: one sequence per echo time, one walk each, the signals placed at their rows
            kw = dict(seed=seed, T2=T2, T1=T1, r0=r0, walker_batch_size=walker_batch_size, require_gpu=require_gpu,
                      sub_steps=sub_steps)
            return _acq.scatter([simulate(n_walkers, diffusivity, seq, geometry, **kw) for seq in _acq])
        if geometry.replay_parity and _replay_gap(geometry, diffusivity) is None:
            pack, tissue, pose = _walk_pack(n_walkers, diffusivity, _acq, geometry, seed=seed, T2=T2, T1=T1, r0=r0,
                                            require_gpu=require_gpu, walker_batch_size=walker_batch_size,
                                            sub_steps=sub_steps)
            return _replay_signal(pack, _acq, tissue, pose)
        from ._fused import simulate_fused
        return simulate_fused(n_walkers, diffusivity, _acq, geometry, seed=seed, T2=T2, T1=T1, r0=r0,
                              walker_batch_size=walker_batch_size, require_gpu=require_gpu, sub_steps=sub_steps)


def simulate_mixture(compartments, waveform, seed=123, sub_steps=None):
    """Run a two- (or multi-) compartment simulation with no exchange.

    Each compartment is simulated independently; the final signal is the
    volume-fraction-weighted sum.

    Parameters
    ----------
    compartments : list of dicts, each with keys:
        - 'fraction'     : float, volume fraction (must sum to 1).
        - 'n_walkers'    : int, walkers for this compartment.
        - 'diffusivity'  : float, D in m²/s.
        - 'geometry'     : Geometry instance.
    waveform : ScannerSequence
    seed : int
        Base seed; each compartment gets seed + compartment_index.

    Returns
    -------
    signals : np.ndarray of shape (n_measurements,), float32
    """
    total = sum(c['fraction'] for c in compartments)
    if abs(total - 1.0) > 1e-6:
        raise ValueError(f"Compartment fractions must sum to 1, got {total}")

    signal = None
    for i, comp in enumerate(compartments):
        s = simulate(
            n_walkers=comp['n_walkers'],
            diffusivity=comp['diffusivity'],
            waveform=waveform,
            geometry=comp['geometry'],
            seed=seed + i,
            sub_steps=sub_steps,
        )
        weighted = comp['fraction'] * s
        signal = weighted if signal is None else signal + weighted

    return signal


def simulate_cpmg(n_walkers, diffusivity, waveform, geometry, *,
                  T2=None, seed=123, r0=None, sub_steps=None,
                  walker_batch_size=None, require_gpu=None):
    """Multi-echo CPMG signal from a SINGLE diffusion walk.

    One coefficient walk over the whole train, one replay per echo: the signal at echo ``k`` is the replay of the
    train up to that echo (its gradient and its refocusing pulses to the echo time), so the relaxation each echo
    has seen is the time elapsed to it. Build ``waveform`` with :func:`dmipy_sim.cpmg` (whose ``readout`` is
    every echo). A geometry the coefficient walk cannot serve (:func:`_replay_gap`) is refused: the multi-echo
    readout has no fused counterpart.

    Parameters
    ----------
    n_walkers : int
    diffusivity : float
        Bulk diffusivity (m²/s).
    waveform : ScannerSequence
        A multi-echo sequence, its ``readout`` every echo (e.g. from ``cpmg``).
    geometry : Geometry
    T2 : float, optional
        Transverse relaxation time (s) in every pool.
    r0 : array-like of shape (n_walkers, 3), optional
        Explicit start positions in metres.  Default: ``geometry.init_positions(n, key)``,
        which on a mesh means ``intra=True`` -- INSIDE the surface.  Pass this whenever the
        pool you want is not the geometry's inside; a fibre bundle's extra-axonal pool is
        the case that occurs, and getting it wrong silently walks the intra pool instead.
        See :func:`dmipy_sim.geometry.initial_positions`.
    sub_steps : int, optional
        Fine sub-steps per step; overrides the per-geometry auto-tune.
    seed, walker_batch_size, require_gpu : see :func:`simulate`.

    Returns
    -------
    signals : np.ndarray, shape (n_echoes, n_measurements), float32
        Signal at each echo (echo k = k·TE), one column per gradient direction.
    """
    with Run("simulate_cpmg", params=dict(n_walkers=n_walkers, diffusivity=diffusivity, geometry=type(geometry).__name__)):
        from ..spec.build import as_geometry
        geometry = as_geometry(geometry)               # a spec, a spec file or a dict is a substrate too
        if hasattr(waveform, 'waveform'):
            waveform = waveform.waveform
        echo_indices = [int(i) for i in waveform.readout]
        if len(echo_indices) < 2:
            raise ValueError("simulate_cpmg needs a multi-echo readout: build the train with cpmg(...)")
        gap = _replay_gap(geometry, diffusivity)
        if gap is not None:
            raise NotImplementedError(f"simulate_cpmg replays one coefficient walk at every echo, and cannot serve this run: {gap}")
        pack, tissue, pose = _walk_pack(n_walkers, diffusivity, waveform, geometry, seed=seed, T2=T2, T1=None, r0=r0,
                                        require_gpu=require_gpu, walker_batch_size=walker_batch_size,
                                        sub_steps=sub_steps)
        return np.stack([_replay_signal(pack, _until(waveform, i), tissue, pose) for i in echo_indices])


def _until(waveform, i):
    """``waveform`` up to its sample ``i``: the gradient to there and the RF events at or before it, read at
    that sample (a train cut at one of its echoes)."""
    t = i * float(waveform.dt)
    rf = [e for e in waveform.rf if e.t_s <= t + 1e-12]
    return dataclasses.replace(waveform, G=waveform.G[:, :i + 1], rf=RFSchedule(rf), readout=None)



def _record_channels(tiers):
    """Whether a walk records its surface / compartment channels: ``"all"`` yes, ``()`` no."""
    if tiers == "all":
        return True
    if tiers is None or (isinstance(tiers, (tuple, list, set, frozenset)) and len(tiers) == 0):
        return False
    raise ValueError(f"tiers must be 'all' (record every channel the geometry supports) or () "
                     f"(positions only), got {tiers!r}")

class _GridRead:
    """The field basis channels at a walker's position, read from a :class:`FieldGrid` passed to the walk as device
    arrays ``(grids, origin, voxel_size)`` (:meth:`FieldGrid.device_arrays`): ``read(r) -> (n_ch,)``."""

    def __init__(self, arrays, periodic):
        self.arrays, self.periodic, self.n_ch = arrays, periodic, int(arrays[0].shape[0])

    def __call__(self, r):
        from ..fields.susceptibility_field import grid_channels_at
        return grid_channels_at(*self.arrays, self.periodic, r)


class _Rows:
    """The rows of one walk channel, written into a single preallocated array as the walker batches arrive.

    The host holds the channel once. A list of batches concatenated at the end held it twice, the list beside
    its concatenation, at the moment the walk is largest (dmrai-lab/dmipy-sim#438).
    """

    def __init__(self, n_rows):
        self.n_rows = int(n_rows); self.array_ = None; self.filled = 0

    def append(self, rows):
        rows = np.asarray(rows)
        if self.array_ is None:
            self.array_ = np.empty((self.n_rows,) + rows.shape[1:], rows.dtype)
        self.array_[self.filled:self.filled + rows.shape[0]] = rows
        self.filled += rows.shape[0]

    def array(self):
        if self.array_ is None:
            return np.empty((0,), np.float32)
        return self.array_ if self.filled == self.n_rows else self.array_[:self.filled]


DEFAULT_WALKER_BATCH = 50_000
"""Walkers per batch of :func:`simulate_trajectories` unless the caller says otherwise."""

#: the length each sub-step rule divides, as the :class:`~dmipy_sim.geometry.base.LengthScales` field it reads;
#: ``None`` where the rule's scale is not a length (the crossing rule reads the permeability, the binding rule the dwell time)
_RULE_LENGTH = {"step law": "min_feature", "regime": "min_feature", "collision lookup": "lookup_cell",
                "surface local time": "surface_pore", "crossing": None, "binding": None}


class WalkResolution(NamedTuple):
    """What a trajectory walk resolves from its request before it walks: the blocks, the sub-step count and the
    rules that gave it, the field read and the :class:`~dmipy_sim.engine.backends.WalkRequest` a backend is handed."""
    record: bool              #: the surface / compartment channels are recorded beside the positions
    n_blocks: int             #: blocks over the walk
    block_h: float            #: the block length ``h`` (s)
    f_band: float             #: the band (Hz) the step law resolves
    rules: dict               #: the sub-step count each applicable rule asks for, by name
    override: object          #: the caller's pinned sub-step count, or None
    sub_steps: int            #: physics steps per block
    dt_sim: float             #: seconds per physics step
    step_rule: dict           #: ``CoefficientWalk.work["step_rule"]`` without ``dt_p`` / ``sub_steps``
    binding: str              #: the rule that sized the step (``"override"`` for a pinned count)
    length_scales: object     #: the geometry's :class:`~dmipy_sim.geometry.base.LengthScales`
    request: object           #: the :class:`~dmipy_sim.engine.backends.WalkRequest` of this walk
    jax_only: list            #: what this walk uses that only the JAX kernels implement
    K: int = None             #: the bands the walk accumulates per axis and channel, per window
    block_order: int = 3      #: the order ``m`` of the block table
    block_rule: str = None    #: how the block was set: ``"derived"`` (omega_K h = BLOCK_OMEGA_H) or ``"given"`` (``block_h``)
    n_windows: int = 1        #: the storage windows the walk accumulates separately
    K_c3: object = None       #: the field's band (modes per window), or None for "a pilot derives it"
    field_read: object = None #: the field read's record (``dict``: rule, ``dt_f``, ``n_read``, the term's inputs), or None without a field
    crossing: bool = False    #: the walkers can change pool (the occupancy is an event list)
    binding_mt: bool = False  #: the walk binds (an MT bound column)
    events_capacity: object = None  #: the events buffer per walker and column, or None for "a pilot sizes it"


def _resolve_walk(geometry, diffusivity, T_max, block_h=None, *, tiers="all", sub_steps=None, kappa_MT=0.0, dwell_time=0.0,
                  f_band=None, seed=42, enforce_compartment=False, field_basis=None, K=None, block_order=None, window_T=None,
                  K_c3=None, field_read_interval=None, events_capacity=None, raw_subset=0, envelope=None):
    """The resolution every walk shares: its storage windows, its band ``K`` per window, its blocks (the band's own,
    :func:`~dmipy_sim.engine.accumulate.block_length`, unless ``block_h`` gives one), the sub-step count per block
    (``physics.resolve_sub_steps`` over the rules ``physics.sub_step_rules`` at the band ``K / (2 T)``), the field
    read's interval (a given one at the largest divisor of the step count within it, else the step: the pilot's read,
    which :func:`simulate_trajectories` replaces by the read the pilot derives,
    :func:`~dmipy_sim.engine.accumulate.field_read_of_pilot`), the geometries this walk cannot represent
    (refused), and the :class:`WalkResolution` a backend is handed. :func:`simulate_trajectories` walks what this
    returns and :func:`walk_cost` reports it, so a cost and a walk state one count."""
    from .accumulate import (block_length, BLOCK_ORDER, default_K, window_count, field_rule_scale, field_read_of_pilot,
                             FIELD_READ_FLOOR)
    record = _record_channels(tiers)
    n_windows, T_win = window_count(T_max, window_T)
    K = default_K(T_win) if K is None else int(K)
    if K < 1:
        raise ValueError(f"K is the number of bands a walk accumulates per window, at least 1; got {K}")
    if block_h is None:
        h, block_rule = block_length(T_win, K), "derived"     # omega_K h = BLOCK_OMEGA_H at the top band (#740, cuda#74)
    else:
        if float(block_h) > float(T_win) * (1.0 + 1e-9):
            raise ValueError(f"block_h = {float(block_h):.6g} s is longer than the storage window of {float(T_win):.6g} s")
        h, block_rule = float(T_win) / max(int(round(float(T_win) / float(block_h))), 1), "given"
    n_bw = int(round(float(T_win) / h))
    n_blocks = n_windows * n_bw
    dt_actual = float(T_win) / n_bw
    block_order = BLOCK_ORDER if block_order is None else int(block_order)
    if not 0 <= block_order <= 3:
        raise ValueError(f"block_order is the order m of the block table, 0..3; got {block_order}")
    if block_order == 0 and K > n_bw - 1:
        raise ValueError(f"at block_order=0 a window's bands are read from its {n_bw} blocks, which hold {n_bw - 1} "
                         f"bands; K={K} asks for more")

    # A walk that records the boundary local time is resolved at the surface criterion too, so a replayed rho
    # reproduces the fused engine's; MT binding adds its own.
    from .physics import (resolve_sub_steps as _resolve_sub_steps, sub_step_rules as _sub_step_rules,
                          length_scales_of as _length_scales_of, STEP_TOLERANCE)
    ls = _length_scales_of(geometry)
    # the band the step law resolves: the caller's (walk_spec passes the scanner class's), else the walk's own K / (2 T)
    f_band = float(K / (2.0 * float(T_win)) if f_band is None else f_band)
    mt_dwell = dwell_time if kappa_MT > 0.0 else None
    rules = _sub_step_rules(geometry, diffusivity, dt_actual, surface=bool(record), mt_dwell_time=mt_dwell, f_band=f_band)
    override = None if sub_steps is None else int(sub_steps)
    n_sub = _resolve_sub_steps(geometry, diffusivity, dt_actual, surface=bool(record), mt_dwell_time=mt_dwell,
                               override=sub_steps, f_band=f_band)
    binding = "override" if override else max(rules.items(), key=lambda kv: kv[1])[0]
    # what sized the step (CoefficientWalk.work["step_rule"]): the count each rule asked for, the tolerance, the
    # band, and the caller's override when there was one
    step_rule = dict(eps_D=float(STEP_TOLERANCE), f_band_hz=f_band, rules=dict(rules), override=override)
    dt_sim = dt_actual / n_sub
    # the soundness bound of physics.make_step_fn, here too because a permeable mesh routes through the REPLAY
    # backend (see _replay_gap) and never reaches make_step_fn
    from .physics import _warn_if_step_outruns_the_lookup as _warn_step
    _warn_step(geometry, diffusivity, dt_actual, n_sub, "trajectory walk")

    # Reject geometries whose boundaries this path cannot represent. The one source of truth is the geometry's own
    # `trajectory_walk_gap` (never a class-name list), which `spec.walk.walk_spec` also checks before the walk.
    gap = geometry.trajectory_walk_gap()
    if gap is not None:
        raise NotImplementedError(f"simulate_trajectories cannot walk a {type(geometry).__name__}: {gap}")
    if geometry._is_packed_myelinated and not record:
        raise NotImplementedError(
            "simulate_trajectories on PackedMyelinatedCylinders requires the recorded channels "
            "(tiers='all'), which route to the fused kernel "
            "physics.make_packed_myelin_traj_step_fn. The position-only path has no "
            "compartment state and would return an unrestricted walk.")

    permeability = geometry.permeability
    # the field read: the caller's interval, or the pilot's (RPK.md 9.5.5) -- until the pilot has measured the read
    # series, the read is the step (the pilot's own, and the cost's bound)
    field_read = None
    if field_basis is not None:
        B0, chi = field_rule_scale(geometry.spec, envelope)
        if field_read_interval is not None and not float(field_read_interval) > 0.0:
            raise ValueError(f"field_read_interval is the interval (s) between the walk's reads of the field, positive; got {field_read_interval!r}")
        field_read = field_read_of_pilot({}, B0=B0, chi=chi, step_s=dt_sim, steps=n_sub, T_e=float(T_win), floor=FIELD_READ_FLOOR,
                                         given=(dt_sim if field_read_interval is None else float(field_read_interval)))
        if field_read_interval is None:
            field_read["rule"] = "pilot"
    # which occupancy the walk keeps: events where a walker can change pool (a permeable wall, a binding walk)
    crossing = bool(record) and (bool(permeability is not None and float(permeability) > 0.0)
                                 or (geometry._is_myelinated or geometry._is_packed_myelinated) and (
                                     bool(np.any(np.asarray(geometry._kappa_inner_jax) > 0)) or bool(np.any(np.asarray(geometry._kappa_outer_jax) > 0))))
    binding_mt = bool(record) and kappa_MT > 0.0
    jax_only = [what for what, on in (
        ("the myelinated substrates", geometry._is_packed_myelinated or geometry._is_myelinated),
        ("MT binding", kappa_MT > 0.0), ("in-walk field sampling", field_basis is not None),
        ("several storage windows (a backend accumulates one window per walk)", n_windows > 1),
        ("the compartment guard", bool(enforce_compartment))) if on]
    # the C1 / bound event capacity a backend is handed before the pilot sized the buffer: the crossing rule's bound
    # (backends.event_capacity); the walk's batches carry the pilot's capacity (or the caller's), an impermeable
    # walk records none
    events = (int(events_capacity) if events_capacity is not None
              else event_capacity(int(n_blocks) * int(n_sub)) if (crossing or binding_mt) else 0)
    request = WalkRequest(
        geometry=geometry, block_h=float(dt_actual), n_blocks=int(n_blocks), sub_steps=int(n_sub), dt_sim=float(dt_sim),
        K=int(K), m=int(block_order), K_c3=(0 if K_c3 is None else int(K_c3)),
        field_read_interval=(0.0 if field_read is None else float(field_read["dt_f"])),
        events=int(events), raw_subset=int(raw_subset),
        diffusivity=float(diffusivity), record=bool(record),
        kappa_over_D=(float(permeability) / float(diffusivity) if permeability is not None else 0.0),
        count_walls=bool(geometry.count_walls), bounce_budget=geometry.bounce_budget, seed=int(seed))
    return WalkResolution(record, n_blocks, dt_actual, f_band, dict(rules), override, int(n_sub), dt_sim, step_rule, binding,
                          ls, request, jax_only, K, block_order, block_rule, n_windows,
                          (None if K_c3 is None else int(K_c3)), field_read, crossing, binding_mt,
                          (None if events_capacity is None else int(events_capacity)))


def _backend_refusal(backend, name, res, geometry):
    """Why ``backend`` (named ``name``) does not walk the resolved request ``res``, or None: the paths only the JAX
    kernels implement are refused for any other backend rather than fallen back from, and the backend may refuse
    the walk itself."""
    if res.jax_only:
        return (f"backend {name!r}: this walk uses {', '.join(res.jax_only)}, which only the jax kernels "
                f"implement; it does not fall back")
    reason = backend.refuses(res.request)
    if reason:
        return f"backend {name!r} refuses this walk of {type(geometry).__name__}: {reason}"
    return None


@dataclass(frozen=True)
class WalkMeasured:
    """A walk's cost measured on the real device and backend by :func:`walk_cost` (``measure=True``): the pilot timed at
    two lengths, so that the per-call part (seeding, setup, the transfer to the host) is separated from the part that
    grows with the blocks."""
    batch: int                        #: walkers of the pilot, the walk's real batch
    pilot_blocks: tuple                #: ``(n1, n2)``: the block counts of the two pilot lengths
    seconds_per_block: float           #: ``(t(n2) - t(n1)) / (n2 - n1)``, each ``t`` the second call of its shape (compile excluded)
    fixed_seconds_per_call: float     #: ``t(n1) - seconds_per_block``: what a call costs whatever its blocks
    seconds_per_walker_step: float    #: ``seconds_per_block / (batch * sub_steps)``
    projected_s: float                #: the full walk: ``walker_batches * (fixed_seconds_per_call + seconds_per_block * (n_blocks - 1))``


@dataclass(frozen=True)
class WalkProfile:
    """How a walk's rate scales with the batch size, measured by :func:`walk_cost` (``measure="profile"``): the same
    pilot at B/2, B, 2B and 4B (B the walk's batch), each the second call of its shape."""
    walker_steps_per_s: dict          #: batch size -> rate, from the seconds per block
    seconds_per_block: dict            #: batch size -> seconds per block of the pilot (the per-call part removed)
    plateau_walker_steps_per_s: float #: the largest rate
    batch_fraction_of_plateau: float  #: the walk's batch rate over the plateau
    recommended_batch: int            #: the timed batch with the highest rate: for a fixed walker count it minimises the total time
    stopped_by: str                   #: why the profile ends: ``"given"`` (the caller's batch with B/2..4B), ``"rate fell"`` (the sweep passed
                                      #: its best), or the limit the sweep reached while the rate still rose: ``"memory"``, ``"n_walkers"``, ``"r0"``
    max_batch_memory: object          #: the largest batch the device's free memory holds, or None without a memory cap
    bytes_per_walker: int             #: device bytes of one walker through a batch (a lower bound)
    memory_cap: str                   #: where ``max_batch_memory`` comes from, or why there is none
    verdict: str                      #: about the walk's batch B: ``"under-filled"`` when 2B takes under :data:`UNDERFILLED_DOUBLING` x the
                                      #: time of B (the extra walkers cost under 20 %), measured whenever 2B fits the device, whatever ``n_walkers``;
                                      #: ``"memory-bound"`` when 2B does not fit; else ``"saturated"``; ``"unresolved"`` when 2B could not be timed for want of ``r0`` rows


@dataclass(frozen=True)
class WalkCost:
    """What a trajectory walk costs, stated before it runs: the resolution :func:`simulate_trajectories` walks, in work
    and bytes, and optionally the time measured on the real device."""
    n_walkers: int                    #: walkers of the walk
    walker_batch_size: int            #: walkers per batch (``n_walkers`` for a single batch)
    walker_batch_rule: str            #: what set it: ``"given"``, ``"recommended by the sweep"`` or ``"n_walkers capped by memory"``
    walker_batches: int               #: batches of the walk
    T_max: float                      #: seconds of diffusion time
    n_blocks: int                     #: blocks of the walk
    block_s: float                    #: the block length ``h`` (s)
    K: int                            #: bands per axis and channel, per window
    block_order: int                  #: the order ``m`` of the block table
    n_windows: int                    #: the storage windows the walk accumulates separately
    f_band_hz: float                  #: the band (Hz) the step law resolves: ``f_band``, else ``K / (2 T)``
    sub_steps: int                    #: physics steps per block
    dt_sim: float                     #: seconds per physics step
    sub_step_rules: dict              #: the sub-step count each applicable rule asks for, by name
    binding_rule: str                 #: the rule that sized the step (``"override"`` for a pinned count)
    binding_length: object            #: the :class:`~dmipy_sim.geometry.base.LengthScales` field that rule divides, or None
    binding_length_m: object          #: that length (m), or None
    length_scales: dict               #: ``min_feature``, ``min_gap``, ``lookup_cell``, ``surface_pore`` (m)
    walker_steps: int                 #: ``n_walkers * n_blocks * sub_steps``
    bytes_per_walker: int             #: device bytes of one walker through a batch (inputs, accumulators and outputs; a lower bound)
    accumulator_bytes_per_walker: int     #: the walk's accumulators per walker (level 2, a ring's moments, a block's moments, the field's, the events)
    backend: str                      #: the backend that would walk it
    backends_accepting: list          #: the other installed backends that do not refuse this request
    backends_refusing: dict           #: the other installed backends that refuse it, ``{name: reason}``
    K_c3: object = None               #: the field's band per window, or None for "a pilot derives it" (no field: 0)
    field_read_interval: object = None  #: the field read's interval (s), or None without a field
    field_reads_per_block: object = None  #: reads per block, or None without a field
    events_capacity: object = None    #: the events buffer per walker and column, or None for "a pilot sizes it" (no events: 0)
    measured: object = None           #: :class:`WalkMeasured`, or None when not measured
    profile: object = None            #: :class:`WalkProfile`, or None unless ``measure="profile"``


def _cost_of(res, geometry, n_walkers, walker_batch_size, backend_name, *, batch_rule="given", n_ch=0):
    """The :class:`WalkCost` of a resolved walk (``walker_batch_size`` None is one batch)."""
    from .backends import installed
    n_walkers = int(n_walkers)
    size = n_walkers if (walker_batch_size is None or int(walker_batch_size) >= n_walkers) else int(walker_batch_size)
    ls = res.length_scales
    field = _RULE_LENGTH.get(res.binding)
    length = getattr(ls, field) if field else None
    if field == "surface_pore" and length is None:
        field, length = "min_feature", ls.min_feature
    accepting, refusing = [], {}
    for other in sorted(set(installed()) - {backend_name}):
        try:
            reason = _backend_refusal(resolve_backend(other), other, res, geometry)
        except Exception as e:                               # an installed backend that cannot load costs nothing
            reason = f"backend {other!r} could not be loaded: {e!r}"
        if reason:
            refusing[other] = reason
        else:
            accepting.append(other)
    f = lambda x: None if x is None else float(x)
    has_events = res.crossing or res.binding_mt
    return WalkCost(
        n_walkers=n_walkers, walker_batch_size=int(size), walker_batch_rule=batch_rule, walker_batches=-(-n_walkers // int(size)),
        T_max=float(res.block_h) * int(res.n_blocks), n_blocks=int(res.n_blocks), block_s=float(res.block_h),
        K=int(res.K), block_order=int(res.block_order), n_windows=int(res.n_windows), f_band_hz=float(res.f_band),
        sub_steps=int(res.sub_steps), dt_sim=float(res.dt_sim),
        sub_step_rules={k: int(v) for k, v in res.rules.items()},
        binding_rule=res.binding, binding_length=field, binding_length_m=f(length),
        length_scales=dict(min_feature=f(ls.min_feature), min_gap=f(ls.min_gap), lookup_cell=f(ls.lookup_cell),
                           surface_pore=f(ls.surface_pore)),
        walker_steps=n_walkers * int(res.n_blocks) * int(res.sub_steps),
        bytes_per_walker=_device_bytes_per_walker(res, n_ch=n_ch),
        accumulator_bytes_per_walker=_accumulator_bytes(res, n_ch=n_ch),
        backend=backend_name, backends_accepting=accepting, backends_refusing=refusing,
        K_c3=(0 if res.field_read is None else res.K_c3),
        field_read_interval=(None if res.field_read is None else res.field_read["dt_f"]),
        field_reads_per_block=(None if res.field_read is None else int(res.sub_steps) // int(res.field_read["n_read"])),
        events_capacity=(res.events_capacity if has_events else 0))


def _announce_cost(run, cost):
    """The cost in the run's record (a ``cost`` row) and the log; a walk on the JAX kernels that an installed
    backend accepts also warns, naming the backend, the walker-steps and the rule that sized the step."""
    run.cost(**dataclasses.asdict(cost))
    log.info("  cost: %.3g walker-steps (%d walkers x %d blocks x %d sub-steps; %s rule) on the %s backend",
             cost.walker_steps, cost.n_walkers, cost.n_blocks, cost.sub_steps, cost.binding_rule, cost.backend)
    if cost.backend == JAX and cost.backends_accepting:
        names = ", ".join(cost.backends_accepting)
        msg = (f"this walk of {cost.walker_steps:.3g} walker-steps (the {cost.binding_rule} rule, "
               f"{cost.sub_steps} sub-steps) runs on the jax kernels; the installed backend {names} accepts the "
               f"same request: pass backend={cost.backends_accepting[0]!r}")
        run.warning(msg, walker_steps=cost.walker_steps, binding_rule=cost.binding_rule,
                    backends=cost.backends_accepting)
        log.warning("%s", msg)


#: ``walk_cost(measure="profile")`` calls a batch under-filled when doubling it costs less than this multiple of
#: its time: 1.2 is "the extra walkers cost under 20 %", the definition of a nearly free doubling
UNDERFILLED_DOUBLING = 1.2


def _accumulator_bytes(res, *, n_ch=0):
    from .accumulate import accumulator_bytes
    K_c3 = 0 if res.field_read is None else (res.K_c3 or 0)
    cap = res.events_capacity or 0
    return accumulator_bytes(res.K, res.block_order, res.record, K_c3=K_c3, n_ch=(n_ch if res.field_read is not None else 0),
                             events_cap=(cap * ((1 if res.crossing else 0) + (1 if res.binding_mt else 0))))


def _device_bytes_per_walker(res, *, n_ch=0):
    """Bytes one walker holds on the device through a batch: its start (3 float32) and key (2 uint32), its
    accumulators (:func:`~dmipy_sim.engine.accumulate.accumulator_bytes`) and its coefficients per window (the bands
    per axis and channel, two exact endpoints each; the field's modes) -- independent of the block count. The output
    buffers only: the walk's working set is not counted, so this is a lower bound."""
    ch = 3 + (1 if res.record else 0)
    per_window = 4 * ch * (int(res.K) + 2) + 4 * 4
    if res.field_read is not None:
        per_window += 4 * int(n_ch) * ((res.K_c3 or 0) + 1)
    return 20 + _accumulator_bytes(res, n_ch=n_ch) + int(res.n_windows) * per_window


def _free_device_bytes():
    """Free bytes of the first JAX device (``bytes_limit - bytes_in_use``), or None where the device reports no
    memory statistics (the CPU)."""
    try:
        st = jax.devices()[0].memory_stats()
    except Exception:
        st = None
    if not st or "bytes_limit" not in st:
        return None
    return int(st["bytes_limit"]) - int(st.get("bytes_in_use", 0))


def _recommended(per_block):
    """The batch of ``per_block`` ({batch: seconds per block}) with the highest walkers per second."""
    return max(per_block, key=lambda b: b / per_block[b])


def _sweep(per_block_of, cap, limit):
    """The pilot timed (``per_block_of(batch)`` -> seconds per block) at 4096 * 2^k walkers below ``cap`` and at ``cap``
    itself (``limit`` names what ``cap`` is), stopping at the first point whose rate is below the best so far.
    Returns ``({batch: seconds per block}, stopped_by)``."""
    sizes = [b for b in (4096 * 2 ** k for k in range(64)) if b < cap] + [cap]
    per, best = {}, 0.0
    for b in sizes:
        per[b] = per_block_of(b)
        rate = b / per[b]
        if rate < best:
            return per, "rate fell"
        best = rate
    return per, limit


def _profile(per_block, vb, res, max_batch, stopped_by, n_ch=0):
    """The :class:`WalkProfile` of the seconds per block ``per_block`` ({batch: seconds}), the verdict read at batch ``vb``:
    the seconds per block of 2B against those of B."""
    rates = {b: b * res.sub_steps / tt for b, tt in sorted(per_block.items())}
    plateau = max(rates.values())
    if 2 * vb in per_block:
        verdict = "under-filled" if per_block[2 * vb] < UNDERFILLED_DOUBLING * per_block[vb] else "saturated"
    else:
        verdict = "memory-bound" if (max_batch is not None and 2 * vb > max_batch) else "unresolved"
    return WalkProfile(walker_steps_per_s=rates, seconds_per_block={b: per_block[b] for b in sorted(per_block)},
                       plateau_walker_steps_per_s=plateau, batch_fraction_of_plateau=rates[vb] / plateau,
                       recommended_batch=int(_recommended(per_block)), stopped_by=stopped_by,
                       max_batch_memory=None if max_batch is None else int(max_batch),
                       bytes_per_walker=_device_bytes_per_walker(res, n_ch=n_ch),
                       memory_cap=("device free memory" if max_batch is not None
                                   else "unavailable: the device reports no memory statistics"),
                       verdict=verdict)


def _fit_blocks(seconds, batch, max_blocks):
    """``(seconds_per_block, fixed_seconds_per_call, (n1, n2))`` of a ``batch`` from ``seconds(batch, blocks)``, the
    second call of each shape: n1 = 1 block, n2 from 2 doubling until ``t(n2) >= 2 t(n1)`` (the blocks then weigh at
    least as much as the fixed part of a call), at most ``max_blocks``."""
    t1 = seconds(batch, 1)
    if max_blocks <= 1:
        return t1, 0.0, (1, 1)
    n2 = 2
    t2 = seconds(batch, n2)
    while t2 < 2 * t1 and n2 < max_blocks:
        n2 = min(2 * n2, max_blocks)
        t2 = seconds(batch, n2)
    per = (t2 - t1) / (n2 - 1)
    if per <= 0:                                  # the blocks were lost in the timing noise: the longer call is the bound
        per = t2 / n2
    return per, max(t1 - per, 0.0), (1, n2)


def walk_cost(n_walkers, diffusivity, geometry, T_max, block_h=None, *, walker_batch_size=DEFAULT_WALKER_BATCH,
              backend=None, measure=False, **walk_kwargs):
    """What a trajectory walk costs, stated before it runs: a :class:`WalkCost`.

    It takes the arguments of :func:`simulate_trajectories`, with its names and defaults, and costs that call: the
    request is resolved exactly as the walk resolves it (one code path: ``walk.sub_steps`` equals
    ``walk_cost(...).sub_steps``) and refused where the walk refuses. ``walk_kwargs`` are the walk's other keywords
    (``K``, ``block_order``, ``tiers``, ``sub_steps``, ``f_band``, ``kappa_MT``, ``dwell_time``, ``seed``,
    ``bounce_budget``, ``field_basis``, ``K_c3``, ``field_read_interval``, ``events_capacity``, ``r0``, ...); a keyword
    the walk does not take is refused. A band the walk would derive by a pilot (``K_c3``) or a buffer it would size by
    one (``events_capacity``) is reported as None: the cost of a walk with those is stated once they are given.

    Parameters
    ----------
    n_walkers, diffusivity, geometry, T_max, block_h
        As :func:`simulate_trajectories`. ``T_max`` scales the walker-steps linearly; the bands ``K`` set the
        accumulators' bytes per walker and the memory-limited batch, and, without ``f_band``, the band the step law
        resolves; ``block_h`` is the block when the band's own is not wanted.
    walker_batch_size : int or None
        Walkers per batch, default the walk's own. ``None`` finds the batch: with ``measure="profile"`` the pilot is
        timed at 4096 * 2^k walkers below ``min(n_walkers, the memory-limited batch)`` and at that cap, stopping at the
        first batch whose walker-steps per second is below the best so far; the batch is the one with the highest
        rate (:attr:`WalkProfile.recommended_batch`), which for a fixed walker count minimises the total time.
        Otherwise it is ``n_walkers`` capped by the memory-limited batch (one batch where it fits).
        ``WalkCost.walker_batch_rule`` says which. The memory-limited batch is the device's free memory over
        ``bytes_per_walker`` (no cap where the device reports no memory statistics).
    backend : str or Backend, optional
        The backend that would walk it (default the JAX kernels).
    measure : {False, True, "profile"}
        ``True`` walks the real batch (``min(walker_batch_size, n_walkers)`` walkers) on the real device and backend and
        times it at two pilot lengths, each the second call of its shape (the compile is in the first):
        :attr:`WalkCost.measured`. The walk is the production program (``walk_kwargs`` included; ``r0`` is sliced
        to the batch). A timed call is the whole of :func:`simulate_trajectories`: seeding, setup, the walk and the
        transfer to the host. Only the walk grows with the blocks; the rest is paid once per call, so one pilot
        length would multiply that fixed part by every block of the walk and flatten the rate curve. The pilot is
        therefore timed at n1 = 1 block and at n2 blocks (the walk's own block, ``K`` and step), n2 starting at 2 and
        doubling until ``t(n2) >= 2 t(n1)`` (the blocks weigh at least as much as the fixed part), at most the walk's
        own blocks; the seconds per block are ``(t(n2) - t(n1)) / (n2 - n1)``, the fixed part ``t(n1)`` less one
        block, and the projection is ``walker_batches * (fixed + seconds_per_block * n_blocks)``, linear in the
        blocks. ``"profile"`` also times the pilot over the batch sizes above -- the sweep for a batch not given, else
        B/2, B, 2B and 4B for the given B (the larger sizes only where the device's free memory holds them) -- and
        rates every batch by its seconds per block: :attr:`WalkCost.profile`. Its verdict asks whether more walkers
        are nearly free at the walk's batch B: the seconds per block of 2B (timed whenever it fits the device) are under
        :data:`UNDERFILLED_DOUBLING` = 1.2 x those of B, i.e. the extra walkers cost under 20 %.

    Returns
    -------
    WalkCost
    """
    import inspect
    import time
    unknown = set(walk_kwargs) - set(inspect.signature(simulate_trajectories).parameters)
    if unknown:
        raise TypeError(f"walk_cost got keywords simulate_trajectories does not take: {sorted(unknown)}")
    kw = dict(walk_kwargs)
    be = resolve_backend(backend)
    name = JAX if be is None else str(be.name)
    from ..spec.build import as_geometry
    with _bounce_budget_override(geometry, kw.get("bounce_budget")):
        geometry = as_geometry(geometry)
        fb = kw.get("field_basis")
        res = _resolve_walk(geometry, diffusivity, T_max, block_h, tiers=kw.get("tiers", "all"),
                            sub_steps=kw.get("sub_steps"), kappa_MT=kw.get("kappa_MT", 0.0),
                            dwell_time=kw.get("dwell_time", 0.0), f_band=kw.get("f_band"), seed=kw.get("seed", 42),
                            enforce_compartment=kw.get("enforce_compartment", False),
                            field_basis=fb, K=kw.get("K"), block_order=kw.get("block_order"),
                            window_T=kw.get("window_T"), K_c3=kw.get("K_c3"), field_read_interval=kw.get("field_read_interval"),
                            events_capacity=kw.get("events_capacity"), raw_subset=kw.get("raw_subset", 0), envelope=kw.get("envelope"))
        n_ch = 0 if fb is None else len(fb.channel_names)
        if be is not None:
            refusal = _backend_refusal(be, name, res, geometry)
            if refusal:
                raise ValueError(refusal)
        free = _free_device_bytes()
        max_batch = None if free is None else max(1, free // _device_bytes_per_walker(res, n_ch=n_ch))
        given = walker_batch_size is not None
        batch = int(walker_batch_size) if given else min(int(n_walkers), max_batch or int(n_walkers))
        cost = _cost_of(res, geometry, n_walkers, batch, name,
                        batch_rule="given" if given else "n_walkers capped by memory", n_ch=n_ch)
    if measure:
        max_blocks = cost.n_blocks
        r0 = None if kw.get("r0") is None else np.asarray(kw["r0"])
        kw.update(sub_steps=res.sub_steps, f_band=res.f_band, K=res.K, block_order=res.block_order)

        def seconds(b, blocks):
            """Seconds of the second call of this shape (the first holds the compile)."""
            k = dict(kw, **({} if r0 is None else dict(r0=r0[:b])))
            walk = lambda: simulate_trajectories(b, diffusivity, geometry, blocks * res.block_h, res.block_h,
                                                 walker_batch_size=b, backend=backend, **k)
            walk()
            t0 = time.perf_counter(); walk()
            return time.perf_counter() - t0

        fits = {}

        def fit(b):
            if b not in fits:
                fits[b] = _fit_blocks(seconds, b, max_blocks)
            return fits[b]

        per_block_of = lambda b: fit(b)[0]
        available = None if r0 is None else len(r0)
        profile = None
        if measure == "profile" and not given:
            limits = {"memory": max_batch, "n_walkers": int(n_walkers), "r0": available}
            limit = min((k for k, v in limits.items() if v is not None), key=lambda k: limits[k])
            per, stopped = _sweep(per_block_of, limits[limit], limit)
            B = _recommended(per)
            if 2 * B not in per and (max_batch is None or 2 * B <= max_batch) and (available is None or 2 * B <= available):
                per[2 * B] = per_block_of(2 * B)                 # the verdict's doubling, whatever n_walkers
            profile = _profile(per, B, res, max_batch, stopped, n_ch=n_ch)
            cost = dataclasses.replace(cost, walker_batch_size=int(B), walker_batch_rule="recommended by the sweep",
                                       walker_batches=-(-int(n_walkers) // int(B)))
        else:
            B = cost.walker_batch_size
            per_block_of(B)
            if measure == "profile":
                per = {B: per_block_of(B)}
                if B // 2 >= 1:
                    per[B // 2] = per_block_of(B // 2)
                for m in (2, 4):
                    if (max_batch is not None and m * B > max_batch) or (available is not None and m * B > available):
                        break
                    per[m * B] = per_block_of(m * B)
                profile = _profile(per, B, res, max_batch, "given", n_ch=n_ch)
        ps, fixed, pilot = fit(B)
        measured = WalkMeasured(batch=int(B), pilot_blocks=pilot, seconds_per_block=ps, fixed_seconds_per_call=fixed,
                                seconds_per_walker_step=ps / (B * res.sub_steps),
                                projected_s=cost.walker_batches * (fixed + ps * max_blocks))
        cost = dataclasses.replace(cost, measured=measured, profile=profile)
    return cost


def _compact_events(n, steps, labels, dt_sim):
    """One batch's events buffers ``(n (nb,), steps (nb, cap), labels (nb, cap))`` as the host's list: ``(counts,
    t (float64, seconds), label (int8))`` of the first ``n`` entries per walker, exact in time (``step x dt_sim``)."""
    n = np.asarray(n, np.int64); steps = np.asarray(steps); labels = np.asarray(labels)
    cap = steps.shape[1] if steps.ndim == 2 else 0
    keep = np.minimum(n, cap)
    mask = np.arange(cap)[None, :] < keep[:, None]
    return keep, np.asarray(steps[mask], np.float64) * float(dt_sim), np.asarray(labels[mask], np.int8)


def simulate_trajectories(
    n_walkers: int,
    diffusivity: float,
    geometry,
    T_max: float,
    block_h: float = None,
    seed: int = 42,
    walker_batch_size: int = DEFAULT_WALKER_BATCH,
    tiers="all",
    sub_steps: int = None,
    require_gpu=None,
    r0=None,
    kappa_MT: float = 0.0,
    dwell_time: float = 0.0,
    equilibrate_binding="auto",
    K: int = None,
    block_order: int = None,
    window_T: float = None,
    raw_subset: int = 0,
    enforce_compartment: bool = False,
    field_basis=None,
    K_c3: int = None,
    field_read_interval: float = None,
    events_capacity: int = None,
    envelope=None,
    bounce_budget: int = None,
    f_band: float = None,
    backend="jax",
):
    """Walk the spins ONCE and keep, per walker, the bands a replay pack stores: the producer of the replay path.

    Unlike :func:`simulate`, this applies NO gradient waveform: it keeps every walker's path as its exact start and
    end and the ``K`` sine bands of the bridge between them (the cumulative boundary local time in the same form, the
    field read along the path as cosine modes), accumulated while the walk runs (:mod:`dmipy_sim.engine.accumulate`),
    so any waveform / relaxation hypothesis is applied afterwards from the coefficients
    (:func:`~dmipy_sim.replay.bank.build_replay_pack`). This is the walk-once half of the replay invariant (the bands
    depend only on ``geometry, diffusivity, seed``).

    The walk runs in blocks of ``h``: the band's own (:func:`~dmipy_sim.engine.accumulate.block_length`: ``omega_K h =
    0.8`` at the top band ``omega_K = pi K / T``, a whole number of blocks of one step count each; 100 us at K = 256
    over 100 ms) unless ``block_h`` gives one. Inside a block the steps add the moments of their increments about the
    block's centre (order :data:`~dmipy_sim.engine.accumulate.BLOCK_ORDER` = 3 by default) and every ring of blocks is
    projected onto the cosine basis (:func:`~dmipy_sim.replay.compression.block_table`) into a float32 level-2
    accumulator; at the end of a window the projections become the bridge's bands by integration by parts
    (:func:`~dmipy_sim.replay.compression.cosine_band_scale`). Nothing is stored per block but the ``raw_subset``; a
    walker's device and host bytes do not grow with the block count. A walk longer than one storage window
    (``window_T``, 100 ms) is accumulated window by window (RPK.md 4.3).

    A walk whose walkers can change pool (a permeable wall, MT binding) keeps the occupancy as EVENTS, ``(t, label)``
    at every change, exact in the walk's own time, in a per-walker buffer of ``events_capacity`` entries sized by a
    pilot of :data:`~dmipy_sim.engine.accumulate.PILOT_WALKERS` walkers (the largest count reached, with
    :data:`~dmipy_sim.engine.accumulate.EVENTS_MARGIN`) unless given, and refused by name on overflow; an impermeable
    walk keeps one label per walker. The MT bound column keeps its bind / unbind events the same way.

    Sub-stepping: :func:`dmipy_sim.engine.physics.resolve_sub_steps` chooses the steps per block from the geometry's
    length scales, the channels recorded (the surface criterion when the surface channel is recorded, the binding
    criterion when ``kappa_MT > 0``) and the band ``K / (2 T)`` (``f_band`` overrides it), the same dispatch the fused
    engine uses. The walk states its cost before it starts: :func:`walk_cost` states the same resolution as a
    :class:`WalkCost`, and the walk writes it as the ``cost`` row of its run.

    Parameters
    ----------
    n_walkers : int
        Total number of walkers.
    diffusivity : float
        Diffusion coefficient in m²/s.
    geometry : Geometry
        Boundary geometry (a spec, a spec file or its dict too).
    T_max : float
        Total simulation duration in seconds.
    block_h : float, optional
        The block length (s) in place of the band's own.
    seed : int
        Master PRNG seed.
    walker_batch_size : int
        Number of walkers per batch.  Reduce if OOM (the batch loop also auto-halves on an OOM exception).
    tiers : {"all", ()}
        Which replay tiers the walk records beyond the positions. ``"all"`` (default) records every
        channel the geometry supports -- the boundary local time (surface relaxivity, rho/D = 1) and
        the compartment occupancy (per-compartment T2/T1); with ``kappa_MT > 0`` also the bound
        column. ``()`` records positions only, a cheaper walk (no surface sub-step criterion) for a
        gradient-only replay.
    require_gpu : {None, True, False}
        GPU guard against a silent CPU fallback.  ``True`` raises if no GPU is
        visible; ``False`` opts out; ``None`` (default) warns for large CPU runs.
    r0 : array-like of shape (n_walkers, 3), optional
        Caller-supplied seed positions; ``geometry.init_positions()`` is skipped.
    kappa_MT : float
        Magnetization-transfer surface reactivity (m/s): ``> 0`` binds free water at the walls (the myelin walls of a
        ``PackedMyelinatedCylinders``, the wall of any geometry with ``has_walls``) and records the bound column's
        bind / unbind events. At ``kappa_MT = 0`` the walk is the plain one.
    dwell_time : float
        Mean bound-pool residence time (s); must be ``> 0`` when ``kappa_MT > 0``.
    equilibrate_binding : {'auto', 'burnin', 'fast', 'off'}
        How the bound pool reaches thermal-equilibrium occupancy before t=0 (MT only);
        see :func:`dmipy_sim.engine.mt.resolve_equilibrate_mode`.
    K : int, optional
        The bands per axis and channel per window; the storage band by default
        (:data:`~dmipy_sim.engine.accumulate.STORAGE_BAND_HZ`, K = 256 per 100 ms).
    block_order : int, optional
        The order ``m`` of the block table, 0..3 (:data:`~dmipy_sim.engine.accumulate.BLOCK_ORDER` = 3): the
        quadrature error :func:`~dmipy_sim.replay.compression.quadrature_rel` (at the derived block 1.3e-7 of the top
        band's phase variance; at ``m = 0`` the block ends' DST-I read in the continuous normalisation).
    window_T : float, optional
        The storage window (s) a longer walk is accumulated in; :data:`~dmipy_sim.replay.bank.WINDOW_T` by default.
    raw_subset : int
        Keep the first ``raw_subset`` walkers' positions, contact, pool label and bound state at every block end and
        their field reads at the walk's read instants beside the coefficients, as plain arrays
        (``CoefficientWalk.raw_positions``, ``raw_boundary``, ``raw_compartment``, ``raw_bound``, ``raw_field``):
        the opt-in raw output, for a measured certificate, a visualisation or a dense oracle.
    field_basis : FieldGrid, optional
        A gridded susceptibility field basis (:class:`~dmipy_sim.fields.susceptibility_field.FieldGrid`) the walk
        reads itself at the kernel's position (a periodic grid wraps) every ``field_read_interval`` seconds, the
        trapezoid of consecutive reads accumulated onto ``K_c3 + 1`` cosine modes per window
        (``CoefficientWalk.field_coeffs``), what the pack's field channel stores. A strand basis is read by
        :func:`~dmipy_sim.engine.adaptive.simulate_trajectories_adaptive`.
    K_c3 : int, optional
        The field's band (modes per window). None derives it on a pilot of
        :data:`~dmipy_sim.engine.accumulate.PILOT_WALKERS` walkers that keeps its block means
        (:func:`~dmipy_sim.replay.bank.derive_field_band`: the first band whose codec error on the certificate's
        battery is within its floor); a fill inherits the certifying pack's.
    field_read_interval : float, optional
        The interval (s) between the walk's reads of the field. None derives it from a pilot of
        :data:`~dmipy_sim.engine.accumulate.PILOT_WALKERS` walkers that reads every step
        (:func:`~dmipy_sim.engine.accumulate.field_read_of_pilot`: per pool the measured quadratic variation of the
        read series, ``v_read`` under the floor's square at the envelope's ``B0`` and the spec's ``chi``, the finest
        over the pools; the trapezoid's constant measured per channel on the same pilot); the walk reads at the
        largest divisor of its step count within it, the step itself when the rule asks for less
        (``CoefficientWalk.block["field_read"]`` records the rule, the pools and the constants, and the pack's
        certificate charges the term from the walk's own quadratic variation).
    events_capacity : int, optional
        The events buffer per walker and column, in place of the pilot's.
    envelope : dict, optional
        The certificate's envelope (:func:`~dmipy_sim.replay.compression.default_envelope`) the field read's ``B0``
        and the field band's pilot are derived at; the defaults otherwise.

    Returns
    -------
    CoefficientWalk
        The coefficients per window, the contact, the occupancy (one label per walker, with the events of a walk
        whose walkers change pool), the field modes, the quadratic variation per channel, the block record ``(h, m,
        rule, ...)``, the raw subset when asked, ``illegal_crossings`` and the work record.
    bounce_budget : int, optional
        The bounce budget of this walk in place of the geometry's derived one (``geometry.bounce_loop.budget``),
        set on the geometry for the walk and restored after it. The derived budget is the worst case a step can
        need; a smaller one is certified by the walk itself: ``work["exhausted_steps"]`` is the number of
        walker-steps whose budget ran out with path untested, zero meaning the trajectories of any larger budget
        bit for bit. :func:`certify_bounce_budget` sizes one from a pilot walk. Reported on every walk, counting
        or not; a walk with exhausted steps warns and :func:`~dmipy_sim.replay.bank.build_replay_pack` refuses it.
    backend : str or Backend
        Which execution of the physics walks: ``"jax"`` (the default, the kernels of this module), the name of
        an installed backend (the ``dmipy_sim.backends`` entry point), or a backend object. An unknown name is
        refused naming what is installed; a backend that does not implement this walk (an engine class, a
        tier, a wall) refuses by name, and the paths only the JAX kernels implement (the myelinated substrates,
        MT binding, in-walk field sampling, several storage windows, the compartment guard) are refused for any
        other backend rather than fallen back from. A backend walks each batch in the coefficient form
        (:class:`~dmipy_sim.engine.backends.CoefficientBatch`: the bands it accumulated, its events, the raw
        subset's block ends) and the walk takes it as it takes its own, converting nothing; a backend that returns
        a walk of saves is refused by name. ``work["backend"]`` and the run record say which backend walked. See
        :mod:`dmipy_sim.engine.backends`.
    """
    _backend = resolve_backend(backend)
    _backend_name = JAX if _backend is None else str(_backend.name)
    with Run("simulate_trajectories", params=dict(n_walkers=n_walkers, diffusivity=diffusivity, geometry=type(geometry).__name__, T_max=T_max, block_h=block_h, backend=_backend_name, K=K)) as run, _bounce_budget_override(geometry, bounce_budget):
        from ..spec.build import as_geometry
        geometry = as_geometry(geometry)               # a spec, a spec file or a dict is a substrate too
        # GPU guard — never silently fall back to CPU for a heavy walk (CLAUDE rule).
        from .gpu import check_gpu
        check_gpu(n_walkers, require_gpu, what="simulate_trajectories")
        if kappa_MT > 0.0 and dwell_time <= 0.0:
            raise ValueError("dwell_time must be > 0 when kappa_MT > 0.")
        res = _resolve_walk(geometry, diffusivity, T_max, block_h, tiers=tiers, sub_steps=sub_steps, kappa_MT=kappa_MT,
                            dwell_time=dwell_time, f_band=f_band, seed=seed, enforce_compartment=enforce_compartment,
                            field_basis=field_basis, K=K, block_order=block_order, window_T=window_T, K_c3=K_c3,
                            field_read_interval=field_read_interval, events_capacity=events_capacity, raw_subset=raw_subset,
                            envelope=envelope)
        record, n_blocks, dt_actual, sub_steps, dt_sim, step_rule = res.record, res.n_blocks, res.block_h, res.sub_steps, res.dt_sim, res.step_rule
        R_geom = res.length_scales.min_feature
        step_l_sim = jnp.float32(jnp.sqrt(6.0 * diffusivity * dt_sim))

        log.info(f"  sub_steps={sub_steps}, dt_sim={dt_sim*1e6:.3f} µs, "
              f"step_l={float(step_l_sim)*1e6:.4f} µm"
              + (f", step_l/R={float(step_l_sim)/float(R_geom):.4f}" if R_geom else ""))

        # ── In-walk field read: a gridded basis read at the kernel's position every n_read steps, passed to the
        # walk as arguments
        _sampling = field_basis is not None
        if _sampling:
            from ..fields.susceptibility_field import FieldGrid
            if not isinstance(field_basis, FieldGrid):
                raise TypeError(f"field_basis must be a FieldGrid (a gridded field read in the walk), got "
                                f"{type(field_basis).__name__}; a strand basis is read by simulate_trajectories_adaptive")
            _f_arrays = field_basis.device_arrays()
            _f_periodic = tuple(bool(p) for p in field_basis.periodic)
            _f_key = ("field", tuple(_f_arrays[0].shape), _f_periodic)
        else:
            if K_c3 is not None or field_read_interval is not None:
                raise ValueError("K_c3 and field_read_interval are the field channel's band and read interval: they go with field_basis=")
            _f_arrays, _f_periodic, _f_key = (), None, None
        _n_ch = int(_f_arrays[0].shape[0]) if _sampling else 0

        def _reader(arrays):
            return _GridRead(arrays, _f_periodic) if _sampling else None

        # the cost: what this walk costs and which rule sized it, in the run's record before the walk starts
        _announce_cost(run, _cost_of(res, geometry, n_walkers, walker_batch_size, _backend_name, n_ch=_n_ch))

        permeability = geometry.permeability
        has_permeability = permeability is not None
        has_reflect_with_log_weight = geometry.has_walls
        from .accumulate import (Kind, Events, events_init, moment_scan as _moment_scan, ring_scan, coefficient_form,
                                 field_mode_form, PRECISION, ACCUMULATION, RING, PILOT_WALKERS, EVENTS_MARGIN)

        # ── The walk's work: wall encounters, crossings and budget exhaustions, summed per walker in the
        # carry and over the batches on the host, only when the geometry counts (`Geometry.count_walls`);
        # a walk that does not count carries nothing and is the same program as before.
        _counting = bool(geometry.count_walls)
        _work = [0, 0, 0, False]                    # hits, crossings, exhausted steps; whether a kernel reported them

        def _work0():
            # hits and crossings only when counting; the budget's exhaustion on every walk (it is the AND of two
            # values the loop already holds, and one integer add per sub-step here)
            zero = jnp.int32(0)
            return (zero if _counting else None, zero if _counting else None, zero)

        def _work_add(w, hit):
            exh = w[2] + hit.exhausted.astype(jnp.int32)
            if not _counting:
                return (None, None, exh)
            n = hit.n_hits if hit.n_hits is not None else jnp.int32(0)
            return (w[0] + n, w[1] + hit.crossed.astype(jnp.int32), exh)

        def _work_acc(w):
            if w is not None:
                for i in range(3):
                    if w[i] is not None:
                        _work[i] += int(np.asarray(w[i], np.int64).sum())
                _work[3] = True

        # ── Standard path (position-only) ─────────────────────────────────────────
        _carries_side = False   # set below only where the geometry accepts a carried side
        if has_permeability:
            kappa_over_D = jnp.float32(float(permeability) / diffusivity)
            permeate = geometry.permeate

            # Does this geometry support carried-compartment bookkeeping? If so the walker OWNS its
            # side and only a granted crossing changes it. Deriving the compartment from the position
            # each step is what allows a walker whose step merely ROUNDS onto the surface to change
            # compartment without moving -- measured at 0.675% of intra walkers per 30k steps on a
            # dense packing at kappa = 0, and ~10% over a production walk. The sentinel inside
            # `permeate` ejects such a walker back to its own side; this carries the label.
            # Modelled on MC/DC's deportation check, which compares the final position against the
            # walker's `initial_location` and re-runs it (dynamicsSimulation.finalPositionCheck).
            _carries_side = geometry.carries_side
            interact = geometry.interact

            def inner_step(carry, _):
                r, key, side, bad, work = carry
                key, step_key, perm_key = jax.random.split(key, 3)
                unit_noise = pair_step(step_key)
                step = unit_noise * step_l_sim
                if _carries_side:
                    hit = permeate(r, step, kappa_over_D, jnp.float32(0.0), perm_key, side)
                    # a granted crossing is the ONLY thing that flips the carried side
                    side = jnp.where(hit.crossed, -side, side)
                else:
                    # `interact`, not `permeate`: a geometry that reports its own refusal returns it on the
                    # WallHit, and a mesh does (#479), so a step held still is counted here rather than lost.
                    hit = interact(r, step, kappa_over_D=kappa_over_D, key=perm_key)
                bad = bad + hit.illegal.astype(jnp.int32)
                return (hit.r, key, side, bad, _work_add(work, hit)), None
        else:
            # `interact`, not `reflect`: the same function at kappa = 0 (`Geometry.interact`), and the WallHit
            # carries the refusal and the work that `reflect` drops
            interact = geometry.interact

            def inner_step(carry, _):
                r, key, side, bad, work = carry
                key, subkey = jax.random.split(key)
                unit_noise = pair_step(subkey)
                step = unit_noise * step_l_sim
                hit = interact(r, step)
                bad = bad + hit.illegal.astype(jnp.int32)
                return (hit.r, key, side, bad, _work_add(work, hit)), None

        # ── Universal compartment sentinel ───────────────────────────────────────
        # `permeate(..., side=)` gives PackedCylinders an exact ejection, but every other
        # geometry re-derives the compartment from the position and shares the same defect: an
        # adversarial probe that steps walkers EXACTLY onto a wall (bisecting to the boundary
        # with the geometry's own classifier) flips 23-81% of them on every analytic geometry,
        # against 0% once a side is carried.
        #
        # At kappa <= 0 the rule needs no geometry knowledge: an impermeable wall grants no
        # crossings, so ANY change of compartment label is illegal and the move that caused it
        # can be rejected outright. One check therefore covers spheres, cylinders, ellipsoids,
        # packed spheres, slabs, shells and meshes alike. At kappa > 0 a label change may be a
        # genuine crossing, so there a geometry must opt in through the `side` API instead.
        #
        # Rejecting rather than ejecting costs a walker one step of displacement, at a measured
        # rate of ~1e-7 per walker-step -- unmeasurable against the walk, and strictly better
        # than silently relabelling the walker.
        # Cost is why this is opt-in rather than the default: a classify_position per sub-step
        # roughly DOUBLES the walk (+88% measured on Sphere, +69% on PackedCylinders), and walk
        # time is the entire cost of the engine. The geometries that carry the largest risk --
        # dense packings, where the rate scales with surface-to-volume -- instead get an exact
        # O(1) sentinel inside their own step for +0.8%, so they are protected by default and
        # do not need this. Use it to validate a geometry that has no in-step sentinel yet.
        _cls_fn = geometry.classify_position
        _use_guard = (bool(enforce_compartment) and not _carries_side
                      and (not has_permeability or float(permeability) <= 0.0))
        if _use_guard:
            _inner_raw = inner_step

            def inner_step(carry, _):
                r_old = carry[0]
                carry, out = _inner_raw(carry, _)
                r_new = carry[0]
                keep = _cls_fn(r_new) == _cls_fn(r_old)
                bad = carry[3] + jnp.where(keep, 0, 1)
                return (jnp.where(keep, r_new, r_old),) + carry[1:3] + (bad,) + carry[4:], out

        # ── Compartment channel (relaxation path only) ────────────────────────────
        # The geometry's own pool id, collapsed to two pools (0 extra, 1 enclosed): relaxation is
        # per pool, and an object id would overflow int8 above 127 objects. The label is CARRIED:
        # seeded once from the exact classifier and updated per sub-step through
        # `classify_position_carry`, so a geometry whose per-step classifier is undecidable away
        # from its walls (a mesh) keeps the label it had, and recorded as an event at every change (exact in time).
        def _pool2(comp):
            return jnp.minimum(comp, 1).astype(jnp.int8)

        # ── Relaxation-data path (position + boundary log-weight with rho/D=1) ────
        is_packed_myelin_geom = geometry._is_packed_myelinated
        is_myelin_geom = geometry._is_myelinated
        # Both concentric-cylinder geometries (one axon in an open domain, or a periodic pack)
        # share the one trajectory step below -- make_myelin_traj_step_fn / make_packed_myelin_
        # traj_step_fn, both built on physics.make_myelin_substep -- which carries the compartment
        # id the generic position-only walk cannot express.
        uses_myelin_traj = is_packed_myelin_geom or is_myelin_geom

        # kappa_MT binds free water into a bound pool: PackedMyelinatedCylinders' trajectory step
        # (make_packed_myelin_traj_step_fn) carries it at the myelin walls; any geometry with walls binds at its
        # wall through the impact-angle rule (the former mt_walk producer, now the "mt" kind below). The isolated
        # MyelinatedCylinder's step has no such state and is refused here, before any device work (#536).
        if kappa_MT > 0.0 and not is_packed_myelin_geom and (is_myelin_geom or not has_reflect_with_log_weight):
            raise NotImplementedError(
                f"simulate_trajectories: kappa_MT > 0 (magnetization transfer) is carried by PackedMyelinatedCylinders' "
                f"trajectory step (make_packed_myelin_traj_step_fn) and by the walls of a geometry with has_walls; "
                f"{type(geometry).__name__} has no bound pool to bind into. Pass kappa_MT=0.0 (the default), or walk a "
                f"PackedMyelinatedCylinders or a walled geometry.")
        _mt_on = kappa_MT > 0.0 and record
        _mt_walls = _mt_on and not is_packed_myelin_geom            # the "mt" kind: binding at a walled geometry's wall

        step_fn_traj_pm = None       # set below when the myelin trajectory step runs; read by the work record
        if record and uses_myelin_traj:
            # (Packed or isolated) myelinated cylinder: use the stripped trajectory step fn
            # (geometry + permeability only, rho/D=1 at all walls).  comp_id is the encoded id
            # (0=extra, 1..N_max=intra, >N_max=myelin); compress to 0/1/2 at save.
            # The walk steps the in-cell position and records the continuous (unwrapped) one,
            # as every other periodic substrate's trajectory does (the isolated cylinder has no
            # cell to unwrap: the two are the same position).
            # Magnetization transfer (kappa_MT > 0, PackedMyelinatedCylinders only -- refused above
            # for every other geometry): the step fn binds free water at the myelin walls.
            # kappa_MT == 0 keeps the pre-MT walk bit-for-bit (RNG stream + positions unchanged).
            _mt_on_pm = kappa_MT > 0.0
            step_fn_traj_pm = (make_packed_myelin_traj_step_fn(
                                  geometry, dt_sim, kappa_MT=kappa_MT, dwell_time=dwell_time)
                              if is_packed_myelin_geom else make_myelin_traj_step_fn(geometry, dt_sim))

            def _compress_comp_pm(comp_id):
                return geometry.pool_of(comp_id).astype(jnp.int8)

            def _inner_pm(carry, _):
                return step_fn_traj_pm(carry, None)

        if record and not uses_myelin_traj and not _mt_walls:
            if has_permeability:
                kappa_over_D_relax = jnp.float32(float(permeability) / diffusivity)
                permeate_relax = geometry.permeate
                interact_relax = geometry.interact

                def inner_step_relax(carry, _):
                    r, key, dlog_cum, side, bad, comp, work = carry
                    key, step_key, perm_key = jax.random.split(key, 3)
                    unit_noise = pair_step(step_key)
                    step = unit_noise * step_l_sim
                    if _carries_side:
                        hit = permeate_relax(r, step, kappa_over_D_relax, jnp.float32(1.0), perm_key, side)
                        r_new, dlog_w_unit = hit.r, hit.dlog_w
                        side = jnp.where(hit.crossed, -side, side)
                        bad = bad + hit.illegal.astype(jnp.int32)
                        # Label from the CARRIED side, not from the position. This is the
                        # channel the events are read from, so re-deriving it here would put
                        # the relabelling straight back in even with the sentinel correcting
                        # the coordinate. 0 = extra, 1 = intra, as `_get_comp_id`.
                        comp = jnp.where(side < 0, jnp.int32(1), jnp.int32(0))
                    else:
                        hit = interact_relax(r, step, kappa_over_D=kappa_over_D_relax,
                                             rho_over_D=jnp.float32(1.0), key=perm_key)
                        r_new, dlog_w_unit = hit.r, hit.dlog_w
                        bad = bad + hit.illegal.astype(jnp.int32)
                        comp = geometry.classify_position_carry(r_new, comp)
                    return (r_new, key, dlog_cum + dlog_w_unit, side, bad, comp, _work_add(work, hit)), None

            elif has_reflect_with_log_weight:
                # `interact`, not `reflect_with_log_weight`: for an impermeable wall the two are the
                # same function at the same arguments (`Geometry.interact`), but the WallHit also
                # carries `illegal`, so a geometry that REFUSES a step -- a voxel wall's reject-escape,
                # where the walker is held still -- is counted here instead of being invisible.
                interact = geometry.interact

                def inner_step_relax(carry, _):
                    r, key, dlog_cum, side, bad, comp, work = carry
                    key, subkey = jax.random.split(key)
                    unit_noise = pair_step(subkey)
                    step = unit_noise * step_l_sim
                    hit = interact(r, step, rho_over_D=jnp.float32(1.0))
                    r_new, dlog_w_unit = hit.r, hit.dlog_w
                    bad = bad + hit.illegal.astype(jnp.int32)
                    comp = geometry.classify_position_carry(r_new, comp)
                    return (r_new, key, dlog_cum + dlog_w_unit, side, bad, comp, _work_add(work, hit)), None

            else:
                # FreeDiffusion: no boundaries → dlog_boundary_unit is always 0.
                reflect_free = geometry.reflect

                def inner_step_relax(carry, _):
                    r, key, dlog_cum, side, bad, comp, work = carry
                    key, subkey = jax.random.split(key)
                    unit_noise = pair_step(subkey)
                    step = unit_noise * step_l_sim
                    r_new = reflect_free(r, step)
                    comp = geometry.classify_position_carry(r_new, comp)
                    return (r_new, key, dlog_cum, side, bad, comp, work), None

        if _mt_walls:
            # ── MT binding at a walled geometry's wall (the former mt_walk producer): the impact-angle stick rule,
            # p = min(1, (kappa_MT / D) local_time), on the same local time the surface channel records; a stuck
            # walker freezes for an exponential dwell of mean dwell_time and is released. A binding encounter
            # contributes nothing to the free pool's local time (the spin left the free pool), so rho stays a pure
            # replay knob. Carry (r, key, bound_rem, dlog_cum, bad, work).
            from ..geometry._boundary import bind_probability
            _kappa_over_D_mt = jnp.float32(float(kappa_MT) / float(diffusivity))
            _dwell_steps_mean = jnp.float32(float(dwell_time) / dt_sim)
            if hasattr(geometry, "reflect_with_binding"):
                _reflect_bind = geometry.reflect_with_binding

                def _move_mt(r, step):
                    r_free, dlog_free, blt = _reflect_bind(r, step, jnp.float32(1.0))
                    return r_free, dlog_free, blt, None
            else:
                _interact_mt = geometry.interact

                def _move_mt(r, step):
                    hit = _interact_mt(r, step, rho_over_D=jnp.float32(1.0))
                    return hit.r, hit.dlog_w, -hit.dlog_w, hit          # symmetric: the binding local time is -dlog

            def inner_step_mt(carry, _):
                r, key, bound_rem, dlog_cum, bad, work = carry
                key, step_key, stick_key, dwell_key = jax.random.split(key, 4)
                is_bound = bound_rem > jnp.float32(0.0)
                step = pair_step(step_key) * step_l_sim
                r_free, dlog_free, local_time, hit = _move_mt(r, step)
                p_stick = bind_probability(_kappa_over_D_mt, local_time)
                newly = (~is_bound) & (jax.random.uniform(stick_key, dtype=jnp.float32) < p_stick)
                u_dwell = jax.random.uniform(dwell_key, dtype=jnp.float32)
                dwell_draw = -jnp.log(jnp.maximum(u_dwell, jnp.float32(1e-20))) * _dwell_steps_mean
                r_next = jnp.where(is_bound, r, r_free)
                bound_next = jnp.where(is_bound, bound_rem - jnp.float32(1.0), jnp.where(newly, dwell_draw, jnp.float32(0.0)))
                dlog_cum = dlog_cum + jnp.where(is_bound | newly, jnp.float32(0.0), dlog_free)
                if hit is not None:
                    bad = bad + hit.illegal.astype(jnp.int32)
                    work = _work_add(work, hit)
                return (r_next, key, bound_next, dlog_cum, bad, work), None

        # Seed each walker's carried compartment ONCE, from its t=0 position, and let only a
        # granted crossing change it thereafter (MC/DC's `initial_location`).
        _illegal_crossings = [0]

        if _carries_side:
            _cls = geometry.classify_position

            def _side0(r_b):
                # -1 = intra (inside some object), +1 = extra. Matches `side < 0 == intra`.
                ids = jax.vmap(_cls)(r_b)
                return jnp.where(ids > 0, jnp.int8(-1), jnp.int8(1))
        else:
            def _side0(r_b):
                return jnp.zeros((r_b.shape[0],), dtype=jnp.int8)

        _, r0_all, walker_keys_all = seed_walkers(geometry, n_walkers, seed, r0)   # r0_all (n_walkers, 3)

        comp0_all = (jnp.asarray(geometry._init_compartments)
                     if (record and uses_myelin_traj) else None)

        # ── MT bound-pool equilibration (kappa_MT > 0) ──
        # An all-free start under-fills the macromolecular pool and biases the transfer;
        # equilibrate the bound occupancy (and spatial state) to f_b BEFORE t=0 and discard
        # the preamble.  kappa_MT == 0 leaves brem0_all at zero and skips this entirely.
        brem0_all = jnp.zeros((n_walkers,), dtype=jnp.float32)
        if _mt_on:
            from . import mt as _mt
            _mode = _mt.resolve_equilibrate_mode(equilibrate_binding, geometry)
            if _mode == "fast":                 # seed f_b walkers bound with a residual dwell (a known S/V)
                S_V = _mt.surface_to_volume(geometry)
                P_eq = _mt.bound_fraction(kappa_MT, dwell_time, S_V)
                ek = jax.random.fold_in(jax.random.PRNGKey(int(seed)), 0xB0117E)
                u_state, u_dwell = jax.random.uniform(ek, (2, n_walkers), dtype=jnp.float32)
                brem0_all = jnp.where(u_state < jnp.float32(P_eq),
                                      -jnp.log(jnp.maximum(u_dwell, jnp.float32(1e-20))) * jnp.float32(float(dwell_time) / dt_sim),
                                      jnp.float32(0.0))
            elif _mode == "burnin":
                _n_chunk = max(4, int(round(float(dwell_time) / float(dt_sim))))
                if _mt_walls:
                    def _burn_walker(r_w, key_w, comp_w, brem_w):
                        (r_f, key_f, brem_f, _d, _b, _w), _ = jax.lax.scan(
                            inner_step_mt, (r_w, key_w, brem_w, jnp.float32(0.0), jnp.int32(0), _work0()), None, length=_n_chunk)
                        return r_f, key_f, comp_w, brem_f, (brem_f > 0).astype(jnp.float32)
                else:
                    def _burn_walker(r_w, key_w, comp_w, brem_w):
                        (r_f, _r_uw, key_f, _da, comp_f, brem_f, bacc, _w), _ = jax.lax.scan(
                            _inner_pm, (r_w, r_w, key_w, jnp.float32(0.0), comp_w, brem_w,
                                        jnp.float32(0.0), _work0()), None, length=_n_chunk)
                        return r_f, key_f, comp_f, brem_f, bacc / jnp.float32(_n_chunk)
                _burn = jax.jit(jax.vmap(_burn_walker, in_axes=(0, 0, 0, 0)))

                _comp = [comp0_all if comp0_all is not None else jnp.zeros((n_walkers,), jnp.int32)]   # the compartment state rides along the chunks

                def _chunk(r, k, brem):
                    r, k, _comp[0], brem, bf = _burn(r, k, _comp[0], brem)
                    return r, k, brem, jnp.mean(bf)
                r0_all, walker_keys_all, brem0_all, _occ_prev, _converged = _mt.equilibrate_burnin_plateau(
                    _chunk, r0_all, walker_keys_all, brem0_all)
                if comp0_all is not None:
                    comp0_all = _comp[0]
                log.info(f"  [mt] equilibrate 'burnin': <bound>={_occ_prev:.4f}")
                if not _converged:
                    import warnings
                    warnings.warn("equilibrate_binding: bound occupancy did not plateau within "
                                  "the burn-in cap; the saved walk may be under-equilibrated.",
                                  stacklevel=2)

        def _work_record():
            """The refused-step count and the work record of the walk, with their warnings."""
            # ── Refused-step report ─────────────────────────────────────────────────
            # A step that leaves the walker in another pool WITHOUT a granted crossing is illegal by
            # definition, and a geometry that detects one holds the walker still instead. So this is a
            # diagnostic rather than a loss -- but a nonzero count is the engine refusing part of the
            # walk, and the number belongs in the open where it can be seen.
            illegal = int(_illegal_crossings[0])
            if illegal:
                import warnings
                warnings.warn(
                    f"{illegal} walker-steps were refused because they ended in another pool without a "
                    f"granted crossing (permeability={permeability!r}); each walker was held at its own "
                    f"position for that step. See CoefficientWalk.illegal_crossings.",
                    RuntimeWarning, stacklevel=3)

            # ── The work record: what the walk cost in wall encounters, for a cost model per substrate class
            # (dmrai-lab/tessera#12) and for sizing the bounce budget -- a budget no lane exhausted gives the same
            # trajectories as any larger one, by the loop's own rule. A myelinated geometry has no `bounce_loop`
            # of its own (its loop is sized per walk, from the step's `step_max`, inside the kernel build
            # -- physics.make_myelin_substep / geometry.myelin.concentric_wall_kernel -- and carried on the
            # trajectory step function instead); read THAT loop when the geometry has none, so the record names
            # the budget that actually ran rather than reporting None for a walk that did use one.
            _loop = geometry.bounce_loop
            if _loop is None and step_fn_traj_pm is not None:
                _loop = step_fn_traj_pm.bounce_loop
            _counted = _counting and _work[3]
            work = dict(walker_steps=int(n_walkers) * int(n_blocks) * int(sub_steps),
                        backend=_backend_name,
                        step_rule=dict(step_rule, dt_p=float(dt_sim), sub_steps=int(sub_steps)),
                        bounce_budget=(None if _loop is None else int(_loop.budget)),
                        n_hits=(int(_work[0]) if _counted else None),
                        n_crossings=(int(_work[1]) if _counted else None),
                        exhausted_steps=(int(_work[2]) if _work[3] else None))
            if _work[3] and _work[2]:
                import warnings
                warnings.warn(
                    f"{_work[2]} walker-steps exhausted the bounce budget of {work['bounce_budget']} with path left "
                    f"untested; the budget is too small for this substrate at this step. See CoefficientWalk.work.",
                    RuntimeWarning, stacklevel=3)

            return illegal, work

        if _backend is not None:
            # another execution of the same physics (#635): the paths only the JAX kernels implement are refused,
            # never fallen back from; a backend walks each batch in the coefficient form (#747)
            reason = _backend_refusal(_backend, _backend_name, res, geometry)
            if reason:
                raise ValueError(reason)

        from ..replay.compression import cosine_band_scale
        from ..coefficient_walk import CoefficientWalk, EventList
        K, m = int(res.K), int(res.block_order)
        n_win = int(res.n_windows)
        n_bw = n_blocks // n_win                                  # blocks per window
        T_win = n_bw * float(dt_actual)
        h_f32 = jnp.float32(dt_actual)
        scale = jnp.asarray(cosine_band_scale(K, T_win), jnp.float32)
        n_raw = int(min(max(int(raw_subset), 0), n_walkers))
        crossing, binding = bool(res.crossing), bool(res.binding_mt)
        # the kind: what the accumulation reads off the physics step
        if record and uses_myelin_traj:
            kind_name = "pm"
            pos_of = (lambda c: c[1])
            kind = Kind(inner=_inner_pm, pos_of=pos_of, read_pos_of=(lambda c: c[0]), blt_of=(lambda c: c[3]),
                        label_of=(lambda c: _compress_comp_pm(c[4])),
                        bound_of=((lambda c: (c[5] > 0).astype(jnp.int8)) if _mt_on else None))
        elif _mt_walls:
            kind_name = "mt"
            kind = Kind(inner=inner_step_mt, pos_of=(lambda c: c[0]), read_pos_of=(lambda c: c[0]), blt_of=(lambda c: c[3]),
                        label_of=None, bound_of=(lambda c: (c[2] > 0).astype(jnp.int8)))
        elif record:
            kind_name = "relax"
            kind = Kind(inner=inner_step_relax, pos_of=(lambda c: c[0]), read_pos_of=(lambda c: c[0]), blt_of=(lambda c: c[2]),
                        label_of=(lambda c: _pool2(c[5])), bound_of=None)
        else:
            kind_name = "gen"
            kind = Kind(inner=inner_step, pos_of=(lambda c: c[0]), read_pos_of=(lambda c: c[0]))
        has_blt = kind.blt_of is not None
        has_lab = kind.label_of is not None
        ev_comp_on = has_lab and crossing
        ev_bound_on = kind.bound_of is not None and binding

        def _walker(raw, cap, K_c3, rd):
            """One walker's walk, window by window: per window its coefficients, contact, field modes, quadratic
            variation and the float32 chain's residual; the events of its label and bound state (a buffer of
            ``cap`` entries each, counted past it); for the raw subset its positions, contact, label and bound state
            at every block end and its field reads. ``K_c3`` is the field's band (``None``: the field is read for the
            pilot alone); ``rd = (n_read, dt_f, checks, reads)`` the read every ``n_read`` steps of ``dt_f`` seconds
            in all, the trapezoid's ``checks`` of a pilot that reads every step, and whether the raw subset keeps the
            reads at their instants (``reads``) or each block's mean (a pilot's, for the field band)."""
            _n_read, _dt_f, _checks, _keep_reads = rd
            def one(r0_w, key_w, side_w, comp0_w, brem0_w, f_arrays):
                read = _reader(f_arrays)
                if kind_name == "gen":
                    phys0 = (r0_w, key_w, side_w, jnp.int32(0), _work0())
                    lab0 = jnp.int8(0)
                elif kind_name == "relax":
                    phys0 = (r0_w, key_w, jnp.float32(0.0), side_w, jnp.int32(0), comp0_w, _work0())
                    lab0 = _pool2(comp0_w)
                elif kind_name == "mt":
                    phys0 = (r0_w, key_w, brem0_w, jnp.float32(0.0), jnp.int32(0), _work0())
                    lab0 = _pool2(comp0_w)
                else:
                    phys0 = ((r0_w, r0_w, key_w, jnp.float32(0.0), comp0_w, brem0_w, jnp.float32(0.0), _work0()) if _mt_on
                             else (r0_w, r0_w, key_w, jnp.float32(0.0), comp0_w, _work0()))
                    lab0 = _compress_comp_pm(comp0_w)
                bnd0 = (brem0_w > 0).astype(jnp.int8)
                b_prev0 = read(kind.read_pos_of(phys0)) if read is not None else None
                ev_c0 = events_init(lab0, cap) if ev_comp_on else None
                ev_b0 = events_init(bnd0, cap) if ev_bound_on else None
                carry0 = (phys0, b_prev0, ev_c0, ev_b0)

                def interval(c, b):
                    phys_a = c[0]
                    blt_a = kind.blt_of(phys_a) if has_blt else None
                    c, mom = _moment_scan(kind, c, sub_steps, m, read=read, n_ch=_n_ch, n_read=_n_read, dt_f=_dt_f,
                                          step0=b * sub_steps, checks=_checks, keep_reads=(raw and _keep_reads))
                    phys = c[0]
                    ys = (kind.pos_of(phys), (kind.blt_of(phys) - blt_a) if has_blt else None,
                          kind.label_of(phys) if has_lab else None, kind.bound_of(phys) if kind.bound_of is not None else None,
                          (None if read is None else mom[8] if _keep_reads else mom[4][0] / h_f32))
                    return c, ys, mom

                emit = (lambda ys: ys) if raw else None
                count = None if (not has_lab or crossing) else (lambda ys: ys[2] != lab0)

                def window(c, w):
                    r_start = kind.pos_of(c[0])
                    c, ap, ab, af, e, n, (qp, qb, qf, qd, ec), = ring_scan(interval, c, n_bw, K, m, has_blt=has_blt, K_c3=K_c3,
                                                                           n_ch=_n_ch, emit=emit, count=count, offset=w * n_bw,
                                                                           n_checks=len(_checks))
                    C, resid = coefficient_form(r_start, kind.pos_of(c[0]), ap, scale)
                    blt = (jnp.zeros((0,), jnp.float32), jnp.float32(0.0)) if not has_blt else (scale * ab[1:], ab[0])
                    fc = jnp.zeros((0, 0), jnp.float32) if af is None else field_mode_form(af, T_win)
                    return c, (C, resid, blt, qp, (jnp.float32(0.0) if qb is None else qb), fc,
                               (jnp.zeros((0,), jnp.float32) if qf is None else qf), (jnp.zeros((0,), jnp.float32) if qd is None else qd),
                               (jnp.zeros((2, 0, 0), jnp.float32) if ec is None else ec), (jnp.int32(0) if n is None else n), e)

                c, (C, resid, blt, qp, qb, fc, qf, qd, ec, n, e) = jax.lax.scan(window, carry0, jnp.arange(n_win, dtype=jnp.int32))
                flat = lambda a: None if a is None else a.reshape((n_win * n_bw,) + a.shape[2:])
                e = (None,) * 5 if e is None else tuple(flat(x) for x in e)
                z = jnp.zeros((0,), jnp.float32)
                pos = jnp.concatenate([r0_w[None], e[0]], axis=0).astype(jnp.float32) if raw else z     # stored float32
                dlog = jnp.concatenate([jnp.zeros((1,), jnp.float32), e[1]]).astype(jnp.float32) if (raw and has_blt) else z
                occ = jnp.concatenate([lab0[None], e[2]]) if (raw and e[2] is not None) else jnp.zeros((0,), jnp.int8)
                bnd = jnp.concatenate([bnd0[None], e[3]]) if (raw and e[3] is not None) else jnp.zeros((0,), jnp.int8)
                fs = (jnp.concatenate([b_prev0[None], e[4].reshape((-1, _n_ch))], axis=0) if (raw and e[4] is not None)
                      else jnp.zeros((0, 0), jnp.float32))         # the reads at their instants, or each block's mean
                phys, _bp, ev_c, ev_b = c
                evz = (jnp.int32(0), jnp.zeros((0,), jnp.int32), jnp.zeros((0,), jnp.int8))
                ev_c_out = evz if ev_c is None else (ev_c.n, ev_c.steps, ev_c.labels)
                ev_b_out = evz if ev_b is None else (ev_b.n, ev_b.steps, ev_b.labels)
                work_f = phys[-1]
                bad = phys[3] if kind_name == "gen" else (phys[4] if kind_name in ("relax", "mt") else jnp.int32(0))
                return (C, resid, blt[1], blt[0], lab0, bnd0, fc, qf, qd, ec, occ, bnd, fs, pos, dlog, qp, qb,
                        ev_c_out, ev_b_out, n.sum(), bad, work_f)
            return one

        def _batch_fn(raw, cap, K_c3, rd):
            return cached_batch(geometry, ("coefficient walk", kind_name, raw, cap, K_c3, crossing, binding, n_blocks, n_win,
                                           sub_steps, float(dt_sim), diffusivity, K, m, kappa_MT, dwell_time, _counting,
                                           _f_key, rd),
                                lambda: jax.jit(jax.vmap(_walker(raw, cap, K_c3, rd), in_axes=(0, 0, 0, 0, 0, None))))

        names = ("C", "resid", "blt_end", "blt_bands", "lab", "bnd", "fc", "qv_f", "qv_d", "chk", "occ", "bound", "fs", "pos", "dlog",
                 "qv_pos", "qv_blt")
        dev = [0]
        _rd = [(1, float(dt_sim), (), True)]                # the read the batches take: the step's, then the walk's

        def _call(a, b, raw, cap, K_c3, rd=None):
            sel = slice(a, b) if b is not None else np.asarray(a)          # a range of walkers, or a pilot's indices
            r0_b = r0_all[sel]
            comp0_b = (comp0_all[sel] if kind_name == "pm" else
                       jnp.asarray(geometry.classify_positions_exact(r0_b), jnp.int32) if kind_name in ("relax", "mt")
                       else jnp.zeros((r0_b.shape[0],), jnp.int32))
            out = _batch_fn(raw, cap, K_c3, _rd[0] if rd is None else rd)(r0_b, walker_keys_all[sel], _side0(r0_b), comp0_b,
                                                                       brem0_all[sel], _f_arrays)
            *arrs, ev_c, ev_b, n_dev, bad_f, work_f = out
            got = {k: np.asarray(v) for k, v in zip(names, arrs)}
            for key, ev in (("c", ev_c), ("b", ev_b)):                      # the events compacted per batch: (counts, t, label)
                n_ev, steps_ev, labs_ev = (np.asarray(x) for x in ev)
                got["ev_" + key] = _compact_events(n_ev, steps_ev, labs_ev, dt_sim)
                got["ev_max_" + key] = int(n_ev.max()) if n_ev.size else 0  # the count reached, buffer or no buffer
            return got, int(jnp.sum(bad_f)), work_f, int(jnp.sum(n_dev))

        _backend_block = {}                                                  # how the backend accumulated: precision, window, summation

        def _backend_call(a, b, raw, cap, K_c3, rd=None):
            """One batch through the backend (#747): its :class:`~dmipy_sim.engine.backends.CoefficientBatch` laid out
            under the names of the JAX batch, the walk converting nothing -- a backend that hands back a walk of saves
            is refused by name."""
            from .backends import CoefficientBatch
            sel = slice(a, b) if b is not None else np.asarray(a)          # a range of walkers, or a pilot's indices
            n_b = int(r0_all[sel].shape[0])
            req = res.request._replace(raw_subset=(n_b if raw else 0), K_c3=int(K_c3 or 0),
                                       events=(int(cap) if cap else res.request.events))
            r0_b = np.ascontiguousarray(np.asarray(r0_all[sel]), np.float32)
            keys_b = np.ascontiguousarray(np.asarray(walker_keys_all[sel]), np.uint32)
            batch = _backend.walk_batch(req, r0_b, keys_b)
            if not isinstance(batch, CoefficientBatch):
                raise ValueError(f"backend {_backend_name!r} returned a {type(batch).__name__} for the coefficient walk; the "
                                 f"producer takes a CoefficientBatch (the bands accumulated in the walk) and converts "
                                 f"nothing: a walk of saves is refused (dmipy-sim#747)")
            if (int(batch.K), int(batch.m), int(batch.sub_steps)) != (K, m, int(sub_steps)):
                raise ValueError(f"backend {_backend_name!r} accumulated (K, m, sub_steps) = {(batch.K, batch.m, batch.sub_steps)} "
                                 f"for a request of {(K, m, int(sub_steps))}")
            C = np.asarray(batch.position_coeffs, np.float64)
            if C.shape != (n_b, K + 2, 3):
                raise ValueError(f"backend {_backend_name!r} returned position coefficients of shape {C.shape}; "
                                 f"the batch of {n_b} walkers at K = {K} has {(n_b, K + 2, 3)}")
            _backend_block.update(precision=str(batch.precision), ring=batch.ring, accumulation=batch.accumulation,
                                  resid_known=batch.k0_residual is not None)
            z1 = np.zeros((n_b, 1), np.float32)
            got = {"C": C[:, None].astype(np.float32),
                   "resid": (np.zeros((n_b, 3), np.float32) if batch.k0_residual is None else np.asarray(batch.k0_residual, np.float32)),
                   "blt_end": (z1 if batch.boundary_endpoint is None else np.asarray(batch.boundary_endpoint, np.float32)[:, None]),
                   "blt_bands": (np.zeros((n_b, 1, K), np.float32) if batch.boundary_bands is None else np.asarray(batch.boundary_bands, np.float32)[:, None]),
                   "lab": (np.zeros((n_b,), np.int8) if batch.compartment is None else np.asarray(batch.compartment, np.int8)),
                   "bnd": np.zeros((n_b,), np.int8),
                   "fc": (np.zeros((n_b, 1, 0, 0), np.float32) if batch.field_bands is None
                          else np.transpose(np.asarray(batch.field_bands, np.float32), (0, 2, 1))[:, None]),
                   "qv_f": (np.zeros((n_b, 1, 0), np.float32) if batch.field_qv is None else np.asarray(batch.field_qv, np.float32)[:, None]),
                   "qv_d": (np.zeros((n_b, 1, 0), np.float32) if batch.field_qv_increments is None
                            else np.asarray(batch.field_qv_increments, np.float32)[:, None]),
                   "chk": np.zeros((n_b, 1, 2, 0, 0), np.float32),
                   "qv_pos": (np.zeros((n_b, 1, 3), np.float32) if batch.qv_positions is None else np.asarray(batch.qv_positions, np.float32)[:, None]),
                   "qv_blt": (z1 if batch.qv_boundary is None else np.asarray(batch.qv_boundary, np.float32)[:, None])}
            sv = batch.saves
            got["pos"] = np.zeros((0,), np.float32) if sv is None else np.asarray(sv.positions, np.float32)
            got["dlog"] = np.zeros((0,), np.float32) if (sv is None or sv.boundary_local_time is None) else np.asarray(sv.boundary_local_time, np.float32)
            got["occ"] = np.zeros((0,), np.int8) if (sv is None or sv.compartment is None) else np.rint(np.asarray(sv.compartment)).astype(np.int8)
            got["bound"] = np.zeros((0,), np.int8); got["fs"] = np.zeros((0, 0), np.float32)
            for key, ev, on in (("c", batch.events, ev_comp_on), ("b", batch.bound_events, ev_bound_on)):
                if ev is None:
                    if on and key == "c":
                        raise ValueError(f"backend {_backend_name!r} returned no C1 event list for a walk whose walkers can change "
                                         f"pool; the event list is the walk's occupancy (RPK.md 9.5.7) and nothing is derived in its place")
                    got["ev_" + key] = (np.zeros((n_b,), np.int64), np.zeros((0,), np.float64), np.zeros((0,), np.int8))
                    got["ev_max_" + key] = 0
                    continue
                off, t, lab = (np.asarray(x) for x in ev)
                counts = np.diff(off.astype(np.int64))
                got["ev_" + key] = (counts, np.asarray(t, np.float64), np.asarray(lab, np.int8))
                got["ev_max_" + key] = int(counts.max()) if counts.size else 0
            work_b = tuple(None if w is None else np.asarray(w) for w in batch.work)
            return got, int(np.asarray(batch.illegal, np.int64).sum()), work_b, 0

        _walk_batch = _call if _backend is None else _backend_call

        # ── The pilots: the events buffer's capacity (the largest count a pilot walker reached, with a margin) and
        # the field's band (the ladder on the pilot's block means), each a walk of the first PILOT_WALKERS walkers
        # thrown away afterwards; the production program is compiled once they are known.
        from .accumulate import pilot_walkers
        pidx = pilot_walkers(n_walkers)                    # spread over the walk: its walkers are seeded pool by pool
        n_pilot = int(pidx.shape[0])
        cap = res.events_capacity
        events_record = None
        if (ev_comp_on or ev_bound_on) and cap is None:
            run.phase("events pilot", walkers=n_pilot)
            got, _, _, _ = _walk_batch(pidx, None, False, 0, None)
            top = max(got["ev_max_c"] if ev_comp_on else 0, got["ev_max_b"] if ev_bound_on else 0)
            cap = int(np.ceil(EVENTS_MARGIN[0] * top)) + int(EVENTS_MARGIN[1])
            events_record = dict(rule="pilot", pilot_walkers=n_pilot, pilot_max_changes=int(top), capacity=cap,
                                 margin=dict(factor=float(EVENTS_MARGIN[0]), plus=int(EVENTS_MARGIN[1])))
        elif ev_comp_on or ev_bound_on:
            cap = int(cap)
            events_record = dict(rule="given", capacity=cap)
        else:
            cap = 0
        K_c3_ = res.K_c3
        field_band = None
        if _sampling:
            # the field's pilot reads every step: the read series' quadratic variation per pool (the read's rule), the
            # trapezoid's constant at every coarser read the block allows (what the walk's read is charged at, given
            # or derived), and the block means (the field band's ladder)
            from .accumulate import read_checks, field_read_of_pilot, field_rule_scale, FIELD_READ_FLOOR
            import time as _time
            checks = read_checks(sub_steps)
            run.phase("field pilot", walkers=n_pilot)
            _t0 = _time.time()
            got, _, _, _ = _walk_batch(pidx, None, K_c3_ is None, 0, None, (1, float(dt_sim), checks, False))
            B0, chi = field_rule_scale(geometry.spec, envelope)
            qd = np.asarray(got["qv_d"], np.float64)                                   # (n_pilot, n_win, n_ch)
            chk = np.asarray(got["chk"], np.float64)                                   # (n_pilot, n_win, 2, n_checks, n_ch)
            labels = np.asarray(got["lab"]).astype(np.int64) if has_lab else np.zeros(qd.shape[0], np.int64)
            name_of = {int(p.id): p.name for p in (geometry.spec.pools if geometry.spec is not None else [])}
            pools = {name_of.get(int(l), str(int(l))): (qd[labels == l].mean(axis=(0, 1)), int((labels == l).sum()))
                     for l in np.unique(labels)}
            field_read = field_read_of_pilot(pools, B0=B0, chi=chi, step_s=float(dt_sim), steps=int(sub_steps), T_e=float(T_win),
                                             floor=FIELD_READ_FLOOR, given=field_read_interval, checks=checks,
                                             err_sq=(chk[:, :, 0] ** 2).mean(axis=(0, 1)) if checks else None,
                                             qv_checks=chk[:, :, 1].mean(axis=(0, 1)) if checks else None, qv_all=qd.mean(axis=(0, 1)),
                                             channels=list(field_basis.channel_names))
            field_read.update(pilot_walkers=int(n_pilot), pilot_seconds=float(_time.time() - _t0))
            res = res._replace(field_read=field_read)
            if K_c3_ is None:
                from ..replay.bank import derive_field_band
                K_c3_, bits, field_band = derive_field_band(np.asarray(got["fs"], np.float32)[:, 1:], list(field_basis.channel_names),
                                                             float(dt_actual), envelope, K_max=K, B0=B0, chi=chi,
                                                             floor_scale=float(np.sqrt(n_pilot / float(n_walkers))))
                field_band = dict(field_band, pilot_walkers=n_pilot, production_walkers=int(n_walkers), bits=int(bits))
        if _sampling:
            K_c3_ = int(K_c3_)
            if K_c3_ < 1 or K_c3_ > n_bw:
                raise ValueError(f"K_c3 is the field's modes per window, 1..{n_bw} (the blocks of a window); got {K_c3_}")
            _n_read, _dt_f = int(res.field_read["n_read"]), float(res.field_read["dt_f"])
            _rd[0] = (_n_read, _dt_f, (), True)                                        # the walk's read; the raw subset keeps the reads
        else:
            K_c3_ = None

        rows = {k: _Rows(n_walkers) for k in names}
        raw_rows = {k: _Rows(n_raw) for k in ("pos", "dlog", "occ", "bound", "fs")} if n_raw else None
        ev_parts = dict(c=[], b=[])
        ev_overflow = dict(c=0, b=0)

        def _run(a, b, raw):
            try:
                return _walk_batch(a, b, raw, cap, K_c3_)
            except Exception as e:
                msg = str(e)
                if not ("OOM" in msg or "out of memory" in msg.lower() or "RESOURCE_EXHAUSTED" in msg) or b - a < 2000:
                    raise
                mid = (a + b) // 2
                log.info(f"  OOM: halving the batch to {mid - a}")
                lo, hi = _run(a, mid, raw), _run(mid, b, raw)
                got = {k: np.concatenate([lo[0][k], hi[0][k]], axis=0) for k in names}
                for key in ("ev_c", "ev_b"):
                    got[key] = tuple(np.concatenate([x, y], axis=0) for x, y in zip(lo[0][key], hi[0][key]))
                    got["ev_max_" + key[-1]] = max(lo[0]["ev_max_" + key[-1]], hi[0]["ev_max_" + key[-1]])
                return got, lo[1] + hi[1], None, lo[3] + hi[3]

        run.phase("walk")
        for start, end in run.batches(n_walkers, walker_batch_size):
            pieces = ([(start, min(end, n_raw), True)] if start < n_raw else []) + \
                     ([(max(start, n_raw), end, False)] if end > n_raw else [])
            for a, b, raw in pieces:
                got, bad, work_f, n_dev = _run(a, b, raw)
                _illegal_crossings[0] += bad
                _work_acc(work_f)
                dev[0] += n_dev
                for k in names:
                    if k in ("pos", "dlog", "occ", "bound", "fs"):
                        continue
                    if k == "chk" or (k in ("fc", "qv_f", "qv_d") and not _sampling):
                        continue
                    rows[k].append(got[k])
                if raw:
                    raw_rows["pos"].append(np.asarray(got["pos"], np.float32))
                    raw_rows["dlog"].append(np.asarray(got["dlog"], np.float32))
                    raw_rows["occ"].append(np.asarray(got["occ"], np.int8))
                    raw_rows["bound"].append(np.asarray(got["bound"], np.int8))
                    raw_rows["fs"].append(np.asarray(got["fs"], np.float32))
                for key, on in (("c", ev_comp_on), ("b", ev_bound_on)):
                    if on:
                        ev_overflow[key] = max(ev_overflow[key], int(got["ev_max_" + key]))
                        ev_parts[key].append(got["ev_" + key])
        for key, on, what in (("c", ev_comp_on, "pool crossings"), ("b", ev_bound_on, "bind / unbind events")):
            if on and ev_overflow[key] > cap:
                raise ValueError(f"a walker recorded {ev_overflow[key]} {what} where the events buffer holds {cap}: the walk is "
                                 f"refused rather than an event dropped (RPK.md 9.5.7). Walk again with events_capacity={int(np.ceil(1.5 * ev_overflow[key])) + 8} "
                                 f"(the pilot of {n_pilot} walkers saw at most {events_record.get('pilot_max_changes', '?')})")
        illegal, work = _work_record()
        resid = rows["resid"].array()
        disp = rows["C"].array()[:, :, 1, :]
        n_dev = int(dev[0])
        if n_dev:
            import warnings
            warnings.warn(f"{n_dev} walker-blocks ended on a pool label other than the walker's own on a substrate no walker "
                          f"leaves its pool in; the walk records each walker's start label (CoefficientWalk.compartment)",
                          RuntimeWarning, stacklevel=2)
        records_surface = record and (has_reflect_with_log_weight or geometry._is_myelinated or geometry._is_packed_myelinated)
        qv_p = rows["qv_pos"].array()
        if _backend is None:
            precision, ring, accumulation, resid_known = PRECISION, int(RING), ACCUMULATION, True
        else:                                         # the backend's own accumulation, as it reported it
            precision = _backend_block.get("precision", "?"); ring = _backend_block.get("ring")
            accumulation = _backend_block.get("accumulation") or f"accumulated by the {_backend_name} backend"
            resid_known = bool(_backend_block.get("resid_known", False))
        block = dict(h=float(dt_actual), m=m, precision=precision, n_blocks=n_blocks, rule=res.block_rule, ring=ring,
                     accumulation=accumulation,
                     k0_residual_m=((float(np.abs(resid).max()) if resid.size else 0.0) if resid_known else None),
                     k0_residual_relative=((float(np.abs(resid).max() / max(np.sqrt(np.mean(disp ** 2)), 1e-300))
                                            if resid.size else 0.0) if resid_known else None),
                     quadratic_variation=dict(
                         positions_mean_m2=[float(x) for x in qv_p.sum(axis=1).mean(axis=0)],
                         positions_over_2DT=([float(x) / (2.0 * float(diffusivity) * float(T_max)) for x in qv_p.sum(axis=1).mean(axis=0)]
                                             if diffusivity else None),
                         boundary_mean_m2=(float(rows["qv_blt"].array().sum(axis=1).mean()) if records_surface else None)))
        if _sampling:
            block["field_read"] = dict(res.field_read)
            block["field_band"] = dict(field_band) if field_band is not None else dict(rule="given", K=int(K_c3_))
        if events_record is not None:
            block["events"] = events_record
        work = dict(work, block=block, label_deviations=n_dev)

        def _events(key):
            parts = ev_parts[key]
            counts = np.concatenate([p[0] for p in parts]) if parts else np.zeros(0, np.int64)
            return EventList(np.concatenate([[0], np.cumsum(counts)]), np.concatenate([p[1] for p in parts]) if parts else np.zeros(0),
                             np.concatenate([p[2] for p in parts]) if parts else np.zeros(0, np.int8))

        comp = rows["lab"].array().astype(np.int8) if has_lab else None
        comp_events = _events("c") if ev_comp_on else None
        bound_start = rows["bnd"].array().astype(np.int8) if ev_bound_on else None
        bound_events = _events("b") if ev_bound_on else None
        raw_comp = raw_bound = raw_field = None
        if n_raw:
            raw_comp = raw_rows["occ"].array() if has_lab else None
            raw_bound = raw_rows["bound"].array() if ev_bound_on else None
            raw_field = raw_rows["fs"].array() if _sampling else None
        cw = CoefficientWalk(
            position_coeffs=np.asarray(rows["C"].array(), np.float32), T=float(T_max), K=K, m=m, block=block,
            n_windows=n_win, dt_sim=float(dt_sim), sub_steps=int(sub_steps),
            boundary_endpoint=(rows["blt_end"].array().astype(np.float32) if records_surface else None),
            boundary_bands=(rows["blt_bands"].array().astype(np.float32) if records_surface else None),
            compartment=comp, comp_events=comp_events, bound_start=bound_start, bound_events=bound_events,
            field_basis=(field_basis if _sampling else None),
            field_coeffs=(np.asarray(rows["fc"].array(), np.float32) if _sampling else None),
            K_c3=K_c3_, field_read_interval=(float(_dt_f) if _sampling else None),
            qv_field=(np.asarray(rows["qv_f"].array(), np.float32) if _sampling else None),
            qv_field_increments=(np.asarray(rows["qv_d"].array(), np.float32) if _sampling else None),
            raw_positions=(None if not n_raw else np.asarray(raw_rows["pos"].array(), np.float32)),
            raw_boundary=(None if not (n_raw and records_surface) else raw_rows["dlog"].array()),
            raw_compartment=raw_comp, raw_bound=raw_bound, raw_field=raw_field,
            qv_positions=qv_p.astype(np.float32),
            qv_boundary=(rows["qv_blt"].array().astype(np.float32) if records_surface else None),
            illegal_crossings=illegal, seed=int(seed), diffusivity=(None if diffusivity is None else float(diffusivity)),
            geometry=geometry, work=work)
        object.__setattr__(cw, "run", run)
        return cw


def certify_bounce_budget(geometry, n_walkers, diffusivity, T_max, block_h=None, *, margin=1, seed=0, **walk_kwargs):
    """Size a bounce budget from a pilot walk: the smallest budget no walker-step of the pilot exhausted, plus
    ``margin`` collisions.

    The derived budget (``geometry.bounce_loop.budget``) is the worst case a step of the reflection rule can need
    and is paid by every lane of the fixed-length scan on every sub-step; a walk at a smaller budget that reports
    zero exhausted steps gives the same trajectories as any larger budget, by the loop's own rule. The pilot walks
    at the derived budget first (and refuses, naming the count, if even that is exhausted: the derivation does
    not cover this substrate at this step), then one budget lower at a time until a walker-step is exhausted.
    Measured on the L40S: the sphere's 7 against a certified 5 is 20 % of the walk, the icosphere mesh's 10
    against a certified 7 is 28 %.

    Returns ``dict(derived, smallest_certified, recommended, exhausted={budget: count}, seconds={budget: s})``;
    ``recommended`` is what to pass as ``bounce_budget=`` to the production walk, which certifies itself again.
    """
    import time
    if geometry.bounce_loop is None:
        raise ValueError(f"{type(geometry).__name__} has no bounce loop to size")
    derived = int(geometry.bounce_loop.budget)
    exhausted, seconds = {}, {}
    budget = derived
    while budget >= 1:
        t0 = time.perf_counter()
        w = simulate_trajectories(n_walkers, diffusivity, geometry, T_max=T_max, block_h=block_h, seed=seed, tiers=(),
                                  bounce_budget=budget, **walk_kwargs)
        seconds[budget] = time.perf_counter() - t0
        exhausted[budget] = int(w.work["exhausted_steps"])
        if budget == derived and exhausted[budget]:
            raise ValueError(f"the derived bounce budget {derived} of {type(geometry).__name__} was exhausted on "
                             f"{exhausted[budget]} walker-steps of the pilot: the derivation does not cover this "
                             f"substrate at this step, and no smaller budget can")
        if exhausted[budget]:
            break
        budget -= 1
    smallest = min(b for b, n in exhausted.items() if n == 0)
    return dict(derived=derived, smallest_certified=smallest, recommended=min(derived, smallest + int(margin)),
                exhausted=exhausted, seconds=seconds)
