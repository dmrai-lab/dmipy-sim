"""Main simulation entry point.

simulate() vmaps over walkers, scans over timesteps, and extracts signal.
Magnetisation is treated as fully transverse throughout (instantaneous ideal
pulses), so transverse T2 is accumulated per-walker inside the scan body via
log-weight:

    log_w += -dt / T2    (transverse decay)

Signal = mean(cos(phi) * exp(log_w)) over walkers.
"""

import jax
import jax.numpy as jnp
import contextlib
import dataclasses
from dataclasses import dataclass
from typing import NamedTuple
import numpy as np

from .backends import JAX, WalkRequest, resolve as resolve_backend
import logging

log = logging.getLogger(__name__)

from ..acquisition.waveforms import band_hz
from .physics import (make_step_fn, make_myelin_step_fn, make_packed_myelin_step_fn,
                      make_myelin_traj_step_fn, make_packed_myelin_traj_step_fn,
                      seed_walkers, pair_step)
from ..persistent_walk import PersistentWalk
from ..run import Run, current
from ..acquisition.scanner_sequence import Protocol


# ═══════════════════════════════════════════════════════════════════════════
# ENGINE ROUTING TABLE (Phase 5)
# ═══════════════════════════════════════════════════════════════════════════
# simulate(engine=) selects the backend that turns a walk into a signal:
#   "fused"  -- the single-pass jax.lax.scan step kernels in this module (the validation oracle).
#   "replay" -- walk once (simulate_trajectories -> PersistentWalk), then apply the waveform,
#               scalar / per-compartment T2, T1 and surface relaxivity to the stored walk.
#   "auto"   -- the default: replay when `geometry.replay_parity` (the producer's walk IS the fused
#               walk, validated in tests/test_replay_parity.py) and _replay_gap() finds nothing the
#               replay cannot serve (single-pass outputs, per-compartment D, a missing diffusivity);
#               otherwise fused. The capability is declared on the geometry, never read off its name.

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


def _ensemble_signal(spin_w, phi, log_w=None):
    """Spin-density-weighted ensemble signal ``Re<w exp(log_w) e^{i phi}> / sum w``.

    ``spin_w`` is the per-walker proton-density weight (myelin water < 1); homogeneous placement plus
    this weight avoids per-geometry placement re-weighting. ``log_w`` (per walker) is the relaxation and
    surface log-weight; ``None`` when nothing wrote it.
    """
    amp = spin_w if log_w is None else spin_w * jnp.exp(log_w)
    return jnp.sum(amp[:, None] * jnp.cos(phi), axis=0) / jnp.sum(spin_w)


def _replay_gap(geometry, *, return_positions, return_compartments,
                return_walker_signals, diffusivity):
    """Return a string naming why the replay backend cannot serve this run
    exactly, or ``None`` when replay is a valid substitute for the fused engine.

    This is the single source of truth for both ``engine='replay'`` (raises with
    the reason) and ``engine='auto'`` (falls back to fused)."""
    if return_positions is not False:
        return "return_positions is a fused single-pass internal (no replay equivalent)"
    if return_compartments is not False:
        return "return_compartments is a fused single-pass internal (no replay equivalent)"
    if return_walker_signals:
        return "return_walker_signals is a fused single-pass internal (no replay equivalent)"
    if geometry._is_myelinated:
        return "MyelinatedCylinder uses a dedicated fused step kernel (no replay walk)"
    if geometry._is_packed_myelinated:
        return ("PackedMyelinatedCylinders fused single-reflection kernel is not "
                "position-parity with the multi-bounce replay walk")
    if geometry._D_comp_jax is not None:
        # Per-compartment D changes the STEP LENGTH per compartment, so it alters the walk
        # itself — not a replay knob. (Per-compartment T2/T1 ARE replay knobs: they gate
        # only log_w and are applied off the saved compartment channel — see
        # _simulate_via_replay.)
        return "per-compartment diffusivity (Mesh intra/extra D) alters the walk (fused-only)"
    if diffusivity is None:
        return "replay needs an explicit diffusivity for the sub-step auto-tune"
    return None


def _simulate_via_replay(n_walkers, diffusivity, waveform, geometry, *, seed,
                         T2, T1, r0, require_gpu, walker_batch_size, sub_steps=None):
    """Signal via the replay backend: walk once, then apply the waveform +
    relaxation.  The producer walk depends only on (geometry, diffusivity,
    seed) — the replay invariant — so it reproduces the fused ``simulate()`` to
    the MC-noise floor on the supported matrix (see the routing table).

    Handles gradient phase, scalar T2, T1 (chi_perp-gated longitudinal storage),
    surface relaxivity (ρ replayed off the recorded unit boundary local time),
    and the stimulated-echo 0.5 factor.  Assumes ``_replay_gap()`` already
    cleared the run (no permeability / myelin / per-comp / extra-output)."""
    from ..replay.trajectories import replay

    G = np.asarray(waveform.G_eff, dtype=np.float32)      # (n_meas, n_t, 3), the effective gradient
    dt = float(waveform.dt)
    n_t = G.shape[1]
    T_max = dt * (n_t - 1)

    # Acquisition rotation: place the substrate in the bore by rotating G into
    # the geometry's native frame (walk stays native), exactly as fused does.
    _orient_R = geometry._orient_R
    if _orient_R is not None:
        G = G @ np.asarray(_orient_R, dtype=np.float32)

    # Transverse-coherence schedule (spin echo = all ones; PGSTE zeros the TM).
    chi_perp = getattr(waveform, 'chi_perp', None)
    chi = (np.asarray(chi_perp, dtype=np.float64).reshape(n_t)
           if chi_perp is not None else np.ones(n_t, dtype=np.float64))

    # Surface relaxivity is a wall effect recorded (with ρ/D = 1) during the
    # walk and multiplied by ρ/D at replay — only then do we need the boundary
    # local-time channel.  Scalar T2/T1 are walker-independent replay knobs.
    rho = geometry.surface_relaxivity_t2
    has_surf = (rho is not None and float(rho) > 0.0
                and geometry.has_walls)

    # Per-compartment T2/T1 (Mesh intra/extra dicts) are pure replay knobs: they gate only
    # log_w and are applied off the saved compartment channel (comp_traj, indexed by pool id:
    # 0 extra, 1 intra, matching the geometry's _T2_comp/_T1_comp ordering). Requesting them
    # forces record so the compartment channel is recorded.
    T2_comp = geometry._T2_comp
    T1_comp = geometry._T1_comp
    has_per_comp = T2_comp is not None or T1_comp is not None

    save_relax = has_surf or has_per_comp
    st_kwargs = dict(seed=seed, require_gpu=require_gpu,
                     tiers=("all" if save_relax else ()))
    if r0 is not None:
        st_kwargs['r0'] = r0
    if walker_batch_size is not None:
        st_kwargs['walker_batch_size'] = walker_batch_size

    if sub_steps:
        st_kwargs['sub_steps'] = sub_steps
    # the walk on the waveform's own grid, every walker kept at its block ends (the raw subset is the whole walk), the
    # step resolved at that grid's Nyquist frequency as the fused engine resolves it; one band, which nothing reads
    walk = simulate_trajectories(n_walkers, diffusivity, geometry, T_max, dt, K=1, f_band=0.5 / dt, raw_subset=n_walkers,
                                 window_T=T_max, **st_kwargs).saves()
    traj, dt_traj, dlog, comp = walk.positions, walk.dt, walk.boundary_local_time, walk.compartment

    relax_kw = dict(T2=T2, T1=T1)
    if has_per_comp:                                   # per-compartment overrides scalar
        relax_kw['comp_traj'] = comp
        if T2_comp is not None:
            relax_kw['T2_per_comp'] = np.asarray(T2_comp, dtype=np.float64)
        if T1_comp is not None:
            relax_kw['T1_per_comp'] = np.asarray(T1_comp, dtype=np.float64)
    if has_surf:
        relax_kw.update(dlog_boundary_unit=dlog, surface_relaxivity=float(rho),
                        D=float(diffusivity))

    signals = replay(
        traj, dt_traj, G, dt, chi_perp=chi,
        stimulated_echo=bool(getattr(waveform, 'stimulated_echo', False)),
        **relax_kw)
    return np.asarray(signals, dtype=np.float32)


def simulate(
    n_walkers: int,
    diffusivity=None,
    waveform=None,
    geometry=None,
    seed: int = 123,
    T2: float = None,
    T1: float = None,
    return_positions: bool = False,
    return_compartments=False,
    return_walker_signals: bool = False,
    r0=None,
    walker_batch_size: int = None,
    require_gpu=None,
    engine: str = "auto",
    sub_steps: int = None,
    _allow_oom_backoff: bool = True,
):
    """Run Monte Carlo diffusion simulation.

    Parameters
    ----------
    n_walkers : int
        Number of random walkers.
    diffusivity : float, optional
        Diffusion coefficient in m²/s. Required for standard geometries.
        Omit for MyelinatedCylinder (D values are in the geometry).
    waveform : ScannerSequence
        Gradient waveform. G has shape (n_measurements, n_t, 3).
    geometry : Geometry
        Boundary geometry. Provides init_positions() and reflect().
    seed : int
        Master PRNG seed (split into per-walker keys internally).
    T2 : float, optional
        Transverse relaxation time in seconds. When set, accumulated
        per-walker inside the scan body as ``-chi_t*dt/T2`` each step, where
        ``chi_t`` is the waveform's transverse-coherence flag (1 transverse,
        0 stored longitudinally). A plain spin echo has ``chi_t ≡ 1``.
    T1 : float, optional
        Longitudinal relaxation time in seconds. When set, accumulated
        per-walker as ``-(1-chi_t)*dt/T1`` each step — i.e. only during the
        longitudinal-storage intervals (the ``chi_perp == 0`` block of a
        stimulated echo, e.g. the mixing time of a PGSTE). With an all-
        transverse waveform (spin echo) T1 never acts.
    return_positions : {False, True, 'full'}, optional
        False (default): no positions.  True: final walker positions,
        (n_walkers, 3).  'full': per-timestep positions, (n_walkers, n_timesteps,
        3) — trajectory export for visualisation/analysis (e.g. combine with
        return_compartments='full' to select walkers that permeated).  Supported
        for standard geometries including Mesh; not the myelin step-fn paths.
    return_compartments : {False, 'final', 'full'}, optional
        Controls compartment-ID output.  Default False (no change to return
        value).

        - ``False``: no compartment output.
        - ``'final'``: return ``(compartment_origin, compartment_current_final)``
          as additional outputs.  Both are int32 arrays of shape
          ``(n_walkers,)``.
        - ``'full'``: return ``(compartment_origin, compartment_current_full)``
          where ``compartment_current_full`` has shape
          ``(n_walkers, n_timesteps)`` containing the compartment ID at every
          timestep.

    r0 : array-like of shape (n_walkers, 3), optional
        Custom initial walker positions in metres (lab frame, float32).
        When provided, ``geometry.init_positions()`` is skipped and these
        positions are used directly.  Useful for mixed initial conditions
        (e.g., f·N walkers inside cylinders, (1-f)·N walkers outside)
        required for Karger-model validation.  Default None (use geometry
        default positions).

        Compartment ids follow one convention: 0 is the extra-cellular / free pool and
        enclosed pools are positive.

        - ``Cylinder``, ``Sphere``, ``Ellipsoid``, ``Mesh``: 1 = intra, 0 = extra.
        - ``Box1D``: 1 (the slab).
        - ``MyelinatedCylinder``, ``PackedMyelinatedCylinders``: 0 = extra-axonal,
          1 = intra-axonal, 2 = myelin.
        - ``PackedCylinders``, ``PackedSpheres``: 0 = extra, 1..N = the object the walker
          is in.

    walker_batch_size : int, optional
        If set and smaller than ``n_walkers``, the run is split into walker
        chunks of this size, run one at a time, and recombined.  Peak device
        memory is bounded to one chunk — use this on a small GPU.  Each chunk
        uses an independent sub-seed, so the ensemble signal is statistically
        identical to a single-shot run (not bit-identical).  Default None
        (all walkers at once).
    require_gpu : {None, True, False}, optional
        GPU guard against a silent CPU fallback.  ``True`` raises if no GPU is
        visible; ``False`` opts out (e.g. a CPU float64 reference check);
        ``None`` (default) warns when a large run is about to use the CPU.
    engine : {'auto', 'replay', 'fused'}, optional
        Which backend computes the signal (default ``'auto'``).  See the
        ENGINE ROUTING TABLE at the top of this module.

        - ``'fused'`` — the inline single-pass ``jax.lax.scan`` step-kernels in
          this file (the validation oracle / universal fallback; byte-for-byte
          the pre-replay code).
        - ``'replay'`` — walk once with :func:`simulate_trajectories` then apply
          the waveform + relaxation with
          :func:`~dmipy_sim.replay.trajectories.replay`
          (gradient phase + scalar/per-comp T2 + T1 + surface relaxivity).
          Raises :class:`NotImplementedError` (naming the gap) for a path the
          replay backend cannot serve exactly — MyelinatedCylinder /
          PackedMyelinatedCylinders, per-compartment D meshes, or the
          ``return_positions``/``return_compartments``/``return_walker_signals``
          single-pass internals.
        - ``'auto'`` — route to replay where it is validated-equivalent AND
          keeps the test suite green, else fall back to fused (transparent).

    Returns
    -------
    signals : np.ndarray of shape (n_measurements,), float32
        Normalised signal: Re(<exp(i·phi)>) averaged over walkers.
    positions : np.ndarray of shape (n_walkers, 3), float32
        Final walker positions. Only returned when return_positions=True.
    compartment_origin : np.ndarray of shape (n_walkers,), int32
        Compartment ID at t=0 (set once, immutable). Only returned when
        return_compartments is not False.
    compartment_current : np.ndarray
        - shape (n_walkers,) when return_compartments='final'.
        - shape (n_walkers, n_timesteps) when return_compartments='full'.
        Only returned when return_compartments is not False.
    """
    with Run("simulate", params=dict(n_walkers=n_walkers, diffusivity=diffusivity, geometry=type(geometry).__name__)) as run:
        from ..spec.build import as_geometry
        geometry = as_geometry(geometry)               # a spec, a spec file or a dict is a substrate too
        if return_compartments not in (False, 'final', 'full'):
            raise ValueError(
                "return_compartments must be False, 'final', or 'full'; "
                f"got {return_compartments!r}")
        if return_positions not in (False, True, 'full'):
            raise ValueError(
                "return_positions must be False, True, or 'full'; "
                f"got {return_positions!r}")
        if engine not in ("auto", "replay", "fused"):
            raise ValueError(
                f"engine must be 'auto', 'replay', or 'fused'; got {engine!r}")
        want_pos_full = return_positions == 'full'

        _acq = waveform.waveform if hasattr(waveform, 'waveform') else waveform
        if isinstance(_acq, Protocol):
            # a multi-TE scheme: one sequence per echo time, one walk each, the signals placed at their rows
            if return_positions or return_compartments:
                raise ValueError("positions and compartments are one sequence's: simulate each sequence of the "
                                 "Protocol on its own to get them")
            kw = dict(seed=seed, T2=T2, T1=T1, return_walker_signals=return_walker_signals, r0=r0,
                      walker_batch_size=walker_batch_size, require_gpu=require_gpu, engine=engine, sub_steps=sub_steps)
            parts = [simulate(n_walkers, diffusivity, seq, geometry, **kw) for seq in _acq]
            if return_walker_signals:                                   # (signal, walker_signals) per sequence
                return _acq.scatter([p[0] for p in parts]), _acq.scatter([p[1] for p in parts], axis=-1)
            return _acq.scatter(parts)

        # ── Engine routing (Phase 5) ────────────────────────────────────────────
        # Decide replay vs fused BEFORE the fused-only OOM/batch machinery so the
        # fused code path below stays byte-for-byte the pre-replay engine.  The
        # replay backend produces a bare signal array only; every extra-output /
        # single-pass-internal request is a fused-only gap.
        if engine != "fused":
            _wf_r = waveform.waveform if hasattr(waveform, 'waveform') else waveform
            _gap = _replay_gap(
                geometry, return_positions=return_positions,
                return_compartments=return_compartments,
                return_walker_signals=return_walker_signals,
                diffusivity=diffusivity)
            if engine == "replay":
                if _gap is not None:
                    raise NotImplementedError(
                        f"engine='replay' cannot serve this run: {_gap}. "
                        "Use engine='fused' (or 'auto').")
                return _simulate_via_replay(
                    n_walkers, diffusivity, _wf_r, geometry, seed=seed,
                    T2=T2, T1=T1, r0=r0, require_gpu=require_gpu,
                    walker_batch_size=walker_batch_size, sub_steps=sub_steps)
            # engine == "auto": replay only where the walk is the fused walk and nothing is missing.
            if _gap is None and geometry.replay_parity:
                return _simulate_via_replay(
                    n_walkers, diffusivity, _wf_r, geometry, seed=seed,
                    T2=T2, T1=T1, r0=r0, require_gpu=require_gpu,
                    walker_batch_size=walker_batch_size, sub_steps=sub_steps)
            # else: fall through to the fused engine (pin so recursion stays fused).
            engine = "fused"

        # GPU guard — never silently fall back to CPU for a heavy run (CLAUDE rule).
        from .gpu import check_gpu
        check_gpu(n_walkers, require_gpu, what="simulate")

        # Automatic GPU-OOM backoff: try the requested plan; if device memory is
        # exhausted, split walkers into progressively smaller batches (down to 1)
        # rather than dying with a raw XLA traceback. Pin walker_batch_size to skip.
        if _allow_oom_backoff:
            try:
                from jaxlib.xla_extension import XlaRuntimeError
            except Exception:
                XlaRuntimeError = RuntimeError
            bs = walker_batch_size
            while True:
                try:
                    return simulate(
                        n_walkers, diffusivity=diffusivity, waveform=waveform,
                        geometry=geometry, seed=seed, T2=T2, T1=T1,
                        return_positions=return_positions,
                        return_compartments=return_compartments,
                        return_walker_signals=return_walker_signals, r0=r0,
                        walker_batch_size=bs, require_gpu=require_gpu,
                        engine="fused", sub_steps=sub_steps, _allow_oom_backoff=False)
                except (XlaRuntimeError, RuntimeError) as exc:
                    m = str(exc)
                    if not ('RESOURCE_EXHAUSTED' in m or 'out of memory' in m.lower()):
                        raise
                    cur = bs if bs is not None else n_walkers
                    nxt = cur // 2
                    if nxt < 1:
                        raise
                    import warnings
                    warnings.warn(
                        "simulate() hit GPU OOM at walker_batch_size={}; retrying at "
                        "{}.".format(cur, nxt), RuntimeWarning, stacklevel=2)
                    bs = nxt

        # Walker batching: split into chunks so peak device memory is one chunk.
        if walker_batch_size is not None and walker_batch_size < n_walkers:
            return _simulate_in_walker_batches(
                n_walkers, walker_batch_size, seed=seed,
                diffusivity=diffusivity, waveform=waveform, geometry=geometry,
                T2=T2, T1=T1, r0=r0,
                return_positions=return_positions,
                return_compartments=return_compartments,
                return_walker_signals=return_walker_signals, sub_steps=sub_steps)

        # Accept AcquisitionScheme (any object with .waveform) or raw ScannerSequence
        if hasattr(waveform, 'waveform'):
            waveform = waveform.waveform
        G = waveform.G_eff      # (n_measurements, n_t, 3), the effective gradient
        dt = waveform.dt
        f_band = band_hz(G, dt)                 # the band the step law resolves (physics.tolerance_sub_steps)

        # Substrate placement in the bore (e.g. Mesh with orientation/R): the walk runs
        # in the geometry's native frame, so rotate the ACQUISITION into that frame
        # instead of rotating the geometry.  A gradient g in the lab (B0=+z) frame is
        # g_mesh = R^T g for a mesh->lab rotation R, i.e. G_mesh = G @ R.
        _orient_R = geometry._orient_R
        if _orient_R is not None:
            G = G @ jnp.asarray(_orient_R, G.dtype)

        n_measurements, n_t, _ = G.shape


        # Transpose G for scan: (n_t, n_measurements, 3).  Each step also receives a
        # scalar transverse-coherence flag chi_t: 1 where the magnetisation is
        # transverse (T2 + surface relaxivity act), 0 where it is stored
        # longitudinally (only T1 acts).  A waveform with no chi_perp schedule is a
        # spin echo (chi_t == 1 throughout).  step_fn receives inputs = (g_t, chi_t).
        G_scan = jnp.transpose(G, (1, 0, 2))
        chi_perp = getattr(waveform, 'chi_perp', None)
        if chi_perp is not None:
            chi_perp_scan = jnp.asarray(chi_perp, dtype=jnp.float32).reshape(n_t)
        else:
            chi_perp_scan = jnp.ones((n_t,), dtype=jnp.float32)
        scan_inputs = (G_scan, chi_perp_scan)

        _r0_user_supplied = r0 is not None
        _, r0, walker_keys = seed_walkers(geometry, n_walkers, seed, r0)   # r0 (n_walkers, 3)

        # Check if this is a MyelinatedCylinder or LabelMap2D (custom step function path)
        is_myelin = geometry._is_myelinated
        is_packed_myelin = geometry._is_packed_myelinated

        if want_pos_full and (is_myelin or is_packed_myelin):
            raise NotImplementedError(
                "return_positions='full' is supported for standard geometries "
                "(including Mesh), not MyelinatedCylinder / PackedMyelinatedCylinders.")

        # -----------------------------------------------------------------------
        # Compartment origin: determined from initial positions.
        # For MyelinatedCylinder and LabelMap2D, _init_compartments is set by
        # init_positions().  For standard geometries, classify_position() is used.
        # -----------------------------------------------------------------------
        track_comp = return_compartments is not False

        if is_myelin:
            # MyelinatedCylinder: extended carry state (r, phi, log_w, compartment_id, key)
            step_fn = make_myelin_step_fn(geometry, dt, T1=T1, sub_steps=sub_steps, f_band=f_band)
            compartments0 = geometry._init_compartments  # (n_walkers,) int32, pool id
            spin_w = jnp.asarray(geometry._water_fraction_by_pool, jnp.float32)[compartments0]

            emit_comp = track_comp and return_compartments == 'full'
            if track_comp:
                comp_origin_jax = geometry.pool_of(compartments0)   # kernel code -> pool id

            def _build_myelin():
                def simulate_walker(r0_w, key_w, comp0, G_s, chi_s):
                    phi0 = jnp.zeros(n_measurements, dtype=jnp.float32)

                    def body(carry, inputs):
                        new_carry, _ = step_fn(carry, inputs)
                        return new_carry, (geometry.pool_of(new_carry[3]) if emit_comp else None)
                    (r_final, phi_all, log_w, comp_final, _), comp_seq = jax.lax.scan(
                        body, (r0_w, phi0, jnp.float32(0.0), comp0, key_w), (G_s, chi_s))
                    return r_final, phi_all, log_w, geometry.pool_of(comp_final), comp_seq
                return jax.jit(jax.vmap(simulate_walker, in_axes=(0, 0, 0, None, None)))

            simulate_batch = cached_batch(
                geometry, ("myelin", n_measurements, n_t, float(dt), T1, sub_steps, emit_comp), _build_myelin)
            final_r, all_phi, all_log_w, comp_final, comp_seq = simulate_batch(
                r0, walker_keys, compartments0, *scan_inputs)
            signals = _ensemble_signal(spin_w, all_phi, all_log_w)

        elif is_packed_myelin:
            # Fused forward: the SAME per-compartment walk as the trajectory step fn, with
            # gradient phase (on the periodic-unwrapped position) + per-compartment T2 + surface
            # relaxivity accumulated in-scan. No trajectory storage / replay.
            if _r0_user_supplied:
                raise NotImplementedError(
                    "simulate(r0=...) is unsupported for PackedMyelinatedCylinders; it "
                    "initialises walkers (and their compartments) from `seed` via init_positions.")
            step_fn = make_packed_myelin_step_fn(geometry, dt, T1=T1, f_band=f_band)
            compartments0 = geometry._init_compartments        # encoded: 0=extra, 1..N=intra, >N=myelin
            _to3 = geometry.pool_of                            # -> 0 extra, 1 intra, 2 myelin
            spin_w = jnp.where(_to3(compartments0) == jnp.int32(2),
                               jnp.float32(geometry._myelin_proton_density), jnp.float32(1.0))

            def _build_packed():
                def simulate_walker(r0_w, key_w, comp0, G_s, chi_s):
                    phi0 = jnp.zeros(n_measurements, dtype=jnp.float32)

                    def emit(carry, inputs):
                        nc, _ = step_fn(carry, inputs)
                        return nc, _to3(nc[4])                 # nc[4] = compartment_id
                    (r_ic_f, _r_uw_f, phi_all, log_w, comp_f, _), comp_seq_w = jax.lax.scan(
                        emit, (r0_w, r0_w, phi0, jnp.float32(0.0), comp0, key_w), (G_s, chi_s))
                    return r_ic_f, phi_all, log_w, comp_f, comp_seq_w
                return jax.jit(jax.vmap(simulate_walker, in_axes=(0, 0, 0, None, None)))

            simulate_batch = cached_batch(
                geometry, ("packed_myelin", n_measurements, n_t, float(dt), T1), _build_packed)
            final_r, all_phi, all_log_w, _comp_final_enc, _comp_seq = simulate_batch(
                r0, walker_keys, compartments0, *scan_inputs)
            signals = _ensemble_signal(spin_w, all_phi, all_log_w)
            if track_comp:
                comp_origin_jax = _to3(compartments0)
                comp_final = _to3(_comp_final_enc)
                comp_seq = _comp_seq
        else:
            # Standard geometry path: one scan body whose carry is (r, phi, log_w, key, comp). The
            # compartment id is advanced inside the step (per-compartment D/T2/T1 read it) and emitted
            # per timestep only when the caller asked for it.
            step_fn, has_weight = make_step_fn(geometry, diffusivity, dt, T2=T2, T1=T1,
                                               sub_steps=sub_steps, track_compartment=track_comp, f_band=f_band)
            spin_w = jnp.ones((n_walkers,), dtype=jnp.float32)
            per_comp = any(a is not None for a in (geometry._D_comp_jax, geometry._inv_T2_comp_jax,
                                                   geometry._inv_T1_comp_jax))
            if track_comp or per_comp:
                # Initial labels are the one place an exact test is affordable and necessary: there is
                # no previous label to carry, so an undecidable point must be resolved, not defaulted.
                comp_origin_jax = jnp.asarray(geometry.classify_positions_exact(r0), jnp.int32)
            else:
                comp_origin_jax = jnp.zeros((n_walkers,), jnp.int32)   # carried untouched, never read
            emit_pos  = want_pos_full
            emit_comp = track_comp and return_compartments == 'full'

            def _build_standard():
                def simulate_walker(r0_w, key_w, comp0_w, G_s, chi_s):
                    phi0 = jnp.zeros(n_measurements, dtype=jnp.float32)

                    def body(carry, inp):
                        new_carry, _ = step_fn(carry, inp)
                        rn, _, _, _, cn = new_carry
                        return new_carry, (rn if emit_pos else None, cn if emit_comp else None)

                    (r_final, phi_all, log_w, _, comp_final), (pos_ys, comp_ys) = jax.lax.scan(
                        body, (r0_w, phi0, jnp.float32(0.0), key_w, comp0_w), (G_s, chi_s))
                    return r_final, phi_all, log_w, comp_final, pos_ys, comp_ys
                return jax.jit(jax.vmap(simulate_walker, in_axes=(0, 0, 0, None, None)))

            simulate_batch = cached_batch(
                geometry, ("standard", n_measurements, n_t, float(dt), diffusivity, T2, T1, sub_steps,
                           track_comp, emit_pos, emit_comp), _build_standard)
            final_r, all_phi, all_log_w, comp_final, pos_seq, comp_seq = simulate_batch(
                r0, walker_keys, comp_origin_jax, *scan_inputs)
            signals = _ensemble_signal(spin_w, all_phi, all_log_w) if has_weight else _ensemble_signal(spin_w, all_phi)

        # T2/T1 are accumulated per-walker inside the scan body (make_step_fn /
        # make_myelin_step_fn); nothing further to apply here.

        # Stimulated-echo readout: the stimulated echo stores half the
        # magnetisation, an idealized 0.5 amplitude factor.
        if getattr(waveform, 'stimulated_echo', False):
            signals = signals * jnp.float32(0.5)

        # Build return tuple
        result = [np.array(signals)]

        if return_positions == 'full':
            result.append(np.array(pos_seq))        # (n_walkers, n_timesteps, 3)
        elif return_positions:
            result.append(np.array(final_r))

        if track_comp:
            result.append(np.array(comp_origin_jax))
            if return_compartments == 'full':
                # comp_seq: (n_walkers, n_t) — transpose from scan output
                result.append(np.array(comp_seq))
            else:  # 'final'
                result.append(np.array(comp_final))

        if return_walker_signals:
            # Per-walker (log_weight, phi) arrays for population-level signal decomposition.
            # log_w: (n_walkers,), phi: (n_walkers, n_measurements).
            # Walker signal contribution: exp(log_w[i]) * cos(phi[i]).
            result.append(np.array(all_log_w))   # (n_walkers,)
            result.append(np.array(all_phi))     # (n_walkers, n_measurements)

        if len(result) == 1:
            return result[0]
        return tuple(result)


def _simulate_in_walker_batches(n_walkers, walker_batch_size, *, seed,
                                diffusivity, waveform, geometry, T2, T1, r0,
                                return_positions, return_compartments,
                                return_walker_signals, sub_steps=None):
    """Run simulate() over walker chunks and recombine (see simulate's
    ``walker_batch_size``).  The signal is a plain walker-mean, so it recombines
    as a size-weighted mean; per-walker outputs (positions, compartments, walker
    signals) are concatenated.  Each chunk's device buffers are released when its
    (host) results are returned, so peak device memory is one chunk."""
    track_comp = return_compartments is not False
    sig_acc = None
    pos_l, origin_l, comp_l, lw_l, phi_l = [], [], [], [], []

    for b, (start, end) in enumerate(current().batches(n_walkers, walker_batch_size, what="simulate")):
        nb = end - start
        out = simulate(
            n_walkers=nb, diffusivity=diffusivity, waveform=waveform,
            geometry=geometry, seed=seed + 1 + b, T2=T2, T1=T1,
            r0=(None if r0 is None else r0[start:end]),
            return_positions=return_positions,
            return_compartments=return_compartments,
            return_walker_signals=return_walker_signals,
            walker_batch_size=None, require_gpu=False,
            engine="fused", sub_steps=sub_steps, _allow_oom_backoff=False)

        items = list(out) if isinstance(out, tuple) else [out]
        sig = np.asarray(items.pop(0))
        sig_acc = sig * nb if sig_acc is None else sig_acc + sig * nb
        if return_positions:
            pos_l.append(np.asarray(items.pop(0)))
        if track_comp:
            origin_l.append(np.asarray(items.pop(0)))
            comp_l.append(np.asarray(items.pop(0)))
        if return_walker_signals:
            lw_l.append(np.asarray(items.pop(0)))
            phi_l.append(np.asarray(items.pop(0)))

    result = [sig_acc / n_walkers]
    if return_positions:
        result.append(np.concatenate(pos_l, axis=0))
    if track_comp:
        result.append(np.concatenate(origin_l, axis=0))
        result.append(np.concatenate(comp_l, axis=0))
    if return_walker_signals:
        result.append(np.concatenate(lw_l, axis=0))
        result.append(np.concatenate(phi_l, axis=0))
    if len(result) == 1:
        return result[0]
    return tuple(result)


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

    Walks the spin ensemble once through the full CPMG train (ideal instantaneous
    180° refocusing is the sign flips of ``waveform.G_eff``) and samples the
    ensemble signal ``Re<exp(iφ)·exp(log_w)>`` at each echo time.  This is the
    ordinary forward model: one pass through the train, nothing cached or reused.
    Build ``waveform`` with :func:`dmipy_sim.cpmg`
    (whose ``readout`` is every echo).

    Parameters
    ----------
    n_walkers : int
    diffusivity : float or None
        Bulk diffusivity (m²/s); omit for MyelinatedCylinder (D in the geometry).
    waveform : ScannerSequence
        A multi-echo sequence, its ``readout`` every echo (e.g. from ``cpmg``).
    geometry : Geometry
    T2 : float, optional
        Transverse relaxation time (s), accumulated per-walker in the walk.
    r0 : array-like of shape (n_walkers, 3), optional
        Explicit start positions in metres.  Default: ``geometry.init_positions(n, key)``,
        which on a mesh means ``intra=True`` -- INSIDE the surface.  Pass this whenever the
        pool you want is not the geometry's inside; a fibre bundle's extra-axonal pool is
        the case that occurs, and getting it wrong silently walks the intra pool instead.
        See :func:`dmipy_sim.geometry.initial_positions`.
    sub_steps : int, optional
        Fine sub-steps per waveform step; overrides the per-geometry auto-tune.
    seed, walker_batch_size, require_gpu : see :func:`simulate`.

    Returns
    -------
    signals : np.ndarray, shape (n_echoes, n_measurements), float32
        Signal at each echo (echo k = k·TE), one column per gradient direction.
    """
    with Run("simulate_cpmg", params=dict(n_walkers=n_walkers, diffusivity=diffusivity, geometry=type(geometry).__name__)) as run:
        from ..spec.build import as_geometry
        geometry = as_geometry(geometry)               # a spec, a spec file or a dict is a substrate too
        from .gpu import check_gpu
        check_gpu(n_walkers, require_gpu, what="simulate_cpmg")

        if hasattr(waveform, 'waveform'):
            waveform = waveform.waveform
        echo_indices = np.asarray(waveform.readout, dtype=int)
        if echo_indices.shape[0] < 2:
            raise ValueError("simulate_cpmg needs a multi-echo readout: build the train with cpmg(...)")

        # Walker batching: one echo-signal accumulator, size-weighted mean over chunks. Every argument
        # that shapes the WALK has to travel with it -- `sub_steps` did not, so a batched call silently
        # ran at the auto-tuned count and an override was measured as if it had been applied. `r0` is
        # per walker and cannot be split by a recursive call, so it is refused rather than ignored.
        if walker_batch_size is not None and walker_batch_size < n_walkers:
            if r0 is not None:
                raise ValueError(
                    "simulate_cpmg cannot batch an explicit r0: the positions would have to be split "
                    "across the batches, and passing them whole would walk the same starts in each. "
                    "Call it once per batch with that batch's r0, or drop walker_batch_size.")
            acc = None
            for b, (start, end) in enumerate(run.batches(n_walkers, walker_batch_size)):
                nb = end - start
                s = simulate_cpmg(nb, diffusivity, waveform, geometry, T2=T2,
                                  seed=seed + 1 + b, sub_steps=sub_steps, walker_batch_size=None,
                                  require_gpu=False)
                acc = s * nb if acc is None else acc + s * nb
            return acc / n_walkers

        G = waveform.G_eff                 # (n_measurements, n_t, 3), the effective gradient
        dt = waveform.dt
        f_band = band_hz(G, dt)
        n_measurements, n_t, _ = G.shape

        G_scan = jnp.transpose(G, (1, 0, 2))   # (n_t, n_measurements, 3)
        # CPMG is a spin-echo train: magnetisation is transverse throughout, so the
        # coherence flag is 1 at every step (step_fn receives inputs = (g_t, chi_t)).
        chi_perp = getattr(waveform, 'chi_perp', None)
        if chi_perp is not None:
            chi_perp_scan = jnp.asarray(chi_perp, dtype=jnp.float32).reshape(n_t)
        else:
            chi_perp_scan = jnp.ones((n_t,), dtype=jnp.float32)
        scan_inputs = (G_scan, chi_perp_scan)

        _, r0, walker_keys = seed_walkers(geometry, n_walkers, seed, r0)

        step_fn, _ = make_step_fn(geometry, diffusivity, dt, T2=T2, sub_steps=sub_steps, f_band=f_band)
        per_comp = any(a is not None for a in (geometry._D_comp_jax, geometry._inv_T2_comp_jax,
                                               geometry._inv_T1_comp_jax))
        comp0 = (jnp.asarray(geometry.classify_positions_exact(r0), jnp.int32) if per_comp
                 else jnp.zeros((n_walkers,), jnp.int32))

        def _build_cpmg():
            def step_emit(carry, inputs):
                new_carry, _ = step_fn(carry, inputs)
                _, phi, log_w, _, _ = new_carry
                return new_carry, jnp.exp(log_w) * jnp.cos(phi)   # (n_measurements,)

            def walk(r0_w, key_w, comp0_w, G_s, chi_s):
                phi0 = jnp.zeros(n_measurements, dtype=jnp.float32)
                _, s_trace = jax.lax.scan(
                    step_emit, (r0_w, phi0, jnp.float32(0.0), key_w, comp0_w), (G_s, chi_s))
                return s_trace                                    # (n_t, n_measurements)
            return jax.jit(jax.vmap(walk, in_axes=(0, 0, 0, None, None)))

        batch = cached_batch(geometry, ("cpmg", n_measurements, n_t, float(dt), diffusivity, T2, sub_steps),
                             _build_cpmg)
        all_traces = batch(r0, walker_keys, comp0, *scan_inputs)              # (n_w, n_t, n_meas)
        signal_trace = jnp.mean(all_traces, axis=0)                    # (n_t, n_meas)
        echo_signals = signal_trace[echo_indices]                     # (n_echoes, n_meas)
        return np.array(echo_signals)





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
    """What a trajectory walk resolves from its request before it walks: the save grid, the sub-step count and the
    rules that gave it, and the :class:`~dmipy_sim.engine.backends.WalkRequest` a backend is handed."""
    record: bool              #: the surface / compartment channels are recorded beside the positions
    n_t: int                  #: block ends over the walk, the start the first
    dt_actual: float          #: the block length ``h`` (s)
    f_band: float             #: the band (Hz) the step law resolves
    rules: dict               #: the sub-step count each applicable rule asks for, by name
    override: object          #: the caller's pinned sub-step count, or None
    sub_steps: int            #: physics steps per block
    dt_sim: float             #: seconds per physics step
    step_rule: dict           #: ``PersistentWalk.work["step_rule"]`` without ``dt_p`` / ``sub_steps``
    binding: str              #: the rule that sized the step (``"override"`` for a pinned count)
    length_scales: object     #: the geometry's :class:`~dmipy_sim.geometry.base.LengthScales`
    request: object           #: the :class:`~dmipy_sim.engine.backends.WalkRequest` of this walk
    jax_only: list            #: what this walk uses that only the JAX kernels implement
    K: int = None             #: the bands the walk accumulates per axis and channel, per window
    block_order: int = 3      #: the order ``m`` of the block table
    block_rule: str = None    #: how the block was set: ``"derived"`` (omega_K h = BLOCK_OMEGA_H) or ``"given"`` (``dt_save``)
    n_windows: int = 1        #: the storage windows the walk accumulates separately


def _resolve_walk(geometry, diffusivity, T_max, dt_save=None, *, tiers="all", sub_steps=None, kappa_MT=0.0, dwell_time=0.0,
                  f_band=None, seed=42, enforce_compartment=False, field_basis=None, K=None, block_order=None, window_T=None):
    """The resolution every walk shares: its storage windows, its band ``K`` per window, its blocks (the band's own,
    :func:`~dmipy_sim.engine.accumulate.block_length`, unless ``dt_save`` gives one), the sub-step count per block
    (``physics.resolve_sub_steps`` over the rules ``physics.sub_step_rules`` at the band ``K / (2 T)``), the geometries
    this walk cannot represent (refused), and the :class:`WalkResolution` a backend is handed.
    :func:`simulate_trajectories` walks what this returns and :func:`walk_cost` reports it, so a cost and a walk state
    one count."""
    from .accumulate import block_length, BLOCK_ORDER, default_K, window_count
    record = _record_channels(tiers)
    n_windows, T_win = window_count(T_max, window_T)
    K = default_K(T_win) if K is None else int(K)
    if K < 1:
        raise ValueError(f"K is the number of bands a walk accumulates per window, at least 1; got {K}")
    if dt_save is None:
        h, block_rule = block_length(T_win, K), "derived"     # omega_K h = BLOCK_OMEGA_H at the top band (#740, cuda#74)
    else:
        h, block_rule = float(T_win) / max(int(round(float(T_win) / float(dt_save))), 1), "given"
    n_bw = int(round(float(T_win) / h))
    n_t = n_windows * n_bw + 1
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
    # what sized the step (PersistentWalk.work["step_rule"]): the count each rule asked for, the tolerance, the
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
    jax_only = [what for what, on in (
        ("the myelinated substrates", geometry._is_packed_myelinated or geometry._is_myelinated),
        ("MT binding", kappa_MT > 0.0), ("in-walk field sampling", field_basis is not None),
        ("the band accumulation (dmipy-sim-cuda#73)", True), ("the compartment guard", bool(enforce_compartment))) if on]
    request = WalkRequest(
        geometry=geometry, n_t=int(n_t), dt_save=float(dt_actual), sub_steps=int(n_sub), dt_sim=float(dt_sim),
        diffusivity=float(diffusivity), record=bool(record),
        kappa_over_D=(float(permeability) / float(diffusivity) if permeability is not None else 0.0),
        count_walls=bool(geometry.count_walls), bounce_budget=geometry.bounce_budget, seed=int(seed))
    return WalkResolution(record, n_t, dt_actual, f_band, dict(rules), override, int(n_sub), dt_sim, step_rule, binding,
                          ls, request, jax_only, K, block_order, block_rule, n_windows)


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
    pilot_blocks: tuple                #: ``(n1, n2)``: the save intervals of the two pilot lengths
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
    accumulator_bytes_per_walker: int     #: the walk's accumulators per walker (level 2, a ring's moments, a block's moments)
    backend: str                      #: the backend that would walk it
    backends_accepting: list          #: the other installed backends that do not refuse this request
    backends_refusing: dict           #: the other installed backends that refuse it, ``{name: reason}``
    measured: object = None           #: :class:`WalkMeasured`, or None when not measured
    profile: object = None            #: :class:`WalkProfile`, or None unless ``measure="profile"``


def _cost_of(res, geometry, n_walkers, walker_batch_size, backend_name, *, batch_rule="given"):
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
    return WalkCost(
        n_walkers=n_walkers, walker_batch_size=int(size), walker_batch_rule=batch_rule, walker_batches=-(-n_walkers // int(size)),
        T_max=float(res.dt_actual) * (int(res.n_t) - 1), n_blocks=int(res.n_t) - 1, block_s=float(res.dt_actual),
        K=int(res.K), block_order=int(res.block_order), n_windows=int(res.n_windows), f_band_hz=float(res.f_band),
        sub_steps=int(res.sub_steps), dt_sim=float(res.dt_sim),
        sub_step_rules={k: int(v) for k, v in res.rules.items()},
        binding_rule=res.binding, binding_length=field, binding_length_m=f(length),
        length_scales=dict(min_feature=f(ls.min_feature), min_gap=f(ls.min_gap), lookup_cell=f(ls.lookup_cell),
                           surface_pore=f(ls.surface_pore)),
        walker_steps=n_walkers * (int(res.n_t) - 1) * int(res.sub_steps),
        bytes_per_walker=_device_bytes_per_walker(res),
        accumulator_bytes_per_walker=_accumulator_bytes(res.K, res.block_order, res.record),
        backend=backend_name, backends_accepting=accepting, backends_refusing=refusing)


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


def _accumulator_bytes(K, m, record):
    from .accumulate import accumulator_bytes
    return accumulator_bytes(K, m, record)


def _device_bytes_per_walker(res):
    """Bytes one walker holds on the device through a batch: its start (3 float32) and key (2 uint32), its
    accumulators (:func:`~dmipy_sim.engine.accumulate.accumulator_bytes`) and its coefficients per window (the bands
    per axis and channel, two exact endpoints each) -- independent of the block count -- and the occupancy track of a
    walk whose walkers can change pool. The output buffers only: the walk's working set is not counted, so this is a
    lower bound."""
    ch = 3 + (1 if res.record else 0)
    per_window = 4 * ch * (int(res.K) + 2) + 4 * 4
    track = 4 * int(res.n_t) if (res.record and res.request.kappa_over_D > 0.0) else 0
    return 20 + _accumulator_bytes(res.K, res.block_order, res.record) + int(res.n_windows) * per_window + track


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


def _profile(per_block, vb, res, max_batch, stopped_by):
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
                       bytes_per_walker=_device_bytes_per_walker(res),
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


def walk_cost(n_walkers, diffusivity, geometry, T_max, dt_save=None, *, walker_batch_size=DEFAULT_WALKER_BATCH,
              backend=None, measure=False, **walk_kwargs):
    """What a trajectory walk costs, stated before it runs: a :class:`WalkCost`.

    It takes the arguments of :func:`simulate_trajectories`, with its names and defaults, and costs that call: the
    request is resolved exactly as the walk resolves it (one code path: ``walk.sub_steps`` equals
    ``walk_cost(...).sub_steps``) and refused where the walk refuses. ``walk_kwargs`` are the walk's other keywords
    (``K``, ``block_order``, ``tiers``, ``sub_steps``, ``f_band``, ``kappa_MT``, ``dwell_time``, ``seed``,
    ``bounce_budget``, ``field_basis``, ``r0``, ...); a keyword the walk does not take is refused.

    Parameters
    ----------
    n_walkers, diffusivity, geometry, T_max, dt_save
        As :func:`simulate_trajectories`. ``T_max`` scales the walker-steps linearly; the bands ``K`` set the
        accumulators' bytes per walker and the memory-limited batch, and, without ``f_band``, the band the step law
        resolves; ``dt_save`` is the block when the band's own is not wanted.
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
        res = _resolve_walk(geometry, diffusivity, T_max, dt_save, tiers=kw.get("tiers", "all"),
                            sub_steps=kw.get("sub_steps"), kappa_MT=kw.get("kappa_MT", 0.0),
                            dwell_time=kw.get("dwell_time", 0.0), f_band=kw.get("f_band"), seed=kw.get("seed", 42),
                            enforce_compartment=kw.get("enforce_compartment", False),
                            field_basis=kw.get("field_basis"), K=kw.get("K"), block_order=kw.get("block_order"),
                            window_T=kw.get("window_T"))
        if be is not None:
            refusal = _backend_refusal(be, name, res, geometry)
            if refusal:
                raise ValueError(refusal)
        free = _free_device_bytes()
        max_batch = None if free is None else max(1, free // _device_bytes_per_walker(res))
        given = walker_batch_size is not None
        batch = int(walker_batch_size) if given else min(int(n_walkers), max_batch or int(n_walkers))
        cost = _cost_of(res, geometry, n_walkers, batch, name,
                        batch_rule="given" if given else "n_walkers capped by memory")
    if measure:
        max_blocks = cost.n_blocks
        r0 = None if kw.get("r0") is None else np.asarray(kw["r0"])
        kw.update(sub_steps=res.sub_steps, f_band=res.f_band, K=res.K, block_order=res.block_order)

        def seconds(b, blocks):
            """Seconds of the second call of this shape (the first holds the compile)."""
            k = dict(kw, **({} if r0 is None else dict(r0=r0[:b])))
            walk = lambda: simulate_trajectories(b, diffusivity, geometry, blocks * res.dt_actual, res.dt_actual,
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
            profile = _profile(per, B, res, max_batch, stopped)
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
                profile = _profile(per, B, res, max_batch, "given")
        ps, fixed, pilot = fit(B)
        measured = WalkMeasured(batch=int(B), pilot_blocks=pilot, seconds_per_block=ps, fixed_seconds_per_call=fixed,
                                seconds_per_walker_step=ps / (B * res.sub_steps),
                                projected_s=cost.walker_batches * (fixed + ps * max_blocks))
        cost = dataclasses.replace(cost, measured=measured, profile=profile)
    return cost


def simulate_trajectories(
    n_walkers: int,
    diffusivity: float,
    geometry,
    T_max: float,
    dt_save: float = None,
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
    field_sample_every: int = 1,
    bounce_budget: int = None,
    f_band: float = None,
    backend="jax",
):
    """Walk the spins ONCE and keep, per walker, the bands a replay pack stores: the producer of the replay path.

    Unlike :func:`simulate`, this applies NO gradient waveform: it keeps every walker's path as its exact start and
    end and the ``K`` sine bands of the bridge between them (and the cumulative boundary local time in the same
    form), accumulated while the walk runs (:mod:`dmipy_sim.engine.accumulate`), so any waveform / relaxation
    hypothesis is applied afterwards from the coefficients (:func:`~dmipy_sim.replay.bank.build_replay_pack`). This
    is the walk-once half of the replay invariant (the bands depend only on ``geometry, diffusivity, seed``).

    The walk runs in blocks of ``h``: the band's own (:func:`~dmipy_sim.engine.accumulate.block_length`: ``omega_K h =
    0.8`` at the top band ``omega_K = pi K / T``, a whole number of blocks of one step count each; 100 us at K = 256
    over 100 ms) unless ``dt_save`` gives one. Inside a block the steps add the moments of their increments about the
    block's centre (order :data:`~dmipy_sim.engine.accumulate.BLOCK_ORDER` = 3 by default) and every ring of blocks is
    projected onto the cosine basis (:func:`~dmipy_sim.replay.compression.block_table`) into a float32 level-2
    accumulator; at the end of a window the projections become the bridge's bands by integration by parts
    (:func:`~dmipy_sim.replay.compression.cosine_band_scale`). Nothing is stored per block but the occupancy track of a
    walk whose walkers can change pool, the bound fraction of a binding walk, the field's block means of a walk that
    samples one, and the ``raw_subset``; a walker's device and host bytes do not grow with the block count otherwise.
    A walk longer than one storage window (``window_T``, 100 ms) is accumulated window by window (RPK.md 4.3).

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
    dt_save : float, optional
        The block length (s) in place of the band's own.
    seed : int
        Master PRNG seed.
    walker_batch_size : int
        Number of walkers per batch.  Reduce if OOM (the batch loop also auto-halves on an OOM exception).
    tiers : {"all", ()}
        Which replay tiers the walk records beyond the positions. ``"all"`` (default) records every
        channel the geometry supports -- the boundary local time (surface relaxivity, rho/D = 1) and
        the compartment occupancy (per-compartment T2/T1); with ``kappa_MT > 0`` also the bound
        fraction. ``()`` records positions only, a cheaper walk (no surface sub-step criterion) for a
        gradient-only replay.
    require_gpu : {None, True, False}
        GPU guard against a silent CPU fallback.  ``True`` raises if no GPU is
        visible; ``False`` opts out; ``None`` (default) warns for large CPU runs.
    r0 : array-like of shape (n_walkers, 3), optional
        Caller-supplied seed positions; ``geometry.init_positions()`` is skipped.
    kappa_MT : float
        Magnetization-transfer surface reactivity (m/s) for the PackedMyelinatedCylinders path; ``> 0`` binds free
        water at the myelin walls and records the bound fraction per block.
    dwell_time : float
        Mean bound-pool residence time (s); must be ``> 0`` when ``kappa_MT > 0``.
    equilibrate_binding : {'auto', 'burnin', 'off'}
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
        Keep the first ``raw_subset`` walkers' positions, contact and pool label at every block end beside the
        coefficients (``CoefficientWalk.raw_positions``; :meth:`~dmipy_sim.coefficient_walk.CoefficientWalk.saves`
        is that subset as a walk of saves): the opt-in raw output, for a measured certificate, a visualisation or a
        dense oracle.
    field_basis : FieldGrid, optional
        A gridded susceptibility field basis (:class:`~dmipy_sim.fields.susceptibility_field.FieldGrid`) the walk
        samples itself: at every step the grid's channels are read at the walker's position (a periodic grid wraps)
        and averaged over the block -- ``CoefficientWalk.field_samples``, sample 0 the start position's, what the
        pack's path channel encodes. A strand basis is sampled by
        :func:`~dmipy_sim.engine.adaptive.simulate_trajectories_adaptive`.
    field_sample_every : int
        The field's own grid: the block means of every that-many-th block only.

    Returns
    -------
    CoefficientWalk
        The coefficients per window, the contact, the occupancy (one label per walker on a substrate no walker leaves
        its pool in, else per block), the quadratic variation per channel, the block record ``(h, m, rule, ...)``, the
        raw subset when asked, ``illegal_crossings`` and the work record.
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
        MT binding, in-walk field sampling, the band accumulation (until dmipy-sim-cuda#73), the compartment guard) are refused for any
        other backend rather than fallen back from. ``work["backend"]`` and the run record say which backend
        walked. See :mod:`dmipy_sim.engine.backends`.
    """
    _backend = resolve_backend(backend)
    _backend_name = JAX if _backend is None else str(_backend.name)
    with Run("simulate_trajectories", params=dict(n_walkers=n_walkers, diffusivity=diffusivity, geometry=type(geometry).__name__, T_max=T_max, dt_save=dt_save, backend=_backend_name, K=K)) as run, _bounce_budget_override(geometry, bounce_budget):
        from ..spec.build import as_geometry
        geometry = as_geometry(geometry)               # a spec, a spec file or a dict is a substrate too
        # GPU guard — never silently fall back to CPU for a heavy walk (CLAUDE rule).
        from .gpu import check_gpu
        check_gpu(n_walkers, require_gpu, what="simulate_trajectories")
        res = _resolve_walk(geometry, diffusivity, T_max, dt_save, tiers=tiers, sub_steps=sub_steps, kappa_MT=kappa_MT,
                            dwell_time=dwell_time, f_band=f_band, seed=seed, enforce_compartment=enforce_compartment,
                            field_basis=field_basis, K=K, block_order=block_order, window_T=window_T)
        record, n_t, dt_actual, sub_steps, dt_sim, step_rule = res.record, res.n_t, res.dt_actual, res.sub_steps, res.dt_sim, res.step_rule
        R_geom = res.length_scales.min_feature
        step_l_sim = jnp.float32(jnp.sqrt(6.0 * diffusivity * dt_sim))

        log.info(f"  sub_steps={sub_steps}, dt_sim={dt_sim*1e6:.3f} µs, "
              f"step_l={float(step_l_sim)*1e6:.4f} µm"
              + (f", step_l/R={float(step_l_sim)/float(R_geom):.4f}" if R_geom else ""))

        # the cost: what this walk costs and which rule sized it, in the run's record before the walk starts
        _announce_cost(run, _cost_of(res, geometry, n_walkers, walker_batch_size, _backend_name))

        permeability = geometry.permeability
        has_permeability = permeability is not None
        has_reflect_with_log_weight = geometry.has_walls

        # ── In-walk field sampling: a gridded basis read at every sub-step, passed to the walk as arguments ──
        _sampling = field_basis is not None
        _f_every = int(field_sample_every)
        if _sampling:
            from ..fields.susceptibility_field import FieldGrid
            if not isinstance(field_basis, FieldGrid):
                raise TypeError(f"field_basis must be a FieldGrid (a gridded field read at the sub-step), got "
                                f"{type(field_basis).__name__}; a strand basis is sampled by simulate_trajectories_adaptive")
            if _f_every < 1:
                raise ValueError(f"field_sample_every must be a positive integer, got {field_sample_every!r}")
            _f_arrays = field_basis.device_arrays()
            _f_periodic = tuple(bool(p) for p in field_basis.periodic)
            _f_key = ("field", tuple(_f_arrays[0].shape), _f_periodic, _f_every)
        else:
            if _f_every != 1:
                raise ValueError("field_sample_every is the save grid of a field sampled in the walk: it goes with field_basis=")
            _f_arrays, _f_periodic, _f_key = (), None, None

        def _reader(arrays):
            return _GridRead(arrays, _f_periodic) if _sampling else None

        # ── The band accumulation: each block's step scan also forms the moments of the increments of the recorded
        # position and of the cumulative local time (engine/accumulate.py), and the ring scan projects them onto the
        # cosine basis
        from .accumulate import moment_scan as _moment_scan
        _m_blk = int(res.block_order)
        _nch = int(_f_arrays[0].shape[0]) if _sampling else 0
        _iscan_gen = lambda inner, init, n, read: _moment_scan(inner, init, n, lambda c: c[0], None, _m_blk, read=read, n_ch=_nch)
        _iscan_relax = lambda inner, init, n, read: _moment_scan(inner, init, n, lambda c: c[0], lambda c: c[2], _m_blk,
                                                                 read=read, n_ch=_nch)
        _iscan_pm = lambda inner, init, n, read: _moment_scan(inner, init, n, lambda c: c[1], lambda c: c[3], _m_blk,
                                                              read=read, n_ch=_nch)

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

        def interval(carry, read):
            carry_final, mean = _iscan_gen(inner_step, carry, sub_steps, read)
            return carry_final, carry_final[0], mean

        # ── Compartment channel (relaxation path only) ────────────────────────────
        # The geometry's own pool id, collapsed to two pools (0 extra, 1 enclosed): relaxation is
        # per pool, and an object id would overflow int8 above 127 objects. The label is CARRIED:
        # seeded once from the exact classifier and updated per sub-step through
        # `classify_position_carry`, so a geometry whose per-step classifier is undecidable away
        # from its walls (a mesh) keeps the label it had. Recorded per sub-step at f32, so it is
        # exact where re-classifying stored positions is not (issue #78).
        def _pool2(comp):
            return jnp.minimum(comp, 1).astype(jnp.float32)

        # ── Relaxation-data path (position + boundary log-weight with rho/D=1) ────
        is_packed_myelin_geom = geometry._is_packed_myelinated
        is_myelin_geom = geometry._is_myelinated
        # Both concentric-cylinder geometries (one axon in an open domain, or a periodic pack)
        # share the one trajectory step below -- make_myelin_traj_step_fn / make_packed_myelin_
        # traj_step_fn, both built on physics.make_myelin_substep -- which carries the compartment
        # id the generic position-only walk cannot express.
        uses_myelin_traj = is_packed_myelin_geom or is_myelin_geom

        # kappa_MT binds free water into a bound pool that only PackedMyelinatedCylinders'
        # trajectory step (make_packed_myelin_traj_step_fn) carries; every other geometry's step
        # -- the isolated MyelinatedCylinder's make_myelin_traj_step_fn included -- has no such
        # state, so a kappa_MT > 0 given to it would silently do nothing (the #536 failure mode).
        # Refused here, where the step function is chosen, before any device work.
        if kappa_MT > 0.0 and not is_packed_myelin_geom:
            raise NotImplementedError(
                f"simulate_trajectories: kappa_MT > 0 (magnetization transfer) is carried only by "
                f"PackedMyelinatedCylinders' trajectory step (make_packed_myelin_traj_step_fn); "
                f"{type(geometry).__name__} has no bound pool to bind into. Pass kappa_MT=0.0 "
                f"(the default), or walk a PackedMyelinatedCylinders.")

        step_fn_traj_pm = None       # set below when the myelin trajectory step runs; read by the work record
        if record and uses_myelin_traj:
            # (Packed or isolated) myelinated cylinder: use the stripped trajectory step fn
            # (geometry + permeability only, rho/D=1 at all walls).  comp_id is the encoded id
            # (0=extra, 1..N_max=intra, >N_max=myelin); compress to 0/1/2 at save.
            # The walk steps the in-cell position and records the continuous (unwrapped) one,
            # as every other periodic substrate's trajectory does (the isolated cylinder has no
            # cell to unwrap: the two are the same position).
            # Magnetization transfer (kappa_MT > 0, PackedMyelinatedCylinders only -- refused above
            # for every other geometry): the step fn binds free water at the myelin walls and
            # records the per-save bound occupancy.  kappa_MT == 0 keeps the pre-MT walk
            # bit-for-bit (RNG stream + positions unchanged).
            _mt_on_pm = kappa_MT > 0.0
            step_fn_traj_pm = (make_packed_myelin_traj_step_fn(
                                  geometry, dt_sim, kappa_MT=kappa_MT, dwell_time=dwell_time)
                              if is_packed_myelin_geom else make_myelin_traj_step_fn(geometry, dt_sim))

            def _compress_comp_pm(comp_id):
                return geometry.pool_of(comp_id).astype(jnp.int8)

            def _inner_pm(carry, _):
                return step_fn_traj_pm(carry, None)

            if not _mt_on_pm:
                # ── without MT (5-element carry) ──
                def interval_pm(carry, read):
                    r, r_uw, key, comp_id, work = carry
                    # dlog_accum resets each save so the emitted value is the per-save delta.
                    inner_init = (r, r_uw, key, jnp.float32(0.0), comp_id, work)
                    (r_final, r_uw_final, key_final, dlog_accum, comp_final, work_f), mean = _iscan_pm(
                        _inner_pm, inner_init, sub_steps, read)
                    return (r_final, r_uw_final, key_final, comp_final, work_f), \
                           (r_uw_final, dlog_accum, _compress_comp_pm(comp_final)), mean

            else:
                # ── MT path: bound_rem persists across saves ──
                def interval_pm(carry, read):
                    r, r_uw, key, comp_id, bound_rem, work = carry
                    inner_init = (r, r_uw, key, jnp.float32(0.0), comp_id, bound_rem, jnp.float32(0.0), work)
                    (r_final, r_uw_final, key_final, dlog_accum, comp_final, bound_rem_f, bound_acc, work_f), mean = \
                        _iscan_pm(_inner_pm, inner_init, sub_steps, read)
                    bound_frac = bound_acc / jnp.float32(sub_steps)
                    return (r_final, r_uw_final, key_final, comp_final, bound_rem_f, work_f), \
                           (r_uw_final, dlog_accum, _compress_comp_pm(comp_final), bound_frac), mean

        if record and not uses_myelin_traj:
            if has_permeability:
                kappa_over_D_relax = jnp.float32(float(permeability) / diffusivity)
                permeate_relax = geometry.permeate
                interact_relax = geometry.interact

                def inner_step_relax(carry, _):
                    r, key, dlog_accum, comp_sum, side, bad, comp, work = carry
                    key, step_key, perm_key = jax.random.split(key, 3)
                    unit_noise = pair_step(step_key)
                    step = unit_noise * step_l_sim
                    if _carries_side:
                        hit = permeate_relax(r, step, kappa_over_D_relax, jnp.float32(1.0), perm_key, side)
                        r_new, dlog_w_unit = hit.r, hit.dlog_w
                        side = jnp.where(hit.crossed, -side, side)
                        bad = bad + hit.illegal.astype(jnp.int32)
                        # Label from the CARRIED side, not from the position. This is the
                        # channel `comp_traj` is built from, so re-deriving it here would put
                        # the relabelling straight back in even with the sentinel correcting
                        # the coordinate. 0 = extra, 1 = intra, as `_get_comp_id`.
                        comp_id = jnp.where(side < 0, jnp.float32(1.0), jnp.float32(0.0))
                    else:
                        hit = interact_relax(r, step, kappa_over_D=kappa_over_D_relax,
                                             rho_over_D=jnp.float32(1.0), key=perm_key)
                        r_new, dlog_w_unit = hit.r, hit.dlog_w
                        bad = bad + hit.illegal.astype(jnp.int32)
                        comp = geometry.classify_position_carry(r_new, comp)
                        comp_id = _pool2(comp)
                    # Per-sub-step compartment id -> fractional occupancy (resolves
                    # intra-save crossings without a finer dt_save).
                    comp_sum = comp_sum + comp_id
                    return (r_new, key, dlog_accum + dlog_w_unit, comp_sum, side, bad, comp, _work_add(work, hit)), None

            elif has_reflect_with_log_weight:
                # `interact`, not `reflect_with_log_weight`: for an impermeable wall the two are the
                # same function at the same arguments (`Geometry.interact`), but the WallHit also
                # carries `illegal`, so a geometry that REFUSES a step -- a voxel wall's reject-escape,
                # where the walker is held still -- is counted here instead of being invisible.
                interact = geometry.interact

                def inner_step_relax(carry, _):
                    r, key, dlog_accum, comp_sum, side, bad, comp, work = carry
                    key, subkey = jax.random.split(key)
                    unit_noise = pair_step(subkey)
                    step = unit_noise * step_l_sim
                    hit = interact(r, step, rho_over_D=jnp.float32(1.0))
                    r_new, dlog_w_unit = hit.r, hit.dlog_w
                    bad = bad + hit.illegal.astype(jnp.int32)
                    comp = geometry.classify_position_carry(r_new, comp)
                    comp_sum = comp_sum + _pool2(comp)
                    return (r_new, key, dlog_accum + dlog_w_unit, comp_sum, side, bad, comp, _work_add(work, hit)), None

            else:
                # FreeDiffusion: no boundaries → dlog_boundary_unit is always 0.
                reflect_free = geometry.reflect

                def inner_step_relax(carry, _):
                    r, key, dlog_accum, comp_sum, side, bad, comp, work = carry
                    key, subkey = jax.random.split(key)
                    unit_noise = pair_step(subkey)
                    step = unit_noise * step_l_sim
                    r_new = reflect_free(r, step)
                    comp = geometry.classify_position_carry(r_new, comp)
                    comp_sum = comp_sum + _pool2(comp)
                    return (r_new, key, dlog_accum, comp_sum, side, bad, comp, work), None

            def interval_relax(carry, read):
                r, key, side, bad, comp, work = carry
                inner_init = (r, key, jnp.float32(0.0), jnp.float32(0.0), side, bad, comp, work)
                (r_final, key_final, dlog_accum, comp_sum, side_f, bad_f, comp_f, work_f), mean = _iscan_relax(
                    inner_step_relax, inner_init, sub_steps, read)
                # Fractional occupancy of pool 1 (the enclosed pool) over the saved interval.
                comp_occ = comp_sum / jnp.float32(sub_steps)
                return (r_final, key_final, side_f, bad_f, comp_f, work_f), (r_final, dlog_accum, comp_occ), mean

        # Seed each walker's carried compartment ONCE, from its t=0 position, and let only a
        # granted crossing change it thereafter (MC/DC's `initial_location`). The wrapper keeps
        # the (r0, keys) -> positions signature every call site below already uses.
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

        # ── MT bound-pool equilibration (packed myelin, kappa_MT > 0) ──
        # An all-free start under-fills the macromolecular pool and biases the transfer;
        # equilibrate the bound occupancy (and spatial state) to f_b BEFORE t=0 and discard
        # the preamble.  kappa_MT == 0 leaves brem0_all at zero and skips this entirely.
        _mt_on = kappa_MT > 0.0 and record and is_packed_myelin_geom
        brem0_all = jnp.zeros((n_walkers,), dtype=jnp.float32)
        if _mt_on:
            from . import mt as _mt
            if dwell_time <= 0.0:
                raise ValueError("dwell_time must be > 0 when kappa_MT > 0.")
            _mode = _mt.resolve_equilibrate_mode(equilibrate_binding, geometry)
            if _mode == 'burnin':
                _n_chunk = max(4, int(round(float(dwell_time) / float(dt_sim))))

                def _burn_walker(r_w, key_w, comp_w, brem_w):
                    (r_f, _r_uw, key_f, _da, comp_f, brem_f, bacc, _w), _ = jax.lax.scan(
                        _inner_pm, (r_w, r_w, key_w, jnp.float32(0.0), comp_w, brem_w,
                                    jnp.float32(0.0), _work0()), None, length=_n_chunk)
                    return r_f, key_f, comp_f, brem_f, bacc / jnp.float32(_n_chunk)
                _burn = jax.jit(jax.vmap(_burn_walker, in_axes=(0, 0, 0, 0)))

                _comp = [comp0_all]                                   # the compartment state rides along the chunks

                def _chunk(r, k, brem):
                    r, k, _comp[0], brem, bf = _burn(r, k, _comp[0], brem)
                    return r, k, brem, jnp.mean(bf)
                r0_all, walker_keys_all, brem0_all, _occ_prev, _converged = _mt.equilibrate_burnin_plateau(
                    _chunk, r0_all, walker_keys_all, brem0_all)
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
                    f"position for that step. See PersistentWalk.illegal_crossings.",
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
            work = dict(walker_steps=int(n_walkers) * (int(n_t) - 1) * int(sub_steps),
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
                    f"untested; the budget is too small for this substrate at this step. See PersistentWalk.work.",
                    RuntimeWarning, stacklevel=3)

            return illegal, work

        if _backend is not None:
            # another execution of the same physics (#635): the paths only the JAX kernels implement are refused,
            # never fallen back from -- the band accumulation is one of them until dmipy-sim-cuda#73
            raise ValueError(_backend_refusal(_backend, _backend_name, res, geometry))

        from .accumulate import ring_scan, coefficient_form, PRECISION, ACCUMULATION, RING
        from ..replay.compression import cosine_band_scale
        from ..coefficient_walk import CoefficientWalk
        K, m = int(res.K), int(res.block_order)
        n_win = int(res.n_windows)
        n_blocks = int(n_t) - 1
        n_bw = n_blocks // n_win                                  # blocks per window
        T_win = n_bw * float(dt_actual)
        scale = jnp.asarray(cosine_band_scale(K, T_win), jnp.float32)
        n_raw = int(min(max(int(raw_subset), 0), n_walkers))
        # which occupancy a block emits: the track where a walker can change pool (a permeable wall, a binding walk),
        # else nothing -- the walker's start label, the blocks that disagree with it counted
        if record and uses_myelin_traj:
            kind = "pm"
            crossing = bool(_mt_on) or bool(np.any(np.asarray(geometry._kappa_inner_jax) > 0)) \
                or bool(np.any(np.asarray(geometry._kappa_outer_jax) > 0))
            pos_of = lambda c: c[1]
        elif record:
            kind, crossing, pos_of = "relax", bool(has_permeability), (lambda c: c[0])
        else:
            kind, crossing, pos_of = "gen", False, (lambda c: c[0])
        _f_nch = int(_f_arrays[0].shape[0]) if _sampling else 0

        def _walker(raw):
            """One walker's walk, window by window: per window its coefficients, contact, quadratic variation and the
            float32 chain's residual; per block what the walk keeps there (the occupancy of a walker that can change
            pool, the bound fraction, the field's block means, and for the raw subset its positions, contact and
            label at every block end)."""
            def one(r0_w, key_w, side_w, comp0_w, brem0_w, f_arrays):
                read = _reader(f_arrays)
                if kind == "gen":
                    carry0, interval_fn, has_blt = (r0_w, key_w, side_w, jnp.int32(0), _work0()), interval, False
                    lab0 = jnp.int8(0)
                    emit = (lambda ys: (ys, None, None, None, None)) if raw else None
                    count = None
                elif kind == "relax":
                    carry0, interval_fn, has_blt = (r0_w, key_w, side_w, jnp.int32(0), comp0_w, _work0()), interval_relax, True
                    lab0 = _pool2(comp0_w)
                    emit = ((lambda ys: (ys[0] if raw else None, ys[1] if raw else None, ys[2] if (raw or crossing) else None,
                                         None, None)) if (raw or crossing) else None)
                    count = None if crossing else (lambda ys: jnp.rint(ys[2]) != lab0)
                else:
                    carry0 = ((r0_w, r0_w, key_w, comp0_w, brem0_w, _work0()) if _mt_on
                              else (r0_w, r0_w, key_w, comp0_w, _work0()))
                    interval_fn, has_blt = interval_pm, True
                    lab0 = _compress_comp_pm(comp0_w)
                    emit = ((lambda ys: (ys[0] if raw else None, ys[1] if raw else None, ys[2] if (raw or crossing) else None,
                                         ys[3] if _mt_on else None, None)) if (raw or crossing or _mt_on) else None)
                    count = None if crossing else (lambda ys: ys[2] != lab0)

                def window(c, w):
                    r_start = pos_of(c)
                    c, ap, ab, e, n, (qp, qb), f = ring_scan(interval_fn, c, n_bw, K, m, has_blt=has_blt, emit=emit,
                                                             count=count, read=read, every=_f_every, offset=w * n_bw)
                    C, resid = coefficient_form(r_start, pos_of(c), ap, scale)
                    blt = (jnp.zeros((0,), jnp.float32), jnp.float32(0.0)) if not has_blt else (scale * ab[1:], ab[0])
                    return c, (C, resid, blt, qp, (jnp.float32(0.0) if qb is None else qb),
                               (jnp.int32(0) if n is None else n), e, f)

                c, (C, resid, blt, qp, qb, n, e, f) = jax.lax.scan(window, carry0, jnp.arange(n_win, dtype=jnp.int32))
                flat = lambda a: None if a is None else a.reshape((n_win * n_bw,) + a.shape[2:])
                e = (None,) * 5 if e is None else tuple(flat(x) for x in e)
                z = jnp.zeros((0,), jnp.float32)
                pos = jnp.concatenate([r0_w[None], e[0]], axis=0).astype(jnp.float32) if raw else z     # stored float32
                dlog = jnp.concatenate([jnp.zeros((1,), jnp.float32), e[1]]).astype(jnp.float32) if (raw and has_blt) else z
                occ = jnp.concatenate([jnp.asarray(lab0, e[2].dtype)[None], e[2]]) if e[2] is not None else z
                bf = jnp.concatenate([(brem0_w > 0).astype(jnp.float32)[None], e[3]]) if e[3] is not None else z
                fs = jnp.concatenate([read(r0_w)[None], flat(f)[_f_every - 1::_f_every]], axis=0) if read is not None else z
                work_f = c[-1]
                bad = c[3] if kind in ("gen", "relax") else jnp.int32(0)
                return (C, resid, blt[1], blt[0], jnp.asarray(lab0).astype(jnp.int8), occ, bf, fs, pos, dlog, qp, qb,
                        n.sum(), bad, work_f)
            return one

        def _batch_fn(raw):
            return cached_batch(geometry, ("coefficient walk", kind, raw, crossing, n_t, n_win, sub_steps, float(dt_sim),
                                           diffusivity, K, m, kappa_MT, dwell_time, _counting, _f_key),
                                lambda: jax.jit(jax.vmap(_walker(raw), in_axes=(0, 0, 0, 0, 0, None))))

        names = ("C", "resid", "blt_end", "blt_bands", "lab", "occ", "bfrac", "fs", "pos", "dlog", "qv_pos", "qv_blt")
        rows = {k: _Rows(n_walkers) for k in names}
        raw_rows = {k: _Rows(n_raw) for k in ("pos", "dlog", "occ")} if n_raw else None
        dev = [0]

        def _call(a, b, raw):
            r0_b = r0_all[a:b]
            comp0_b = (comp0_all[a:b] if kind == "pm" else
                       jnp.asarray(geometry.classify_positions_exact(r0_b), jnp.int32) if kind == "relax"
                       else jnp.zeros((b - a,), jnp.int32))
            out = _batch_fn(raw)(r0_b, walker_keys_all[a:b], _side0(r0_b), comp0_b, brem0_all[a:b], _f_arrays)
            *arrs, n_dev, bad_f, work_f = out
            _illegal_crossings[0] += int(jnp.sum(bad_f))
            _work_acc(work_f)
            dev[0] += int(jnp.sum(n_dev))
            return {k: np.asarray(v) for k, v in zip(names, arrs)}

        def _run(a, b, raw):
            try:
                return _call(a, b, raw)
            except Exception as e:
                msg = str(e)
                if not ("OOM" in msg or "out of memory" in msg.lower() or "RESOURCE_EXHAUSTED" in msg) or b - a < 2000:
                    raise
                mid = (a + b) // 2
                log.info(f"  OOM: halving the batch to {mid - a}")
                lo, hi = _run(a, mid, raw), _run(mid, b, raw)
                return {k: np.concatenate([lo[k], hi[k]], axis=0) for k in names}

        for start, end in run.batches(n_walkers, walker_batch_size):
            pieces = ([(start, min(end, n_raw), True)] if start < n_raw else []) + \
                     ([(max(start, n_raw), end, False)] if end > n_raw else [])
            for a, b, raw in pieces:
                got = _run(a, b, raw)
                for k in names:
                    if k in ("pos", "dlog") or (k == "occ" and not crossing) or (k == "bfrac" and not _mt_on) \
                            or (k == "fs" and not _sampling):
                        continue
                    rows[k].append(got[k])
                if raw:
                    raw_rows["pos"].append(np.asarray(got["pos"], np.float32))
                    raw_rows["dlog"].append(np.asarray(got["dlog"], np.float32))
                    o = np.asarray(got["occ"])                       # converted per batch: the walk held once
                    raw_rows["occ"].append(o.astype(np.float32) if (crossing and kind == "relax") else np.rint(o).astype(np.int8))
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
        block = dict(h=float(dt_actual), m=m, precision=PRECISION, n_blocks=n_blocks, rule=res.block_rule, ring=int(RING),
                     accumulation=ACCUMULATION,
                     k0_residual_m=float(np.abs(resid).max()) if resid.size else 0.0,
                     k0_residual_relative=(float(np.abs(resid).max() / max(np.sqrt(np.mean(disp ** 2)), 1e-300))
                                           if resid.size else 0.0),
                     quadratic_variation=dict(
                         positions_mean_m2=[float(x) for x in qv_p.sum(axis=1).mean(axis=0)],
                         positions_over_2DT=([float(x) / (2.0 * float(diffusivity) * float(T_max)) for x in qv_p.sum(axis=1).mean(axis=0)]
                                             if diffusivity else None),
                         boundary_mean_m2=(float(rows["qv_blt"].array().sum(axis=1).mean()) if records_surface else None)))
        work = dict(work, block=block, label_deviations=n_dev)
        comp = bfrac = None
        raw_comp = None
        if record:
            comp = rows["occ"].array() if crossing else rows["lab"].array().astype(np.int8)
            if crossing and kind == "pm":
                comp = comp.astype(np.int8)
            if _mt_on:
                bfrac = rows["bfrac"].array().astype(np.float32)
            if n_raw:
                raw_comp = raw_rows["occ"].array()
        cw = CoefficientWalk(
            position_coeffs=np.asarray(rows["C"].array(), np.float32), T=float(T_max), K=K, m=m, block=block,
            n_windows=n_win, dt_sim=float(dt_sim), sub_steps=int(sub_steps),
            boundary_endpoint=(rows["blt_end"].array().astype(np.float32) if records_surface else None),
            boundary_bands=(rows["blt_bands"].array().astype(np.float32) if records_surface else None),
            compartment=comp, bound_frac=bfrac,
            field_basis=(field_basis if _sampling else None),
            field_samples=(rows["fs"].array().astype(np.float32) if _sampling else None),
            field_sample_every=_f_every,
            raw_positions=(None if not n_raw else np.asarray(raw_rows["pos"].array(), np.float32)),
            raw_boundary=(None if not (n_raw and records_surface) else raw_rows["dlog"].array()),
            raw_compartment=raw_comp,
            qv_positions=qv_p.astype(np.float32),
            qv_boundary=(rows["qv_blt"].array().astype(np.float32) if records_surface else None),
            illegal_crossings=illegal, seed=int(seed), diffusivity=(None if diffusivity is None else float(diffusivity)),
            geometry=geometry, work=work)
        object.__setattr__(cw, "run", run)
        return cw


def certify_bounce_budget(geometry, n_walkers, diffusivity, T_max, dt_save, *, margin=1, seed=0, **walk_kwargs):
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
        w = simulate_trajectories(n_walkers, diffusivity, geometry, T_max=T_max, dt_save=dt_save, seed=seed, tiers=(),
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
