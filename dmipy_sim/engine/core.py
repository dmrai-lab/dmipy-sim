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
import numpy as np
import logging

log = logging.getLogger(__name__)

from .physics import (make_step_fn, make_myelin_step_fn, make_packed_myelin_step_fn,
                      make_myelin_traj_step_fn, make_packed_myelin_traj_step_fn,
                      seed_walkers, isotropic_unit_step)
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
    walk = simulate_trajectories(n_walkers, diffusivity, geometry, T_max, dt, **st_kwargs)
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
            step_fn = make_myelin_step_fn(geometry, dt, T1=T1, sub_steps=sub_steps)
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
            step_fn = make_packed_myelin_step_fn(geometry, dt, T1=T1)
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
                                               sub_steps=sub_steps, track_compartment=track_comp)
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

        step_fn, _ = make_step_fn(geometry, diffusivity, dt, T2=T2, sub_steps=sub_steps)
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


def _interval_scan(inner, init, length, read):
    """``inner`` stepped ``length`` times from ``init``: ``(carry, mean)``, where ``mean`` is the mean of ``read``
    at the position (``carry[0]``) after every step of the interval, or None without a ``read``."""
    if read is None:
        out, _ = jax.lax.scan(inner, init, None, length=length)
        return out, None

    def body(c, _):
        c_in, acc = c
        c_out, _ = inner(c_in, None)
        return (c_out, acc + read(c_out[0])), None
    (out, acc), _ = jax.lax.scan(body, (init, jnp.zeros(read.n_ch, jnp.float32)), None, length=length)
    return out, acc / jnp.float32(length)


def _save_scan(interval, carry, n_saves, read, every):
    """The ``n_saves`` save intervals of one walker: ``interval(carry, read) -> (carry, ys, mean)`` walks one
    (``mean`` the interval mean of ``read``, :func:`_interval_scan`). Returns ``(carry, ys, field)``: ``ys`` stacked
    per save, and ``field`` the interval means of the saves ``every``, ``2 every``, ... (the field's own save grid;
    only those intervals read the field, chosen per save by a condition no walker differs in, so a vmapped walk
    runs the one branch and the walk itself is the same program whatever ``every`` is), None without a ``read``."""
    def plain(c, _):
        c, ys, _m = interval(c, None)
        return c, ys
    if read is None:
        carry, ys = jax.lax.scan(plain, carry, None, length=n_saves)
        return carry, ys, None

    def save(c, t):
        c, buf = c

        def reading(cb):
            c, buf = cb
            c, ys, m = interval(c, read)
            return (c, buf.at[t // every - 1].set(m)), ys

        def not_reading(cb):
            c, buf = cb
            c, ys, _m = interval(c, None)
            return (c, buf), ys
        return jax.lax.cond(t % every == 0, reading, not_reading, (c, buf))
    buf = jnp.zeros((n_saves // every, read.n_ch), jnp.float32)
    (carry, field), ys = jax.lax.scan(save, (carry, buf), jnp.arange(1, n_saves + 1, dtype=jnp.int32))
    return carry, ys, field


def _field_series(read, r0, field):
    """A walker's field samples: the channels at its start (sample 0), then the interval means; an empty row
    without a ``read``, so the batch functions keep one output signature."""
    if read is None:
        return jnp.zeros((0,), jnp.float32)
    return jnp.concatenate([read(r0)[None], field])


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


def simulate_trajectories(
    n_walkers: int,
    diffusivity: float,
    geometry,
    T_max: float,
    dt_save: float,
    seed: int = 42,
    walker_batch_size: int = 50_000,
    tiers="all",
    sub_steps: int = None,
    require_gpu=None,
    storage_dtype=np.float32,
    r0=None,
    kappa_MT: float = 0.0,
    dwell_time: float = 0.0,
    equilibrate_binding="auto",
    compress: int = None,
    enforce_compartment: bool = False,
    field_basis=None,
    field_sample_every: int = 1,
) -> PersistentWalk:
    """Walk the spins ONCE and save positions at every saved time step — the
    producer for the replay path (:mod:`dmipy_sim.replay.trajectories`).

    Unlike :func:`simulate`, this applies NO gradient waveform: it stores
    ``r(t)`` for all walkers so any waveform / relaxation hypothesis can be
    applied post-hoc via
    :func:`~dmipy_sim.replay.trajectories.replay`.  This is the
    walk-once half of the replay invariant (positions depend only on
    ``geometry, diffusivity, seed`` — see the module CLAUDE guide).

    Sub-stepping: :func:`dmipy_sim.engine.physics.resolve_sub_steps` chooses the fine
    sub-steps per saved step from the geometry's length scales and the channels
    recorded (the surface criterion when the surface channel is recorded, the binding
    criterion when ``kappa_MT > 0``), the same dispatch the fused engine uses.  Only
    the position after each group of ``sub_steps`` inner steps is saved, so storage
    is always ``(n_walkers, n_t, 3)`` at ``dt_save`` granularity.

    Parameters
    ----------
    n_walkers : int
        Total number of walkers.
    diffusivity : float
        Diffusion coefficient in m²/s.
    geometry : Geometry
        Boundary geometry.  Provides ``init_positions(n, key)`` and
        ``reflect(r, step)``; for the recorded channels also
        ``reflect_with_log_weight`` (surface local time) and/or ``permeate``.
    T_max : float
        Total simulation duration in seconds.
    dt_save : float
        Time step between saved positions in seconds.
    seed : int
        Master PRNG seed.
    walker_batch_size : int
        Number of walkers per GPU batch.  Reduce if OOM (the batch loop also
        auto-halves on an OOM exception).
    tiers : {"all", ()}
        Which replay tiers the walk records beyond the positions. ``"all"`` (default) records every
        channel the geometry supports -- the boundary local time (surface relaxivity, rho/D = 1) and
        the compartment occupancy (per-compartment T2/T1); with ``kappa_MT > 0`` also the bound
        fraction. ``()`` records positions only, a cheaper walk (no surface sub-step criterion) for a
        gradient-only replay. FreeDiffusion records an all-zero boundary channel.
    require_gpu : {None, True, False}
        GPU guard against a silent CPU fallback.  ``True`` raises if no GPU is
        visible; ``False`` opts out; ``None`` (default) warns for large CPU runs.
    r0 : array-like of shape (n_walkers, 3), optional
        Caller-supplied seed positions (e.g. extra-axonal water outside a mesh).
        When provided, ``geometry.init_positions()`` is skipped.
    kappa_MT : float
        Magnetization-transfer surface reactivity (m/s) for the PackedMyelinated-
        Cylinders path.  ``0`` (default) leaves the walk
        byte-for-byte identical to the pre-MT path (RNG stream + positions unchanged);
        ``> 0`` binds free water at the myelin walls and returns a 7th ``bound_frac``
        channel.  (For analytic geometries use :func:`dmipy_sim.simulate_mt_trajectories`.)
    dwell_time : float
        Mean bound-pool residence time (s); must be ``> 0`` when ``kappa_MT > 0``.
    equilibrate_binding : {'auto', 'burnin', 'off'}
        How the bound pool reaches thermal-equilibrium occupancy before t=0 (MT only);
        see :func:`dmipy_sim.engine.mt.resolve_equilibrate_mode`.
    field_basis : FieldGrid, optional
        A gridded susceptibility field basis (:class:`~dmipy_sim.fields.susceptibility_field.FieldGrid`: a mesh,
        a myelinated cylinder, a packed cell) the walk samples itself: at every sub-step the grid's channels are
        read at the walker's position (the one the step kernel holds; a periodic grid wraps, :attr:`FieldGrid.periodic`)
        and averaged over the save interval -- ``PersistentWalk.field_samples``, sample 0 the start position's,
        what the pack's path channel encodes. A strand basis is sampled by
        :func:`~dmipy_sim.engine.adaptive.simulate_trajectories_adaptive`.
    field_sample_every : int
        The field's own save grid: the samples are the interval means of every that-many-th save interval only
        (the saves ``every``, ``2 every``, ...), ``ceil(n_t / every)`` of them at ``every * dt``; the positions
        keep every save.

    Returns
    -------
    PersistentWalk
        ``positions`` (n_walkers, n_t, 3) in ``storage_dtype`` -- save ``k`` at ``t = k dt``, save 0 the start
        (the initial position; its contact / occupancy entries are the initial state) -- ``dt`` (= T_max / (n_t - 1)),
        ``sub_steps``, ``dt_sim``; with the default ``tiers="all"`` also ``boundary_local_time``
        (n_walkers, n_t), the per-step boundary log-weight at rho/D = 1 (``-2 * sum d_perp`` over
        the step's wall hits, non-positive), and ``compartment`` (n_walkers, n_t); with
        ``kappa_MT > 0`` also ``bound_frac``; with ``field_basis`` also ``field_basis``, ``field_samples``
        ``(n_walkers, ceil(n_t / field_sample_every), 7 | 13)`` and ``field_sample_every``. ``illegal_crossings``
        counts the rejected wrong-side steps. With ``compress=K`` a compressed master dict is returned instead (see
        :func:`trajectories.replay`).
    """
    with Run("simulate_trajectories", params=dict(n_walkers=n_walkers, diffusivity=diffusivity, geometry=type(geometry).__name__, T_max=T_max, dt_save=dt_save)) as run:
        from ..spec.build import as_geometry
        geometry = as_geometry(geometry)               # a spec, a spec file or a dict is a substrate too
        # GPU guard — never silently fall back to CPU for a heavy walk (CLAUDE rule).
        from .gpu import check_gpu
        check_gpu(n_walkers, require_gpu, what="simulate_trajectories")
        record = _record_channels(tiers)

        n_t = int(round(T_max / dt_save)) + 1
        dt_actual = T_max / (n_t - 1)

        # --- Sub-stepping: one dispatch for every driver (physics.resolve_sub_steps) ---
        # A walk that records the boundary local time is resolved at the surface criterion too, so a
        # replayed rho reproduces the fused engine's; MT binding adds its own.
        from .physics import resolve_sub_steps as _resolve_sub_steps, length_scales_of as _length_scales_of
        R_geom = _length_scales_of(geometry).min_feature
        sub_steps = _resolve_sub_steps(
            geometry, diffusivity, dt_actual, surface=bool(record),
            mt_dwell_time=(dwell_time if kappa_MT > 0.0 else None), override=sub_steps)
        dt_sim = dt_actual / sub_steps
        step_l_sim = jnp.float32(jnp.sqrt(6.0 * diffusivity * dt_sim))
        # Same soundness bound as in physics.make_step_fn -- the guard has to live here too, because a permeable
        # mesh now routes through the REPLAY backend (see _replay_gap) and so never reaches make_step_fn.
        from .physics import _warn_if_step_outruns_the_lookup as _warn_step
        _warn_step(geometry, diffusivity, dt_actual, sub_steps, "trajectory walk")

        log.info(f"  sub_steps={sub_steps}, dt_sim={dt_sim*1e6:.3f} µs, "
              f"step_l={float(step_l_sim)*1e6:.4f} µm"
              + (f", step_l/R={float(step_l_sim)/float(R_geom):.4f}" if R_geom else ""))

        # ── Reject geometries whose boundaries this path cannot represent ────────
        # A multi-compartment geometry may be stepped by a fused kernel that CARRIES the
        # compartment id; a geometry with no such state and no usable `reflect` would fall
        # through to one that cannot express its boundaries -- both used to do so silently, e.g.
        # a MyelinatedCylinder came back confined to R_inner (myelin and extra water absent) and a
        # PackedMyelinatedCylinders came back at 1.02x free diffusion through 1 um axons. The one
        # source of truth is the geometry's own `trajectory_walk_gap` (never a class-name list),
        # which `spec.walk.walk_spec` also checks before this function is even called.
        _traj_gap = geometry.trajectory_walk_gap()
        if _traj_gap is not None:
            raise NotImplementedError(
                f"simulate_trajectories cannot walk a {type(geometry).__name__}: {_traj_gap}")
        if geometry._is_packed_myelinated and not record:
            raise NotImplementedError(
                "simulate_trajectories on PackedMyelinatedCylinders requires the recorded channels "
                "(tiers='all'), which route to the fused kernel "
                "physics.make_packed_myelin_traj_step_fn. The position-only path has no "
                "compartment state and would return an unrestricted walk.")

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
            if compress is not None:
                raise NotImplementedError("compress= returns the walk as position coefficients; the in-walk field samples "
                                          "have no coefficient form there. Walk uncompressed and pack with build_replay_pack.")
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

        # ── The walk's work: wall encounters, crossings and budget exhaustions, summed per walker in the
        # carry and over the batches on the host, only when the geometry counts (`Geometry.count_walls`);
        # a walk that does not count carries nothing and is the same program as before.
        _counting = bool(geometry.count_walls)
        _work = [0, 0, 0, False]                    # hits, crossings, exhausted steps; whether a kernel reported them

        def _work0():
            return (jnp.int32(0), jnp.int32(0), jnp.int32(0)) if _counting else None

        def _work_add(w, hit):
            if not _counting:
                return None
            n = hit.n_hits if hit.n_hits is not None else jnp.int32(0)
            return (w[0] + n, w[1] + hit.crossed.astype(jnp.int32), w[2] + hit.exhausted.astype(jnp.int32))

        def _work_acc(w):
            if w is not None:
                for i in range(3):
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
                unit_noise = isotropic_unit_step(step_key)
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
                unit_noise = isotropic_unit_step(subkey)
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
            carry_final, mean = _interval_scan(inner_step, carry, sub_steps, read)
            return carry_final, carry_final[0], mean

        def simulate_one_walker(r0_w, key_w, side_w, f_arrays):
            # save 0 is the start (t = 0); saves 1..n_t-1 follow n_t-1 blocks of sub-steps
            read = _reader(f_arrays)
            (_, _, side_f, bad_f, work_f), positions, field = _save_scan(
                interval, (r0_w, key_w, side_w, jnp.int32(0), _work0()), n_t - 1, read, _f_every)
            positions = jnp.concatenate([r0_w[None, :], positions], axis=0)
            # (n_t, 3), carried side, illegal-crossing count, the work, field samples
            return positions, side_f, bad_f, work_f, _field_series(read, r0_w, field)

        # ── Storage dtype for the returned channels ─────────────────────────────
        # f32 by DEFAULT. The walk is f32, the pack is f32 (compression.pack_position_arrays)
        # and SPEC 5.1 requires float32/float64 for `positions`, so f16 used to be the only
        # non-f32 link in the chain -- and a biased one: f16's smallest normal is 6.1e-5, so a
        # micron-scale coordinate in METRES is subnormal, with a flat ~6e-8 m quantum (12.5% of
        # a 0.5 um coordinate). A compartment is populated up to its wall and empty beyond, so
        # that quantum can only EJECT walkers, never recruit them: 3.1% of a confined population
        # at R=0.5um, scaling as ~1/R. The signal never noticed (|dS| <= 1.3e-5) but compartment
        # counting did. Pass storage_dtype=np.float16 to halve peak RAM on large walks, knowing
        # positions are then unfit for inside/outside classification -- use `comp_traj` instead,
        # which is computed here at f32 and stored as int8.
        _sdt = np.dtype(storage_dtype).type
        if _sdt not in (np.float16, np.float32, np.float64):
            raise ValueError(f"storage_dtype must be float16/32/64, got {storage_dtype!r}")

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
                    r, r_uw, key, comp_id = carry
                    # dlog_accum resets each save so the emitted value is the per-save delta.
                    inner_init = (r, r_uw, key, jnp.float32(0.0), comp_id)
                    (r_final, r_uw_final, key_final, dlog_accum, comp_final), mean = _interval_scan(
                        _inner_pm, inner_init, sub_steps, read)
                    return (r_final, r_uw_final, key_final, comp_final), \
                           (r_uw_final, dlog_accum, _compress_comp_pm(comp_final)), mean

                def simulate_one_walker_pm(r0_w, key_w, comp0_w, brem0_w, f_arrays):  # brem0 unused
                    read = _reader(f_arrays)
                    _, (positions, dlog_boundary, comp_types), field = _save_scan(
                        interval_pm, (r0_w, r0_w, key_w, comp0_w), n_t - 1, read, _f_every)
                    # save 0 is the start: the initial position, no contact yet, the initial pool
                    positions = jnp.concatenate([r0_w[None, :], positions], axis=0)
                    dlog_boundary = jnp.concatenate([jnp.zeros((1,), dlog_boundary.dtype), dlog_boundary])
                    comp_types = jnp.concatenate([_compress_comp_pm(comp0_w)[None], comp_types])
                    z = jnp.zeros_like(dlog_boundary)                       # placeholder bound_frac
                    return positions, dlog_boundary, comp_types, z, _field_series(read, r0_w, field)
            else:
                # ── MT path: bound_rem persists across saves ──
                def interval_pm(carry, read):
                    r, r_uw, key, comp_id, bound_rem = carry
                    inner_init = (r, r_uw, key, jnp.float32(0.0), comp_id, bound_rem, jnp.float32(0.0))
                    (r_final, r_uw_final, key_final, dlog_accum, comp_final, bound_rem_f, bound_acc), mean = \
                        _interval_scan(_inner_pm, inner_init, sub_steps, read)
                    bound_frac = bound_acc / jnp.float32(sub_steps)
                    return (r_final, r_uw_final, key_final, comp_final, bound_rem_f), \
                           (r_uw_final, dlog_accum, _compress_comp_pm(comp_final), bound_frac), mean

                def simulate_one_walker_pm(r0_w, key_w, comp0_w, brem0_w, f_arrays):
                    read = _reader(f_arrays)
                    _, (positions, dlog_boundary, comp_types, bound_frac), field = _save_scan(
                        interval_pm, (r0_w, r0_w, key_w, comp0_w, brem0_w), n_t - 1, read, _f_every)
                    # save 0 is the start: the initial position, no contact yet, the initial pool and bound state
                    positions = jnp.concatenate([r0_w[None, :], positions], axis=0)
                    dlog_boundary = jnp.concatenate([jnp.zeros((1,), dlog_boundary.dtype), dlog_boundary])
                    comp_types = jnp.concatenate([_compress_comp_pm(comp0_w)[None], comp_types])
                    bound_frac = jnp.concatenate([(brem0_w > 0).astype(bound_frac.dtype)[None], bound_frac])
                    return positions, dlog_boundary, comp_types, bound_frac, _field_series(read, r0_w, field)

            _simulate_batch_pm_raw = cached_batch(
                geometry, ("traj_packed_myelin", n_t, sub_steps, float(dt_sim), kappa_MT, dwell_time, _f_key),
                lambda: jax.jit(jax.vmap(simulate_one_walker_pm, in_axes=(0, 0, 0, 0, None))))

            def simulate_batch_pm(r0_b, keys_b, comp0_b, brem0_b):
                return _simulate_batch_pm_raw(r0_b, keys_b, comp0_b, brem0_b, _f_arrays)

        if record and not uses_myelin_traj:
            if has_permeability:
                kappa_over_D_relax = jnp.float32(float(permeability) / diffusivity)
                permeate_relax = geometry.permeate
                interact_relax = geometry.interact

                def inner_step_relax(carry, _):
                    r, key, dlog_accum, comp_sum, side, bad, comp, work = carry
                    key, step_key, perm_key = jax.random.split(key, 3)
                    unit_noise = isotropic_unit_step(step_key)
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
                    unit_noise = isotropic_unit_step(subkey)
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
                    unit_noise = isotropic_unit_step(subkey)
                    step = unit_noise * step_l_sim
                    r_new = reflect_free(r, step)
                    comp = geometry.classify_position_carry(r_new, comp)
                    comp_sum = comp_sum + _pool2(comp)
                    return (r_new, key, dlog_accum, comp_sum, side, bad, comp, work), None

            def interval_relax(carry, read):
                r, key, side, bad, comp, work = carry
                inner_init = (r, key, jnp.float32(0.0), jnp.float32(0.0), side, bad, comp, work)
                (r_final, key_final, dlog_accum, comp_sum, side_f, bad_f, comp_f, work_f), mean = _interval_scan(
                    inner_step_relax, inner_init, sub_steps, read)
                # Fractional occupancy of pool 1 (the enclosed pool) over the saved interval.
                comp_occ = comp_sum / jnp.float32(sub_steps)
                return (r_final, key_final, side_f, bad_f, comp_f, work_f), (r_final, dlog_accum, comp_occ), mean

            def simulate_one_walker_relax(r0_w, key_w, side_w, comp0_w, f_arrays):
                read = _reader(f_arrays)
                (_, _, _side_f, bad_f, _comp_f, work_f), (positions, dlog_boundary, comp_ids), field = _save_scan(
                    interval_relax, (r0_w, key_w, side_w, jnp.int32(0), comp0_w, _work0()), n_t - 1, read, _f_every)
                # save 0 is the start: the initial position, no contact yet, the initial occupancy
                positions = jnp.concatenate([r0_w[None, :], positions], axis=0)
                dlog_boundary = jnp.concatenate([jnp.zeros((1,), dlog_boundary.dtype), dlog_boundary])
                comp_ids = jnp.concatenate([jnp.asarray(_pool2(comp0_w), comp_ids.dtype)[None], comp_ids])   # the same collapse as the kernel's
                return positions, dlog_boundary, comp_ids, bad_f, work_f, _field_series(read, r0_w, field)

            _simulate_batch_relax_raw = cached_batch(
                geometry, ("traj_relax", n_t, sub_steps, float(dt_sim), diffusivity, _f_key, _counting),
                lambda: jax.jit(jax.vmap(simulate_one_walker_relax, in_axes=(0, 0, 0, 0, None))))

            def simulate_batch_relax(r0_b, keys_b):
                comp0_b = jnp.asarray(geometry.classify_positions_exact(r0_b), jnp.int32)
                pos, dlog, comp, bad_f, work_f, fs = _simulate_batch_relax_raw(r0_b, keys_b, _side0(r0_b), comp0_b, _f_arrays)
                _illegal_crossings[0] += int(jnp.sum(bad_f))
                _work_acc(work_f)
                return pos, dlog, comp, fs

        _simulate_batch_raw = cached_batch(
            geometry, ("traj", n_t, sub_steps, float(dt_sim), diffusivity, _f_key, _counting),
            lambda: jax.jit(jax.vmap(simulate_one_walker, in_axes=(0, 0, 0, None))))

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

        def simulate_batch(r0_b, keys_b):
            positions, _side_f, bad_f, work_f, fs = _simulate_batch_raw(r0_b, keys_b, _side0(r0_b), _f_arrays)
            _illegal_crossings[0] += int(jnp.sum(bad_f))
            _work_acc(work_f)
            return positions, fs

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
                    (r_f, _r_uw, key_f, _da, comp_f, brem_f, bacc), _ = jax.lax.scan(
                        _inner_pm, (r_w, r_w, key_w, jnp.float32(0.0), comp_w, brem_w,
                                    jnp.float32(0.0)), None, length=_n_chunk)
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

        all_batches = _Rows(n_walkers)                   # each channel written into one array as its batches arrive
        all_dlog_batches = _Rows(n_walkers) if record else None
        all_comp_batches = _Rows(n_walkers) if record else None
        all_bound_batches = _Rows(n_walkers) if _mt_on else None
        all_field_batches = _Rows(n_walkers) if _sampling else None

        # compress=K: each batch leaves the device as the pack's own C0/C2 coefficients (two exact endpoints and K
        # sine bands of the bridge per axis; the cumulative local time in the same form), so the host holds a
        # batch of raw positions at a time and the walk at K+2 numbers per axis per walker
        _compress = compress is not None
        _cx = {"K": int(compress) if _compress else 0, "n_t": None}
        all_blt_endpoints = _Rows(n_walkers) if (_compress and record) else None
        all_blt_starts = _Rows(n_walkers) if (_compress and record) else None
        if _compress and uses_myelin_traj:
            raise NotImplementedError("compress= is not wired for myelinated-cylinder walks (the MT bound channel has "
                                      "no bridge form); walk them uncompressed and pack with build_replay_pack.")
        from ..replay.compression import bridge_coefficients_device, boundary_coefficients_device

        def _compress_pos(pos_dev):
            """The batch's positions in the pack's C0 form, ``(b, K+2, 3)``, formed on the device: only the
            coefficients cross to the host (#446)."""
            C, K = bridge_coefficients_device(pos_dev, _cx["K"])
            _cx["K"], _cx["n_t"] = int(K), int(pos_dev.shape[1])
            return C

        for batch_idx, (start, end) in enumerate(run.batches(n_walkers, walker_batch_size)):
            batch_size = end - start

            current_r0 = r0_all[start:end]
            current_keys = walker_keys_all[start:end]
            if record and uses_myelin_traj:
                current_comp0 = comp0_all[start:end]
                current_brem0 = brem0_all[start:end]

            success = False
            while not success:
                try:
                    if record and uses_myelin_traj:
                        pos_f32, dlog_f32, comp_f32, bfrac_f32, fs_f32 = simulate_batch_pm(
                            current_r0, current_keys, current_comp0, current_brem0)
                        all_batches.append(np.array(pos_f32).astype(_sdt))
                        all_dlog_batches.append(np.array(dlog_f32).astype(_sdt))
                        all_comp_batches.append(np.array(comp_f32).astype(np.int8))
                        if _mt_on:
                            all_bound_batches.append(np.array(bfrac_f32).astype(_sdt))
                    elif record:
                        pos_f32, dlog_f32, comp_f32, fs_f32 = simulate_batch_relax(current_r0, current_keys)
                        if _compress:
                            all_batches.append(_compress_pos(pos_f32))
                            # the cumulative local time in the pack's C2 form, formed on the device
                            _start, _end, _bmodes = boundary_coefficients_device(dlog_f32, _cx["K"])
                            all_blt_starts.append(_start)
                            all_blt_endpoints.append(_end)
                            all_dlog_batches.append(_bmodes)
                        else:
                            all_batches.append(np.array(pos_f32).astype(_sdt))
                            all_dlog_batches.append(np.array(dlog_f32).astype(_sdt))
                        # Permeable: fractional occupancy; else a discrete label, ROUNDED: the occupancy
                        # is comp_sum / sub_steps and XLA may form it as comp_sum * (1/sub_steps), which
                        # for some sub-step counts is 0.99999994 -- truncating that to int8 labelled every
                        # intra walker "extra" (measured at sub_steps = 97).
                        all_comp_batches.append(np.array(comp_f32).astype(_sdt) if has_permeability
                                                else np.rint(np.array(comp_f32)).astype(np.int8))
                    else:
                        positions_f32, fs_f32 = simulate_batch(current_r0, current_keys)
                        if _compress:
                            all_batches.append(_compress_pos(positions_f32))
                        else:
                            all_batches.append(np.array(positions_f32).astype(_sdt))
                    if _sampling:
                        all_field_batches.append(np.asarray(fs_f32, np.float32))
                    success = True
                except Exception as e:
                    err_str = str(e)
                    if ("OOM" in err_str or "out of memory" in err_str.lower()
                            or "RESOURCE_EXHAUSTED" in err_str):
                        if _compress:
                            # The compressed path already keeps peak device memory low; a
                            # further sub-batch split here would need to compress each sub-slice
                            # too. Not wired yet — surface a clear message instead of raw OOM.
                            raise RuntimeError(
                                "compress= hit GPU OOM within a walker batch; lower "
                                "walker_batch_size (the compressed sub-batch fallback is not "
                                "yet implemented).") from e
                        new_sub_batch = batch_size // 2
                        if new_sub_batch < 1000:
                            raise RuntimeError(f"Batch size too small after OOM: {e}") from e
                        log.info(f"  OOM: halving sub-batch to {new_sub_batch}")
                        sub_pos_list = []
                        sub_dlog_list = [] if record else None
                        sub_comp_list = [] if record else None
                        sub_bound_list = [] if _mt_on else None
                        sub_field_list = [] if _sampling else None
                        for ss in range(0, batch_size, new_sub_batch):
                            se = min(ss + new_sub_batch, batch_size)
                            if record and uses_myelin_traj:
                                sp, sd, sc, sbf, sfs = simulate_batch_pm(
                                    current_r0[ss:se], current_keys[ss:se],
                                    current_comp0[ss:se], current_brem0[ss:se])
                                sub_pos_list.append(np.array(sp).astype(_sdt))
                                sub_dlog_list.append(np.array(sd).astype(_sdt))
                                sub_comp_list.append(np.array(sc).astype(np.int8))
                                if _mt_on:
                                    sub_bound_list.append(np.array(sbf).astype(_sdt))
                            elif record:
                                sp, sd, sc, sfs = simulate_batch_relax(
                                    current_r0[ss:se], current_keys[ss:se])
                                sub_pos_list.append(np.array(sp).astype(_sdt))
                                sub_dlog_list.append(np.array(sd).astype(_sdt))
                                sub_comp_list.append(np.array(sc).astype(_sdt) if has_permeability
                                                     else np.rint(np.array(sc)).astype(np.int8))
                            else:
                                sp, sfs = simulate_batch(current_r0[ss:se], current_keys[ss:se])
                                sub_pos_list.append(np.array(sp).astype(_sdt))
                            if _sampling:
                                sub_field_list.append(np.asarray(sfs, np.float32))
                        all_batches.append(np.concatenate(sub_pos_list, axis=0))
                        if record:
                            all_dlog_batches.append(np.concatenate(sub_dlog_list, axis=0))
                            all_comp_batches.append(np.concatenate(sub_comp_list, axis=0))
                            if _mt_on:
                                all_bound_batches.append(np.concatenate(sub_bound_list, axis=0))
                        if _sampling:
                            all_field_batches.append(np.concatenate(sub_field_list, axis=0))
                        success = True
                    else:
                        raise

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
                RuntimeWarning, stacklevel=2)

        # ── The work record: what the walk cost in wall encounters, for a cost model per substrate class
        # (dmrai-lab/tessera#12) and for sizing the bounce budget -- a budget no lane exhausted gives the same
        # trajectories as any larger one, by the loop's own rule.
        _loop = geometry.bounce_loop
        _counted = _counting and _work[3]           # the myelin kernels do not report their work yet
        work = dict(walker_steps=int(n_walkers) * (int(n_t) - 1) * int(sub_steps),
                    bounce_budget=(None if _loop is None else int(_loop.budget)),
                    n_hits=(int(_work[0]) if _counted else None),
                    n_crossings=(int(_work[1]) if _counted else None),
                    exhausted_steps=(int(_work[2]) if _counted else None))
        if _counted and _work[2]:
            import warnings
            warnings.warn(
                f"{_work[2]} walker-steps exhausted the bounce budget of {work['bounce_budget']} with path left "
                f"untested; the budget is too small for this substrate at this step. See PersistentWalk.work.",
                RuntimeWarning, stacklevel=2)

        if _compress:
            # the walk in the pack's own coefficient form: `pos_modes` holds [r(0), r(T) - r(0), the K sine bands]
            # per axis, which `build_replay_pack` stores as they are and `compression.decode` expands
            master = {
                "compressed": True, "method": "bridge_dst", "K": _cx["K"],
                "n_t": int(_cx["n_t"]), "dt_traj": dt_actual,
                "sub_steps": sub_steps, "dt_sim": dt_sim,
                "pos_modes": all_batches.array(),        # (N, K+2, 3) f32
            }
            if record:
                master["blt_endpoint"] = all_blt_endpoints.array()  # (N,)
                master["blt_start"] = all_blt_starts.array()        # (N,)
                master["blt_modes"] = all_dlog_batches.array()      # (N, K)
                master["comp_traj"] = all_comp_batches.array()      # (N, n_t)
            return master

        D_walk = None if diffusivity is None else float(diffusivity)
        walk = PersistentWalk(all_batches.array(), float(dt_actual), int(sub_steps),
                          float(dt_sim), illegal_crossings=illegal, seed=int(seed), diffusivity=D_walk,
                          geometry=geometry, work=work)
        if record:
            # a geometry without a log-weight reflection accumulates no boundary local time: the channel is then
            # NOT a record of zero contact but the absence of a record, and a pack built from it must not claim the
            # surface tier (a replay at rho would return an unattenuated signal without a word)
            records_surface = has_reflect_with_log_weight or geometry._is_myelinated or geometry._is_packed_myelinated
            blt = all_dlog_batches.array() if records_surface else None   # the myelin kernels accumulate it too
            walk = PersistentWalk(walk.positions, walk.dt, walk.sub_steps, walk.dt_sim,
                              boundary_local_time=blt,
                              compartment=all_comp_batches.array(),
                              bound_frac=(all_bound_batches.array() if _mt_on else None),
                              illegal_crossings=illegal, seed=int(seed), diffusivity=D_walk, geometry=geometry, work=work)
        if _sampling:
            import dataclasses
            walk = dataclasses.replace(walk, field_basis=field_basis, field_samples=all_field_batches.array(),
                                       field_sample_every=_f_every)
        object.__setattr__(walk, "run", run)
        return walk
