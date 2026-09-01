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

from .physics import (make_step_fn, make_myelin_step_fn, make_packed_myelin_step_fn,
                      make_packed_myelin_traj_step_fn)
from .waveforms import Waveform


# ═══════════════════════════════════════════════════════════════════════════
# ENGINE ROUTING TABLE (Phase 5)
# ═══════════════════════════════════════════════════════════════════════════
# simulate(engine=) selects the backend that turns a walk into a signal:
#
#   "fused"  — the inline single-pass jax.lax.scan step-kernels in this module.
#              The validation ORACLE and the universal fallback; byte-for-byte
#              the pre-replay engine (nothing below the routing block changed).
#   "replay" — walk ONCE (simulate_trajectories(save_relaxation_data=…)) then
#              replay (gradient phase + scalar/per-comp
#              T2 + T1 + surface relaxivity).  This mirrors the private
#              packed-myelin unification, generalised to the geometries Phase-1
#              proved equivalent at the MC-noise floor (test_replay_parity).
#   "auto"   — default; replay where validated-equivalent AND the suite stays
#              green, else transparently fused.
#
# "auto" routing decided EMPIRICALLY by keeping the full suite green:
#
#   geometry \ effect    | gradient | scalar T2 | T1 | surface ρ  → engine
#   ---------------------+----------+-----------+----+-----------------------
#   FreeDiffusion        |  REPLAY  |  REPLAY   | R  |    n/a
#   Box1D                |  REPLAY  |  REPLAY   | R  |  REPLAY
#   Sphere               |  REPLAY  |  REPLAY   | R  |  REPLAY
#   Cylinder             |  REPLAY  |  REPLAY   | R  |  REPLAY
#   Ellipsoid            |  REPLAY  |  REPLAY   | R  |  REPLAY
#   PackedCylinders      |  REPLAY  |  REPLAY   | R  |  REPLAY
#   PackedSpheres        |  REPLAY  |  REPLAY   | R  |  REPLAY
#   Mesh (impermeable)   |  REPLAY  |  REPLAY   | R  |  REPLAY
#   ---------------------+----------+-----------+----+-----------------------
#   MyelinatedCylinder   |  FUSED  (dedicated fused step kernel; no replay walk)
#   PackedMyelinatedCyl. |  FUSED  (fused single-reflection kernel is NOT
#                         position-parity with the multi-bounce replay walk —
#                         the private repo unified these by REPLACING the fused
#                         kernel; the public repo keeps both, so rerouting would
#                         shift test_packed_myelinated_cylinders)
#   permeability (any)   |  FUSED  (membrane crossing is single-pass walk
#                         semantics; scalar replay has no Mz reservoir for
#                         exchange across a longitudinal-storage mixing time)
#   per-comp D/T2 Mesh   |  FUSED  (intra/extra dicts resolved per-step in
#                         make_step_fn; not wired into the replay operator here)
#
# Fused-only REQUESTS (any geometry) — replay raises NotImplementedError, auto
# falls back to fused: return_positions, return_compartments,
# return_walker_signals (single-pass internals).  simulate_cpmg / simulate_
# mixture / simulate_bloch stay fused (out of Phase-5 scope).
# ═══════════════════════════════════════════════════════════════════════════

# Geometry families "auto" is allowed to route to replay (see table above).  A
# geometry qualifies only if it is one of these AND _replay_gap() returns None.
_REPLAY_AUTO_GEOM_NAMES = frozenset({
    "FreeDiffusion", "Box1D", "Sphere", "Cylinder", "Ellipsoid",
    "PackedCylinders", "PackedSpheres", "Mesh",
})


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
    if getattr(geometry, '_is_myelinated', False):
        return "MyelinatedCylinder uses a dedicated fused step kernel (no replay walk)"
    if getattr(geometry, '_is_packed_myelinated', False):
        return ("PackedMyelinatedCylinders fused single-reflection kernel is not "
                "position-parity with the multi-bounce replay walk")
    if getattr(geometry, 'permeability', None) is not None:
        return ("membrane permeability is single-pass walk semantics "
                "(fused-only; scalar replay has no exchange reservoir)")
    if getattr(geometry, '_D_comp_jax', None) is not None:
        # Per-compartment D changes the STEP LENGTH per compartment, so it alters the walk
        # itself — not a replay knob. (Per-compartment T2/T1 ARE replay knobs: they gate
        # only log_w and are applied off the saved compartment channel — see
        # _simulate_via_replay.)
        return "per-compartment diffusivity (Mesh intra/extra D) alters the walk (fused-only)"
    if diffusivity is None:
        return "replay needs an explicit diffusivity for the sub-step auto-tune"
    return None


def _replay_auto_allowed(geometry):
    """Whether ``engine='auto'`` may route this geometry to replay.  Restricted
    to the families proven suite-green; a stricter gate than _replay_gap()."""
    return type(geometry).__name__ in _REPLAY_AUTO_GEOM_NAMES


def _simulate_via_replay(n_walkers, diffusivity, waveform, geometry, *, seed,
                         T2, T1, r0, require_gpu, walker_batch_size):
    """Signal via the replay backend: walk once, then apply the waveform +
    relaxation.  The producer walk depends only on (geometry, diffusivity,
    seed) — the replay invariant — so it reproduces the fused ``simulate()`` to
    the MC-noise floor on the supported matrix (see the routing table).

    Handles gradient phase, scalar T2, T1 (chi_perp-gated longitudinal storage),
    surface relaxivity (ρ replayed off the recorded unit boundary local time),
    and the stimulated-echo 0.5 factor.  Assumes ``_replay_gap()`` already
    cleared the run (no permeability / myelin / per-comp / extra-output)."""
    from .trajectories import replay

    G = np.asarray(waveform.G, dtype=np.float32)          # (n_meas, n_t, 3)
    dt = float(waveform.dt)
    n_t = G.shape[1]
    T_max = dt * (n_t - 1)

    # Acquisition rotation: place the substrate in the bore by rotating G into
    # the geometry's native frame (walk stays native), exactly as fused does.
    _orient_R = getattr(geometry, '_orient_R', None)
    if _orient_R is not None:
        G = G @ np.asarray(_orient_R, dtype=np.float32)

    # Transverse-coherence schedule (spin echo = all ones; PGSTE zeros the TM).
    chi_perp = getattr(waveform, 'chi_perp', None)
    chi = (np.asarray(chi_perp, dtype=np.float64).reshape(n_t)
           if chi_perp is not None else np.ones(n_t, dtype=np.float64))

    # Surface relaxivity is a wall effect recorded (with ρ/D = 1) during the
    # walk and multiplied by ρ/D at replay — only then do we need the boundary
    # local-time channel.  Scalar T2/T1 are walker-independent replay knobs.
    rho = getattr(geometry, 'surface_relaxivity_t2', None)
    has_surf = (rho is not None and float(rho) > 0.0
                and hasattr(geometry, 'reflect_with_log_weight'))

    # Per-compartment T2/T1 (Mesh intra/extra dicts) are pure replay knobs: they gate only
    # log_w and are applied off the saved compartment channel (comp_traj, index 0=intra /
    # 1=extra, matching the geometry's _T2_comp/_T1_comp ordering). Requesting them forces
    # save_relaxation_data so the compartment channel is recorded.
    T2_comp = getattr(geometry, '_T2_comp', None)
    T1_comp = getattr(geometry, '_T1_comp', None)
    has_per_comp = T2_comp is not None or T1_comp is not None

    save_relax = has_surf or has_per_comp
    st_kwargs = dict(seed=seed, require_gpu=require_gpu,
                     save_relaxation_data=save_relax)
    if r0 is not None:
        st_kwargs['r0'] = r0
    if walker_batch_size is not None:
        st_kwargs['walker_batch_size'] = walker_batch_size

    out = simulate_trajectories(n_walkers, diffusivity, geometry, T_max, dt,
                                **st_kwargs)
    if save_relax:
        traj, dt_traj, _sub, _dtsim, dlog, comp = out
    else:
        traj, dt_traj, _sub, _dtsim = out
        dlog = comp = None

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
    waveform : Waveform
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

        Compartment integer IDs:

        - ``Cylinder``, ``Sphere``, ``Ellipsoid``, ``Box1D``:
          0 = intra, 1 = extra.
        - ``MyelinatedCylinder``:
          0 = intra-axonal, 1 = myelin, 2 = extra-axonal.
        - ``PackedCylinders``:
          0 = extra-axonal, 1..N = intra cylinder k (1-indexed).

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
          :func:`~dmipy_sim.trajectories.replay`
          (gradient phase + scalar/per-comp T2 + T1 + surface relaxivity).
          Raises :class:`NotImplementedError` (naming the gap) for a path the
          replay backend cannot serve exactly — MyelinatedCylinder /
          PackedMyelinatedCylinders, membrane permeability, per-compartment
          D/T2 meshes, or the ``return_positions``/``return_compartments``/
          ``return_walker_signals`` single-pass internals.
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
                walker_batch_size=walker_batch_size)
        # engine == "auto": replay only where validated-equivalent AND green.
        if _gap is None and _replay_auto_allowed(geometry):
            return _simulate_via_replay(
                n_walkers, diffusivity, _wf_r, geometry, seed=seed,
                T2=T2, T1=T1, r0=r0, require_gpu=require_gpu,
                walker_batch_size=walker_batch_size)
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
                    engine="fused", _allow_oom_backoff=False)
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
            return_walker_signals=return_walker_signals)

    # Accept AcquisitionScheme (any object with .waveform) or raw Waveform
    if hasattr(waveform, 'waveform'):
        waveform = waveform.waveform
    G = waveform.G          # (n_measurements, n_t, 3)
    dt = waveform.dt
    echo_idx = waveform.echo_idx

    # Substrate placement in the bore (e.g. Mesh with orientation/R): the walk runs
    # in the geometry's native frame, so rotate the ACQUISITION into that frame
    # instead of rotating the geometry.  A gradient g in the lab (B0=+z) frame is
    # g_mesh = R^T g for a mesh->lab rotation R, i.e. G_mesh = G @ R.
    _orient_R = getattr(geometry, '_orient_R', None)
    if _orient_R is not None:
        G = G @ jnp.asarray(_orient_R, G.dtype)

    n_measurements, n_t, _ = G.shape

    # Spin-density-weighted ensemble signal Re(<w_spin . exp(log_w) . e^{i phi}>)/Σw_spin.
    # w_spin is the per-walker n(r0) proton-density weight (myelin < 1); homogeneous
    # placement + this weight avoids per-geometry placement re-weighting.
    def _ens(sw, logw, phi):
        return jnp.sum(sw[:, None] * jnp.exp(logw[:, None]) * jnp.cos(phi), axis=0) / jnp.sum(sw)
    def _ens_np(sw, phi):
        return jnp.sum(sw[:, None] * jnp.cos(phi), axis=0) / jnp.sum(sw)

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

    # Build per-walker PRNG keys
    master_key = jax.random.PRNGKey(seed)
    pos_key, walker_key = jax.random.split(master_key)
    walker_keys = jax.random.split(walker_key, n_walkers)

    # Initial positions — use caller-supplied r0 or let geometry place walkers
    _r0_user_supplied = r0 is not None
    if r0 is None:
        r0 = geometry.init_positions(n_walkers, pos_key)  # (n_walkers, 3)
    else:
        r0 = jnp.array(r0, dtype=jnp.float32)           # (n_walkers, 3)

    # Check if this is a MyelinatedCylinder or LabelMap2D (custom step function path)
    is_myelin = getattr(geometry, '_is_myelinated', False)
    is_packed_myelin = getattr(geometry, '_is_packed_myelinated', False)

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
    if track_comp:
        if is_myelin or is_packed_myelin:
            # compartments0 is set during init_positions() call above.
            # We read it after simulate to avoid forward-reference issues.
            pass  # set later after the geometry-specific init
        else:
            # Standard geometry: vmap classify_position over initial positions
            classify_fn = geometry.classify_position
            # Initial labels are the one place an exact test is affordable and necessary: there is no
            # previous label to carry, so an undecidable point must be resolved rather than defaulted.
            exact_fn = getattr(geometry, 'classify_positions_exact', None)
            comp_origin_jax = (exact_fn(r0) if exact_fn is not None
                               else jax.vmap(classify_fn)(r0))     # (n_walkers,) int32

    if is_myelin:
        # MyelinatedCylinder: extended carry state (r, phi, log_w, compartment_id, key)
        step_fn = make_myelin_step_fn(geometry, dt, T1=T1)
        compartments0 = geometry._init_compartments  # (n_walkers,) int32
        spin_w = jnp.asarray(geometry.water_fractions, jnp.float32)[compartments0]

        if track_comp:
            comp_origin_jax = compartments0

            if return_compartments == 'full':
                def simulate_walker(r0_w, key_w, comp0):
                    phi0   = jnp.zeros(n_measurements, dtype=jnp.float32)
                    log_w0 = jnp.float32(0.0)
                    # Emit compartment_id at every step
                    def step_with_comp(carry, inputs):
                        new_carry, _ = step_fn(carry, inputs)
                        comp_out = new_carry[3]  # compartment_id at carry position 3
                        return new_carry, comp_out

                    (r_final, phi_all, log_w, comp_final, _), comp_seq = jax.lax.scan(
                        step_with_comp, (r0_w, phi0, log_w0, comp0, key_w), scan_inputs)
                    return r_final, phi_all, log_w, comp_final, comp_seq

                simulate_batch = jax.vmap(simulate_walker, in_axes=(0, 0, 0))
                final_r, all_phi, all_log_w, comp_final, comp_seq = simulate_batch(
                    r0, walker_keys, compartments0)
                signals = _ens(spin_w, all_log_w, all_phi)

            else:  # 'final'
                def simulate_walker(r0_w, key_w, comp0):
                    phi0   = jnp.zeros(n_measurements, dtype=jnp.float32)
                    log_w0 = jnp.float32(0.0)
                    (r_final, phi_all, log_w, comp_final, _), _ = jax.lax.scan(
                        step_fn, (r0_w, phi0, log_w0, comp0, key_w), scan_inputs)
                    return r_final, phi_all, log_w, comp_final

                simulate_batch = jax.vmap(simulate_walker, in_axes=(0, 0, 0))
                final_r, all_phi, all_log_w, comp_final = simulate_batch(
                    r0, walker_keys, compartments0)
                signals = _ens(spin_w, all_log_w, all_phi)

        else:
            def simulate_walker(r0_w, key_w, comp0):
                phi0   = jnp.zeros(n_measurements, dtype=jnp.float32)
                log_w0 = jnp.float32(0.0)
                (r_final, phi_all, log_w, comp_final, _), _ = jax.lax.scan(
                    step_fn, (r0_w, phi0, log_w0, comp0, key_w), scan_inputs)
                return r_final, phi_all, log_w

            simulate_batch = jax.vmap(simulate_walker, in_axes=(0, 0, 0))
            final_r, all_phi, all_log_w = simulate_batch(r0, walker_keys, compartments0)
            signals = _ens(spin_w, all_log_w, all_phi)

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

        def _to3(cid):                                     # -> 0=intra, 1=myelin, 2=extra
            return jnp.where(cid == jnp.int32(0), jnp.int32(2),
                    jnp.where(cid > jnp.int32(geometry.N_max), jnp.int32(1), jnp.int32(0)))
        spin_w = jnp.where(_to3(compartments0) == jnp.int32(1),
                           jnp.float32(geometry._myelin_proton_density), jnp.float32(1.0))

        def simulate_walker(r0_w, key_w, comp0):
            phi0 = jnp.zeros(n_measurements, dtype=jnp.float32)

            def emit(carry, inputs):
                nc, _ = step_fn(carry, inputs)
                return nc, _to3(nc[4])                     # nc[4] = compartment_id
            (r_ic_f, _r_uw_f, phi_all, log_w, comp_f, _), comp_seq_w = jax.lax.scan(
                emit, (r0_w, r0_w, phi0, jnp.float32(0.0), comp0, key_w), scan_inputs)
            return r_ic_f, phi_all, log_w, comp_f, comp_seq_w

        final_r, all_phi, all_log_w, _comp_final_enc, _comp_seq = jax.vmap(
            simulate_walker, in_axes=(0, 0, 0))(r0, walker_keys, compartments0)
        signals = _ens(spin_w, all_log_w, all_phi)
        if track_comp:
            comp_origin_jax = _to3(compartments0)
            comp_final = _to3(_comp_final_enc)
            comp_seq = _comp_seq
    else:
        # Standard geometry path
        # Build scan body for this geometry and diffusivity
        # T2/T1 are passed in so they are accumulated per-walker inside the scan
        step_fn, has_weight = make_step_fn(geometry, diffusivity, dt, T2=T2, T1=T1)
        spin_w = jnp.ones((n_walkers,), dtype=jnp.float32)

        if want_pos_full:
            # Per-timestep position export (additive path; existing True/'final'
            # scans are untouched).  Emits r at every step, plus the compartment
            # id when tracking — e.g. to select walkers that permeated and plot
            # only their trajectories.  pos_seq: (n_walkers, n_timesteps, 3).
            classify_fn = geometry.classify_position
            if has_weight:
                def simulate_walker(r0_w, key_w):
                    phi0 = jnp.zeros(n_measurements, dtype=jnp.float32)

                    def body(carry, inp):
                        (rn, pn, ln, kn), _ = step_fn(carry, inp)
                        return (rn, pn, ln, kn), ((rn, classify_fn(rn)) if track_comp else rn)
                    (r_final, phi_all, log_w, _), ys = jax.lax.scan(
                        body, (r0_w, phi0, jnp.float32(0.0), key_w), scan_inputs)
                    return r_final, phi_all, log_w, ys

                final_r, all_phi, all_log_w, ys = jax.vmap(
                    simulate_walker, in_axes=(0, 0))(r0, walker_keys)
                signals = _ens(spin_w, all_log_w, all_phi)
            else:
                def simulate_walker(r0_w, key_w):
                    phi0 = jnp.zeros(n_measurements, dtype=jnp.float32)

                    def body(carry, inp):
                        (rn, pn, kn), _ = step_fn(carry, inp)
                        return (rn, pn, kn), ((rn, classify_fn(rn)) if track_comp else rn)
                    (r_final, phi_all, _), ys = jax.lax.scan(
                        body, (r0_w, phi0, key_w), scan_inputs)
                    return r_final, phi_all, ys

                final_r, all_phi, ys = jax.vmap(
                    simulate_walker, in_axes=(0, 0))(r0, walker_keys)
                signals = _ens_np(spin_w, all_phi)
            if track_comp:
                pos_seq, comp_seq = ys          # (n_w, n_t, 3), (n_w, n_t)
                comp_final = comp_seq[:, -1]
            else:
                pos_seq = ys

        elif track_comp:
            # Need a classify_position closure for the scan body. Where the geometry can say that a
            # position is undecidable (no wall within reach), prefer carrying the previous label: the
            # walker cannot have crossed a boundary it was never near, and re-deriving would make its
            # compartment depend on the local mesh resolution. See Mesh.classify_position_carry.
            classify_fn = geometry.classify_position
            carry_fn = getattr(geometry, 'classify_position_carry', None)

            if has_weight:
                # carry = (r, phi, log_weight, compartment_current, key)
                def step_fn_comp(carry, inputs):
                    r, phi, log_weight, comp_cur, key = carry
                    # Run the original step_fn with its expected carry format
                    orig_carry = (r, phi, log_weight, key)
                    (r_new, phi_new, log_new, key_new), _ = step_fn(orig_carry, inputs)
                    comp_new = (carry_fn(r_new, comp_cur) if carry_fn is not None
                                else classify_fn(r_new))
                    return (r_new, phi_new, log_new, comp_new, key_new), comp_new

                if return_compartments == 'full':
                    def simulate_walker(r0_w, key_w, comp0):
                        phi0   = jnp.zeros(n_measurements, dtype=jnp.float32)
                        log_w0 = jnp.float32(0.0)
                        (r_final, phi_all, log_w, comp_final, _), comp_seq = jax.lax.scan(
                            step_fn_comp, (r0_w, phi0, log_w0, comp0, key_w), scan_inputs)
                        return r_final, phi_all, log_w, comp_final, comp_seq

                    simulate_batch = jax.vmap(simulate_walker, in_axes=(0, 0, 0))
                    final_r, all_phi, all_log_w, comp_final, comp_seq = simulate_batch(
                        r0, walker_keys, comp_origin_jax)
                    signals = _ens(spin_w, all_log_w, all_phi)

                else:  # 'final'
                    def simulate_walker(r0_w, key_w, comp0):
                        phi0   = jnp.zeros(n_measurements, dtype=jnp.float32)
                        log_w0 = jnp.float32(0.0)
                        (r_final, phi_all, log_w, comp_final, _), _ = jax.lax.scan(
                            step_fn_comp, (r0_w, phi0, log_w0, comp0, key_w), scan_inputs)
                        return r_final, phi_all, log_w, comp_final

                    simulate_batch = jax.vmap(simulate_walker, in_axes=(0, 0, 0))
                    final_r, all_phi, all_log_w, comp_final = simulate_batch(
                        r0, walker_keys, comp_origin_jax)
                    signals = _ens(spin_w, all_log_w, all_phi)

            else:
                # carry = (r, phi, compartment_current, key)
                def step_fn_comp(carry, inputs):
                    r, phi, comp_cur, key = carry
                    orig_carry = (r, phi, key)
                    (r_new, phi_new, key_new), _ = step_fn(orig_carry, inputs)
                    comp_new = (carry_fn(r_new, comp_cur) if carry_fn is not None
                                else classify_fn(r_new))
                    return (r_new, phi_new, comp_new, key_new), comp_new

                if return_compartments == 'full':
                    def simulate_walker(r0_w, key_w, comp0):
                        phi0 = jnp.zeros(n_measurements, dtype=jnp.float32)
                        (r_final, phi_all, comp_final, _), comp_seq = jax.lax.scan(
                            step_fn_comp, (r0_w, phi0, comp0, key_w), scan_inputs)
                        return r_final, phi_all, comp_final, comp_seq

                    simulate_batch = jax.vmap(simulate_walker, in_axes=(0, 0, 0))
                    final_r, all_phi, comp_final, comp_seq = simulate_batch(
                        r0, walker_keys, comp_origin_jax)
                    signals = _ens_np(spin_w, all_phi)

                else:  # 'final'
                    def simulate_walker(r0_w, key_w, comp0):
                        phi0 = jnp.zeros(n_measurements, dtype=jnp.float32)
                        (r_final, phi_all, comp_final, _), _ = jax.lax.scan(
                            step_fn_comp, (r0_w, phi0, comp0, key_w), scan_inputs)
                        return r_final, phi_all, comp_final

                    simulate_batch = jax.vmap(simulate_walker, in_axes=(0, 0, 0))
                    final_r, all_phi, comp_final = simulate_batch(
                        r0, walker_keys, comp_origin_jax)
                    signals = _ens_np(spin_w, all_phi)

        else:
            # Original code path (no compartment tracking)
            if has_weight:
                # Surface relaxation path: carry includes per-walker log-weight
                def simulate_walker(r0_w, key_w):
                    phi0   = jnp.zeros(n_measurements, dtype=jnp.float32)
                    log_w0 = jnp.float32(0.0)
                    (r_final, phi_all, log_w, _), _ = jax.lax.scan(
                        step_fn, (r0_w, phi0, log_w0, key_w), scan_inputs)
                    return r_final, phi_all, log_w

                simulate_batch = jax.vmap(simulate_walker, in_axes=(0, 0))
                final_r, all_phi, all_log_w = simulate_batch(r0, walker_keys)
                # Signal: Re(<w · exp(i·phi)>) = <exp(log_w) · cos(phi)>
                # all_log_w: (n_walkers,); all_phi: (n_walkers, n_measurements)
                # [:, None] keeps broadcasting as (n_walkers, 1) × (n_walkers, n_meas)
                signals = _ens(spin_w, all_log_w, all_phi)

            else:
                # Standard path (no surface relaxation, no permeability)
                def simulate_walker(r0_w, key_w):
                    phi0 = jnp.zeros(n_measurements, dtype=jnp.float32)
                    (r_final, phi_all, _), _ = jax.lax.scan(
                        step_fn, (r0_w, phi0, key_w), scan_inputs)
                    return r_final, phi_all

                simulate_batch = jax.vmap(simulate_walker, in_axes=(0, 0))
                final_r, all_phi = simulate_batch(r0, walker_keys)
                # Signal: Re(<exp(i*phi)>) = <cos(phi)>
                signals = _ens_np(spin_w, all_phi)  # (n_measurements,)

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
                                return_walker_signals):
    """Run simulate() over walker chunks and recombine (see simulate's
    ``walker_batch_size``).  The signal is a plain walker-mean, so it recombines
    as a size-weighted mean; per-walker outputs (positions, compartments, walker
    signals) are concatenated.  Each chunk's device buffers are released when its
    (host) results are returned, so peak device memory is one chunk."""
    n_batches = (n_walkers + walker_batch_size - 1) // walker_batch_size
    track_comp = return_compartments is not False
    sig_acc = None
    pos_l, origin_l, comp_l, lw_l, phi_l = [], [], [], [], []

    for b in range(n_batches):
        start = b * walker_batch_size
        end = min(start + walker_batch_size, n_walkers)
        nb = end - start
        print(f"  simulate: walkers {start}–{end - 1} "
              f"({int(100 * end / n_walkers)}%)...", flush=True)
        out = simulate(
            n_walkers=nb, diffusivity=diffusivity, waveform=waveform,
            geometry=geometry, seed=seed + 1 + b, T2=T2, T1=T1,
            r0=(None if r0 is None else r0[start:end]),
            return_positions=return_positions,
            return_compartments=return_compartments,
            return_walker_signals=return_walker_signals,
            walker_batch_size=None, require_gpu=False,
            engine="fused", _allow_oom_backoff=False)

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


def simulate_mixture(compartments, waveform, seed=123):
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
    waveform : Waveform
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
        )
        weighted = comp['fraction'] * s
        signal = weighted if signal is None else signal + weighted

    return signal


def simulate_cpmg(n_walkers, diffusivity, waveform, geometry, *,
                  T2=None, seed=123, walker_batch_size=None, require_gpu=None):
    """Multi-echo CPMG signal from a SINGLE diffusion walk.

    Walks the spin ensemble once through the full CPMG train (ideal instantaneous
    180° refocusing is encoded as the sign flips of ``waveform.G``) and samples the
    ensemble signal ``Re<exp(iφ)·exp(log_w)>`` at each echo time.  This is the
    ordinary forward model: one pass through the train, nothing cached or reused.
    Build ``waveform`` with :func:`dmipy_sim.cpmg`
    (which sets ``echo_indices``).

    Parameters
    ----------
    n_walkers : int
    diffusivity : float or None
        Bulk diffusivity (m²/s); omit for MyelinatedCylinder (D in the geometry).
    waveform : Waveform
        A multi-echo waveform carrying ``echo_indices`` (e.g. from ``cpmg``).
    geometry : Geometry
    T2 : float, optional
        Transverse relaxation time (s), accumulated per-walker in the walk.
    seed, walker_batch_size, require_gpu : see :func:`simulate`.

    Returns
    -------
    signals : np.ndarray, shape (n_echoes, n_measurements), float32
        Signal at each echo (echo k = k·TE), one column per gradient direction.
    """
    from .gpu import check_gpu
    check_gpu(n_walkers, require_gpu, what="simulate_cpmg")

    if hasattr(waveform, 'waveform'):
        waveform = waveform.waveform
    echo_indices = getattr(waveform, 'echo_indices', None)
    if echo_indices is None:
        raise ValueError(
            "simulate_cpmg needs a multi-echo waveform with echo_indices set; "
            "build it with dmipy_sim.cpmg(...).")
    echo_indices = np.asarray(echo_indices, dtype=int)

    # Walker batching: one echo-signal accumulator, size-weighted mean over chunks.
    if walker_batch_size is not None and walker_batch_size < n_walkers:
        n_batches = (n_walkers + walker_batch_size - 1) // walker_batch_size
        acc = None
        for b in range(n_batches):
            start = b * walker_batch_size
            nb = min(walker_batch_size, n_walkers - start)
            print(f"  simulate_cpmg: walkers {start}–{start + nb - 1} "
                  f"({int(100 * (start + nb) / n_walkers)}%)...", flush=True)
            s = simulate_cpmg(nb, diffusivity, waveform, geometry, T2=T2,
                              seed=seed + 1 + b, walker_batch_size=None,
                              require_gpu=False)
            acc = s * nb if acc is None else acc + s * nb
        return acc / n_walkers

    G = waveform.G                     # (n_measurements, n_t, 3)
    dt = waveform.dt
    n_measurements, n_t, _ = G.shape

    # Spin-density-weighted ensemble signal Re(<w_spin . exp(log_w) . e^{i phi}>)/Σw_spin.
    # w_spin is the per-walker n(r0) proton-density weight (myelin < 1); homogeneous
    # placement + this weight avoids per-geometry placement re-weighting.
    def _ens(sw, logw, phi):
        return jnp.sum(sw[:, None] * jnp.exp(logw[:, None]) * jnp.cos(phi), axis=0) / jnp.sum(sw)
    def _ens_np(sw, phi):
        return jnp.sum(sw[:, None] * jnp.cos(phi), axis=0) / jnp.sum(sw)
    G_scan = jnp.transpose(G, (1, 0, 2))   # (n_t, n_measurements, 3)
    # CPMG is a spin-echo train: magnetisation is transverse throughout, so the
    # coherence flag is 1 at every step (step_fn receives inputs = (g_t, chi_t)).
    chi_perp = getattr(waveform, 'chi_perp', None)
    if chi_perp is not None:
        chi_perp_scan = jnp.asarray(chi_perp, dtype=jnp.float32).reshape(n_t)
    else:
        chi_perp_scan = jnp.ones((n_t,), dtype=jnp.float32)
    scan_inputs = (G_scan, chi_perp_scan)

    master_key = jax.random.PRNGKey(seed)
    pos_key, walker_key = jax.random.split(master_key)
    walker_keys = jax.random.split(walker_key, n_walkers)
    r0 = geometry.init_positions(n_walkers, pos_key)

    step_fn, has_weight = make_step_fn(geometry, diffusivity, dt, T2=T2)

    if has_weight:
        def step_emit(carry, inputs):
            new_carry, _ = step_fn(carry, inputs)
            _, phi, log_w, _ = new_carry
            return new_carry, jnp.exp(log_w) * jnp.cos(phi)   # (n_measurements,)

        def walk(r0_w, key_w):
            phi0 = jnp.zeros(n_measurements, dtype=jnp.float32)
            _, s_trace = jax.lax.scan(
                step_emit, (r0_w, phi0, jnp.float32(0.0), key_w), scan_inputs)
            return s_trace                                    # (n_t, n_measurements)
    else:
        def step_emit(carry, inputs):
            new_carry, _ = step_fn(carry, inputs)
            _, phi, _ = new_carry
            return new_carry, jnp.cos(phi)

        def walk(r0_w, key_w):
            phi0 = jnp.zeros(n_measurements, dtype=jnp.float32)
            _, s_trace = jax.lax.scan(step_emit, (r0_w, phi0, key_w), scan_inputs)
            return s_trace

    all_traces = jax.vmap(walk, in_axes=(0, 0))(r0, walker_keys)   # (n_walkers, n_t, n_meas)
    signal_trace = jnp.mean(all_traces, axis=0)                    # (n_t, n_meas)
    echo_signals = signal_trace[echo_indices]                     # (n_echoes, n_meas)
    return np.array(echo_signals)


def simulate_trajectories(
    n_walkers: int,
    diffusivity: float,
    geometry,
    T_max: float,
    dt_save: float,
    seed: int = 42,
    walker_batch_size: int = 50_000,
    save_relaxation_data: bool = False,
    require_gpu=None,
    r0=None,
    kappa_MT: float = 0.0,
    dwell_time: float = 0.0,
    equilibrate_binding="auto",
    compress: int = None,
) -> tuple:
    """Walk the spins ONCE and save positions at every saved time step — the
    producer for the replay path (:mod:`dmipy_sim.trajectories`).

    Unlike :func:`simulate`, this applies NO gradient waveform: it stores
    ``r(t)`` for all walkers so any waveform / relaxation hypothesis can be
    applied post-hoc via
    :func:`~dmipy_sim.trajectories.replay`.  This is the
    walk-once half of the replay invariant (positions depend only on
    ``geometry, diffusivity, seed`` — see the module CLAUDE guide).

    Sub-stepping: for small geometries a coarse ``dt_save`` produces a step_l
    comparable to the geometry size, biasing reflection.  ``sub_steps =
    ceil(dt_save / dt_phys_max)`` is auto-tuned so ``step_l < R/6`` per inner
    step (``dt_phys_max = R² / (216·D)`` for reflection; ``R² / (3750·D)`` — i.e.
    ``step_l ≈ R/25`` — when the geometry is permeable, since membrane crossing
    is step-size sensitive).  Only the position after each group of ``sub_steps``
    inner steps is saved, so storage is always ``(n_walkers, n_t, 3)`` at
    ``dt_save`` granularity.

    Parameters
    ----------
    n_walkers : int
        Total number of walkers.
    diffusivity : float
        Diffusion coefficient in m²/s.
    geometry : Geometry
        Boundary geometry.  Provides ``init_positions(n, key)`` and
        ``reflect(r, step)``; for ``save_relaxation_data`` also
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
    save_relaxation_data : bool
        If True, also save per-step boundary log-weight (with rho/D=1) and a
        per-step compartment channel for each walker — enabling replay of
        surface relaxivity (rho) and per-compartment T2/T1 without re-simulating.
        Requires the geometry to have ``reflect_with_log_weight`` (Cylinder,
        Sphere, Box1D, Ellipsoid, PackedCylinders/Spheres, Mesh).  For
        FreeDiffusion (no boundaries), ``dlog_boundary_unit`` is all zeros.
    require_gpu : {None, True, False}
        GPU guard against a silent CPU fallback.  ``True`` raises if no GPU is
        visible; ``False`` opts out; ``None`` (default) warns for large CPU runs.
    r0 : array-like of shape (n_walkers, 3), optional
        Caller-supplied seed positions (e.g. extra-axonal water outside a mesh).
        When provided, ``geometry.init_positions()`` is skipped.
    kappa_MT : float
        Magnetization-transfer surface reactivity (m/s) for the PackedMyelinated-
        Cylinders + ``save_relaxation_data`` path.  ``0`` (default) leaves the walk
        byte-for-byte identical to the pre-MT path (RNG stream + positions unchanged);
        ``> 0`` binds free water at the myelin walls and returns a 7th ``bound_frac``
        channel.  (For analytic geometries use :func:`dmipy_sim.simulate_mt_trajectories`.)
    dwell_time : float
        Mean bound-pool residence time (s); must be ``> 0`` when ``kappa_MT > 0``.
    equilibrate_binding : {'auto', 'burnin', 'off'}
        How the bound pool reaches thermal-equilibrium occupancy before t=0 (MT only);
        see :func:`dmipy_sim.mt.resolve_equilibrate_mode`.

    Returns
    -------
    trajectories : np.ndarray, shape (n_walkers, n_t, 3), float16
        Walker positions in metres at each saved time step.
    dt_actual : float
        Saved time step (= T_max / (n_t - 1)).
    sub_steps : int
        Number of physics sub-steps per saved point.
    dt_sim : float
        Actual simulation time step (= dt_actual / sub_steps).
    dlog_boundary_unit : np.ndarray, shape (n_walkers, n_t), float16
        Only when ``save_relaxation_data=True``.  Per-step accumulated boundary
        log-weight assuming rho/D = 1, i.e.
        ``dlog_boundary_unit[w, t] = -2 * sum_k(d_perp_k)`` over the boundary
        hits in the ``sub_steps`` inner steps of saved step t.  Non-positive.
        Replay surface relaxivity rho via
        ``log_w[m,w] += (rho/D) * sum_t(chi_perp[m,t] * dlog_boundary_unit[w,t])``.
    comp_traj : np.ndarray, shape (n_walkers, n_t)
        Only when ``save_relaxation_data=True``.  For PERMEABLE 2-compartment
        geometries (Sphere/Cylinder with permeability): float16 FRACTIONAL
        OCCUPANCY of compartment 1 (outside) over each saved interval — the mean
        of the sub-step compartment ids (resolves intra-save membrane crossings).
        For all OTHER geometries (impermeable, surface relaxivity): int8 discrete
        compartment ID (always 0 for single-compartment; 0/1/2 for packed
        myelin).  Consumed by ``replay`` with
        ``T2_per_comp``/``T1_per_comp``.
    bound_frac : np.ndarray, shape (n_walkers, n_t), float16
        ONLY for the packed-myelin path with ``kappa_MT > 0`` — appended as a 7th
        return value.  Per-save MT bound-pool occupancy, consumed by
        ``replay_bloch(bound_frac=...)`` to blend the bound pool.
    """
    # GPU guard — never silently fall back to CPU for a heavy walk (CLAUDE rule).
    from .gpu import check_gpu
    check_gpu(n_walkers, require_gpu, what="simulate_trajectories")

    n_t = int(round(T_max / dt_save)) + 1
    dt_actual = T_max / (n_t - 1)

    # --- Sub-stepping: guarantee step_l < R/6 for accurate reflection ---
    R_geom = getattr(geometry, 'radius', None)
    if R_geom is None:
        R_geom = getattr(geometry, 'sphere_radius', None)
    if R_geom is None:
        # Box1D (slab) confines over its width -> use `length` as the sub-step scale;
        # without this the auto-tune fell through to sub_steps=1 (step_l = sqrt(6 D dt_save),
        # far coarser than a small slab), garbling the recorded boundary local time at small R.
        R_geom = getattr(geometry, 'length', None)
    if R_geom is None:
        _radii = getattr(geometry, '_radii_np', None)
        if _radii is not None and len(_radii) > 0:
            R_geom = float(np.min(_radii))
    if R_geom is None:
        _inner_radii = getattr(geometry, '_inner_radii_np', None)
        if _inner_radii is not None and len(_inner_radii) > 0:
            R_geom = (float(np.min(_inner_radii[_inner_radii > 0]))
                      if np.any(_inner_radii > 0) else None)
    if R_geom is None:
        # Free diffusion or unknown: no sub-stepping needed
        sub_steps = 1
    else:
        # Impermeable reflection is exact at any step, so step_l = R/6 suffices
        # (divisor 216 = 6·6²).  Membrane crossing is step-size sensitive
        # (over-permeates at coarse steps), so use a finer step_l ≈ R/25 when
        # permeability is active (divisor 3750 = 6·25²).
        _has_perm = getattr(geometry, 'permeability', None) is not None
        _divisor = 3750.0 if _has_perm else 216.0
        dt_phys_max = float(R_geom) ** 2 / (_divisor * diffusivity)
        sub_steps = max(1, int(np.ceil(dt_actual / dt_phys_max)))

    dt_sim = dt_actual / sub_steps
    step_l_sim = jnp.float32(jnp.sqrt(6.0 * diffusivity * dt_sim))

    print(f"  sub_steps={sub_steps}, dt_sim={dt_sim*1e6:.3f} µs, "
          f"step_l={float(step_l_sim)*1e6:.4f} µm"
          + (f", step_l/R={float(step_l_sim)/float(R_geom):.4f}" if R_geom else ""),
          flush=True)

    permeability = getattr(geometry, 'permeability', None)
    has_permeability = permeability is not None
    has_reflect_with_log_weight = hasattr(geometry, 'reflect_with_log_weight')

    # ── Standard path (position-only) ─────────────────────────────────────────
    if has_permeability:
        kappa_over_D = jnp.float32(float(permeability) / diffusivity)
        permeate = geometry.permeate

        def inner_step(carry, _):
            r, key = carry
            key, step_key, perm_key = jax.random.split(key, 3)
            noise = jax.random.normal(step_key, (3,), dtype=jnp.float32)
            unit_noise = noise / jnp.linalg.norm(noise)
            step = unit_noise * step_l_sim
            r_new, _dlog_w = permeate(r, step, kappa_over_D, jnp.float32(0.0), perm_key)
            return (r_new, key), None
    else:
        reflect = geometry.reflect

        def inner_step(carry, _):
            r, key = carry
            key, subkey = jax.random.split(key)
            noise = jax.random.normal(subkey, (3,), dtype=jnp.float32)
            unit_noise = noise / jnp.linalg.norm(noise)
            step = unit_noise * step_l_sim
            r_new = reflect(r, step)
            return (r_new, key), None

    def outer_step(carry, _):
        carry_final, _ = jax.lax.scan(inner_step, carry, None, length=sub_steps)
        r_final = carry_final[0]
        return carry_final, r_final

    def simulate_one_walker(r0_w, key_w):
        (_, _), positions = jax.lax.scan(outer_step, (r0_w, key_w), None, length=n_t)
        return positions  # (n_t, 3)

    # ── Compartment ID detection (relaxation path only) ─────────────────────
    # Permeable Cylinder (has _R and radius): 0=inside, 1=outside.  Permeable
    # Sphere (radius, no _R): 0=inside, 1=outside.  Everything else: always 0.
    if save_relaxation_data:
        if has_permeability and hasattr(geometry, '_R') and hasattr(geometry, 'radius'):
            R_val = jnp.float32(float(geometry.radius))
            _R_jax = jnp.array(geometry._R, dtype=jnp.float32)
            _is_id_R = bool(np.allclose(np.array(geometry._R), np.eye(3)))

            def _get_comp_id(r):
                r_c = r if _is_id_R else _R_jax @ r
                return jnp.int8(jnp.where(jnp.linalg.norm(r_c[:2]) < R_val, 0, 1))
        elif has_permeability and hasattr(geometry, 'radius'):
            R_val = jnp.float32(float(geometry.radius))

            def _get_comp_id(r):
                return jnp.int8(jnp.where(jnp.linalg.norm(r) < R_val, 0, 1))
        else:
            def _get_comp_id(r):
                return jnp.int8(0)

    # ── Relaxation-data path (position + boundary log-weight with rho/D=1) ────
    is_packed_myelin_geom = getattr(geometry, '_is_packed_myelinated', False)

    if save_relaxation_data and is_packed_myelin_geom:
        # PackedMyelinatedCylinders: use the stripped trajectory step fn (geometry
        # + permeability only, rho/D=1 at all walls).  comp_id is the encoded id
        # (0=extra, 1..N_max=intra, >N_max=myelin); compress to 0/1/2 at save.
        N_max_pm = geometry.N_max
        # Magnetization transfer (kappa_MT > 0): the step fn binds free water at the
        # myelin walls and records the per-save bound occupancy.  kappa_MT == 0 keeps
        # the pre-MT walk bit-for-bit (RNG stream + positions unchanged).
        _mt_on_pm = kappa_MT > 0.0
        step_fn_traj_pm = make_packed_myelin_traj_step_fn(
            geometry, dt_sim, kappa_MT=kappa_MT, dwell_time=dwell_time)

        def _compress_comp_pm(comp_id):
            return jnp.where(comp_id == jnp.int32(0), jnp.int8(0),
                   jnp.where(comp_id <= jnp.int32(N_max_pm), jnp.int8(1),
                             jnp.int8(2)))

        def _inner_pm(carry, _):
            return step_fn_traj_pm(carry, None)

        if not _mt_on_pm:
            # ── pre-MT path (4-element carry) — UNCHANGED, kept bit-for-bit ──
            def outer_step_pm(carry, _):
                r, key, comp_id = carry
                # dlog_accum resets each save so the emitted value is the per-save delta.
                inner_init = (r, key, jnp.float32(0.0), comp_id)
                (r_final, key_final, dlog_accum, comp_final), _ = jax.lax.scan(
                    _inner_pm, inner_init, None, length=sub_steps)
                return (r_final, key_final, comp_final), \
                       (r_final, dlog_accum, _compress_comp_pm(comp_final))

            def simulate_one_walker_pm(r0_w, key_w, comp0_w, brem0_w):  # brem0 unused
                (_, _, _), (positions, dlog_boundary, comp_types) = jax.lax.scan(
                    outer_step_pm, (r0_w, key_w, comp0_w), None, length=n_t)
                z = jnp.zeros_like(dlog_boundary)                       # placeholder bound_frac
                return positions, dlog_boundary, comp_types, z
        else:
            # ── MT path (6-element carry): bound_rem persists across saves ──
            def outer_step_pm(carry, _):
                r, key, comp_id, bound_rem = carry
                inner_init = (r, key, jnp.float32(0.0), comp_id, bound_rem, jnp.float32(0.0))
                (r_final, key_final, dlog_accum, comp_final, bound_rem_f, bound_acc), _ = \
                    jax.lax.scan(_inner_pm, inner_init, None, length=sub_steps)
                bound_frac = bound_acc / jnp.float32(sub_steps)
                return (r_final, key_final, comp_final, bound_rem_f), \
                       (r_final, dlog_accum, _compress_comp_pm(comp_final), bound_frac)

            def simulate_one_walker_pm(r0_w, key_w, comp0_w, brem0_w):
                (_, _, _, _), (positions, dlog_boundary, comp_types, bound_frac) = \
                    jax.lax.scan(outer_step_pm, (r0_w, key_w, comp0_w, brem0_w),
                                 None, length=n_t)
                return positions, dlog_boundary, comp_types, bound_frac

        simulate_batch_pm = jax.jit(
            jax.vmap(simulate_one_walker_pm, in_axes=(0, 0, 0, 0)))

    if save_relaxation_data and not is_packed_myelin_geom:
        if has_permeability:
            kappa_over_D_relax = jnp.float32(float(permeability) / diffusivity)
            permeate_relax = geometry.permeate

            def inner_step_relax(carry, _):
                r, key, dlog_accum, comp_sum = carry
                key, step_key, perm_key = jax.random.split(key, 3)
                noise = jax.random.normal(step_key, (3,), dtype=jnp.float32)
                unit_noise = noise / jnp.linalg.norm(noise)
                step = unit_noise * step_l_sim
                r_new, dlog_w_unit = permeate_relax(
                    r, step, kappa_over_D_relax, jnp.float32(1.0), perm_key)
                # Per-sub-step compartment id -> fractional occupancy (resolves
                # intra-save crossings without a finer dt_save).
                comp_sum = comp_sum + _get_comp_id(r_new).astype(jnp.float32)
                return (r_new, key, dlog_accum + dlog_w_unit, comp_sum), None

        elif has_reflect_with_log_weight:
            reflect_with_log_weight = geometry.reflect_with_log_weight

            def inner_step_relax(carry, _):
                r, key, dlog_accum, comp_sum = carry
                key, subkey = jax.random.split(key)
                noise = jax.random.normal(subkey, (3,), dtype=jnp.float32)
                unit_noise = noise / jnp.linalg.norm(noise)
                step = unit_noise * step_l_sim
                r_new, dlog_w_unit = reflect_with_log_weight(r, step, jnp.float32(1.0))
                comp_sum = comp_sum + _get_comp_id(r_new).astype(jnp.float32)
                return (r_new, key, dlog_accum + dlog_w_unit, comp_sum), None

        else:
            # FreeDiffusion: no boundaries → dlog_boundary_unit is always 0.
            reflect_free = geometry.reflect

            def inner_step_relax(carry, _):
                r, key, dlog_accum, comp_sum = carry
                key, subkey = jax.random.split(key)
                noise = jax.random.normal(subkey, (3,), dtype=jnp.float32)
                unit_noise = noise / jnp.linalg.norm(noise)
                step = unit_noise * step_l_sim
                r_new = reflect_free(r, step)
                comp_sum = comp_sum + _get_comp_id(r_new).astype(jnp.float32)
                return (r_new, key, dlog_accum, comp_sum), None

        def outer_step_relax(carry, _):
            r, key = carry
            inner_init = (r, key, jnp.float32(0.0), jnp.float32(0.0))
            (r_final, key_final, dlog_accum, comp_sum), _ = jax.lax.scan(
                inner_step_relax, inner_init, None, length=sub_steps)
            # Fractional occupancy of compartment 1 over the saved interval.  For
            # single-compartment geometries this is identically 0.
            comp_occ = comp_sum / jnp.float32(sub_steps)
            return (r_final, key_final), (r_final, dlog_accum, comp_occ)

        def simulate_one_walker_relax(r0_w, key_w):
            (_, _), (positions, dlog_boundary, comp_ids) = jax.lax.scan(
                outer_step_relax, (r0_w, key_w), None, length=n_t)
            return positions, dlog_boundary, comp_ids

        simulate_batch_relax = jax.jit(
            jax.vmap(simulate_one_walker_relax, in_axes=(0, 0)))

    simulate_batch = jax.jit(jax.vmap(simulate_one_walker, in_axes=(0, 0)))

    master_key = jax.random.PRNGKey(seed)
    pos_key, walker_key = jax.random.split(master_key)

    if r0 is not None:
        r0_all = jnp.asarray(r0, dtype=jnp.float32)
        if r0_all.shape != (n_walkers, 3):
            raise ValueError(f"r0 must have shape ({n_walkers}, 3), got {r0_all.shape}")
    else:
        r0_all = geometry.init_positions(n_walkers, pos_key)   # (n_walkers, 3)
    walker_keys_all = jax.random.split(walker_key, n_walkers)

    comp0_all = (jnp.asarray(geometry._init_compartments)
                 if (save_relaxation_data and is_packed_myelin_geom) else None)

    # ── MT bound-pool equilibration (packed myelin, kappa_MT > 0) ──
    # An all-free start under-fills the macromolecular pool and biases the transfer;
    # equilibrate the bound occupancy (and spatial state) to f_b BEFORE t=0 and discard
    # the preamble.  kappa_MT == 0 leaves brem0_all at zero and skips this entirely.
    _mt_on = kappa_MT > 0.0 and save_relaxation_data and is_packed_myelin_geom
    brem0_all = jnp.zeros((n_walkers,), dtype=jnp.float32)
    if _mt_on:
        from . import mt as _mt
        if dwell_time <= 0.0:
            raise ValueError("dwell_time must be > 0 when kappa_MT > 0.")
        _mode = _mt.resolve_equilibrate_mode(equilibrate_binding, geometry)
        if _mode == 'burnin':
            _n_chunk = max(4, int(round(float(dwell_time) / float(dt_sim))))

            def _burn_walker(r_w, key_w, comp_w, brem_w):
                (r_f, key_f, _da, comp_f, brem_f, bacc), _ = jax.lax.scan(
                    _inner_pm, (r_w, key_w, jnp.float32(0.0), comp_w, brem_w,
                                jnp.float32(0.0)), None, length=_n_chunk)
                return r_f, key_f, comp_f, brem_f, bacc / jnp.float32(_n_chunk)
            _burn = jax.jit(jax.vmap(_burn_walker, in_axes=(0, 0, 0, 0)))

            _r, _k, _c = r0_all, walker_keys_all, comp0_all
            _brem = brem0_all
            _occ_prev, _converged = -1.0, False
            for _ in range(40):
                _r, _k, _c, _brem, _bf = _burn(_r, _k, _c, _brem)
                _occ = float(jnp.mean(_bf))
                if _occ_prev >= 0.0 and abs(_occ - _occ_prev) <= 0.01 * max(_occ, 1e-6):
                    _occ_prev, _converged = _occ, True
                    break
                _occ_prev = _occ
            r0_all, walker_keys_all, comp0_all, brem0_all = _r, _k, _c, _brem
            print(f"  [mt] equilibrate 'burnin': <bound>={_occ_prev:.4f}", flush=True)
            if not _converged:
                import warnings
                warnings.warn("equilibrate_binding: bound occupancy did not plateau within "
                              "the burn-in cap; the saved walk may be under-equilibrated.",
                              stacklevel=2)

    all_batches = []
    all_dlog_batches = [] if save_relaxation_data else None
    all_comp_batches = [] if save_relaxation_data else None
    all_bound_batches = [] if _mt_on else None
    n_batches = (n_walkers + walker_batch_size - 1) // walker_batch_size

    # ── IR-basis streaming compression (piece 1) ────────────────────────────────
    # When compress=K, DCT each batch's positions/boundary channel ON DEVICE and pull
    # only (batch, K, 3) / (batch, K+1) to the host — the raw (batch, n_t, 3) trajectory
    # never leaves the GPU, so the producer's host-RAM footprint drops ~n_t/K. The DCT is
    # a matmul against the SAME orthonormal DCT-II basis compression.py uses (built from
    # scipy so the stored modes decode/mode-space-replay bit-consistently).
    _compress = compress is not None
    _cx = {"K": int(compress) if _compress else 0, "Phi": None, "Psi": None, "ramp": None, "n_t": None}
    all_blt_endpoints = [] if (_compress and save_relaxation_data) else None
    if _compress and is_packed_myelin_geom:
        raise NotImplementedError(
            "compress= is not yet wired for packed-myelin walks (extra MT bound channel); "
            "piece 1 covers the generic reflect/surface-relaxivity producer.")

    def _dct_basis(n_t):
        from scipy.fft import dct as _sdct
        # dct(I, axis=0)[k, n] = DCT-II coeff k of basis vector e_n  ->  Phi @ x == dct(x)
        Phi = _sdct(np.eye(n_t, dtype=np.float64), type=2, norm="ortho", axis=0)[:_cx["K"]]
        return jnp.asarray(Phi, jnp.float32)

    def _sine_basis(n_t):
        from scipy.fft import dst as _sdst
        # dst(I, axis=0)[k, n] over the INTERIOR: Psi @ u_int == dst-I(u_int)
        Psi = _sdst(np.eye(n_t - 2, dtype=np.float64), type=1, norm="ortho",
                    axis=0)[:_cx["K"]]
        return jnp.asarray(Psi, jnp.float32)

    def _compress_pos(pos_dev):                  # (b, n_t, 3) device -> (b, K+2, 3) host
        """Positions in the same representation build_replay_pack writes: two exact endpoints
        then sine bands of the pinned residual.  Producing DCT bands here instead would put a
        second, differently-meaning C0 layout into masters that the reader cannot distinguish
        by shape."""
        n_t = int(pos_dev.shape[1])
        if _cx["Psi"] is None:
            _cx["n_t"] = n_t; _cx["Psi"] = _sine_basis(n_t)
        a = pos_dev[:, 0, :]
        v = pos_dev[:, -1, :] - a
        tau = jnp.linspace(jnp.float32(0.0), jnp.float32(1.0), n_t)
        u = pos_dev - (a[:, None, :] + v[:, None, :] * tau[None, :, None])
        B = jnp.einsum("kt,btd->bkd", _cx["Psi"], u[:, 1:-1, :])
        return np.asarray(jnp.concatenate([a[:, None, :], v[:, None, :], B],
                                          axis=1)).astype(np.float32)

    def _compress_blt(dlog_dev):                     # (b, n_t) -> endpoint (b,), modes (b, K)
        n_t = int(dlog_dev.shape[1])
        if _cx["Phi"] is None:
            _cx["n_t"] = n_t; _cx["Phi"] = _dct_basis(n_t)
        if _cx["ramp"] is None:
            _cx["ramp"] = jnp.linspace(jnp.float32(0.0), jnp.float32(1.0), n_t)
        B = jnp.cumsum(dlog_dev, axis=1)
        endpoint = B[:, -1]
        resid = B - endpoint[:, None] * _cx["ramp"][None, :]      # 0 at both ends
        modes = jnp.einsum("kt,bt->bk", _cx["Phi"], resid)
        return np.asarray(endpoint).astype(np.float32), np.asarray(modes).astype(np.float32)

    for batch_idx in range(n_batches):
        start = batch_idx * walker_batch_size
        end = min(start + walker_batch_size, n_walkers)
        batch_size = end - start
        pct = int(100 * end / n_walkers)
        print(f"  Simulating walkers {start}–{end - 1} ({pct}% done)...", flush=True)

        current_r0 = r0_all[start:end]
        current_keys = walker_keys_all[start:end]
        if save_relaxation_data and is_packed_myelin_geom:
            current_comp0 = comp0_all[start:end]
            current_brem0 = brem0_all[start:end]

        success = False
        while not success:
            try:
                if save_relaxation_data and is_packed_myelin_geom:
                    pos_f32, dlog_f32, comp_f32, bfrac_f32 = simulate_batch_pm(
                        current_r0, current_keys, current_comp0, current_brem0)
                    all_batches.append(np.array(pos_f32).astype(np.float16))
                    all_dlog_batches.append(np.array(dlog_f32).astype(np.float16))
                    all_comp_batches.append(np.array(comp_f32).astype(np.int8))
                    if _mt_on:
                        all_bound_batches.append(np.array(bfrac_f32).astype(np.float16))
                elif save_relaxation_data:
                    pos_f32, dlog_f32, comp_f32 = simulate_batch_relax(current_r0, current_keys)
                    if _compress:
                        all_batches.append(_compress_pos(pos_f32))
                        _end, _bmodes = _compress_blt(dlog_f32)
                        all_blt_endpoints.append(_end)
                        all_dlog_batches.append(_bmodes)
                    else:
                        all_batches.append(np.array(pos_f32).astype(np.float16))
                        all_dlog_batches.append(np.array(dlog_f32).astype(np.float16))
                    # Permeable: fractional occupancy (float16); else discrete (int8).
                    all_comp_batches.append(np.array(comp_f32).astype(
                        np.float16 if has_permeability else np.int8))
                else:
                    if _compress:
                        all_batches.append(_compress_pos(simulate_batch(current_r0, current_keys)))
                    else:
                        positions_f32 = np.array(simulate_batch(current_r0, current_keys))
                        all_batches.append(positions_f32.astype(np.float16))
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
                    print(f"  OOM: halving sub-batch to {new_sub_batch}", flush=True)
                    sub_pos_list = []
                    sub_dlog_list = [] if save_relaxation_data else None
                    sub_comp_list = [] if save_relaxation_data else None
                    sub_bound_list = [] if _mt_on else None
                    for ss in range(0, batch_size, new_sub_batch):
                        se = min(ss + new_sub_batch, batch_size)
                        if save_relaxation_data and is_packed_myelin_geom:
                            sp, sd, sc, sbf = simulate_batch_pm(
                                current_r0[ss:se], current_keys[ss:se],
                                current_comp0[ss:se], current_brem0[ss:se])
                            sub_pos_list.append(np.array(sp).astype(np.float16))
                            sub_dlog_list.append(np.array(sd).astype(np.float16))
                            sub_comp_list.append(np.array(sc).astype(np.int8))
                            if _mt_on:
                                sub_bound_list.append(np.array(sbf).astype(np.float16))
                        elif save_relaxation_data:
                            sp, sd, sc = simulate_batch_relax(
                                current_r0[ss:se], current_keys[ss:se])
                            sub_pos_list.append(np.array(sp).astype(np.float16))
                            sub_dlog_list.append(np.array(sd).astype(np.float16))
                            sub_comp_list.append(np.array(sc).astype(
                                np.float16 if has_permeability else np.int8))
                        else:
                            sp = np.array(simulate_batch(
                                current_r0[ss:se], current_keys[ss:se]))
                            sub_pos_list.append(sp.astype(np.float16))
                    all_batches.append(np.concatenate(sub_pos_list, axis=0))
                    if save_relaxation_data:
                        all_dlog_batches.append(np.concatenate(sub_dlog_list, axis=0))
                        all_comp_batches.append(np.concatenate(sub_comp_list, axis=0))
                        if _mt_on:
                            all_bound_batches.append(np.concatenate(sub_bound_list, axis=0))
                    success = True
                else:
                    raise

    if _compress:
        # Compressed master: IR modes instead of the raw trajectory. Decode with
        # compression.decode / replay via compression.mode_space_signal (piece 2 wires
        # this into replay() directly). `pos_modes` are [r(0), r(T)-r(0), sine bands].
        master = {
            "compressed": True, "method": "bridge_dst", "K": _cx["K"],
            "n_t": int(_cx["n_t"]), "dt_traj": dt_actual,
            "sub_steps": sub_steps, "dt_sim": dt_sim,
            "pos_modes": np.concatenate(all_batches, axis=0),        # (N, K, 3) f32
        }
        if save_relaxation_data:
            master["blt_endpoint"] = np.concatenate(all_blt_endpoints, axis=0)  # (N,)
            master["blt_modes"] = np.concatenate(all_dlog_batches, axis=0)      # (N, K)
            master["comp_traj"] = np.concatenate(all_comp_batches, axis=0)      # (N, n_t)
        return master

    trajectories = np.concatenate(all_batches, axis=0)  # (n_walkers, n_t, 3) float16
    if save_relaxation_data:
        dlog_boundary_unit = np.concatenate(all_dlog_batches, axis=0)  # (n_walkers, n_t) float16
        comp_traj = np.concatenate(all_comp_batches, axis=0)           # (n_walkers, n_t)
        if _mt_on:
            # 7th channel: per-save MT bound-pool occupancy (packed myelin, kappa_MT>0).
            bound_frac = np.concatenate(all_bound_batches, axis=0)     # (n_walkers, n_t) float16
            return (trajectories, dt_actual, sub_steps, dt_sim,
                    dlog_boundary_unit, comp_traj, bound_frac)
        return trajectories, dt_actual, sub_steps, dt_sim, dlog_boundary_unit, comp_traj
    return trajectories, dt_actual, sub_steps, dt_sim
