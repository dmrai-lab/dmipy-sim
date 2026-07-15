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

from .physics import (make_step_fn, make_myelin_step_fn, make_packed_myelin_step_fn)
from .waveforms import Waveform


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
    want_pos_full = return_positions == 'full'

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
                    _allow_oom_backoff=False)
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
            comp_origin_jax = jax.vmap(classify_fn)(r0)  # (n_walkers,) int32

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
            # Need a classify_position closure for the scan body
            classify_fn = geometry.classify_position

            if has_weight:
                # carry = (r, phi, log_weight, compartment_current, key)
                def step_fn_comp(carry, inputs):
                    r, phi, log_weight, comp_cur, key = carry
                    # Run the original step_fn with its expected carry format
                    orig_carry = (r, phi, log_weight, key)
                    (r_new, phi_new, log_new, key_new), _ = step_fn(orig_carry, inputs)
                    comp_new = classify_fn(r_new)
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
                    comp_new = classify_fn(r_new)
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
            _allow_oom_backoff=False)

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
    ordinary forward model — not trajectory replay: one walk, no saved trajectories,
    no coherence-pathway bank.  Build ``waveform`` with :func:`dmipy_sim.cpmg`
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
