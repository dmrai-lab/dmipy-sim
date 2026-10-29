"""The fused forward engine: the oracle.

One ``jax.lax.scan`` per walker integrates the gradient phase, the relaxation log-weight and the pool label along
the walk and returns the ensemble signal. It is the single-pass physics every coefficient-form route is admitted
against -- the parity admissions, the step-law certificate, the tier certificates -- and the route of the runs a
coefficient walk cannot serve (:func:`~dmipy_sim.engine.core._replay_gap`). Nothing exports it:
:func:`~dmipy_sim.engine.core.simulate` is the public entry.
"""

import jax
import jax.numpy as jnp
import numpy as np

from dmipy_replay.acquisition.waveforms import band_hz
from ..run import Run, current
from .core import cached_batch
from .physics import (make_step_fn, make_myelin_step_fn, make_packed_myelin_step_fn, seed_walkers)


def _ensemble_signal(spin_w, phi, log_w=None):
    """Spin-density-weighted ensemble signal ``Re<w exp(log_w) e^{i phi}> / sum w``.

    ``spin_w`` is the per-walker proton-density weight (myelin water < 1); homogeneous placement plus
    this weight avoids per-geometry placement re-weighting. ``log_w`` (per walker) is the relaxation and
    surface log-weight; ``None`` when nothing wrote it.
    """
    amp = spin_w if log_w is None else spin_w * jnp.exp(log_w)
    return jnp.sum(amp[:, None] * jnp.cos(phi), axis=0) / jnp.sum(spin_w)


def simulate_fused(
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
    sub_steps: int = None,
    _allow_oom_backoff: bool = True,
):
    """The fused forward signal of ``waveform`` on ``geometry``: the oracle.

    Parameters
    ----------
    n_walkers : int
        Number of random walkers.
    diffusivity : float, optional
        Diffusion coefficient in m²/s. Required for standard geometries.
        Omit for MyelinatedCylinder (D values are in the geometry).
    waveform : ScannerSequence
        One sequence (a multi-TE :class:`~dmipy_sim.acquisition.scanner_sequence.Protocol` is the public
        entry's). Its effective gradient has shape (n_measurements, n_t, 3).
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
    with Run("simulate_fused", params=dict(n_walkers=n_walkers, diffusivity=diffusivity, geometry=type(geometry).__name__)):
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
                    return simulate_fused(
                        n_walkers, diffusivity=diffusivity, waveform=waveform,
                        geometry=geometry, seed=seed, T2=T2, T1=T1,
                        return_positions=return_positions,
                        return_compartments=return_compartments,
                        return_walker_signals=return_walker_signals, r0=r0,
                        walker_batch_size=bs, require_gpu=require_gpu,
                        sub_steps=sub_steps, _allow_oom_backoff=False)
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
                        "simulate_fused() hit GPU OOM at walker_batch_size={}; retrying at "
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
    """Run simulate_fused() over walker chunks and recombine (see simulate_fused's
    ``walker_batch_size``).  The signal is a plain walker-mean, so it recombines
    as a size-weighted mean; per-walker outputs (positions, compartments, walker
    signals) are concatenated.  Each chunk's device buffers are released when its
    (host) results are returned, so peak device memory is one chunk."""
    track_comp = return_compartments is not False
    sig_acc = None
    pos_l, origin_l, comp_l, lw_l, phi_l = [], [], [], [], []

    for b, (start, end) in enumerate(current().batches(n_walkers, walker_batch_size, what="simulate")):
        nb = end - start
        out = simulate_fused(
            n_walkers=nb, diffusivity=diffusivity, waveform=waveform,
            geometry=geometry, seed=seed + 1 + b, T2=T2, T1=T1,
            r0=(None if r0 is None else r0[start:end]),
            return_positions=return_positions,
            return_compartments=return_compartments,
            return_walker_signals=return_walker_signals,
            walker_batch_size=None, require_gpu=False,
            sub_steps=sub_steps, _allow_oom_backoff=False)

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
