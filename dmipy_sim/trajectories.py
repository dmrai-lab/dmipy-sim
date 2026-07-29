"""Waveform application to pre-computed walker trajectories (the replay path).

Decouples MC geometry simulation from gradient encoding + relaxation, enabling
fast re-use of a single walker library for arbitrary waveforms, T2/T1 schedules
and surface relaxivity.

The replay invariant (see ``core.simulate_trajectories``): walker positions
``r(t)`` and boundary events depend ONLY on ``(geometry, diffusivity, seed)`` —
never on ``G(t)``, T2, T1 or ρ.  Those gate ``log_w``/phase only, so
``simulate_trajectories(..., save_relaxation_data=True)`` walks once and the
``apply_waveform_*`` functions below replay many acquisition/relaxation
hypotheses off that one walk.
"""

import numpy as np

from .constants import GAMMA

# JAX optional — dmipy-sim does not hard-require JAX for the NumPy replay path.
try:
    import jax
    import jax.numpy as jnp
    _JAX_AVAILABLE = True
except ImportError:
    _JAX_AVAILABLE = False

_GAMMA_JAX = float(GAMMA)  # rad/(s·T) — exactly GAMMA, so the JAX and NumPy
                           # replay phases can never drift from a stale literal.


# ── susceptibility off-resonance: sample a provider's field along the walk ───────
def _resolve_field_fn(susceptibility):
    """Return the pure-JAX ``r -> ΔBz`` callable of a susceptibility provider.

    Accepts a :mod:`dmipy_sim.susceptibility` provider (exposes ``delta_bz_fn()``)
    or a bare ``r -> ΔBz`` callable.  ``None`` returns ``None``.
    """
    if susceptibility is None:
        return None
    if hasattr(susceptibility, "delta_bz_fn"):
        return susceptibility.delta_bz_fn()
    return susceptibility


def _sample_delta_bz(field_fn, trajectories):
    """Sample ``ΔBz`` (T) along every stored position: ``(n_w, n_t, 3) -> (n_w, n_t)``.

    ``field_fn`` is a pure-JAX ``r(3,) -> ΔBz`` callable; the whole trajectory is
    evaluated with a doubly-vmapped map.  This is the geometry-agnostic public
    susceptibility-replay primitive — replay any B0 / orientation / χ by re-evaluating
    a different provider on the SAME walk (no re-simulation).
    """
    if not _JAX_AVAILABLE:
        raise ImportError("JAX is required to sample a susceptibility field provider.")
    tj = jnp.asarray(np.asarray(trajectories, dtype=np.float32))
    samp = jax.vmap(jax.vmap(field_fn))          # over walkers, then time
    return np.asarray(samp(tj), dtype=np.float64)


def pathway_sign_se(n_t, dt, TE, te_frac=0.5):
    """Refocusing pathway sign ``ε_P(t) ∈ {+1, −1}`` for a spin echo (flip at ``TE/2``).

    Susceptibility (and any static off-resonance) that accrues before the 180° pulse is
    conjugated after it, so in the SCALAR replay path
    (:func:`apply_waveform_with_relaxation`) the static phase must be summed with a sign
    that flips at the pulse: ``ε_P = +1`` for ``t < te_frac·TE`` and ``−1`` after.  (The
    vector-Bloch replay :func:`apply_waveform_bloch` needs no ``ε_P`` — its explicit 180°
    rotation conjugates the phase emergently.)  For PGSTE, zero ``ε_P`` across the mixing
    time (stored transverse phase is parked on ``Mz``); build that mask directly.

    Parameters
    ----------
    n_t : int      number of samples on the grid.
    dt : float     grid time step (s).
    TE : float     echo time (s); the 180° fires at ``te_frac·TE``.
    te_frac : float  fractional pulse position (0.5 = mid-TE spin echo).
    """
    t = np.arange(int(n_t), dtype=np.float64) * float(dt)
    return np.where(t < float(te_frac) * float(TE), 1.0, -1.0)


def unwrap_periodic(traj, cell_size, periodic_axes=(0, 1)):
    """Reconstruct CONTINUOUS positions from periodic-wrapped trajectory positions.

    Packed periodic substrates (PackedCylinders/Spheres/MyelinatedCylinders) store
    walker positions folded into the cell ``[-L/2, L/2)`` along the periodic axes.
    The gradient phase ``gamma * integral G.r dt`` must use the CONTINUOUS lab-frame
    position, else every periodic boundary crossing injects a spurious ``~q*L``
    phase into the (refocused) encoding -- artificially attenuating the
    diffusion-weighted signal, worse in smaller cells.

    This undoes the wrapping along ``periodic_axes`` by detecting per-save jumps
    larger than ``L/2`` (a wrap) and re-integrating the minimal-image displacement
    from t=0.  Non-periodic axes (e.g. the fibre axis z of a cylinder pack) are
    left untouched.

    VALIDITY: assumes the true per-save displacement is < L/2 on the periodic axes
    (so a jump > L/2 is unambiguously a wrap).  This holds whenever the save step
    is fine enough that ``sqrt(2 D dt_save) < L/2`` -- the normal case.  A warning
    is emitted if a periodic axis shows displacements suspiciously close to L/2.

    Parameters
    ----------
    traj : np.ndarray, shape (..., n_t, 3)
        Wrapped positions (any leading batch dims; time is axis -2).
    cell_size : float
        Periodic cell side length L (m).
    periodic_axes : tuple of int
        Spatial axes (0=x,1=y,2=z) that are periodic.  Cylinders/myelin: (0,1);
        spheres: (0,1,2).

    Returns
    -------
    np.ndarray, float32, same shape as ``traj`` — continuous positions.
    """
    out = np.asarray(traj, dtype=np.float32).copy()
    L = np.float32(cell_size)
    for ax in periodic_axes:
        x = out[..., ax]                              # (..., n_t)
        dx = np.diff(x, axis=-1)
        dx -= L * np.round(dx / L)                    # minimal-image displacement
        if np.any(np.abs(dx) > 0.45 * float(L)):
            import warnings
            warnings.warn(
                "unwrap_periodic: per-save displacement approaches L/2 on a "
                "periodic axis; the trajectory save step may be too coarse for "
                "unambiguous unwrapping. Use a finer dt_save.", RuntimeWarning)
        x_cont = np.empty_like(x)
        x_cont[..., 0] = x[..., 0]
        x_cont[..., 1:] = x[..., :1] + np.cumsum(dx, axis=-1)
        out[..., ax] = x_cont
    return out


def apply_waveform_jax(
    G,
    traj,
    dt_traj: float,
    dt_wf=None,
    walker_weights=None,
    stimulated_echo: bool = False,
):
    """JAX-differentiable waveform replay on stored walker trajectories.

    Computes the diffusion-weighted signal by accumulating the gradient-
    position dot product along each walker's trajectory using JAX operations:

        φ_m,w = γ · dt_traj · Σ_t  G_resampled[m,t,:] · traj[w,t,:]
        E[m]  = mean_w( cos(φ_m,w) )   or weighted mean if walker_weights given

    Fully differentiable with respect to ``G`` via ``jax.grad``.

    Parameters
    ----------
    G : jnp.ndarray, shape (n_meas, n_t_wf, 3)
        Gradient waveform array in T/m.  JAX-traced (differentiable).
    traj : jnp.ndarray, shape (n_walkers, n_t_traj, 3)
        Walker positions in metres.  Treated as a constant (not differentiated).
        float16 or float32; cast to float32 internally.
    dt_traj : float
        Trajectory time step in seconds.
    dt_wf : float or None
        Waveform time step in seconds.  If None, assumed equal to ``dt_traj``
        (no resampling).
    walker_weights : jnp.ndarray, shape (n_walkers,) or None
        Optional per-walker importance weights.  If None, uniform mean is used.
        Weights are normalised to sum to 1 internally.
    stimulated_echo : bool
        If True, multiply the returned signal by 0.5 to account for the
        cos(phi1) storage step in PGSTE sequences.  Pass
        ``stimulated_echo=wf.stimulated_echo`` when replaying a Waveform
        built by :func:`pgste`.

    Returns
    -------
    signals : jnp.ndarray, shape (n_meas,)
        Signal attenuation E = <cos(φ)> in [0, 1], differentiable w.r.t. G.
        Scaled by 0.5 when ``stimulated_echo=True``.

    Notes
    -----
    When ``dt_wf != dt_traj``, G is resampled to the trajectory time grid
    using linear interpolation (``jnp.interp`` vmapped over measurements and
    axes) — this is JAX-differentiable.
    """
    if not _JAX_AVAILABLE:
        raise ImportError(
            "JAX is required for apply_waveform_jax. "
            "Install jax or use apply_waveform_to_trajectories for NumPy."
        )

    n_meas, n_t_wf, _ = G.shape
    n_walkers, n_t_traj, _ = traj.shape

    # --- Resample G to trajectory time grid if time steps differ ---
    if dt_wf is None or abs(dt_wf - dt_traj) / max(abs(dt_traj), 1e-30) < 1e-9:
        # Same time step: align by length
        if n_t_wf == n_t_traj:
            G_r = G.astype(jnp.float32)
        elif n_t_wf > n_t_traj:
            G_r = G[:, :n_t_traj, :].astype(jnp.float32)
        else:
            pad = jnp.zeros((n_meas, n_t_traj - n_t_wf, 3), dtype=jnp.float32)
            G_r = jnp.concatenate([G.astype(jnp.float32), pad], axis=1)
    else:
        # Linear interpolation of G to trajectory time grid
        t_wf   = jnp.arange(n_t_wf,   dtype=jnp.float32) * float(dt_wf)
        t_traj = jnp.arange(n_t_traj, dtype=jnp.float32) * float(dt_traj)

        # vmap over measurements (axis 0 of G) and axes (axis 2)
        def interp_one(g_1d):
            # left/right=0: no gradient outside the waveform window.
            return jnp.interp(t_traj, t_wf, g_1d, left=0.0, right=0.0)

        interp_ax   = jax.vmap(interp_one)          # over 3 axes
        interp_meas = jax.vmap(interp_ax)           # over n_meas

        G_t   = G.astype(jnp.float32).transpose(0, 2, 1)   # (n_meas, 3, n_t_wf)
        G_r_t = interp_meas(G_t)                            # (n_meas, 3, n_t_traj)
        G_r   = G_r_t.transpose(0, 2, 1)                   # (n_meas, n_t_traj, 3)

    # --- Phase accumulation: phi[m, w] ---
    traj_f32 = traj.astype(jnp.float32)
    phi = _GAMMA_JAX * float(dt_traj) * jnp.einsum('mtx,wtx->mw', G_r, traj_f32)

    # --- Weighted average ---
    if walker_weights is None:
        signals = jnp.mean(jnp.cos(phi), axis=1)
    else:
        w_norm = walker_weights / (jnp.sum(walker_weights) + 1e-30)
        signals = jnp.einsum('mw,w->m', jnp.cos(phi), w_norm)

    if stimulated_echo:
        signals = signals * jnp.float32(0.5)
    return signals


def apply_waveform_to_trajectories(
    trajectories,
    dt_traj,
    G,
    dt_wf,
    echo_idx=None,
    stimulated_echo: bool = False,
):
    """Apply a gradient waveform to saved walker trajectories.

    Computes the diffusion-weighted signal by integrating the gradient-
    position dot product along each walker's trajectory:

        φ_walker = γ Σ_t G(t) · r_walker(t) · dt_traj

        E = mean_walkers(cos(φ))

    The gradient waveform G is resampled to the trajectory time grid if
    dt_wf != dt_traj. The echo_idx marks the 180° RF pulse position:
    phase accumulated before echo_idx is negated (spin echo refocusing).
    Conventional waveforms (PGSE, OGSE) already encode the sign flip in G
    directly (second lobe is negative), so echo_idx is only used for
    FWF-style waveforms where the AB file concatenates two positive-lobe
    segments around a zero-gap.

    Parameters
    ----------
    trajectories : np.ndarray, shape (n_walkers, n_t_traj, 3)
        Walker positions in metres. float16 or float32.
    dt_traj : float
        Trajectory time step in seconds.
    G : np.ndarray, shape (n_meas, n_t_wf, 3)
        Gradient waveform in T/m.
    dt_wf : float
        Waveform time step in seconds.
    echo_idx : int or None
        Time index in the WAVEFORM grid where the 180° RF pulse fires.
        If None, no sign flip is applied (waveform already handles sign).
    stimulated_echo : bool
        If True, multiply the returned signal by 0.5 to account for the
        cos(phi1) storage step in PGSTE sequences.

    Returns
    -------
    signals : np.ndarray, shape (n_meas,), float64
        Signal attenuation E = <cos(φ)> in [0, 1].
        Scaled by 0.5 when ``stimulated_echo=True``.
    """
    G = np.asarray(G, dtype=np.float32)
    n_meas, n_t_wf, _ = G.shape
    n_walkers, n_t_traj, _ = trajectories.shape

    # --- Step 1: resample G to trajectory time grid if needed ---
    rel_dt_diff = abs(dt_wf - dt_traj) / max(abs(dt_traj), 1e-30)
    if rel_dt_diff > 1e-9:
        t_wf = np.arange(n_t_wf, dtype=np.float64) * dt_wf
        t_traj = np.arange(n_t_traj, dtype=np.float64) * dt_traj
        G_traj = np.zeros((n_meas, n_t_traj, 3), dtype=np.float32)
        for m in range(n_meas):
            for ax in range(3):
                # left/right=0: no gradient outside the waveform window.  Without
                # this, np.interp constant-extrapolates G[-1] across any trajectory
                # tail past the waveform (e.g. TE > Delta+delta), injecting a
                # spurious constant gradient if the waveform does not zero-terminate.
                G_traj[m, :, ax] = np.interp(t_traj, t_wf, G[m, :, ax],
                                             left=0.0, right=0.0)
    else:
        # Time grids match: align by length
        if n_t_wf == n_t_traj:
            G_traj = G
        elif n_t_wf > n_t_traj:
            G_traj = G[:, :n_t_traj, :]
        else:
            # Pad with zeros
            G_traj = np.zeros((n_meas, n_t_traj, 3), dtype=np.float32)
            G_traj[:, :n_t_wf, :] = G

    # --- Step 2: apply echo sign flip if requested ---
    if echo_idx is not None:
        echo_idx_traj = int(round(echo_idx * dt_wf / dt_traj))
        G_traj = G_traj.copy()
        G_traj[:, echo_idx_traj:, :] *= -1.0

    # --- Step 3: phase accumulation ---
    # phi[m, w] = GAMMA * dt_traj * sum_t G_traj[m,t,:] · traj[w,t,:]
    mem_bytes = n_walkers * n_t_traj * 3 * 4  # float32
    chunk_size = 100_000  # walkers per chunk

    G_flat = G_traj.reshape(n_meas, n_t_traj * 3).astype(np.float64)

    if mem_bytes <= 8 * 1024**3:
        # Fits in ~8 GB — one shot
        traj_f32 = trajectories.reshape(n_walkers, n_t_traj * 3).astype(np.float64)
        phi = GAMMA * dt_traj * (G_flat @ traj_f32.T)  # (n_meas, n_walkers)
        signals = np.mean(np.cos(phi), axis=1)
    else:
        # Chunked over walkers
        cos_sum = np.zeros(n_meas, dtype=np.float64)
        for w_start in range(0, n_walkers, chunk_size):
            w_end = min(w_start + chunk_size, n_walkers)
            chunk = trajectories[w_start:w_end].reshape(
                w_end - w_start, n_t_traj * 3
            ).astype(np.float64)
            phi_chunk = GAMMA * dt_traj * (G_flat @ chunk.T)  # (n_meas, chunk)
            cos_sum += np.sum(np.cos(phi_chunk), axis=1)
        signals = cos_sum / n_walkers

    if stimulated_echo:
        signals = signals * 0.5
    return signals


def apply_waveform_with_relaxation(
    trajectories,
    dt_traj: float,
    G,
    dt_wf: float,
    chi_perp,
    dlog_boundary_unit=None,
    T2=None,
    T1=None,
    rho=None,
    D=None,
    stimulated_echo: bool = False,
    comp_traj=None,
    T2_per_comp=None,
    T1_per_comp=None,
    return_walker_signals: bool = False,
    susceptibility=None,
    eps_P=None,
):
    """Apply a gradient waveform and relaxation weights to saved walker trajectories.

    Extends apply_waveform_to_trajectories() with chi_perp-gated T2/T1 decay
    and surface relaxivity (rho) replay over pre-saved boundary hit data.

    Susceptibility off-resonance (opt-in) is replayed by sampling a public
    :mod:`dmipy_sim.susceptibility` provider's ``delta_bz_fn(r)`` along the stored
    positions and adding ``γ·dt·Σ_t ε_P(t)·ΔBz(r(t))`` to the gradient phase.  The
    provider bakes in B0 / fibre orientation / χ, so one walk replays any field by
    re-evaluating a different provider (no re-simulation).  ``eps_P`` is the pathway
    sign for spin-echo refocusing (``+1`` before the 180° at ``TE/2``, ``−1`` after —
    see :func:`pathway_sign_se`); pass ``eps_P=None`` for a gradient-echo / FID (no
    sign flip) and ``eps_P`` zeroed across the mixing time for PGSTE storage.

    NOTE (public vs private): the private engine also carried a packed-myelin phasor
    fast-path (precomputed Φ_C/Φ_S/Φ_0 field maps + scalar Δχ_a/B0/θ/α); the public
    path is provider-driven (``delta_bz_fn``) and geometry-agnostic instead.  The
    phasor fast-path is a later addition and is intentionally omitted here (no public
    geometry exposes precomputed field maps yet).

    Physics
    -------
    For each measurement m and walker w:

        phi[m,w]   = γ · dt_traj · Σ_t  G_r[m,t,:] · r[w,t,:]
        log_w[m,w] = -Σ_t chi[m,t] · dt_traj / T2                  (T2 term)
                   - Σ_t (1 - chi[m,t]) · dt_traj / T1             (T1 term)
                   + (rho/D) · Σ_t chi[m,t] · dlog_bnd_unit[w,t]   (rho term)
        E[m] = mean_w( exp(log_w[m,w]) · cos(phi[m,w]) )

    The T2 and T1 terms are walker-independent for single-compartment geometries
    and are computed as scalar multipliers per measurement.  With
    ``T2_per_comp``/``T1_per_comp`` they become walker-dependent (via
    ``comp_traj``).  The rho term is walker-dependent and requires
    ``dlog_boundary_unit``.

    Parameters
    ----------
    trajectories : np.ndarray, shape (n_walkers, n_t_traj, 3)
        Walker positions in metres.  float16 or float32.
    dt_traj : float
        Trajectory time step in seconds.
    G : np.ndarray, shape (n_meas, n_t_wf, 3)
        Gradient waveform in T/m.
    dt_wf : float
        Waveform time step in seconds.
    chi_perp : np.ndarray, shape (n_t_wf,) or (n_meas, n_t_wf)
        Transverse gating schedule: 1 during encoding/decoding (T2 active),
        0 during mixing time (T1 active, T2 suspended).  Resampled to the
        trajectory time grid internally.  1D input is broadcast over all
        measurements; 2D input (n_meas, n_t_wf) allows per-measurement chi.
    dlog_boundary_unit : np.ndarray, shape (n_walkers, n_t_traj), or None
        Per-step accumulated boundary log-weight with rho/D = 1, as returned
        by simulate_trajectories(save_relaxation_data=True).  Required when
        rho is not None.  Values are non-positive (boundary hits reduce signal).
    T2 : float or None
        Transverse relaxation time constant in seconds.
    T1 : float or None
        Longitudinal relaxation time constant in seconds.
    rho : float or None
        Surface relaxivity in m/s.  Requires D and dlog_boundary_unit.
    D : float or None
        Diffusion coefficient in m²/s.  Required when rho is not None.
    stimulated_echo : bool
        If True, multiply signal by 0.5 for PGSTE cos(phi1) storage factor.
    comp_traj : np.ndarray, shape (n_walkers, n_t_traj), or None
        Per-walker compartment ID at each trajectory time step, as returned by
        simulate_trajectories(save_relaxation_data=True).  Required when
        T2_per_comp or T1_per_comp is provided.  An integer array indexes the
        per-compartment arrays directly; a float array is the fractional
        occupancy of compartment 1 in a 2-compartment permeable geometry.
    T2_per_comp : array-like, shape (n_comp,), or None
        Per-compartment T2 in seconds.  If set, overrides scalar T2.
        Requires comp_traj.
    T1_per_comp : array-like, shape (n_comp,), or None
        Per-compartment T1 in seconds.  If set, overrides scalar T1.
        Requires comp_traj.
    return_walker_signals : bool
        If True, return a 3-tuple ``(phi, log_w_total, signals)`` instead of
        just ``signals``.

    Returns
    -------
    signals : np.ndarray, shape (n_meas,), float64
        Signal attenuation E = <exp(log_w) · cos(φ)>.  Scaled by 0.5 when
        stimulated_echo=True.  Returned as a bare array when
        ``return_walker_signals=False`` (default).
    phi : np.ndarray, shape (n_meas, n_walkers), float64
        Per-(measurement, walker) phase.  Only when ``return_walker_signals=True``.
    log_w_total : np.ndarray, shape (n_meas, n_walkers) or (n_meas, 1), float64
        Per-(measurement, walker) log-weight.  Only when
        ``return_walker_signals=True``.

    Notes
    -----
    chi_perp resampling uses nearest-neighbour (via round indexing) on the
    trajectory grid — appropriate since chi_perp is a binary step function not
    suited to linear interpolation.  G is resampled linearly (same as
    apply_waveform_to_trajectories).
    """
    G = np.asarray(G, dtype=np.float32)
    chi_perp = np.asarray(chi_perp, dtype=np.float64)
    n_meas, n_t_wf, _ = G.shape
    n_walkers, n_t_traj, _ = trajectories.shape

    per_meas_chi = chi_perp.ndim == 2  # (n_meas, n_t_wf) vs (n_t_wf,)

    # ── Resample G to trajectory grid ────────────────────────────────────────
    rel_dt_diff = abs(dt_wf - dt_traj) / max(abs(dt_traj), 1e-30)
    if rel_dt_diff > 1e-9:
        t_wf   = np.arange(n_t_wf,   dtype=np.float64) * dt_wf
        t_traj = np.arange(n_t_traj, dtype=np.float64) * dt_traj
        G_traj = np.zeros((n_meas, n_t_traj, 3), dtype=np.float32)
        for m in range(n_meas):
            for ax in range(3):
                # left/right=0: no gradient outside the waveform window (see
                # apply_waveform_to_trajectories) -- avoids extrapolating G[-1]
                # across a trajectory tail past the waveform end.
                G_traj[m, :, ax] = np.interp(t_traj, t_wf, G[m, :, ax],
                                             left=0.0, right=0.0)
    else:
        if n_t_wf == n_t_traj:
            G_traj = G
        elif n_t_wf > n_t_traj:
            G_traj = G[:, :n_t_traj, :]
        else:
            G_traj = np.zeros((n_meas, n_t_traj, 3), dtype=np.float32)
            G_traj[:, :n_t_wf, :] = G

    # ── Resample chi_perp to trajectory grid (nearest-neighbour) ─────────────
    t_wf   = np.arange(n_t_wf,   dtype=np.float64) * dt_wf
    t_traj = np.arange(n_t_traj, dtype=np.float64) * dt_traj

    def _resample_chi(chi_1d):
        # Nearest-neighbour inside waveform range; 0.0 outside (beyond waveform
        # end the echo has already formed — no further relaxation contribution).
        dt_wf_eff = t_wf[-1] / max(n_t_wf - 1, 1) if n_t_wf > 1 else dt_traj
        idx = np.round(t_traj / dt_wf_eff).astype(int)
        in_range = (idx >= 0) & (idx < n_t_wf)
        idx_clipped = np.clip(idx, 0, n_t_wf - 1)
        return np.where(in_range, chi_1d[idx_clipped], 0.0)

    if per_meas_chi:
        chi_r = np.stack([_resample_chi(chi_perp[m]) for m in range(n_meas)])
        # chi_r: (n_meas, n_t_traj)
    else:
        chi_r_1d = _resample_chi(chi_perp)
        chi_r = chi_r_1d[np.newaxis, :]  # (1, n_t_traj) — broadcast over measurements

    # ── Phase accumulation (same as apply_waveform_to_trajectories) ──────────
    G_flat    = G_traj.reshape(n_meas, n_t_traj * 3).astype(np.float64)
    traj_flat = trajectories.reshape(n_walkers, n_t_traj * 3).astype(np.float64)
    phi = GAMMA * dt_traj * (G_flat @ traj_flat.T)  # (n_meas, n_walkers)

    # ── T2/T1 log-weights — scalar (walker-independent) and per-walker ────────
    log_w_scalar = np.zeros(n_meas, dtype=np.float64)          # (n_meas,)
    log_w_per_walker = np.zeros((n_meas, 1), dtype=np.float64) # (n_meas, 1) or (n_meas, n_walkers)

    def _inv_rate_at_step(inv_arr, comp):
        """Per-(walker, step) inverse-relaxation rate from comp_traj.

        Discrete (integer) comp_traj indexes ``inv_arr`` directly.  Fractional
        (float) comp_traj is the time-fraction in compartment 1 of a
        2-compartment permeable geometry; the inverse rate is the
        occupancy-weighted average ``(1-f)·inv_arr[0] + f·inv_arr[1]`` (exact
        for compartment-weighted bulk relaxation, resolving intra-save crossings).
        """
        comp = np.asarray(comp)
        if np.issubdtype(comp.dtype, np.floating):
            if inv_arr.shape[0] != 2:
                raise ValueError(
                    "Fractional (float) comp_traj requires exactly 2 entries in "
                    f"the per-compartment relaxation array; got {inv_arr.shape[0]}.")
            f = comp.astype(np.float64)
            return inv_arr[0] + f * (inv_arr[1] - inv_arr[0])   # (n_walkers, n_t_traj)
        return inv_arr[comp]                                     # discrete index

    if T2_per_comp is not None:
        if comp_traj is None:
            raise ValueError("comp_traj required when T2_per_comp is provided.")
        T2_arr = np.asarray(T2_per_comp, dtype=np.float64)
        inv_T2_at_step = _inv_rate_at_step(1.0 / T2_arr, comp_traj)   # (n_walkers, n_t_traj)
        # log_w_t2[m,w] = -dt * chi_r[m,:] @ inv_T2_at_step[w,:].T
        log_w_t2 = -dt_traj * (chi_r @ inv_T2_at_step.T)   # (n_meas, n_walkers)
        log_w_per_walker = log_w_per_walker + log_w_t2
    elif T2 is not None:
        log_w_scalar -= (dt_traj / T2) * chi_r.sum(axis=1)  # (n_meas,)

    if T1_per_comp is not None:
        if comp_traj is None:
            raise ValueError("comp_traj required when T1_per_comp is provided.")
        T1_arr = np.asarray(T1_per_comp, dtype=np.float64)
        inv_T1_at_step = _inv_rate_at_step(1.0 / T1_arr, comp_traj)   # (n_walkers, n_t_traj)
        chi_inv_r = 1.0 - chi_r  # (n_meas, n_t_traj) or (1, n_t_traj)
        log_w_t1 = -dt_traj * (chi_inv_r @ inv_T1_at_step.T)   # (n_meas, n_walkers)
        log_w_per_walker = log_w_per_walker + log_w_t1
    elif T1 is not None:
        log_w_scalar -= (dt_traj / T1) * (n_t_traj - chi_r.sum(axis=1))  # (n_meas,)

    # ── Surface relaxivity walker-dependent log-weight ────────────────────────
    # log_w_rho[m,w] = (rho/D) * chi_r[m,:] @ dlog_bnd_unit[w,:].T
    if rho is not None:
        if D is None:
            raise ValueError("D (diffusivity) must be provided when rho is not None.")
        if dlog_boundary_unit is None:
            raise ValueError(
                "dlog_boundary_unit must be provided when rho is not None.  "
                "Re-run simulate_trajectories with save_relaxation_data=True.")
        dlog_bnd = np.asarray(dlog_boundary_unit, dtype=np.float64)  # (n_walkers, n_t_traj)
        log_w_rho = (rho / D) * (chi_r @ dlog_bnd.T)                 # (n_meas, n_walkers)
        log_w_per_walker = log_w_per_walker + log_w_rho

    log_w_total = log_w_scalar[:, np.newaxis] + log_w_per_walker  # (n_meas, n_walkers) or (n_meas, 1)

    # ── Susceptibility off-resonance phase (opt-in, provider-driven) ────────────
    # phi_susc[m,w] = γ · dt · Σ_t ε_P[m,t] · ΔBz(r[w,t]).  The 180° refocusing of the
    # static field is handled by the ε_P sign flip at TE/2 (see pathway_sign_se); with
    # eps_P=None it defaults to chi_r (FID / gradient-echo: no sign flip, so any static
    # field dephases for the full readout — WARNING: do not use for a spin echo).
    if susceptibility is not None:
        field_fn = _resolve_field_fn(susceptibility)
        dB = _sample_delta_bz(field_fn, trajectories)           # (n_walkers, n_t_traj)
        if eps_P is not None:
            eps_arr = np.asarray(eps_P, dtype=np.float64)
            if eps_arr.ndim == 2:
                eps_r = np.stack([_resample_chi(eps_arr[m]) for m in range(n_meas)])
            else:
                eps_r = _resample_chi(eps_arr)[np.newaxis, :]   # (1, n_t_traj)
        else:
            eps_r = chi_r                                       # FID approximation
        phi = phi + GAMMA * dt_traj * (eps_r @ dB.T)            # (n_meas, n_walkers)

    # ── Signal ────────────────────────────────────────────────────────────────
    signals = np.mean(np.exp(log_w_total) * np.cos(phi), axis=1)  # (n_meas,)

    if stimulated_echo:
        signals = signals * 0.5

    if return_walker_signals:
        return phi, log_w_total, signals
    return signals


def apply_waveform_with_relaxation_jax(
    G,
    traj,
    dt_traj: float,
    chi_perp,
    dt_wf=None,
    dlog_boundary_unit=None,
    T2=None,
    T1=None,
    rho=None,
    D: float = None,
    walker_weights=None,
    stimulated_echo: bool = False,
):
    """JAX-differentiable waveform replay with T2, T1, and surface-relaxivity weights.

    Extends :func:`apply_waveform_jax` with chi_perp-gated T2/T1 decay and
    surface relaxivity (rho) replay over pre-saved boundary hit data.  All
    physical parameters (G, T2, T1, rho, chi_perp) are JAX-traced and fully
    differentiable via ``jax.grad`` or ``jax.jacobian``.

    Physics
    -------
    For each measurement *m* and walker *w*:

        phi[m,w]   = gamma * dt_traj * sum_t  G_r[m,t,:] . traj[w,t,:]

        log_w[m,w] = - (dt_traj / T2) * sum_t  chi_r[m,t]           (T2 term)
                     - (dt_traj / T1) * sum_t  (1 - chi_r[m,t])     (T1 term)
                     + (rho / D)      * sum_t  chi_r[m,t] * dlog_bnd_unit[w,t]
                                                                     (rho term)

        E[m] = mean_w( exp(log_w[m,w]) * cos(phi[m,w]) )

    Scalar T2/T1 only (per-compartment is not differentiable here — use the
    NumPy :func:`apply_waveform_with_relaxation`).

    Parameters
    ----------
    G : jnp.ndarray, shape (n_meas, n_t_wf, 3)
        Gradient waveform array in T/m.  JAX-traced (differentiable).
    traj : jnp.ndarray, shape (n_walkers, n_t_traj, 3)
        Walker positions in metres.  Constant (not differentiated).  float16 or
        float32; cast to float32 internally.
    dt_traj : float
        Trajectory time step in seconds.
    chi_perp : jnp.ndarray, shape (n_t_wf,) or (n_meas, n_t_wf)
        Transverse gating schedule.  Resampled to the trajectory grid with
        nearest-neighbour; values beyond the waveform end are set to 0.0.
    dt_wf : float or None
        Waveform time step in seconds.  If None, assumed equal to ``dt_traj``.
    dlog_boundary_unit : jnp.ndarray, shape (n_walkers, n_t_traj), or None
        Boundary log-weight with rho/D = 1 (see simulate_trajectories).  Required
        when ``rho`` is not None.
    T2, T1 : scalar or None
        Transverse / longitudinal relaxation times in seconds (differentiable).
    rho : scalar or None
        Surface relaxivity in m/s (differentiable).  Requires D and
        dlog_boundary_unit.
    D : float or None
        Diffusion coefficient in m²/s.  Required when ``rho`` is not None.
    walker_weights : jnp.ndarray, shape (n_walkers,), or None
        Optional per-walker importance weights (normalised internally).
    stimulated_echo : bool
        If True, multiply the returned signal by 0.5 (PGSTE storage factor).

    Returns
    -------
    signals : jnp.ndarray, shape (n_meas,)
        Signal attenuation E = <exp(log_w) * cos(phi)>.  Differentiable w.r.t.
        G, T2, T1, rho, chi_perp.  Scaled by 0.5 when ``stimulated_echo=True``.
    """
    if not _JAX_AVAILABLE:
        raise ImportError(
            "JAX is required for apply_waveform_with_relaxation_jax. "
            "Install jax or use apply_waveform_with_relaxation for NumPy."
        )

    n_meas, n_t_wf, _ = G.shape
    n_walkers, n_t_traj, _ = traj.shape

    if dt_wf is None:
        dt_wf = dt_traj

    # ── G resampling (same as apply_waveform_jax) ────────────────────────────
    if abs(dt_wf - dt_traj) / max(abs(dt_traj), 1e-30) < 1e-9:
        if n_t_wf == n_t_traj:
            G_r = G.astype(jnp.float32)
        elif n_t_wf > n_t_traj:
            G_r = G[:, :n_t_traj, :].astype(jnp.float32)
        else:
            pad = jnp.zeros((n_meas, n_t_traj - n_t_wf, 3), dtype=jnp.float32)
            G_r = jnp.concatenate([G.astype(jnp.float32), pad], axis=1)
    else:
        t_wf_g   = jnp.arange(n_t_wf,   dtype=jnp.float32) * float(dt_wf)
        t_traj_g = jnp.arange(n_t_traj, dtype=jnp.float32) * float(dt_traj)

        def interp_one(g_1d):
            return jnp.interp(t_traj_g, t_wf_g, g_1d, left=0.0, right=0.0)

        interp_ax   = jax.vmap(interp_one)
        interp_meas = jax.vmap(interp_ax)

        G_t   = G.astype(jnp.float32).transpose(0, 2, 1)  # (n_meas, 3, n_t_wf)
        G_r_t = interp_meas(G_t)                           # (n_meas, 3, n_t_traj)
        G_r   = G_r_t.transpose(0, 2, 1)                  # (n_meas, n_t_traj, 3)

    # ── chi_perp resampling (nearest-neighbour, zero outside waveform) ───────
    t_wf_jax   = jnp.arange(n_t_wf,   dtype=jnp.float32) * float(dt_wf)
    t_traj_jax = jnp.arange(n_t_traj, dtype=jnp.float32) * float(dt_traj)
    dt_wf_eff  = float(dt_wf)
    in_range   = t_traj_jax <= t_wf_jax[-1]  # (n_t_traj,) bool

    def _resample_chi_1d(chi_1d):
        idx = jnp.round(t_traj_jax / dt_wf_eff).astype(jnp.int32)
        idx_clipped = jnp.clip(idx, 0, n_t_wf - 1)
        return jnp.where(in_range, chi_1d[idx_clipped], jnp.float32(0.0))

    chi_perp_arr = jnp.asarray(chi_perp, dtype=jnp.float32)
    if chi_perp_arr.ndim == 1:
        chi_r = _resample_chi_1d(chi_perp_arr)[jnp.newaxis, :]  # (1, n_t_traj)
    else:
        chi_r = jax.vmap(_resample_chi_1d)(chi_perp_arr)         # (n_meas, n_t_traj)

    # ── Phase accumulation ────────────────────────────────────────────────────
    traj_f32 = traj.astype(jnp.float32)
    phi = _GAMMA_JAX * float(dt_traj) * jnp.einsum('mtx,wtx->mw', G_r, traj_f32)

    # ── T2/T1 scalar log-weight per measurement (walker-independent) ──────────
    log_w_scalar = jnp.zeros(n_meas, dtype=jnp.float32)  # (n_meas,)

    if T2 is not None:
        T2_f = jnp.float32(T2)
        log_w_scalar = log_w_scalar - (jnp.float32(dt_traj) / T2_f) * jnp.sum(
            chi_r, axis=1)

    if T1 is not None:
        T1_f = jnp.float32(T1)
        log_w_scalar = log_w_scalar - (jnp.float32(dt_traj) / T1_f) * jnp.sum(
            jnp.float32(1.0) - chi_r, axis=1)

    # ── Surface relaxivity walker-dependent log-weight ────────────────────────
    if rho is not None:
        if D is None:
            raise ValueError("D required when rho is not None")
        if dlog_boundary_unit is None:
            raise ValueError(
                "dlog_boundary_unit required when rho is not None. "
                "Re-run simulate_trajectories with save_relaxation_data=True.")
        dlog_bnd = jnp.asarray(dlog_boundary_unit, dtype=jnp.float32)  # (n_walkers, n_t_traj)
        rho_over_D = jnp.float32(rho) / jnp.float32(D)
        log_w_rho = rho_over_D * jnp.einsum('mt,wt->mw', chi_r, dlog_bnd)
        log_w_total = log_w_scalar[:, jnp.newaxis] + log_w_rho  # (n_meas, n_walkers)
    else:
        log_w_total = log_w_scalar[:, jnp.newaxis]  # (n_meas, 1) broadcasts over walkers

    # ── Signal ────────────────────────────────────────────────────────────────
    weights = jnp.exp(log_w_total) * jnp.cos(phi)  # (n_meas, n_walkers)

    if walker_weights is None:
        signals = jnp.mean(weights, axis=1)
    else:
        w_norm = walker_weights / (jnp.sum(walker_weights) + jnp.float32(1e-30))
        signals = jnp.einsum('mw,w->m', weights, w_norm)

    if stimulated_echo:
        signals = signals * jnp.float32(0.5)

    return signals


# ══════════════════════════════════════════════════════════════════════════════
# Vector-Bloch replay: evolve M = (Mx, My, Mz) on the stored trajectory
# ══════════════════════════════════════════════════════════════════════════════
# Where the scalar path accumulates a single phase and gates ``log_w``, the
# vector-Bloch replay carries the full magnetisation vector through the ACTUAL
# sequence operators (RF rotations + gradient/off-resonance precession + per-comp
# T2/T1 + surface relaxivity + an optional MT bound-pool blend), re-evolved off the
# same field-independent walk.  Spin echoes / CPMG refocusing are EMERGENT: the 180°
# rotation conjugates the accumulated phase, so no ``eps_P`` sign is needed here (pass
# the PHYSICAL same-sign gradient, ``Waveform.G``, NOT the bipolar/effective one).
# This mirrors the forward engine ``bloch.simulate_bloch`` one-to-one.


def _rf_increment(M, flip, ax):
    """Rotate M (3, N) by ``flip`` rad about the in-plane B1 axis ``ax`` rad (Rodrigues,
    axis (cos ax, sin ax, 0)).  One partial B1 step of a finite pulse; the free
    precession between successive increments (in the main loop) supplies the
    off-resonance tilt.  ``flip`` may be scalar (uniform B1) or a per-walker (N,) array
    (B1+ transmit inhomogeneity)."""
    ux, uy = np.cos(ax), np.sin(ax)
    c, s = np.cos(flip), np.sin(flip)                          # scalar or (N,)
    omc = 1.0 - c
    Mx, My, Mz = M[0], M[1], M[2]
    Mx2 = (c + ux * ux * omc) * Mx + (ux * uy * omc) * My + (uy * s) * Mz
    My2 = (ux * uy * omc) * Mx + (c + uy * uy * omc) * My + (-ux * s) * Mz
    Mz2 = (-uy * s) * Mx + (ux * s) * My + c * Mz
    return np.stack([Mx2, My2, Mz2])


def _rf_matrix(event):
    """3x3 hard-RF rotation matrix (rotation by flip_deg about (cos ax, sin ax, 0))."""
    flip = np.deg2rad(float(event.get('flip_deg', 180.0)))
    ax = np.deg2rad(float(event.get('axis_deg', 0.0)))
    ux, uy = np.cos(ax), np.sin(ax)
    c, s = np.cos(flip), np.sin(flip)
    omc = 1.0 - c
    return np.array([
        [c + ux * ux * omc, ux * uy * omc, uy * s],
        [ux * uy * omc, c + uy * uy * omc, -ux * s],
        [-uy * s, ux * s, c],
    ], dtype=np.float64)


def finite_180_longitudinal_dwell(phi_pre, tau_180, phi_B1=0.0):
    r"""Per-walker longitudinal dwell during a finite 180° refocusing pulse.

    A 180° rotates M about the B1 axis.  The component ALONG B1 stays transverse the
    whole pulse; only the PERPENDICULAR component swings transverse → longitudinal →
    transverse, spending time at z where T1 (not T2 / surface / susceptibility) acts.
    For a walker arriving with transverse azimuth ``phi_pre`` (its accumulated
    gradient + susceptibility phase relative to the static-spin reference), the
    longitudinal dwell is

        tau_par = sin²(phi_pre − phi_B1) · tau_180 / 2 .

    Meiboom-Gill (``phi_B1 = 0``) keeps a coherent on-resonance spin off z
    (tau_par = 0), while a uniformly dephased ensemble (<sin²> = 1/2) gives
    tau_par → tau_180 / 4 — the per-walker, tissue-blind replacement for the scalar
    ``|cos|`` / ``cos²`` profiles.

    Parameters
    ----------
    phi_pre : array-like   per-walker accumulated transverse phase at the pulse (rad).
    tau_180 : float        refocusing pulse duration (s).
    phi_B1 : float         azimuth of the B1 axis vs the static reference (rad); 0 = MG.

    Returns
    -------
    np.ndarray   per-walker longitudinal dwell ``tau_par`` (s).  Move this much dwell
        from the T2 to the T1 channel.
    """
    return np.sin(np.asarray(phi_pre, dtype=np.float64) - phi_B1) ** 2 * (0.5 * tau_180)


def pre_pulse_gradient_phase(trajectories, dt_traj, G, dt_wf, cutoff_wf_idx):
    """Per-walker accumulated gradient phase up to a waveform index (the azimuth a
    finite refocusing pulse sees).

    Mirrors the phase accumulation of :func:`apply_waveform_with_relaxation` but
    truncated at ``cutoff_wf_idx`` (the 180° index in the waveform grid):

        phi_pre[m,w] = γ · dt_traj · Σ_{t < cutoff_traj} G_r[m,t,:] · r[w,t,:]

    Feed the result to :func:`finite_180_longitudinal_dwell`.  Susceptibility
    off-resonance adds to this azimuth separately (only the pre-pulse ε=+1 part).
    """
    G = np.asarray(G, dtype=np.float32)
    n_meas, n_t_wf, _ = G.shape
    n_walkers, n_t_traj, _ = trajectories.shape
    cutoff_traj = int(round(cutoff_wf_idx * dt_wf / dt_traj))
    cutoff_traj = max(0, min(cutoff_traj, n_t_traj))
    t_wf = np.arange(n_t_wf, dtype=np.float64) * dt_wf
    t_traj = np.arange(n_t_traj, dtype=np.float64) * dt_traj
    G_pre = np.zeros((n_meas, n_t_traj, 3), dtype=np.float64)
    for m in range(n_meas):
        for ax in range(3):
            G_pre[m, :, ax] = np.interp(t_traj, t_wf, G[m, :, ax], left=0.0, right=0.0)
    G_pre[:, cutoff_traj:, :] = 0.0
    G_flat = G_pre.reshape(n_meas, n_t_traj * 3)
    traj_flat = trajectories.reshape(n_walkers, n_t_traj * 3).astype(np.float64)
    return GAMMA * dt_traj * (G_flat @ traj_flat.T)  # (n_meas, n_walkers)


def apply_waveform_bloch(trajectories, dt_traj, G, dt_wf, rf_events,
                         T2=None, T1=None, comp_traj=None,
                         T2_per_comp=None, T1_per_comp=None,
                         susceptibility=None, extra_phase_per_step=None,
                         dlog_boundary_unit=None, rho=None, D=None,
                         echo_steps=None, echo_per_walker=False,
                         weights=None, return_walker_signals=False,
                         b1_scale=None, slice_offsets=None, slice_gradient=0.0,
                         bound_frac=None, T2_bound=None, T1_bound=None,
                         off_resonance_bound=0.0):
    """Emergent per-walker vector-Bloch replay on the stored (field-independent) walk.

    Propagates each walker's magnetisation ``M = (Mx, My, Mz)`` through the ACTUAL
    sequence operators — RF rotations from ``rf_events``, the physical gradient
    precession ``dφ = γ·G(t)·r_w(t)·dt``, per-comp T2/T1, surface relaxivity, an
    optional susceptibility off-resonance field, and an optional MT bound-pool blend.
    The forward-engine counterpart is :func:`dmipy_sim.bloch.simulate_bloch`.

    Coherence pathways / refocusing are EMERGENT: the 180° rotation conjugates the
    accumulated phase, so the spin echo forms by itself — there is no ``eps_P`` sign.
    **Pass the PHYSICAL (same-sign-lobe) gradient** (``Waveform.G``); the bipolar /
    effective convention must NOT be used here.

    Parameters
    ----------
    rf_events : list of dict
        Each ``{'t_s', 'flip_deg', 'axis_deg', 'duration_s', 'offset_hz'}``; ``axis_deg``
        is the B1 phase (0 = x, 90 = y); ``duration_s = 0`` is an instantaneous hard
        pulse; ``offset_hz`` gives an off-resonance carrier over the pulse (what makes an
        MT-prep saturation pulse saturate the broad bound pool).
    T2, T1 / T2_per_comp, T1_per_comp, comp_traj : as in apply_waveform_with_relaxation.
    susceptibility : provider or callable, optional
        A :mod:`dmipy_sim.susceptibility` provider (``delta_bz_fn()``) or a bare
        ``r -> ΔBz`` callable; sampled along the walk and added to the free precession
        every step as ``γ·ΔBz(r(t))·dt`` — refocused emergently by the sequence's 180°.
        Mutually exclusive with ``extra_phase_per_step`` (which is the pre-baked
        per-step phase increment ``γ·ΔBz·dt`` if you already have it).
    bound_frac, T2_bound, T1_bound, off_resonance_bound : magnetization transfer.
        ``bound_frac`` (n_w, n_t) is the per-step bound-pool occupancy from
        :func:`dmipy_sim.mt_walk.simulate_mt_trajectories`; when given, the per-step
        relaxation RATE is blended toward the bound-pool ``T2_bound``/``T1_bound`` (and
        ``off_resonance_bound``) by that occupancy.  RF rotates bound spins too, so MT
        saturation transfer is EMERGENT.  None → no MT.

    Returns
    -------
    signals : (n_meas,) complex   walker-mean ``Mx + i My`` at the last step, or
        ``(n_meas, n_echo)`` when ``echo_steps`` is given, or ``(M_final, signals)``
        with the per-walker (3, n_w) final magnetisation when ``return_walker_signals``.
    """
    G = np.asarray(G, dtype=np.float64)
    n_meas, n_t_wf, _ = G.shape
    n_w, n_t, _ = trajectories.shape
    traj = trajectories.astype(np.float64)

    # ── resample G to the trajectory grid (nearest within window, 0 outside) ──
    t_wf = np.arange(n_t_wf, dtype=np.float64) * dt_wf
    t_tr = np.arange(n_t, dtype=np.float64) * dt_traj
    G_tr = np.zeros((n_meas, n_t, 3), dtype=np.float64)
    for m in range(n_meas):
        for ax in range(3):
            G_tr[m, :, ax] = np.interp(t_tr, t_wf, G[m, :, ax], left=0.0, right=0.0)

    # ── per-walker decay factors per step ──
    if T2_per_comp is not None:
        invT2 = (1.0 / np.asarray(T2_per_comp, np.float64))[comp_traj]   # (n_w, n_t)
    else:
        invT2 = np.full((n_w, n_t), 0.0 if T2 is None else 1.0 / T2)
    if T1_per_comp is not None:
        invT1 = (1.0 / np.asarray(T1_per_comp, np.float64))[comp_traj]
    else:
        invT1 = np.full((n_w, n_t), 0.0 if T1 is None else 1.0 / T1)

    # ── magnetization transfer: blend to the bound-pool relaxation by occupancy ──
    # bound_frac[w,t] in [0,1] is the time fraction walker w spent bound during step t
    # (from simulate_mt_trajectories).  A bound spin relaxes with the very short
    # T2_bound / T1_bound; blending the RATE by occupancy is exact for the fully
    # bound/free steps (frac 0 or 1) and a linear interpolation for partial saves.  RF
    # still rotates bound spins (the loop rotates ALL walkers), so MT saturation and its
    # exchange transfer to the free pool are EMERGENT — no bound-pool lineshape imposed.
    has_mt = bound_frac is not None
    if has_mt:
        if T2_bound is None or T1_bound is None:
            raise ValueError("T2_bound and T1_bound are required when bound_frac is set.")
        bf = np.asarray(bound_frac, np.float64)                 # (n_w, n_t)
        invT2 = (1.0 - bf) * invT2 + bf * (1.0 / float(T2_bound))
        invT1 = (1.0 - bf) * invT1 + bf * (1.0 / float(T1_bound))

    E2 = np.exp(-dt_traj * invT2)        # (n_w, n_t)
    E1 = np.exp(-dt_traj * invT1)

    # bound-pool off-resonance: a per-step transverse phase for the bound occupancy.
    dphi_bound = None
    if has_mt and float(off_resonance_bound) != 0.0:
        dphi_bound = bf * (2.0 * np.pi * float(off_resonance_bound) * dt_traj)

    # ── susceptibility: per-step transverse phase increment (refocusing emergent) ──
    # The 180° rotation conjugates the accumulated phase, so SE refocusing of the static
    # field needs no eps_P — it is automatic.  A provider is sampled along the walk;
    # extra_phase_per_step lets a caller pass a pre-baked γ·ΔBz·dt array directly.
    if susceptibility is not None and extra_phase_per_step is None:
        field_fn = _resolve_field_fn(susceptibility)
        extra_phase_per_step = GAMMA * dt_traj * _sample_delta_bz(field_fn, trajectories)
    has_susc = extra_phase_per_step is not None
    if has_susc:
        dphi_susc = np.asarray(extra_phase_per_step, np.float64)        # (n_w, n_t)

    # ── surface relaxivity: per-step transverse attenuation at wall contacts ──
    has_surf = rho is not None and dlog_boundary_unit is not None and float(rho) != 0.0
    if has_surf:
        if D is None:
            raise ValueError("D required when rho is not None.")
        surf = np.exp((float(rho) / float(D)) * np.asarray(dlog_boundary_unit, np.float64))

    # ── RF events → per-step incremental B1 rotations ──
    # A finite pulse (duration_s > 0) is spread over its REAL duration as partial B1
    # rotations, one per trajectory step, with the free precession (gradient + static
    # susceptibility off-resonance) acting BETWEEN them — so imperfect, off-resonant
    # flips EMERGE.  duration_s = 0 is the instantaneous hard-pulse limit (one step).
    rf_at = {}
    rf_active = np.zeros(n_t, dtype=bool)
    rf_offset_dphi = np.zeros(n_t, dtype=np.float64)
    for e in rf_events:
        i0 = int(round(float(e['t_s']) / dt_traj))
        dur = float(e.get('duration_s', 0.0) or 0.0)
        nsub = max(1, int(round(dur / dt_traj))) if dur > 0.0 else 1
        # CENTRE the pulse on its nominal time t_s (the echo refocuses about the pulse
        # centre), so a finite 180 at TE/2 still refocuses at TE.  The excitation (t_s=0)
        # clamps to run forward from 0.
        i_start = max(0, i0 - nsub // 2)
        total = np.deg2rad(float(e.get('flip_deg', 180.0)))
        ax = np.deg2rad(float(e.get('axis_deg', 0.0)))
        off_hz = float(e.get('offset_hz', 0.0) or 0.0)         # carrier off-resonance
        off_dphi = 2.0 * np.pi * off_hz * dt_traj
        env = e.get('b1_envelope', None)
        if env is not None and nsub > 1:                       # shaped pulse
            env = np.asarray(env, dtype=np.float64)
            env = np.interp(np.linspace(0.0, 1.0, nsub),
                            np.linspace(0.0, 1.0, len(env)), env)
            dflips = total * env / (env.sum() + 1e-30)         # preserve total flip
        else:
            dflips = np.full(nsub, total / nsub)               # flat rectangle
        for j in range(nsub):
            i = min(i_start + j, n_t - 1)
            rf_at.setdefault(i, []).append((float(dflips[j]), ax))
            rf_active[i] = True
            rf_offset_dphi[i] += off_dphi
    has_rf_offset = bool(np.any(rf_offset_dphi != 0.0))

    # spin weights for the ensemble average (None = uniform)
    if weights is None:
        wmean = lambda v: np.mean(v)
    else:
        wn = np.asarray(weights, np.float64); wsum = wn.sum()
        wmean = lambda v: (v * wn).sum() / wsum

    # B1+ transmit inhomogeneity: per-walker actual-flip multiplier (1.0 = ideal)
    b1s = 1.0 if b1_scale is None else np.asarray(b1_scale, np.float64)
    # slice-select off-resonance: a spin at slice position z sees Δω = γ·Gss·z during a
    # pulse, so the slice profile emerges.  slice_offsets (n_w,) in metres.
    has_slice = slice_offsets is not None and float(slice_gradient) != 0.0
    if has_slice:
        slice_dphi = (GAMMA * float(slice_gradient)
                      * np.asarray(slice_offsets, np.float64) * dt_traj)   # (n_w,)

    echo_set = None if echo_steps is None else set(int(e) for e in echo_steps)
    signals = np.empty(n_meas, dtype=np.complex128)
    echo_out = None
    M_last = None
    for m in range(n_meas):
        M = np.zeros((3, n_w), dtype=np.float64)
        M[2] = 1.0                                   # equilibrium along +z
        rec = []
        for t in range(n_t):
            if t in rf_at:
                for dflip, ax in rf_at[t]:           # partial B1 rotation(s) this step
                    M = _rf_increment(M, dflip * b1s, ax)   # b1s scales actual flip (B1+)
            # free precession EVERY step (incl. RF steps): the spin precesses during the
            # pulse dt too, keeping dephase/rephase intervals symmetric (essential CPMG).
            dphi = GAMMA * dt_traj * (traj[:, t, :] @ G_tr[m, t])   # (n_w,)
            if has_susc:
                dphi = dphi + dphi_susc[:, t]
            if dphi_bound is not None:               # bound-pool off-resonance
                dphi = dphi + dphi_bound[:, t]
            if has_rf_offset:                        # off-resonance RF carrier (MT-prep)
                dphi = dphi + rf_offset_dphi[t]
            if has_slice and rf_active[t]:           # slice-select off-resonance in pulse
                dphi = dphi + slice_dphi
            c, s = np.cos(dphi), np.sin(dphi)
            Mx = c * M[0] - s * M[1]
            My = s * M[0] + c * M[1]
            M = np.stack([Mx, My, M[2]])
            if has_surf:                             # transverse wall attenuation
                M = np.stack([M[0] * surf[:, t], M[1] * surf[:, t], M[2]])
            # uniform per-step relaxation (each step gets exactly one dt of decay)
            M = np.stack([M[0] * E2[:, t], M[1] * E2[:, t], M[2] * E1[:, t]])
            if echo_set is not None and t in echo_set:
                rec.append((M[0] + 1j * M[1]).copy() if echo_per_walker
                           else wmean(M[0] + 1j * M[1]))
        signals[m] = wmean(M[0] + 1j * M[1])
        M_last = M
        if echo_set is not None:
            if echo_per_walker:
                if echo_out is None:
                    echo_out = np.empty((n_meas, len(rec), n_w), dtype=np.complex128)
                echo_out[m] = np.asarray(rec)
            else:
                if echo_out is None:
                    echo_out = np.empty((n_meas, len(rec)), dtype=np.complex128)
                echo_out[m] = rec
    if echo_steps is not None:
        return echo_out                              # (n_meas, n_echo)
    if return_walker_signals:
        return M_last, signals                       # M_last: (3, n_w)
    return signals


def apply_waveform_bloch_jax(trajectories, dt_traj, G, dt_wf, rf_events,
                             T2=None, T1=None, comp_traj=None,
                             T2_per_comp=None, T1_per_comp=None,
                             susceptibility=None, extra_phase_per_step=None,
                             dlog_boundary_unit=None, rho=None, D=None,
                             echo_steps=None):
    """GPU/JAX vectorisation of :func:`apply_waveform_bloch` (hard-pulse rotations).

    Same physics and outputs as the numpy primitive for hard (instantaneous) pulses,
    evaluated with a jitted ``lax.scan`` over time.  Returns ``echo_out`` (n_meas,
    n_echo) when ``echo_steps`` is given, else ``signals`` (n_meas,) complex at the
    last step.  (MT bound-pool blend, finite-pulse spreading and slice-select are not
    vectorised here — use the numpy path for those.)
    """
    if not _JAX_AVAILABLE:
        raise RuntimeError("JAX not available; use apply_waveform_bloch.")
    G = np.asarray(G, np.float64)
    n_meas, n_t_wf, _ = G.shape
    n_w, n_t, _ = trajectories.shape
    traj = trajectories.astype(np.float64)

    t_wf = np.arange(n_t_wf) * dt_wf
    t_tr = np.arange(n_t) * dt_traj
    G_tr = np.zeros((n_meas, n_t, 3))
    for m in range(n_meas):
        for ax in range(3):
            G_tr[m, :, ax] = np.interp(t_tr, t_wf, G[m, :, ax], left=0.0, right=0.0)

    invT2 = ((1.0 / np.asarray(T2_per_comp, float))[comp_traj] if T2_per_comp is not None
             else np.full((n_w, n_t), 0.0 if T2 is None else 1.0 / T2))
    invT1 = ((1.0 / np.asarray(T1_per_comp, float))[comp_traj] if T1_per_comp is not None
             else np.full((n_w, n_t), 0.0 if T1 is None else 1.0 / T1))
    E2 = jnp.asarray(np.exp(-dt_traj * invT2).T)         # (n_t, n_w)
    E1 = jnp.asarray(np.exp(-dt_traj * invT1).T)
    if rho is not None and dlog_boundary_unit is not None and float(rho) != 0.0:
        surf = jnp.asarray(np.exp((float(rho) / float(D))
                                  * np.asarray(dlog_boundary_unit, float)).T)
    else:
        surf = jnp.ones((n_t, n_w))
    if susceptibility is not None and extra_phase_per_step is None:
        field_fn = _resolve_field_fn(susceptibility)
        extra_phase_per_step = GAMMA * dt_traj * _sample_delta_bz(field_fn, trajectories)
    dphi_susc = (jnp.asarray(np.asarray(extra_phase_per_step, float).T)
                 if extra_phase_per_step is not None else jnp.zeros((n_t, n_w)))

    # per-step RF rotation matrices (identity off-pulse; hard pulses only)
    Rs = np.broadcast_to(np.eye(3), (n_t, 3, 3)).copy()
    for e in rf_events:
        idx = max(0, min(int(round(float(e['t_s']) / dt_traj)), n_t - 1))
        Rs[idx] = _rf_matrix(e)
    Rs = jnp.asarray(Rs)

    echo_idx = (np.asarray(sorted(int(e) for e in echo_steps))
                if echo_steps is not None else None)

    def run_meas(Gm):                                    # Gm: (n_t, 3)
        dphi_grad = _GAMMA_JAX * dt_traj * jnp.einsum('td,wtd->tw',
                                                      jnp.asarray(Gm), jnp.asarray(traj))
        dphi = dphi_grad + dphi_susc                     # (n_t, n_w)

        def step(M, x):
            Rt, dphi_t, e2, e1, sf = x
            M = Rt @ M
            c, s = jnp.cos(dphi_t), jnp.sin(dphi_t)
            Mx = (c * M[0] - s * M[1]) * sf * e2
            My = (s * M[0] + c * M[1]) * sf * e2
            M = jnp.stack([Mx, My, M[2] * e1])
            return M, jnp.mean(M[0] + 1j * M[1])
        M0 = jnp.stack([jnp.zeros(n_w), jnp.zeros(n_w), jnp.ones(n_w)])
        _, sig_t = jax.lax.scan(step, M0, (Rs, dphi, E2, E1, surf))
        return sig_t                                     # (n_t,) mean Mxy per step

    sig_all = jax.vmap(run_meas)(jnp.asarray(G_tr))      # (n_meas, n_t)
    sig_all = np.asarray(sig_all)
    if echo_idx is not None:
        return sig_all[:, echo_idx]
    return sig_all[:, -1]
