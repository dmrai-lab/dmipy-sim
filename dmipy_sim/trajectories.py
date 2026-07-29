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
):
    """Apply a gradient waveform and relaxation weights to saved walker trajectories.

    Extends apply_waveform_to_trajectories() with chi_perp-gated T2/T1 decay
    and surface relaxivity (rho) replay over pre-saved boundary hit data.

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
