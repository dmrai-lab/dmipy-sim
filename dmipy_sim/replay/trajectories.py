"""ScannerSequence application to pre-computed walker trajectories (the replay path).

Decouples MC geometry simulation from gradient encoding + relaxation, enabling
fast re-use of a single walker library for arbitrary waveforms, T2/T1 schedules
and surface relaxivity.

The replay invariant (see ``core.simulate_trajectories``): walker positions
``r(t)`` and boundary events depend ONLY on ``(geometry, diffusivity, seed)`` —
never on ``G(t)``, T2, T1 or ρ.  Those gate ``log_w``/phase only, so
``simulate_trajectories(...)`` walks once and the
``replay*`` functions below replay many acquisition/relaxation
hypotheses off that one walk.
"""

import functools
from typing import NamedTuple

import numpy as np

from ..constants import GAMMA
from ..acquisition.rf import RFSchedule
from ._replay_kernel import (effective_gradient, effective_gradient_jax, gradient_phase,
                             gradient_phase_jax, piece_phase_weights, gate_weights, bin_gate)

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

    Accepts a :mod:`dmipy_sim.fields.susceptibility` provider (exposes ``delta_bz_fn()``)
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
    (:func:`replay`) the static phase must be summed with a sign
    that flips at the pulse: ``ε_P = +1`` for ``t < te_frac·TE`` and ``−1`` after.  (The
    vector-Bloch replay :func:`replay_bloch` needs no ``ε_P`` — its explicit 180°
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


def replay_jax(
    G,
    trajectory,
    dt_traj: float,
    dt_wf=None,
    weights=None,
    stimulated_echo: bool = False,
):
    """JAX-differentiable waveform replay on stored walker trajectories.

    Computes the diffusion-weighted signal by accumulating the gradient-
    position dot product along each walker's trajectory using JAX operations:

        φ_m,w = γ · dt_traj · Σ_t  G_resampled[m,t,:] · traj[w,t,:]
        E[m]  = mean_w( cos(φ_m,w) )   or weighted mean if weights given

    Fully differentiable with respect to ``G`` via ``jax.grad``.

    Parameters
    ----------
    G : jnp.ndarray, shape (n_meas, n_t_wf, 3)
        Gradient waveform array in T/m.  JAX-traced (differentiable).
    trajectory : jnp.ndarray, shape (n_walkers, n_t_traj, 3)
        Walker positions in metres.  Treated as a constant (not differentiated).
        float16 or float32; cast to float32 internally.
    dt_traj : float
        Trajectory time step in seconds.
    dt_wf : float or None
        ScannerSequence time step in seconds.  If None, assumed equal to ``dt_traj``
        (no resampling).
    weights : jnp.ndarray, shape (n_walkers,) or None
        Optional per-walker importance weights.  If None, uniform mean is used.
        Weights are normalised to sum to 1 internally.
    stimulated_echo : bool
        If True, multiply the returned signal by 0.5 to account for the
        cos(phi1) storage step in PGSTE sequences.  Pass
        ``stimulated_echo=wf.stimulated_echo`` when replaying a ScannerSequence
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
            "JAX is required for replay_jax. "
            "Install jax or use replay for NumPy."
        )

    n_walkers, n_t_traj, _ = trajectory.shape
    G_r = effective_gradient_jax(G, dt_wf, n_t_traj, dt_traj)
    phi = gradient_phase_jax(G_r, trajectory, dt_traj)                 # (n_meas, n_walkers)

    # --- Weighted average ---
    if weights is None:
        signals = jnp.mean(jnp.cos(phi), axis=1)
    else:
        w_norm = weights / (jnp.sum(weights) + 1e-30)
        signals = jnp.einsum('mw,w->m', jnp.cos(phi), w_norm)

    if stimulated_echo:
        signals = signals * jnp.float32(0.5)
    return signals


def _replay_compressed(master, G, dt_wf, *, chi_perp, T2, T1, surface_relaxivity, D,
                       stimulated_echo, comp_traj, T2_per_comp, T1_per_comp,
                       return_walker_signals, susceptibility, eps_P):
    """Replay a COMPRESSED master (IR modes from simulate_trajectories(compress=K)).

    The gradient phase is computed in mode space (compression.mode_space_phi: contract the
    stored temporal-DCT position bands against DCT(G)) — the (N, n_t, 3) trajectory is NEVER
    reconstructed. Surface relaxivity uses the stored cumulative endpoint (ungated / full
    walk) or the decoded per-save boundary channel (chi-gated). Per-compartment T2/T1 use
    the raw compartment channel carried in the master. Susceptibility is unsupported (its
    off-resonance phase is nonlinear in position — see docs/replay_compression.md)."""
    from . import compression as _cx
    if susceptibility is not None:
        raise NotImplementedError(
            "susceptibility replay from a compressed master is unsupported: the off-resonance "
            "phase samples a field nonlinearly in position and does not commute with the basis. "
            "Replay susceptibility from a raw walk (see docs/replay_compression.md).")

    K = int(master["K"]); n_t = int(master["n_t"]); dt_traj = float(master["dt_traj"])
    _cx.require_position_method(master.get("method", "bridge_dst"))
    pos_modes = np.asarray(master["pos_modes"], np.float64)          # (N, K+2, 3)
    meta = {"method": master.get("method", "bridge_dst"), "K": K, "n_t": n_t}

    # ── Gradient phase in mode space (no trajectory reconstruction) ──────────────
    n_meas = np.asarray(G).shape[0]
    phi = _cx.mode_space_phi(_cx.pack_position_arrays(pos_modes, np.float64),
                             meta, G, dt_traj, dt_wf=dt_wf).T                # (n_meas, N), exact in time

    # ── the coherence gate on the save grid, without resampling: averaged over each save's accumulation interval
    ungated = chi_perp is None
    ones_r = np.ones((1, n_t))
    chi_r = ones_r if ungated else bin_gate(chi_perp, dt_wf, n_t, dt_traj)

    # ── Relaxation / surface log-weights ─────────────────────────────────────────
    log_w_scalar = np.zeros(n_meas); log_w_pw = np.zeros((n_meas, 1))

    def _inv_rate_at_step(inv_arr, comp):
        comp = np.asarray(comp)
        if np.issubdtype(comp.dtype, np.floating):
            f = comp.astype(np.float64)
            return inv_arr[0] + f * (inv_arr[1] - inv_arr[0])
        return inv_arr[comp]

    if T2_per_comp is not None:
        comp = comp_traj if comp_traj is not None else master.get("comp_traj")
        if comp is None:
            raise ValueError("comp_traj required when T2_per_comp is provided.")
        inv = _inv_rate_at_step(1.0 / np.asarray(T2_per_comp, np.float64), comp)
        log_w_pw = log_w_pw - dt_traj * (chi_r @ inv.T)
    elif T2 is not None:
        log_w_scalar -= (dt_traj / T2) * chi_r.sum(1)

    if T1_per_comp is not None:
        comp = comp_traj if comp_traj is not None else master.get("comp_traj")
        if comp is None:
            raise ValueError("comp_traj required when T1_per_comp is provided.")
        inv = _inv_rate_at_step(1.0 / np.asarray(T1_per_comp, np.float64), comp)
        log_w_pw = log_w_pw - dt_traj * ((ones_r - chi_r) @ inv.T)
    elif T1 is not None:
        log_w_scalar -= (dt_traj / T1) * (ones_r.sum() - chi_r.sum(1))

    if surface_relaxivity is not None:
        if D is None:
            raise ValueError("D must be provided when surface_relaxivity is not None.")
        if "blt_endpoint" not in master:
            raise ValueError("compressed master lacks the boundary channel "
                             "(re-run simulate_trajectories with tiers=\"all\").")
        if ungated:
            # (ρ/D)·Σ_t ℓ_t = (ρ/D)·B(T) — the stored endpoint; no reconstruction.
            surf = (surface_relaxivity / D) * np.asarray(master["blt_endpoint"], np.float64)
            log_w_pw = log_w_pw + surf[None, :]
        else:
            # chi-gated surface needs per-save ℓ(t): reconstruct the (N, n_t) boundary
            # channel (ONE channel — 1/3 the positions; a mode-space contraction that
            # avoids this is a follow-up). Positions are still never reconstructed.
            blt = _cx.decode_boundary_bridge(
                {"blt_bridge_dst": master["blt_modes"], "blt_start": master["blt_start"],
                 "blt_endpoint": master["blt_endpoint"]},
                {"n_t": n_t, "K": K}).astype(np.float64)
            log_w_pw = log_w_pw + (surface_relaxivity / D) * (chi_r @ blt.T)

    log_w_total = log_w_scalar[:, None] + log_w_pw                  # (n_meas, N) or (n_meas, 1)
    signals = np.mean(np.exp(log_w_total) * np.cos(phi), axis=1)
    if stimulated_echo:
        signals = signals * 0.5
    if return_walker_signals:
        return phi, log_w_total, signals
    return signals


def replay(
    trajectory,
    dt_traj: float,
    G,
    dt_wf: float,
    *,
    chi_perp=None,
    dlog_boundary_unit=None,
    T2=None,
    T1=None,
    surface_relaxivity=None,
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

    The core replay operator: with no relaxation arguments this is a pure
    gradient replay (``chi_perp`` defaults to all-ones), and it extends that with
    chi_perp-gated T2/T1 decay and surface relaxivity (``surface_relaxivity``)
    replay over pre-saved boundary hit data.

    Susceptibility off-resonance (opt-in) is replayed by sampling a public
    :mod:`dmipy_sim.fields.susceptibility` provider's ``delta_bz_fn(r)`` along the stored
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
                   + (surface_relaxivity/D) · Σ_t chi[m,t] · dlog_bnd_unit[w,t]
                                                     (surface_relaxivity term)
        E[m] = mean_w( exp(log_w[m,w]) · cos(phi[m,w]) )

    The T2 and T1 terms are walker-independent for single-compartment geometries
    and are computed as scalar multipliers per measurement.  With
    ``T2_per_comp``/``T1_per_comp`` they become walker-dependent (via
    ``comp_traj``).  The surface_relaxivity term is walker-dependent and requires
    ``dlog_boundary_unit``.

    Parameters
    ----------
    trajectory : np.ndarray, shape (n_walkers, n_t_traj, 3)
        Walker positions in metres.  float16 or float32.
    dt_traj : float
        Trajectory time step in seconds.
    G : np.ndarray, shape (n_meas, n_t_wf, 3)
        Gradient waveform in T/m.
    dt_wf : float
        ScannerSequence time step in seconds.
    chi_perp : np.ndarray, shape (n_t_wf,) or (n_meas, n_t_wf), or None
        Transverse gating schedule: 1 during encoding/decoding (T2 active),
        0 during mixing time (T1 active, T2 suspended).  Resampled to the
        trajectory time grid internally.  1D input is broadcast over all
        measurements; 2D input (n_meas, n_t_wf) allows per-measurement chi.
        ``None`` (default) uses all-ones (no gating) — a pure gradient replay.
    dlog_boundary_unit : np.ndarray, shape (n_walkers, n_t_traj), or None
        Per-step accumulated boundary log-weight with surface_relaxivity/D = 1, as
        returned by simulate_trajectories().  Required when
        surface_relaxivity is not None.  Non-positive (boundary hits reduce signal).
    T2 : float or None
        Transverse relaxation time constant in seconds.
    T1 : float or None
        Longitudinal relaxation time constant in seconds.
    surface_relaxivity : float or None
        Surface relaxivity in m/s.  Requires D and dlog_boundary_unit.
    D : float or None
        Diffusion coefficient in m²/s.  Required when surface_relaxivity is not None.
    stimulated_echo : bool
        If True, multiply signal by 0.5 for PGSTE cos(phi1) storage factor.
    comp_traj : np.ndarray, shape (n_walkers, n_t_traj), or None
        Per-walker compartment ID at each trajectory time step, as returned by
        simulate_trajectories().  Required when
        T2_per_comp or T1_per_comp is provided.  An integer array indexes the
        per-compartment arrays directly; a float array is the fractional
        occupancy of pool 1 (the enclosed pool) in a 2-compartment permeable geometry.
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
    suited to linear interpolation.  G is resampled linearly.

    Compressed master: if ``trajectory`` is the dict returned by
    ``simulate_trajectories(compress=K)`` (IR-basis modes), the gradient phase is
    replayed in mode space (never reconstructing positions) and the surface/relaxation
    weights come from the stored boundary/compartment channels — see
    :func:`_replay_compressed`.  ``dt_traj`` is then taken from the master.
    """
    if isinstance(trajectory, dict) and trajectory.get("compressed"):
        return _replay_compressed(
            trajectory, G, dt_wf, chi_perp=chi_perp, T2=T2, T1=T1,
            surface_relaxivity=surface_relaxivity, D=D, stimulated_echo=stimulated_echo,
            comp_traj=comp_traj, T2_per_comp=T2_per_comp, T1_per_comp=T1_per_comp,
            return_walker_signals=return_walker_signals, susceptibility=susceptibility,
            eps_P=eps_P)

    G = np.asarray(G, dtype=np.float64)
    n_meas, n_t_wf, _ = G.shape
    n_walkers, n_t_traj, _ = trajectory.shape

    if chi_perp is None:
        chi_perp = np.ones(n_t_wf)          # no gating → pure gradient replay
    chi_perp = np.asarray(chi_perp, dtype=np.float64)

    per_meas_chi = chi_perp.ndim == 2  # (n_meas, n_t_wf) vs (n_t_wf,)

    G_traj = effective_gradient(G, dt_wf, n_t_traj, dt_traj)         # exact per-save weights (n_meas, n_t_traj, 3)

    # ── the coherence gate on the save grid, without resampling: a save's contact and occupancy are accumulated
    # over the step ending at it, so the gate is averaged over that interval (bin_gate); the sampled field below
    # reads the gate through the path interpolant (gate_weights)
    chi_r = bin_gate(chi_perp, dt_wf, n_t_traj, dt_traj)              # (n_meas | 1, n_t_traj)
    ones_r = np.ones((1, n_t_traj))

    phi = gradient_phase(G_traj, trajectory, dt_traj)                 # (n_meas, n_walkers)

    # ── T2/T1 log-weights — scalar (walker-independent) and per-walker ────────
    log_w_scalar = np.zeros(n_meas, dtype=np.float64)          # (n_meas,)
    log_w_per_walker = np.zeros((n_meas, 1), dtype=np.float64) # (n_meas, 1) or (n_meas, n_walkers)

    def _inv_rate_at_step(inv_arr, comp):
        """Per-(walker, step) inverse-relaxation rate from comp_traj.

        Discrete (integer) comp_traj indexes ``inv_arr`` directly.  Fractional
        (float) comp_traj is the time-fraction in pool 1 (the enclosed pool) of a
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
        chi_inv_r = ones_r - chi_r  # the longitudinal periods (n_meas, n_t_traj) or (1, n_t_traj)
        log_w_t1 = -dt_traj * (chi_inv_r @ inv_T1_at_step.T)   # (n_meas, n_walkers)
        log_w_per_walker = log_w_per_walker + log_w_t1
    elif T1 is not None:
        log_w_scalar -= (dt_traj / T1) * (ones_r.sum() - chi_r.sum(axis=1))  # (n_meas,)

    # ── Surface relaxivity walker-dependent log-weight ────────────────────────
    # log_w_surf[m,w] = (surface_relaxivity/D) * chi_r[m,:] @ dlog_bnd_unit[w,:].T
    if surface_relaxivity is not None:
        if D is None:
            raise ValueError(
                "D (diffusivity) must be provided when surface_relaxivity is not None.")
        if dlog_boundary_unit is None:
            raise ValueError(
                "dlog_boundary_unit must be provided when surface_relaxivity is not None.  "
                "Re-run simulate_trajectories with tiers=\"all\".")
        dlog_bnd = np.asarray(dlog_boundary_unit, dtype=np.float64)  # (n_walkers, n_t_traj)
        log_w_surf = (surface_relaxivity / D) * (chi_r @ dlog_bnd.T) # (n_meas, n_walkers)
        log_w_per_walker = log_w_per_walker + log_w_surf

    log_w_total = log_w_scalar[:, np.newaxis] + log_w_per_walker  # (n_meas, n_walkers) or (n_meas, 1)

    # ── Susceptibility off-resonance phase (opt-in, provider-driven) ────────────
    # phi_susc[m,w] = γ · dt · Σ_t ε_P[m,t] · ΔBz(r[w,t]).  The 180° refocusing of the
    # static field is handled by the ε_P sign flip at TE/2 (see pathway_sign_se); with
    # eps_P=None it defaults to chi_r (FID / gradient-echo: no sign flip, so any static
    # field dephases for the full readout — WARNING: do not use for a spin echo).
    if susceptibility is not None:
        field_fn = _resolve_field_fn(susceptibility)
        dB = _sample_delta_bz(field_fn, trajectory)             # (n_walkers, n_t_traj)
        if eps_P is not None:
            eps_r = gate_weights(eps_P, dt_wf, n_t_traj, dt_traj)   # (n_meas | 1, n_t_traj)
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
    """Gradient phase accumulated before an instant: ``phi_pre[m, w] = gamma int_0^{t_c} G . r dt`` with
    ``t_c = cutoff_wf_idx * dt_wf`` (the 180's index on the waveform grid), exact for the piecewise-linear path
    (the last piece ends at ``t_c`` wherever it falls between saves). Feed the result to
    :func:`finite_180_longitudinal_dwell`. Susceptibility off-resonance adds to this azimuth separately."""
    traj = np.asarray(trajectories, np.float64)
    n_w, n_t, _ = traj.shape
    T = (n_t - 1) * float(dt_traj)
    t_c = float(np.clip(cutoff_wf_idx * float(dt_wf), 0.0, T))
    saves = np.arange(n_t) * float(dt_traj)
    edges = np.append(saves[saves < t_c * (1.0 - 1e-12)], t_c)
    if edges.size < 2:
        return np.zeros((np.asarray(G).shape[0], n_w))
    w_lo, w_hi, k = piece_phase_weights(G, dt_wf, edges, n_t, dt_traj)          # (n_meas, n_p, 3)
    return (np.einsum("mpd,wpd->mw", w_lo, traj[:, k, :]) + np.einsum("mpd,wpd->mw", w_hi, traj[:, k + 1, :]))


_BLOCH_PHASE_TABLE_BYTES = 512 * 2 ** 20      # device budget for one batch of per-piece phase tables


class _BlochTerms(NamedTuple):
    """The operator of a vector-Bloch replay per PIECE of the walk -- the save grid cut at every RF instant -- in
    the order the propagator applies them: the RF rotations that open the piece, free precession over it,
    transverse wall attenuation, relaxation."""
    k: np.ndarray               # (n_p,) save interval each piece lies in
    w_lo: np.ndarray            # (n_meas, n_p, 3) precession weight on r[k] (gamma included)
    w_hi: np.ndarray            # (n_meas, n_p, 3) precession weight on r[k + 1]
    flips: np.ndarray           # (n_p, n_ev) RF rotation opening each piece, per event slot, rad (0 = none)
    axes: np.ndarray            # (n_p, n_ev) B1 phase of each slot, rad
    b1: object                  # transmit scale: float, or (n_w,) per walker
    dphi_extra: object          # (n_p, n_w) precession beyond the gradient over the piece, or None
    surf: object                # (n_p, n_w) transverse wall factor over the piece, or None
    E2: np.ndarray              # (n_p, n_w) or (n_p, 1) transverse relaxation factor over the piece
    E1: np.ndarray              # (n_p, n_w) or (n_p, 1) longitudinal relaxation factor over the piece
    weights: object             # (n_w,) normalised ensemble weights, or None (plain mean)
    echo_pieces: object         # tuple of piece indices after which the echoes are read, or None


def _bloch_timeline(rf_events, n_t, dt_traj):
    """Cut the walk at the saves and at every RF instant. Returns the edges (sorted, ``(n_p + 1,)``), the
    rotations opening each piece ``[(piece, flip, axis), ...]`` and the pulse windows
    ``[(t0, t1, carrier rad/s), ...]`` over which a carrier offset and slice-select off-resonance act."""
    T = (int(n_t) - 1) * float(dt_traj)
    tol = 1e-9 * float(dt_traj)
    saves = np.arange(int(n_t)) * float(dt_traj)
    rots, windows, instants = [], [], []
    for e in RFSchedule(rf_events):
        t_s, dur = e.t_s, e.duration_s
        if t_s < -tol or t_s > T + tol:
            raise ValueError(f"RF event at t = {t_s:.6g} s lies outside the walk [0, {T:.6g}] s")
        nsub = max(1, int(round(dur / float(dt_traj)))) if dur > 0.0 else 1
        ts = (np.clip(t_s - dur / 2.0 + (np.arange(nsub) + 0.5) * dur / nsub, 0.0, T) if dur > 0.0
              else np.array([t_s]))
        dflips, axes = e.flip_split(nsub)                                  # even, or by the pulse's envelope
        for t, f, ax in zip(ts, dflips, axes):
            rots.append((float(t), float(f), float(ax)))
            instants.append(float(t))
        windows.append((max(0.0, t_s - dur / 2.0), min(T, t_s + dur / 2.0), 2.0 * np.pi * e.offset_hz))
    times = np.sort(np.concatenate([saves, np.asarray(instants, np.float64)]))
    edges = [times[0]]
    for t in times[1:]:
        if t - edges[-1] > tol:
            edges.append(t)
    edges = np.asarray(edges)
    if rots and any(abs(t - edges[-1]) <= tol for t, _, _ in rots):
        edges = np.append(edges, edges[-1])                                  # a rotation at T opens an empty piece
    def piece_of(t):
        i = int(np.argmin(np.abs(edges[:-1] - t)))
        return i
    rotations = [(piece_of(t), f, ax) for t, f, ax in rots]
    return edges, rotations, windows


def _bloch_replay_terms(trajectory, dt_traj, G, dt_wf, rf_events, *, T2, T1, comp_traj,
                        T2_per_comp, T1_per_comp, susceptibility, extra_phase_per_step,
                        dlog_boundary_unit, surface_relaxivity, D, b1_scale, slice_offsets,
                        slice_gradient, bound_frac, T2_bound, T1_bound, off_resonance_bound,
                        weights, echo_steps=None):
    """Resolve the sequence and substrate inputs of :func:`replay_bloch` into per-piece terms.

    Both Bloch replays consume this; the numpy loop and the JAX scan then differ only in how they iterate the
    identical operator, which is what makes the numpy one the reference.

    The walk is cut at the saves and at every RF instant (:func:`_bloch_timeline`). Over each piece the gradient
    precession is exact for the piecewise-linear path (:func:`_replay_kernel.piece_phase_weights`), relaxation
    runs for the piece's duration, and the per-save wall contact and off-resonance are shared by duration. A
    hard pulse is one rotation at its instant; a finite pulse of ``duration_s`` is ``round(duration/dt)``
    sub-rotations spread over its window with the flip split evenly (or by its envelope); its carrier
    ``offset_hz`` and, with ``slice_offsets``/``slice_gradient``, the slice-select off-resonance act over the
    window (a hard pulse has no window). An MT ``bound_frac`` blends the relaxation rates toward the bound
    pool's per save and adds the bound pool's off-resonance precession by occupancy.
    """
    G = np.asarray(G, np.float64)
    n_meas = G.shape[0]
    n_w, n_t, _ = trajectory.shape
    dt = float(dt_traj)
    edges, rotations, windows = _bloch_timeline(rf_events, n_t, dt)
    w_lo, w_hi, k = piece_phase_weights(G, dt_wf, edges, n_t, dt)              # (n_meas, n_p, 3), (n_p,)
    n_p = k.size
    ell = np.diff(edges)                                                      # (n_p,) piece durations
    frac = ell / dt

    def rate(per_comp, scalar):
        if per_comp is not None:
            if comp_traj is None:
                raise ValueError("comp_traj is required with per-compartment relaxation.")
            return (1.0 / np.asarray(per_comp, np.float64))[np.asarray(comp_traj)].T    # (n_t, n_w)
        return np.full((n_t, 1), 0.0 if scalar is None else 1.0 / float(scalar))
    invT2 = rate(T2_per_comp, T2)
    invT1 = rate(T1_per_comp, T1)
    # a save's occupancy, contact and bound fraction are accumulated over the step ENDING at it, so the piece in
    # save interval [t_k, t_{k+1}] reads them at k + 1; a sampled quantity (a field at the saves) is read through
    # the path interpolant with the piece's unit moments (lo on save k, hi on save k + 1)
    k1 = np.minimum(k + 1, n_t - 1)
    a_rel, b_rel = edges[:-1] - k * dt, edges[1:] - k * dt
    u_hi = (b_rel ** 2 - a_rel ** 2) / (2.0 * dt)                            # int (t - t_k) dt / dt
    u_lo = ell - u_hi

    dphi_extra = None
    def add(term):
        nonlocal dphi_extra
        dphi_extra = term if dphi_extra is None else dphi_extra + term

    if bound_frac is not None:
        if T2_bound is None or T1_bound is None:
            raise ValueError("T2_bound and T1_bound are required when bound_frac is set.")
        bf = np.asarray(bound_frac, np.float64).T                          # (n_t, n_w)
        invT2 = (1.0 - bf) * invT2 + bf * (1.0 / float(T2_bound))
        invT1 = (1.0 - bf) * invT1 + bf * (1.0 / float(T1_bound))
        if float(off_resonance_bound) != 0.0:
            add(bf[k1] * (2.0 * np.pi * float(off_resonance_bound) * ell)[:, None])
    E2 = np.exp(-ell[:, None] * invT2[k1])                                    # (n_p, n_w | 1)
    E1 = np.exp(-ell[:, None] * invT1[k1])

    if susceptibility is not None and extra_phase_per_step is None:
        extra_phase_per_step = GAMMA * dt * _sample_delta_bz(_resolve_field_fn(susceptibility), trajectory)
    if extra_phase_per_step is not None:
        x = np.asarray(extra_phase_per_step, np.float64).T / dt                  # (n_t, n_w): the sampled rate
        add(u_lo[:, None] * x[k] + u_hi[:, None] * x[k1])

    surf = None
    if (surface_relaxivity is not None and dlog_boundary_unit is not None
            and float(surface_relaxivity) != 0.0):
        if D is None:
            raise ValueError("D required when surface_relaxivity is not None.")
        surf = np.exp((float(surface_relaxivity) / float(D)) * np.asarray(dlog_boundary_unit, np.float64).T[k1]
                      * frac[:, None])

    slots = {}
    for pi, f, ax in rotations:
        slots.setdefault(pi, []).append((f, ax))
    n_ev = max([len(v) for v in slots.values()] + [1])
    flips = np.zeros((n_p, n_ev)); axes = np.zeros((n_p, n_ev))
    for pi, lst in slots.items():
        for j, (f, a) in enumerate(lst):
            flips[pi, j], axes[pi, j] = f, a
    if windows:
        overlap = np.zeros(n_p); carrier = np.zeros(n_p)
        for t0, t1, w_off in windows:
            o = np.clip(np.minimum(edges[1:], t1) - np.maximum(edges[:-1], t0), 0.0, None)
            overlap += o; carrier += w_off * o
        if np.any(carrier != 0.0):
            add(carrier[:, None])
        if slice_offsets is not None and float(slice_gradient) != 0.0:
            add(overlap[:, None] * (GAMMA * float(slice_gradient) * np.asarray(slice_offsets, np.float64))[None, :])

    b1 = 1.0 if b1_scale is None else np.asarray(b1_scale, np.float64)
    wn = None
    if weights is not None:
        wn = np.asarray(weights, np.float64)
        wn = wn / wn.sum()
    echo_pieces = None
    if echo_steps is not None:
        ends = edges[1:]
        echo_pieces = tuple(int(np.argmin(np.abs(ends - int(e) * dt))) for e in echo_steps)
    return _BlochTerms(k, w_lo, w_hi, flips, axes, b1, dphi_extra, surf, E2, E1, wn, echo_pieces)


def _bloch_replay_output(sig_last, M_last, echo_out, echo_steps, return_walker_signals):
    if echo_steps is not None:
        return echo_out
    if return_walker_signals:
        return M_last, sig_last
    return sig_last


def replay_bloch(trajectory, dt_traj, G, dt_wf, rf_events, *,
                 T2=None, T1=None, comp_traj=None,
                 T2_per_comp=None, T1_per_comp=None,
                 susceptibility=None, extra_phase_per_step=None,
                 dlog_boundary_unit=None, surface_relaxivity=None, D=None,
                 echo_steps=None, echo_per_walker=False,
                 weights=None, return_walker_signals=False,
                 b1_scale=None, slice_offsets=None, slice_gradient=0.0,
                 bound_frac=None, T2_bound=None, T1_bound=None,
                 off_resonance_bound=0.0):
    """Emergent per-walker vector-Bloch replay on the stored (field-independent) walk.

    Propagates each walker's magnetisation ``M = (Mx, My, Mz)`` through the ACTUAL sequence operators -- RF
    rotations from ``rf_events``, the physical gradient precession ``gamma int G(t) . r_w(t) dt``, per-comp
    T2/T1, surface relaxivity, an optional susceptibility off-resonance field, and an optional MT bound-pool
    blend -- piece by piece, the walk cut at every save and every RF instant, so the phase accrued before a
    pulse is flipped by it exactly wherever the pulse falls (:func:`_bloch_replay_terms`). The forward-engine
    counterpart is :func:`dmipy_sim.engine.bloch.simulate_bloch`.

    This is the numpy REFERENCE: a Python loop over measurements and pieces written so each operator is one
    readable line. :func:`replay_bloch_jax` is the same operator as a jitted scan and takes the same arguments.

    Coherence pathways / refocusing are EMERGENT: the 180 conjugates the accumulated phase, so the spin echo
    forms by itself. **Pass the PHYSICAL (same-sign-lobe) gradient** (``ScannerSequence.G``, on its own grid).

    Parameters
    ----------
    rf_events : list of dict
        Each ``{'t_s', 'flip_deg', 'axis_deg', 'duration_s', 'offset_hz'}``; ``axis_deg`` is the B1 phase
        (0 = x, 90 = y); ``duration_s = 0`` is an instantaneous hard pulse at ``t_s`` (any instant, not a save);
        ``offset_hz`` an off-resonance carrier over a finite pulse.
    T2, T1 / T2_per_comp, T1_per_comp, comp_traj : as in :func:`replay`.
    susceptibility : a :mod:`dmipy_sim.fields.susceptibility` provider or ``r -> dBz`` callable, sampled along
        the walk; mutually exclusive with ``extra_phase_per_step`` (the pre-baked per-save increment).
    bound_frac, T2_bound, T1_bound, off_resonance_bound : the MT bound-pool blend per save.
    b1_scale : transmit (B1+) scale of every flip; per walker for an inhomogeneous field.
    slice_offsets, slice_gradient : slice-select off-resonance ``gamma G_slice z_w`` during finite pulses.
    weights : (n_w,) ensemble weights of the walker mean.
    echo_steps, echo_per_walker : record ``Mxy`` at these SAVE indices, per walker if asked.

    Returns
    -------
    signals : (n_meas,) complex walker-mean ``Mx + i My`` at the end, or ``(n_meas, n_echo)`` (``(n_meas,
        n_echo, n_w)`` with ``echo_per_walker``) when ``echo_steps`` is given, or ``(M_final, signals)`` with the
        per-walker (3, n_w) final magnetisation of the last measurement when ``return_walker_signals``.
    """
    tm = _bloch_replay_terms(trajectory, dt_traj, G, dt_wf, rf_events, T2=T2, T1=T1,
                             comp_traj=comp_traj, T2_per_comp=T2_per_comp, T1_per_comp=T1_per_comp,
                             susceptibility=susceptibility, extra_phase_per_step=extra_phase_per_step,
                             dlog_boundary_unit=dlog_boundary_unit,
                             surface_relaxivity=surface_relaxivity, D=D, b1_scale=b1_scale,
                             slice_offsets=slice_offsets, slice_gradient=slice_gradient,
                             bound_frac=bound_frac, T2_bound=T2_bound, T1_bound=T1_bound,
                             off_resonance_bound=off_resonance_bound, weights=weights, echo_steps=echo_steps)
    traj = np.asarray(trajectory, np.float64)
    n_meas = tm.w_lo.shape[0]
    n_w = traj.shape[0]
    n_p = tm.k.size
    wmean = (lambda v: np.mean(v)) if tm.weights is None else (lambda v: (v * tm.weights).sum())
    echo_set = None if tm.echo_pieces is None else set(tm.echo_pieces)
    signals = np.empty(n_meas, np.complex128)
    echo_out = None
    M = None
    for m in range(n_meas):
        M = np.zeros((3, n_w))
        M[2] = 1.0                                              # equilibrium along +z
        rec = []
        for p in range(n_p):
            for f, a in zip(tm.flips[p], tm.axes[p]):
                if f != 0.0:
                    M = _rf_increment(M, f * tm.b1, a)
            kp = tm.k[p]
            dphi = traj[:, kp, :] @ tm.w_lo[m, p] + traj[:, kp + 1, :] @ tm.w_hi[m, p]     # (n_w,)
            if tm.dphi_extra is not None:
                dphi = dphi + tm.dphi_extra[p]
            c, s_ = np.cos(dphi), np.sin(dphi)
            Mx = c * M[0] - s_ * M[1]
            My = s_ * M[0] + c * M[1]
            if tm.surf is not None:
                Mx, My = Mx * tm.surf[p], My * tm.surf[p]
            M = np.stack([Mx * tm.E2[p], My * tm.E2[p], M[2] * tm.E1[p]])
            if echo_set is not None and p in echo_set:
                mxy = M[0] + 1j * M[1]
                for _ in range(tm.echo_pieces.count(p)):
                    rec.append(mxy.copy() if echo_per_walker else wmean(mxy))
        signals[m] = wmean(M[0] + 1j * M[1])
        if echo_set is not None:
            if echo_out is None:
                echo_out = np.empty((n_meas, len(rec)) + ((n_w,) if echo_per_walker else ()), np.complex128)
            echo_out[m] = np.asarray(rec)
    return _bloch_replay_output(signals, M, echo_out, echo_steps, return_walker_signals)


def replay_bloch_jax(trajectory, dt_traj, G, dt_wf, rf_events, *,
                     T2=None, T1=None, comp_traj=None,
                     T2_per_comp=None, T1_per_comp=None,
                     susceptibility=None, extra_phase_per_step=None,
                     dlog_boundary_unit=None, surface_relaxivity=None, D=None,
                     echo_steps=None, echo_per_walker=False,
                     weights=None, return_walker_signals=False,
                     b1_scale=None, slice_offsets=None, slice_gradient=0.0,
                     bound_frac=None, T2_bound=None, T1_bound=None,
                     off_resonance_bound=0.0, phase_table_bytes=_BLOCH_PHASE_TABLE_BYTES):
    """:func:`replay_bloch` as a jitted ``lax.scan`` over the pieces of the walk, vectorised over measurements.

    Same arguments, same per-piece operator and same outputs as the numpy reference. Arithmetic is float32 on
    the device, so signals agree with the reference to about 1e-5 of the phase scale. Each measurement needs
    an ``(n_pieces, n_w)`` float32 phase table on the device; measurements are batched so that the tables of one
    batch stay within ``phase_table_bytes``, and a single measurement whose table exceeds it runs alone.
    """
    if not _JAX_AVAILABLE:
        raise RuntimeError("JAX not available; use replay_bloch.")
    tm = _bloch_replay_terms(trajectory, dt_traj, G, dt_wf, rf_events, T2=T2, T1=T1,
                             comp_traj=comp_traj, T2_per_comp=T2_per_comp, T1_per_comp=T1_per_comp,
                             susceptibility=susceptibility, extra_phase_per_step=extra_phase_per_step,
                             dlog_boundary_unit=dlog_boundary_unit,
                             surface_relaxivity=surface_relaxivity, D=D, b1_scale=b1_scale,
                             slice_offsets=slice_offsets, slice_gradient=slice_gradient,
                             bound_frac=bound_frac, T2_bound=T2_bound, T1_bound=T1_bound,
                             off_resonance_bound=off_resonance_bound, weights=weights, echo_steps=echo_steps)
    n_w, n_t, _ = trajectory.shape
    n_p = tm.k.size
    f32 = jnp.float32
    traj = jnp.asarray(trajectory, f32)
    zeros_p = np.zeros((n_p, 1))
    arrays = (traj[:, jnp.asarray(tm.k), :], traj[:, jnp.asarray(tm.k + 1), :],           # (n_w, n_p, 3) each
              jnp.asarray(tm.flips, f32), jnp.asarray(tm.axes, f32), jnp.asarray(tm.b1, f32),
              jnp.asarray(tm.dphi_extra if tm.dphi_extra is not None else zeros_p, f32),
              jnp.asarray(tm.surf if tm.surf is not None else zeros_p + 1.0, f32),
              jnp.asarray(tm.E2, f32), jnp.asarray(tm.E1, f32),
              jnp.asarray(tm.weights if tm.weights is not None else np.full(n_w, 1.0 / n_w), f32))
    n_meas = tm.w_lo.shape[0]
    per_batch = max(1, int(phase_table_bytes // (4 * n_p * n_w)))
    W = jnp.asarray(np.stack([tm.w_lo, tm.w_hi], axis=1), f32)                        # (n_meas, 2, n_p, 3)
    parts = [_bloch_scan_batch(W[a:a + per_batch], *arrays, echo_idx=tm.echo_pieces, per_walker=bool(echo_per_walker))
             for a in range(0, n_meas, per_batch)]
    M_last = parts[-1][0][-1]
    rec = np.concatenate([np.asarray(p[1], np.complex128) for p in parts])
    last = np.concatenate([np.asarray(p[2], np.complex128) for p in parts])
    echo_out = None if tm.echo_pieces is None else rec
    return _bloch_replay_output(last, np.asarray(M_last, np.float64), echo_out,
                                echo_steps, return_walker_signals)


def _bloch_rotate(M, flip, ax):
    """:func:`_rf_increment` on traced operands."""
    ux, uy = jnp.cos(ax), jnp.sin(ax)
    c, s = jnp.cos(flip), jnp.sin(flip)
    omc = 1.0 - c
    Mx, My, Mz = M[0], M[1], M[2]
    return jnp.stack([(c + ux * ux * omc) * Mx + (ux * uy * omc) * My + (uy * s) * Mz,
                      (ux * uy * omc) * Mx + (c + uy * uy * omc) * My + (-ux * s) * Mz,
                      (-uy * s) * Mx + (ux * s) * My + c * Mz])


if _JAX_AVAILABLE:
    @functools.partial(jax.jit, static_argnames=("echo_idx", "per_walker"))
    def _bloch_scan_batch(W_b, r_lo, r_hi, flips, axes, b1, dphi_extra, surf, E2, E1, wn, *, echo_idx, per_walker):
        """One batch of measurements of :func:`replay_bloch_jax`: ``vmap`` over ``W_b`` ``(n_b, 2, n_p, 3)`` (the
        piece weights on the two bounding saves) of a ``lax.scan`` over the pieces. Module-level and jitted on
        array arguments, so the executable is reused by every call of the same shapes."""
        n_w = r_lo.shape[0]
        n_ev = flips.shape[1]

        def readout(M):
            mxy = M[0] + 1j * M[1]
            return mxy if per_walker else jnp.sum(mxy * wn)

        def run_meas(Wm):
            dphi = (jnp.einsum("pd,wpd->pw", Wm[0], r_lo, precision=jax.lax.Precision.HIGHEST)
                    + jnp.einsum("pd,wpd->pw", Wm[1], r_hi, precision=jax.lax.Precision.HIGHEST) + dphi_extra)

            def step(M, x):
                fl, ax, dphi_t, e2, e1, sf = x
                for j in range(n_ev):
                    M = _bloch_rotate(M, fl[j] * b1, ax[j])
                c, s = jnp.cos(dphi_t), jnp.sin(dphi_t)
                Mx = (c * M[0] - s * M[1]) * sf * e2
                My = (s * M[0] + c * M[1]) * sf * e2
                M = jnp.stack([Mx, My, M[2] * e1])
                return M, readout(M)
            M0 = jnp.stack([jnp.zeros(n_w, r_lo.dtype), jnp.zeros(n_w, r_lo.dtype), jnp.ones(n_w, r_lo.dtype)])
            M_last, out_t = jax.lax.scan(step, M0, (flips, axes, dphi, E2, E1, surf))
            rec = out_t[-1] if echo_idx is None else out_t[jnp.asarray(echo_idx)]
            last = jnp.sum(out_t[-1] * wn) if per_walker else out_t[-1]
            return M_last, rec, last
        return jax.vmap(run_meas)(W_b)
