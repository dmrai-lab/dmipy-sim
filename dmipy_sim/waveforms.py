"""Gradient waveform representation and constructors.

Primary representation: G array of shape (n_measurements, n_t, 3), float32, in T/m.
All constructors return a Waveform dataclass.
"""

from dataclasses import dataclass
import numpy as np
import jax.numpy as jnp
from warnings import warn

from .constants import GAMMA, DEFAULT_SLEW_RATE, resolve_slew as _resolve_slew


def _fill_lobe(arr, m, i0, n_pulse, amp_vec, n_rise):
    """Fill a (possibly trapezoidal) gradient lobe of peak ``amp_vec`` over steps
    [i0, i0+n_pulse) of measurement m.  ``n_rise=0`` gives a rectangle (the square
    / instantaneous limit -- bit-identical to the pre-slew construction); n_rise>0
    adds linear ramp-up / flat-top / ramp-down (matching trapezoidal_ogse)."""
    i1 = i0 + n_pulse
    n_rise = min(int(n_rise), n_pulse // 2)
    if n_rise > 0:
        up = np.linspace(0.0, 1.0, n_rise, endpoint=False)
        arr[m, i0:i0 + n_rise, :] = up[:, None] * amp_vec
        arr[m, i0 + n_rise:i1 - n_rise, :] = amp_vec
        dn = np.linspace(1.0, 0.0, n_rise, endpoint=False)
        arr[m, i1 - n_rise:i1, :] = dn[:, None] * amp_vec
    else:
        arr[m, i0:i1, :] = amp_vec


def _lobe_n_rise(square, G_peak, slew_rate, dt, n_pulse):
    """Ramp length (steps) for a slew-limited lobe; 0 for the square limit."""
    if square or G_peak <= 0:
        return 0
    return min(max(0, round((G_peak / float(slew_rate)) / dt)), n_pulse // 2)


@dataclass
class Waveform:
    """Gradient waveform for Monte Carlo simulation.

    Attributes
    ----------
    G : jnp.ndarray
        Shape (n_measurements, n_t, 3), float32. Gradient vectors in T/m at
        each time point for each measurement.
    dt : float
        Uniform time step in seconds.
    echo_idx : int
        Index in [0, n_t) at which the signal is sampled (last time point for
        spin echo). Signal extraction uses phi at this time step.
    rf_events : list, optional
        Ideal (instantaneous) RF markers for visualisation only, each
        ``{'t_s', 'label', 'flip_deg'}``. Do not affect the simulation.
    G_display : np.ndarray, optional
        Physical scanner gradient for visualisation only (same-sign second lobe,
        as the 180° would deliver it). None → fall back to G for display. The
        simulation always uses G (bipolar convention; calc_b / physics depend
        on it).
    echo_indices : np.ndarray, optional
        Time-step indices of the successive echoes for a multi-echo train (e.g.
        CPMG). None → single echo at ``echo_idx``. :func:`dmipy_sim.simulate_cpmg`
        samples the signal at each of these indices from a single walk.

    Magnetisation is treated as fully transverse throughout (ideal
    instantaneous pulses), so there is no χ_⊥ schedule or longitudinal storage.
    """
    G: jnp.ndarray
    dt: float
    echo_idx: int
    rf_events: list = None   # [{'t_s': float, 'label': str, 'flip_deg': int}, ...]
    G_display: np.ndarray = None
    echo_indices: np.ndarray = None


def pgse(delta, DELTA, G_magnitude, bvecs, n_t,
         slew_rate=DEFAULT_SLEW_RATE):
    """Build a PGSE gradient waveform.

    Slew-limited (realizable, trapezoidal lobes) by default -- dmipy-sim is the
    forward truth.  Pass ``slew_rate=np.inf`` for the idealized instantaneous
    (square) limit (e.g. A/B against an analytic solution); ``slew_rate`` must be
    a positive T/m/s value or ``np.inf`` (``None`` is rejected).  b is set later
    via :func:`set_b`; slew-limiting changes the lobe SHAPE (hence the restricted-
    diffusion signal), which is the point.

    Parameters
    ----------
    delta : float
        Gradient pulse duration in seconds.
    DELTA : float
        Diffusion time in seconds (centre-to-centre of gradient pulses).
    G_magnitude : float or array of shape (n_measurements,)
        Gradient amplitude in T/m. Use set_b() to scale to target b-values.
    bvecs : array of shape (n_measurements, 3)
        Unit gradient direction vectors.
    n_t : int
        Number of time points. Total duration = DELTA + delta.

    Returns
    -------
    Waveform
    """
    bvecs = np.asarray(bvecs, dtype=np.float32)
    n_measurements = bvecs.shape[0]
    G_mag = np.broadcast_to(np.asarray(G_magnitude, dtype=np.float32),
                             (n_measurements,))

    T_total = DELTA + delta
    dt = T_total / (n_t - 1)

    # Use integer-based indexing (like disimpy) to avoid floating-point boundary issues.
    # Time-based masks with strict < can exclude boundary points for small delta.
    n_pulse = max(1, round(delta / dt))
    n_DELTA = round(DELTA / dt)

    # Gradient-only waveform (n_t steps, fully transverse throughout for PGSE).
    # Simulation uses bipolar convention: second lobe is -G so that the phase
    # integral refocuses at TE without explicitly modeling the 180° spin flip.
    # G_display uses same-sign second lobe (physical scanner convention).
    _, square = _resolve_slew(slew_rate)
    G_peak = float(np.max(np.abs(G_mag)))
    n_rise = _lobe_n_rise(square, G_peak, slew_rate, dt, n_pulse)

    G_grad = np.zeros((n_measurements, n_t, 3), dtype=np.float32)
    G_disp = np.zeros((n_measurements, n_t, 3), dtype=np.float32)
    for m in range(n_measurements):
        gpos = G_mag[m] * bvecs[m]
        _fill_lobe(G_grad, m, 0, n_pulse, gpos, n_rise)
        _fill_lobe(G_grad, m, n_DELTA, n_pulse, -gpos, n_rise)
        _fill_lobe(G_disp, m, 0, n_pulse, gpos, n_rise)
        _fill_lobe(G_disp, m, n_DELTA, n_pulse, gpos, n_rise)

    # Ideal instantaneous 90°/180° markers (visualisation only): the 180° sits
    # midway between the two gradient lobes.  The bipolar G already encodes the
    # refocusing, so no χ_⊥ schedule or free-precession tail is needed.
    gap_mid = (n_pulse + n_DELTA) // 2
    rf_events = [
        {'t_s': 0.0,           'label': 'Mz→Mxy', 'flip_deg': 90},
        {'t_s': gap_mid * dt,  'label': 'refocus', 'flip_deg': 180},
    ]

    return Waveform(G=jnp.array(G_grad), dt=float(dt),
                    echo_idx=n_t - 1,
                    rf_events=rf_events,
                    G_display=G_disp)


def ogse(frequency, T_total, G_magnitude, bvecs, n_t, kind='cosine'):
    """Build an OGSE (oscillating gradient spin echo) waveform.

    Two equal blocks straddle the 180 at T_total/2; the second block's sign is
    flipped (spin-echo refocusing). To make a *valid, frequency-selective* OGSE
    the oscillation must contain an INTEGER number of periods in each half-echo,
    so that (i) the gradient moment nulls at the echo and (ii) the cosine form is
    DC-free (its gradient power spectrum is peaked at ``f``). The requested
    ``frequency`` is therefore snapped to the nearest such value,
    ``f_eff = N/(T_total/2)`` with ``N = round(frequency*T_total/2)``; a warning is
    issued when the snap exceeds 10%.

    ``kind``:
      ``'cosine'`` (default) -- ``G(t)=cos(2*pi*f*t)``; ``q=int G`` is zero-mean,
        the encoding spectrum peaks sharply at ``f`` (the standard frequency-
        selective OGSE; suppresses low-frequency / static-dephasing terms).
      ``'sine'`` -- ``G(t)=sin(2*pi*f*t)``; ``q`` is single-signed and carries a
        DC component, so it is only quasi-frequency-selective (provided for
        completeness; ``trapezoidal_ogse`` is the hardware-realisable cosine).

    Parameters
    ----------
    frequency : float
        Requested OGSE frequency in Hz (snapped to integer periods per half-echo).
    T_total : float
        Total waveform duration in seconds.
    G_magnitude : float or array of shape (n_measurements,)
        Peak gradient amplitude in T/m.
    bvecs : array of shape (n_measurements, 3)
        Unit gradient direction vectors.
    n_t : int
        Number of time points.
    kind : {'cosine', 'sine'}
        Oscillation form (see above).

    Returns
    -------
    Waveform
    """
    bvecs = np.asarray(bvecs, dtype=np.float32)
    n_measurements = bvecs.shape[0]
    G_mag = np.broadcast_to(np.asarray(G_magnitude, dtype=np.float32),
                             (n_measurements,))

    dt = T_total / (n_t - 1)
    t = np.arange(n_t, dtype=np.float64) * dt

    # Snap to an integer number of periods per half-echo so the T/2 sign flip
    # lands on a full-period boundary (valid, DC-free, frequency-selective OGSE).
    half = T_total / 2.0
    N = max(1, int(round(frequency * half)))
    f_eff = N / half
    if abs(f_eff - frequency) > 0.1 * max(frequency, 1e-9):
        warn(f"ogse: frequency snapped {frequency:.1f} -> {f_eff:.1f} Hz "
             f"({N} periods per half-echo of {half*1e3:.1f} ms) for a valid OGSE")
    if kind == 'cosine':
        envelope = np.cos(2 * np.pi * f_eff * t)
    elif kind == 'sine':
        envelope = np.sin(2 * np.pi * f_eff * t)
    else:
        raise ValueError(f"ogse kind must be 'cosine' or 'sine', got {kind!r}")
    # Spin-echo refocusing: the post-180 lobe is the negated time-mirror of the
    # pre-180 lobe. This makes the gradient moment null EXACTLY at the echo
    # (q(TE)=0 to one sample), independent of frequency -- a discrete sign-flip at
    # t>=T/2 instead leaves an O(f*dt) residual because the first block falls one
    # sample short of an integer number of periods.
    mid = n_t // 2
    envelope[n_t - mid:] = -envelope[:mid][::-1]
    if n_t % 2 == 1:
        envelope[mid] = 0.0          # gradient off at the 180 instant -> exact null

    G_grad = np.zeros((n_measurements, n_t, 3), dtype=np.float32)
    for m in range(n_measurements):
        G_grad[m, :, :] = (G_mag[m] * envelope)[:, None] * bvecs[m]

    # Ideal instantaneous 90°/180° markers (visualisation only); the bipolar
    # (time-mirrored) G already refocuses at the echo.
    gap_mid = (n_t - 1) // 2
    rf_events = [
        {'t_s': 0.0,          'label': 'Mz→Mxy', 'flip_deg': 90},
        {'t_s': gap_mid * dt, 'label': 'refocus', 'flip_deg': 180},
    ]

    return Waveform(G=jnp.array(G_grad), dt=float(dt),
                    echo_idx=n_t - 1,
                    rf_events=rf_events)


def trapezoidal_ogse(N, delta, DELTA, G_magnitude, bvecs, n_t,
                     rise_time=None, slew_rate=200e3):
    """Build a trapezoidal N-lobe OGSE (oscillating gradient spin echo) waveform.

    This is the waveform family used in Drobnjak et al. MRM 2016 and MISST.
    N=1 is a PGSE with finite rise/fall ramps (trapezoidal pulses).

    Structure::

        [0,        delta)    : Block 1 — N lobes, alternating +/- G, trapezoidal
        [delta,    DELTA)    : Zero  — 180° RF pulse gap
        [DELTA,    DELTA+delta): Block 2 — N lobes, signs flipped (spin echo)

    Each lobe has duration ``delta/N``.  Within each lobe::

        ramp up (tr) → flat top (delta/N - 2*tr) → ramp down (tr)

    The b-value formula for this waveform is given by Eq. (2) of Drobnjak 2016
    and can be evaluated with :func:`b_trapezoidal_ogse`.

    Parameters
    ----------
    N : int
        Number of gradient lobes per block (N=1 → trapezoidal PGSE).
    delta : float
        Duration of each gradient block (= total lobe time) in seconds.
    DELTA : float
        Time from start of block 1 to start of block 2 in seconds.
        Typically DELTA = delta + P180 where P180 ≈ 10 ms.
    G_magnitude : float or array of shape (n_measurements,)
        Peak gradient amplitude in T/m.
    bvecs : array of shape (n_measurements, 3)
        Unit gradient direction vectors.
    n_t : int
        Number of time points.  Total duration = DELTA + delta.
    rise_time : float or None
        Gradient rise time in seconds (linear ramp each edge of every lobe).
        If None, computed from slew_rate: tr = max(G_magnitude) / slew_rate.
    slew_rate : float
        Maximum slew rate in T/m/s, used only when rise_time is None.
        Default 200e3 T/m/s (paper value).

    Returns
    -------
    Waveform
    """
    bvecs = np.asarray(bvecs, dtype=np.float32)
    n_measurements = bvecs.shape[0]
    G_mag = np.broadcast_to(np.asarray(G_magnitude, dtype=np.float32),
                             (n_measurements,))

    G_peak = float(np.max(np.abs(G_mag)))
    if rise_time is None:
        rise_time = G_peak / slew_rate if G_peak > 0 else 0.0

    T_total = DELTA + delta
    dt = T_total / (n_t - 1)

    # Steps per block (round to multiple of N for equal lobe widths).
    n_lobe = max(1, round(delta / (N * dt)))
    n_block = n_lobe * N
    n_rise = min(max(0, round(rise_time / dt)), n_lobe // 2)
    n_DELTA = round(DELTA / dt)   # start of block 2

    G = np.zeros((n_measurements, n_t, 3), dtype=np.float32)

    def _fill_block(G, m, block_start, signs):
        """Fill N lobes into G[m] starting at block_start, with per-lobe signs."""
        for k in range(N):
            s = signs[k]
            i0 = block_start + k * n_lobe
            i1 = block_start + (k + 1) * n_lobe
            if n_rise > 0:
                # ramp up
                ramp = np.linspace(0.0, s * G_mag[m], n_rise, endpoint=False)
                G[m, i0:i0 + n_rise, :] = ramp[:, None] * bvecs[m]
                # flat top
                G[m, i0 + n_rise:i1 - n_rise, :] = (s * G_mag[m]) * bvecs[m]
                # ramp down
                ramp_d = np.linspace(s * G_mag[m], 0.0, n_rise, endpoint=False)
                G[m, i1 - n_rise:i1, :] = ramp_d[:, None] * bvecs[m]
            else:
                G[m, i0:i1, :] = (s * G_mag[m]) * bvecs[m]

    signs1 = [(-1) ** k for k in range(N)]          # block 1: +, -, +, ...
    signs2 = [-((-1) ** k) for k in range(N)]        # block 2: -, +, -, ...

    G_grad = np.zeros((n_measurements, n_t, 3), dtype=np.float32)
    for m in range(n_measurements):
        _fill_block(G_grad, m, 0, signs1)
        _fill_block(G_grad, m, n_DELTA, signs2)

    # Ideal instantaneous 90°/180° markers (visualisation only); the sign-flipped
    # second block already refocuses at the echo.
    gap_mid = (n_block + n_DELTA) // 2
    rf_events = [
        {'t_s': 0.0,          'label': 'Mz→Mxy', 'flip_deg': 90},
        {'t_s': gap_mid * dt, 'label': 'refocus', 'flip_deg': 180},
    ]

    return Waveform(G=jnp.array(G_grad), dt=float(dt),
                    echo_idx=n_t - 1,
                    rf_events=rf_events)


def b_trapezoidal_ogse(N, delta, DELTA, G, rise_time):
    """Analytical b-value for trapezoidal N-lobe OGSE (Eq. 2, Drobnjak 2016).

    Parameters
    ----------
    N : int
        Number of gradient lobes per block.
    delta : float
        Gradient block duration (s).
    DELTA : float
        Block separation (s).
    G : float
        Peak gradient strength (T/m).
    rise_time : float
        Gradient ramp time per edge (s).

    Returns
    -------
    float
        b-value in s/m².
    """
    tr = rise_time
    x = tr * N / delta   # dimensionless ramp fraction
    # Eq. 2, first term: oscillation contribution
    b1 = (2 * G**2 * GAMMA**2 * delta**3 / (15 * N**2) *
          (5 - (15/2)*x - (5/4)*x**2 + 4*x**3))
    # Eq. 2, second term: cross-block correlation (non-zero only for odd N)
    q_net = (1 - (-1)**N) * (delta - N * tr) / (2 * N)
    b2 = G**2 * GAMMA**2 * (DELTA - delta) * q_net**2
    return float(b1 + b2)


def calc_b(waveform):
    """Compute b-values for each measurement in a Waveform (s/m²).

    Uses the rectangular (left-point) rule to accumulate q(t), then
    integrates |q(t)|² with the trapezoidal rule.  This is internally
    consistent with the simulation's phase accumulation
    (physics.py: dphi = GAMMA*dt * dot(G[t], r_new)), which is also
    a rectangular rule evaluated at the new position.

    Note: disimpy's gradients.calc_b() uses a trapezoidal rule for q(t)
    accumulation (gradients.py:60-66).  The two agree to O(dt) and are
    indistinguishable for smooth (interpolated) waveforms.  For sharp-edged
    waveforms with very few steps per pulse (n_pulse < ~10), the trapezoidal
    rule underestimates q_max by ≈ 0.5/n_pulse, causing set_b to over-scale
    G and introducing a systematic error of order (0.5/n_pulse)² in b.
    The rectangular rule avoids this inconsistency.

    Returns
    -------
    b_values : np.ndarray of shape (n_measurements,)
    """
    G = np.array(waveform.G)  # (n_measurements, n_t, 3)
    dt = waveform.dt
    q = np.cumsum(G * dt, axis=1) * GAMMA  # (n_measurements, n_t, 3)
    q_sq = np.sum(q ** 2, axis=2)          # (n_measurements, n_t)
    b = np.trapezoid(q_sq, dx=dt, axis=1)      # (n_measurements,)
    return b.astype(np.float64)


def calc_btensor(waveform):
    """Compute the B-tensor for each measurement.

    B_ij = ∫ q_i(t) q_j(t) dt   where q(t) = γ ∫₀ᵗ G(t') dt'

    Evaluated with the same rectangular rule as calc_b(), so
    trace(calc_btensor(wf)) == calc_b(wf) to float64 precision.

    Parameters
    ----------
    waveform : Waveform

    Returns
    -------
    B : np.ndarray, shape (n_measurements, 3, 3), float64
        B-tensor in s/m².
    """
    G = np.array(waveform.G)               # (n_meas, n_t, 3)
    dt = waveform.dt
    q = np.cumsum(G * dt, axis=1) * GAMMA  # (n_meas, n_t, 3)
    # B_ij = ∫ q_i(t) q_j(t) dt — trapezoidal rule, consistent with calc_b
    # so that trace(calc_btensor(wf)) == calc_b(wf) to float64 precision.
    qq = q[:, :, :, None] * q[:, :, None, :]   # (n_meas, n_t, 3, 3)
    B = np.trapezoid(qq, dx=dt, axis=1)         # (n_meas, 3, 3)
    return B.astype(np.float64)


def btensor_invariants(B):
    """Extract scalar invariants from B-tensor array.

    Parameters
    ----------
    B : np.ndarray, shape (n_measurements, 3, 3)
        B-tensor in s/m² (e.g. from calc_btensor).

    Returns
    -------
    b : np.ndarray, shape (n_measurements,)
        Scalar b-value = trace(B).
    b_delta : np.ndarray, shape (n_measurements,)
        Normalized anisotropy = (λ_max − λ_min) / b.
        LTE → 1,  STE → 0,  PTE → −0.5.
    b_eta : np.ndarray, shape (n_measurements,)
        Normalized asymmetry = (λ_max − 2λ_mid + λ_min) / b.
        Zero for axially symmetric tensors (LTE, STE, PTE).

    Notes
    -----
    b_delta sign convention (Szczepankiewicz et al. J Neurosci Methods 2021):

        prolate (LTE-like, λ_max farther from b/3 than λ_min):
            b_delta = (λ_1 − (λ_2 + λ_3)/2) / b   → positive, LTE=1

        oblate (PTE-like, λ_min farther from b/3 than λ_max):
            b_delta = (λ_3 − (λ_1 + λ_2)/2) / b   → negative, PTE=−0.5

    b_eta measures asymmetry of the equatorial eigenvalues:
        prolate: b_eta = (λ_2 − λ_3) / b
        oblate:  b_eta = (λ_1 − λ_2) / b
    Zero for axially symmetric tensors (LTE, STE, PTE).
    """
    B = np.asarray(B)
    eigvals = np.linalg.eigvalsh(B)         # (n_meas, 3), ascending
    eigvals = eigvals[:, ::-1]              # descending: λ_1 ≥ λ_2 ≥ λ_3
    lam1, lam2, lam3 = eigvals[:, 0], eigvals[:, 1], eigvals[:, 2]
    b = lam1 + lam2 + lam3                 # = trace(B)
    b_safe = np.where(b > 0, b, 1.0)       # avoid division by zero at b=0
    b_third = b / 3.0
    # Prolate when λ_1 is farther from b/3 than λ_3 is
    is_prolate = np.abs(lam1 - b_third) >= np.abs(lam3 - b_third)
    b_delta_prolate = (lam1 - (lam2 + lam3) / 2.0) / b_safe
    b_delta_oblate  = (lam3 - (lam1 + lam2) / 2.0) / b_safe
    b_delta = np.where(is_prolate, b_delta_prolate, b_delta_oblate)
    b_eta_prolate = (lam2 - lam3) / b_safe
    b_eta_oblate  = (lam1 - lam2) / b_safe
    b_eta = np.where(is_prolate, b_eta_prolate, b_eta_oblate)
    return b, b_delta, b_eta


def ste(delta, DELTA, G_magnitude, n_t, slew_rate=DEFAULT_SLEW_RATE):
    """Build a spherical tensor encoding (STE) waveform (b_delta = 0).

    Slew-limited (trapezoidal lobes) by default; ``slew_rate=np.inf`` for the
    square limit (``None`` rejected).  b_delta = 0 is preserved under slew-
    limiting: each axis's q(t) still returns to zero within its own back-to-back
    pair, so the off-diagonal B-tensor elements remain exactly zero.

    Three sequential bipolar gradient pairs, one per Cartesian axis (x, y, z),
    played back-to-back within the total duration DELTA + delta.

    Structure over T_total = DELTA + delta, divided into 6 equal segments::

        segs 0-1 : +G_x then -G_x  →  q_x rises and returns to 0
        segs 2-3 : +G_y then -G_y  →  q_y rises and returns to 0
        segs 4-5 : +G_z then -G_z  →  q_z rises and returns to 0

    Because each axis's q(t) has strictly non-overlapping support with every
    other axis, the off-diagonal B-tensor elements are exactly zero.  By
    symmetry B_xx = B_yy = B_zz, so B = (b/3) I and b_delta = 0.

    This is not time-optimal (q-MAS waveforms achieve higher b for given G and
    duration) but is trivially verifiable from first principles.

    Parameters
    ----------
    delta, DELTA : float
        Only their sum T_total = DELTA + delta is used; the waveform does not
        follow the PGSE δ/Δ structure.  Pass the same values you would use for
        a PGSE with the same total duration.
    G_magnitude : float
        Gradient amplitude in T/m.  Use set_b() to scale to a target b.
    n_t : int
        Number of time points.

    Returns
    -------
    Waveform
        Single measurement.  G shape = (1, n_t, 3).
    """
    T_total = DELTA + delta
    dt = T_total / (n_t - 1)
    n_seg = max(1, round(n_t / 6))

    _, square = _resolve_slew(slew_rate)
    n_rise = _lobe_n_rise(square, abs(float(G_magnitude)), slew_rate, dt, n_seg)
    eye = np.eye(3, dtype=np.float32)
    G_grad = np.zeros((1, n_t, 3), dtype=np.float32)
    for i in range(3):
        enc_start = 2 * i * n_seg
        enc_end   = min((2 * i + 1) * n_seg, n_t)
        dec_end   = min((2 * i + 2) * n_seg, n_t)
        _fill_lobe(G_grad, 0, enc_start, enc_end - enc_start,  G_magnitude * eye[i], n_rise)
        _fill_lobe(G_grad, 0, enc_end,   dec_end - enc_end,   -G_magnitude * eye[i], n_rise)

    return Waveform(G=jnp.array(G_grad), dt=float(dt),
                    echo_idx=n_t - 1,
                    rf_events=[{'t_s': 0.0, 'label': 'Mz→Mxy', 'flip_deg': 90}])


def pte(delta, DELTA, G_magnitude, plane_normal, n_t, slew_rate=DEFAULT_SLEW_RATE):
    """Build a planar tensor encoding (PTE) waveform (b_delta = -0.5).

    Slew-limited (trapezoidal lobes) by default; ``slew_rate=np.inf`` for the
    square limit (``None`` rejected).  b_delta = -0.5 is preserved: each in-plane
    axis's q(t) still returns to zero within its own pair (B_uv = 0).

    Two sequential bipolar gradient pairs, one per in-plane axis (u, v),
    played back-to-back within the total duration DELTA + delta.

    Structure over T_total = DELTA + delta, divided into 4 equal segments::

        segs 0-1 : +G_u then -G_u  →  q_u rises and returns to 0
        segs 2-3 : +G_v then -G_v  →  q_v rises and returns to 0

    Because u and v have non-overlapping q(t) support, B_uv = 0 exactly.
    Since B_ww = 0 (no gradient along the plane normal w), the eigenvalues
    are (b/2, b/2, 0) and b_delta = -0.5.

    Parameters
    ----------
    delta, DELTA : float
        Only their sum T_total = DELTA + delta is used.
    G_magnitude : float
        Gradient amplitude in T/m per in-plane axis.  Use set_b() to scale.
    plane_normal : array of shape (3,)
        Unit vector normal to the encoding plane.
    n_t : int
        Number of time points.

    Returns
    -------
    Waveform
        Single measurement.  G shape = (1, n_t, 3).
    """
    n = np.asarray(plane_normal, dtype=np.float64)
    n = n / np.linalg.norm(n)

    # Two orthonormal vectors in the encoding plane via Gram-Schmidt.
    ref = np.array([1.0, 0.0, 0.0]) if abs(n[0]) < 0.9 else np.array([0.0, 1.0, 0.0])
    u = ref - np.dot(ref, n) * n
    u /= np.linalg.norm(u)
    v = np.cross(n, u)
    v /= np.linalg.norm(v)

    T_total = DELTA + delta
    dt = T_total / (n_t - 1)
    n_seg = max(1, round(n_t / 4))

    _, square = _resolve_slew(slew_rate)
    n_rise = _lobe_n_rise(square, abs(float(G_magnitude)), slew_rate, dt, n_seg)
    G_grad = np.zeros((1, n_t, 3), dtype=np.float32)
    for i, axis in enumerate([u, v]):
        enc_start = 2 * i * n_seg
        enc_end   = min((2 * i + 1) * n_seg, n_t)
        dec_end   = min((2 * i + 2) * n_seg, n_t)
        avec = (G_magnitude * axis).astype(np.float32)
        _fill_lobe(G_grad, 0, enc_start, enc_end - enc_start,  avec, n_rise)
        _fill_lobe(G_grad, 0, enc_end,   dec_end - enc_end,   -avec, n_rise)

    return Waveform(G=jnp.array(G_grad), dt=float(dt),
                    echo_idx=n_t - 1,
                    rf_events=[{'t_s': 0.0, 'label': 'Mz→Mxy', 'flip_deg': 90}])


def set_b(waveform, b_target):
    """Return a new Waveform scaled so each measurement has the given b-value.

    Parameters
    ----------
    waveform : Waveform
    b_target : float or array of shape (n_measurements,)
        Target b-values in **s/m²** (SI units), consistent with ``calc_b``.
        Typical clinical values: 1e8–3e9 s/m² (= 100–3000 s/mm²).

        .. warning::

           A common mistake is passing b-values in **s/mm²** (e.g. 1000) instead
           of **s/m²** (e.g. 1e9).  ``set_b`` will silently produce gradients
           that are 1000× too small, giving essentially b≈0 signals.
           Convert: ``b_si = b_mm2 * 1e6``.

    Returns
    -------
    Waveform with scaled G.
    """
    import warnings
    b_current = calc_b(waveform)
    b_arr = np.asarray(b_target, dtype=np.float64).ravel()
    b_nonzero = b_arr[b_arr > 0]
    if b_nonzero.size > 0 and b_nonzero.max() < 1e5:
        warnings.warn(
            f"set_b: b_target values appear very small (max={b_nonzero.max():.4g} s/m²). "
            "Did you pass b-values in s/mm² instead of s/m²? "
            "Multiply by 1e6 to convert: set_b(wf, b_mm2 * 1e6). "
            "Typical SI b-values are 1e8–3e9 s/m² (100–3000 s/mm²).",
            UserWarning,
            stacklevel=2,
        )
    b_target = np.broadcast_to(np.asarray(b_target, dtype=np.float64),
                                b_current.shape)
    # b scales as G², so G scales as sqrt(b_target / b_current)
    scale = np.sqrt(b_target / b_current).astype(np.float32)  # (n_measurements,)
    G_new = np.array(waveform.G) * scale[:, None, None]
    G_disp = waveform.G_display
    if G_disp is not None:
        G_disp = (np.array(G_disp) * scale[:, None, None]).astype(np.float32)
    return Waveform(G=jnp.array(G_new.astype(np.float32)),
                    dt=waveform.dt,
                    echo_idx=waveform.echo_idx,
                    rf_events=waveform.rf_events,
                    G_display=G_disp)


def rotate_waveform(waveform, R=None, *, theta=None):
    """Rotate the gradient vectors of a waveform: ``G_new = G @ R.T``.

    Pass **either** a rotation matrix ``R`` **or** a polar angle ``theta``:

    - ``R`` (3, 3): an explicit rotation matrix.  Since
      ``dphi = GAMMA * dt * dot(G, r)``, rotating the substrate to orientation
      ``n`` is equivalent to rotating ``G`` by the inverse rotation — so this
      simulates a rotated geometry without rebuilding it.
    - ``theta`` (radians): rotation about the y-axis mapping the z-axis to
      ``(sin theta, 0, cos theta)`` — probes a single-fibre (z) substrate at
      polar angle ``theta`` from the fibre axis (used by SH fibre-response
      sampling).

    Parameters
    ----------
    waveform : Waveform
        Input waveform with G of shape (n_measurements, n_t, 3).
    R : np.ndarray, shape (3, 3), optional
        Rotation matrix (mutually exclusive with ``theta``).
    theta : float, optional
        Polar angle in radians (mutually exclusive with ``R``).

    Returns
    -------
    Waveform
        New Waveform with rotated G, same dt and echo_idx.
    """
    if (R is None) == (theta is None):
        raise ValueError("rotate_waveform: pass exactly one of R or theta.")
    if theta is not None:
        c, s = np.cos(float(theta)), np.sin(float(theta))
        R = np.array([[c, 0.0, s], [0.0, 1.0, 0.0], [-s, 0.0, c]], dtype=np.float32)
    R = np.asarray(R, dtype=np.float32)
    G = np.array(waveform.G)  # (n_meas, n_t, 3)
    # G_rotated[m, t, :] = G[m, t, :] @ R.T = R.T @ G[m, t, :]^T
    G_rot = np.einsum('mtj,ij->mti', G, R)  # G @ R.T via einsum
    return Waveform(G=jnp.array(G_rot, dtype=jnp.float32),
                    dt=waveform.dt,
                    echo_idx=waveform.echo_idx,
                    rf_events=waveform.rf_events)


def tile_waveform(waveform, n_copies):
    """Tile a waveform along the measurement dimension.

    Creates ``n_copies`` copies of the waveform stacked along the first
    (measurement) axis.  Used to batch multiple orientations into a single
    ``simulate()`` call.

    Parameters
    ----------
    waveform : Waveform
        Input waveform with G of shape (n_measurements, n_t, 3).
    n_copies : int
        Number of copies.

    Returns
    -------
    Waveform
        New Waveform with G of shape (n_copies * n_measurements, n_t, 3).
    """
    G = np.array(waveform.G)  # (n_meas, n_t, 3)
    G_tiled = np.tile(G, (n_copies, 1, 1))
    return Waveform(G=jnp.array(G_tiled, dtype=jnp.float32),
                    dt=waveform.dt,
                    echo_idx=waveform.echo_idx,
                    rf_events=waveform.rf_events)


def cpmg(n_echoes, TE, G_magnitude, bvecs, n_t_per_echo=100):
    """Instantaneous-pulse CPMG (Carr–Purcell–Meiboom–Gill) echo train.

    A plain :class:`Waveform` for the multi-echo spin-echo: a 90° excitation at
    ``t=0`` and ``n_echoes`` ideal 180° refocusing pulses at ``(k+½)·TE``, with
    echoes forming at ``k·TE``.  Magnetisation is fully transverse throughout
    (ideal instantaneous pulses) — there is no coherence-pathway machinery
    here; this is a gradient+RF schedule for visualisation
    (:mod:`dmipy_sim.pedagogy`) and for the final-echo :func:`simulate` sample.

    The (constant) diffusion-weighting gradient is stored physically in
    ``G_display``; the ``G`` used by the walk carries the effective sign flip
    imposed by each 180° (so static-spin phase refocuses at every echo).  Use
    ``G_magnitude=0`` for a pure-T2 demonstration.

    Parameters
    ----------
    n_echoes : int
        Number of refocusing pulses / echoes.
    TE : float
        Echo spacing in seconds (echoes at k·TE, k=1..n_echoes).
    G_magnitude : float or array of shape (n_measurements,)
        Constant gradient amplitude in T/m (0 for a pure-T2 train).
    bvecs : array of shape (n_measurements, 3)
        Unit gradient direction vectors.
    n_t_per_echo : int
        Time samples per echo interval.

    Returns
    -------
    Waveform
    """
    bvecs = np.asarray(bvecs, dtype=np.float32)
    n_measurements = bvecs.shape[0]
    G_mag = np.broadcast_to(np.asarray(G_magnitude, dtype=np.float32),
                            (n_measurements,))

    n_t = n_echoes * n_t_per_echo + 1
    T_total = n_echoes * TE
    dt = T_total / (n_t - 1)
    t = np.arange(n_t) * dt

    # Effective sign: +1, flipped by each 180° at (k+1/2)*TE (spin-echo conjugation).
    ref_times = [(k + 0.5) * TE for k in range(n_echoes)]
    sign = np.ones(n_t, dtype=np.float32)
    for rt in ref_times:
        sign[t >= rt] *= -1.0

    G_phys = np.zeros((n_measurements, n_t, 3), dtype=np.float32)
    G_eff  = np.zeros((n_measurements, n_t, 3), dtype=np.float32)
    for m in range(n_measurements):
        g = G_mag[m] * bvecs[m]
        G_phys[m, :, :] = g
        G_eff[m, :, :]  = sign[:, None] * g

    rf_events = [{'t_s': 0.0, 'label': 'Mz→Mxy', 'flip_deg': 90}]
    rf_events += [{'t_s': (k + 0.5) * TE, 'label': 'refocus', 'flip_deg': 180}
                  for k in range(n_echoes)]

    # Echoes form at k*TE, k=1..n_echoes → step indices k*n_t_per_echo.
    echo_indices = np.arange(1, n_echoes + 1) * n_t_per_echo

    return Waveform(G=jnp.array(G_eff), dt=float(dt),
                    echo_idx=n_t - 1,
                    rf_events=rf_events,
                    G_display=G_phys,
                    echo_indices=echo_indices)
