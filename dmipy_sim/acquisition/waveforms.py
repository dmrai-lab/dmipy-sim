"""The gradient integrals and the measurement-axis transforms of a :class:`ScannerSequence`.

``b_from_gradient`` / ``btensor_from_gradient`` are THE b and B-tensor integrals of a gradient array
``(n_measurements, n_t, 3)`` in T/m on a ``dt`` grid (rectangular q, trapezoidal integral); ``calc_b`` /
``calc_btensor`` read a sequence's EFFECTIVE gradient; ``btensor_invariants`` names a B-tensor's shape;
``set_b`` / ``rotate_waveform`` / ``tile_waveform`` return a sequence rescaled, rotated or tiled along its
measurement axis, its ``Encoding`` following. The builders live in :mod:`dmipy_sim.sequences`.
"""

from dataclasses import replace
import numpy as np

from ..constants import GAMMA


def b_from_gradient(G, dt):
    """b (s/m²) per measurement from a gradient array ``(n_measurements, n_t, 3)`` and its step.

    ``q(t)`` accumulates by the rectangular (left-point) rule, matching the phase accumulation of
    the walk (``dphi = GAMMA dt G[t] . r``); ``b = ∫ |q|² dt`` by the trapezoidal rule. This is
    the one b integral of the package: :func:`calc_b`, :meth:`Sequence.btensor` and every
    constructor's numeric scaling read it.
    """
    G = np.asarray(G, dtype=np.float64)
    q = np.cumsum(G * float(dt), axis=1) * GAMMA        # (n_m, n_t, 3)
    return np.trapezoid(np.sum(q ** 2, axis=2), dx=float(dt), axis=1).astype(np.float64)


def btensor_from_gradient(G, dt):
    """B-tensor ``B_ij = ∫ q_i q_j dt`` per measurement, ``(n_measurements, 3, 3)``, by the same
    rules as :func:`b_from_gradient`, so ``trace(B) == b`` to float64 precision."""
    G = np.asarray(G, dtype=np.float64)
    q = np.cumsum(G * float(dt), axis=1) * GAMMA
    qq = q[:, :, :, None] * q[:, :, None, :]
    return np.trapezoid(qq, dx=float(dt), axis=1).astype(np.float64)


def calc_b(waveform):
    """Compute b-values for each measurement of a ScannerSequence (s/m²).

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
    return b_from_gradient(waveform.G_eff, waveform.dt)


def calc_btensor(waveform):
    """Compute the B-tensor for each measurement.

    B_ij = ∫ q_i(t) q_j(t) dt   where q(t) = γ ∫₀ᵗ G(t') dt'

    Evaluated with the same rectangular rule as calc_b(), so
    trace(calc_btensor(wf)) == calc_b(wf) to float64 precision.

    Parameters
    ----------
    waveform : ScannerSequence

    Returns
    -------
    B : np.ndarray, shape (n_measurements, 3, 3), float64
        B-tensor in s/m².
    """
    return btensor_from_gradient(waveform.G_eff, waveform.dt)


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


def set_b(waveform, b_target):
    """Return a new ScannerSequence scaled so each measurement has the given b-value.

    Parameters
    ----------
    waveform : ScannerSequence
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
    ScannerSequence with scaled G.
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
    b_target = np.broadcast_to(np.asarray(b_target, dtype=np.float64), b_current.shape)
    # b scales as G², so G scales as sqrt(b_target / b_current)
    scale = np.where(b_current > 0, np.sqrt(b_target / np.where(b_current > 0, b_current, 1.0)), 1.0)
    G_new = np.asarray(waveform.G, dtype=np.float64) * scale[:, None, None]
    enc = waveform.encoding
    if enc is not None:                                     # the declared b and its amplitude follow the gradient
        enc = replace(enc, bvalues=b_target,
                      gradient_strengths=None if enc.gradient_strengths is None else enc.gradient_strengths * scale,
                      qvalues=None if enc.qvalues is None else enc.qvalues * scale)
    return replace(waveform, G=G_new.astype(np.float32), encoding=enc)


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
    waveform : ScannerSequence
        Input waveform with G of shape (n_measurements, n_t, 3).
    R : np.ndarray, shape (3, 3), optional
        Rotation matrix (mutually exclusive with ``theta``).
    theta : float, optional
        Polar angle in radians (mutually exclusive with ``R``).

    Returns
    -------
    ScannerSequence
        New ScannerSequence with rotated G, same dt and readout.
    """
    if (R is None) == (theta is None):
        raise ValueError("rotate_waveform: pass exactly one of R or theta.")
    if theta is not None:
        c, s = np.cos(float(theta)), np.sin(float(theta))
        R = np.array([[c, 0.0, s], [0.0, 1.0, 0.0], [-s, 0.0, c]], dtype=np.float32)
    R = np.asarray(R, dtype=np.float64)
    G = np.asarray(waveform.G, dtype=np.float64)  # (n_meas, n_t, 3)
    G_rot = np.einsum('mtj,ij->mti', G, R)        # G @ R.T
    enc = waveform.encoding
    if enc is not None:                            # the declared directions turn with the gradient
        enc = replace(enc, gradient_directions=np.asarray(enc.gradient_directions) @ R.T)
    return replace(waveform, G=G_rot.astype(np.float32), encoding=enc)


def tile_waveform(waveform, n_copies):
    """Tile a waveform along the measurement dimension.

    Creates ``n_copies`` copies of the waveform stacked along the first
    (measurement) axis.  Used to batch multiple orientations into a single
    ``simulate()`` call.

    Parameters
    ----------
    waveform : ScannerSequence
        Input waveform with G of shape (n_measurements, n_t, 3).
    n_copies : int
        Number of copies.

    Returns
    -------
    ScannerSequence
        New ScannerSequence with G of shape (n_copies * n_measurements, n_t, 3).
    """
    G = np.array(waveform.G)  # (n_meas, n_t, 3)
    G_tiled = np.tile(G, (n_copies, 1, 1))
    return replace(waveform, G=G_tiled.astype(np.float32), encoding=None)      # n_meas changes: the encoding does not carry over


