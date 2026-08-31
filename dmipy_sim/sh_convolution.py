"""SH convolution for orientation distributions in dmipy-sim.

Theory
------
For any axially-symmetric substrate (cylinder, myelinated cylinder, packed
cylinders along z) the single-fiber signal E(theta) depends only on the polar
angle theta between the gradient waveform axis and the fiber axis.  This means
E(theta) has only m=0 spherical harmonic (zonal/Legendre) terms:

    E(theta) = sum_{l=0,2,...,lmax}  f_l * P_l(cos theta)

where P_l are Legendre polynomials and f_l are the rotational harmonic (RH)
coefficients of the fiber response.

The ensemble-averaged signal for an orientation distribution ODF(n) is:

    E_total = integral  E(n) * ODF(n)  dn
            = sum_l  (4*pi / (2*l+1))  *  f_l  *  c_l^0

where c_l^0 is the m=0 (zonal) SH coefficient of the ODF in the real
symmetric SH convention (Tournier/dipy descoteaux, same as dmipy-core).

SH convention (same as dmipy-core / Tournier / MRtrix):
    Y_l^0(theta, phi) = sqrt((2l+1)/(4*pi)) * P_l(cos theta)
    Y_0^0 = 1/(2*sqrt(pi))
    c_0^0 of any normalised PDF = 1/(2*sqrt(pi))

Compact even-order SH array layout (m=0 indices):
    l=0: start=0, m=0 at index 0
    l=2: start=1, m=0 at index 3   (5 coefficients, m=-2,-1,0,1,2)
    l=4: start=6, m=0 at index 10  (9 coefficients)
    l=6: start=15, m=0 at index 21 (13 coefficients)
    l=8: start=28, m=0 at index 36 (17 coefficients)
    General: M=l//2; start = M*(2M-1) for M>0, else 0; m0_idx = start + l

References
----------
Kaden, Knosche & Anwander (2007), NeuroImage. (SH convolution theory)
dmipy-core dmipy/utils/spherical_convolution.py (validated convention)
"""

from __future__ import annotations

import numpy as np
from .waveforms import rotate_waveform
from scipy.special import eval_legendre, roots_legendre


# ---------------------------------------------------------------------------
# Helper: m=0 index in compact even-order SH array
# ---------------------------------------------------------------------------

def _m0_idx(l):
    """Return index of (l, m=0) in compact even-order real SH array.

    Array layout: l=0 has 1 coeff; l=2 has 5; l=4 has 9; ...
    start_of_l = M*(2M-1) where M = l//2 (with M=0 → start=0)
    m=0 offset within order l block = l
    """
    M = l // 2
    start = M * (2 * M - 1) if M > 0 else 0
    return start + l


def _n_sh_coeffs(lmax):
    """Total number of coefficients in compact even-order SH array up to lmax."""
    return sum(2 * l + 1 for l in range(0, lmax + 1, 2))


# ---------------------------------------------------------------------------
# Waveform rotation by polar angle (about y-axis)
# ---------------------------------------------------------------------------



# ---------------------------------------------------------------------------
# Fiber response function via MC simulation + Legendre fit
# ---------------------------------------------------------------------------

def compute_fiber_response(geometry, acquisition_scheme, n_walkers,
                            lmax=8, seed=0, diffusivity=None, T2=None):
    """Simulate single-fiber signal at n_angles polar angles and fit Legendre.

    Uses Gauss-Legendre quadrature nodes mapped to [0, 1] (cos theta in [0,1],
    upper hemisphere) for the numerical integration.

    For each angle theta_k the acquisition waveform is rotated so that the
    gradient direction makes angle theta_k with the fiber axis (z).  Simulation
    is run with the fiber fixed along z and the waveform rotated.

    Legendre coefficients f_l are fit via least squares to the n_angles signal
    values:  E(theta_k) = sum_{l=0,2,...,lmax} f_l * P_l(cos theta_k)

    Parameters
    ----------
    geometry : dmipy-sim geometry
        Any geometry oriented along z (Cylinder, MyelinatedCylinder, etc.).
    acquisition_scheme : Waveform or object with .waveform attribute
        Waveform to simulate.
    n_walkers : int
        Walkers per polar angle.
    lmax : int
        Maximum even SH order (default 8).  Must be even.
    seed : int
        Base random seed; angle k gets seed+k.

    Returns
    -------
    fiber_response : np.ndarray, shape (lmax//2+1, n_measurements)
        Legendre RH coefficients f_l for l=0,2,...,lmax.
    thetas : np.ndarray, shape (n_angles,)
        Polar angles used (radians).
    E_theta : np.ndarray, shape (n_angles, n_measurements)
        Raw simulated signals at each angle (for diagnostics / residuals).
    """
    from .core import simulate

    if lmax % 2 != 0:
        raise ValueError(f"lmax must be even, got {lmax}")

    n_orders = lmax // 2 + 1   # number of even orders: 0, 2, ..., lmax
    # Use 2*n_orders GL points on [-1,1] for full orthogonality
    # (GL with n points integrates polynomials of degree 2n-1 exactly)
    n_angles = 2 * n_orders

    # Unwrap AcquisitionScheme to Waveform if needed
    if hasattr(acquisition_scheme, 'waveform'):
        waveform = acquisition_scheme.waveform
    else:
        waveform = acquisition_scheme

    n_measurements = waveform.G.shape[0]

    # Gauss-Legendre nodes on [-1, 1]
    # x = cos(theta); for cylinder, E(x) = E(-x) (antipodal symmetry).
    # We only need to simulate at unique |x| values (upper hemisphere).
    gl_nodes, gl_weights = roots_legendre(n_angles)   # nodes in [-1, 1]

    # Due to E(x) = E(-x), nodes come in ±x pairs. Exploit symmetry:
    # simulate only the positive-x half (theta in [0, pi/2])
    # and set E at negative-x nodes to the same value.
    cos_thetas_unique = gl_nodes[n_angles // 2:]   # x > 0 (upper half)
    thetas_unique = np.arccos(cos_thetas_unique)    # theta in (0, pi/2)

    # Get D from geometry if available (for standard geometries)
    # Caller-supplied diffusivity takes precedence
    D_val = diffusivity if diffusivity is not None else getattr(geometry, '_D', None)
    T2_val = T2

    # Simulate at each unique angle
    E_unique = np.zeros((n_angles // 2, n_measurements), dtype=np.float64)
    for k, theta in enumerate(thetas_unique):
        waveform_k = rotate_waveform(waveform, theta=theta)
        sig = simulate(
            n_walkers=n_walkers,
            waveform=waveform_k,
            geometry=geometry,
            seed=seed + k,
            diffusivity=D_val,
            T2=T2_val,
        )
        E_unique[k] = np.asarray(sig, dtype=np.float64)

    # Construct E at all GL nodes: E[-x] = E[x]
    # gl_nodes is sorted ascending; negative nodes come first
    E_all = np.concatenate([E_unique[::-1], E_unique], axis=0)  # shape (n_angles, n_meas)
    thetas = np.arccos(gl_nodes)  # for return value

    # Compute Legendre coefficients via GL quadrature (exact for polynomials ≤ 2n-1):
    # f_l = (2l+1)/2 * sum_k w_k E(x_k) P_l(x_k)
    fiber_response = np.zeros((n_orders, n_measurements), dtype=np.float64)
    for j in range(n_orders):
        l = 2 * j
        P_l_vals = eval_legendre(l, gl_nodes)  # shape (n_angles,)
        # (2l+1)/2 * integral_{-1}^1 E(x) P_l(x) dx ≈ (2l+1)/2 * sum_k w_k E_k P_l(x_k)
        fiber_response[j] = ((2 * l + 1) / 2.0) * np.dot(
            gl_weights, E_all * P_l_vals[:, None])  # dot broadcasts correctly

    # E_theta for diagnostics: return the unique angles (upper hemisphere)
    return fiber_response, thetas_unique, E_unique


# ---------------------------------------------------------------------------
# Apply ODF via SH convolution
# ---------------------------------------------------------------------------

def apply_odf(fiber_response, odf_sh, lmax=8):
    """Convolve fiber response with ODF via SH convolution formula.

    Computes:
        E = sum_l  (4*pi / (2*l+1))  *  f_l  *  c_l^0

    where f_l = fiber_response[l//2] is the l-th Legendre (RH) coefficient
    and c_l^0 is the m=0 SH coefficient of the ODF at order l.

    This formula follows from:
        E = integral E(theta(n)) * ODF(n) dn
    with E(theta) = sum_l f_l P_l(cos theta) and ODF in real SH basis.

    Parameters
    ----------
    fiber_response : np.ndarray, shape (lmax//2+1, n_measurements)
        From compute_fiber_response().
    odf_sh : np.ndarray or None
        Real SH coefficients in compact even-order representation.
        None → isotropic (returns fiber_response[0], the l=0 Legendre coeff
        which is the solid-angle average).
    lmax : int

    Returns
    -------
    signal : np.ndarray, shape (n_measurements,)
    """
    if odf_sh is None:
        # Isotropic: return the l=0 Legendre coefficient
        # (which equals the solid-angle-weighted average of E(theta))
        return np.asarray(fiber_response[0], dtype=np.float64)

    odf_sh = np.asarray(odf_sh, dtype=np.float64)
    n_orders = lmax // 2 + 1

    if fiber_response.ndim == 1:
        signal = 0.0
    else:
        signal = np.zeros(fiber_response.shape[1], dtype=np.float64)

    for j in range(n_orders):
        l = 2 * j
        idx = _m0_idx(l)
        if idx >= len(odf_sh):
            break

        c_l0 = odf_sh[idx]
        factor = np.sqrt((4.0 * np.pi) / (2.0 * l + 1.0))
        signal = signal + factor * fiber_response[j] * c_l0

    return signal


# ---------------------------------------------------------------------------
# Watson ODF SH coefficients
# ---------------------------------------------------------------------------

def watson_odf_sh(kappa, mu=None, lmax=8):
    """Compute real SH coefficients for a Watson distribution.

    Delegates to dmipy.utils.sh_analytical.watson_sh, which uses an exact
    erfi-based recurrence (no quadrature, no hyp1f1 overflow issues) and
    supports arbitrary mean orientation mu.

    The output layout (compact even-order real SH, Tournier convention) is
    identical to the one used by apply_odf / compute_fiber_response.

    Parameters
    ----------
    kappa : float
        Concentration parameter.  kappa=0 → isotropic; kappa→inf → single fiber.
    mu : np.ndarray, shape (3,), optional
        Mean orientation as a unit Cartesian vector.  Default: z=[0,0,1].
    lmax : int
        Maximum even SH order.

    Returns
    -------
    odf_sh : np.ndarray, shape (n_sh_coeffs,)
        Compact even-order SH coefficients.
    """
    from .math.sh_analytical import watson_sh

    if mu is None:
        mu = np.array([0., 0., 1.], dtype=np.float64)
    return watson_sh(np.asarray(mu, dtype=np.float64), float(kappa), l_max=lmax)


def isotropic_odf_sh(lmax=8):
    """SH coefficients for uniform sphere (only l=0 term non-zero).

    For ODF(n) = 1/(4*pi):
        c_0^0 = integral (1/(4*pi)) * Y_0^0 dn = Y_0^0 = 1/(2*sqrt(pi))
        All higher-order coefficients = 0.

    Parameters
    ----------
    lmax : int

    Returns
    -------
    odf_sh : np.ndarray
    """
    odf_sh = np.zeros(_n_sh_coeffs(lmax), dtype=np.float64)
    odf_sh[0] = 1.0 / (2.0 * np.sqrt(np.pi))
    return odf_sh


# ---------------------------------------------------------------------------
# Two-axis (susceptibility-aware) composition -- paper Sec. 2.6
# ---------------------------------------------------------------------------

def coupled_spectrum(E, u_nodes, v_nodes, l_g=8, l_b=6):
    """Coupled angular spectrum ``Lambda_{l1 l2}`` of a two-axis response.

    ``E`` is the response sampled on the tensor grid ``u_nodes x v_nodes`` of the two
    invariants ``u = n.g`` and ``v = n.B0`` (shape ``(len(u_nodes), len(v_nodes))``, or with
    a trailing measurement axis).  ``u_nodes``/``v_nodes`` must be Gauss-Legendre nodes on
    [-1, 1]; use :func:`invariant_grid` to build them together with their weights.

    Separability of the physics is the rank-one case; nothing here assumes it.  Use
    :func:`rank1_residual` to report how far a given substrate departs from it.
    """
    from .gaunt import _CACHE  # noqa: F401  (kept for cache-warm symmetry)
    E = np.asarray(E, np.float64)
    u = np.asarray(u_nodes, np.float64); v = np.asarray(v_nodes, np.float64)
    _, wu = roots_legendre(u.size)
    _, wv = roots_legendre(v.size)
    n1, n2 = l_g // 2 + 1, l_b // 2 + 1
    Pu = np.stack([eval_legendre(2 * i, u) for i in range(n1)])      # (n1, nu)
    Pv = np.stack([eval_legendre(2 * j, v) for j in range(n2)])      # (n2, nv)
    norm1 = np.array([(4 * i + 1) / 2.0 for i in range(n1)])
    norm2 = np.array([(4 * j + 1) / 2.0 for j in range(n2)])
    lam = np.einsum("iu,jv,uv...,u,v->ij...", Pu, Pv, E, wu, wv, optimize=True)
    return lam * norm1[:, None] * norm2[None, :] if lam.ndim == 2 else \
        lam * norm1[:, None, None] * norm2[None, :, None]


def invariant_grid(n_u=24, n_v=24):
    """Gauss-Legendre nodes/weights in the two invariants ``(n.g, n.B0)``."""
    u, wu = roots_legendre(n_u)
    v, wv = roots_legendre(n_v)
    return u, v, wu, wv


def rank1_residual(lam):
    """``||Lambda - rank1(Lambda)|| / ||Lambda||`` -- how far the response is from separable.

    Zero means susceptibility acts as a magnitude weight multiplying the diffusion
    attenuation (the assumption of the analytic operator).  A non-zero value is a measured
    property of the substrate: the gradient and susceptibility phases add *inside* the
    ensemble average, so the expectation need not factor.
    """
    lam = np.asarray(lam, np.float64)
    s = np.linalg.svd(lam, compute_uv=False)
    return float(np.sqrt(max(np.sum(s[1:] ** 2), 0.0)) / np.linalg.norm(lam))


def coupled_spectrum_at(response, g_dir, b0_dir, l_g=8, l_b=6, n_theta=48, n_phi=96):
    """Build ``Lambda`` for the *exact* geometry a scanner requests -- no table, no interpolation.

    ``Lambda`` is a scanner-side object: it depends on the gradient and field directions but
    not on the orientation distribution or the voxel, so it is built once per measurement and
    then composed against every FOD in a phantom (:func:`apply_odf_coupled`).  That split is
    what the Gaunt route buys -- the alternative, integrating over the sphere per voxel, repeats
    the response evaluation for every voxel.

    ``response`` is called as ``response(dirs)`` with ``dirs`` an ``(n, 3)`` array of fibre
    directions and must return the replayed signal for a substrate pointing each way, i.e. the
    pack evaluated at the counter-rotated ``(g, B0)``.

    The projection is on the **sphere**, not on a ``(u, v)`` tensor grid: for a given
    ``(g, B0)`` the two invariants are constrained to an ellipse rather than the square, so
    ``{P_l1(u) P_l2(v)}`` is not orthogonal there and a least-squares solve is required.  The
    family is also rank-deficient when ``g`` is parallel to ``B0`` (then ``u == v``), which is a
    common acquisition geometry -- hence ``lstsq`` and never ``solve``.  Degeneracy is harmless:
    the minimum-norm solution still reconstructs the response exactly.

    Returns ``(Lambda, residual, rank)``.  ``residual`` is the relative L2 misfit of the
    truncated expansion and is the honest per-measurement error indicator: the composition
    error tracks it.
    """
    from .gaunt import sphere_quadrature
    g = np.asarray(g_dir, np.float64); g = g / np.linalg.norm(g)
    b = np.asarray(b0_dir, np.float64); b = b / np.linalg.norm(b)
    dirs, w = sphere_quadrature(n_theta, n_phi)
    E = np.asarray(response(dirs), np.float64)
    n1, n2 = l_g // 2 + 1, l_b // 2 + 1
    B = np.stack([eval_legendre(2 * i, dirs @ g) * eval_legendre(2 * j, dirs @ b)
                  for i in range(n1) for j in range(n2)], axis=1)
    sw = np.sqrt(w)
    A, y = B * sw[:, None], E * sw
    lam, *_ = np.linalg.lstsq(A, y, rcond=None)
    sv = np.linalg.svd(A, compute_uv=False)
    rank = int((sv > sv[0] * 1e-10).sum())
    resid = float(np.linalg.norm(A @ lam - y) / max(np.linalg.norm(y), 1e-300))
    return lam.reshape(n1, n2), resid, rank


def apply_odf_coupled(lam, odf_sh, g_dir, b0_dir, l_fod=8, l_g=8, l_b=6):
    """Compose a two-axis response over an orientation distribution (paper Eq. 19).

    ``lam`` is ``Lambda_{l1 l2}`` from :func:`coupled_spectrum`; ``odf_sh`` the FOD in the
    compact even-order orthonormal real SH layout; ``g_dir``/``b0_dir`` unit vectors.
    Returns the composed signal.

    With ``Lambda`` supported on ``l2 = 0`` (no susceptibility) this reduces exactly to
    :func:`apply_odf`.
    """
    from .gaunt import gaunt_table, real_sh, sh_block, n_sh_coeffs
    lam = np.asarray(lam, np.float64)
    f = np.asarray(odf_sh, np.float64)
    nf = n_sh_coeffs(l_fod)
    if f.size < nf:
        f = np.concatenate([f, np.zeros(nf - f.size)])
    G = gaunt_table(l_fod, l_g, l_b)
    Yg = real_sh(l_g, np.asarray(g_dir, np.float64)[None, :] /
                 np.linalg.norm(g_dir))[0]
    Yb = real_sh(l_b, np.asarray(b0_dir, np.float64)[None, :] /
                 np.linalg.norm(b0_dir))[0]
    out = 0.0
    for i in range(lam.shape[0]):
        l1 = 2 * i
        for j in range(lam.shape[1]):
            l2 = 2 * j
            w = (4 * np.pi) ** 2 / ((2 * l1 + 1) * (2 * l2 + 1))
            blk = np.einsum("abc,a,b,c->", G[:nf, sh_block(l1), sh_block(l2)],
                            f[:nf], Yg[sh_block(l1)], Yb[sh_block(l2)])
            out = out + lam[i, j] * w * blk
    return out
