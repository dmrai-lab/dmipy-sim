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

def separability(response, g_dir, b0_dir, l_g=8, l_b=6, n_theta=48, n_phi=96, n_als=30):
    """How far a response is from the separable model ``E = A(n.g) * Xi(n.B0)``.

    Separability -- susceptibility acting as a magnitude weight on the diffusion attenuation,
    which is what the analytic operator assumes -- needs the response to lie in the rank-one
    even--even zonal sector.  Two things take it out of there: the chiral sector, and odd--odd
    ``p=0`` pairs (a physical ``A`` is even in ``n.g`` and ``Xi`` even in ``n.B0``, so a
    separable response has no odd content at all).

    Measured in ``L^2`` on the sphere, **not** on the coefficients.  The zonal family is not
    orthogonal there, so coefficient norms are not energies -- near-degenerate columns produce
    large cancelling coefficients and would report a departure that is not in the function.

    Returns fractions of ``||E||``: ``'non_separable_sector'`` (energy the even--even zonal
    block cannot reach), ``'rank1'`` (departure from rank one within that block), and
    ``'total'`` (distance to the best separable approximation).
    """
    from .gaunt import sphere_quadrature
    g = np.asarray(g_dir, np.float64); g = g / np.linalg.norm(g)
    b = np.asarray(b0_dir, np.float64); b = b / np.linalg.norm(b)
    dirs, w = sphere_quadrature(n_theta, n_phi)
    E = np.asarray(response(dirs))
    if np.iscomplexobj(E):
        E = np.abs(E)
    sw = np.sqrt(w)
    u, v = dirs @ g, dirs @ b
    n1, n2 = l_g // 2 + 1, l_b // 2 + 1
    Pu = np.stack([eval_legendre(2 * i, u) for i in range(n1)])
    Pv = np.stack([eval_legendre(2 * j, v) for j in range(n2)])

    def fit(cols):
        A = cols * sw[:, None]
        c, *_ = np.linalg.lstsq(A, E * sw, rcond=None)
        return cols @ c

    ee = np.stack([Pu[i] * Pv[j] for i in range(n1) for j in range(n2)], axis=1)
    E_ee = fit(ee)                                   # best even--even zonal approximation

    # best rank one within that block: alternating least squares on A(u) and Xi(v).
    # Converges by ~20 sweeps on the responses tested; the default leaves margin.
    a = np.ones(n1); a[0] = 1.0
    for _ in range(n_als):
        Xa = (Pu.T @ a)                              # A(u) on the grid
        M = (Pv * Xa[None, :]).T * sw[:, None]
        cb, *_ = np.linalg.lstsq(M, E_ee * sw, rcond=None)
        Xb = Pv.T @ cb
        M = (Pu * Xb[None, :]).T * sw[:, None]
        a, *_ = np.linalg.lstsq(M, E_ee * sw, rcond=None)
    E_sep = (Pu.T @ a) * (Pv.T @ cb)

    nE = np.linalg.norm(E * sw)
    return {"non_separable_sector": float(np.linalg.norm((E - E_ee) * sw) / nE),
            "rank1": float(np.linalg.norm((E_ee - E_sep) * sw) / nE),
            "total": float(np.linalg.norm((E - E_sep) * sw) / nE)}


def coupled_spectrum_at(response, g_dir, b0_dir, l_g=8, l_b=6, n_theta=48, n_phi=96,
                        chiral=True, _grid=None):
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
    dirs, w = _grid if _grid is not None else sphere_quadrature(n_theta, n_phi)
    E = np.asarray(response(dirs))
    u, v = dirs @ g, dirs @ b
    terms = [(l1, l2, 0) for l1 in range(l_g + 1) for l2 in range(l_b + 1)
             if (l1 + l2) % 2 == 0]
    kv = np.cross(g, b)
    s_k = float(np.linalg.norm(kv))
    if chiral and s_k > 1e-12:
        khat = kv / s_k
        wt = dirs @ khat
        terms += [(l1, l2, 1) for l1 in range(l_g + 1) for l2 in range(l_b + 1)
                  if (l1 + l2) % 2 == 1]
    else:
        wt = np.zeros(dirs.shape[0])
    B = np.stack([eval_legendre(l1, u) * eval_legendre(l2, v) * (wt ** p)
                  for l1, l2, p in terms], axis=1)
    sw = np.sqrt(w)
    # a replayed response is complex; the fit is linear, so solve both parts and keep the
    # phase rather than silently discarding it
    A, y = B * sw[:, None], E * sw
    if np.iscomplexobj(y):
        lr, *_ = np.linalg.lstsq(A, y.real, rcond=None)
        li, *_ = np.linalg.lstsq(A, y.imag, rcond=None)
        lam = lr + 1j * li
    else:
        lam, *_ = np.linalg.lstsq(A, y, rcond=None)
    sv = np.linalg.svd(A, compute_uv=False)
    rank = int((sv > sv[0] * 1e-10).sum())
    resid = float(np.linalg.norm(A @ lam - y) / max(np.linalg.norm(y), 1e-300))
    out = {"terms": terms, "coeffs": lam, "l_g": int(l_g), "l_b": int(l_b),
           "chiral": bool(chiral and s_k > 1e-12)}
    return out, resid, rank


def apply_odf_coupled(lam, odf_sh, g_dir, b0_dir, l_fod=8, l_g=8, l_b=6):
    """Compose a response over an orientation distribution (paper Eq. 19).

    ``lam`` is the dict returned by :func:`coupled_spectrum_at`: terms ``(l1, l2, p)`` with
    ``p=1`` marking the chiral sector.

    The ``p=0`` terms contract through the three-index Gaunt coefficients.  The ``p=1`` terms
    carry the extra zonal factor ``P_1(n . k)`` with ``k = (g x B0)/|g x B0|`` and so contract
    through a four-index table -- the same construction with one more harmonic.  That sector is
    what represents a handed substrate, whose response sees the triple product
    ``n . (g x B0)`` and is therefore not a function of the two invariants alone.

    Reduces exactly to :func:`apply_odf` when only ``(l1, 0, 0)`` terms are populated.
    """
    from .gaunt import gaunt_table, real_sh, sh_block, n_sh_coeffs
    f = np.asarray(odf_sh)
    nf = n_sh_coeffs(l_fod)
    if f.size < nf:
        f = np.concatenate([f, np.zeros(nf - f.size, f.dtype)])
    f = f[:nf]
    g = np.asarray(g_dir, np.float64); g = g / np.linalg.norm(g)
    b = np.asarray(b0_dir, np.float64); b = b / np.linalg.norm(b)

    terms, coeffs = lam["terms"], np.asarray(lam["coeffs"])
    lg, lb = int(lam["l_g"]), int(lam["l_b"])

    need_chiral = any(p for _, _, p in terms)
    G3 = gaunt_table(l_fod, lg, lb, full_g=True, full_b=True)
    Yg = real_sh(lg, g[None, :], full=True)[0]
    Yb = real_sh(lb, b[None, :], full=True)[0]
    if need_chiral:
        kv = np.cross(g, b); s_k = np.linalg.norm(kv)
        khat = kv / s_k if s_k > 1e-12 else np.array([0., 0., 1.])
        G4 = gaunt_table(l_fod, lg, lb, full_g=True, full_b=True, l_k=1)
        Yk = real_sh(1, khat[None, :], full=True)[0][sh_block(1, True)]

    out = 0.0
    for (l1, l2, p), c in zip(terms, coeffs):
        if c == 0:
            continue
        wgt = (4 * np.pi) ** 2 / ((2 * l1 + 1) * (2 * l2 + 1))
        b1 = sh_block(l1, True); b2 = sh_block(l2, True)
        if p == 0:
            blk = np.einsum("abc,a,b,c->", G3[:nf, b1, b2], f, Yg[b1], Yb[b2])
        else:
            wgt *= 4 * np.pi / 3.0                     # the extra P_1(n.k) factor
            blk = np.einsum("abcd,a,b,c,d->", G4[:nf, b1, b2, :], f, Yg[b1], Yb[b2], Yk)
        out = out + c * wgt * blk
    return out


def _rotations_from_axis(axis, dirs):
    """Stack of rotations taking ``axis`` to each row of ``dirs`` (Rodrigues, vectorised)."""
    a = np.asarray(axis, np.float64); a = a / np.linalg.norm(a)
    n = np.asarray(dirs, np.float64)
    n = n / np.linalg.norm(n, axis=1, keepdims=True)
    v = np.cross(a, n)                                   # (N, 3)
    c = n @ a                                            # (N,)
    R = np.broadcast_to(np.eye(3), (n.shape[0], 3, 3)).copy()
    K = np.zeros((n.shape[0], 3, 3))
    K[:, 0, 1], K[:, 0, 2] = -v[:, 2], v[:, 1]
    K[:, 1, 0], K[:, 1, 2] = v[:, 2], -v[:, 0]
    K[:, 2, 0], K[:, 2, 1] = -v[:, 1], v[:, 0]
    ok = c > -1.0 + 1e-12
    denom = np.where(ok, 1.0 + c, 1.0)[:, None, None]
    R = R + K + (K @ K) / denom
    if (~ok).any():                                      # antipodal: pi about any perpendicular
        p = np.eye(3)[np.argmin(np.abs(a))]
        p = p - (p @ a) * a; p /= np.linalg.norm(p)
        R[~ok] = 2.0 * np.outer(p, p) - np.eye(3)
    return R


def pack_response(pack, profile, g_dir, b0_dir, *, amplitude=1.0, B0=3.0,
                  chi_iso=0.0, chi_aniso=0.0, refocus_time=None,
                  fibre_axis=(0., 0., 1.), relaxation=True, chunk=256):
    """Response callable for :func:`coupled_spectrum_at`, backed by a real replay pack.

    Returns ``response(dirs) -> (n_dirs,)`` complex signal: the pack replayed as though its
    substrate pointed along each ``dirs`` row, which by pose covariance is the pack replayed at
    the counter-rotated ``(g, B0)``.

    Nothing is reconstructed.  The gradient phase is linear in position and the susceptibility
    field is linear in its stored basis, so both reduce to per-walker constants obtained by
    contracting the stored coefficients once:

        q_w   = gamma dt amp * sum_k profile_k * C_{w,k,:}      (a 3-vector per walker)
        Psi_w = gamma dt     * sum_k gate_k * B_{w,c,k}         (13 per walker)

    after which each orientation costs ``phi_G = g' . q_w`` and
    ``phi_chi = chi_iso B0 (Psi_0 - Q(b0').Psi_P) + chi_aniso B0 Q(b0').Psi_G`` -- the ``Q(H)``
    contraction being quadratic in the field direction, so the field orientation factors out of
    the walk exactly as it does in Eq. (15).  The two phases are summed *before* the ensemble
    average, so the diffusion x susceptibility cross-term is kept: that is what makes the
    resulting coupled spectrum able to depart from rank one.
    """
    from scipy.fft import dct
    from .compression import read_position_coeffs, decode_occupancy, relaxation_logweight
    from .constants import GAMMA
    from .susceptibility_field import _q_of_H

    arrays, meta = pack.arrays, pack.meta
    comp_meta = meta.get("compression", {})
    chans = comp_meta.get("channels", {}) or {}
    pm = chans.get("susceptibility_path")
    if pm is None:
        raise ValueError("pack_response needs the susc_path channel (C3 path route); this pack "
                         "carries none. Grid-route packs must decode positions instead.")
    dt = float(pack.dt)
    n_t = int(pm["n_t"])
    prof = np.asarray(profile, np.float64)
    if prof.size != n_t:
        raise ValueError(f"profile has {prof.size} samples, pack walk has n_t={n_t}")

    C = read_position_coeffs(arrays, dtype=np.float64)                 # (n_w, K, 3)
    n_w, K = C.shape[0], C.shape[1]
    prof_hat = dct(prof, type=2, norm="ortho")[:K]
    q = (GAMMA * dt * amplitude) * np.einsum("k,wkd->wd", prof_hat, C)  # (n_w, 3)

    from .bank import susc_path_coeffs
    Cs, names = susc_path_coeffs(arrays, pm)                           # (n_w, 13, Ks), dequantised
    Ks = Cs.shape[2]
    gate = _se_gate_local(n_t, dt, refocus_time)
    gate_hat = dct(gate, type=2, norm="ortho")[:Ks]
    Psi = (GAMMA * dt) * np.einsum("k,wck->wc", gate_hat, Cs)          # (n_w, n_ch)
    i_p = names.index("iso_P_xx")
    Psi0, PsiP = Psi[:, names.index("iso_local")], Psi[:, i_p:i_p + 6]
    has_aniso = "aniso_G_xx" in names
    PsiG = Psi[:, names.index("aniso_G_xx"):][:, :6] if has_aniso else None

    w0 = np.asarray(arrays.get("spin_weights", np.ones(n_w)), np.float64)
    ew = w0.copy()
    if relaxation and "comp_rle_vals" in arrays and meta.get("per_comp", {}).get("T2"):
        comp = decode_occupancy(arrays, chans["compartment"])["comp"]
        pc = meta["per_comp"]
        ew = ew * np.exp(relaxation_logweight(comp, pc["T2"], pc.get("T1"), dt))
    norm = w0.sum()

    g = np.asarray(g_dir, np.float64); g /= np.linalg.norm(g)
    b = np.asarray(b0_dir, np.float64); b /= np.linalg.norm(b)

    def response(dirs):
        dirs = np.asarray(dirs, np.float64).reshape(-1, 3)
        out = np.empty(dirs.shape[0], np.complex128)
        for s in range(0, dirs.shape[0], chunk):
            R = _rotations_from_axis(fibre_axis, dirs[s:s + chunk])     # (n, 3, 3)
            gp = np.einsum("nji,j->ni", R, g)                           # R^T g
            bp = np.einsum("nji,j->ni", R, b)                           # R^T B0
            phi = gp @ q.T                                              # (n, n_w)
            Q = np.stack([bp[:, 0] ** 2, bp[:, 1] ** 2, bp[:, 2] ** 2,
                          2 * bp[:, 0] * bp[:, 1], 2 * bp[:, 0] * bp[:, 2],
                          2 * bp[:, 1] * bp[:, 2]], axis=1)             # (n, 6)
            if chi_iso:
                phi = phi + chi_iso * B0 * (Psi0[None, :] - Q @ PsiP.T)
            if chi_aniso and PsiG is not None:
                phi = phi + chi_aniso * B0 * (Q @ PsiG.T)
            out[s:s + chunk] = (ew[None, :] * np.exp(1j * phi)).sum(1) / norm
        return out

    return response


def _se_gate_local(n_t, dt, refocus_time):
    """Transverse-phase gate; mirrors bank._se_gate (kept local to avoid a bank import cycle)."""
    if refocus_time is None:
        return np.ones(n_t)
    t = np.arange(n_t) * dt
    s = np.sign(refocus_time - t).astype(float)
    d = int(round(s.sum()))
    if d != 0:
        side = np.where(s == np.sign(d))[0]
        s[side[np.argsort(-np.abs(t[side] - refocus_time))[:abs(d)]]] = 0.0
    return s


class PackResponder:
    """Pack + field direction + gradient profile, with everything direction-independent hoisted.

    Building ``Lambda`` per measurement re-did work that does not depend on the gradient
    direction: ``q_w`` follows the gradient *profile*, ``Psi_w`` the refocusing gate, and
    ``Q(R^T B0)`` -- hence the whole susceptibility phase -- only ``B0`` and the quadrature.  At
    fixed ``B0`` that phase is identical for every gradient direction in a shell.  This object
    computes it once and reuses it, which is what makes a shell affordable.

    ``backend='jax'`` runs the remaining kernel -- an elementwise map and a reduction over
    (directions x walkers) -- on an accelerator.  Correctness is identical; only the timing
    differs.
    """

    def __init__(self, pack, profile, b0_dir, *, amplitude=1.0, B0=3.0, chi_iso=0.0,
                 chi_aniso=0.0, refocus_time=None, fibre_axis=(0., 0., 1.),
                 relaxation=True, n_theta=32, n_phi=64, backend="numpy", chunk=512):
        from scipy.fft import dct
        from .gaunt import sphere_quadrature
        from .compression import read_position_coeffs, decode_occupancy, relaxation_logweight
        from .constants import GAMMA

        arrays, meta = pack.arrays, pack.meta
        chans = (meta.get("compression", {}).get("channels", {}) or {})
        pm = chans.get("susceptibility_path")
        if pm is None:
            raise ValueError("PackResponder needs the susc_path channel (C3 path route)")
        dt = float(pack.dt)
        n_t = int(pm["n_t"])
        prof = np.asarray(profile, np.float64)
        if prof.size != n_t:
            raise ValueError(f"profile has {prof.size} samples, pack walk has n_t={n_t}")

        C = read_position_coeffs(arrays, dtype=np.float64)
        n_w, K = C.shape[0], C.shape[1]
        self.q = (GAMMA * dt * amplitude) * np.einsum(
            "k,wkd->wd", dct(prof, type=2, norm="ortho")[:K], C)          # profile, not direction

        from .bank import susc_path_coeffs
        Cs, names = susc_path_coeffs(arrays, pm)                        # dequantised, zz re-inserted
        gate = dct(_se_gate_local(n_t, dt, refocus_time), type=2, norm="ortho")[:Cs.shape[2]]
        Psi = (GAMMA * dt) * np.einsum("k,wck->wc", gate, Cs)             # gate, not direction

        self.dirs, self.w = sphere_quadrature(n_theta, n_phi)
        self.b0 = np.asarray(b0_dir, np.float64) / np.linalg.norm(b0_dir)
        self.R = _rotations_from_axis(fibre_axis, self.dirs)
        bp = np.einsum("nji,j->ni", self.R, self.b0)
        Q = np.stack([bp[:, 0] ** 2, bp[:, 1] ** 2, bp[:, 2] ** 2,
                      2 * bp[:, 0] * bp[:, 1], 2 * bp[:, 0] * bp[:, 2],
                      2 * bp[:, 1] * bp[:, 2]], axis=1)
        i_p = names.index("iso_P_xx")
        phi_chi = np.zeros((self.dirs.shape[0], n_w))
        if chi_iso:
            phi_chi += chi_iso * B0 * (Psi[:, names.index("iso_local")][None, :]
                                       - Q @ Psi[:, i_p:i_p + 6].T)
        if chi_aniso and "aniso_G_xx" in names:
            ia = names.index("aniso_G_xx")
            phi_chi += chi_aniso * B0 * (Q @ Psi[:, ia:ia + 6].T)
        # the shared factor: exp(i phi_chi) is the same for every gradient direction at fixed B0
        self.Echi = np.exp(1j * phi_chi.astype(np.float32))

        w0 = np.asarray(arrays.get("spin_weights", np.ones(n_w)), np.float64)
        ew = w0.copy()
        if relaxation and "comp_rle_vals" in arrays and meta.get("per_comp", {}).get("T2"):
            comp = decode_occupancy(arrays, chans["compartment"])["comp"]
            pc = meta["per_comp"]
            ew = ew * np.exp(relaxation_logweight(comp, pc["T2"], pc.get("T1"), dt))
        self.norm = float(w0.sum())
        self.backend, self.chunk = backend, int(chunk)
        if backend == "jax":
            import jax, jax.numpy as jnp
            self._jq = jnp.asarray(self.q, jnp.float32)
            self._jE = jnp.asarray(self.Echi * ew[None, :].astype(np.float32))

            @jax.jit
            def _kern(gp, q, Ew):
                return (Ew * jnp.exp(1j * (gp @ q.T))).sum(1)
            self._kern = _kern
            self._jnp = jnp
        else:
            self.ew = ew

    def evaluate(self, g_dir):
        """Replayed signal on the responder's quadrature grid for one gradient direction."""
        g = np.asarray(g_dir, np.float64); g = g / np.linalg.norm(g)
        gp = np.einsum("nji,j->ni", self.R, g)
        if self.backend == "jax":
            jnp = self._jnp
            out = self._kern(jnp.asarray(gp, jnp.float32), self._jq, self._jE)
            return np.asarray(out) / self.norm
        out = np.empty(self.dirs.shape[0], np.complex128)
        for s in range(0, self.dirs.shape[0], self.chunk):
            pg = (gp[s:s + self.chunk] @ self.q.T).astype(np.float32)
            out[s:s + self.chunk] = ((self.ew[None, :] * self.Echi[s:s + self.chunk])
                                     * np.exp(1j * pg)).sum(1)
        return out / self.norm

    def spectrum_at(self, g_dir, l_g=8, l_b=6, chiral=True):
        """``Lambda`` for one gradient direction, projected on this responder's own grid."""
        return coupled_spectrum_at(lambda d: self.evaluate(g_dir), g_dir, self.b0,
                                   l_g=l_g, l_b=l_b, n_theta=None, n_phi=None,
                                   chiral=chiral, _grid=(self.dirs, self.w))


def free_water_response(b_value, diffusivity):
    """Closed-form isotropic Gaussian response, ``exp(-b D)`` -- the analytic substrate.

    Free water is the case where storing walkers is not merely wasteful but unusable. Its
    signal decays exponentially in ``b`` while a Monte-Carlo floor decays only as
    ``1/sqrt(N_w)``, so the *relative* error grows like ``exp(+bD)/sqrt(N_w)``.  Measured at
    ``D = 3.0e-9`` with 4000 walkers, the floor is 0.1x the signal at ``b = 1000 s/mm^2``, 113x
    at 3000 and 8.5e4 at 5000; reaching 1% relative accuracy at ``b = 3000`` would take
    ~5e11 walkers and at 5000 ~3e17.  The closed form has no such error, and is also
    orientation-independent, so it needs no orientation distribution.
    """
    return float(np.exp(-np.asarray(b_value, np.float64) * float(diffusivity)))


def compose_voxel(spectra, substrate_id, geometric_fraction, orientation, m0,
                  g_dir, b0_dir, *, l_fod=8, l_g=8, l_b=6, signal_bearing=None, atol=1e-3):
    """One voxel of a replay phantom (RPH.md Sec. 6).

        S_v = sum_p  geometric_fraction[p] * m0[substrate_id[p]] * <F_p, E_{substrate_id[p]}>

    ``spectra`` is indexed by substrate.  An entry that is a spectrum dict is composed against
    that slot's orientation; an entry that is a **scalar** is an analytic, orientation-
    independent response (see :func:`free_water_response`) and is used as it stands.
    ``orientation`` is ``(P, n_c)`` SH coefficients.  ``signal_bearing`` marks which substrates
    carry a response at all; an inert substrate contributes nothing.  Slots with
    ``substrate_id < 0`` are skipped.

    **Fractions must sum to one.** A voxel is always full -- there is no vacuum in a sample --
    so a row that sums to less is not a voxel with a void in it, it is a voxel whose remainder
    was not modelled, and composing it would return a signal that is quietly too low.  Anything
    that is not tissue is declared as a substrate: background air is a non-signal-bearing
    substrate with ``m0 = 0``, and free water is a substrate with its own pack.  Partial volume
    is carried by the *ratio* of the fractions, which needs no slack in the sum.
    """
    sid = np.asarray(substrate_id)
    frac = np.asarray(geometric_fraction, np.float64)
    ori = np.asarray(orientation)
    m0 = np.asarray(m0, np.float64)
    live = sid >= 0
    total = float(frac[live].sum())
    if abs(total - 1.0) > atol:
        raise ValueError(
            f"geometric_fraction sums to {total:.6g}, not 1 (tol {atol:g}). A voxel is always "
            f"full: declare what occupies the rest -- background air as a non-signal-bearing "
            f"substrate with m0=0, free water as a substrate with its own pack -- rather than "
            f"leaving a remainder, which would return a signal that is quietly too low "
            f"(RPH.md Sec. 3).")
    out = 0.0
    for p in np.flatnonzero(live):
        i = int(sid[p])
        if frac[p] == 0.0:
            continue
        if signal_bearing is not None and not signal_bearing[i]:
            continue                                     # inert: occupies volume, emits nothing
        sp = spectra[i]
        if isinstance(sp, dict):
            resp = apply_odf_coupled(sp, ori[p], g_dir, b0_dir,
                                     l_fod=l_fod, l_g=l_g, l_b=l_b)
        else:
            resp = sp                                    # analytic, orientation-independent
        out = out + frac[p] * m0[i] * resp
    return out
