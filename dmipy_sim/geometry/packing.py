"""Packing helpers -- generate non-overlapping object positions for the packed geometries."""
import numpy as np


def pack_cylinders(radii, target_vf=None, L=None, seed=0, max_attempts=100_000):
    """Pack N parallel cylinders in a periodic 2-D square domain using RSA.

    Random Sequential Addition (RSA) places cylinders one by one, rejecting
    positions that would cause overlap with any previously placed cylinder
    (including periodic images).  Large cylinders are placed first to maximise
    the achievable packing fraction.  RSA typically reaches VF ≈ 0.45 for
    monodisperse and ≈ 0.55 for polydisperse populations.

    The returned centres are in the 2-D cross-section plane (the plane
    perpendicular to the cylinder axis).  Pass them to ``PackedCylinders``
    together with the same radii, L, and orientation.

    Parameters
    ----------
    radii : array-like, shape (N,)
        Cylinder radii in metres.  All must be positive.
    target_vf : float, optional
        Target intra-cylindrical volume fraction Σπrᵢ²/L².  Used to derive L.
        Mutually exclusive with ``L``.
    L : float, optional
        Side-length of the periodic square domain in metres.
        Mutually exclusive with ``target_vf``.
    seed : int
        NumPy RNG seed for reproducible packing.
    max_attempts : int
        Maximum random placement trials per cylinder before raising
        ``RuntimeError``.  Increase for high target_vf.

    Returns
    -------
    centers : np.ndarray, shape (N, 2)
        Cylinder centre positions in the 2-D cross-section plane, metres.
    L : float
        Side-length actually used.
    achieved_vf : float
        Achieved intra-cylindrical volume fraction = Σπrᵢ² / L².

    Raises
    ------
    ValueError
        On invalid inputs (conflicting L/target_vf, non-positive radii, radii
        exceeding L/2).
    RuntimeError
        If RSA cannot place a cylinder within ``max_attempts`` trials.  Try
        reducing ``target_vf`` or increasing ``max_attempts``.
    """
    radii = np.asarray(radii, dtype=np.float64).ravel()
    if len(radii) == 0:
        raise ValueError("radii must contain at least one element.")
    if np.any(radii <= 0):
        raise ValueError("All radii must be positive.")
    if (target_vf is None) == (L is None):
        raise ValueError("Provide exactly one of target_vf or L.")

    if target_vf is not None:
        if not 0.0 < float(target_vf) < 1.0:
            raise ValueError(f"target_vf must be in (0, 1), got {target_vf}.")
        L = float(np.sqrt(np.pi * np.sum(radii ** 2) / float(target_vf)))
    else:
        L = float(L)

    if np.any(radii > L / 2.0):
        raise ValueError(
            f"Largest radius ({np.max(radii) * 1e6:.2f} µm) exceeds L/2 "
            f"({L / 2.0 * 1e6:.2f} µm).  Reduce target_vf or supply a larger L.")

    rng = np.random.default_rng(int(seed))
    # Place largest cylinders first — improves RSA packing fraction.
    order = np.argsort(radii)[::-1]
    radii_s   = radii[order]
    centers_s = np.zeros((len(radii), 2))

    for i, r_new in enumerate(radii_s):
        placed = False
        for _ in range(max_attempts):
            c_new = rng.uniform(-L / 2.0, L / 2.0, 2)
            ok = True
            for j in range(i):
                dq = c_new - centers_s[j]
                dq -= L * np.round(dq / L)   # minimum-image distance
                if np.linalg.norm(dq) < r_new + radii_s[j]:
                    ok = False
                    break
            if ok:
                centers_s[i] = c_new
                placed = True
                break
        if not placed:
            raise RuntimeError(
                f"RSA failed after {max_attempts} attempts placing cylinder {i} "
                f"(r = {r_new * 1e6:.2f} µm).  "
                f"The target packing fraction may exceed what RSA can achieve; "
                f"try reducing target_vf or increasing max_attempts.")

    # Restore original cylinder ordering
    centers_out = np.empty_like(centers_s)
    centers_out[order] = centers_s
    achieved_vf = float(np.pi * np.sum(radii ** 2) / L ** 2)
    return centers_out, L, achieved_vf


def pack_spheres(radii, target_vf=None, L=None, seed=0, max_attempts=100_000):
    """Pack N spheres in a periodic 3-D cubic domain using RSA.

    Random Sequential Addition (RSA) places spheres one by one, rejecting
    positions that would cause overlap with any previously placed sphere
    (including periodic images).  Large spheres are placed first.  RSA
    typically reaches VF ≈ 0.38 for monodisperse spheres.

    Parameters
    ----------
    radii : array-like, shape (N,)
        Sphere radii in metres.  All must be positive.
    target_vf : float, optional
        Target intra-sphere volume fraction Σ(4/3)πrᵢ³/L³.  Used to derive L.
        Mutually exclusive with ``L``.
    L : float, optional
        Side-length of the periodic cubic domain in metres.
        Mutually exclusive with ``target_vf``.
    seed : int
        NumPy RNG seed for reproducible packing.
    max_attempts : int
        Maximum random placement trials per sphere before raising RuntimeError.

    Returns
    -------
    centers : np.ndarray, shape (N, 3)
        Sphere centre positions in metres.
    L : float
        Side-length actually used.
    achieved_vf : float
        Achieved volume fraction = Σ(4/3)πrᵢ³ / L³.

    Raises
    ------
    ValueError
        On invalid inputs.
    RuntimeError
        If RSA cannot place a sphere within ``max_attempts`` trials.
    """
    radii = np.asarray(radii, dtype=np.float64).ravel()
    if len(radii) == 0:
        raise ValueError("radii must contain at least one element.")
    if np.any(radii <= 0):
        raise ValueError("All radii must be positive.")
    if (target_vf is None) == (L is None):
        raise ValueError("Provide exactly one of target_vf or L.")

    if target_vf is not None:
        if not 0.0 < float(target_vf) < 1.0:
            raise ValueError(f"target_vf must be in (0, 1), got {target_vf}.")
        L = float(((4.0 / 3.0) * np.pi * np.sum(radii ** 3) / float(target_vf))
                  ** (1.0 / 3.0))
    else:
        L = float(L)

    if np.any(radii > L / 2.0):
        raise ValueError(
            f"Largest radius ({np.max(radii) * 1e6:.2f} µm) exceeds L/2 "
            f"({L / 2.0 * 1e6:.2f} µm).  Reduce target_vf or supply a larger L.")

    rng = np.random.default_rng(int(seed))
    order = np.argsort(radii)[::-1]   # largest first
    radii_s   = radii[order]
    centers_s = np.zeros((len(radii), 3))

    for i, r_new in enumerate(radii_s):
        placed = False
        for _ in range(max_attempts):
            c_new = rng.uniform(-L / 2.0, L / 2.0, 3)
            ok = True
            for j in range(i):
                dq = c_new - centers_s[j]
                dq -= L * np.round(dq / L)   # minimum-image
                if np.linalg.norm(dq) < r_new + radii_s[j]:
                    ok = False
                    break
            if ok:
                centers_s[i] = c_new
                placed = True
                break
        if not placed:
            raise RuntimeError(
                f"RSA failed after {max_attempts} attempts placing sphere {i} "
                f"(r = {r_new * 1e6:.2f} µm).  "
                f"The target packing fraction may exceed what RSA can achieve "
                f"(monodisperse RSA limit ≈ 0.38); try reducing target_vf or "
                f"increasing max_attempts.")

    centers_out = np.empty_like(centers_s)
    centers_out[order] = centers_s
    achieved_vf = float((4.0 / 3.0) * np.pi * np.sum(radii ** 3) / L ** 3)
    return centers_out, L, achieved_vf


def pack_myelinated_cylinders(inner_radii, g_ratios, target_packing,
                               cell_size=None, seed=0, max_attempts=100_000):
    """Place myelinated cylinders in a periodic square RVE using RSA.

    Each cylinder has an inner (axon) radius and an outer (myelin) radius
    given by outer_radius = inner_radius / g_ratio.  Placement ensures that
    outer boundaries do not overlap each other (including periodic images).

    Parameters
    ----------
    inner_radii : array-like, shape (N,)
        Axon radii in metres.
    g_ratios : array-like, shape (N,) or scalar
        g-ratio per cylinder (outer = inner / g_ratio).  Scalar is broadcast.
    target_packing : float
        Target packing fraction = sum(pi*R_outer^2) / cell_size^2.  Used to
        derive cell_size when ``cell_size`` is None.
    cell_size : float, optional
        Side-length of the periodic square cell in metres.  If provided,
        ``target_packing`` is ignored.
    seed : int
        NumPy RNG seed.
    max_attempts : int
        RSA placement attempts per cylinder.

    Returns
    -------
    inner_radii : np.ndarray, shape (N,)
    g_ratios    : np.ndarray, shape (N,)
    centers     : np.ndarray, shape (N, 2)
        Cylinder centers in metres, ``[-L/2, L/2)`` convention.
    """
    inner_radii = np.asarray(inner_radii, dtype=np.float64).ravel()
    N = len(inner_radii)
    g_ratios_arr = np.broadcast_to(
        np.asarray(g_ratios, dtype=np.float64).ravel(), (N,)).copy()
    outer_radii = inner_radii / g_ratios_arr

    if cell_size is None:
        if target_packing is None:
            raise ValueError("Provide either cell_size or target_packing.")
        cell_size = float(np.sqrt(np.pi * np.sum(outer_radii ** 2) / target_packing))
    L = float(cell_size)

    rng = np.random.default_rng(int(seed))
    order = np.argsort(outer_radii)[::-1]   # place largest first
    outer_s = outer_radii[order]
    centers_s = np.zeros((N, 2))

    for i, r_out in enumerate(outer_s):
        placed = False
        for _ in range(max_attempts):
            c = rng.uniform(-L / 2.0, L / 2.0, 2)
            ok = True
            for j in range(i):
                dq = c - centers_s[j]
                dq -= L * np.round(dq / L)
                if np.linalg.norm(dq) < r_out + outer_s[j]:
                    ok = False
                    break
            if ok:
                centers_s[i] = c
                placed = True
                break
        if not placed:
            raise RuntimeError(
                f"RSA failed after {max_attempts} attempts placing cylinder {i}.")

    centers_out = np.empty_like(centers_s)
    centers_out[order] = centers_s
    return inner_radii, g_ratios_arr, centers_out
