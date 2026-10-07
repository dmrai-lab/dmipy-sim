"""The eigenmode (matrix) solution of restricted diffusion in a planar, cylindrical or spherical pore under
a generalised gradient waveform -- Callaghan's 1997 matrix formalism (J. Magn. Reson. 129, 74), extended to
these three pores by Codd & Callaghan 1999 (J. Magn. Reson. 137:358-372, doi:10.1006/jmre.1998.1679), and
restated by Barzykin 1999 (J. Magn. Reson. 139, 342) and Grebenkov's review (Rev. Mod. Phys. 79, 1077, 2007).

The Bloch-Torrey equation ``dm/dt = D grad^2 m - i gamma (g(t).r) m`` with reflecting (Neumann) walls is
solved in the Laplacian eigenbasis: expand ``m`` in the orthonormal eigenfunctions of the pore (``u_0``
constant, eigenvalue 0). Over a step of duration ``tau`` at a constant gradient ``g`` along the measured axis
the coefficients advance by ``c <- exp(-tau (D Lambda + i gamma g B)) c``, with ``Lambda`` the eigenvalues and
``B_{mn} = <u_m| ghat.r |u_n>`` the position-projection matrix; the signal is the echo's ``(0, 0)`` entry of
the ordered propagator product. This is exact up to the number of eigenmodes kept, whose convergence is an
argument rather than assumed; the Gaussian-phase models (Van Gelderen for the cylinder, Murday-Cotts for the
sphere) are its low-b limit, Callaghan's narrow-pulse expression its short-delta limit.

A piecewise-constant waveform has a handful of distinct RUNS of constant amplitude, so the exact product costs
one matrix exponential per run (:func:`_runs`) rather than one per sample -- a slew-limited PGSE has a few
amplitudes from its ramps but just as few runs and stays exact at that cost. A waveform whose amplitude changes
at (almost) every sample (an OGSE cosine) is genuinely dense; past ``_MAX_EXACT_RUNS`` runs the module falls
back to a Strang split of the diffusion and precession operators, whose Trotter error is ``O(dt^2)``.

The eigenvalue roots and the dimensionless position matrix are quadratured once per ``(shape, n_modes)`` on
the UNIT geometry (radius or thickness 1) and cached; a pore of size ``a`` reads ``lambda = lambda_unit / a**2``
and ``B = a * B_unit`` (a scalar times a cached matrix, not a requadrature), so a sweep over pore size pays
one eigendecomposition. The eigenfunction norms and ``B`` come from direct quadrature rather than closed-form
Lommel integrals, so the only approximations are the number of modes and the quadrature resolution, both
arguments (module constants, since both shapes and sizes share one resolution).
"""
from __future__ import annotations

from functools import lru_cache

import numpy as np
from scipy import linalg, special
from scipy.optimize import brentq

from dmipy_sim.constants import GAMMA

#: Quadrature nodes across the unit pore's radius/thickness and around its angle.
RADIAL_NODES = 4000
ANGULAR_NODES = 2048

#: The run-product path costs one matrix exponential per run of constant amplitude (``_runs``), not per
#: distinct amplitude -- a slew-limited PGSE has a handful of amplitudes from its ramps but just as few runs
#: (12 at the ``pgse`` default 200 T/m/s, 32 at 50 T/m/s) and is exact at that cost. A waveform whose amplitude
#: changes at (almost) every sample -- an OGSE cosine, ~1000 runs at n_t=1000 -- is genuinely dense and this
#: ceiling sends it to the Strang split instead.
_MAX_EXACT_RUNS = 64

__all__ = ["MatrixPore", "matrix_restricted_signal", "matrix_restricted_batch"]


def _cyl_neumann_roots(m, n_roots):
    """The first ``n_roots`` non-negative roots of ``J_m'(x) = 0``; ``m = 0`` includes the constant mode at 0."""
    if m == 0:
        return np.concatenate([[0.0], special.jnp_zeros(0, n_roots - 1)])
    return special.jnp_zeros(m, n_roots)


def _sph_neumann_roots(l, n_roots):
    """The first ``n_roots`` non-negative roots of ``j_l'(x) = 0``; ``l = 0`` includes the constant mode at 0."""
    xs = np.linspace(1e-6, 4 * (n_roots + l + 2) + 20, 40000)
    sign = np.sign(special.spherical_jn(l, xs, derivative=True))
    roots = []
    for i in np.where(np.diff(sign) != 0)[0]:
        try:
            r = brentq(lambda x: special.spherical_jn(l, x, derivative=True), xs[i], xs[i + 1])
        except Exception:
            continue
        if not roots or r - roots[-1] > 1e-3:
            roots.append(r)
        if len(roots) >= (n_roots - 1 if l == 0 else n_roots):
            break
    roots = np.asarray(roots)
    return np.concatenate([[0.0], roots]) if l == 0 else roots


def _plane_modes_unit(n_max, nr):
    """The unit slab ``x in [0, 1]`` with reflecting walls, gradient along its normal: ``cos(n pi x)`` with
    eigenvalue ``(n pi)^2``, and ``B`` the position operator centred on the slab's midplane -- the same
    convention the cylinder/sphere get for free from their radial coordinate running outward from the pore's
    centre, so ``B[0, 0] == 0`` and a non-refocused waveform carries no spurious global phase."""
    x = np.linspace(0.0, 1.0, nr)
    xc = x - 0.5
    U = [np.cos(n * np.pi * x) for n in range(n_max + 1)]
    lam = np.asarray([(n * np.pi) ** 2 for n in range(n_max + 1)])
    norm = [np.trapezoid(u * u, x) for u in U]
    N = len(U)
    B = np.zeros((N, N))
    for i in range(N):
        for j in range(N):
            B[i, j] = np.trapezoid(U[i] * xc * U[j], x) / np.sqrt(norm[i] * norm[j])
    return lam, B


def _cyl_modes_unit(m_max, n_max, nr, nt):
    """The unit disk ``r in [0, 1]``, gradient across it: states ``(m, n)`` with radial ``J_m(alpha r)`` and
    angular ``cos(m theta)``, orthonormal on ``r dr dtheta``, and ``B`` the operator ``x = r cos theta``."""
    r = np.linspace(0.0, 1.0, nr)
    states, lam, rnorm, rfun = [], [], [], []
    for m in range(m_max + 1):
        for n, al in enumerate(_cyl_neumann_roots(m, n_max)):
            R = special.jv(m, al * r)
            states.append((m, n)); lam.append(al ** 2)
            rnorm.append(np.trapezoid(R * R * r, r)); rfun.append(R)
    t = np.linspace(0.0, 2 * np.pi, nt)
    ang_norm = [2 * np.pi if m == 0 else np.pi for m in range(m_max + 2)]
    N = len(states)
    B = np.zeros((N, N))
    for i in range(N):
        mi, _ = states[i]
        for j in range(N):
            mj, _ = states[j]
            if abs(mi - mj) != 1:                       # cos t couples only neighbouring angular orders
                continue
            rad = np.trapezoid(rfun[i] * rfun[j] * r * r, r)
            ang = np.trapezoid(np.cos(mi * t) * np.cos(t) * np.cos(mj * t), t)
            B[i, j] = rad * ang / np.sqrt(rnorm[i] * ang_norm[mi] * rnorm[j] * ang_norm[mj])
    return np.asarray(lam), B


def _sph_modes_unit(l_max, n_max, nr, nt):
    """The unit ball ``r in [0, 1]``: states ``(l, n)`` with radial ``j_l(beta r)`` and angular ``P_l(cos th)``,
    orthonormal on ``r^2 dr sin th dth dphi``, and ``B`` the operator ``z = r cos th``."""
    r = np.linspace(0.0, 1.0, nr)
    states, lam, rnorm, rfun = [], [], [], []
    for l in range(l_max + 1):
        for n, be in enumerate(_sph_neumann_roots(l, n_max)):
            R = special.spherical_jn(l, be * r)
            states.append((l, n)); lam.append(be ** 2)
            rnorm.append(np.trapezoid(R * R * r * r, r)); rfun.append(R)
    th = np.linspace(0.0, np.pi, nt)
    ct, st = np.cos(th), np.sin(th)
    P = {l: special.eval_legendre(l, ct) for l in range(l_max + 2)}
    ang_norm = {l: 2 * np.pi * np.trapezoid(P[l] * P[l] * st, th) for l in range(l_max + 2)}
    N = len(states)
    B = np.zeros((N, N))
    for i in range(N):
        li, _ = states[i]
        for j in range(N):
            lj, _ = states[j]
            if abs(li - lj) != 1:
                continue
            rad = np.trapezoid(rfun[i] * rfun[j] * r * r * r, r)
            ang = 2 * np.pi * np.trapezoid(P[li] * ct * P[lj] * st, th)
            B[i, j] = rad * ang / np.sqrt(rnorm[i] * ang_norm[li] * rnorm[j] * ang_norm[lj])
    return np.asarray(lam), B


@lru_cache(maxsize=None)
def _unit_modes(shape, n_modes, radial_nodes, angular_nodes):
    """``(lambda_unit, B_unit, beta_unit, U_unit)`` on the UNIT geometry (radius/thickness 1), cached per
    ``(shape, n_modes, radial_nodes, angular_nodes)`` -- computed once however many pore SIZES are evaluated,
    since a pore of size ``a`` scales as ``lambda = lambda_unit / a**2``, ``B = a * B_unit`` (eigenvectors of
    ``B`` are unchanged by the positive scalar ``a``, so ``beta = a * beta_unit``, ``U = U_unit``)."""
    if shape == "plane":
        if not isinstance(n_modes, (int, np.integer)):
            raise ValueError(f"a plane has one mode index: n_modes is the count of cosine modes (an int), not {n_modes!r}")
        lam, B = _plane_modes_unit(int(n_modes), radial_nodes)
    elif shape in ("cylinder", "sphere"):
        if isinstance(n_modes, (int, np.integer)) or len(n_modes) != 2:
            raise ValueError(f"a {shape} has an angular and a radial mode index: n_modes is the pair (angular, radial), not {n_modes!r}")
        m_max, n_max = (int(x) for x in n_modes)
        if shape == "cylinder":
            lam, B = _cyl_modes_unit(m_max, n_max, radial_nodes, angular_nodes)
        else:
            lam, B = _sph_modes_unit(m_max, n_max, radial_nodes, angular_nodes)
    else:
        raise ValueError(f"a closed form is known for a plane, a cylinder and a sphere, not {shape!r}")
    beta, U = np.linalg.eigh(B)
    return lam, B, beta, U


def _runs(g):
    """A per-step amplitude array as ``[(amplitude, n_steps)]``: what makes a piecewise-constant waveform cost
    a handful of matrix exponentials rather than one per step."""
    out, cur, cnt = [], g[0], 1
    for x in g[1:]:
        if x == cur:
            cnt += 1
        else:
            out.append((float(cur), cnt)); cur, cnt = x, 1
    out.append((float(cur), cnt))
    return out


#: the series truncation when none is given: the cosine count for the plane, the (angular, radial) pair for the
#: round pores
DEFAULT_MODES = {"plane": 24, "cylinder": (24, 18), "sphere": (24, 18)}


class MatrixPore:
    """The closed form of one pore: ``MatrixPore(shape, size_m, D).signal(g, dt)``.

    ``shape`` is ``"plane"``, ``"cylinder"`` or ``"sphere"``; ``size_m`` the plate separation (plane) or the
    diameter (cylinder, sphere); ``n_modes`` the truncation of the series -- the count of cosine modes (an int)
    for the plane, whose eigenfunctions carry one index, and the ``(angular, radial)`` pair for the cylinder and
    the sphere; :data:`DEFAULT_MODES` when not given. It is the only approximation besides the quadrature
    resolution, refined to measure the reference's own uncertainty.
    """

    def __init__(self, shape, size_m, D, *, n_modes=None):
        self.shape, self.D, self.size = str(shape), float(D), float(size_m)
        if n_modes is None:
            n_modes = DEFAULT_MODES.get(self.shape)
        key = int(n_modes) if isinstance(n_modes, (int, np.integer)) else tuple(int(x) for x in n_modes)
        lam_unit, B_unit, beta_unit, U_unit = _unit_modes(self.shape, key, RADIAL_NODES, ANGULAR_NODES)
        a = self.size if self.shape == "plane" else 0.5 * self.size
        self.lam = lam_unit / a ** 2
        self.B = a * B_unit
        self._beta = a * beta_unit
        self._U = U_unit

    def signal(self, g_axis, dt):
        """``S`` for the per-step gradient amplitudes ``g_axis`` (T/m along the measured axis) at step ``dt``.

        Exact for a piecewise-constant waveform: one matrix exponential per run of constant amplitude, a
        zero-amplitude run being pure mode decay and therefore diagonal -- so the cost is ``len(_runs(g))``,
        not the number of distinct amplitudes, and a slew-limited waveform (many amplitudes, few runs) stays
        on this path. A genuinely dense waveform (an OGSE cosine, amplitude changing almost every sample) goes
        through the Strang split instead, whose ``O(dt^2)`` error the caller bounds by refining ``dt``.
        """
        g = np.asarray(g_axis, np.float64)
        runs = _runs(g)
        if len(runs) > _MAX_EXACT_RUNS:
            return self._signal_strang(g, float(dt))
        M = np.eye(len(self.lam), dtype=complex)
        cache = {}
        for amp, ns in runs:
            if ns <= 0:
                continue
            tau = ns * float(dt)
            if amp == 0.0:
                M = np.exp(-tau * self.D * self.lam)[:, None] * M
                continue
            key = (round(amp, 18), ns)
            if key not in cache:
                cache[key] = linalg.expm(-tau * (self.D * np.diag(self.lam) + 1j * GAMMA * amp * self.B))
            M = cache[key] @ M
        return complex(M[0, 0])

    def _signal_strang(self, g, dt):
        half = np.exp(-0.5 * dt * self.D * self.lam)
        c = np.zeros(len(self.lam), complex)
        c[0] = 1.0
        for gk in g:
            c = half * c
            if gk != 0.0:
                c = self._U @ (np.exp(-1j * GAMMA * gk * dt * self._beta) * (self._U.T @ c))
            c = half * c
        return complex(c[0])


def matrix_restricted_signal(shape, g_axis, dt, D, size_m, *, n_modes=None):
    """The restricted signal's magnitude for a single projected 1-D gradient schedule (T/m) at step ``dt``
    (s), diffusivity ``D`` (m^2/s) and pore ``size_m`` -- the functional entry point for a caller (such as
    dmipy-fit's compartment models) that wants a number rather than :class:`MatrixPore`'s object API."""
    return float(abs(MatrixPore(shape, size_m, D, n_modes=n_modes).signal(g_axis, dt)))


def matrix_restricted_batch(shape, g_axes, dt, D, size_m, *, n_modes=None):
    """``matrix_restricted_signal`` over a stack of projected 1-D schedules ``g_axes`` (n_meas, n_t) sharing
    one ``dt``, ``D`` and ``size_m`` -- one :class:`MatrixPore` (one unit-mode cache hit, one eigendecomposition
    scaled once) evaluated per measurement rather than rebuilt per call."""
    pore = MatrixPore(shape, size_m, D, n_modes=n_modes)
    return np.array([float(abs(pore.signal(g, dt))) for g in g_axes])
