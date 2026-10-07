"""The closed form of restricted diffusion in a planar, cylindrical or spherical pore: the eigenmode (matrix)
solution of the Bloch-Torrey equation under a generalised gradient waveform.

This is the reference the canonical-pores family reproduces -- Codd & Callaghan 1999 (J. Magn. Reson.
137:358-372, doi:10.1006/jmre.1998.1679), who extend Callaghan's 1997 matrix formalism to exactly these three
pores. It is an independent computation: it shares no code path, no random number and no discretisation with the
Monte-Carlo walk, and its only error is where the series is cut off, which :class:`MatrixPore` makes measurable
by taking the mode counts as an argument.

The method. Expand the transverse magnetisation ``m`` in the orthonormal Neumann-Laplacian eigenfunctions of the
pore, ``u_0`` constant with eigenvalue 0. Over a step of duration ``tau`` at a constant gradient ``g`` along the
measured axis the coefficients advance by ``c <- exp(-tau (D Lambda + i gamma g B)) c``, with ``Lambda`` the
eigenvalues and ``B_{mn} = <u_m| ghat.r |u_n>`` the position-projection matrix; the signal is ``c_0`` at the
echo, i.e. the ``(0, 0)`` entry of the ordered product. A piecewise-constant waveform has a handful of distinct
amplitudes, so the whole product is a handful of matrix exponentials however fine its grid.

The eigenfunction norms and ``B`` are computed by direct quadrature rather than from closed-form Lommel
integrals, so the only approximations are the number of modes and the quadrature resolution, both of which are
arguments. GPA (van Gelderen for the cylinder, Murday-Cotts for the sphere) is this solution's low-b limit,
Callaghan's narrow-pulse expression its short-delta limit, and MISST is the same matrix method computed
elsewhere.
"""
from __future__ import annotations

import numpy as np
from scipy import linalg, special
from scipy.optimize import brentq

#: rad/s/T, the proton gyromagnetic ratio -- the same value as ``dmipy_sim.constants.GAMMA``.
GAMMA = 2.6751525e8

#: Quadrature nodes across the pore's radius and around its angle. A thumb rule would be a fourth
#: approximation with no argument to refine; these are the values the truncation study was run at.
RADIAL_NODES = 4000
ANGULAR_NODES = 2048

#: The run-product path costs one matrix exponential per run of constant amplitude (``_runs``), not per
#: distinct amplitude -- a slew-limited PGSE has a handful of amplitudes from its ramps but just as few runs
#: (12 at the ``pgse`` default 200 T/m/s, 32 at 50 T/m/s) and is exact at that cost. A waveform whose amplitude
#: changes at (almost) every sample -- an OGSE cosine, ~1000 runs at n_t=1000 -- is genuinely dense and this
#: ceiling sends it to the Strang split instead.
_MAX_EXACT_RUNS = 64

__all__ = ["MatrixPore", "GAMMA"]


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


def _plane_modes(L, n_max):
    """The slab of thickness ``L`` with reflecting walls, gradient along its normal: ``cos(n pi x / L)`` with
    eigenvalue ``(n pi / L)^2``, and ``B`` the position operator centred on the slab's midplane -- the same
    convention ``_cyl_modes``/``_sph_modes`` get for free from their radius running outward from the pore's
    centre, so ``B[0, 0] == 0`` and a non-refocused waveform carries no spurious global phase."""
    x = np.linspace(0.0, L, RADIAL_NODES)
    xc = x - 0.5 * L
    U = [np.cos(n * np.pi * x / L) for n in range(n_max + 1)]
    lam = np.asarray([(n * np.pi / L) ** 2 for n in range(n_max + 1)])
    norm = [np.trapezoid(u * u, x) for u in U]
    N = len(U)
    B = np.zeros((N, N))
    for i in range(N):
        for j in range(N):
            B[i, j] = np.trapezoid(U[i] * xc * U[j], x) / np.sqrt(norm[i] * norm[j])
    return lam, B


def _cyl_modes(a, m_max, n_max):
    """The disk of radius ``a``, gradient across it: states ``(m, n)`` with radial ``J_m(alpha r / a)`` and
    angular ``cos(m theta)``, orthonormal on ``r dr dtheta``, and ``B`` the operator ``x = r cos theta``."""
    r = np.linspace(0.0, a, RADIAL_NODES)
    states, lam, rnorm, rfun = [], [], [], []
    for m in range(m_max + 1):
        for n, al in enumerate(_cyl_neumann_roots(m, n_max)):
            R = special.jv(m, al * r / a)
            states.append((m, n)); lam.append((al / a) ** 2)
            rnorm.append(np.trapezoid(R * R * r, r)); rfun.append(R)
    t = np.linspace(0.0, 2 * np.pi, ANGULAR_NODES)
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


def _sph_modes(a, l_max, n_max):
    """The ball of radius ``a``: states ``(l, n)`` with radial ``j_l(beta r / a)`` and angular ``P_l(cos th)``,
    orthonormal on ``r^2 dr sin th dth dphi``, and ``B`` the operator ``z = r cos th``."""
    r = np.linspace(0.0, a, RADIAL_NODES)
    states, lam, rnorm, rfun = [], [], [], []
    for l in range(l_max + 1):
        for n, be in enumerate(_sph_neumann_roots(l, n_max)):
            R = special.spherical_jn(l, be * r / a)
            states.append((l, n)); lam.append((be / a) ** 2)
            rnorm.append(np.trapezoid(R * R * r * r, r)); rfun.append(R)
    th = np.linspace(0.0, np.pi, ANGULAR_NODES)
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


class MatrixPore:
    """The closed form of one pore: ``MatrixPore(shape, size_m, D).signal(g, dt)``.

    ``shape`` is ``"plane"``, ``"cylinder"`` or ``"sphere"``; ``size_m`` the plate separation or the diameter;
    ``n_modes`` the ``(angular, radial)`` truncation of the series, which is the only approximation and is
    refined to measure the reference's own uncertainty.
    """

    def __init__(self, shape, size_m, D, *, n_modes=(24, 18)):
        self.shape, self.D, self.size = str(shape), float(D), float(size_m)
        m_max, n_max = (int(x) for x in n_modes)
        if self.shape == "plane":
            self.lam, self.B = _plane_modes(self.size, m_max * n_max)
        elif self.shape == "cylinder":
            self.lam, self.B = _cyl_modes(0.5 * self.size, m_max, n_max)
        elif self.shape == "sphere":
            self.lam, self.B = _sph_modes(0.5 * self.size, m_max, n_max)
        else:
            raise ValueError(f"a closed form is known for a plane, a cylinder and a sphere, not {shape!r}")
        self._beta, self._U = np.linalg.eigh(self.B)

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
