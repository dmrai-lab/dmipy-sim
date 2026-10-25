"""The walk's step law: what one isotropic step of the walk does to diffusion, and the step length law the
engine draws (dmrai-lab/dmipy-sim#714, #715).

**The mechanism.** A step of length ``delta`` in a direction uniform over the sphere has a normal component
uniform on ``[-delta, delta]``, so it multiplies a bulk mode ``exp(i k x)`` by ``sinc(k delta)``; a step whose
length follows a law ``p(delta)`` multiplies it by ``E[sinc(k delta)]``. Diffusion over ``dt`` multiplies it by
``exp(-D k^2 dt)``. The Gaussian step of per-axis variance ``2 D dt`` (``|r|^2 / (2 D dt)`` chi-squared with
three degrees of freedom) reproduces that exactly; a law with the same ``E[delta^2] = 6 D dt`` errs by the
mismatch of its higher even moments, the ``x^4`` term of ``sinc`` weighted by ``E[delta^4]`` and so on. The
FIXED length (``E[delta^4] = 36 D^2 dt^2`` against the Gaussian's 60) decays every mode at a rate wrong by
``lambda dt / 5``: a first-order integrator whose error is the step's fourth moment and nothing else.

**The law the engine draws** is the two-point Gauss-Laguerre quadrature (``alpha = 1/2``) of the chi-squared
law of ``delta^2``: two lengths and a weight that match ``E[delta^2]``, ``E[delta^4]`` and ``E[delta^6]`` of the
Gaussian step, so the bulk error is third order in ``dt``, and the longest step is bounded (``PAIR_LONG`` times
``sqrt(6 D dt)``), which every wall rule of the engine is built on: the collision lookup, the bounce budget,
the candidate reach, the crossing rule. In closed form ``delta^2 / (6 D dt) = (5 -/+ sqrt(10)) / 3`` with
probabilities ``1/2 +/- 1/sqrt(10)``. Inside a pore every law that matches the fourth moment gives the same
error, the curved-wall term, which is first order for every symmetric step (the symmetrised Euler scheme of
Bossy, Gobet and Talay, 2004): 0.44 of ``lambda dt / 5`` on a cylinder's slowest mode, 0.78 on a sphere's,
0 on a slab, measured with :class:`SteppedPore` below.

**The stepped matrix method** (:class:`SteppedPore`) is the oracle of all of this: the one-step operator of
a walk with a given step law and the geometry's OWN ``reflect``, projected on the pore's Laplacian eigenmodes
by quadrature, then the signal of a sequence of physics steps as ``c <- Phase(g dt_p) T c`` from the uniform
state, with the engine's endpoint phase rule. For the slab the fixed-length operator is ``diag(sinc(k delta))``
exactly; for a cylinder or sphere it is whatever the curved reflection makes it, computed, not assumed. Its
reference :meth:`SteppedPore.continuous` is the exact propagator between the samples with the same phase rule,
so the difference isolates the step law.
"""
from __future__ import annotations

from functools import lru_cache

import numpy as np
from scipy import special

from dmipy_sim.constants import GAMMA
from .matrix_method import _cyl_neumann_roots, _sph_neumann_roots

__all__ = ["PAIR_SHORT", "PAIR_LONG", "PAIR_P_LONG", "PAIR_LENGTHS", "PAIR_WEIGHTS", "LONGEST_STEP", "MEAN_STEP",
           "FIXED_LENGTHS", "FIXED_WEIGHTS", "WALL_COEFFICIENT_ENVELOPE", "PORE_RATE_ENVELOPE",
           "SPHERE_K1", "SPHERE_WALL_COEFFICIENT", "CYLINDER_K1", "CYLINDER_WALL_COEFFICIENT",
           "PILOT_STEP_RATIO", "diffusion_spectrum", "step_certificate", "step_moments", "bulk_multiplier",
           "pore_step", "band_step", "band_reach", "SteppedPore"]

#: The pair's lengths in units of ``sqrt(6 D dt)`` and the probability of the long one.
PAIR_SHORT = float(np.sqrt((5.0 - np.sqrt(10.0)) / 3.0))      # 0.78267
PAIR_LONG = float(np.sqrt((5.0 + np.sqrt(10.0)) / 3.0))       # 1.64947
PAIR_P_LONG = float(0.5 - 1.0 / np.sqrt(10.0))                # 0.18377
PAIR_LENGTHS = (PAIR_SHORT, PAIR_LONG)
PAIR_WEIGHTS = (1.0 - PAIR_P_LONG, PAIR_P_LONG)
#: The longest step the walk takes, in units of ``sqrt(6 D dt)``: the number every displacement bound uses.
LONGEST_STEP = PAIR_LONG
#: The pair's mean length in units of ``delta_0``: what a wall's hit rate per step, ``(S/V) E[delta] / 4``, reads.
MEAN_STEP = (1.0 - PAIR_P_LONG) * PAIR_SHORT + PAIR_P_LONG * PAIR_LONG       # 0.94222
#: The fixed-length law, for the oracle's comparisons.
FIXED_LENGTHS = (1.0,)
FIXED_WEIGHTS = (1.0,)
#: The first-order coefficient of the specular reflection on a curved wall, in units of ``lambda_1 dt / 5`` on the
#: pore's slowest mode, for a surface that declares none: the largest measured (the sphere, 0.83 at
#: ``delta_0 / a = 0.5``, 0.65 at 0.1; the cylinder 0.41 to 0.51; a flat wall 0), so the envelope of every
#: convex pore in the first-order regime ``delta_0 <= a / 2``.
WALL_COEFFICIENT_ENVELOPE = 0.85
#: The slowest mode of each analytic pore, as the root ``K1`` of its Neumann condition (rate ``(K1 / a)^2`` per unit
#: D): the sphere's ``l = 1`` and the cylinder's ``m = 1``; and the wall coefficient measured on it with the stepped
#: matrix method (the sphere's bound over the first-order regime, the cylinder's 0.41 to 0.51).
SPHERE_K1, SPHERE_WALL_COEFFICIENT = 2.0815759778181, 0.85
CYLINDER_K1, CYLINDER_WALL_COEFFICIENT = 1.8411837813407, 0.5
#: The pore rate per ``min_feature^2`` of a geometry that declares none: a sphere of that radius, the tightest
#: confinement of the pores at one feature (a cylinder's is 3.39, a slab of width ``2 h`` 2.47), so the step it sets
#: holds the tolerance on any of them.
PORE_RATE_ENVELOPE = SPHERE_K1 ** 2


def step_moments(lengths, weights):
    """``E[delta^2], E[delta^4], E[delta^6]`` of a step length law given in units of ``sqrt(6 D dt)``, each
    divided by the Gaussian step's (``6 D dt``, ``60 D^2 dt^2``, ``840 D^3 dt^3``): a matched moment reads 1."""
    s = np.asarray(lengths, float) ** 2
    w = np.asarray(weights, float)
    return float(w @ s), float(w @ s ** 2 * 36.0 / 60.0), float(w @ s ** 3 * 216.0 / 840.0)


def bulk_multiplier(k_delta0, lengths=PAIR_LENGTHS, weights=PAIR_WEIGHTS):
    """``E[sinc(k delta)]`` of a step length law at ``k delta_0``, ``delta_0 = sqrt(6 D dt)``: what one step does
    to a bulk mode ``k``, against diffusion's ``exp(-(k delta_0)^2 / 6)``."""
    x = np.asarray(k_delta0, float)
    return sum(w * np.sinc(x * l / np.pi) for l, w in zip(lengths, weights))


def pore_step(pore_rate, D, eps, wall_coefficient=WALL_COEFFICIENT_ENVELOPE):
    """The physics step (s) at which the pore's slowest mode decays with relative rate error ``eps``:
    ``5 eps / (c lambda_1)`` with ``lambda_1 = D * pore_rate`` the mode's rate (``pore_rate`` its eigenvalue
    per unit D, 1/m^2) and ``c`` the wall coefficient; infinite for a flat wall (``c = 0``) and for a geometry that
    declares no confining mode on its feature (``pore_rate = 0``)."""
    c = float(wall_coefficient)
    if c <= 0.0 or float(pore_rate) <= 0.0:
        return float("inf")
    return 5.0 * float(eps) / (c * float(pore_rate) * float(D))


@lru_cache(maxsize=None)
def band_reach(eps, lengths=PAIR_LENGTHS, weights=PAIR_WEIGHTS):
    """The largest ``k delta_0`` at which a step law's bulk rate error on mode ``k`` stays within ``eps``: where
    ``|ln E[sinc(k delta)] / (-(k delta_0)^2 / 6) - 1|`` first exceeds ``eps`` on a fine grid (the pair: 1.79 at
    1e-3, 1.24 at 1e-4; the fixed step: 0.17 at 1e-3, from ``(k delta_0)^2 / 30``)."""
    x = np.linspace(1e-3, 3.0, 30000)
    m = bulk_multiplier(x, lengths, weights)
    ok = m > 0
    err = np.full_like(x, np.inf)
    err[ok] = np.abs(np.log(m[ok]) / (-x[ok] ** 2 / 6.0) - 1.0)
    beyond = np.flatnonzero(err > float(eps))
    return float(x[beyond[0] - 1]) if len(beyond) and beyond[0] > 0 else float(x[-1])


def band_step(f_band, eps, lengths=PAIR_LENGTHS, weights=PAIR_WEIGHTS):
    """The physics step (s) at which the bulk mode a band of ``f_band`` (Hz) reads, ``k = sqrt(2 pi f / D)``,
    decays with relative rate error ``eps``: ``k delta_0 = sqrt(12 pi f dt)``, so ``dt = reach^2 / (12 pi f)``
    with :func:`band_reach`; infinite for no band."""
    if f_band is None or f_band <= 0.0:
        return float("inf")
    return band_reach(float(eps), tuple(lengths), tuple(weights)) ** 2 / (12.0 * np.pi * float(f_band))


def _trapz_weights(x):
    w = np.empty_like(x)
    dx = np.diff(x)
    w[0] = dx[0] / 2
    w[-1] = dx[-1] / 2
    w[1:-1] = (dx[:-1] + dx[1:]) / 2
    return w


class SteppedPore:
    """The stepped matrix method on one analytic pore: ``shape`` in ``plane``, ``cylinder``, ``sphere`` of the
    given ``diameter`` (m) at diffusivity ``D`` (m^2/s), walked with ``geometry`` -- the engine's own
    :class:`~dmipy_sim.geometry.Box1D`, :class:`~dmipy_sim.geometry.Cylinder` or
    :class:`~dmipy_sim.geometry.Sphere` of that size, whose ``reflect`` is the operator's wall. The slab's
    walls are at ``x = 0`` and ``x = diameter`` like ``Box1D``'s; the round pores are centred at the origin.

    ``n_modes`` is ``(angular, radial)`` for the round pores and the count of cosine modes for the plane. The
    modes carry the gradient direction: along the slab's normal, across the cylinder, along the sphere's axis.
    """

    def __init__(self, shape, diameter, D, geometry, *, n_modes=(10, 10), radial_nodes=6000, angular_nodes=4096):
        self.shape, self.D, self.geometry = shape, float(D), geometry
        self.L, self.a = float(diameter), float(diameter) / 2.0
        self._nr = int(radial_nodes)
        a = self.a
        if shape == "plane":
            L = self.L
            n_cos = max(4, int(n_modes if isinstance(n_modes, (int, np.integer)) else n_modes[0] * n_modes[1] // 8))
            ks = np.arange(n_cos) * np.pi / L
            x = np.linspace(0.0, L, self._nr)
            self.rgrid = x
            U = np.array([np.cos(k * x) for k in ks])
            U /= np.sqrt(np.trapezoid(U * U, x, axis=1))[:, None]
            self.k, self.ang, self.Rf = ks, np.zeros(len(ks), int), U
            w = _trapz_weights(x)
            self.B = (U * (w * x)[None, :]) @ U.T
            self.p = (U @ w) / np.sqrt(L)
        elif shape == "cylinder":
            r = np.linspace(0.0, a, self._nr)
            self.rgrid = r
            ks, ang, Rf = [], [], []
            for m in range(n_modes[0] + 1):
                for k in _cyl_neumann_roots(m, n_modes[1]) / a:
                    R = special.jv(m, k * r)
                    R = R / np.sqrt(np.trapezoid(R * R * r, r) * (2 * np.pi if m == 0 else np.pi))
                    ks.append(k); ang.append(m); Rf.append(R)
            self.k, self.ang, self.Rf = np.array(ks), np.array(ang), np.array(Rf)
            t = np.linspace(0, 2 * np.pi, int(angular_nodes))
            M = n_modes[0] + 1
            angB = np.array([[np.trapezoid(np.cos(mi * t) * np.cos(t) * np.cos(mj * t), t) if abs(mi - mj) == 1 else 0.0
                              for mj in range(M)] for mi in range(M)])
            w = _trapz_weights(r)
            self.B = ((self.Rf * (w * r * r)[None, :]) @ self.Rf.T) * angB[self.ang[:, None], self.ang[None, :]]
            self.p = (self.Rf @ (w * r)) * np.where(self.ang == 0, 2 * np.pi, 0.0) / np.sqrt(np.pi * a * a)
        elif shape == "sphere":
            r = np.linspace(0.0, a, self._nr)
            self.rgrid = r
            th = np.linspace(0, np.pi, int(angular_nodes))
            ct, st = np.cos(th), np.sin(th)
            P = {l: special.eval_legendre(l, ct) for l in range(n_modes[0] + 2)}
            ang_norm = {l: 2 * np.pi * np.trapezoid(P[l] * P[l] * st, th) for l in range(n_modes[0] + 2)}
            ks, ang, Rf = [], [], []
            for l in range(n_modes[0] + 1):
                for k in _sph_neumann_roots(l, n_modes[1]) / a:
                    R = special.spherical_jn(l, k * r)
                    R = R / np.sqrt(np.trapezoid(R * R * r * r, r) * ang_norm[l])
                    ks.append(k); ang.append(l); Rf.append(R)
            self.k, self.ang, self.Rf = np.array(ks), np.array(ang), np.array(Rf)
            Lm = n_modes[0] + 1
            angB = np.array([[2 * np.pi * np.trapezoid(P[li] * ct * P[lj] * st, th) if abs(li - lj) == 1 else 0.0
                              for lj in range(Lm)] for li in range(Lm)])
            w = _trapz_weights(r)
            self.B = ((self.Rf * (w * r ** 3)[None, :]) @ self.Rf.T) * angB[self.ang[:, None], self.ang[None, :]]
            self.p = (self.Rf @ (w * r * r)) * np.where(self.ang == 0, 4 * np.pi, 0.0) / np.sqrt(4 / 3 * np.pi * a ** 3)
        else:
            raise ValueError(f"the stepped matrix method knows a plane, a cylinder and a sphere, not {shape!r}")
        self.lam = self.k ** 2                      # eigenvalues per unit D: the mode's rate is D * lam
        self._beta, self._U = np.linalg.eigh(self.B)

    def modes_at(self, pts):
        """The modes at ``pts`` (n, 3): ``(n_modes, n)``, the radial functions interpolated on the fine grid."""
        if self.shape == "plane":
            rad, angf = pts[:, 0], None
        elif self.shape == "cylinder":
            rad = np.hypot(pts[:, 0], pts[:, 1])
            phi = np.arctan2(pts[:, 1], pts[:, 0])
            angf = np.cos(self.ang[:, None] * phi[None, :])
        else:
            rad = np.linalg.norm(pts, axis=1)
            ct = np.where(rad > 0, pts[:, 2] / np.maximum(rad, 1e-30), 1.0)
            Pl = {l: special.eval_legendre(l, ct) for l in np.unique(self.ang)}
            angf = np.stack([Pl[l] for l in self.ang])
        g = self.rgrid
        nr = self._nr
        x = np.clip(rad, g[0], g[-1]) / (g[-1] - g[0]) * (nr - 1)
        i0 = np.clip(np.floor(x).astype(int), 0, nr - 2)
        f = x - i0
        R = self.Rf[:, i0] * (1 - f)[None, :] + self.Rf[:, i0 + 1] * f[None, :]
        return R if angf is None else R * angf

    def positions(self, n_rad=120, n_th=24):
        """Quadrature points ``(n, 3)`` and weights summing to the volume (the cylinder's azimuth is applied in
        :meth:`operator`, so its weights sum to the cross-section over ``2 pi``)."""
        xr, wr = np.polynomial.legendre.leggauss(n_rad)
        if self.shape == "plane":
            x = 0.5 * self.L * (xr + 1)
            w = 0.5 * self.L * wr
            return np.stack([x, np.zeros_like(x), np.zeros_like(x)], 1), w
        r = 0.5 * self.a * (xr + 1)
        wr = 0.5 * self.a * wr
        if self.shape == "cylinder":
            return np.stack([r, np.zeros_like(r), np.zeros_like(r)], 1), r * wr
        xc, wc = np.polynomial.legendre.leggauss(n_th)
        R, C = np.meshgrid(r, xc, indexing="ij")
        W = np.outer(wr * r * r, wc) * 2 * np.pi
        S = np.sqrt(1 - C ** 2)
        return np.stack([(R * S).ravel(), np.zeros(R.size), (R * C).ravel()], 1), W.ravel()

    @staticmethod
    def directions(n_a=64, n_b=128):
        """Directions ``(n, 3)`` and weights of the mean over the sphere."""
        xc, wc = np.polynomial.legendre.leggauss(n_a)
        b = (np.arange(n_b) + 0.5) * 2 * np.pi / n_b
        C, Bm = np.meshgrid(xc, b, indexing="ij")
        S = np.sqrt(1 - C ** 2)
        d = np.stack([(S * np.cos(Bm)).ravel(), (S * np.sin(Bm)).ravel(), C.ravel()], 1)
        w = np.outer(wc, np.full(n_b, 1.0 / n_b)).ravel() / 2.0
        return d, w

    def operator(self, delta, *, n_rad=120, n_th=24, n_a=64, n_b=128, chunk=64):
        """The one-step operator of a step of fixed length ``delta`` (m) on the modes,
        ``T[m, n] = (1/V) int u_m(r) E_dir[u_n(reflect(r, delta dir))] dV`` by quadrature, with its checks:
        ``asym`` (the operator is symmetric), ``inv`` (the uniform state is invariant), ``diag_err`` (the
        diagonal against the slab's ``sinc(k delta)``, exact there only)."""
        import jax
        import jax.numpy as jnp
        pts, wp = self.positions(n_rad, n_th)
        dirs, wd = self.directions(n_a, n_b)
        V = wp.sum() * (2 * np.pi if self.shape == "cylinder" else 1.0)
        steps = jnp.asarray(delta * dirs, jnp.float32)
        reflect = jax.jit(jax.vmap(jax.vmap(self.geometry.reflect, in_axes=(None, 0)), in_axes=(0, None)))
        Um = self.modes_at(pts)
        F = np.zeros_like(Um)
        for i in range(0, len(pts), chunk):
            ends = np.asarray(reflect(jnp.asarray(pts[i:i + chunk], jnp.float32), steps), np.float64)
            E = self.modes_at(ends.reshape(-1, 3)).reshape(len(self.k), ends.shape[0], ends.shape[1])
            F[:, i:i + chunk] = E @ wd
        T = (Um * wp[None, :]) @ F.T
        if self.shape == "cylinder":
            same = self.ang[:, None] == self.ang[None, :]
            T = T * np.where(same, np.where(self.ang[:, None] == 0, 2 * np.pi, np.pi), 0.0)
        return T, dict(V=V, asym=float(np.abs(T - T.T).max()), inv=float(np.abs(T @ self.p - self.p).max()),
                       diag_err=float(np.abs(np.diag(T) - np.sinc(self.k * delta / np.pi)).max()))

    def law_operator(self, dt_p, lengths=PAIR_LENGTHS, weights=PAIR_WEIGHTS, **quadrature):
        """The one-step operator of a physics step ``dt_p`` (s) under a step length law, the weighted sum of the
        fixed-length operators at ``lengths * sqrt(6 D dt_p)``."""
        delta0 = np.sqrt(6.0 * self.D * dt_p)
        T = 0.0
        for l, w in zip(lengths, weights):
            T = T + w * self.operator(delta0 * l, **quadrature)[0]
        return T

    def rate_error(self, T, dt_p):
        """Each mode's relative rate error under one step of ``T``: ``-ln T[i, i] / (dt_p D lam_i) - 1``, with
        the constant mode's (undefined) entry as 0."""
        rate = self.D * self.lam
        with np.errstate(divide="ignore", invalid="ignore"):
            e = -np.log(np.diag(T)) / dt_p / rate - 1.0
        return np.where(rate > 0, e, 0.0)

    def signal(self, T, g, dt, sub, *, phase="endpoint"):
        """``|S|`` of a walk under the per-sample gradient ``g`` (T/m) at the sample interval ``dt``, ``sub``
        physics steps of ``T`` per sample; ``T`` ``None`` steps with the exact propagator ``exp(-D lam dt_p)``,
        which isolates the phase rule. ``phase`` is the engine's ``endpoint`` rule (the step, then the whole
        step's phase at its end point) or ``midpoint`` (half the phase, the step, half the phase: second order)."""
        dt_p = dt / sub
        U, beta = self._U, self._beta
        exact = np.exp(-dt_p * self.D * self.lam)
        c = self.p.astype(complex).copy()
        for gk in g:
            ph = np.exp(-1j * GAMMA * gk * dt_p * beta)
            half = np.exp(-0.5j * GAMMA * gk * dt_p * beta)
            for _ in range(sub):
                if phase == "midpoint":
                    c = U @ (half * (U.T @ c))
                c = exact * c if T is None else T @ c
                c = U @ ((half if phase == "midpoint" else ph) * (U.T @ c))
        return float(abs(self.p @ c))

    def continuous(self, g, dt):
        """``|S|`` with the exact propagator over each sample interval, Strang-split around the sample's phase:
        the reference the stepped signal is held against."""
        half = np.exp(-0.5 * dt * self.D * self.lam)
        U, beta = self._U, self._beta
        c = self.p.astype(complex).copy()
        for gk in g:
            c = half * c
            c = U @ (np.exp(-1j * GAMMA * gk * dt * beta) * (U.T @ c))
            c = half * c
        return float(abs(self.p @ c))


#: The factor between the two pilot walks' physics steps: the coarse walk steps this many times longer in time (twice
#: as long in length) than the fine one, so the first-order difference is three times the error at the fine step
#: for a quarter of the fine walk's cost, and the coarse step stays in the first-order regime the rule sizes
#: (``delta_0 <= min_feature / 2``) wherever the fine step is under a quarter of it.
PILOT_STEP_RATIO = 4


def diffusion_spectrum(C, T, *, f_max=None):
    """``(f, D, se)``: the walk's diffusion spectrum read off its stored bands, ``D(f)`` in m^2/s at the bands'
    frequencies ``f_k = k / (2 T)`` up to ``f_max`` (the top band ``K / (2 T)`` when None), the mean over the three
    axes, the windows and the walkers, ``se`` its standard error over walkers.

    ``C`` is a :class:`~dmipy_sim.coefficient_walk.CoefficientWalk`'s ``position_coeffs``, ``(n_walkers, n_windows,
    K + 2, 3)`` (or one window, ``(n_walkers, K + 2, 3)``), ``T`` the window's duration. Band ``k`` of the bridge on
    ``s_k = sqrt(2 / T) sin(pi k t / T)`` is, by parts, ``beta_k = sqrt(2 T) / (pi k) int_0^T cos(pi k t / T) v(t)
    dt``: a cosine transform of the velocity, so ``D(f_k) = (pi k)^2 <beta_k^2> / (2 T^2)`` per axis with
    ``D(f) = (1/2) int <v(0) v(t)> cos(2 pi f t) dt``. A free walk's bands are its Karhunen-Loeve coefficients, of
    variance ``2 D T^2 / (pi k)^2`` exactly, so its estimate is flat at ``D``; a confined walk's falls from ``D`` above
    the pore rate to zero at zero frequency. The transform over a finite window smooths a structured spectrum by its
    kernel of width ``1 / T``, and the block table weights each band by its quadrature; the two walks of
    :func:`step_certificate` share ``T``, ``K``, the block and its order, so both cancel in their difference. There
    is no ``k = 0`` bin: the window's drift is its exact displacement, not a band."""
    C = np.asarray(C, np.float64)
    if C.ndim == 3:
        C = C[:, None]
    if C.ndim != 4 or C.shape[-1] != 3 or C.shape[2] < 3:
        raise ValueError(f"the bands are position_coeffs (n_walkers, n_windows, K + 2, 3); got shape {C.shape}")
    T = float(T)
    K = int(C.shape[2]) - 2
    k = np.arange(1, K + 1, dtype=np.float64)
    beta2 = C[:, :, 2:, :] ** 2                                     # (n_w, n_win, K, 3)
    d_w = (np.pi * k[None, :]) ** 2 * beta2.mean(axis=(1, 3)) / (2.0 * T * T)      # (n_w, K): per walker
    f = k / (2.0 * T)
    D = d_w.mean(axis=0)
    se = d_w.std(axis=0, ddof=1) / np.sqrt(d_w.shape[0]) if d_w.shape[0] > 1 else np.full_like(D, np.inf)
    keep = slice(None) if f_max is None else (f <= float(f_max) * (1.0 + 1e-12))
    return f[keep], D[keep], se[keep]


def step_certificate(C_fine, C_coarse, T, ratio, *, f_band, eps_declared):
    """What two pilot walks of one substrate, accumulated over windows of ``T`` at one ``K``, one block and one
    block order, the coarse one stepping ``ratio`` times longer in time than the fine one, measure of the fine
    step's error: the relative first-order error of the diffusion spectrum over the band, ``eps_measured``, with its
    standard error, and whether the fine step holds ``eps_declared``. ``C_fine`` / ``C_coarse`` are the two walks'
    ``position_coeffs`` (:func:`diffusion_spectrum`).

    The law is first order in the step, so ``D_coarse(f) - D_fine(f) = (ratio - 1) * err_fine(f)``; each band up to
    ``f_band`` gives ``err_fine(f) / D_fine(f)`` with its own standard error. A mode's rate error enters the spectrum
    with one sign below the mode's rate and the other above it, so the band's error is read as the root mean square
    over the bands, its noise subtracted (``E[rel^2] = err^2 + se^2``); the slowest mode's rate error, which the rule
    sizes the step for, bounds it from above. ``floor`` is the rms standard error of the bands, what the pilot can
    resolve, and ``resolves`` whether that is under ``eps_declared``: a pilot that cannot resolve the tolerance
    bounds the error only to its floor, the record and the card say so, and the rule the walk was sized by is then
    what holds the tolerance. A walk whose bands stop below ``f_band`` (``K < 2 T f_band``) cannot read the band it
    certifies and is refused, as are two walks of different band counts or windows."""
    Cf, Cc = np.asarray(C_fine), np.asarray(C_coarse)
    if Cf.shape[1:] != Cc.shape[1:]:
        raise ValueError(f"the two pilots carry bands of shape {Cf.shape[1:]} and {Cc.shape[1:]}; the certificate "
                         f"compares two walks of one window, K and block")
    K = int(Cf.shape[-2]) - 2
    T = float(T)
    if K < 2.0 * T * float(f_band) * (1.0 - 1e-9):
        raise ValueError(f"the band {float(f_band):g} Hz is above the pilots' top band K / (2 T) = {K / (2.0 * T):g} Hz "
                         f"(K = {K} over {T:g} s): the step certificate reads the band it certifies, so the pilots "
                         f"walk at K >= 2 T f_band = {int(np.ceil(2.0 * T * float(f_band)))}")
    _, Df, sf = diffusion_spectrum(Cf, T, f_max=f_band)
    _, Dc, sc = diffusion_spectrum(Cc, T, f_max=f_band)
    if Df.size < 1:
        raise ValueError(f"the band {float(f_band):g} Hz holds no band of a window of {T:g} s (the first is at "
                         f"{0.5 / T:g} Hz): the pilot's window is too short to read the band it certifies")
    r = float(ratio)
    rel = (Dc - Df) / (r - 1.0) / Df
    se = np.sqrt(sc ** 2 + sf ** 2) / (r - 1.0) / Df
    n = int(Df.size)
    floor = float(np.sqrt(np.mean(se ** 2)))
    eps = float(np.sqrt(max(float(np.mean(rel ** 2) - np.mean(se ** 2)), 0.0)))
    eps_se = float(np.sqrt(np.sum(2.0 * se ** 4 + 4.0 * rel ** 2 * se ** 2)) / n / (2.0 * max(eps, floor)))
    return dict(eps_measured=eps, eps_se=eps_se, floor=floor, eps_declared=float(eps_declared), ratio=r,
                f_band_hz=float(f_band), n_bins=n, K=K, T_s=T, n_walkers=int(Cf.shape[0]),
                resolves=bool(floor < float(eps_declared)), per_bin_max=float(np.abs(rel).max()),
                rule="eps_measured = the noise-corrected root mean square over the bands k / (2 T) <= f_band of "
                     "(D_coarse - D_fine) / ((ratio - 1) D_fine), D(f_k) = (pi k)^2 <beta_k^2> / (2 T^2) read off "
                     "the stored bands, the first-order error of the fine step's diffusion spectrum, which the "
                     "slowest mode's rate error bounds; the gate holds it to eps_declared within its pass band")


def assess_step_certificate(certificate, *, z=3.0):
    """The measured step diagnostic, separate from an absolute MR-signal accuracy certificate.

    A pass needs a resolved pilot and an upper error estimate below the declared tolerance. The confidence
    is a one-sided normal/delta-method approximation, not a deterministic bound or a signal-error conversion.
    A noisy pilot is unresolved rather than evidence that the theoretical step rule has been measured.
    """
    import math
    z = float(z)
    if not math.isfinite(z) or z <= 0:
        raise ValueError("the step check needs a finite positive confidence multiplier")
    fields = ("eps_measured", "eps_se", "floor", "eps_declared")
    values = {key:certificate.get(key) for key in fields}
    if any(value is None or not math.isfinite(float(value)) or float(value) < 0 for value in values.values()):
        return dict(status="unresolved", upper=None, metric="relative_diffusion_spectrum_rms")
    measured, se, floor, declared = (float(values[key]) for key in fields)
    if declared <= 0:
        raise ValueError("the declared step tolerance must be positive")
    upper = measured + z * se
    resolved = bool(certificate.get("resolves", False)) and floor < declared
    status = "unresolved" if not resolved else "passed" if upper <= declared else "over_budget"
    return dict(certificate, status=status, upper=upper, declared=declared, z=z,
                confidence=0.5 * (1.0 + math.erf(z / math.sqrt(2.0))),
                metric="relative_diffusion_spectrum_rms", method="one_sided_normal_delta_approximation")
