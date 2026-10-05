"""Factorised pose responses: one pass over the walkers per (waveform shape, tissue, field), and every encoding class
of a machine an evaluation of that pass's series (dmrai-lab/dmipy-sim#610).

**The setting.** A machine plays one commanded acquisition differently at every voxel: its coils' nonlinearity
scales and tilts each row, its own gradient ``g0`` adds a constant through every pulse and dead time, and its Maxwell
term adds a gradient quadratic in the coils'. Binned to a tolerance of ``b`` these are the encoding classes
(:func:`~dmipy_sim.phantom.bore.encoding_classes`), thousands over a head on a permanent magnet, and the closed form
(:meth:`~dmipy_sim.replay.replay.ReplayPack.pose_responses`) contracts the walkers once per class. The classes are
one timing class: what changes from one to the next is each row's amplitude and direction, the background's
magnitude ``|g0|`` and direction, and the small part of each row that is not its commanded shape.

**The phase.** In the pack's own coefficient space a row is ``W_i`` ``(n_c, 3)`` (the bridge projection of its
effective gradient per window, :func:`~dmipy_sim.replay.replay._compile_effective`), and walker ``w``'s phase is
``sum_k W_i[k] . C_w[k]`` with ``C_w`` its stored coefficients. A shape is a unit vector ``u`` of that space; a row
of the shape is written ``W_i = u v_i^T + D_i`` with ``v_i = W_i^T u`` (its least-squares vector: amplitude ``a_i =
|v_i|``, direction ``g^_i = v_i / a_i``) and ``D_i`` its departure. With ``mu_w = sum_k u[k] C_w[k]`` (the walker's
shape moment) and ``n_w`` its background moment (the effective gate's), the phase at pose ``R`` is

    phi_iw(R) = a_i g^_i . R mu_w + beta g^0 . R n_w + d_iw(R),       d_iw(R) = sum_k (R^T D_i[k]) . C_w[k].

**The joint series.** The closed form's body of a row (#197, #565) is

    B[(l, n), c] = sum_w w_w j_l(a mu_w) Y_ln(mu^_w) F_wc(beta),

``F_w(beta)`` the background factor's columns ``4 pi i^l2 j_l2(beta |n_w|) Y_l2(n^_w)`` (coupled with the field
factor walker by walker when there is one, #582/#611). Everything else in a row's coefficients is its lab side
(``Y_l(g^_i)``, ``Y_l2(g^0)`` and the field direction), which is exact and per row. ``B`` depends on the class only
through ``(a, beta)``, and is an entire function of both, so on a closed interval of each it is a convergent
Chebyshev series:

    B(a, beta) = sum_{k <= N_a} sum_{q <= N_b} T_k(tau) T_q(sigma) Bhat_kq,     tau = (a - a_c) / h_a,  sigma = (beta - b_c) / h_b,

``Bhat`` the Chebyshev coefficients of the bodies at the ``(N_a+1)(N_b+1)`` Chebyshev points of the second kind --
which ARE bodies of the closed form, at those amplitudes and magnitudes: the walker pass. Its coupling on the body
index (the real Clebsch-Gordan tables) is linear, so it is applied to the coefficients once and a class reads only
coupled numbers. A class is then ``T_k(tau_i) T_q(sigma)`` per row and the lab side.

Why Chebyshev and not powers: the power series of ``j_l(a x)`` in ``a`` about zero (#588) has terms that grow to about
``e^x / 2x`` before they cancel, so at the brain's phases (tens of radians) it loses every digit. About the centre of
a shell's amplitude interval the Chebyshev series has no cancellation (``|T_k| <= 1`` and its coefficients decay
geometrically) and needs about ``e x h / 2`` terms, ``x h`` the phase's swing over the interval.

**The convergence radius and the truncation bounds.** ``j_l`` is entire, so the series in each variable converges on
every Bernstein ellipse ``E_rho`` (foci -1, 1, semi-axes ``(rho +- 1/rho) / 2``). On it ``|Im a| <= h_a (rho - 1/rho)
/ 2``, and from ``j_l(z) = z^l / (2^{l+1} l!) int_{-1}^1 e^{izt} (1 - t^2)^l dt``,

    |j_l(z)| <= min(1, |z|^l / (2l+1)!!) e^{|Im z|},

so ``M_lw(rho) = min(1, (x_w (a_c + h_a (rho + 1/rho) / 2))^l / (2l+1)!!) exp(x_w h_a (rho - 1/rho) / 2)`` bounds
``j_l(a x_w)`` there, and the interpolant in the ``N + 1`` Chebyshev points misses it by at most ``4 M rho^{-N} /
(rho - 1)`` everywhere on the interval (Trefethen, *Approximation Theory and Approximation Practice*, Thm 8.2). The
plane wave is ``sum_l (2l+1) i^l j_l P_l`` with ``|P_l| <= 1``, so per walker the amplitude series moves it by at most

    I_aw = sum_{l <= L} (2l+1) 4 M_lw(rho) rho^{-N_a} / (rho - 1),

and the background series by ``I_bw``, the same sum over ``l2 <= L_b`` on ``beta`` in ``[0, beta_max]``. ``rho`` is
chosen per interval to minimise the walkers' weighted sum, and ``N`` is the least degree that brings it below
``tol``. The weighted sums are taken over the walkers binned by ``x_w`` at each bin's upper edge (``M`` grows with
``x``), so they are bounds and cost a few hundred terms.

**The departure.** ``|e^{i(phi + d)} - e^{i phi}| <= |d|`` and, for an orthonormal basis ``V`` of the coefficient
space, ``|d_iw(R)| <= sum_j |D_i^T V_j| |nu_jw|`` with ``nu_jw = sum_k V[k, j] C_w[k]`` at every pose. So

    dev_i = sum_j |D_i^T V_j| A_j,      A_j = sum_w |w_w| |nu_jw|,

one pass over the walkers for ``A`` and a product per row. ``V`` is the complete SVD basis of the departures of the
first acquisitions read: a machine's departure (the Maxwell term through a ramp, float32 rounding) lies along a
few courses, so their ``A_j`` carry it and #573's per-walker bound is recovered; anything else is still bounded,
since ``V`` spans the whole space. This is #573's bound with the courses fixed once for every class.

**The misfit.** Walker by walker, with ``A``, ``B`` the gradient's and the background's plane waves and ``Ã``, ``B̃``
their truncated series, ``|A e^{id} B Phi - Ã B̃ Phi| <= |d| + rho_a + (1 + rho_a) rho_b`` (``|B| = |Phi| = 1``), where
``rho_a`` is the band tail (orders above ``L``) plus ``I_a`` and ``rho_b`` the background's band tail plus ``I_b``.
Weighted as the signal weighs the walkers, each row's misfit is

    dev_i + tail_a + I_a + (tail_b(beta) + I_b) (1 + max_w rho_aw),

``tail_a`` the closed form's band tail at the interval's largest amplitude (``j_l`` rises on ``[0, l]`` and ``L``
exceeds every phase, so the largest amplitude bounds the rest), ``tail_b(beta) = sum_{l2 > L_b} (2l2+1) beta^l2
sum_w |w_w| |n_w|^l2 / (2l2+1)!!`` at the class's own ``|g0|``, and ``max_w rho_aw`` the closed form's largest
per-walker band remainder plus the largest ``I_aw``. A row whose departure exceeds ``direction_tol`` (a tenth of the
ensemble's floor by default, the closed form's own limit) is refused.

**When it is exact, what is refused.** The bodies are exact at the nodes (they are the closed form's bodies there);
the series reproduces the closed form at every class to its truncation bounds, and the misfit carries them. Every
class must share the series' gate and timing (the effective gate, the coherence gate, the waveform grid, the pulse
pathway): one series per timing class, tissue and field (#611), refused by name otherwise. A class read after the
build must lie in the series' intervals: an amplitude outside its shape's interval, a ``|g0|`` above ``beta_max``, a
row that is no shape the series holds, or a b-tensor encoding, is refused by name, never extrapolated.
"""
from __future__ import annotations

import numpy as np

__all__ = ["PoseSeries", "pose_series"]

SERIES_GAP = 0.02
"""Rows of one shape are one series interval while their amplitudes lie within this fraction of each other's: a gap
wider than it (two shells of one shape, square pulses at two b-values) starts a new interval, so each shell's
series spans its own spread (a machine's percent) rather than the distance between shells."""

SERIES_COURSES = 8
"""Acquisitions whose departures fix the departure basis ``V``: the first ones read. Any basis bounds every
departure; this one makes the bound #573's (the departures' own courses carry them)."""

SERIES_MAX_DEGREE = 160
"""The largest Chebyshev degree an interval takes. An interval that needs more is one whose phase swings more than
some sixty radians across it: refused, naming the interval, rather than summed at a degree that would be a walker
pass per node anyway."""

SERIES_PAD = 1e-6
"""Each amplitude interval and the largest ``|g0|`` are widened by this fraction: a waveform is stored in float32, so
two plays of one class differ by its rounding (~1e-7), and a class read again must lie in the interval it was built
in. The bounds are taken over the widened intervals."""

SERIES_BINS = 512
"""Bins of the walkers' moment lengths over which the series' bounds are summed (each at its bin's upper edge)."""

ZERO_ROW = 1e-6
"""A row whose encoding (the magnet's gradient taken out) is at most this fraction of the magnet's gradient on every
sample is a ``b = 0`` row: no amplitude, its whole projection a departure."""


def _cheb_points(N):
    """The ``N + 1`` Chebyshev points of the second kind, ``cos(j pi / N)`` (``[0.0]`` for ``N = 0``)."""
    return np.array([0.0]) if N == 0 else np.cos(np.pi * np.arange(N + 1) / N)


def _cheb_matrix(N):
    """``(N+1, N+1)``: values at :func:`_cheb_points` -> Chebyshev coefficients of the interpolant,
    ``c_k = (2 / N) sum''_j f_j cos(k j pi / N)`` with the first and last terms of the sum and ``c_0``, ``c_N`` halved."""
    if N == 0:
        return np.ones((1, 1))
    j = np.arange(N + 1)
    T = np.cos(np.pi * np.outer(j, j) / N)                               # (k, j)
    T[:, 0] *= 0.5; T[:, -1] *= 0.5
    T *= 2.0 / N
    T[0] *= 0.5; T[-1] *= 0.5
    return T


def _cheb_T(N, t, xp=np):
    """``(n, N+1)``: ``T_k(t)`` for ``k <= N`` by the three-term recurrence."""
    out = [xp.ones_like(t)]
    if N >= 1:
        out.append(t)
    for _k in range(2, N + 1):
        out.append(2 * t * out[-1] - out[-2])
    return xp.stack(out, -1)


def _log_double_factorial(L):
    """``log((2l+1)!!)`` for ``l = 0..L``."""
    return np.cumsum(np.log(np.arange(1, 2 * L + 2, 2, dtype=np.float64)))


def _series_bound(x, wabs, centre, half, L, tol):
    """The Chebyshev degree of the series of ``j_l(s x)`` (``l <= L``) in ``s`` on ``[centre - half, centre + half]``
    and its bounds: ``(N, I, I_max)`` with ``I`` the walkers' weighted bound ``sum_w |w_w| I_w`` and ``I_max`` the
    largest ``I_w`` (module docstring). ``x`` ``(n_w,)`` the walkers' moment lengths, ``wabs`` their ``|w|``. The
    walkers are binned by ``x`` and each bin is bounded at its upper edge (every term grows with ``x``)."""
    if half <= 0 or not x.size or float(x.max()) == 0.0:
        return 0, 0.0, 0.0
    x_max = float(x.max())
    edges = np.linspace(0.0, x_max, SERIES_BINS + 1)[1:]
    which = np.minimum(np.searchsorted(edges, x, side="left"), SERIES_BINS - 1)
    wb = np.bincount(which, weights=wabs, minlength=SERIES_BINS)
    keep = wb > 0
    xb, wb = edges[keep], wb[keep]
    rho = np.geomspace(1.0 + 1e-3, 1e6, 400)                              # (n_rho,)
    eta = np.outer(xb, half * (rho - 1.0 / rho) / 2.0)                   # (n_b, n_rho): |Im z| on the ellipse
    z = np.outer(xb, centre + half * (rho + 1.0 / rho) / 2.0)           # the largest |z| on it
    l = np.arange(L + 1)
    with np.errstate(divide="ignore"):
        logz = np.log(z)
    lm = np.minimum(0.0, l[None, None, :] * logz[:, :, None] - _log_double_factorial(L)[None, None, :])
    logS = np.log(np.sum((2 * l + 1)[None, None, :] * np.exp(lm), axis=2))   # (n_b, n_rho): sum_l (2l+1) min(1, z^l/(2l+1)!!)
    base = np.log(4.0) + eta + logS - np.log(rho - 1.0)[None, :]        # log of 4 M rho^0 / (rho - 1), per bin
    lw = np.log(wb)[:, None]
    m = (base + lw).max(axis=0)
    logW = m + np.log(np.exp(base + lw - m[None, :]).sum(axis=0))      # (n_rho,): log sum_b w_b 4 M / (rho - 1)
    for N in range(1, SERIES_MAX_DEGREE + 1):
        lb = logW - N * np.log(rho)
        k = int(np.argmin(lb))
        if lb[k] <= np.log(tol):
            return N, float(np.exp(lb[k])), float(np.exp(base[:, k].max() - N * np.log(rho[k])))
    raise ValueError(
        f"an amplitude interval [{centre - half:.4g}, {centre + half:.4g}] swings the phase by {x_max * 2 * half:.1f} "
        f"radians: its series would need more than {SERIES_MAX_DEGREE} terms. Expand these acquisitions per class "
        f"(ReplayPack.pose_responses), or split them into narrower sets")


class PoseSeries:
    """The pose responses of a set of encoding classes (one timing class, tissue and field) as one series: built by
    :meth:`~dmipy_sim.replay.replay.ReplayPack.pose_series` from one pass over the walkers, evaluated per class by
    :meth:`responses`. The module docstring is the derivation; :attr:`bounds` states what each truncation costs.

    Attributes worth reading: ``lmax``, ``nmax`` (the band kept), ``L`` (the gradient's band), ``field_lmax``,
    ``background_lmax``, ``pieces`` (per amplitude interval: its shape, ``[a_lo, a_hi]``, degree ``N_a`` and bound),
    ``degree_b`` and ``beta_max`` (the background's series), ``n_classes`` (the classes it was built on), ``floor``,
    ``seconds`` (the build's phases)."""

    def __init__(self, **state):
        self.__dict__.update(state)

    @property
    def n_classes(self):
        return len(self._located)

    @property
    def bounds(self):
        """What each part of the misfit is, largest over the classes the series was built on: ``departure`` (the rows'
        departures from their shapes), ``band`` (the gradient's band tail), ``series_a`` (the amplitude series),
        ``background`` (the background's band tail and series, with the gradient's remainder factor) and ``floor``."""
        dep = max((float(r["dep"].max()) for r in self._located), default=0.0)
        return dict(departure=dep, band=float(max(p["tail"] for p in self.pieces)),
                    series_a=float(max(p["I"] for p in self.pieces)),
                    background=float((self._tail_b(self.beta_max) + self.series_b) * (1.0 + self.r_a)),
                    floor=self.floor)

    def _tail_b(self, beta):
        """The background's band tail at magnitude ``beta`` (module docstring): ``sum_{l2 > L_b} (2l2+1) beta^l2
        S_l2 / (2l2+1)!!`` over the three orders the closed form sums."""
        beta = np.asarray(beta, np.float64)
        out = np.zeros_like(beta)
        for l, S in self._bg_moments:
            out = out + (2 * l + 1) * np.exp(l * np.log(np.maximum(beta, 1e-300)) - _log_double_factorial(l)[-1]) * S
        return np.where(beta > 0, out, 0.0)

    def responses(self, waveforms=None, *, select=None):
        """One :class:`~dmipy_sim.replay.replay.PoseResponse` per class, ``route == "series"``: the classes the series was
        built on (``select`` an index list into them, all of them when None), or ``waveforms`` read now (each within
        the series' intervals and of its timing class, refused by name otherwise)."""
        if waveforms is not None:
            if select is not None:
                raise ValueError("responses takes waveforms or select, not both")
            located = [self._locate(wf) for wf in waveforms]
        else:
            idx = range(self.n_classes) if select is None else select
            located = [self._located[int(i)] for i in idx]
        out = []
        chunk = max(1, int(self.eval_rows // max(1, self.n_meas)))
        for lo in range(0, len(located), chunk):
            out += self._evaluate(located[lo:lo + chunk])
        return out

    # ---- a class located in the series ----------------------------------------------------------------------------
    def _locate(self, wf):
        """``wf``'s rows in the series: their interval, amplitude, direction, departure bound and the class's
        background (magnitude and direction per row), after the checks that it is of the series' timing class."""
        loc = _project(self._ctx, wf)
        return _place(self._ctx, loc, self.pieces)

    # ---- evaluation -----------------------------------------------------------------------------------------------
    def _evaluate(self, located):
        """The coefficients and misfits of the classes ``located``, all their rows in one batch, on the series' device."""
        from . import so3
        from .replay import PoseResponse, _factor_lab
        xp, put, cdt = self._xp, self._put, self._cdtype
        on_torch = xp is not np
        cplx = (lambda x: put(np.asarray(x, np.complex128)).to(cdt)) if on_torch else (lambda x: np.asarray(x, np.complex128))
        n_rows = [len(r["a"]) for r in located]
        rows = int(sum(n_rows))
        if rows == 0:
            return []
        cat = lambda k: np.concatenate([r[k] for r in located])
        piece, a, g_hat, beta, u, dep = cat("piece"), cat("a"), cat("g_hat"), cat("beta"), cat("u"), cat("dep")
        # every row's coupled bodies: the series at its (amplitude, |g0|)
        sigma = np.zeros(rows) if self.degree_b == 0 else np.clip(beta / self.beta_max * 2.0 - 1.0, -1.0, 1.0)
        Tb = _cheb_T(self.degree_b, sigma)                                     # (rows, N_b+1)
        body = (self._torch.zeros((rows, self._n_cpl), dtype=cdt, device=self._dev) if on_torch
                else np.zeros((rows, self._n_cpl), np.complex128))
        for p, pc in enumerate(self.pieces):
            sel = np.flatnonzero(piece == p)
            if not sel.size:
                continue
            tau = np.zeros(sel.size) if pc["half"] == 0 else np.clip((a[sel] - pc["centre"]) / pc["half"], -1.0, 1.0)
            Tab = (_cheb_T(pc["N"], tau)[:, :, None] * Tb[sel][:, None, :]).reshape(sel.size, -1)   # (n, (N_a+1)(N_b+1))
            body[put(sel) if on_torch else sel] = cplx(Tab) @ self._coef[p]
        # the lab side: the rows' gradient directions and the background's (with the field's) per row
        Yg = cplx(so3.real_sh(self._l_lab, g_hat, full=True))
        u_dirs, u_of = np.unique(u, axis=0, return_inverse=True)                  # a class's rows share its magnet's gradient
        u_of = np.asarray(u_of).reshape(-1)
        u_of = put(u_of) if on_torch else u_of
        lam = [cplx(x)[u_of] for x in _factor_lab(self._channels, so3.real_sh(self.background_lmax, u_dirs, full=True),
                                                   self._Yb, self._K_fb)]
        coeffs = (self._torch.zeros((rows, self._n_feat), dtype=cdt, device=self._dev) if on_torch
                  else np.zeros((rows, self._n_feat), np.complex128))
        for (l, Lam), (cis, cols, K_lab, Ls, lab_sl, body_sl, gains) in self._groups.items():
            bl = so3.sh_block(l, True)
            Rl = (Yg[:, bl] @ K_lab).reshape(rows, 2 * Lam + 1, -1)           # (rows, 2Lam+1, sum 2Lc+1)
            lam_c = xp.stack([lam[ci] for ci in cis], 1)                      # (rows, n_ci, 2Lam+1)
            body_c = body[:, cols].reshape(rows, len(cis), -1)                # (rows, n_ci, sum 2kk+1)
            Bl = xp.einsum("icM,ics->iMs", lam_c, body_c)
            for Lc in Ls:
                o, n_o = self._offs[Lc]
                blk = xp.einsum("iMa,iMb->iab", Rl[:, :, lab_sl[Lc]], Bl[:, :, body_sl[Lc]]).reshape(rows, -1)
                coeffs[:, o:o + n_o] += gains[Lc] * blk
        coeffs = np.asarray(coeffs.cpu().numpy() if on_torch else coeffs, np.complex128)
        # the misfit per row (module docstring)
        tail = np.array([pc["tail"] for pc in self.pieces])[piece]
        I_a = np.array([pc["I"] for pc in self.pieces])[piece]
        misfit = dep + tail + I_a + (self._tail_b(beta) + self.series_b) * (1.0 + self.r_a)
        out, o = [], 0
        for r, n in zip(located, n_rows):
            sl = slice(o, o + n); o += n
            f = r["voxel"]
            resp = PoseResponse(coeffs[sl] * f[:, None], self.lmax, self.nmax, misfit=misfit[sl] * np.abs(f), floor=self.floor,
                                phase_amplitude=float(np.max(r["a"] * self._mu_max[r["shape"]], initial=0.0)), n_samples=0)
            resp.n_bodies = len(self.pieces)
            resp.field_lmax = self.field_lmax
            resp.background_lmax = self.background_lmax
            resp.route = "series"
            out.append(resp)
        return out


# ---- the projection of a class into the pack's coefficient space -----------------------------------------------------

def _window_maps(pack, P, n_t_wf, dt_wf):
    """``(n_c, n_t_wf)``: the linear map from a waveform's samples (one axis) to the pack's coefficient space, every
    window the acquisition reaches side by side -- the same :func:`effective_gradient` and bridge projection every
    closed-form route reads, applied to the unit samples (exact: both are linear)."""
    from ._replay_kernel import effective_gradient
    from .replay import _compile_effective
    dt = P["dt"]
    maps = []
    for _seg, t0, n_s in P["windows"]:
        M = np.empty((pack.K + 2, n_t_wf))
        for lo in range(0, n_t_wf, 256):
            hi = min(lo + 256, n_t_wf)
            unit = np.zeros((hi - lo, n_t_wf, 1)); unit[np.arange(hi - lo), np.arange(lo, hi), 0] = 1.0
            G = effective_gradient(unit, dt_wf, n_s, dt, t0=t0) if pack.n_segments > 1 else \
                effective_gradient(unit, dt_wf, P["n_t"], dt)
            M[:, lo:hi] = _compile_effective(G, dt, pack.K, n_s)
        maps.append(M)
    return np.concatenate(maps, 0)


def _timing_key(wf):
    """What every class of a series shares: the waveform grid, the effective gate, the coherence gate, the pathway."""
    from ..acquisition.epg import pathway_weight
    chi = wf.chi_perp
    return (float(wf.dt), int(np.asarray(wf.G).shape[1]), np.asarray(wf.effective_gate, np.float64).tobytes(),
            None if chi is None else np.asarray(chi, np.float64).tobytes(), float(pathway_weight(wf)))


def _project(ctx, wf):
    """A class's rows in the pack's coefficient space, ``W`` ``(n_meas, n_c, 3)`` in the specimen frame, the magnet's
    gradient taken out (``g0`` ``(n_meas, 3)``), the ``b = 0`` rows marked and the voxel factor; refused when it is
    not of the series' timing class."""
    from .replay import _background_of, ScannerSequence
    wf = wf.waveform if hasattr(wf, "waveform") else wf
    if not isinstance(wf, ScannerSequence):
        raise TypeError(f"a pose series reads ScannerSequences; got {type(wf).__name__}")
    if _timing_key(wf) != ctx["key"]:
        raise ValueError("this acquisition is not of the series' timing class (its waveform grid, effective gate, "
                         "coherence gate or pulse pathway differ): a series is one per timing class, tissue and field")
    G = np.asarray(wf.G_eff, np.float64)
    n_meas = G.shape[0]
    bg = _background_of(wf)
    g0 = np.zeros((n_meas, 3))
    if bg is not None:
        g0, gate = bg
        G = G - g0[:, None, :] * gate[None, :, None]
    zero = ~np.any(G != 0.0, axis=(1, 2)) | (np.abs(G).max(axis=(1, 2)) <= ZERO_ROW * np.linalg.norm(g0, axis=1))
    xp = ctx["xp"]
    if xp is np:
        W = np.einsum("ct,mtb->mcb", ctx["M"], G)
    else:
        torch = ctx["torch"]
        Gd = torch.as_tensor(G, device=ctx["dev"])
        W = torch.einsum("ct,mtb->mcb", ctx["M_d"], Gd)
    R_s = ctx["R_s"]
    if R_s is not None:                                                       # the specimen frame: G @ R_s per sample
        W = W @ (R_s if xp is np else ctx["R_s_d"])
        g0 = g0 @ R_s
    return dict(W=W, g0=g0, zero=zero, voxel=np.asarray(wf.voxel_factor(), np.float64))


def _shapes(ctx, loc):
    """The shapes of a class: its rows' principal profiles in the coefficient space, unit-normed and signed by their
    largest entry, in clusters of amplitude (a gap wider than :data:`SERIES_GAP` between two rows' amplitudes starts
    a new one: a shell, whose ramps a slew-limited machine plays at the shell's own length) and within each grouped
    to :data:`~dmipy_sim.replay.replay.SHELL_RTOL`: ``(n_h, n_c)``."""
    from .replay import SHELL_RTOL, _group_waveforms
    W = loc["W"] if ctx["xp"] is np else loc["W"].cpu().numpy()
    live = np.flatnonzero(~loc["zero"])
    if not live.size:
        return np.zeros((0, W.shape[1]))
    Wl = W[live]
    _ev, vec = np.linalg.eigh(np.einsum("mcb,mcd->mbd", Wl, Wl))
    p = np.einsum("mcb,mb->mc", Wl, vec[:, :, -1])
    amp = np.linalg.norm(p, axis=1)
    p = p / amp[:, None]
    p = p * np.sign(p[np.arange(len(p)), np.argmax(np.abs(p), axis=1)])[:, None]
    order = np.argsort(amp, kind="stable")
    cut = np.flatnonzero(np.diff(amp[order]) > SERIES_GAP * amp[order][1:]) + 1
    U = []
    for members in np.split(order, cut):
        _grp, first = _group_waveforms(p[members], rtol=SHELL_RTOL)
        U.append(p[members][first])
    return np.concatenate(U, 0)


def _fit(ctx, loc):
    """The rows of a projected class on the series' shapes: each row's shape (the one that carries most of it), its
    vector ``v = W^T u`` (amplitude and direction) and its departure ``W - u v^T`` ``(n_meas, n_c, 3)``."""
    xp = ctx["xp"]
    W, U = loc["W"], ctx["U"] if xp is np else ctx["U_d"]
    v = xp.einsum("mcb,hc->mhb", W, U)                                        # (n_meas, n_h, 3)
    h = (v * v).sum(-1).argmax(1) if xp is np else (v * v).sum(-1).argmax(1)
    if xp is np:
        vh = v[np.arange(v.shape[0]), h]
        D = W - U[h][:, :, None] * vh[:, None, :]
    else:
        ar = ctx["torch"].arange(v.shape[0], device=ctx["dev"])
        vh = v[ar, h]
        D = W - U[h][:, :, None] * vh[:, None, :]
    zero = loc["zero"]
    if zero.any():                                                            # a b = 0 row: all of it is departure
        zi = np.flatnonzero(zero)
        zi_d = zi if xp is np else ctx["torch"].as_tensor(zi, device=ctx["dev"])
        vh[zi_d] = 0.0
        D[zi_d] = W[zi_d]
    return h, vh, D


def _place(ctx, loc, pieces=None):
    """A projected class's rows as the series reads them: shape, amplitude, direction, departure bound, background
    magnitude and direction per row; with ``pieces`` (a built series) also each row's interval, refused when it lies
    outside every one."""
    xp = ctx["xp"]
    h, vh, D = _fit(ctx, loc)
    c = xp.einsum("mcb,cj->mjb", D, ctx["V"] if xp is np else ctx["V_d"])       # the departure in the basis V
    nrm = (c * c).sum(-1) ** 0.5
    dep = nrm @ (ctx["A"] if xp is np else ctx["A_d"])
    if xp is not np:
        h, vh, dep = h.cpu().numpy(), vh.cpu().numpy(), dep.cpu().numpy()
    a = np.linalg.norm(vh, axis=1)
    g_hat = np.where(a[:, None] > 0, vh / np.where(a > 0, a, 1.0)[:, None], (0.0, 0.0, 1.0))
    zero = loc["zero"]
    shape = np.where(zero, -1, h)
    g0 = loc["g0"]
    beta = np.linalg.norm(g0, axis=1)
    u = np.where(beta[:, None] > 0, g0 / np.where(beta > 0, beta, 1.0)[:, None], (0.0, 0.0, 1.0))
    if np.any(dep > ctx["limit"]):
        i = int(np.argmax(dep))
        raise ValueError(f"row {i} departs from the series' shape {int(shape[i])} by {dep[i]:.3g} of signal at some pose, "
                         f"beyond the direction tolerance {ctx['limit']:.3g}: it is not one of the series' shapes. Expand "
                         f"this acquisition per class (ReplayPack.pose_responses)")
    out = dict(shape=shape, a=a, g_hat=g_hat, dep=dep, beta=beta, u=u, voxel=loc["voxel"])
    if pieces is not None:
        out["piece"] = _piece_of(pieces, shape, a, beta, ctx)
    return out


def _piece_of(pieces, shape, a, beta, ctx):
    """Each row's series interval, refused by name for a row outside every interval of its shape, or a ``|g0|``
    above the series' largest."""
    piece = np.full(len(a), -1, np.int64)
    for p, pc in enumerate(pieces):
        inside = (shape == pc["shape"]) & (a >= pc["lo"]) & (a <= pc["hi"])
        piece[inside & (piece < 0)] = p
    if np.any(piece < 0):
        i = int(np.flatnonzero(piece < 0)[0])
        raise ValueError(f"row {i} plays amplitude {a[i]:.6g} on shape {int(shape[i])}, outside every interval the series "
                         f"was built over: build the series on a set of classes that includes it")
    if np.any(beta > ctx["beta_max"]):
        raise ValueError(f"a |g0| of {beta.max():.4g} T/m exceeds the series' largest, {ctx['beta_max']:.4g}: build the series "
                         f"on a set of classes that includes it")
    return piece


# ---- the build ----------------------------------------------------------------------------------------------------

def pose_series(pack, waveforms, *, tissue=None, scanner=None, pose=None, compartment=None, keep=None, tol=1e-8,
                l_cap=64, direction_tol=None, backend="jax", device=None):
    """:meth:`~dmipy_sim.replay.replay.ReplayPack.pose_series` (the module docstring is the derivation)."""
    import time
    from . import so3
    from .compression import read_position_coeffs
    from .pose_device import bessel_tails, route, field_bodies, field_bodies_torch
    from .replay import (_pose_matrix, _factor_channels, _factor_of, _coupling_tables, PoseResponse, TORCH_GROUPS)
    from ..run import Run, current
    if backend not in ("jax", "torch"):
        raise ValueError(f"backend is 'jax' or 'torch', got {backend!r}")
    kernels = "auto" if backend == "jax" else ("torch" if device is None else f"torch:{device}")
    view = pack._at_tissue(tissue)
    if view is not pack:
        return pose_series(view, waveforms, tissue=tissue, scanner=scanner, pose=pose, compartment=compartment, keep=keep,
                           tol=tol, l_cap=l_cap, direction_tol=direction_tol, backend=backend, device=device)
    kind, t_dev = route(kernels)
    seconds = {}
    t_start = time.perf_counter()
    it = iter(waveforms)
    try:
        first = next(it)
    except StopIteration:
        raise ValueError("a pose series needs at least one acquisition") from None
    first = first.waveform if hasattr(first, "waveform") else first
    R_s = _pose_matrix(pose)
    with Run("pose_series", params=dict(id=pack.id, n_meas=int(first.n_meas), keep=None if keep is None else list(keep))):
        run = current()
        _ph = (lambda name, **f: run.phase(name, **f)) if run is not None else (lambda name, **f: None)
        from ..acquisition.waveforms import rotate_waveform
        wf0 = rotate_waveform(first, R_s.T) if R_s is not None else first
        P0 = pack._prepare(wf0, tissue=tissue, scanner=scanner, orientation=None, compartment=compartment)
        if R_s is not None:
            P0["b0_dir"] = tuple(R_s.T @ np.array([0.0, 0.0, 1.0]))
        if np.any(P0["voxel"] != 1.0):
            pass                                                              # applied per class, as the closed form does
        ew, norm = P0["pathway"] * P0["ew"], P0["norm"]
        n_w = ew.shape[0]
        w = np.asarray(ew, np.float64) / float(norm)
        wabs = np.abs(w)
        limit = (0.1 / np.sqrt(n_w)) if direction_tol is None else float(direction_tol)
        n_t_wf, dt_wf = int(np.asarray(first.G).shape[1]), float(first.dt)
        M = _window_maps(pack, P0, n_t_wf, dt_wf)                             # (n_c, n_t_wf)
        n_c = M.shape[0]
        ctx = dict(key=_timing_key(first), M=M, R_s=R_s, limit=limit, beta_max=np.inf)
        if kind == "torch":
            import torch
            put = lambda x: torch.as_tensor(x, device=t_dev)
            ctx.update(xp=torch, torch=torch, dev=t_dev, M_d=put(M), R_s_d=None if R_s is None else put(R_s))
        else:
            torch = None
            put = lambda x: x
            ctx.update(xp=np)
        # the first acquisitions: the shapes and the departure basis
        _ph("shapes")
        head = [first]
        for wf in it:
            head.append(wf)
            if len(head) >= SERIES_COURSES:
                break
        projected = [_project(ctx, wf) for wf in head]
        U = _shapes(ctx, projected[0])
        if not U.shape[0] and any((~p["zero"]).any() for p in projected):
            U = _shapes(ctx, next(p for p in projected if (~p["zero"]).any()))
        n_h = U.shape[0]
        ctx["U"] = U if n_h else np.zeros((1, n_c))
        if kind == "torch":
            ctx["U_d"] = put(ctx["U"])
        D_all = []
        for p in projected:
            _h, _v, D = _fit(ctx, p)
            D_all.append(D if kind != "torch" else D.cpu().numpy())
        X = np.concatenate([D.transpose(0, 2, 1).reshape(-1, n_c) for D in D_all], 0)
        from .._blas import lapack_threads
        with lapack_threads():
            _u, _s, Vt = np.linalg.svd(X, full_matrices=True)
        V = Vt.T                                                              # (n_c, n_c): an orthonormal basis
        ctx["V"] = V
        # ---- the walker pass: shape moments, background moment, departure courses
        _ph("moments", n_w=int(n_w), n_shapes=int(n_h))
        t0 = time.perf_counter()
        gate = np.asarray(first.effective_gate, np.float64)
        w_gate = M @ gate                                                     # (n_c,): the background's course
        F = pack.substrate_frame
        mu = np.zeros((n_w, max(n_h, 1), 3)); n_bg = np.zeros((n_w, 3)); A = np.zeros(n_c)
        row0 = 0
        nu = None
        for seg, _t0, _n_s in P0["windows"]:
            C = read_position_coeffs(seg.arrays, dtype=np.float64).reshape(n_w, -1, 3)   # (n_w, K+2, 3)
            k = C.shape[1]
            sl = slice(row0, row0 + k); row0 += k
            if kind == "torch":
                Cd = put(C)
                mu += torch.einsum("wkb,hk->whb", Cd, put(ctx["U"][:, sl])).cpu().numpy()
                n_bg += torch.einsum("wkb,k->wb", Cd, put(w_gate[sl])).cpu().numpy()
                part = torch.einsum("wkb,kj->wjb", Cd, put(V[sl]))
                nu = part if nu is None else nu + part
                del Cd
            else:
                mu += np.einsum("wkb,hk->whb", C, ctx["U"][:, sl])
                n_bg += np.einsum("wkb,k->wb", C, w_gate[sl])
                part = np.einsum("wkb,kj->wjb", C, V[sl])
                nu = part if nu is None else nu + part
            del C
        if kind == "torch":
            A = (put(wabs) @ torch.linalg.vector_norm(nu, dim=2)).cpu().numpy()
        else:
            A = wabs @ np.linalg.norm(nu, axis=2)
        del nu
        ctx["A"] = A
        if kind == "torch":
            ctx["V_d"], ctx["A_d"] = put(V), put(A)
        mu = mu @ F                                                           # stored -> canonical, per walker
        n_bg = n_bg @ F
        x_mu = np.linalg.norm(mu, axis=2)                                     # (n_w, n_h): |mu_w| per shape
        mu_hat = mu / np.where(x_mu > 0, x_mu, 1.0)[:, :, None]
        mu_hat[x_mu == 0] = (0.0, 0.0, 1.0)
        r_n = np.linalg.norm(n_bg, axis=1)
        n_hat = n_bg / np.where(r_n > 0, r_n, 1.0)[:, None]
        n_hat[r_n == 0] = (0.0, 0.0, 1.0)
        seconds["moments"] = time.perf_counter() - t0
        # ---- every class located: the series' domain
        _ph("locate")
        t0 = time.perf_counter()
        located = [_place(ctx, p) for p in projected]
        del projected
        for wf in it:
            located.append(_place(ctx, _project(ctx, wf)))
        seconds["locate"] = time.perf_counter() - t0
        shape_all = np.concatenate([r["shape"] for r in located])
        a_all = np.concatenate([r["a"] for r in located])
        beta_max = float(max((r["beta"].max() for r in located), default=0.0)) * (1.0 + SERIES_PAD)
        ctx["beta_max"] = beta_max
        pieces = []
        for hh in range(n_h):
            amps = np.unique(a_all[shape_all == hh])
            if not amps.size:
                continue
            cut = np.flatnonzero(np.diff(amps) > SERIES_GAP * amps[1:]) + 1
            for grp in np.split(amps, cut):
                pieces.append(dict(shape=hh, lo=float(grp[0]), hi=float(grp[-1])))
        if np.any(shape_all < 0):
            pieces.append(dict(shape=-1, lo=0.0, hi=0.0))
        for pc in pieces:                                                     # widened by a float32 waveform's rounding
            pc["lo"], pc["hi"] = pc["lo"] * (1.0 - SERIES_PAD), pc["hi"] * (1.0 + SERIES_PAD)
            pc["centre"], pc["half"] = (pc["lo"] + pc["hi"]) / 2.0, (pc["hi"] - pc["lo"]) / 2.0
        # ---- the gradient's band: the closed form's rule at every interval's largest amplitude
        _ph("band")
        x_of = lambda pc: (np.zeros(n_w) if pc["shape"] < 0 else x_mu[:, pc["shape"]])
        kappa_hi = np.stack([pc["hi"] * x_of(pc) for pc in pieces], 1)       # (n_w, n_pieces)
        k_max = float(kappa_hi.max()) if kappa_hi.size else 0.0
        L = int(np.ceil(k_max)) + 2
        T_ab = bessel_tails(kappa_hi, w, min(l_cap, L + 12), device=kernels)
        while L < l_cap:
            if L + 1 >= T_ab.shape[0]:
                T_ab = bessel_tails(kappa_hi, w, min(l_cap, T_ab.shape[0] + 12), device=kernels)
            if (2 * (L + 1) + 1) * T_ab[L + 1].max() < tol:
                break
            L += 1
        for p, pc in enumerate(pieces):
            pc["tail"] = float(sum((2 * l + 1) * T_ab[l, p] for l in range(L + 1, min(L + 4, T_ab.shape[0]))))
        J_top = so3.spherical_jn_all(L + 3, np.array([k_max]))[:, 0]
        r_a = float(sum((2 * l + 1) * abs(J_top[l]) for l in range(L + 1, L + 4)))
        # ---- the amplitude series per interval
        I_max = 0.0
        for pc in pieces:
            pc["N"], pc["I"], im = _series_bound(x_of(pc), wabs, pc["centre"], pc["half"], L, tol)
            I_max = max(I_max, im)
        # ---- the field factor
        field = pack._field_quadratic(P0, wf0) if P0["B0"] is not None else None
        if field is None:
            L_f, F_sh = 0, None
        else:
            _ph("field")
            F_sh, L_f = pack._field_harmonics(field, tol=tol, l_cap=l_cap, device=kernels)
        # ---- the background: its band and its series in |g0| on [0, beta_max]
        _ph("background", beta_max=beta_max)
        L_b, N_b, I_b, bg_moments = 0, 0, 0.0, []
        if beta_max > 0:
            x = beta_max * r_n
            T_bg = bessel_tails(x[:, None], w, min(l_cap, int(np.ceil(float(x.max()))) + 12), device=kernels)
            while L_b < l_cap:
                if L_b + 1 >= T_bg.shape[0]:
                    T_bg = bessel_tails(x[:, None], w, min(l_cap, T_bg.shape[0] + 12), device=kernels)
                if (2 * (L_b + 1) + 1) * T_bg[L_b + 1].max() < tol:
                    break
                L_b += 1
            N_b, I_b, _ = _series_bound(r_n, wabs, beta_max / 2.0, beta_max / 2.0, L_b, tol)
            bg_moments = [(l, float(wabs @ r_n ** l)) for l in range(L_b + 1, L_b + 4)]
        # ---- the coupling tables and the factor's channels (the closed form's own)
        Lam_max = L_f + L_b
        L_tot = L + Lam_max
        want_l, want_n = (None, None) if keep is None else (keep[0], keep[1])
        keep_l = L_tot if want_l is None else min(int(want_l), L_tot)
        keep_n = L_tot if want_n is None else min(int(want_n), L_tot)
        n_feat = so3.n_so3_coeffs(keep_l, keep_n)
        l_used = [l for l in range(L + 1) if l <= keep_l + Lam_max]
        offs, tables = _coupling_tables(l_used, Lam_max, keep_l, keep_n)
        channels, n_cols, K_fb = _factor_channels(None if F_sh is None else L_f, L_b)
        Yb = None
        if F_sh is not None:
            b_lab = np.asarray(P0["b0_dir"], np.float64); b_lab = b_lab / np.linalg.norm(b_lab)
            Yb = so3.real_sh(L_f, b_lab[None, :], full=True)[0]
        # the coupled body layout: per (l, channel) its sum 2kk+1 columns
        cpl, n_cpl = {}, 0
        for l in l_used:
            for ci, (Lam, _cols, _k) in enumerate(channels):
                if (l, Lam) in tables:
                    width = tables[(l, Lam)][2].shape[1]
                    cpl[(l, ci)] = slice(n_cpl, n_cpl + width); n_cpl += width
        _ph("nodes", L=int(L), L_f=int(L_f), L_b=int(L_b), n_pieces=len(pieces), N_b=int(N_b),
            nodes=int(sum(pc["N"] + 1 for pc in pieces) * (N_b + 1)))
        # ---- the walker pass at the nodes: the closed form's bodies at every (amplitude, |g0|) node, coupled
        t0 = time.perf_counter()
        R_rows = sum(2 * l + 1 for l in l_used)
        l_off = dict(zip(l_used, np.cumsum([0] + [2 * l + 1 for l in l_used[:-1]])))
        n_bessel = max(l_used) + 24 + int(np.ceil(k_max))
        a_nodes = [pc["centre"] + pc["half"] * _cheb_points(pc["N"]) for pc in pieces]
        kap = np.concatenate([np.outer(x_of(pc), an) for pc, an in zip(pieces, a_nodes)], 1)   # (n_w, n_nodes)
        mh = np.concatenate([np.repeat((mu_hat[:, pc["shape"], None, :] if pc["shape"] >= 0 else
                                        np.broadcast_to(np.array([0.0, 0.0, 1.0]), (n_w, 1, 3))), len(an), axis=1)
                             for pc, an in zip(pieces, a_nodes)], 1)                         # (n_w, n_nodes, 3)
        n_nodes = kap.shape[1]
        b_nodes = beta_max / 2.0 * (1.0 + _cheb_points(N_b)) if beta_max > 0 else np.zeros(1)
        if kind == "torch":
            new = lambda shape: torch.empty(shape, dtype=torch.complex128, device=t_dev)
            Y_n = put(so3.real_sh(L_b, n_hat, full=True)); F_sh_k = None if F_sh is None else put(F_sh)
            K_fb_k = {pr: {Lm: put(K.conj()) for Lm, K in Kp.items()} for pr, Kp in K_fb.items()}
            K_body = {key: put(t[2]) for key, t in tables.items()}
        else:
            new = lambda shape: np.empty(shape, np.complex128)
            Y_n = so3.real_sh(L_b, n_hat, full=True); F_sh_k = F_sh
            K_fb_k = {pr: {Lm: K.conj() for Lm, K in Kp.items()} for pr, Kp in K_fb.items()}
            K_body = {key: t[2] for key, t in tables.items()}
        values = np.empty((n_nodes, len(b_nodes), n_cpl), np.complex128)
        for q, bq in enumerate(b_nodes):
            Fq = _factor_of(put(so3.spherical_jn_all(L_b, bq * r_n)), Y_n, F_sh_k, K_fb_k, channels, L_b, n_cols, new)
            if kind == "torch":
                F_re, F_im = Fq.real.to(torch.float32).contiguous(), Fq.imag.to(torch.float32).contiguous()
            else:
                F_re, F_im = np.ascontiguousarray(Fq.real), np.ascontiguousarray(Fq.imag)
            del Fq
            step = TORCH_GROUPS if kind == "torch" else max(1, int(2.5e8 / (8 * n_w * (L + 1) ** 2)))
            for lo in range(0, n_nodes, step):
                sl = slice(lo, min(lo + step, n_nodes)); nc = sl.stop - sl.start
                if kind == "torch":
                    B3 = field_bodies_torch(kap[:, sl], mh[:, sl], w, F_re, F_im, L, l_used, n_bessel=n_bessel, device=t_dev)
                else:
                    B_c = field_bodies(kap[:, sl], mh[:, sl], w, F_re, F_im, L, l_used, n_bessel=n_bessel, device=kernels)
                    B3 = np.empty((nc, R_rows, B_c.shape[1]), np.complex128)
                    row = 0
                    for l in l_used:
                        B3[:, l_off[l]:l_off[l] + 2 * l + 1] = B_c[row:row + nc * (2 * l + 1)].reshape(nc, 2 * l + 1, -1)
                        row += nc * (2 * l + 1)
                parts = []
                for (l, ci), cs in cpl.items():
                    Lam, cols, _k = channels[ci]
                    parts.append(B3[:, l_off[l]:l_off[l] + 2 * l + 1, cols].reshape(nc, -1) @ K_body[(l, Lam)])
                vals = torch.cat(parts, 1).cpu().numpy() if kind == "torch" else np.concatenate(parts, 1)
                values[sl, q] = vals
                del B3, parts
        # Chebyshev coefficients of every interval, in amplitude and in |g0|
        Cb = _cheb_matrix(N_b)
        coef, o = [], 0
        for pc, an in zip(pieces, a_nodes):
            v = values[o:o + len(an)]; o += len(an)
            coef.append(np.einsum("kj,qr,jrc->kqc", _cheb_matrix(pc["N"]), Cb, v))
        del values
        seconds["nodes"] = time.perf_counter() - t0
        # ---- the evaluation's tables, grouped by (l, Lambda)
        groups = {}
        for (l, Lam), (Ls, K_lab3, _Kb, lab_sl, body_sl) in tables.items():
            cis = [ci for ci, (Lm, _c, _k) in enumerate(channels) if Lm == Lam and (l, ci) in cpl]
            if not cis:
                continue
            cols = np.concatenate([np.arange(cpl[(l, ci)].start, cpl[(l, ci)].stop) for ci in cis])
            gains = {Lc: 4 * np.pi * (1j ** l) / np.sqrt(2 * Lc + 1) for Lc in Ls}
            groups[(l, Lam)] = (cis, cols, K_lab3.reshape(2 * l + 1, -1), Ls, lab_sl, body_sl, gains)
        offs_n = {Lc: (offs[Lc], (2 * Lc + 1) * (2 * (so3._n_cols(Lc, keep_n) // 2) + 1)) for Lc in offs}
        if kind == "torch":
            cdt = torch.complex128
            groups = {k: (cis, put(cols), put(K.astype(np.complex128)).to(cdt), Ls, lab_sl, body_sl, gains)
                      for k, (cis, cols, K, Ls, lab_sl, body_sl, gains) in groups.items()}
            coef_d = [put(c.reshape(-1, c.shape[2])).to(cdt) for c in coef]
        else:
            cdt = np.complex128
            coef_d = [c.reshape(-1, c.shape[2]) for c in coef]
        mu_max = np.append(x_mu.max(axis=0) if n_w else np.zeros(max(n_h, 1)), 0.0)   # index -1: the b = 0 rows
        series = PoseSeries(
            lmax=keep_l, nmax=keep_n, L=L, field_lmax=L_f, background_lmax=L_b, pieces=pieces, degree_b=N_b,
            beta_max=beta_max, series_b=I_b, r_a=r_a + I_max, floor=1.0 / np.sqrt(n_w), n_meas=int(first.n_meas),
            seconds=seconds, eval_rows=16384 if kind == "torch" else 1024,
            _ctx=ctx, _located=[], _xp=(torch if kind == "torch" else np), _torch=torch, _dev=t_dev, _put=put,
            _cdtype=cdt, _coef=coef_d, _groups=groups, _offs=offs_n,
            _n_feat=n_feat, _n_cpl=n_cpl, _channels=channels, _Yb=Yb, _K_fb=K_fb, _l_lab=max(l_used), _mu_max=mu_max,
            _bg_moments=bg_moments)
        t0 = time.perf_counter()
        for r in located:
            r["piece"] = _piece_of(pieces, r["shape"], r["a"], r["beta"], ctx)
        series._located = located
        seconds["place"] = time.perf_counter() - t0
        seconds["build"] = time.perf_counter() - t_start
        return series
