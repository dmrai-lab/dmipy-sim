"""The concomitant term held against magnetostatics, with no formula in common (dmipy-sim#377, #364 tier 1).

Every other test of the Maxwell term is comparative (two routes sharing ``with_concomitant``) or structural
(zero at isocentre, ``1/B0``, unrefocused by a spin echo), and all of those pass for a wrong coefficient. The
oracle here is the Biot-Savart field of an explicit wire: what a spin sees is ``|B0 n + B_coil(r, t)|``, and
nothing in ``coil.py`` knows the expansion the sequence applies.

Two checks. The first holds the sequence transform alone: the extra gradient ``with_concomitant`` adds must be
the spatial gradient of the coil's exact ``|B_perp|^2 / 2 B0``. The second is end to end through a replayed
walk: the pack replayed at the acquisition as the coils actually deliver it must give the signal a walker
integrating the TRUE field magnitude along its own path gives -- the frame rotation, the position convention
and the composition with the nonlinearity all inside the comparison.

What it validates and what it cannot. A Maxwell pair has ``alpha = 1/2`` by axial symmetry, so this proves the
expansion's FORM, its derivative, its frame and its wiring into a phase. It says nothing about a machine whose
coil symmetry is not published; the Swoop's ``alpha`` is an inference (see ``test_coil_oracle``), and this
oracle cannot make it a measurement.

What it found, and what changed because of it. The usual first-order form of the term,
``|B_perp|^2 / 2 B0``, is short of the full ``|B0 n + B|`` by about ``1.3 B_n / B0`` in its gradient with
``B_n = G . r``: 4 per cent at 8 cm and 67 mT/m at 64 mT, which reached the signal as 8 per cent of the
concomitant effect. ``with_concomitant`` now applies the exact magnitude ``sqrt((B0 + B_n)^2 + |B_perp|^2) -
B0 - B_n``, and the end-to-end residual is the coil's own: 0.2 per cent of the effect. The truncation is
recorded as its own test so the reason stays measurable (dmipy-sim#377).
"""
import numpy as np
import pytest

import dmipy_sim as d
from dmipy_sim import sequences
from dmipy_sim.acquisition import coil
from dmipy_sim.constants import GAMMA
from dmipy_sim.replay import read_rpk
from dmipy_sim.replay.bank import build_replay_pack

B0 = 0.064                    # the field at which the term is a low-field problem: 47x what 3 T gives
R_TURN = 0.30                 # coil radius, m; the probe sits well inside, where the Golay's nonlinearity is small
SEGMENTS = 120                # per loop; the discretisation error is 1/n^2 of 6e-6 at 720, so ~2e-4 here

#: the frame the oracle is run in: B0 along the bore, and B0 across it (the bi-planar case). The coils, the
#: gradient and the position are all turned by the same rotation, so the physics is re-described, not changed.
FRAMES = {
    "bore": np.eye(3),
    "across": np.array([[1.0, 0.0, 0.0], [0.0, 0.0, -1.0], [0.0, 1.0, 0.0]]),      # z -> y
}


def _coils(R, linear=True):
    """Unit-gradient x and z coils, turned into the frame ``R``: ``{axis: (Coil, g_nominal)}``.

    The z coil is a Maxwell pair, whose transverse field is exactly the symmetric ``alpha = 1/2`` form near
    the axis. The x coil is a Golay saddle at the classic linearising geometry (arcs of 120 degrees at
    0.39 R and 2.14 R), which nulls its third-order term, so the pair is what the expansion assumes: LINEAR
    coils. ``linear=False`` gives the symmetric saddle instead, whose third-order term is 3.5 per metre
    squared -- a real coil's nonlinearity, and what the expansion does not carry.
    """
    out = {}
    golay = (coil.golay_saddle(radius=R_TURN, z_inner=0.39 * R_TURN, z_outer=2.14 * R_TURN, n=60) if linear
             else coil.golay_saddle(radius=R_TURN, n=60))
    for axis, c in (("x", golay), ("z", coil.maxwell_pair(radius=R_TURN, n=SEGMENTS))):
        g = coil.GradientCoil(c, axis).nominal_gradient()
        out[axis] = (coil.Coil([(v @ R.T, i) for v, i in c.turns], name=c.name), g)
    return out


def _field(coils, G_vec, points, R):
    """``B_coil`` at ``points`` for a commanded gradient ``G_vec`` (in the turned frame), by superposition."""
    Gx, _Gy, Gz = np.asarray(G_vec, np.float64) @ R           # the command in the coils' own axes
    cx, gx = coils["x"]
    cz, gz = coils["z"]
    return cx.field(points) * (Gx / gx) + cz.field(points) * (Gz / gz)


def _nonlinearity(coils, r, R, h=1e-4):
    """``L(r)`` of these coils by finite differences of the TRUE ``B_z``: column j is ``grad(b_j . n) / g_j``."""
    n = R @ np.array([0.0, 0.0, 1.0])
    L = np.zeros((3, 3))
    for j, e in enumerate(np.eye(3)):
        for i, f in enumerate(np.eye(3)):
            hi = _field(coils, e, (r + h * f)[None], R)[0] @ n
            lo = _field(coils, e, (r - h * f)[None], R)[0] @ n
            L[i, j] = (hi - lo) / (2.0 * h)
    return L


def _extra_gradient(coils, R, G_vec, r, B0_T=B0):
    """``(got, want)``: the extra gradient ``with_concomitant`` adds at ``r`` for the constant command
    ``G_vec``, and the numerical gradient of the coils' exact ``|B_perp|^2 / 2 B0`` there."""
    n = R @ np.array([0.0, 0.0, 1.0])

    def B_c(points):
        B = _field(coils, G_vec, np.atleast_2d(points), R)
        perp = B - np.outer(B @ n, n)
        return np.sum(perp ** 2, axis=-1) / (2.0 * B0_T)

    h = 1e-4
    want = np.array([(B_c(r + h * e) - B_c(r - h * e))[0] / (2.0 * h) for e in np.eye(3)])
    seq = sequences.pgse([[1.0, 0.0, 0.0]], 0.01, 0.03, gradient_strengths=[1.0], n_t=201)
    G = np.zeros_like(np.asarray(seq.G, np.float64)); G[:, :, :] = G_vec               # constant, so pointwise
    played = seq.with_gradient(G).with_concomitant(r, B0_T, b0_axis=tuple(n))
    return np.asarray(played.G, np.float64)[0, 0] - G_vec, want


@pytest.mark.parametrize("frame", list(FRAMES))
def test_the_extra_gradient_is_the_gradient_of_the_coils_exact_concomitant_field(frame):
    """``with_concomitant`` adds ``grad(|B_perp|^2 / 2 B0)`` -- held against that quantity computed from the
    wire, differenced numerically, at an oblique position where every term of the expansion is non-zero."""
    R = FRAMES[frame]
    G_vec = R @ np.array([0.018, 0.0, 0.012])                  # T/m, an oblique command: Gx Gz cross term live
    r = R @ np.array([0.05, 0.03, -0.04])
    got, want = _extra_gradient(_coils(R), R, G_vec, r)

    # what is left is the coils' residual nonlinearity, the finite differences and the wire discretisation
    assert np.linalg.norm(got - want) < 0.02 * np.linalg.norm(want), (got, want)
    assert np.linalg.norm(want) > 1e-6, "the probe sits where the term vanishes; the check would be empty"


def test_a_coils_nonlinearity_moves_the_concomitant_gradient_by_more_than_it_moves_the_field():
    """What the expansion does NOT carry, measured. The symmetric saddle's field is within 2 per cent of the
    ideal linear coil at this point, and the concomitant GRADIENT the expansion predicts is off by more than
    ten per cent in one component -- the derivative of the third-order term competes with the Maxwell
    pair's own transverse slope. A machine with a large catalogued nonlinearity (the Swoop's is 0.45 per
    metre) has a concomitant term the expansion states to this kind of accuracy and no better, and the
    catalogue's ``concomitant_alpha`` cannot repair it: alpha is one number and this is a shape."""
    R = np.eye(3)
    G_vec = np.array([0.018, 0.0, 0.012])
    r = np.array([0.05, 0.03, -0.04])
    got, want = _extra_gradient(_coils(R, linear=False), R, G_vec, r)
    worst = np.max(np.abs(got - want) / np.abs(want))
    assert worst > 0.10, f"the nonlinear coil moved the term by only {worst:.1%}; the caveat above is overstated"
    assert np.linalg.norm(got - want) < 0.3 * np.linalg.norm(want)


@pytest.mark.parametrize("B0_T", [0.064, 0.256])
def test_the_first_order_form_is_short_by_the_gradients_own_field_over_b0_and_the_exact_one_is_not(B0_T):
    """The truncation law, measured, and why ``with_concomitant`` does not use the usual form. The first-order
    expansion keeps ``|B_perp|^2 / 2 B0`` and drops the next order, ``-B_n |B_perp|^2 / 2 B0^2`` with
    ``B_n = G . r`` the gradient's own field; its GRADIENT is then short of the full ``|B0 n + B|``'s by about
    ``1.3 B_n / B0``, the factor coming from ``grad B_n = G`` multiplying ``|B_perp|^2`` as well. Linear in
    ``B_n / B0``, so halving the field doubles it: at 64 mT and 67 mT/m it is 4 per cent of the term at 8 cm
    and nothing at 3 T. The exact magnitude the sequence applies is within the coil's residual of the wire."""
    R = np.eye(3)
    n = np.array([0.0, 0.0, 1.0])
    coils = _coils(R)
    G_vec = 0.067 * np.array([2.0, 0.0, 1.0]) / np.sqrt(5.0)
    r = np.array([0.06, 0.02, -0.05])
    h = 1e-4

    def full(points):
        return np.linalg.norm(B0_T * n + _field(coils, G_vec, np.atleast_2d(points), R), axis=-1)

    def b_z(points):
        return _field(coils, G_vec, np.atleast_2d(points), R) @ n

    def truncated(points):
        B = _field(coils, G_vec, np.atleast_2d(points), R)
        perp = B - np.outer(B @ n, n)
        return np.sum(perp ** 2, axis=-1) / (2.0 * B0_T)

    grad = lambda f: np.array([(f(r + h * e) - f(r - h * e))[0] / (2.0 * h) for e in np.eye(3)])
    beyond_bz = grad(full) - grad(b_z)                          # everything the term is meant to carry
    short = np.linalg.norm(grad(truncated) - beyond_bz) / np.linalg.norm(beyond_bz)
    ratio = short / (b_z(r[None])[0] / B0_T)
    assert 1.0 < ratio < 1.6, f"the truncation is {short:.1%} at B_n / B0 = {b_z(r[None])[0] / B0_T:.3f}"
    if B0_T < 0.1:
        assert short > 0.03, "at the Swoop's own field and gradient the first-order form is short by percent"
    # and the sequence's own term, the exact magnitude, is not short: what is left is the coil's residual
    got, _want = _extra_gradient(coils, R, G_vec, r, B0_T=B0_T)
    exact = np.linalg.norm(got - beyond_bz) / np.linalg.norm(beyond_bz)
    assert exact < 0.02, f"the exact form misses the full magnitude's gradient by {exact:.1%}"
    if B0_T < 0.1:
        assert exact < 0.5 * short          # at 256 mT the truncation is already below the coil's own residual


@pytest.fixture(scope="module")
def pack(tmp_path_factory):
    """A free-diffusion walk on the acquisition's own save grid, so the replay's per-save weights and the
    oracle's per-save sum are the same quadrature and only the physics can differ."""
    seq = _acquisition()
    n_t, dt = int(seq.n_t), float(seq.dt)
    walk = d.simulate_trajectories(1500, 2.0e-9, d.FreeDiffusion(), (n_t - 1) * dt, dt, seed=3, require_gpu=False)
    out = tmp_path_factory.mktemp("pk") / "free.rpk"
    build_replay_pack(walk, id="test/free", license="x", citation="x", K=40, out_path=str(out))
    return read_rpk(str(out))


def _acquisition():
    return sequences.pgse([[2.0, 0.0, 1.0], [1.0, 0.0, -2.0]], 0.012, 0.030, bvalues=[1.2e9, 1.2e9], n_t=121)


@pytest.mark.parametrize("frame", list(FRAMES))
def test_a_replayed_walk_sees_the_field_the_coils_actually_make(pack, frame):
    """End to end, two ways, both against the TRUE field magnitude of the wire and neither knowing the
    expansion.

    Walker by walker: each walker's phase from ``|B0 n + B_coil(r_v + u(t), t)|`` along its own decoded
    path, against the same sum over the same path with the gradient the composed sequence plays,
    ``L(r_v) G`` plus the Maxwell term. Same quadrature on both sides, so only the physics can differ; the
    second-order term in ``u`` (microns against a coil of decimetres) is inside the comparison.

    Through the pack: the pack replayed at the composed sequence, against the pack replayed at the
    numerical gradient of ``|B|`` at the voxel -- the replay's own contraction on both sides, so the
    sequence transforms are held to magnetostatics with the codec and the exact per-save weights in the
    loop.

    And the replay WITHOUT the term must miss both, so neither check can pass by the term being small."""
    R = FRAMES[frame]
    n = R @ np.array([0.0, 0.0, 1.0])
    coils = _coils(R)
    r_v = R @ np.array([0.04, 0.015, -0.03])                   # the voxel, 5 cm out; the walk is micron-scale
    seq = _acquisition()
    G_grid = np.asarray(seq.G, np.float64) @ R.T                # the command, re-described in the turned frame
    seq = seq.with_gradient(G_grid)
    n_t, dt = int(seq.n_t), float(seq.dt)
    assert n_t == pack.n_t and abs(dt - pack.dt) < 1e-12
    sign = np.asarray(seq.rf.sign(np.arange(n_t) * dt), np.float64)
    n_meas = seq.n_meas

    def magnitude(points, m, t):
        """``|B0 n + B_coil|`` at ``points`` for measurement ``m`` at save ``t``, from the wire."""
        return np.linalg.norm(B0 * n + _field(coils, G_grid[m, t], points, R), axis=-1)

    # (1) walker by walker, along the decoded path
    u = pack.positions()                                        # (n_w, n_t, 3), the substrate's own frame
    n_w = u.shape[0]
    L = _nonlinearity(coils, r_v, R)
    with_term = seq.with_gradient_nonlinearity(L).with_concomitant(r_v, B0, b0_axis=tuple(n))
    without = seq.with_gradient_nonlinearity(L)
    G_with = np.asarray(with_term.G, np.float64)
    G_without = np.asarray(without.G, np.float64)
    phi_true = np.zeros((n_w, n_meas)); phi_with = np.zeros((n_w, n_meas)); phi_without = np.zeros((n_w, n_meas))
    for m in range(n_meas):
        for t in range(n_t):
            if sign[t] == 0.0 or not np.any(G_grid[m, t]):
                continue
            mag = magnitude(r_v[None, :] + u[:, t, :], m, t) - magnitude(r_v[None, :], m, t)[0]
            phi_true[:, m] += sign[t] * mag * dt
            phi_with[:, m] += sign[t] * (u[:, t, :] @ G_with[m, t]) * dt
            phi_without[:, m] += sign[t] * (u[:, t, :] @ G_without[m, t]) * dt
    S = lambda phi: np.abs(np.mean(np.exp(1j * GAMMA * phi), axis=0))
    effect = np.abs(S(phi_true) - S(phi_without)).max()
    err = np.abs(S(phi_true) - S(phi_with)).max()
    # what is left is the coil's residual nonlinearity: 0.2 per cent of the effect here. The first-order form
    # of the term left 8.3 per cent (the law below); a frame, sign or alpha error is the whole term
    assert effect > 1.5e-3, f"the Maxwell term moves this signal by only {effect:.2e}; the check would be empty"
    assert err < 0.01 * effect, f"walker by walker, the composed gradient misses the true field by {err:.2e} " \
                                f"against an effect of {effect:.2e}"

    # (2) through the pack: the numerical gradient of |B| at the voxel, per save, as the played gradient
    h = 1e-4
    g_true = np.zeros_like(G_grid)
    for m in range(n_meas):
        for t in range(n_t):
            if not np.any(G_grid[m, t]):
                continue
            for i, e in enumerate(np.eye(3)):
                g_true[m, t, i] = (magnitude((r_v + h * e)[None], m, t) - magnitude((r_v - h * e)[None], m, t))[0] / (2.0 * h)
    S_true = pack.replay(seq.with_gradient(g_true))
    S_with = pack.replay(with_term)
    S_without = pack.replay(without)
    effect_pack = np.abs(S_true - S_without).max()
    err_pack = np.abs(S_true - S_with).max()
    assert effect_pack > 1.5e-3
    assert err_pack < 0.01 * effect_pack, f"through the pack, the composed sequence misses the true field's " \
                                          f"gradient by {err_pack:.2e} against an effect of {effect_pack:.2e}"
    # and the walker-level sum with that same numerical gradient IS the true path integral: the term's
    # second order in the walker's own excursion is nothing, so the residual above is all at the voxel
    S_g = lambda G: np.abs(np.mean(np.exp(1j * GAMMA * np.einsum("wtd,mtd->wm", u, G * sign[None, :, None] * dt)), axis=0))
    assert np.abs(S(phi_true) - S_g(g_true)).max() < 1e-5
