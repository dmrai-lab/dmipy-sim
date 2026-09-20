"""The gradient a voxel actually receives, and the two routes to it (dmipy-sim#369).

Three things stand between the prescribed gradient and the delivered one -- the coils' nonlinearity, the
magnet's background, and the coils' concomitant term. All three were modelled and none reached a signal.
"""
import numpy as np
import pytest

from dmipy_sim import sequences
from dmipy_sim.acquisition.scanners import ScannerLimits
from dmipy_sim.phantom import Grid
from dmipy_sim.phantom.bore import (_effective, delivered_b, delivered_gradient, delivered_weights,
                                    gradient_tensor_map)
from dmipy_sim.replay.replay import _compile_effective

SH = (3, 3, 3)
K, N_T = 16, 48


@pytest.fixture(scope="module")
def setup():
    s = ScannerLimits.of("swoop")
    grid = Grid(shape=SH, voxel_size_m=(0.03,) * 3,
                origin_m=tuple(-0.03 * (n - 1) / 2 for n in SH), isocenter_m=(0.0, 0.0, 0.0))
    seq = sequences.pgse([[1, 0, 0], [0, 1, 0], [0, 0, 1]], 0.01, 0.03, bvalues=[1e9], n_t=N_T)
    return s, grid, seq


@pytest.mark.parametrize("flags", [
    {"nonlinearity": True, "background": False, "concomitant": False},
    {"nonlinearity": False, "background": True, "concomitant": False},
    {"nonlinearity": False, "background": False, "concomitant": True},
    {"nonlinearity": True, "background": True, "concomitant": False},
    {"nonlinearity": True, "background": False, "concomitant": True},
    {"nonlinearity": False, "background": True, "concomitant": True},
    {"nonlinearity": True, "background": True, "concomitant": True},
])
def test_the_fast_route_is_the_reference_route_in_another_order(setup, flags):
    """``delivered_weights`` works in the replay's coefficient space at a few thousand flops per voxel, by
    exploiting that ``bridge_projection`` is exactly linear in the gradient. That is a reordering of the
    same sum, not an approximation, so it must agree with building the sequence per voxel.

    EVERY combination is checked, not the terms one at a time. Individually all three agreed to 1e-8 while
    the combination was wrong by 4e-4, because the concomitant term is quadratic in the gradient the COILS
    deliver and composing it as if it were independent drops the cross terms."""
    s, grid, seq = setup
    G = delivered_gradient(s, grid, seq, **flags)
    ref = np.stack([_compile_effective(_effective(seq, g), seq.dt, K, N_T) for g in G])
    fast = delivered_weights(s, grid, seq, K=K, n_t=N_T, dt_pack=seq.dt, **flags)
    rel = np.abs(ref - fast).max() / max(np.abs(ref).max(), 1e-300)
    assert rel < 1e-6, f"{flags} disagree by {rel:.2e}"


def test_the_concomitant_term_is_the_coils_and_not_the_magnets(setup):
    """The Maxwell term is a quadratic form in the field the GRADIENT COILS produce. A static inhomogeneity
    is not produced by them and contributes none of it -- ``with_concomitant`` reads ``designed_gradient``
    for exactly that reason. Feeding it the background as though it were coil gradient moved the answer by
    4e-4, which is how this was found."""
    s, grid, seq = setup
    r = np.array([0.03, -0.02, 0.04])
    g0 = np.atleast_2d(s.b0_gradient(r[None]))[0]
    chained = seq.with_background_gradient(g0).with_concomitant(r, s.field_T).G
    background_only = seq.with_background_gradient(g0).G
    coils_only = seq.with_concomitant(r, s.field_T).G - np.asarray(seq.G, np.float64)
    # the concomitant increment must be the one the bare coils make, unchanged by the background
    # tolerance is float32 roundoff on the gradient arrays, not a fudge: the quantity being compared is
    # ~7e-2 T/m and the residual is 3e-11, which is 4e-10 relative.
    scale = np.abs(np.asarray(seq.G, np.float64)).max()
    assert np.abs((chained - background_only) - coils_only).max() < 1e-6 * scale


def test_the_tensor_enters_as_a_similarity_and_not_a_scale(setup):
    """b becomes |L u|^2 b along a direction that is no longer u. A model applying only the magnitude keeps
    the direction and loses the part that mixes measurements -- and a TRANSPOSED tensor would pass any test
    that used a symmetric L, so this one is deliberately asymmetric."""
    _s, _grid, seq = setup
    L = np.eye(3)
    L[1, 1], L[0, 1] = 1.10, 0.05                        # asymmetric on purpose
    got = seq.with_gradient_tensor(L).b() / seq.b()
    want = [float(np.sum((L @ u) ** 2)) for u in np.eye(3)]
    np.testing.assert_allclose(got, want, rtol=1e-6)
    assert not np.allclose(got, [float(np.sum((L.T @ u) ** 2)) for u in np.eye(3)], rtol=1e-6), \
        "this L is symmetric enough that a transpose would pass -- the test cannot catch the bug it guards"


def test_the_delivered_b_agrees_with_the_gradient_route(setup):
    """The acceptance criterion of #369: ``delivered_b`` computes the b a voxel receives by its own route (a
    polynomial in position fitted from probe sequences), and ``delivered_gradient`` builds the acquisition
    per voxel. Two independent paths to the same number."""
    s, grid, seq = setup
    G = delivered_gradient(s, grid, seq, nonlinearity=False)      # delivered_b covers background+concomitant
    from dmipy_sim.acquisition.waveforms import b_from_gradient
    from dataclasses import replace
    direct = np.stack([replace(seq, G=g).b() for g in G])
    poly = delivered_b(s, grid, seq)
    rel = np.abs(direct - poly).max() / np.abs(poly).max()
    assert rel < 5e-3, f"the two routes to the delivered b differ by {rel:.2e}"


def test_the_encoding_varies_across_the_grid_and_not_merely_in_scale(setup):
    """A uniform L would pass a test that only checked the numbers moved. What must be asserted is the
    SPATIAL pattern: the delivered b differs voxel to voxel, and the encoding DIRECTION tilts, which is the
    part a diagonal or scalar model cannot produce."""
    s, grid, seq = setup
    G = delivered_gradient(s, grid, seq)
    b = np.stack([np.linalg.norm(g.reshape(len(g), -1), axis=1) for g in G])
    assert np.ptp(b, axis=0).max() / b.mean() > 1e-3, "the encoding does not vary across the grid"

    L = gradient_tensor_map(s, grid)(grid.positions_m(grid.every_voxel))
    u = np.array([0.0, 1.0, 0.0])
    tilt = [np.rad2deg(np.arccos(np.clip(abs((M @ u) @ u) / np.linalg.norm(M @ u), 0, 1))) for M in L]
    assert max(tilt) > 0.5, f"the encoding direction never tilts (max {max(tilt):.2f} deg)"
