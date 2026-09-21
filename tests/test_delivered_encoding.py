"""The gradient a voxel actually receives, and the two routes to it (dmipy-sim#369).

Three things stand between the prescribed gradient and the delivered one -- the coils' nonlinearity, the
magnet's background, and the coils' concomitant term. All three were modelled and none reached a signal.
"""
import numpy as np
import pytest

from dmipy_sim import sequences
from dmipy_sim.acquisition.scanners import ScannerLimits
from dmipy_sim.phantom import Grid
from dmipy_sim.phantom.bore import (delivered_b, delivered_gradient, delivered_weights,
                                    gradient_tensor_map)
from dmipy_sim.replay.replay import _compile_effective

SH = (3, 3, 3)
K, N_T = 16, 48


@pytest.fixture(scope="module")
def setup():
    s = ScannerLimits.of("swoop")
    grid = Grid(shape=SH, voxel_size_m=(0.03,) * 3,
                origin_m=tuple(-0.03 * (n - 1) / 2 for n in SH), isocenter_m=(0.0, 0.0, 0.0))
    # OBLIQUE on purpose. With cardinal directions the Gx Gz x z and Gy Gz y z cross terms of the Maxwell
    # formula are identically zero, so 40 per cent of it is unreachable and deleting it passes every test
    # in this file.
    seq = sequences.pgse([[1, 1, 1], [1, 0, 1], [2, -1, 3]], 0.01, 0.03, bvalues=[1e9], n_t=N_T)
    return s, grid, seq


#: A genuine rotation (Rz31 Ry17 Rx44), orthonormal to 1e-12. A hand-typed near-rotation is refused by
#: gradient_tensor_map, correctly -- a grid's axes are orthonormal in the scanner.
ROT = np.array([[0.81971317, -0.19639803, 0.53805031],
                [0.49253336, 0.72119799, -0.48711841],
                [-0.29237170, 0.66430510, 0.68790807]])


def _reference(seq, G, K, n_t, dt_pack=None):
    """W built the way ``ReplayPack._prepare`` builds it: ``ScannerSequence.G_eff``, resampled onto the
    pack's save grid, then projected.

    Deliberately not built from ``bore`` itself: a reference built from the module under test cannot fail
    when that module is wrong, and the "carried through the RF sign" claim needs coverage of its own.
    The RESAMPLE is here for the same reason -- weights on the sequence's own grid are self-consistent and
    incompatible with a pack, and leaving it out of both routes made the parity test blind to it.
    """
    from dataclasses import replace
    from dmipy_sim.replay._replay_kernel import effective_gradient
    dt = seq.dt if dt_pack is None else dt_pack
    return np.stack([_compile_effective(
        effective_gradient(replace(seq, G=np.asarray(g, np.float32)).G_eff, seq.dt, n_t, dt), dt, K, n_t)
        for g in G])


@pytest.mark.parametrize("rotation", [None, ROT], ids=["aligned", "oblique"])
@pytest.mark.parametrize("flags", [
    {"nonlinearity": True, "background": False, "concomitant": False},
    {"nonlinearity": False, "background": True, "concomitant": False},
    {"nonlinearity": False, "background": False, "concomitant": True},
    {"nonlinearity": True, "background": True, "concomitant": False},
    {"nonlinearity": True, "background": False, "concomitant": True},
    {"nonlinearity": False, "background": True, "concomitant": True},
    {"nonlinearity": True, "background": True, "concomitant": True},
])
def test_the_fast_route_is_the_reference_route_in_another_order(setup, flags, rotation):
    """``delivered_weights`` works in the replay's coefficient space at a few thousand flops per voxel, by
    exploiting that ``bridge_projection`` is exactly linear in the gradient. That is a reordering of the
    same sum, not an approximation, so it must agree with building the sequence per voxel.

    EVERY combination is checked, not the terms one at a time. Individually all three agreed to 1e-8 while
    the combination was wrong by 4e-4, because the concomitant term is quadratic in the gradient the COILS
    deliver and composing it as if it were independent drops the cross terms."""
    s, grid, seq = setup
    G = delivered_gradient(s, grid, seq, to_scanner=rotation, **flags)
    ref = _reference(seq, G, K, N_T)
    fast = delivered_weights(s, grid, seq, K=K, n_t=N_T, dt_pack=seq.dt, to_scanner=rotation, **flags)
    rel = np.abs(ref - fast).max() / max(np.abs(ref).max(), 1e-300)
    assert rel < 1e-6, f"{flags} at rotation={rotation is not None} disagree by {rel:.2e}"


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
    G = np.asarray(seq.G, dtype=np.float64)                       # the direction actually played
    u = np.stack([g[np.argmax(np.linalg.norm(g, axis=-1))] for g in G])
    u = u / np.linalg.norm(u, axis=-1, keepdims=True)
    got = seq.with_gradient_nonlinearity(L).b() / seq.b()
    want = [float(np.sum((L @ v) ** 2)) for v in u]
    np.testing.assert_allclose(got, want, rtol=1e-6)
    assert not np.allclose(got, [float(np.sum((L.T @ v) ** 2)) for v in u], rtol=1e-6), \
        "this L is symmetric enough that a transpose would pass -- the test cannot catch the bug it guards"


def test_the_delivered_b_agrees_with_the_gradient_route(setup):
    """The acceptance criterion of #369: ``delivered_b`` computes the b a voxel receives by its own route (a
    polynomial in position fitted from probe sequences), and ``delivered_gradient`` builds the acquisition
    per voxel. Two independent paths to the same number."""
    from dataclasses import replace
    s, grid, seq = setup
    # delivered_b applies the BACKGROUND only -- not the concomitant and not the nonlinearity. Comparing it
    # against a route that includes them is comparing different quantities; the discrepancy then grows
    # LINEARLY with distance (the concomitant term), which is what gave this away.
    G = delivered_gradient(s, grid, seq, nonlinearity=False, concomitant=False)
    direct = np.stack([replace(seq, G=np.asarray(g, np.float32)).b() for g in G])
    poly = delivered_b(s, grid, seq)
    rel = np.abs(direct - poly).max() / np.abs(poly).max()
    assert rel < 1e-5, f"the two routes to the delivered b differ by {rel:.2e}"


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


def test_the_concomitant_is_computed_for_the_field_direction_the_machine_declares(setup):
    """The formula is written for B0 along the third component, which is right for every cylindrical magnet
    and WRONG for a bi-planar one. The Swoop declares ``b0_axis = (0, 1, 0)``, and nothing on this path
    rotated into the magnet's frame: the whole quadratic form sat 90 degrees out. It moves the concomitant
    contribution to b by about a factor of two and permutes it between directions, so a marginal min/max
    over a symmetric grid would not have noticed."""
    s, _grid, seq = setup
    r = np.array([0.04, 0.03, -0.02])
    assumed = seq.with_concomitant(r, s.field_T).b() / seq.b() - 1.0
    actual = seq.with_concomitant(r, s.field_T, b0_axis=s.b0_axis).b() / seq.b() - 1.0
    assert not np.allclose(assumed, actual, rtol=0.2), "the b0_axis argument changes nothing"

    # a cylindrical magnet must be untouched, and the rotation must be a rotation
    z = seq.with_concomitant(r, s.field_T, b0_axis=(0, 0, 1))
    assert np.abs(np.asarray(z.G, np.float64) - np.asarray(seq.with_concomitant(r, s.field_T).G,
                                                           np.float64)).max() == 0.0
    for bad in ((0, 0, 0), (np.nan, 0, 1), (0, 1)):
        with pytest.raises(ValueError, match="b0_axis"):
            seq.with_concomitant(r, s.field_T, b0_axis=bad)


def test_the_nonlinearity_is_the_coils_gradient_and_not_the_magnets(setup):
    """``imposed_gradient`` means what the MAGNET imposes, and ``designed_gradient`` subtracts it so the
    builder's guarantees still read true. A coil's nonlinearity obeys those guarantees -- it vanishes where
    the commanded gradient does -- and it IS the gradient the coils design, which is what the Maxwell term
    must be a quadratic form in. Booking it as the magnet's left ``designed_gradient`` at the nominal G and
    silently dropped every cross term between the nonlinearity and the concomitant field."""
    _s, _grid, seq = setup
    L = np.eye(3)
    L[1, 1], L[0, 1] = 1.10, 0.05
    got = seq.with_gradient_nonlinearity(L)
    np.testing.assert_allclose(np.asarray(got.designed_gradient, np.float64),
                               np.asarray(got.G, np.float64), atol=0)
    assert got.imposed_gradient is None, "a coil's nonlinearity is not imposed by the magnet"
    # and a background still books as the magnet's, so designed_gradient takes it back out
    g0 = np.array([1e-3, -5e-4, 2e-4])
    both = got.with_background_gradient(g0)
    np.testing.assert_allclose(np.asarray(both.designed_gradient, np.float64),
                               np.asarray(got.G, np.float64), atol=1e-9)


# ── end to end: the delivered gradient through to a signal (dmipy-sim#369) ──────────────────────────
@pytest.fixture(scope="module")
def pack(tmp_path_factory):
    import dmipy_sim as d
    from dmipy_sim.replay.bank import build_replay_pack
    from dmipy_sim.replay import read_rpk
    out = tmp_path_factory.mktemp("pk") / "w.rpk"
    # 40 000, not 400. At 400 the Monte-Carlo floor on this pack is 11-14 per cent in ADC against an
    # effect of a few per cent, so a per-direction bias has an SNR under one and its SIGN is not determined.
    # The assertions below still passed at 400 -- they compare two weight sets on the SAME walkers, which
    # cancels most of the floor -- but any NUMBER read off such a fixture is a draw, not a measurement.
    walk = d.simulate_trajectories(40_000, 2e-9, d.FreeDiffusion(), 0.05, 1e-3, seed=0, require_gpu=False)
    build_replay_pack(walk, id="t", license="x", citation="x", K=8, out_path=str(out))
    return read_rpk(str(out))


def _signal(phi):
    return np.abs(np.mean(np.exp(1j * phi), axis=0))


def test_the_weights_a_caller_supplies_are_the_ones_the_replay_would_have_built(setup, pack):
    """The contract of ``walker_phases(weights=)``. Handing back the replay's OWN weights must change
    nothing, or a per-voxel override is not substituting for the shared path but for something else."""
    from dmipy_sim.replay.replay import _compile_effective
    _s, _grid, seq = setup
    P = pack._prepare(seq, tissue=None, scanner=None, orientation=None, compartment=None)
    W = _compile_effective(P["Geff"], P["dt"], pack.K, P["n_t"])
    _w, _e, base = pack.walker_phases(seq)
    _w, _e, given = pack.walker_phases(seq, weights=W)
    assert np.abs(base - given).max() == 0.0


def test_delivered_weights_land_on_the_packs_save_grid_and_not_the_sequences(setup, pack):
    """``_prepare`` resamples the effective gradient onto the grid the WALK was saved on before projecting.
    Weights built on the sequence's own grid are self-consistent and incompatible with the pack, and a
    parity test between two routes that both skip the resample cannot see it -- which is how this survived.

    The check that catches it: at isocentre every scanner term vanishes, so the delivered weights must
    reproduce the ideal-magnet replay EXACTLY, and they only can if they live on the same grid."""
    s, grid, seq = setup
    W = delivered_weights(s, grid, seq, K=pack.K, n_t=pack.n_t, dt_pack=pack.dt)
    assert W.shape[1:] == (pack.n_coeffs * 3, seq.n_meas)
    iso = int(np.argmin(np.linalg.norm(grid.offset_m(grid.every_voxel).reshape(-1, 3), axis=1)))
    _w, _e, ideal = pack.walker_phases(seq)
    _w, _e, at_iso = pack.walker_phases(seq, weights=W[iso])
    assert np.abs(ideal - at_iso).max() == 0.0, "the scanner does not vanish at its own isocentre"


def test_the_scanner_biases_the_signal_by_position_and_by_direction(setup, pack):
    """What the wiring is FOR. The bias must depend on where the voxel is AND on which way the measurement
    encodes -- a model that only rescaled would move every direction together, and that is the part a
    diagonal L or a scalar b-correction cannot produce."""
    s, grid, seq = setup
    W = delivered_weights(s, grid, seq, K=pack.K, n_t=pack.n_t, dt_pack=pack.dt)
    d_v = grid.offset_m(grid.every_voxel).reshape(-1, 3)
    iso, corner = int(np.argmin(np.linalg.norm(d_v, axis=1))), int(np.argmax(np.linalg.norm(d_v, axis=1)))
    _w, _e, p_iso = pack.walker_phases(seq, weights=W[iso])
    _w, _e, p_cor = pack.walker_phases(seq, weights=W[corner])
    s_iso, s_cor = _signal(p_iso), _signal(p_cor)
    bias = np.log(s_cor / s_iso)
    assert np.abs(bias).max() > 1e-3, "the scanner does not bias the signal at all"
    assert np.ptp(bias) > 0.5 * np.abs(bias).max(), \
        f"the bias is nearly the same for every direction ({bias}) -- that is a rescale, not an encoding error"
    # and it must grow with distance rather than appear only at the edge
    mid = int(np.argmin(np.abs(np.linalg.norm(d_v, axis=1) - 0.5 * np.linalg.norm(d_v[corner]))))
    _w, _e, p_mid = pack.walker_phases(seq, weights=W[mid])
    assert np.abs(np.log(_signal(p_mid) / s_iso)).max() < np.abs(bias).max()


def test_the_delivered_weights_are_covariant_under_a_rigid_re_description(setup, pack):
    """THE test the frame conventions needed, and the one every parity test is structurally blind to.

    Describing the same physics in a rotated grid is not a different experiment. A voxel at bore position q
    with gradient G, re-described in a grid carrying ``to_scanner = R``, sits at grid offset ``q R`` with
    gradient ``G R`` -- and the delivered weights must be the aligned answer with only its AXIS index
    rotated. Nothing about the magnet changed.

    Parity cannot see a frame error because both routes share the convention, and the isocentre anchor
    cannot either because every term vanishes there by construction. Three mutants survived the whole
    90-test neighbourhood: the tensor similarity applied as ``R L R^T`` instead of ``R^T L R`` (worth 4.5
    per cent in log S at a corner -- larger than the effect this branch exists to show, and it flips a
    direction's sign), the concomitant read at a bore position instead of a grid one, and the background
    rotated by ``R^T`` instead of ``R``. This kills all three, with no pack and no second route."""
    from dataclasses import replace
    s, _grid, seq = setup
    R = ROT
    q = np.array([0.03, -0.02, 0.025])                       # one bore position, well inside the anchor
    shape, vs = (1, 1, 1), 0.01

    aligned = Grid(shape=shape, voxel_size_m=(vs,) * 3, origin_m=tuple(q), isocenter_m=(0.0, 0.0, 0.0))
    turned = Grid(shape=shape, voxel_size_m=(vs,) * 3, origin_m=tuple(q @ R), isocenter_m=(0.0, 0.0, 0.0))
    posed = replace(seq, G=(np.asarray(seq.G, np.float64) @ R).astype(np.float32))

    A = delivered_weights(s, aligned, seq, K=pack.K, n_t=pack.n_t, dt_pack=pack.dt)[0]
    B = delivered_weights(s, turned, posed, K=pack.K, n_t=pack.n_t, dt_pack=pack.dt, to_scanner=R)[0]
    A_rot = np.einsum("kjm,ji->kim", A.reshape(-1, 3, A.shape[-1]), R).reshape(A.shape)
    rel = np.abs(B - A_rot).max() / max(np.abs(A).max(), 1e-300)
    assert rel < 1e-5, f"the same physics re-described in a rotated grid gives a different answer: {rel:.2e}"


def test_an_oblique_grid_is_not_silently_evaluated_in_the_wrong_frame(setup, pack):
    """A grid that CARRIES its rotation must give the same answer as one told it explicitly. The tensor map
    was handed the caller's raw ``to_scanner`` while the background and the concomitant got the resolved
    frame, so an oblique grid mixed frames -- rotated for two terms, not for the third."""
    s, _grid, seq = setup
    shape, vs = (2, 2, 2), 0.02
    org = tuple(-vs * (n - 1) / 2 for n in shape)
    carried = Grid(shape=shape, voxel_size_m=(vs,) * 3, origin_m=org, isocenter_m=(0.0, 0.0, 0.0),
                   to_scanner=ROT)
    explicit = Grid(shape=shape, voxel_size_m=(vs,) * 3, origin_m=org, isocenter_m=(0.0, 0.0, 0.0))
    a = delivered_weights(s, carried, seq, K=pack.K, n_t=pack.n_t, dt_pack=pack.dt)
    b = delivered_weights(s, explicit, seq, K=pack.K, n_t=pack.n_t, dt_pack=pack.dt, to_scanner=ROT)
    assert np.abs(a - b).max() / np.abs(b).max() < 1e-9


def test_weights_refuse_the_two_routes_that_would_discard_them(setup, pack):
    """Both were silent. A pose is carried by rotating the gradient BEFORE projection, so supplied weights
    replace it wholesale and the pose vanishes (48 per cent of the signal). And the susceptibility-field
    branches rebuild the gradient from the nominal sequence, so with a field active the result is
    bit-identical to passing no weights at all -- including weights of zero, in exactly the low-field case
    this feature exists for."""
    s, grid, seq = setup
    W = delivered_weights(s, grid, seq, K=pack.K, n_t=pack.n_t, dt_pack=pack.dt)[0]
    with pytest.raises(ValueError, match="cannot both be given"):
        pack.walker_phases(seq, weights=W, orientation=np.eye(3))
    with pytest.raises(ValueError, match="not finite"):
        pack.walker_phases(seq, weights=np.full_like(W, np.nan))
    with pytest.raises(ValueError, match="real floating point"):
        pack.walker_phases(seq, weights=np.ones(W.shape, dtype=np.int64))


def test_the_signal_level_bias_is_the_b_level_prediction(setup, pack):
    """The acceptance criterion of #369, end to end. For gradient nonlinearity alone the delivered b is
    ``|L u|^2 b``, so a free-diffusion pack's ADC fitted against the NOMINAL b must be biased by exactly
    ``|L u|^2 - 1``. That is a closed form, computed from the tensor and the direction, with no replay in
    it -- so agreement tests the whole path from the catalogue through the contraction to a signal.

    Sign convention, stated because I had it inverted: ``S = exp(-b_eff D)``, so fitting against nominal b
    gives ``ADC_fit / D = b_eff / b_nom`` and the bias is ``log S_here / log S_iso - 1``. A voxel where
    ``|L u| > 1`` receives MORE diffusion weighting than prescribed, so its signal is LOWER and its fitted
    ADC HIGHER."""
    s, grid, seq = setup
    W = delivered_weights(s, grid, seq, K=pack.K, n_t=pack.n_t, dt_pack=pack.dt,
                          background=False, concomitant=False)
    d_v = grid.offset_m(grid.every_voxel).reshape(-1, 3)
    iso, corner = int(np.argmin(np.linalg.norm(d_v, axis=1))), int(np.argmax(np.linalg.norm(d_v, axis=1)))
    s_iso = _signal(pack.walker_phases(seq, weights=W[iso])[2])
    s_cor = _signal(pack.walker_phases(seq, weights=W[corner])[2])
    measured = np.log(s_cor) / np.log(s_iso) - 1.0

    L = gradient_tensor_map(s, grid)(grid.positions_m(grid.every_voxel))[corner]
    G = np.asarray(seq.G, np.float64)
    u = np.stack([g[np.argmax(np.linalg.norm(g, axis=-1))] for g in G])
    u = u / np.linalg.norm(u, axis=-1, keepdims=True)
    predicted = np.array([float(np.sum((L @ v) ** 2)) for v in u]) - 1.0

    np.testing.assert_allclose(measured, predicted, atol=2e-3)
    assert np.abs(predicted).max() > 5e-3, "this fixture barely exercises the nonlinearity"
