"""The pose expansion in closed form (#197): for a single-direction encoding the response over poses is a sum of
plane waves whose harmonics are the Rayleigh expansion, computed per walker with no quadrature. Certified against
the direct posed replay (exact), against the quadrature route (to that route's own misfit), and by its band bound."""

import numpy as np
import pytest
from scipy.special import spherical_jn

import dmipy_sim as d
from dmipy_sim import sequences
from dmipy_sim.replay import read_rpk, so3
from dmipy_sim.replay.bank import build_replay_pack
from dmipy_sim.replay.so3 import Distribution
from dmipy_sim.spec.tissue import Tissue
from tests.replay_frames import field_along


@pytest.fixture(scope="module")
def pack(tmp_path_factory):
    g = d.PackedCylinders([1e-6], [[0.0, 0.0]], 10e-6)
    walk = d.simulate_trajectories(500, 2e-9, g, 10e-3, 5e-4, seed=2, require_gpu=False)
    p = tmp_path_factory.mktemp("pk") / "c.rpk"
    build_replay_pack(walk, id="t/c", license="x", citation="x", K=8, out_path=str(p))
    return read_rpk(str(p))


@pytest.fixture(scope="module")
def seq():
    return sequences.pgse([[1, 0, 0], [0, 0, 1], [0.6, 0.8, 0.0], [0.0, 0.6, 0.8]], 2e-3, 5e-3,
                          gradient_strengths=[0.3, 0.3, 0.3, 0.3], TE=10e-3)


def test_the_expansion_is_the_direct_posed_replay_at_every_rotation(pack, seq):
    """Pose covariance, the exact statement: the expansion evaluated at a rotation is the pack replayed there."""
    pr = pack.pose_response(seq, method="closed")
    assert pr.n_samples == 0 and pr.misfit.max() < 1e-7                # nothing sampled; the tail is the bound
    for R in so3.haar_rotations(16, 3):
        np.testing.assert_allclose(pr.at(R), pack.replay(seq, orientation=R, complex_signal=True), atol=1e-8)


def test_the_closed_form_agrees_with_the_quadrature_route_to_its_misfit(pack, seq):
    pc = pack.pose_response(seq, method="closed")
    pq = pack.pose_response(seq, method="quadrature")
    assert pq.misfit.max() > 0                                          # the quadrature route certifies by sampling
    for dist in (Distribution.axis((0.3, 0.5, 0.81), pq.lmax, 0), Distribution.watson(6.0, mu=(0, 0, 1), lmax=pq.lmax, nmax=0),
                 Distribution.pose(so3.rotation_of((0.2, -0.4, 0.89)), pq.lmax, pq.nmax)):
        np.testing.assert_allclose(pc.compose(dist), pq.compose(dist), atol=3 * pq.misfit.max() + 1e-4)


def test_only_the_retained_band_is_computed_and_it_is_the_same_numbers(pack, seq):
    """keep=(L, 0) builds the n = 0 column alone (the Legendre path), and it equals the n = 0 column of the full
    block: an ODF composition never touches a roll."""
    full = pack.pose_response(seq, method="closed")
    odf = pack.pose_response(seq, method="closed", keep=(8, 0))
    assert odf.nmax == 0 and odf.lmax == 8 and odf.coeffs.shape[1] == so3.n_so3_coeffs(8, 0)
    np.testing.assert_allclose(odf.coeffs, full.retained(8, 0), atol=1e-12)
    d8 = Distribution.watson(10.0, mu=(0.3, 0.5, 0.81), lmax=8, nmax=0)
    np.testing.assert_allclose(odf.compose(d8), full.compose(d8), atol=1e-12)


def test_the_band_follows_the_bessel_tail(pack, seq):
    """Truncating the expansion below its own band changes a composed signal by no more than the weighted
    Bessel tail of the orders dropped -- the bound the band is chosen by."""
    pr = pack.pose_response(seq, method="closed")
    dist = Distribution.axis((0.3, 0.5, 0.81), pr.lmax, 0)
    S = pr.compose(dist)
    # the walkers' phase amplitudes, recomputed the way the expansion does
    P = pack._prepare(seq, tissue=pack.nominal, scanner=pack.nominal_field_T, orientation=None, compartment=None)
    w = P["ew"] / P["norm"]
    for L in (2, 4, 6):
        low = so3.truncate_coeffs(pr.coeffs, pr.lmax, pr.nmax, L, 0)
        S_L = low @ so3.truncate_coeffs(dist.coeffs, dist.lmax, dist.nmax, L, 0)
        err = np.abs(S_L - S).max()
        # the bound: sum over dropped orders of (2l+1) sum_w w |j_l(kappa_w)|, with kappa at its largest
        kappa_max = pr.phase_amplitude
        bound = sum((2 * l + 1) * abs(spherical_jn(l, kappa_max)) for l in range(L + 1, pr.lmax + 1))
        assert err <= bound + 1e-12


def test_encodings_the_closed_form_does_not_take_fall_back_or_refuse(pack):
    ste = sequences.ste(6e-3, gradient_strengths=[0.05, 0.05], TE=10e-3)       # three axes: rank 3, not a direction
    pq = pack.pose_response(ste)                                                # auto: the quadrature route
    assert pq.n_samples > 0
    with pytest.raises(ValueError, match="single-direction"):
        pack.pose_response(ste, method="closed")
    with pytest.raises(ValueError, match="method is"):
        pack.pose_response(ste, method="magic")


def test_the_closed_form_composes_the_direct_replay_on_a_real_pack(seq):
    """The 8k-walker CACTUS pack at four measurements: the closed form takes no samples, and its n = 0 column
    composes the roll-averaged direct replay (the wall-clock is a measurement, not a test: #455)."""
    import os
    path = "/home/rutger/dmrai-ws/packs/cactus_demo_xframe.rpk"
    if not os.path.exists(path):
        pytest.skip("the CACTUS demo pack is not on this machine")
    pk = read_rpk(path)
    seq30 = sequences.pgse([[1, 0, 0], [0, 0, 1], [0.6, 0.8, 0.0], [0.0, 0.6, 0.8]], 8e-3, 16e-3, bvalues=[3e9] * 4, TE=30e-3)
    # no tissue, no scanner: this pack's spec declares a nominal field, and a field response still takes the quadrature
    pr = pk.pose_response(seq30, keep=(8, 0))
    assert pr.n_samples == 0 and pr.phase_amplitude > 5.0                    # a sharp response, closed (no samples) at n = 0
    # and the n = 0 column composes the same signal as the direct replay averaged over the roll about an axis
    axis = np.array([0.3, 0.5, 0.81]); axis /= np.linalg.norm(axis)
    rolls = np.linspace(0, 2 * np.pi, 24, endpoint=False)
    direct = np.mean([pk.replay(seq30, orientation=so3.rotation_of(axis, roll=r), complex_signal=True) for r in rolls], axis=0)
    composed = pr.compose(Distribution.axis(axis, 8, 0))
    np.testing.assert_allclose(composed, direct, atol=3e-3)                  # order 8 of a kappa ~ 20 response: the ODF band


def test_a_shell_is_one_body_and_its_directions_are_free(pack):
    """Every measurement on a shell plays the same waveform, so the closed form contracts the walkers once per
    shell and applies each direction as harmonics: a 64-direction shell equals 64 single-measurement calls, with
    one body per shell."""
    rng = np.random.default_rng(4)
    dirs = rng.normal(size=(64, 3)); dirs /= np.linalg.norm(dirs, axis=1, keepdims=True)
    shell = sequences.pgse(dirs, 2e-3, 5e-3, gradient_strengths=[0.3] * 64, TE=10e-3)
    pr = pack.pose_response(shell, method="closed", keep=(8, 0))
    assert pr.n_bodies == 1
    for i in (0, 17, 63):
        one = sequences.pgse(dirs[i:i + 1], 2e-3, 5e-3, gradient_strengths=[0.3], TE=10e-3)
        np.testing.assert_allclose(pack.pose_response(one, method="closed", keep=(8, 0)).coeffs[0], pr.coeffs[i], atol=1e-12)
    two = sequences.pgse(np.concatenate([dirs, dirs]), 2e-3, 5e-3, gradient_strengths=[0.3] * 64 + [0.15] * 64, TE=10e-3)
    assert pack.pose_response(two, method="closed", keep=(8, 0)).n_bodies == 2
    b0 = sequences.pgse(dirs[:3], 2e-3, 5e-3, bvalues=[0.0, 0.0, 0.0], TE=10e-3)       # b = 0 rows: one body, all ones at l = 0
    p0 = pack.pose_response(b0, method="closed", keep=(4, 0))
    assert p0.n_bodies == 1 and np.allclose(np.abs(p0.compose(Distribution.axis((0, 0, 1), 4, 0))), 1.0)


@pytest.fixture(scope="module")
def field_pack():
    import os
    path = "/home/rutger/dmrai-ws/packs/cactus_demo_xframe.rpk"
    if not os.path.exists(path):
        pytest.skip("the CACTUS demo pack (field tier, path route) is not on this machine")
    return read_rpk(path)


def test_the_field_is_composed_in_closed_form_and_agrees_with_every_other_route(field_pack):
    """With a field the response is the product of the gradient's plane-wave expansion and the field factor's
    harmonics, coupled on both indices: exact against the direct posed replay with the field, equal to the
    quadrature route to that route's misfit, and no g x B0 frame anywhere (#197 step 3, #155)."""
    pk = field_pack
    seq = sequences.pgse([[1, 0, 0], [0, 0, 1], [0.6, 0.8, 0.0]], 6e-3, 15e-3, bvalues=[1.5e9] * 3, TE=30e-3)
    kw = dict(scanner=7.0, tissue=Tissue(chi_iso=-1e-7, chi_aniso=-1e-7))
    seq_y, R_y = field_along(seq, (0.0, 1.0, 0.0))                   # the field along y of the specimen: a pose
    pc = pk.pose_response(seq_y, method="closed", pose=R_y, **kw)
    assert pc.n_samples == 0 and pc.field_lmax >= 2
    for R in so3.haar_rotations(6, 2):
        np.testing.assert_allclose(pc.at(R), pk.replay(seq_y, orientation=R_y @ R, complex_signal=True, **kw), atol=2e-6)
    pq = pk.pose_response(seq_y, method="quadrature", pose=R_y, **kw)
    for d in (Distribution.axis((0.3, 0.5, 0.81), pq.lmax, 0), Distribution.watson(6.0, mu=(0, 0, 1), lmax=pq.lmax, nmax=0)):
        np.testing.assert_allclose(pc.compose(d), pq.compose(d), atol=3 * pq.misfit.max())
    # the parallel geometry the quadrature route had to handle specially (#155) is nothing special here
    seq_x, R_x = field_along(seq, (1.0, 0.0, 0.0))
    pp = pk.pose_response(seq_x, method="closed", keep=(8, 0), pose=R_x, **kw)
    for R in so3.haar_rotations(2, 9):
        full = pk.pose_response(seq_x, method="closed", pose=R_x, **kw)
        np.testing.assert_allclose(full.at(R), pk.replay(seq_x, orientation=R_x @ R, complex_signal=True, **kw), atol=2e-6)
    assert np.isfinite(pp.coeffs).all()


def test_a_machines_nearly_single_direction_waveform_takes_the_closed_form_within_its_bound(pack):
    """A trapezoid played by a machine's coils at 8 cm: the Maxwell gradient goes as the square of the ramps where the
    encoding goes as the ramps, so the delivered waveform is one direction only up to a residual. The closed form
    takes its principal direction and adds the residual's largest phase over every pose (the nuclear norm of its
    walker contraction) to the misfit; the expansion then equals the direct posed replay of the played waveform at
    every rotation to that bound. A residual beyond the bound takes the quadrature."""
    base = sequences.pgse([[1, 0, 0], [0.0, 0.6, 0.8]], 2e-3, 5e-3, gradient_strengths=[0.3, 0.3], TE=10e-3, slew_rate=200.0)
    played = base.with_concomitant(np.array([0.03, 0.05, 0.06]), 3.0)
    G = np.asarray(played.G_eff, np.float64)
    assert all(np.linalg.svd(G[i], compute_uv=False)[1] > 1e-6 * np.linalg.svd(G[i], compute_uv=False)[0] for i in range(2))
    pr = pack.pose_response(played, method="closed")
    assert pr.route == "closed" and 0 < pr.misfit.max() < 0.1 / np.sqrt(pack.n_walkers)
    for R in so3.haar_rotations(12, 5):
        err = np.abs(pr.at(R) - pack.replay(played, orientation=R, complex_signal=True))
        assert np.all(err <= pr.misfit + 1e-8), (err, pr.misfit)
    strong = base.with_concomitant(np.array([0.03, 0.05, 0.06]), 0.02)       # a 20 mT magnet: the residual is large
    assert pack.pose_response(strong).route != "closed"


def test_the_residual_bound_is_the_ensembles(pack):
    """The misfit bounds the ensemble's signal, and |sum_w w (e^{i(phi + d)} - e^{i phi})| <= sum_w |w| |d_w|: the
    walkers' residual bounds weighted as the signal weighs them, not the worst walker's. At 0.5 T the worst walker's
    bound (7.0e-3) is beyond a tenth of this pack's floor (4.5e-3) and the ensemble's (2.6e-3) within it: the
    closed form takes it, and still equals the direct posed replay at every rotation to that bound."""
    base = sequences.pgse([[1, 0, 0], [0.0, 0.6, 0.8]], 2e-3, 5e-3, gradient_strengths=[0.3, 0.3], TE=10e-3, slew_rate=200.0)
    played = base.with_concomitant(np.array([0.03, 0.05, 0.06]), 0.5)
    pr = pack.pose_response(played)
    assert pr.route == "closed" and 1e-3 < pr.misfit.max() < 0.1 / np.sqrt(pack.n_walkers)
    for R in so3.haar_rotations(12, 5):
        err = np.abs(pr.at(R) - pack.replay(played, orientation=R, complex_signal=True))
        assert np.all(err <= pr.misfit + 1e-8), (err, pr.misfit)
def _g0(strength):
    u = np.array([0.3, -0.5, 0.81])
    return strength * u / np.linalg.norm(u)


@pytest.mark.parametrize("kind", ["spin echo", "stimulated echo"])
def test_a_magnets_own_gradient_is_a_second_plane_wave_in_closed_form(pack, kind):
    """A magnet's own gradient g0 is on through every pulse and dead time, so every measurement plays two directions
    with two time courses: the encoding and g0 through the effective gate. Its phase g0 . R n_w (n_w the walker's
    background moment) is a second Rayleigh factor coupled to the gradient's, as the field factor is; the expansion
    equals the direct posed replay of the delivered waveform at every rotation, to its misfit (#565). The background
    here is 50 mT/m, far beyond any magnet's, so the factor's band is several orders and the coupling is exercised."""
    dirs = [[1, 0, 0], [0, 0, 1], [0.6, 0.8, 0.0], [0.0, 0.6, 0.8]]
    if kind == "spin echo":
        base = sequences.pgse(dirs, 2e-3, 5e-3, gradient_strengths=[0.3] * 4, TE=10e-3)
    else:
        base = sequences.pgste(dirs, 1.5e-3, 3e-3, gradient_strengths=[0.3] * 4, TE=10e-3)
    played = base.with_background_gradient(_g0(0.05))
    pr = pack.pose_response(played, method="closed")
    assert pr.route == "closed" and pr.n_samples == 0 and pr.background_lmax >= 4
    assert pr.misfit.max() < 1e-7
    moved = 0.0
    for R in so3.haar_rotations(12, 5):
        direct = pack.replay(played, orientation=R, complex_signal=True)
        assert np.all(np.abs(pr.at(R) - direct) <= pr.misfit + 2e-8), (np.abs(pr.at(R) - direct), pr.misfit)
        moved = max(moved, np.abs(direct - pack.replay(base, orientation=R, complex_signal=True)).max())
    assert moved > 1e-2                                                      # and the background is not a perturbation here


def test_no_background_is_the_expansion_without_one_to_the_bit(pack, seq):
    """A zero background is no background: the same numbers to the bit, alone or batched with an acquisition that
    carries one (a batch that mixes the two kinds expands each kind in its own pass)."""
    plain = pack.pose_response(seq, method="closed")
    zero = pack.pose_response(seq.with_background_gradient((0.0, 0.0, 0.0)), method="closed")
    np.testing.assert_array_equal(zero.coeffs, plain.coeffs)
    np.testing.assert_array_equal(zero.misfit, plain.misfit)
    both = pack.pose_responses([seq, seq.with_background_gradient(_g0(0.01))], method="closed")
    np.testing.assert_array_equal(both[0].coeffs, plain.coeffs)
    assert not hasattr(both[0], "background_lmax") and both[1].background_lmax >= 1


def test_the_background_turns_with_the_specimen(pack, seq):
    """The magnet's gradient is in the acquisition's frame and turns with it (#585): with the specimen at a pose in
    the bore, the expansion is the direct replay at that pose composed with every rotation."""
    played = seq.with_background_gradient(_g0(0.02))
    R_s = so3.rotation_of((0.2, -0.4, 0.89), roll=0.7)
    pr = pack.pose_response(played, method="closed", pose=R_s)
    for R in so3.haar_rotations(6, 8):
        direct = pack.replay(played, orientation=R_s @ R, complex_signal=True)
        assert np.all(np.abs(pr.at(R) - direct) <= pr.misfit + 2e-8)


def test_the_background_and_the_field_compose_in_closed_form(field_pack):
    """Field on: the field factor and the background factor are coupled walker by walker into one factor, then
    with the gradient -- three Wigner blocks, two couplings. Exact against the direct posed replay with the field
    to the field route's own precision (2e-6, as without a background)."""
    pk = field_pack
    seq = sequences.pgse([[1, 0, 0], [0, 0, 1], [0.6, 0.8, 0.0]], 6e-3, 15e-3, bvalues=[1.5e9] * 3, TE=30e-3)
    kw = dict(scanner=7.0, tissue=Tissue(chi_iso=-1e-7, chi_aniso=-1e-7))
    played, R_y = field_along(seq.with_background_gradient(_g0(0.005)), (0.0, 1.0, 0.0))
    pc = pk.pose_response(played, method="closed", pose=R_y, **kw)
    assert pc.route == "closed" and pc.field_lmax >= 2 and pc.background_lmax >= 2
    for R in so3.haar_rotations(12, 2):
        np.testing.assert_allclose(pc.at(R), pk.replay(played, orientation=R_y @ R, complex_signal=True, **kw), atol=2e-6)


@pytest.fixture(scope="module")
def long_pack(tmp_path_factory):
    g = d.PackedCylinders([3e-6], [[0.0, 0.0]], 10e-6)
    walk = d.simulate_trajectories(1000, 2e-9, g, 60e-3, 5e-4, seed=3, require_gpu=False)
    p = tmp_path_factory.mktemp("pk") / "swoop.rpk"
    build_replay_pack(walk, id="t/swoop", license="x", citation="x", K=24, out_path=str(p))
    return read_rpk(str(p))


@pytest.mark.parametrize("kind", ["spin echo", "stimulated echo"])
def test_the_swoop_as_delivered_at_a_head_position_takes_the_closed_form(long_pack, kind):
    """The Hyperfine Swoop as it plays a voxel 8 cm off isocentre: the nonlinearity, the magnet's own gradient
    (1 mT/m there), and the Maxwell term at 64 mT through the slew-limited ramps -- every catalogued term, composed
    by bore.encoding_classes as the brain page composes a class. The encoding left once g0 is taken out is one
    direction up to the Maxwell residual (#561), g0 is the second plane wave, and the expansion equals the direct
    posed replay of the delivered waveform at every rotation within its misfit."""
    from dmipy_sim.acquisition.scanners import ScannerLimits
    from dmipy_sim.phantom.bore import encoding_classes
    from dmipy_sim.phantom.grid import Grid
    sw = ScannerLimits.of("swoop")
    dirs = [[1, 0, 0], [0, 0, 1], [0.6, 0.8, 0.0], [0.0, 0.6, 0.8]]
    if kind == "spin echo":
        seq = sequences.pgse(dirs, 12e-3, 26e-3, gradient_strengths=[0.0, 0.02, 0.02, 0.02], TE=56e-3,
                             slew_rate=float(sw.slew_max))
    else:
        seq = sequences.pgste(dirs, 10e-3, 16e-3, gradient_strengths=[0.0, 0.02, 0.02, 0.02], TE=50e-3,
                              slew_rate=float(sw.slew_max))
    grid = Grid(shape=(1, 1, 1), voxel_size_m=(2e-3,) * 3, origin_m=(0.08, 0.0, 0.0), isocenter_m=(0.0, 0.0, 0.0))
    _cls, played = encoding_classes(sw, grid, seq, np.array([[0, 0, 0]]), tolerance=None)
    p = played[0]
    assert np.linalg.norm(p.background_gradient) > 0.9e-3 and p.concomitant is not None
    pr = long_pack.pose_response(p, method="closed")
    assert pr.route == "closed" and pr.background_lmax >= 1
    assert pr.misfit.max() < 0.1 / np.sqrt(long_pack.n_walkers)
    for R in so3.haar_rotations(12, 11):
        err = np.abs(pr.at(R) - long_pack.replay(p, orientation=R, complex_signal=True))
        assert np.all(err <= pr.misfit + 2e-8), (err, pr.misfit)


def test_an_odf_composition_forms_only_the_pairs_its_band_couples(pack, field_pack):
    """keep=(L, 0) with a background: on the host only the products of azimuthal orders |N| = |n| are formed
    (pose_device.paired_bodies), and the result is the n = 0 column of the full expansion, field off and on."""
    played = sequences.pgse([[1, 0, 0], [0, 0, 1], [0.6, 0.8, 0.0]], 2e-3, 5e-3, gradient_strengths=[0.3] * 3,
                            TE=10e-3).with_background_gradient(_g0(0.05))
    full = pack.pose_response(played, method="closed")
    odf = pack.pose_response(played, method="closed", keep=(8, 0))
    np.testing.assert_allclose(odf.coeffs, full.retained(8, 0), atol=1e-12)
    seq = sequences.pgse([[1, 0, 0], [0.6, 0.8, 0.0]], 6e-3, 15e-3, bvalues=[1.5e9] * 2, TE=30e-3)
    kw = dict(scanner=7.0, tissue=Tissue(chi_iso=-1e-7, chi_aniso=-1e-7))
    played, R_y = field_along(seq.with_background_gradient(_g0(0.005)), (0.0, 1.0, 0.0))
    full = field_pack.pose_response(played, method="closed", pose=R_y, **kw)
    odf = field_pack.pose_response(played, method="closed", pose=R_y, keep=(8, 0), **kw)
    np.testing.assert_allclose(odf.coeffs, full.retained(8, 0), atol=1e-12)


def test_a_shell_of_delivered_amplitudes_is_one_body_in_powers_of_its_amplitude(pack, monkeypatch):
    """A machine plays every row of a shell at its own amplitude (the nonlinearity), so its rows are as many groups
    -- but one shape. Their bodies are one polynomial in the amplitude (the Bessel series), the walkers contracted
    once per power: the same numbers as contracting every row on its own (to the float32 rounding of the rows'
    waveforms about their one shape), and the direct posed replay to the misfit."""
    import dmipy_sim.replay.replay as rr
    rng = np.random.default_rng(7)
    dirs = rng.normal(size=(24, 3)); dirs /= np.linalg.norm(dirs, axis=1, keepdims=True)
    tilt = np.eye(3) + 0.05 * rng.normal(size=(24, 3, 3))                      # every row its own delivered amplitude
    played = (sequences.pgse(dirs, 2e-3, 5e-3, gradient_strengths=[0.3] * 24, TE=10e-3)
              .with_gradient_nonlinearity(tilt).with_background_gradient(_g0(0.02)))
    import dmipy_sim.replay.pose_device as pd

    def per_row(*a, **k):
        raise AssertionError("a row of the shell was contracted on its own")

    for keep in (None, (6, 0)):
        with monkeypatch.context() as mp:
            mp.setattr(pd, "field_bodies", per_row); mp.setattr(pd, "paired_bodies", per_row)
            series = pack.pose_response(played, method="closed", keep=keep)
        assert series.n_bodies == 24
        with monkeypatch.context() as mp:
            mp.setattr(rr, "SHELL_SERIES_MAX_PHASE", 0.0)                       # every row contracted on its own
            each = pack.pose_response(played, method="closed", keep=keep)
        # equal to the float32 rounding of each row's delivered waveform about the shell's one shape
        np.testing.assert_allclose(series.coeffs, each.coeffs, atol=2e-9)
    full = pack.pose_response(played, method="closed")
    for R in so3.haar_rotations(4, 13):
        assert np.all(np.abs(full.at(R) - pack.replay(played, orientation=R, complex_signal=True)) <= full.misfit + 2e-8)


@pytest.mark.parametrize("machine", [None, "siemens_magnetom_prisma_3T", "hyperfine_swoop_64mT"])
def test_the_misfit_bounds_what_the_grouping_and_the_rounding_leave_out(pack, machine):
    """Every row's part the expansion does not play -- off its principal direction however small, its difference from
    the group whose body it shares, a b = 0 row's rounding about zero, its |g0|'s difference from its group's -- is in
    its misfit (dmrai-lab/dmipy-sim#617): the direct posed replay is within it at every rotation, on a 3 T machine's
    classes (where the misfit was 727 times short) as on the ideal scanner and the Swoop."""
    from dmipy_sim.acquisition.scanners import ScannerLimits
    from dmipy_sim.phantom.bore import encoding_classes
    from dmipy_sim.phantom.grid import Grid
    rng = np.random.default_rng(3)
    dirs = rng.normal(size=(12, 3)); dirs /= np.linalg.norm(dirs, axis=1, keepdims=True)
    dirs = np.vstack([[0.0, 0.0, 1.0], dirs, dirs])
    seq = sequences.pgse(dirs, 2e-3, 5e-3, gradient_strengths=[0.0] + [0.08] * 12 + [0.048] * 12, TE=10e-3, slew_rate=400.0)
    if machine is None:
        played = [seq]
    else:
        grid = Grid(shape=(3, 1, 1), voxel_size_m=(1e-2,) * 3, origin_m=(-0.01, 0.005, -0.02), isocenter_m=(0.0, 0.0, 0.0))
        played = encoding_classes(ScannerLimits.of(machine), grid, seq, np.array([[i, 0, 0] for i in range(3)]),
                                  tolerance=None)[1]
    for one in played:
        pr = pack.pose_response(one, method="closed")
        for R in so3.haar_rotations(6, 5):
            err = np.abs(pr.at(R) - pack.replay(one, orientation=R, complex_signal=True))
            assert np.all(err <= pr.misfit + 1e-12), (err - pr.misfit).max()


def test_residual_bound_projects_every_mode_in_one_pass(pack, monkeypatch):
    """dmrai-lab/dmipy-sim#698: ``_residual_bounds`` (#617) bounds every row that departs from its group's played
    waveform, a departure a float32-stored ``G`` reconstructed through two independent float64 SVDs never lands on
    exactly zero for even an ideal scanner's rows -- so on a many-direction single-shell batch nearly every row
    gets a residual, and its numerical rank ``k`` runs to the hundreds (unstructured rounding has no preferred
    direction). The numpy branch projected each of those ``k`` modes through its own call into
    :func:`dmipy_sim.replay.replay._compile_effective` (``bridge_projection``'s own DST) and its own matmul against
    every walker, a cost that grows with ``k`` -- where the torch branch beside it already batches every mode into
    ONE projection. Pinning the number of ``_compile_effective`` calls pins that cost: it must stay the few calls
    the per-axis moment contraction and ONE residual batch take, never one per mode."""
    from dmipy_sim.replay import replay as rr
    rng = np.random.default_rng(11)
    dirs = rng.normal(size=(24, 3)); dirs /= np.linalg.norm(dirs, axis=1, keepdims=True)
    seq = sequences.pgse(dirs, 2e-3, 5e-3, bvalues=[1e9] * 24, TE=10e-3, slew_rate=np.inf)
    calls = []
    orig = rr._compile_effective
    def counted(*a, **k):
        calls.append(1)
        return orig(*a, **k)
    monkeypatch.setattr(rr, "_compile_effective", counted)
    pack.pose_response(seq, method="closed")
    # per window: 3 calls for the main moment contraction (one per axis) and ONE for the residual bound's batch
    # (every mode of every residual row, in one call, #698) -- never the dozens of directions, let alone the
    # residual's own numerical rank (in the hundreds here), which is what #698 cost
    assert len(calls) <= 4 * pack.n_segments
