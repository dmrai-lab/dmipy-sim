"""The machine's gradient side reaches a phantom's signal (dmipy-sim#377, closing #369).

A catalogued machine delivers a different gradient at every voxel: the nonlinearity tensor's scale and tilt,
the magnet's own background, the coils' Maxwell term. The pose route replays each pack once per distinct
delivered gradient, the voxels binned to a fraction of ``b``; the Bloch route propagates once per class. The
checks are absolute, not comparative: on free water the voxel's signal is ``exp(-b D)`` at the ``b`` the
acquisition actually delivers there, which no replay computes; at isocentre every term vanishes and the
signal is the ideal magnet's to the bit.
"""
import numpy as np
import pytest

import dmipy_sim as d
from dmipy_sim import sequences
from dmipy_sim.acquisition.prescription import Prescription
from dmipy_sim.acquisition.scanners import ScannerLimits
from dmipy_sim.phantom import Grid, Inert, ODF, PackSubstrate, Phantom
from dmipy_sim.phantom.bore import encoding_classes
from dmipy_sim.replay import read_rpk
from dmipy_sim.replay.bank import build_replay_pack

D0 = 2.0e-9
SH = (3, 3, 1)                      # odd, so one voxel sits exactly at isocentre
VOX = 0.025                         # 2.5 cm voxels: the grid spans 7.5 cm, inside the Swoop's 8 cm anchor


def _seq():
    return sequences.pgse([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [1.0, 1.0, 1.0]], 8e-3, 20e-3,
                          bvalues=[1.0e9] * 3, n_t=121)


@pytest.fixture(scope="module")
def pack_path(tmp_path_factory):
    seq = _seq()
    n_t, dt = int(seq.n_t), float(seq.dt)
    walk = d.simulate_trajectories(1000, D0, d.FreeDiffusion(), (n_t - 1) * dt, dt, seed=11, require_gpu=False)
    out = tmp_path_factory.mktemp("pk") / "free.rpk"
    build_replay_pack(walk, id="test/free", license="x", citation="x", K=16, out_path=str(out))
    return str(out)


def _grid():
    return Grid(shape=SH, voxel_size_m=(VOX,) * 3,
                origin_m=tuple(-0.5 * (n - 1) * VOX for n in SH), isocenter_m=(0.0, 0.0, 0.0))


@pytest.fixture(scope="module")
def phantom(pack_path):
    grid = _grid()
    c = np.zeros(SH + (45,), np.float32)
    c[..., 0] = 1.0 / np.sqrt(4 * np.pi)                                # isotropic: free water has no axis
    wm = PackSubstrate(pack_path, m0=1.0, name="free")
    ph = Phantom.compose(grid, fractions={wm: np.ones(SH, np.float32)},
                         orientation={wm: ODF(c, basis="mrtrix3")}, remainder=Inert(name="bg"))
    return ph, read_rpk(pack_path)


def _iso(ph):
    return tuple(n // 2 for n in SH)


@pytest.fixture(scope="module")
def exact(phantom):
    """The Swoop replayed with one class per voxel, beside the bare field: what the tests below read."""
    ph, pack = phantom
    seq = _seq()
    sw = ScannerLimits.of("swoop")
    rep = {}
    S_sw = ph.replay(seq, scanner=sw, packs={0: pack}, encoding_tolerance=None, report=rep)
    S_b0 = ph.replay(seq, scanner=0.064, packs={0: pack})
    cls, played = encoding_classes(sw, ph.grid, seq, ph.voxel_index, tolerance=None)
    return dict(S_sw=S_sw, S_b0=S_b0, cls=cls, played=played, n=rep["n_encoding_classes"])


def test_the_catalogued_swoop_moves_the_signal_and_a_bare_field_does_not(phantom, exact):
    """The defect this closes: a phantom on ``ScannerLimits.of("swoop")`` replayed bit-identically to one on a
    bare 64 mT field, while the reference route said the delivered b differed by percent."""
    ph, _pack = phantom
    assert exact["n"] == ph.n_voxels
    rel = np.nanmax(np.abs(exact["S_sw"] - exact["S_b0"]) / np.abs(exact["S_b0"]))
    assert rel > 3e-2, f"the machine moved the signal by only {rel:.2e}"
    # and the isocentre voxel is the ideal magnet's to the bit: every term vanishes there by construction
    np.testing.assert_array_equal(exact["S_sw"][_iso(ph)], exact["S_b0"][_iso(ph)])


def test_free_water_decays_at_the_b_the_machine_delivers_in_each_voxel(phantom, exact):
    """The absolute check. On free water the signal is ``exp(-b D)``, so the machine multiplies each voxel's
    signal by ``exp(-(b_played - b) D)`` at the ``b`` the played acquisition delivers THERE -- composed by the
    sequence transforms and never by a replay. The ratio to the bare field is read, since the pack's own
    Monte-Carlo realisation, the same walkers under nearly the same gradient, divides out of it."""
    ph, _pack = phantom
    seq = _seq()
    b = np.stack([exact["played"][c].b() for c in exact["cls"]])         # (n_voxels, n_meas)
    _vi, sw = ph.sparse(exact["S_sw"])
    _vi, b0 = ph.sparse(exact["S_b0"])
    want = np.exp(-(b - np.asarray(seq.b())[None, :]) * D0)
    assert np.abs(sw / b0 - want).max() < 1e-2, np.abs(sw / b0 - want).max()
    assert np.abs(want - 1.0).max() > 3e-2                               # and the effect is well above that


def test_binning_costs_what_the_tolerance_allows_and_no_more(phantom, exact):
    ph, pack = phantom
    seq = _seq()
    sw = ScannerLimits.of("swoop")
    counts = {}
    for tol in (1e-3, 0.2):
        rep = {}
        binned = ph.replay(seq, scanner=sw, packs={0: pack}, encoding_tolerance=tol, report=rep)
        counts[tol] = rep["n_encoding_classes"]
        rel = np.nanmax(np.abs(binned - exact["S_sw"]) / np.abs(exact["S_sw"]))
        assert rel < 4 * tol * 1e9 * D0, f"tolerance {tol} moved the signal by {rel:.2e}"     # b D = 2 here
    # the Maxwell term's position is binned to tolerance * B0 / G_max, three centimetres here at a twentieth
    # (110 mT/m in 64 mT), so only a tolerance wider than the grid collapses it: a fifth is one class
    assert counts[0.2] == 1 and counts[1e-3] == ph.n_voxels


def test_a_three_tesla_magnet_costs_a_handful_of_classes(phantom):
    """A shimmed superconducting magnet catalogues no field shape, so its Maxwell term is one class over a head
    (binned to ``tolerance B0 / G_max``, a quarter of a metre at a hundredth). What a 3 T machine does catalogue
    is a CLASS MODEL of its coils' nonlinearity, inferred and said so, which moves the delivered b by
    ``a_t rho^2`` across the transverse plane, half a percent at this grid's corners: the classes then cost what
    that model costs and no more, and a machine that catalogues neither is one class within a percent of the
    bare field."""
    ph, pack = phantom
    seq = _seq()
    S0 = ph.replay(seq, scanner=3.0, packs={0: pack})
    rep = {}
    S = ph.replay(seq, scanner=ScannerLimits.of("prisma"), packs={0: pack}, encoding_tolerance=1e-2, report=rep)
    prisma = ScannerLimits.of("prisma")
    assert prisma.has_gradient_nonlinearity and not prisma.has_field_law
    r2 = 2 * VOX ** 2                                                     # the corner voxels, transverse to B0
    bound = 1.5 * abs(prisma.b_error_quadratic_transverse) * r2 * 1e9 * D0 + 1e-2       # b D0 = 2 here
    dev = np.nanmax(np.abs(S - S0) / np.abs(S0))
    assert 2e-3 < dev < bound, (dev, bound)                               # the model reaches the signal, and only it
    assert 1 <= rep["n_encoding_classes"] <= ph.n_voxels
    rep = {}
    premier = ScannerLimits.of("premier")
    assert not premier.has_gradient_nonlinearity and not premier.has_field_law
    Sg = ph.replay(seq, scanner=premier, packs={0: pack}, encoding_tolerance=1e-2, report=rep)
    assert rep["n_encoding_classes"] == 1
    assert np.nanmax(np.abs(Sg - S0) / np.abs(S0)) < 1e-2


def test_the_bloch_route_carries_the_same_classes(pack_path):
    """A frames-mode phantom on the RF-aware route propagates once per (substrate, class, pose, ...) and must
    agree with the pose route to the pack's floor."""
    grid = _grid()
    from dmipy_sim.phantom import Frames
    R = np.zeros(SH + (1, 3, 3)); R[..., :, :] = np.eye(3)
    wm = PackSubstrate(pack_path, m0=1.0, name="free")
    ph = Phantom.compose(grid, fractions={wm: np.ones(SH, np.float32)}, orientation=Frames(R), remainder=Inert(name="bg"))
    pack = read_rpk(pack_path)
    seq = _seq()
    sw = ScannerLimits.of("swoop")
    rep = {}
    _, S_b = ph.file.replay_bloch(seq, scanner=sw, packs={0: pack}, encoding_tolerance=None, report=rep)
    _, S_p = ph.file.replay(seq, scanner=sw, packs={0: pack}, encoding_tolerance=None)
    assert rep["n_encoding_classes"] == ph.n_voxels
    assert np.abs(np.abs(S_b) - np.abs(S_p)).max() < 3.0 / np.sqrt(pack.n_walkers)


def test_a_train_replay_says_what_it_does_not_carry(phantom):
    ph, pack = phantom
    train = sequences.splice([[1.0, 0.0, 0.0]], 3e-3, 7e-3, 2, 5e-3, bvalues=[3e8], TE_prep=16e-3,
                             beta_deg=150.0, n_t_per_echo=20)
    with pytest.raises(ValueError, match="every voxel"):
        ph.replay_train(train, scanner=ScannerLimits.of("swoop"), packs={0: pack})
    ph.replay_train(train, scanner=0.064, packs={0: pack})               # the field alone is carried


def test_an_unbalanced_encoding_takes_its_voxel_from_the_grid(phantom):
    """A sequence with no prescription is averaged over THIS phantom's voxel: the same answer as declaring the
    grid's voxel size on the sequence, and not a refusal."""
    ph, pack = phantom
    seq = _seq()
    G = np.asarray(seq.G, np.float64).copy()
    n = int(round(2e-3 / seq.dt))
    G[:, seq.echo_idx - n:seq.echo_idx, 2] += 0.02                      # a spoiler, played in G
    spoiled = seq.with_gradient(G)
    assert spoiled.unbalanced and spoiled.prescription is None
    S = ph.replay(spoiled, scanner=0.064, packs={0: pack})
    stated = spoiled.with_prescription(Prescription(voxel_size_m=(VOX,) * 3, matrix=SH))
    np.testing.assert_array_equal(S, ph.replay(stated, scanner=0.064, packs={0: pack}))
    tiny = spoiled.with_prescription(Prescription(voxel_size_m=(1e-9,) * 3, matrix=SH))
    assert np.nanmax(np.abs(S)) < 0.3 * np.nanmin(np.abs(ph.replay(tiny, scanner=0.064, packs={0: pack})))
