"""Factorised pose responses (dmrai-lab/dmipy-sim#610): one pass over the walkers per (shape, tissue, field), every
encoding class of a machine an evaluation of its Chebyshev series in (amplitude, |g0|) with the exact lab side.

Held here on CI-sized packs: each class's response equals its per-class closed form within the two misfits (the
coefficients' difference is a function on SO(3) bounded pointwise by their sum, so its L2 norm per measurement is
too), the series' misfit bounds the direct posed replay at every rotation, torch is numpy, the truncation bound of
the series is a bound, and a class outside the series is refused by name. The same comparisons under torch on a CUDA
device, on two committed fixture packs (``tools/build_pose_fixtures.py``), are the `gpu` test at the end."""
from pathlib import Path

from tests._envelope import stated
import numpy as np
import pytest
from scipy.special import spherical_jn

import dmipy_sim as d
from dmipy_sim.engine.core import _walk_geometry
from dmipy_replay import read_rpk, sequences
from dmipy_replay.acquisition.scanners import ScannerLimits
from dmipy_replay.acquisition.waveforms import rotate_waveform
from dmipy_replay.phantom.bore import encoding_classes
from dmipy_replay.phantom.grid import Grid
from dmipy_replay.replay import so3
from dmipy_sim.replay.bank import build_replay_pack
from dmipy_replay.replay.pose_series import _cheb_matrix, _cheb_points, _cheb_T, _series_bound
from dmipy_replay.spec.tissue import Tissue
from tests.replay_frames import pose_with_field_along
from dmipy_replay.phantom import Pose

MACHINES = (None, "hyperfine_swoop_64mT", "siemens_magnetom_prisma_3T", "siemens_magnetom_terra_7T")
TISSUE = Tissue(chi_iso=-1e-7, chi_aniso=-5e-8)


@pytest.fixture(scope="module")
def walk_pack():
    """A 400-walker cylinder walk of 12 ms at K = 12: the field-off pack."""
    walk = _walk_geometry(400, 2e-9, d.PackedCylinders([2e-6], [[0.0, 0.0]], 8e-6), 12e-3, 5e-4, seed=4,
                                   require_gpu=False)
    return build_replay_pack(walk, id="test/series", license="x", citation="x", K=12, **stated(of=walk))


def _classes(machine, kind, T, g, R=None):
    """The encoding classes of a b = 0 row and two 12-direction shells (``g`` and ``0.6 g`` T/m, ramps of a few percent
    of ``T``)
    over twelve voxels 1-3 cm off isocentre as ``machine`` plays them (every catalogued term, binned exactly), or the
    commanded acquisition on the ideal scanner; turned by ``R`` (the lab acquisition of a specimen at pose ``R``)."""
    rng = np.random.default_rng(3)
    dirs = rng.normal(size=(12, 3)); dirs /= np.linalg.norm(dirs, axis=1, keepdims=True)
    dirs = np.vstack([[0.0, 0.0, 1.0], dirs, dirs])
    strengths = [0.0] + [g] * 12 + [0.6 * g] * 12
    slew = g / (0.02 * T)                                                 # ramps of a tenth of the lobe or so
    if kind == "spin echo":
        seq = sequences.pgse(dirs, 0.2 * T, 0.5 * T, gradient_strengths=strengths, TE=T, slew_rate=slew)
    else:
        seq = sequences.pgste(dirs, 0.15 * T, 0.6 * T, gradient_strengths=strengths, TE=T, slew_rate=slew)
    if machine is None:
        played = [seq]
    else:
        grid = Grid(shape=(3, 2, 2), voxel_size_m=(1e-2,) * 3, origin_m=(-0.01, 0.005, -0.02), isocenter_m=(0.0, 0.0, 0.0))
        idx = np.array([[i, j, k] for i in range(3) for j in range(2) for k in range(2)])
        _cls, played = encoding_classes(ScannerLimits.of(machine), grid, seq, idx, tolerance=None)
    return played if R is None else [rotate_waveform(p, R) for p in played]


CLOSED_ROUNDING = 1e-12
"""The numpy closed form's misfit bounds its whole residual (#617/#619), so the comparison carries rounding only."""


def _within(series, closed, slack=CLOSED_ROUNDING):
    """Each measurement's coefficients within the two misfits, the closed form read at the series' band."""
    assert series.route == "series" and closed.route == "closed"
    ref = so3.rebanded(closed.coeffs, closed.lmax, closed.nmax, series.lmax, series.nmax) \
        if (closed.lmax, closed.nmax) != (series.lmax, series.nmax) else closed.coeffs
    diff = np.linalg.norm(series.coeffs - ref, axis=1)
    bad = diff > series.misfit + closed.misfit + slack
    assert not bad.any(), (np.flatnonzero(bad), diff[bad], series.misfit[bad], closed.misfit[bad])
    return float(np.max(diff / (series.misfit + closed.misfit + slack)))


@pytest.mark.parametrize("kind", ["spin echo", "stimulated echo"])
@pytest.mark.parametrize("machine", MACHINES)
def test_every_class_is_its_closed_form_within_the_misfit_with_the_field_off(walk_pack, machine, kind):
    """Field off: the ideal scanner and each catalogued machine over twelve voxels, a spin echo and a stimulated echo;
    the series built once over the classes, each class against its own closed form, the full band and an ODF's."""
    played = _classes(machine, kind, 10e-3, 0.08)
    for keep in (None, (6, 0)):
        series = walk_pack.pose_series(played, keep=keep)
        assert series.n_classes == len(played)
        assert (series.background_lmax >= 1) == (machine == "hyperfine_swoop_64mT")
        for got, one in zip(series.responses(), played):
            _within(got, walk_pack.pose_response(one, method="closed", keep=keep))
        assert series.bounds["departure"] < 0.1 * series.floor


@pytest.mark.parametrize("kind", ["spin echo", "stimulated echo"])
@pytest.mark.parametrize("machine", MACHINES)
def test_every_class_is_its_closed_form_within_the_misfit_with_the_field_on(field_pack, machine, kind):
    """Field on (the strand pack's path channel at each machine's field, 3 T on the ideal scanner, the field along an
    oblique direction of the specimen): the field factor coupled with the background's in the series as in the
    closed form."""
    B0 = 3.0 if machine is None else float(ScannerLimits.of(machine).field_T)
    R = pose_with_field_along((0.6, 0.0, 0.8))
    played = _classes(machine, kind, 8e-4, 0.3, R)
    kw = dict(scanner=B0, tissue=TISSUE, pose=Pose(R))
    for keep in (None, (6, 0)):
        series = field_pack.pose_series(played, keep=keep, **kw)
        assert series.field_lmax >= 1
        for got, one in zip(series.responses(), played):
            _within(got, field_pack.pose_response(one, method="closed", keep=keep, **kw))


@pytest.mark.parametrize("field", [False, True])
def test_the_misfit_bounds_the_direct_posed_replay(walk_pack, field_pack, field):
    """The misfit is a true bound: on the Swoop (its own gradient, its Maxwell term through slew-limited ramps), the
    series at a rotation is the pack replayed there within each measurement's misfit, field off and on."""
    machine = "hyperfine_swoop_64mT"
    if field:
        pack, R = field_pack, pose_with_field_along((0.6, 0.0, 0.8))
        played, kw = _classes(machine, "spin echo", 8e-4, 0.3, R), dict(scanner=0.064, tissue=TISSUE, pose=Pose(R))
    else:
        pack, played, kw = walk_pack, _classes(machine, "spin echo", 10e-3, 0.08), {}
    series = pack.pose_series(played, **kw)
    got = series.responses()
    assert all(r.misfit.max() > 0 for r in got)
    for k in (0, len(played) - 1):
        for Q in so3.haar_rotations(4, seed=k):
            # the lab acquisition replayed with the specimen at its pose and the substrate at Q within it
            direct = pack.replay(played[k], orientation=(R @ Q) if field else Q, complex_signal=True, tol=None,
                                 **({} if not field else dict(scanner=0.064, tissue=TISSUE)))
            err = np.abs(got[k].at(Q) - direct)
            assert np.all(err <= got[k].misfit + 2e-8), (err, got[k].misfit)


def test_torch_is_numpy(walk_pack, field_pack):
    """The torch backend builds the same series: coefficients to float32 rounding, the same misfit."""
    pytest.importorskip("torch")
    for pack, played, kw in ((walk_pack, _classes("hyperfine_swoop_64mT", "spin echo", 10e-3, 0.08), {}),
                             (field_pack, _classes("hyperfine_swoop_64mT", "stimulated echo", 8e-4, 0.3),
                              dict(scanner=0.064, tissue=TISSUE))):
        for keep in (None, (6, 0)):
            a = pack.pose_series(played, keep=keep, **kw).responses()
            b = pack.pose_series(played, keep=keep, backend="torch", device="cpu", **kw).responses()
            for x, y in zip(a, b):
                np.testing.assert_allclose(y.coeffs, x.coeffs, rtol=0, atol=1e-5)
                np.testing.assert_allclose(y.misfit, x.misfit, rtol=1e-5, atol=1e-13)


def test_a_class_read_later_is_the_class_built_on(walk_pack):
    """``responses(waveforms)`` locates acquisitions read after the build: a class of the build set read again is the
    same numbers as its ``select``."""
    played = _classes("hyperfine_swoop_64mT", "spin echo", 10e-3, 0.08)
    series = walk_pack.pose_series(played)
    again = series.responses(played[3:5])
    for x, y in zip(again, series.responses(select=[3, 4])):
        np.testing.assert_array_equal(x.coeffs, y.coeffs)
        np.testing.assert_array_equal(x.misfit, y.misfit)


def test_what_the_series_does_not_hold_is_refused(walk_pack):
    """An amplitude outside the series' intervals, a magnet's gradient above its largest, another timing class, a row
    departing from its shape beyond the direction tolerance: each refused by name, never extrapolated."""
    played = _classes("hyperfine_swoop_64mT", "spin echo", 10e-3, 0.08)
    series = walk_pack.pose_series(played[:4])
    louder = _classes("hyperfine_swoop_64mT", "spin echo", 10e-3, 0.09)[0]
    with pytest.raises(ValueError, match="outside every interval"):
        series.responses([louder])
    with pytest.raises(ValueError, match="timing class"):
        series.responses([_classes("hyperfine_swoop_64mT", "spin echo", 12e-3, 0.08)[0]])
    with pytest.raises(ValueError, match="departs from the series' shape"):     # a row no shape holds to the tolerance
        walk_pack.pose_series(played, direction_tol=1e-9)
    seq = _classes(None, "spin echo", 10e-3, 0.08)[0]
    g0 = np.array([3e-4, -2e-4, 5e-4])
    series = walk_pack.pose_series([seq.with_background_gradient(g0)])
    with pytest.raises(ValueError, match="exceeds the series' largest"):
        series.responses([seq.with_background_gradient(2.0 * g0)])


def test_the_series_bound_bounds_the_chebyshev_interpolant():
    """The derivation's bound, held to the numbers: the interpolant of ``j_l(s x)`` in ``N + 1`` Chebyshev points of
    an amplitude interval misses ``j_l`` by no more than the walker's ``I_w`` anywhere on it, for every order of the
    band and phases from a fraction of a radian to tens of them."""
    rng = np.random.default_rng(0)
    for centre, half, x_max in ((1.0, 0.03, 30.0), (1.0, 0.2, 8.0), (0.5, 0.5, 0.3)):
        x = rng.uniform(0.0, x_max, 64)
        L = int(np.ceil(x_max * (centre + half))) + 4
        N, I, I_max = _series_bound(x, np.full(x.size, 1.0 / x.size), centre, half, L, 1e-9)
        assert 1 <= N and I <= 1e-9 and I_max > 0
        nodes = centre + half * _cheb_points(N)
        coef = _cheb_matrix(N) @ np.stack([spherical_jn(l, np.outer(nodes, x)) for l in range(L + 1)], -1).reshape(N + 1, -1)
        s = np.linspace(centre - half, centre + half, 257)
        approx = (_cheb_T(N, (s - centre) / half) @ coef).reshape(s.size, x.size, L + 1)
        exact = np.stack([spherical_jn(l, np.outer(s, x)) for l in range(L + 1)], -1)
        per_walker = ((2 * np.arange(L + 1) + 1) * np.abs(approx - exact)).sum(-1).max(0)    # sum_l (2l+1) |err_l|
        assert per_walker.mean() <= I * (1 + 1e-6) + 1e-15 and per_walker.max() <= I_max * (1 + 1e-6) + 1e-15


# ---- the torch route on a CUDA device, on the committed fixture packs -----------------------------------------------

FIXTURES = Path(__file__).parent / "data"
"""The two packs ``tools/build_pose_fixtures.py`` writes: ``cylinder_k12.rpk`` (400 walkers of a cylinder, K = 12, no field
channel) and ``strands_field_k4.rpk`` (90 walkers among three sheathed strands, K = 4, the susceptibility path channel)."""

FLOAT32 = 1e-5
"""What the torch route's float32 products add to a measurement beyond the misfits that bound it, in the L2 norm of its
coefficients or in a replayed signal: the atol of the coefficient comparisons of ``tests/test_pose_torch.py``."""


def _same_closed_form(got, ref, slack):
    """``got``, the closed form on another backend, within the two misfits and ``slack`` of ``ref`` in each measurement's
    coefficients; the largest share of that bound used."""
    assert got.route == ref.route == "closed" and got.coeffs.shape == ref.coeffs.shape
    diff = np.linalg.norm(got.coeffs - ref.coeffs, axis=1)
    bound = got.misfit + ref.misfit + slack
    assert (diff <= bound).all(), (np.flatnonzero(diff > bound), diff[diff > bound], got.misfit, ref.misfit)
    return float(np.max(diff / bound))


def _torch_route_vs_closed_form(pack, played, k, *, device, keep, **kw):
    """The class ``played[k]`` at replay setting ``kw`` (``scanner=``, ``tissue=``, ``pose=``) read at band ``keep``, under
    torch on ``device``: its closed form and its series (built over all of ``played``) each within the two misfits and
    ``FLOAT32`` of the numpy closed form, and the series' misfit below a tenth of the pack's floor. Returns the largest
    share of each bound used."""
    ref = pack.pose_responses([played[k]], method="closed", keep=keep, **kw)[0]                  # numpy: the float64 route
    closed = pack.pose_responses([played[k]], method="closed", keep=keep, backend="torch", device=device, **kw)[0]
    series = pack.pose_series(played, keep=keep, backend="torch", device=device, **kw)
    assert max(r.misfit.max() for r in series.responses()) < 0.1 * series.floor
    return dict(closed=_same_closed_form(closed, ref, FLOAT32),
                series=_within(series.responses(select=[k])[0], ref, slack=FLOAT32))


def _series_vs_direct_replay(pack, played, k, *, device, R=None, **kw):
    """The whole-band series of the class ``played[k]``, under torch on ``device``, at three rotations ``Q`` within its misfit
    and ``FLOAT32`` of the pack replayed there (the specimen at ``R @ Q`` where ``R`` poses it). Returns the largest share
    of that bound used."""
    got = pack.pose_series(played, backend="torch", device=device, **kw).responses(select=[k])[0]
    replay_kw = {key: v for key, v in kw.items() if key != "pose"}
    used = 0.0
    for Q in so3.haar_rotations(3, seed=1):
        direct = pack.replay(played[k], orientation=Q if R is None else R @ Q, complex_signal=True, tol=None, **replay_kw)
        err = np.abs(got.at(Q) - direct)
        assert (err <= got.misfit + FLOAT32).all(), (err, got.misfit)
        used = max(used, float(np.max(err / (got.misfit + FLOAT32))))
    return used


@pytest.mark.gpu
@pytest.mark.parametrize("kind", ["spin echo", "stimulated echo"])
@pytest.mark.parametrize("field", [False, True], ids=["field off", "field on"])
def test_the_torch_route_on_a_cuda_device_is_the_closed_form_within_the_misfits(field, kind):
    """The torch route on a CUDA device, on the committed fixture packs, one class per kind: the Prisma's class at the last
    of twelve voxels, a spin echo and a stimulated echo, the field off (the cylinder pack) and on (the strand pack at the
    Prisma's 3 T, the field along an oblique direction of the specimen). The closed form and the series are each within the
    two misfits of the numpy closed form, at the full band and an ODF's; the series' misfit is below a tenth of the pack's
    floor; the series at three rotations is within its misfit of the direct posed replay. This is the comparison the CPU tests
    make with torch on the host, made on the device that runs the production route."""
    torch = pytest.importorskip("torch")
    if not torch.cuda.is_available():
        pytest.skip("no CUDA device")
    machine = "siemens_magnetom_prisma_3T"
    if field:
        pack = read_rpk(FIXTURES / "strands_field_k4.rpk")
        R = pose_with_field_along((0.6, 0.0, 0.8))
        played = _classes(machine, kind, 8e-4, 0.3, R)
        kw = dict(scanner=float(ScannerLimits.of(machine).field_T), tissue=TISSUE, pose=Pose(R))
    else:
        pack, R, kw = read_rpk(FIXTURES / "cylinder_k12.rpk"), None, {}
        played = _classes(machine, kind, 10e-3, 0.08)
    k = len(played) - 1
    for keep in (None, (6, 0)):
        _torch_route_vs_closed_form(pack, played, k, device="cuda", keep=keep, **kw)
    _series_vs_direct_replay(pack, played, k, device="cuda", R=R, **kw)
