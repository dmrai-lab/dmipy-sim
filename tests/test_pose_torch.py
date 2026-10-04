"""The torch backend of the closed-form pose expansion (dmrai-lab/dmipy-sim#603) is the JAX/numpy route's expansion:
the same coefficients to float32 rounding, the same band, and the same misfit (the residual bound, the Bessel tails
and the background's tail), for the ideal scanner and each catalogued machine as it plays a voxel off isocentre
(the Swoop with its own gradient, the Prisma and the Terra), a spin echo and a stimulated echo, the field off and on.
"""
import numpy as np
import pytest

import dmipy_sim as d
from dmipy_sim import sequences
from dmipy_sim.acquisition.scanners import ScannerLimits
from dmipy_sim.phantom.bore import encoding_classes
from dmipy_sim.phantom.grid import Grid
from dmipy_sim.replay.bank import build_replay_pack
from dmipy_sim.spec.tissue import Tissue
from tests.replay_frames import field_along

torch = pytest.importorskip("torch")

MACHINES = (None, "hyperfine_swoop_64mT", "siemens_magnetom_prisma_3T", "siemens_magnetom_terra_7T")
TISSUE = Tissue(chi_iso=-1e-7, chi_aniso=-5e-8)
DEVICES = ["cpu", pytest.param("cuda", marks=[pytest.mark.gpu, pytest.mark.skipif(not torch.cuda.is_available(),
                                                                                    reason="no CUDA device")])]


@pytest.fixture(scope="module")
def walk_pack():
    """A 400-walker cylinder walk of 12 ms at K = 12: the field-off pack."""
    walk = d.simulate_trajectories(400, 2e-9, d.PackedCylinders([2e-6], [[0.0, 0.0]], 8e-6), 12e-3, 5e-4, seed=4,
                                   require_gpu=False)
    return build_replay_pack(walk, id="test/torch", license="x", citation="x", K=12)


def _shell(n, rng):
    dirs = rng.normal(size=(n, 3))
    return dirs / np.linalg.norm(dirs, axis=1, keepdims=True)


def _played(machine, kind, T, g):
    """A b = 0 row and a 24-direction shell of amplitude ``g`` (T/m) within ``T`` (s), as ``machine`` plays it at a
    voxel 6 cm off isocentre (every catalogued term: nonlinearity, the magnet's own gradient, the Maxwell term),
    or as commanded on the ideal scanner. Each row of a machine's shell is then its own delivered amplitude."""
    dirs = np.vstack([[0.0, 0.0, 1.0], _shell(24, np.random.default_rng(3))])
    strengths = [0.0] + [g] * 24
    if kind == "spin echo":
        seq = sequences.pgse(dirs, 0.2 * T, 0.5 * T, gradient_strengths=strengths, TE=T, slew_rate=np.inf)
    else:
        seq = sequences.pgste(dirs, 0.15 * T, 0.6 * T, gradient_strengths=strengths, TE=T, slew_rate=np.inf)
    if machine is None:
        return seq
    grid = Grid(shape=(1, 1, 1), voxel_size_m=(2e-3,) * 3, origin_m=(0.05, -0.02, 0.025), isocenter_m=(0.0, 0.0, 0.0))
    _cls, played = encoding_classes(ScannerLimits.of(machine), grid, seq, np.array([[0, 0, 0]]), tolerance=None)
    return played[0]


def _same(ref, got, atol):
    assert got.route == ref.route == "closed"
    assert got.coeffs.shape == ref.coeffs.shape and got.lmax == ref.lmax and got.nmax == ref.nmax
    assert getattr(got, "field_lmax", None) == getattr(ref, "field_lmax", None)
    assert getattr(got, "background_lmax", None) == getattr(ref, "background_lmax", None)
    np.testing.assert_allclose(got.coeffs, ref.coeffs, rtol=0, atol=atol)
    # the misfit is the same bound: the residual bound and the shells' departures in float64 on both, the Bessel
    # tails from float32 values summed in float64
    np.testing.assert_allclose(got.misfit, ref.misfit, rtol=1e-5, atol=1e-13)


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("kind", ["spin echo", "stimulated echo"])
@pytest.mark.parametrize("machine", MACHINES)
def test_the_torch_expansion_is_the_numpy_one_with_the_field_off(walk_pack, machine, kind, device):
    """Field off: every catalogued machine and the ideal scanner, a spin echo and a stimulated echo, the whole
    expansion and an ODF composition's band; equal coefficients to float32 rounding and an equal misfit."""
    played = _played(machine, kind, 10e-3, 0.08)
    for keep in (None, (6, 0)):
        ref = walk_pack.pose_response(played, method="closed", keep=keep)
        got = walk_pack.pose_response(played, method="closed", keep=keep, backend="torch", device=device)
        _same(ref, got, atol=1e-5)
        assert (getattr(got, "background_lmax", 0) >= 1) == (machine == "hyperfine_swoop_64mT")


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("kind", ["spin echo", "stimulated echo"])
@pytest.mark.parametrize("machine", MACHINES)
def test_the_torch_expansion_is_the_numpy_one_with_the_field_on(field_pack, machine, kind, device):
    """Field on (the strand pack's path channel, at each machine's own field; 3 T on the ideal scanner, the field
    along an oblique direction of the specimen): the field factor, its coupling to the gradient and, on the Swoop,
    to the magnet's own gradient; equal coefficients to float32 rounding and an equal misfit."""
    B0 = 3.0 if machine is None else float(ScannerLimits.of(machine).field_T)
    played, R = field_along(_played(machine, kind, 8e-4, 0.3), (0.6, 0.0, 0.8))
    kw = dict(scanner=B0, tissue=TISSUE, pose=R, method="closed")
    for keep in (None, (6, 0)):
        ref = field_pack.pose_response(played, keep=keep, **kw)
        got = field_pack.pose_response(played, keep=keep, backend="torch", device=device, **kw)
        _same(ref, got, atol=1e-5)
        assert got.field_lmax >= 1


def test_a_batch_on_torch_is_each_acquisition_on_its_own(walk_pack):
    """A batch of classes is one pass over the walkers on torch as on the host: each member the numbers of its own
    call (the Swoop's classes at two positions, and the ideal scanner's, which takes its own pass)."""
    a = _played("hyperfine_swoop_64mT", "spin echo", 10e-3, 0.08)
    b = _played(None, "spin echo", 10e-3, 0.08)
    both = walk_pack.pose_responses([a, b], method="closed", backend="torch", device="cpu")
    for one, got in zip((a, b), both):
        alone = walk_pack.pose_response(one, method="closed", backend="torch", device="cpu")
        np.testing.assert_allclose(got.coeffs, alone.coeffs, rtol=0, atol=1e-12)
        np.testing.assert_array_equal(got.misfit, alone.misfit)


def test_the_backend_is_named(walk_pack):
    with pytest.raises(ValueError, match="backend"):
        walk_pack.pose_response(_played(None, "spin echo", 10e-3, 0.08), backend="cupy")
