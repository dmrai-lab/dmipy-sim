"""The concomitant field's VALUE at a voxel, the order-0 term of dmipy-sim#394: one phase per measurement and
readout, ``gamma int s(t) B_c(x_v, t) dt``, recorded by ``with_concomitant`` and applied to every voxel's signal
by the phantom routes. A spin echo with identical lobes cancels it exactly; a gradient echo keeps it."""
import numpy as np
import pytest

import dmipy_sim as d
from dmipy_sim import sequences
from dmipy_sim.acquisition.scanners import ScannerLimits
from dmipy_sim.constants import GAMMA
from dmipy_sim.phantom import Grid, Inert, ODF, PackSubstrate, Phantom
from dmipy_sim.replay.bank import build_replay_pack

D0, B0 = 2.0e-9, 0.064
R_M = (0.05, 0.03, 0.02)                                   # 6 cm from isocentre, inside the Swoop's anchor


def _bernstein_first_order(G, r, B0):
    """``B_c`` to first order in ``|B_perp| / B0`` (Bernstein 1998), the textbook form, per sample."""
    x, y, z = r
    Gx, Gy, Gz = G[..., 0], G[..., 1], G[..., 2]
    return (Gx ** 2 + Gy ** 2) * z ** 2 / (2 * B0) + Gz ** 2 * (x ** 2 + y ** 2) / (8 * B0) - (Gx * Gz * x * z + Gy * Gz * y * z) / (2 * B0)


def test_a_gradient_echo_keeps_the_order_0_phase_and_a_spin_echo_cancels_it():
    gre = sequences.gre(30e-3, gradient_directions=[[1.0, 0.0, 0.0], [0.0, 0.0, 1.0]], bvalues=[1.0e9] * 2,
                        delta=8e-3, Delta=9e-3, n_t=301)
    played = gre.with_concomitant(R_M, B0)
    ph = played.concomitant_phase_rad
    assert ph.shape == (2, 1)
    Bc = _bernstein_first_order(np.asarray(gre.G, np.float64), R_M, B0)          # (n_meas, n_t)
    ref = GAMMA * gre.dt * Bc.sum(axis=1)
    assert np.all(np.abs(ref) > 1.0)                                                 # radians, not a rounding effect
    np.testing.assert_allclose(ph[:, 0], ref, rtol=0.05)                             # the exact form against first order
    se = sequences.pgse([[1.0, 0.0, 0.0], [0.0, 0.0, 1.0]], 8e-3, 20e-3, bvalues=[1.0e9] * 2, n_t=301)
    ph_se = se.with_concomitant(R_M, B0).concomitant_phase_rad
    lobe = GAMMA * se.dt * np.abs(_bernstein_first_order(np.asarray(se.G, np.float64), R_M, B0)).sum(axis=1) / 2
    assert np.all(lobe > 1.0)                                                        # each lobe winds radians
    np.testing.assert_allclose(ph_se[:, 0], 0.0, atol=1e-9 * lobe.max())             # and the 180 cancels them
    assert np.all(gre.concomitant_phase_rad == 0.0)                                  # nothing applied, nothing recorded


@pytest.fixture(scope="module")
def free_phantom(tmp_path_factory):
    seq = sequences.gre(30e-3, gradient_directions=[[1.0, 0.0, 0.0]], bvalues=[1.0e9], delta=8e-3, Delta=9e-3, n_t=301)
    n_t, dt = int(seq.n_t), float(seq.dt)
    walk = d.simulate_trajectories(1000, D0, d.FreeDiffusion(), (n_t - 1) * dt, dt, seed=11, require_gpu=False)
    out = tmp_path_factory.mktemp("pk") / "free.rpk"
    build_replay_pack(walk, id="test/free", license="x", citation="x", K=16, out_path=str(out))
    sh, vox = (3, 3, 1), 0.03
    grid = Grid(shape=sh, voxel_size_m=(vox,) * 3, origin_m=tuple(-0.5 * (n - 1) * vox for n in sh), isocenter_m=(0.0, 0.0, 0.0))
    c = np.zeros(sh + (45,), np.float32); c[..., 0] = 1.0 / np.sqrt(4 * np.pi)
    wm = PackSubstrate(str(out), m0=1.0, name="free")
    ph = Phantom.compose(grid, fractions={wm: np.ones(sh, np.float32)}, orientation={wm: ODF(c, basis="mrtrix3")}, remainder=Inert(name="bg"))
    return ph, seq


def test_the_phantom_turns_each_voxel_by_its_own_concomitant_phase(free_phantom):
    ph, seq = free_phantom
    swoop = ScannerLimits.of("swoop")
    S = ph.replay(seq, scanner=swoop, off_resonance=0.0, complex_signal=True)       # the field law's offset held off
    from dmipy_sim.phantom.bore import concomitant_phase_map
    expect = concomitant_phase_map(swoop, ph.grid, ph._on_this_grid(seq))[:, 0, -1]  # every voxel, in grid order
    got = np.angle(S.reshape(-1, S.shape[-1])[:, 0])
    iso = 4                                                                          # the centre of the 3 x 3
    assert abs(expect[iso]) < 1e-12 and abs(got[iso]) < 1e-9                         # nothing at isocentre
    off = np.flatnonzero(np.abs(expect) > 0.05)
    assert off.size >= 4                                                             # the corners turn by tenths of a radian
    # modulo pi: the Swoop's background gradient is on through a gradient echo, so the voxel's own winding factor
    # (a product of sincs) carries a sign of its own at some voxels; the concomitant phase is what remains
    np.testing.assert_allclose(np.sin(got[off] - expect[off]), 0.0, atol=1e-6)
    assert np.all((np.abs(S) > 0.0) & (np.abs(S) <= 1.0))
