"""Where replay ends: position enters a walker's phase to first order, so the voxel is a factor (dmipy-sim#375).

A walker's phase under an encoding is ``r_v . k + gamma int G . u dt``: its place in the voxel times the net
moment the encoding leaves at the readout, plus its own micron-scale excursion. The second is what a pack
stores. The first is what a spoiler winds across, and a micron-scale substrate cannot wind it geometrically --
so the voxel's signal is the substrate's times the voxel's average of ``exp(i k . r_v)``, which for a box is a
product of sincs. Exactly 1 for a refocused encoding; for an unbalanced one it is the spoiler, derived from the
waveform and the prescription. Without a prescription it is refused, since the un-crushed substrate signal is a
different number, not an approximation.
"""
import numpy as np
import pytest

import dmipy_sim as d
from dmipy_sim import sequences
from dmipy_sim.acquisition.prescription import Prescription
from dmipy_sim.constants import GAMMA
from dmipy_sim.replay import read_rpk
from dmipy_sim.replay.bank import build_replay_pack


def _pgse():
    return sequences.pgse([[1.0, 0.0, 0.0], [0.0, 1.0, 1.0]], 0.008, 0.024, bvalues=[1.0e9, 1.0e9], n_t=121)


def _spoiled(seq, G_T_m=0.02, tau_s=2e-3, axis=2):
    """The same acquisition with a spoiler lobe played in G after the second encoding lobe: ``G_T_m`` for
    ``tau_s`` along ``axis``, in every measurement. It winds ``gamma G tau`` per metre at the readout."""
    G = np.asarray(seq.G, np.float64).copy()
    dt = float(seq.dt)
    n = int(round(tau_s / dt))
    stop = int(seq.echo_idx)
    G[:, stop - n:stop, axis] += G_T_m
    return seq.with_gradient(G)


def _prescribed(seq, L=(2e-3, 2e-3, 2e-3)):
    return seq.with_prescription(Prescription(voxel_size_m=L, matrix=(8, 8, 8)))


@pytest.fixture(scope="module")
def pack(tmp_path_factory):
    seq = _pgse()
    n_t, dt = int(seq.n_t), float(seq.dt)
    walk = d.simulate_trajectories(3000, 2.0e-9, d.FreeDiffusion(), (n_t - 1) * dt, dt, seed=5, require_gpu=False)
    out = tmp_path_factory.mktemp("pk") / "free.rpk"
    build_replay_pack(walk, id="test/free", license="x", citation="x", K=16, out_path=str(out))
    return read_rpk(str(out))


def test_the_net_moment_is_the_winding_the_effective_gradient_leaves_at_the_readout():
    seq = _pgse()
    np.testing.assert_allclose(seq.net_moment, 0.0, atol=1e-3 * GAMMA * np.abs(seq.G).max() * seq.dt)
    assert not seq.unbalanced
    sp = _spoiled(seq)
    assert sp.unbalanced
    k = sp.net_moment
    n = int(round(2e-3 / sp.dt))
    want = -GAMMA * 0.02 * n * sp.dt                             # gamma G tau along z; AFTER the 180, so negative
    assert k.shape == (2, 3)
    np.testing.assert_allclose(k[:, 2], want, rtol=1e-6)
    np.testing.assert_allclose(k[:, :2], 0.0, atol=1e-6 * abs(want))


def test_the_voxel_factor_is_the_box_average_of_the_winding_and_not_a_lattices():
    """``prod_i sinc(k_i L_i / 2)`` against the average of ``exp(i k . r)`` over the box, taken on a fine grid
    with no sinc in it; and the trap the continuous average avoids: a LATTICE of identical cells at spacing
    ``a`` returns to full signal at ``k a = 2 pi``, where the voxel is fully crushed."""
    sp = _prescribed(_spoiled(_pgse()))
    F = sp.voxel_factor()
    k = sp.net_moment
    L = np.asarray(sp.prescription.voxel_size_m)
    g = (np.arange(200) + 0.5) / 200 - 0.5
    X, Y, Z = np.meshgrid(g * L[0], g * L[1], g * L[2], indexing="ij")
    r = np.stack([X, Y, Z], -1).reshape(-1, 3)
    grid = np.array([np.mean(np.exp(1j * r @ k[m])) for m in range(k.shape[0])])
    np.testing.assert_allclose(F, grid.real, atol=2e-4)
    np.testing.assert_allclose(grid.imag, 0.0, atol=2e-4)       # a centred box: real, and it may be negative
    assert np.abs(F).max() < 0.2, f"a spoiler of {np.abs(k).max() * L.max() / (2 * np.pi):.1f} turns barely crushed"
    # the lattice trap
    a = 50e-6
    k_alias = 2.0 * np.pi / a
    cells = (np.arange(int(round(L[2] / a))) - (L[2] / a - 1) / 2) * a
    assert abs(np.mean(np.exp(1j * k_alias * cells))) > 0.999                    # a lattice: no crushing at all
    assert abs(np.sinc(k_alias * L[2] / (2 * np.pi))) < 1e-12                    # the voxel: fully crushed


def test_an_unbalanced_encoding_is_refused_without_a_voxel_size_on_every_route(pack):
    sp = _spoiled(_pgse())
    for call in (lambda: pack.replay(sp), lambda: pack.walker_signals(sp), lambda: pack.walker_phases(sp),
                 lambda: pack.pose_response(sp), lambda: pack.replay_bloch(sp)):
        with pytest.raises(ValueError, match="net moment"):
            call()
    with pytest.raises(ValueError, match="net moment"):
        sp.voxel_factor()


def test_a_balanced_encoding_is_untouched_by_a_prescription(pack):
    seq = _pgse()
    np.testing.assert_array_equal(seq.voxel_factor(), 1.0)
    a = pack.replay(seq, complex_signal=True)
    b = pack.replay(_prescribed(seq), complex_signal=True)
    np.testing.assert_array_equal(a, b)


def test_the_voxel_signal_is_the_substrate_signal_times_the_factor(pack):
    """The identity the boundary rests on, through the pack. A voxel of a nanometre is smaller than any cell
    and its factor is 1 to double precision, so it is the substrate's own signal, legitimately; the millimetre
    voxel must be that times its factor, exactly."""
    sp = _spoiled(_pgse())
    tiny = _prescribed(sp, L=(1e-9, 1e-9, 1e-9))
    np.testing.assert_allclose(tiny.voxel_factor(), 1.0, rtol=1e-10)
    S_sub = pack.replay(tiny, complex_signal=True)
    mm = _prescribed(sp)
    S_vox = pack.replay(mm, complex_signal=True)
    np.testing.assert_allclose(S_vox, S_sub * mm.voxel_factor(), rtol=1e-9)
    S_enc = np.abs(pack.replay(_pgse()))                          # the encoding alone: exp(-bD) and its noise
    assert np.abs(S_sub).min() > 0.8 * S_enc.min(), "the substrate alone barely dephases under this spoiler"
    assert np.abs(S_vox).max() < 0.2 * S_enc.max(), "the voxel is crushed"
    # walker_signals carries the factor in E, so its ensemble sum is the voxel's signal
    w, ew, E = pack.walker_signals(mm)
    np.testing.assert_allclose((ew[:, None] * E).sum(0) / w.sum(), S_vox, rtol=1e-12)


def test_the_pose_expansion_carries_the_factor_and_the_cache_knows_the_voxel(pack, tmp_path):
    sp = _spoiled(_pgse())
    tiny, mm = _prescribed(sp, L=(1e-9,) * 3), _prescribed(sp)
    r_sub = pack.pose_response(tiny)
    r_vox = pack.pose_response(mm, cache=tmp_path)
    np.testing.assert_allclose(r_vox.coeffs, r_sub.coeffs * mm.voxel_factor()[:, None], rtol=1e-10, atol=1e-14)
    other = _prescribed(sp, L=(1e-3, 1e-3, 1e-3))
    r_other = pack.pose_response(other, cache=tmp_path)           # a second voxel size must not read the first's
    np.testing.assert_allclose(r_other.coeffs, r_sub.coeffs * other.voxel_factor()[:, None], rtol=1e-10, atol=1e-14)
    assert len(list(tmp_path.glob("*.npz"))) == 2


@pytest.mark.parametrize("orientation", [None, "posed"])
def test_the_bloch_route_puts_each_walker_at_its_own_place_in_the_voxel(pack, orientation):
    """The same average, per walker: each walker's walk translated to a drawn offset in the voxel. With hard
    pulses and no relaxation the vector route must give the scalar route's crushed signal to within the draw's
    own Monte-Carlo noise, and nowhere near the un-crushed one. A pose turns the substrate, not the voxel,
    and ``k . r_v`` is the same dot product in either frame."""
    from scipy.spatial.transform import Rotation
    R = Rotation.from_euler("zyx", [0.4, -0.3, 0.7]).as_matrix() if orientation else None
    mm = _prescribed(_spoiled(_pgse()))
    scalar = pack.replay(mm, complex_signal=True, orientation=R)
    bloch = pack.replay_bloch(mm, complex_signal=True, orientation=R)
    sigma = 1.0 / np.sqrt(2.0 * pack.n_walkers)                  # the draw's noise on the voxel average
    assert np.abs(bloch - scalar).max() < 4 * sigma, (bloch, scalar)
    tiny = _prescribed(_spoiled(_pgse()), L=(1e-9,) * 3)
    uncrushed = np.abs(pack.replay_bloch(tiny, complex_signal=True, orientation=R))
    # the crushed magnitude sits at the draw's own floor (a mean of n_w near-random phasors), so the
    # statement is that the un-crushed signal is many floors away from it
    assert uncrushed.min() - np.abs(bloch).max() > 4 * sigma, (uncrushed, bloch, sigma)


def test_a_declared_crusher_states_its_winding():
    from dmipy_sim.engine.bloch import _build_crusher
    rate, has = _build_crusher({"windows_s": [(1e-3, 3e-3)], "n_cycles": 4.0}, 1e-4, 100)
    assert has and rate.sum() == pytest.approx(2 * np.pi * 4.0)
    with pytest.raises(ValueError, match="states its winding"):
        _build_crusher({"windows_s": [(1e-3, 3e-3)]}, 1e-4, 100)
