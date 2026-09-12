"""The pose expansion in closed form (#197): for a single-direction encoding the response over poses is a sum of
plane waves whose harmonics are the Rayleigh expansion, computed per walker with no quadrature. Certified against
the direct posed replay (exact), against the quadrature route (to that route's own misfit), and by its band bound."""
import time

import numpy as np
import pytest
from scipy.special import spherical_jn

import dmipy_sim as d
from dmipy_sim import sequences
from dmipy_sim.replay import read_rpk, so3
from dmipy_sim.replay.bank import build_replay_pack
from dmipy_sim.replay.so3 import Distribution


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
    P = pack._prepare(seq, tissue="nominal", T2=None, T1=None, rho=None, D=None, B0=None, b0_dir=(0, 0, 1),
                      chi_iso=None, chi_aniso=0.0, orientation=None, compartment=None)
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


def test_the_closed_form_is_fast_on_a_real_pack(seq):
    """The cost is one contraction over walkers per order: the 8k-walker CACTUS pack at four measurements in well
    under a second, where the quadrature route needs seconds per measurement."""
    import os
    path = "/home/rutger/dmrai-ws/packs/cactus_demo_xframe.rpk"
    if not os.path.exists(path):
        pytest.skip("the CACTUS demo pack is not on this machine")
    pk = read_rpk(path)
    seq30 = sequences.pgse([[1, 0, 0], [0, 0, 1], [0.6, 0.8, 0.0], [0.0, 0.6, 0.8]], 8e-3, 16e-3, bvalues=[3e9] * 4, TE=30e-3)
    # tissue=False: this pack's spec declares a nominal field, and a field response still takes the quadrature
    t0 = time.time(); pr = pk.pose_response(seq30, keep=(8, 0), tissue=False); dt = time.time() - t0
    assert dt < 5.0 and pr.n_samples == 0 and pr.phase_amplitude > 5.0     # a sharp response, still cheap at n = 0
    # and the n = 0 column composes the same signal as the direct replay averaged over the roll about an axis
    axis = np.array([0.3, 0.5, 0.81]); axis /= np.linalg.norm(axis)
    rolls = np.linspace(0, 2 * np.pi, 24, endpoint=False)
    direct = np.mean([pk.replay(seq30, orientation=so3.rotation_of(axis, roll=r), complex_signal=True, tissue=False) for r in rolls], axis=0)
    composed = pr.compose(Distribution.axis(axis, 8, 0))
    np.testing.assert_allclose(composed, direct, atol=3e-3)                  # order 8 of a kappa ~ 20 response: the ODF band
