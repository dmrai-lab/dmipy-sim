"""A pack's pose: one rotation exactly, or a distribution of rotations through the SO(3) response (#157)."""
import numpy as np
import pytest

import dmipy_sim as d
from dmipy_sim import Encoding, ScannerSequence
from dmipy_sim.fields.susceptibility_field import field_grid_of
from dmipy_sim.replay import so3
from dmipy_sim.replay.bank import build_replay_pack
from dmipy_sim.replay.fod import FOD

D0 = 2.0e-9
ENV = dict(bvals=[0.0, 1e8], dirs=[[0, 0, 1], [1, 0, 0]], delta_frac=0.2, Delta_frac=0.5,
           ogse_periods=[1], shortd_b=1e8, shortd_deltas_frac=[0.2])


@pytest.fixture(scope="module")
def hollow():
    """One hollow cylinder along z, alone in a wide periodic cell, with its field basis: an axially symmetric
    substrate, where the azimuthal coefficients must come out at the Monte-Carlo floor."""
    g = d.PackedMyelinatedCylinders([1.0e-6], 0.7, [[0.0, 0.0]], 30e-6, N_max=2, D_intra=D0, D_extra=D0)
    walk = d.simulate_trajectories(3000, D0, g, 6e-3, 3e-4, seed=0, require_gpu=False)
    pk = build_replay_pack(walk, id="test/hollow", license="x", citation="x", K=8, envelope=ENV,
                           field=field_grid_of(g, res=0.2e-6), susc_path_K=16)
    assert pk.has_field and "susceptibility_path" in pk.meta["compression"]["channels"]
    return pk, _pgse(pk, [[1, 0, 0], [1, 0, 0], [0, 0, 1]], [0.0, 1e8, 1e8])


@pytest.fixture(scope="module")
def ellipsoid():
    """A pore with three different semi-axes: manifestly not axially symmetric, so its response depends on the
    substrate's own azimuth -- the case an axis-only representation cannot hold at all."""
    g = d.Ellipsoid(semiaxes=(1.0e-6, 3.5e-6, 8.0e-6))
    walk = d.simulate_trajectories(4000, D0, g, 6e-3, 3e-4, seed=0, require_gpu=False)
    pk = build_replay_pack(walk, id="test/ellipsoid", license="x", citation="x", K=8,
                           envelope=dict(ENV, bvals=[0.0, 6e8], shortd_b=6e8), field=False)
    # a closed pore bounds the displacement, so a high b here is a sharp *angular* response at a modest phase
    # amplitude -- which is what makes this the substrate that shows azimuthal structure
    return pk, _pgse(pk, [[1, 0, 0], [0, 1, 0]], [6e8, 6e8])


def _Acq(G, dt, bvalues=None):
    """A bare gradient waveform on the pack grid (a gradient echo): no pulses, so the effective gradient is the
    gradient itself; the declared b defaults to zero."""
    G = np.asarray(G)
    return ScannerSequence(G=G, dt=dt, family="gre",
                           encoding=Encoding(bvalues=np.zeros(G.shape[0]) if bvalues is None else np.asarray(bvalues, float),
                                             gradient_directions=np.zeros((G.shape[0], 3))))


def _pgse(pk, dirs, bvals, delta=1e-3, Delta=3e-3):
    from dmipy_sim.constants import GAMMA
    n_t, dt = pk.n_t, pk.dt
    nd, ng = int(round(delta / dt)), int(round(Delta / dt))
    G = np.zeros((len(dirs), n_t, 3))
    for i, (g, b) in enumerate(zip(dirs, bvals)):
        g = np.asarray(g, float) / np.linalg.norm(g)
        amp = np.sqrt(b / ((GAMMA * nd * dt) ** 2 * ((ng - nd / 3) * dt))) if b > 0 else 0.0
        G[i, :nd] = amp * g; G[i, ng:ng + nd] = -amp * g
    return _Acq(G, dt, bvals)


KW = dict(B0=3.0, b0_dir=(0.6, 0.0, 0.8), chi_iso=-0.1e-6, chi_aniso=-0.1e-6)
BAND = dict(n_check=200)          # no band: it follows the response's phase amplitude


def test_one_pose_is_the_counter_rotated_acquisition(hollow):
    """A single pose needs no expansion: it is the same walk under a rotated acquisition, exact."""
    pk, seq = hollow
    ref = pk.replay(seq, complex_signal=True, **KW)
    np.testing.assert_allclose(pk.replay(seq, orientation=np.eye(3), complex_signal=True, **KW), ref, rtol=1e-12)
    np.testing.assert_allclose(pk.replay(seq, orientation=(0, 0, 1), complex_signal=True, **KW), ref, rtol=1e-12)
    th = 0.7
    R = np.array([[np.cos(th), 0, np.sin(th)], [0, 1, 0], [-np.sin(th), 0, np.cos(th)]])
    kw = dict(KW, b0_dir=tuple(R.T @ np.asarray(KW["b0_dir"])))
    np.testing.assert_allclose(pk.replay(seq, orientation=R, complex_signal=True, **KW),
                               pk.replay(_Acq(np.asarray(seq.G_eff) @ R, seq.dt), complex_signal=True, **kw),
                               rtol=1e-6)                     # the object stores G in float32: agreement to its rounding
    with pytest.raises(ValueError, match="proper rotation"):
        pk.replay(seq, orientation=np.diag([1, 1, -1]))


def test_the_expansion_reproduces_a_replay_at_any_rotation(hollow):
    """The projection against the thing it represents: the response read out of the coefficients at rotations it
    was not sampled at must match the pack replayed at that pose, to the pack's own noise floor."""
    pk, seq = hollow
    pr = pk.pose_response(seq, **BAND, **KW)
    assert pr.misfit.max() < 2.0 / np.sqrt(pk.n_walkers)
    for R in so3.haar_rotations(6, seed=7):
        np.testing.assert_allclose(pr.at(R), pk.replay(seq, orientation=R, complex_signal=True, **KW),
                                   atol=3.0 / np.sqrt(pk.n_walkers))
    # composing a single pose is that same read-out, so a phantom slot holding one pose needs no other path
    R0 = so3.haar_rotations(1, seed=8)[0]
    np.testing.assert_allclose(pr.compose(so3.Distribution.pose(R0, pr.lmax, pr.nmax)), pr.at(R0), rtol=1e-9)


def test_a_multi_axis_waveform_is_expanded_too(hollow):
    """The contraction is over the waveform's own components, so a gradient that turns during the measurement
    is no different -- the axis-only expansion had to refuse those."""
    pk, seq = hollow
    G = np.asarray(seq.G_eff).copy()
    n_t = G.shape[1]
    G[1, : n_t // 4, 1] = G[1, : n_t // 4, 0] * 0.7                  # a second axis during the first lobe
    twisted = _Acq(G, seq.dt)
    pr = pk.pose_response(twisted, **BAND, **KW)
    for R in so3.haar_rotations(3, seed=9):
        np.testing.assert_allclose(pr.at(R), pk.replay(twisted, orientation=R, complex_signal=True, **KW),
                                   atol=3.0 / np.sqrt(pk.n_walkers))


def test_the_azimuthal_energy_measures_how_symmetric_a_substrate_is(hollow, ellipsoid):
    """What the representation refuses to assume. A hollow cylinder is axially symmetric, so turning it about
    its own axis changes its response only by Monte-Carlo noise and its azimuthal coefficients are negligible;
    a three-axis pore is not, and both numbers say so. Measured here: asymmetry 5e-4 against 1e-2, and a
    quarter-turn moving the signal by 2.9x the pack's floor against 13.4x. The energy separates the two by a
    factor of twenty and is the statistic to read; the roll variation of an axially symmetric substrate does not
    go to zero because a finite ensemble of walkers is not itself symmetric, and the coefficients that carry
    that noise are exactly the ones an axis-only representation would have folded into its answer.
    """
    def roll_variation(pr):
        Rz = so3.rotation_of((0, 0, 1), roll=np.pi / 2)
        return max(float(np.abs(pr.at(R) - pr.at(R @ Rz)).max()) for R in so3.haar_rotations(4, seed=13))

    pk, seq = hollow
    sym = pk.pose_response(seq, **BAND, **KW)
    assert sym.asymmetry < 2e-3
    assert roll_variation(sym) < 4.0 * sym.floor

    pk2, seq2 = ellipsoid
    asym = pk2.pose_response(seq2, tissue=False, **BAND)
    assert asym.asymmetry > 5e-3
    assert roll_variation(asym) > 8.0 * asym.floor
    # and it is still represented, because the azimuth is a dimension of the basis rather than an assumption
    assert asym.misfit.max() < 2.0 * asym.floor
    for R in so3.haar_rotations(4, seed=11):
        np.testing.assert_allclose(asym.at(R), pk2.replay(seq2, orientation=R, tissue=False, complex_signal=True),
                                   atol=3.0 * asym.floor)


def test_an_axis_density_composes_like_averaging_explicit_poses(hollow):
    """The composition against the definition: an ODF states directions and says nothing about the substrate's
    azimuth, so its signal is the ODF-weighted, azimuth-averaged mean of the pack replayed at those poses."""
    pk, seq = hollow
    fod = FOD.watson(3.0, mu=(0.3, 0.5, 0.81), lmax=6)
    S = pk.replay(seq, orientation=fod, complex_signal=True, **KW)
    dirs, w = so3.sphere_quadrature(10, 20)
    rolls = np.arange(6) * (2 * np.pi / 6)
    rho = fod.evaluate(dirs)
    brute = np.zeros(len(seq.encoding.bvalues), np.complex128)
    for n, wn, fn in zip(dirs, w, rho):
        for r in rolls:
            brute += (wn * fn / len(rolls)) * pk.replay(seq, orientation=so3.rotation_of(n, roll=r),
                                                        complex_signal=True, **KW)
    np.testing.assert_allclose(S, brute, atol=4.0 / np.sqrt(pk.n_walkers))
    assert abs(S[0] - pk.replay(seq, complex_signal=True, **KW)[0]) < 1e-2      # b = 0: the density integrates to 1


def test_a_bingham_fan_reduces_to_the_watson_it_contains(ellipsoid):
    """Two concentrations, one frame: equal values are the Watson cone, and unequal ones are a fan that no
    single-parameter dispersion can express -- checked on the substrate whose response varies enough with pose
    for the difference to be visible above its own noise."""
    pk, seq = ellipsoid
    pr = pk.pose_response(seq, tissue=False, **BAND)
    frame = so3.rotation_of((0.3, 0.5, 0.81))
    cone = pr.compose(so3.Distribution.bingham(frame, (6.0, 6.0), lmax=pr.lmax, nmax=pr.nmax))
    watson = pr.compose(so3.Distribution.watson(6.0, mu=(0.3, 0.5, 0.81), lmax=pr.lmax, nmax=pr.nmax))
    np.testing.assert_allclose(cone, watson, atol=1e-9)
    fan = pr.compose(so3.Distribution.bingham(frame, (0.5, 30.0), lmax=pr.lmax, nmax=pr.nmax))
    assert np.abs(fan - cone).max() > 3.0 / np.sqrt(pk.n_walkers)
    # the powder average is the zeroth coefficient alone, and sits between the poses it averages
    powder = pr.compose(so3.Distribution.uniform(pr.lmax, pr.nmax))
    at = np.abs([pr.at(R) for R in so3.haar_rotations(12, seed=3)])
    assert (at.min(axis=0) <= np.abs(powder) + 1e-9).all() and (np.abs(powder) <= at.max(axis=0) + 1e-9).all()


def test_an_orientation_distribution_must_say_what_basis_it_is_in(hollow):
    """A bare coefficient array cannot say which spherical-harmonic convention it is in, and the wrong one is
    silently wrong, so it is refused wherever coefficients enter (RPH.md 4.1)."""
    pk, seq = hollow
    fod = FOD.watson(3.0, mu=(0.3, 0.5, 0.81), lmax=6)
    with pytest.raises(TypeError, match="basis|FOD"):
        so3.Distribution.axis_density(fod.coeffs)
    # the conversion is not a no-op: the same numbers read in the legacy basis are a different distribution
    wrong = so3.Distribution.axis_density(FOD.from_sh(fod.coeffs, basis="tournier07", legacy=True))
    right = so3.Distribution.axis_density(fod)
    assert np.abs(wrong.coeffs - right.coeffs).max() > 1e-2
    # a distribution stated beyond the response's band meets it at the band they share, which is exact: the
    # response has nothing above its band for those terms to multiply. (The closed form's full band is large,
    # so the response is truncated to (6, 6) here rather than the distribution raised above it.)
    pr = pk.pose_response(seq, **BAND, **KW)
    from dmipy_sim.replay.replay import PoseResponse
    low = PoseResponse(so3.truncate_coeffs(pr.coeffs, pr.lmax, pr.nmax, 6, 6), 6, 6, pr.misfit, pr.floor,
                       pr.phase_amplitude, pr.n_samples)
    wide = low.compose(so3.Distribution.watson(3.0, lmax=8, nmax=8))
    same = low.compose(so3.Distribution.watson(3.0, lmax=6, nmax=6))
    np.testing.assert_allclose(wide, same, atol=1e-12)


def test_a_response_the_truncation_cannot_hold_is_refused(ellipsoid):
    """The band is measured, not asserted: at a truncation the response does not fit into, composing would
    return a plausible wrong number, so the projection raises and names the knobs instead."""
    pk, seq = ellipsoid
    with pytest.raises(ValueError, match="not represented at"):
        pk.pose_response(seq, method="quadrature", band=0, n_check=200, tissue=False)   # the sampled route: a constant is not a response
