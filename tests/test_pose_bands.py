"""Two bands, not one: sample from the response's own sharpness, retain what the distribution reaches (#164).

The composition is an inner product of coefficients, so a distribution has no reach above its own band and
none at all outside ``n = 0`` when it leaves the substrate's azimuth unstated. That makes the retained band an
exact economy. What it does *not* make cheap is the sampling: a quadrature exact only for the band being kept
folds everything above it into those coefficients.

Nothing here needs a statistical walk. Half the claims are algebra on synthetic coefficients, and the rest run
on a pack of **static walkers at chosen positions** driven by a single unrefocused gradient lobe, whose response
is the plane-wave sum ``mean_w exp(i q . R r_w)`` in closed form -- so the reference is analytic, the phase
amplitude is exactly ``|q| max|r|`` and can be dialled to whatever sharpness a test needs, and there is no
Monte-Carlo floor in the way. The statistical coverage lives in ``test_replay_pose.py``.
"""
import numpy as np
import pytest

from dmipy_sim.constants import GAMMA
from dmipy_sim.persistent_walk import PersistentWalk
from dmipy_sim.replay import so3
from dmipy_sim.replay.bank import build_replay_pack
from dmipy_sim.replay.fod import FOD

N_T, DT = 65, 2.0e-4
POS = np.array([[1.0, 0, 0], [0, 1.0, 0], [0.6, 0.6, 0.5],
                [-0.8, 0.2, -0.5], [0, 0, 1.0], [-0.3, -0.9, 0.2]]) * 3.0e-6
ENV = dict(bvals=[0.0, 1e9], dirs=[[0, 0, 1]], delta_frac=0.2, Delta_frac=0.5, ogse_periods=[1],
           shortd_b=1e9, shortd_deltas_frac=[0.2])


@pytest.fixture(scope="module")
def pack():
    """Six walkers standing still at chosen positions: a walk with no diffusion and no statistics."""
    walk = PersistentWalk(positions=np.repeat(POS[:, None, :], N_T, axis=1), dt=DT, sub_steps=1, dt_sim=DT)
    pk = build_replay_pack(walk, id="test/plane-wave", license="x", citation="x", K=8, envelope=ENV, field=False)
    assert pk.fidelity["err_max"] < 1e-8                        # a straight line is compressed exactly
    return pk


class _Lobe:
    """One unrefocused gradient lobe: its zeroth moment survives, so a static walker accrues ``q . r``."""

    def __init__(self, amp, axis=(1.0, 0.0, 0.0), dt=DT, n_t=N_T):
        self.G = np.zeros((1, n_t, 3))
        self.G[0, :, :] = amp * np.asarray(axis, float)
        self.dt, self.bvalues, self.rf_events = dt, np.array([0.0]), []
        self.q = GAMMA * amp * dt * (n_t - 1) * np.asarray(axis, float)   # the exact integral of a constant


def analytic(q, R):
    """The response of the static ensemble at pose ``R``, in closed form."""
    return np.mean(np.exp(1j * (POS @ np.asarray(R).T @ q)))


def analytic_over(q, density, n_theta=24, n_phi=48, n_roll=12):
    """The same, averaged over a density on directions with the azimuth uniform."""
    dirs, w = so3.sphere_quadrature(n_theta, n_phi)
    rolls = np.arange(n_roll) * (2 * np.pi / n_roll)
    vals = np.array([[analytic(q, so3.rotation_of(n, roll=r)) for r in rolls] for n in dirs]).mean(axis=1)
    rho = density(dirs)
    return complex((w * rho * vals).sum() / (w * rho).sum())


# ------------------------------------------------------------------ algebra, no pack
def test_a_grid_sized_to_the_retained_band_folds_higher_content_into_it():
    """The mechanism, on a function whose coefficients are known exactly. A rule exact for the retained band
    integrates that band correctly and everything above it wrongly, so higher content lands in the coefficients
    being kept. Oversampling removes it; sizing the grid to the retained band does not.

    (On a pack the same effect measured 0.067 in signal units at a retained band of (4, 0) against a floor of
    0.011, with a susceptibility field, whose azimuthal structure is what folds.)
    """
    keep_l, keep_n, hi = 2, 0, 8
    truth = np.random.default_rng(0).normal(size=so3.n_so3_coeffs(hi, hi))
    exact = so3.truncate_coeffs(truth, hi, hi, keep_l, keep_n)

    def f(R):
        return so3.so3_design(hi, R, hi) @ truth

    Rn, wn, _d, _r = so3.so3_quadrature(keep_l, keep_n)                   # exact for the retained band alone
    naive = so3.project(so3.so3_design(keep_l, Rn, keep_n), wn, f(Rn))
    Ro, wo, Ao = so3.oversampled_design(keep_l, keep_n, hi)               # exact past the function's own band
    np.testing.assert_allclose(so3.project(Ao, wo, f(Ro)), exact, atol=1e-10)
    assert np.abs(naive - exact).max() > 0.1 * np.abs(exact).max()


def test_retaining_the_distributions_band_is_exact_arithmetic():
    """Why the economy is free: the terms dropped are multiplied by coefficients the distribution does not
    have -- above its order, and outside ``n = 0`` when its azimuth is unstated."""
    hi = 6
    c = np.random.default_rng(1).normal(size=so3.n_so3_coeffs(hi, hi))
    fod = FOD.watson(3.0, mu=(0.3, 0.5, 0.81), lmax=4)
    full = so3.Distribution.axis_density(fod, hi, hi)
    off = np.array([0.0 if n == 0 else 1.0 for (_l, _m, n) in so3.so3_index(hi, hi)])
    assert float(np.abs(full.coeffs) @ off) < 1e-9                        # nothing outside n = 0
    for keep in ((hi, 0), (4, 0)):
        part = so3.Distribution.axis_density(fod, *keep)
        np.testing.assert_allclose(so3.truncate_coeffs(c, hi, hi, *keep) @ part.coeffs,
                                   c @ full.coeffs, rtol=1e-12)


# ------------------------------------------------------------------ the pipeline, against the closed form
def test_the_phase_amplitude_is_the_accumulated_phase(pack):
    """`Phi` is what the pose modulates, in radians, and for this ensemble it is exactly ``|q| max|r|`` -- so it
    is derived rather than fitted, and it scales with the gradient as the phase does."""
    for amp in (0.02, 0.2, 0.6):
        seq = _Lobe(amp)
        pr = pack.pose_response(seq, tissue=False)
        expected = float(np.linalg.norm(seq.q) * np.linalg.norm(POS, axis=1).max())
        np.testing.assert_allclose(pr.phase_amplitude, expected, rtol=0.02)


def test_the_expansion_reproduces_the_analytic_response(pack):
    """The projection against the closed form, at poses it never sampled. The residual is the Jacobi-Anger tail
    of a plane wave beyond the retained order, so at a fixed margin it grows with `Phi`: measured 3e-6, 8e-4 and
    4e-3 at `Phi` = 0.2, 2.1 and 6.2 rad."""
    for amp, tol in ((0.02, 1e-5), (0.2, 2e-3), (0.6, 1e-2)):
        seq = _Lobe(amp)
        pr = pack.pose_response(seq, tissue=False)
        for R in so3.haar_rotations(8, seed=2):
            assert abs(float(np.abs(pr.at(R))[0]) - abs(analytic(seq.q, R))) < tol


def test_the_band_follows_the_phase_amplitude_and_a_smaller_one_is_worse(pack):
    """The band is ``ceil(Phi) + margin``, and forcing it below the phase amplitude costs accuracy against the
    closed form -- which is what makes this a rule rather than a default."""
    seq = _Lobe(0.6)
    pr = pack.pose_response(seq, tissue=False)
    assert pr.lmax == int(np.ceil(pr.phase_amplitude)) + 2
    R = so3.haar_rotations(8, seed=3)
    truth = np.abs([analytic(seq.q, r) for r in R])
    good = np.abs([pr.at(r)[0] for r in R])
    with pytest.warns(UserWarning, match="not represented"):
        coarse = pack.pose_response(seq, band=3, tissue=False, strict=False)   # deliberately below Phi
    poor = np.abs([coarse.at(r)[0] for r in R])
    assert np.abs(poor - truth).max() > 10 * np.abs(good - truth).max()


def test_the_reported_misfit_is_the_worst_case_not_a_spread(pack):
    """A spread bounds nothing. With the band forced below the phase amplitude the error is uneven across
    poses, and the number reported must be the largest of them rather than their scatter."""
    seq = _Lobe(0.6)
    with pytest.warns(UserWarning, match="not represented"):
        pr = pack.pose_response(seq, band=3, tissue=False, strict=False)
    R = so3.haar_rotations(120, seed=4)
    err = np.abs(np.array([pr.at(r)[0] for r in R]) - np.array([analytic(seq.q, r) for r in R]))
    assert err.max() > 3 * err.std()                                      # the error really is uneven
    assert pr.misfit.max() > 2 * err.std()                                # so a spread would have hidden it
    np.testing.assert_allclose(pr.misfit.max(), err.max(), rtol=0.4)      # what is reported is the max


def test_composition_equals_the_analytic_pose_average(pack):
    """The composition against its definition: the density-weighted, azimuth-averaged closed form."""
    seq = _Lobe(0.2)
    pr = pack.pose_response(seq, tissue=False)
    for kappa in (1.0, 4.0):
        fod = FOD.watson(kappa, mu=(0.3, 0.5, 0.81), lmax=6)
        got = pr.compose(so3.Distribution.axis_density(fod, pr.lmax, pr.nmax))[0]
        assert abs(abs(got) - abs(analytic_over(seq.q, fod.evaluate))) < 5e-3


def test_one_stated_pose_needs_no_expansion(pack):
    """Rotating the substrate is rotating the acquisition, so one pose is exact with no band, no sampling and
    nothing to certify -- the documented route when a single pose is all that is wanted."""
    seq = _Lobe(0.6)
    for R in so3.haar_rotations(4, seed=5):
        got = pack.replay(seq, orientation=R, tissue=False, complex_signal=True)[0]
        assert abs(abs(got) - abs(analytic(seq.q, R))) < 1e-6


# ------------------------------------------------------------------ guards
def test_an_acquisition_too_sharp_to_expand_is_refused_with_the_reason(pack):
    """Past a point a direct replay per pose is the cheaper and exact route, so the cost is stated."""
    with pytest.raises(ValueError, match="cheaper and exact route|band_cap"):
        pack.pose_response(_Lobe(1.5), band_cap=6, tissue=False)


def test_no_azimuthal_truncation_is_a_valid_request(pack):
    """``nmax=None`` means "all of it" and used to raise ``int(None)``."""
    pr = pack.pose_response(_Lobe(0.2), keep=(4, None), tissue=False)
    assert pr.lmax == 4 and pr.nmax >= 4


def test_a_waveform_on_its_own_grid_composes_correctly(pack):
    """The case that exposed a 20% error: a waveform sampled at its own rate, not the pack's."""
    seq = _Lobe(0.2, dt=DT / 3.0, n_t=3 * (N_T - 1) + 1)
    assert abs(seq.dt - pack.dt) > 1e-9
    pr = pack.pose_response(seq, tissue=False)
    for R in so3.haar_rotations(4, seed=6):
        assert abs(float(np.abs(pr.at(R))[0]) - abs(analytic(seq.q, R))) < 2e-3


def test_the_projection_is_deterministic(pack):
    """Same inputs, same coefficients: a phantom built twice is the same phantom."""
    seq = _Lobe(0.2)
    np.testing.assert_array_equal(pack.pose_response(seq, tissue=False).coeffs,
                                  pack.pose_response(seq, tissue=False).coeffs)
