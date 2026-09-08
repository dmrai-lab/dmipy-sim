"""Pose knobs on `pack.replay`: `orientation=` (one pose) and `fod=` (a distribution of poses through the Gaunt
route), validated on the analytic hollow cylinder against brute-force pose averaging, with a field on so the
gradient-field coupling matters; and the FOD's basis provenance, with DIPY as the oracle for other conventions."""
import numpy as np
import pytest

import dmipy_sim as d
from dmipy_sim.fields.susceptibility_field import field_grid_of
from dmipy_sim.replay.bank import build_replay_pack
from dmipy_sim.replay.fod import FOD
from dmipy_sim.replay.gaunt import sphere_quadrature

D0 = 2e-9
ENV = dict(bvals=[0.0, 1e9, 3e9], dirs=[[1, 0, 0], [0, 0, 1]], ogse_periods=[2], shortd_b=1e9, shortd_deltas_frac=[0.05],
           B0_list=[3.0], theta_deg=[0, 90], delta_frac=0.2, Delta_frac=0.5, rho_list=[1e-5])


@pytest.fixture(scope="module")
def hollow():
    """One hollow cylinder along z, alone in a wide periodic cell, with its field basis: the axially symmetric
    substrate the Gaunt route was derived for (the periodic images are 30 um away)."""
    g = d.PackedMyelinatedCylinders([1.0e-6], 0.7, [[0.0, 0.0]], 30e-6, N_max=2, D_intra=D0, D_extra=D0)
    walk = d.simulate_trajectories(3000, D0, g, 6e-3, 3e-4, seed=0, require_gpu=False)      # enough walkers for a smooth response over poses
    pk = build_replay_pack(walk, id="test/hollow", license="x", citation="x", K=8, envelope=ENV,
                           field=field_grid_of(g, res=0.2e-6), susc_path_K=16)
    assert pk.has_field and "susceptibility_path" in pk.meta["compression"]["channels"]
    return pk, _pgse(pk, [[1, 0, 0], [1, 0, 0], [0, 0, 1]], [0.0, 6e8, 6e8])


class _Acq:
    """A bare gradient waveform on the pack grid (a gradient echo): ``G`` (n_meas, n_t, 3), ``dt``."""
    def __init__(self, G, dt):
        self.G, self.dt = G, dt
        self.bvalues = np.zeros(G.shape[0])


def _pgse(pk, dirs, bvals, delta=1e-3, Delta=3e-3):
    from dmipy_sim.constants import GAMMA
    n_t, dt = pk.n_t, pk.dt
    nd, ng = int(round(delta / dt)), int(round(Delta / dt))
    G = np.zeros((len(dirs), n_t, 3))
    for i, (g, b) in enumerate(zip(dirs, bvals)):
        g = np.asarray(g, float) / np.linalg.norm(g)
        amp = np.sqrt(b / ((GAMMA * nd * dt) ** 2 * ((ng - nd / 3) * dt))) if b > 0 else 0.0
        G[i, :nd] = amp * g; G[i, ng:ng + nd] = -amp * g
    acq = _Acq(G, dt); acq.bvalues = np.asarray(bvals, float)
    return acq


KW = dict(B0=3.0, b0_dir=(0.6, 0.0, 0.8), chi_iso=-0.1e-6, chi_aniso=-0.1e-6, refocus_time=None, complex_signal=True)


def test_orientation_is_the_counter_rotated_acquisition(hollow):
    pk, seq = hollow
    ref = pk.replay(seq, **KW)
    np.testing.assert_allclose(pk.replay(seq, orientation=np.eye(3), **KW), ref, rtol=1e-12)
    np.testing.assert_allclose(pk.replay(seq, orientation=(0, 0, 1), **KW), ref, rtol=1e-12)
    # a pose R is the acquisition rotated by R^T: same walk, no second simulation
    th = 0.7; R = np.array([[np.cos(th), 0, np.sin(th)], [0, 1, 0], [-np.sin(th), 0, np.cos(th)]])
    kw = dict(KW, b0_dir=tuple(R.T @ np.asarray(KW["b0_dir"])))
    np.testing.assert_allclose(pk.replay(seq, orientation=R, **KW), pk.replay(_Acq(np.asarray(seq.G) @ R, seq.dt), **kw), rtol=1e-10)
    # the convention: orientation=n turns the substrate axis onto lab n, so a lab gradient along n is seen along
    # the substrate axis (parallel), and one perpendicular to n stays perpendicular
    Rx = pk._rotation_of((1, 0, 0))
    np.testing.assert_allclose(Rx @ pk.frame_axis, [1, 0, 0], atol=1e-12)
    np.testing.assert_allclose(Rx.T @ np.array([1.0, 0, 0]), pk.frame_axis, atol=1e-12)
    assert abs(Rx.T @ np.array([0, 1.0, 0]) @ pk.frame_axis) < 1e-12
    with pytest.raises(ValueError, match="proper rotation"):
        pk.replay(seq, orientation=np.diag([1, 1, -1]))


def test_fod_composition_equals_brute_force_pose_averaging(hollow):
    """The Gaunt route against the definition: the FOD-weighted average of the pack replayed at every pose, on a
    quadrature the FOD is band-limited on, with a field ON and B0 skew to the gradient (the coupled case)."""
    import warnings
    pk, seq = hollow
    fod = FOD.watson(3.0, mu=(0.3, 0.5, 0.81))
    with warnings.catch_warnings():
        warnings.simplefilter("error", UserWarning)                                  # the response IS band-limited here
        S = pk.replay(seq, fod=fod, **KW)
    dirs, w = sphere_quadrature(20, 40)
    f = fod.evaluate(dirs)
    brute = np.zeros(len(seq.bvalues), np.complex128)
    for n, wn, fn in zip(dirs, w, f):
        brute += wn * fn * pk.replay(seq, orientation=n, **KW)
    np.testing.assert_allclose(S, brute, rtol=3e-3, atol=3e-3)
    assert abs(S[0] - 1.0) < 2e-3                                                   # b = 0: the density integrates to 1
    # without a field the composition is a spherical convolution: the same route, checked the same way
    S0 = pk.replay(seq, fod=fod, tissue=False, complex_signal=True)
    brute0 = sum(wn * fn * pk.replay(seq, orientation=n, tissue=False, complex_signal=True) for n, wn, fn in zip(dirs, w, f))
    np.testing.assert_allclose(S0, brute0, rtol=3e-3, atol=3e-3)


def test_a_bare_array_and_a_multi_axis_waveform_are_refused(hollow):
    pk, seq = hollow
    with pytest.raises(TypeError, match="basis"):
        pk.replay(seq, fod=FOD.watson(3.0).coeffs, tissue=False)
    with pytest.raises(ValueError, match="exclude each other"):
        pk.replay(seq, fod=FOD.isotropic(), orientation=(0, 0, 1), tissue=False)
    G = np.zeros((1, pk.n_t, 3)); G[0, :4, 0] = 0.05; G[0, 4:8, 1] = 0.05                     # two axes in one measurement
    with pytest.raises(ValueError, match="single gradient direction"):
        pk.replay(_Acq(G, pk.dt), fod=FOD.isotropic(), tissue=False)


def test_fod_provenance_named_bases_convert_and_a_wrong_basis_would_be_wrong(hollow):
    """The required basis is DIPY's `real_sh_tournier(legacy=False)`. Coefficients in the legacy (non-orthonormal)
    Tournier basis or in the Descoteaux basis convert exactly per RPH 4.1; fed in unconverted they change the
    signal, which is the silent error the provenance requirement exists for."""
    shm = pytest.importorskip("dipy.reconst.shm")
    from dipy.core.sphere import Sphere
    pk, seq = hollow
    native = FOD.watson(3.0, mu=(0.3, 0.5, 0.81), lmax=8)
    dirs, w = sphere_quadrature(48, 96)
    f = native.evaluate(dirs)
    sph = Sphere(xyz=dirs)
    for basis, legacy in (("tournier07", True), ("tournier07", False), ("descoteaux07", False)):
        B = shm.sh_to_sf_matrix(sph, sh_order_max=8, basis_type=basis, legacy=legacy, return_inv=False)     # (n_sh, n_dirs)
        c_other = np.linalg.lstsq(B.T, f, rcond=None)[0]                            # that basis's coefficients of the same FOD
        assert np.allclose(B.T @ c_other, f, atol=1e-9)
        back = FOD.from_sh(c_other, basis=basis, legacy=legacy)
        np.testing.assert_allclose(back.coeffs, native.coeffs, atol=1e-9)
        if not (basis == "tournier07" and not legacy):
            wrong = FOD.native(c_other, normalize=True)                            # the unconverted mistake
            assert np.abs(wrong.evaluate(dirs) - f).max() > 0.05 * np.abs(f).max()     # a different distribution ...
            S_wrong, S_native = pk.replay(seq, fod=wrong, tissue=False), pk.replay(seq, fod=native, tissue=False)
            assert np.abs(S_wrong - S_native).max() > 1e-5                        # ... and a different signal, silently
    with pytest.raises(ValueError, match="unit-integral"):
        FOD.native(2.0 * native.coeffs)
    assert FOD.native(2.0 * native.coeffs, normalize=True).integral == pytest.approx(1.0)
    with pytest.raises(ValueError, match="unknown basis"):
        FOD.from_sh(native.coeffs, basis="mrtrix")


def test_a_response_the_two_axis_expansion_cannot_represent_is_flagged(hollow, monkeypatch):
    """The two-axis expansion assumes the response depends on the pose only through n.g and n.B0. When the fit
    misfits the response by more than the pack's own floor, the composition says so instead of returning a
    plausible number (the guard is exercised by forcing the misfit; a two-cylinder cell trips it for real only
    at a b and ensemble size too costly for this tier)."""
    from dmipy_sim.replay import sh_convolution as sh
    pk, seq = hollow
    real = sh.coupled_spectrum_at
    monkeypatch.setattr(sh, "coupled_spectrum_at", lambda *a, **k: (lambda out: (out[0], 1.0, out[2]))(real(*a, **k)))
    with pytest.warns(UserWarning, match="not that of an axially symmetric substrate"):
        pk.replay(seq, fod=FOD.watson(3.0), tissue=False)


def test_pose_spectra_are_the_factored_fod_route_and_carry_the_peak_limit(hollow):
    """``pose_spectra`` is what a phantom composes against every voxel: ``compose(fod)`` is ``replay(fod=)``
    exactly, and ``at(n)`` -- the peak, RPH 4 -- is the pack replayed at that one pose within the two-axis
    truncation and the pack's own Monte-Carlo floor."""
    from dmipy_sim.replay import PoseSpectra
    pk, seq = hollow
    ps = pk.pose_spectra(seq, **{k: v for k, v in KW.items() if k != "complex_signal"})
    assert isinstance(ps, PoseSpectra) and ps.n_meas == len(seq.bvalues)
    fod = FOD.watson(3.0, mu=(0.3, 0.5, 0.81))
    np.testing.assert_allclose(ps.compose(fod), pk.replay(seq, fod=fod, **KW), rtol=1e-12)
    # a single pose differs from the spectra by the pack's own roughness: 3000 walkers seen from one side are not
    # axially symmetric at the 1/sqrt(N) level, and the expansion keeps the symmetric part (misfit ~ floor, checked)
    floor = 1.0 / np.sqrt(pk.n_walkers)
    assert ps.misfit < 2.0 * floor
    for n in [(0, 0, 1), (1, 0, 0), (0.3, 0.5, 0.81)]:
        n = np.asarray(n, float) / np.linalg.norm(n)
        np.testing.assert_allclose(ps.at(n), pk.replay(seq, orientation=n, **KW), atol=3.0 * floor)
    # the same spectra without a field: a spherical convolution kernel, same contract
    ps0 = pk.pose_spectra(seq, tissue=False)
    np.testing.assert_allclose(ps0.compose(fod), pk.replay(seq, fod=fod, tissue=False, complex_signal=True), rtol=1e-12)


def test_the_roll_average_is_adaptive_and_free_on_an_axisymmetric_substrate(hollow, monkeypatch):
    """A pose is an axis, not a full rotation, so the response is averaged over the roll a slot does not
    declare. A hollow cylinder is axially symmetric, so one roll already fits inside its Monte-Carlo floor and
    no extra work is done. When the misfit does not come down, the count doubles to the cap and then says so,
    and each doubling refines a uniform average rather than restarting it."""
    import warnings
    from dmipy_sim.replay import sh_convolution as sh
    from dmipy_sim.replay.replay import _van_der_corput
    pk, seq = hollow
    kw = {k: v for k, v in KW.items() if k != "complex_signal"}
    with warnings.catch_warnings():
        warnings.simplefilter("error", UserWarning)
        ps = pk.pose_spectra(seq, **kw)
    assert ps.n_roll == 1 and ps.misfit < 2.0 / np.sqrt(pk.n_walkers)
    real = sh.coupled_spectrum_at
    monkeypatch.setattr(sh, "coupled_spectrum_at", lambda *a, **k: (lambda o: (o[0], 1.0, o[2]))(real(*a, **k)))
    with pytest.warns(UserWarning, match="averaged over 4 rolls"):
        pk.pose_spectra(seq, n_theta=12, n_phi=24, n_roll_max=4, **kw)
    # every power-of-two prefix of the roll schedule is a uniform average over the circle
    for m in (1, 2, 4, 8):
        rolls = np.sort([_van_der_corput(j) for j in range(m)])
        np.testing.assert_allclose(rolls, (np.arange(m) + 0.0) / m, atol=1e-12)
