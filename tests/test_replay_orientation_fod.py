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
    walk = d.simulate_trajectories(240, D0, g, 6e-3, 3e-4, seed=0, require_gpu=False)
    pk = build_replay_pack(walk, id="test/hollow", license="x", citation="x", K=8, envelope=ENV,
                           field=field_grid_of(g, res=0.2e-6), susc_path_K=16)
    assert pk.has_field and "susceptibility_path" in pk.meta["compression"]["channels"]
    return pk, _pgse(pk, [[1, 0, 0], [1, 0, 0], [0, 0, 1]], [0.0, 1e9, 1e9])


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
    pk, seq = hollow
    fod = FOD.watson(3.0, mu=(0.3, 0.5, 0.81))
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
            assert not np.allclose(pk.replay(seq, fod=wrong, tissue=False), pk.replay(seq, fod=native, tissue=False), rtol=1e-3)
    with pytest.raises(ValueError, match="unit-integral"):
        FOD.native(2.0 * native.coeffs)
    assert FOD.native(2.0 * native.coeffs, normalize=True).integral == pytest.approx(1.0)
    with pytest.raises(ValueError, match="unknown basis"):
        FOD.from_sh(native.coeffs, basis="mrtrix")
