"""A calibrated substrate realises itself in one call, the walk keeps its geometry, and the pack
builder reads pools and field basis from it: nothing the objects already know is retyped."""
import numpy as np
import pytest

import dmipy_sim as d
from dmipy_sim.compartments import Compartments, Pool
from dmipy_sim.replay.bank import build_replay_pack
from dmipy_sim.spec import Tissue
from dmipy_sim.substrate import Substrate

ENV = dict(bvals=[0.0, 1e9], dirs=[[1, 0, 0]], ogse_periods=[2], shortd_b=1e9, shortd_deltas_frac=[0.05],
           B0_list=[3.0], theta_deg=[90], delta_frac=0.2, Delta_frac=0.5, rho_list=[1e-5])


def test_substrate_pack_realises_the_diameter_law_by_outer_radius():
    sub = Substrate.canonical(field_T=3.0)
    d_out = sub.sample_outer_diameters(2000, seed=1)
    assert d_out.min() >= sub.d_min and abs(d_out.mean() - sub.gamma_shape_diameter * sub.gamma_scale_diameter) < 0.15e-6
    g = sub.pack(n_axons=16, seed=0)
    assert isinstance(g, d.PackedMyelinatedCylinders) and g.N_actual == 16
    inner = g._inner_radii_np[:16]; outer = g._outer_radii_np[:16]
    np.testing.assert_allclose(inner / outer, sub.g_ratio, rtol=1e-9)         # lumens follow the g-ratio
    assert (2 * outer).min() >= sub.d_min - 1e-12                              # the floor is on the fibre
    assert np.pi * np.sum(outer ** 2) / g._L_float ** 2 == pytest.approx(sub.f_axon, rel=1e-6)
    assert list(g.compartments) == ["extra", "intra", "myelin"]
    assert g.compartments["myelin"].T2 == sub.T2_myelin and float(np.asarray(g._D_extra_jax)[0]) == pytest.approx(sub.D_extra, rel=1e-6)
    assert float(np.asarray(g._rho_inner_jax)[0]) == pytest.approx(sub.rho2)
    assert float(np.asarray(g._kappa_inner_jax)[0]) == pytest.approx(sub.kappa)


def test_the_walk_keeps_its_geometry_and_the_builder_needs_nothing_else():
    sub = Substrate.canonical(field_T=3.0)
    g = sub.pack(n_axons=4, seed=0)
    walk = d.simulate_trajectories(120, sub.D_intra, g, 2e-3, 5e-4, seed=0, require_gpu=False)
    assert walk.geometry is g and walk.diffusivity == sub.D_intra
    pk = build_replay_pack(walk, id="t/sub", license="x", citation="x", K=8, envelope=ENV)
    assert pk.has_relaxation and pk.has_surface and pk.has_field
    assert "per_comp" not in pk.meta and pk.substrate.pools == g.spec.pools     # the spec, never the values
    gm = pk.meta["compression"]["channels"]["susceptibility_grid"]
    assert gm["has_aniso"], "the basis must carry the anisotropic part so chi_aniso is a replay knob"
    G0 = np.zeros((1, pk.n_t, 3))
    e = pk.replay(G0, B0=3.0, b0_dir=(1, 0, 0), chi_iso=1.06e-6, chi_aniso=-0.1e-6, refocus_time=None)
    assert 0 < e[0] < 1
    nominal = Tissue.from_spec(g.spec, B0=3.0, b0_dir=(1, 0, 0))                 # the spec's nominal values
    assert nominal.T2 == [sub.T2_extra, sub.T2_intra, sub.T2_myelin] and nominal.rho == pytest.approx(sub.rho2)
    assert nominal.chi_iso is None, "a Substrate declares the sheath a field source and no chi: a replay knob"
    with pytest.raises(ValueError, match="without chi_iso"):
        pk.replay(G0, tissue=nominal, refocus_time=None)
    tissue = Tissue.from_spec(g.spec, B0=3.0, b0_dir=(1, 0, 0), chi_iso=1.06e-6, chi_aniso=-0.1e-6)
    e_t = pk.replay(G0, tissue=tissue, refocus_time=None)
    np.testing.assert_allclose(e_t, pk.replay(G0, T2=tissue.T2, T1=tissue.T1, rho=tissue.rho, B0=3.0, b0_dir=(1, 0, 0),
                                              chi_iso=1.06e-6, chi_aniso=-0.1e-6, refocus_time=None))
    assert e_t[0] < e[0]                                                          # T2 and rho cost signal
    # field=False leaves the tier out
    pk2 = build_replay_pack(walk, id="t/own", license="x", citation="x", K=8, envelope=ENV, field=False)
    assert pk2.has_relaxation and not pk2.has_field
    # a bare cylinder: occupancy and local time are recorded (T2 / rho come at replay), no field source
    plain = d.simulate_trajectories(60, 2e-9, d.Cylinder(2e-6, (0, 0, 1)), 2e-3, 5e-4, seed=0, require_gpu=False)
    pk3 = build_replay_pack(plain, id="t/plain", license="x", citation="x", K=8, envelope=ENV)
    assert pk3.has_surface and pk3.has_relaxation and not pk3.has_field
