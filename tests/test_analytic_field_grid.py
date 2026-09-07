"""An analytic myelinated substrate gets the same field tier a mesh does: `field_grid_of` rasterises
its sheath onto the field-basis grid, and a pack built from its walk replays with a field."""
import numpy as np
import pytest

import dmipy_sim as d
from dmipy_sim.compartments import Compartments, Pool
from dmipy_sim.fields.susceptibility_field import FieldGrid, field_grid_of, assemble_field, sample_grid
from dmipy_sim.replay.bank import build_replay_pack

D0 = 2e-9
ENV = dict(bvals=[0.0, 1e9], dirs=[[1, 0, 0]], ogse_periods=[2], shortd_b=1e9, shortd_deltas_frac=[0.05],
           B0_list=[3.0], theta_deg=[90], delta_frac=0.2, Delta_frac=0.5, rho_list=[1e-5])


def test_hollow_cylinder_grid_reproduces_the_analytic_field_structure():
    """An isotropic hollow cylinder has exactly zero field in the lumen and a dipolar field outside;
    the anisotropic one a uniform lumen field of half delta_chi_a B0 sin^2(theta) ln(1/g)."""
    g = d.MyelinatedCylinder(2e-6, 3e-6, (0, 0, 1), D0, D0)
    fg = field_grid_of(g, res=0.1e-6)
    assert isinstance(fg, FieldGrid) and fg.basis["shape"][2] == 4 and fg.basis["aniso_G"] is not None
    B0 = 3.0
    iso = assemble_field(fg.basis, (1, 0, 0), B0=B0, chi_iso=1e-6, chi_aniso=0.0)      # B0 perpendicular
    aniso = assemble_field(fg.basis, (1, 0, 0), B0=B0, chi_iso=0.0, chi_aniso=-0.1e-6)
    ang = np.linspace(0, 2 * np.pi, 48, endpoint=False)
    lumen = np.stack([1.2e-6 * np.cos(ang), 1.2e-6 * np.sin(ang), np.zeros_like(ang)], 1)
    outside = np.stack([5e-6 * np.cos(ang), 5e-6 * np.sin(ang), np.zeros_like(ang)], 1)
    vs = fg.basis["voxel_size"]
    f_iso_in = sample_grid(iso, lumen, fg.origin, vs)
    f_iso_out = sample_grid(iso, outside, fg.origin, vs)
    assert np.abs(f_iso_in).max() < 0.01 * 1e-6 * B0                       # zero lumen field, to ringing
    assert np.abs(f_iso_out).max() > 0.02 * 1e-6 * B0                      # a dipolar field outside
    assert abs(np.mean(f_iso_out * np.cos(2 * ang))) > 0.5 * np.abs(f_iso_out).max() / 2   # cos(2 alpha) lobes
    f_an_in = sample_grid(aniso, lumen, fg.origin, vs)
    expect = 0.5 * (-0.1e-6) * B0 * np.log(3e-6 / 2e-6)                     # sin^2(90 deg) = 1
    np.testing.assert_allclose(f_an_in.mean(), expect, rtol=0.1)
    assert f_an_in.std() < 0.1 * abs(expect)                               # uniform


def test_packed_grid_is_the_periodic_cell_and_a_pack_replays_with_the_field():
    L = float(np.sqrt(np.pi * 3 * (1e-6 / 0.7) ** 2 / 0.5))
    _, _, c = d.pack_myelinated_cylinders([1e-6] * 3, 0.7, None, cell_size=L, seed=0)
    pm = d.PackedMyelinatedCylinders([1e-6] * 3, 0.7, c, L, N_max=4, D_intra=D0, D_extra=D0)
    fg = field_grid_of(pm, res=0.1e-6, include_aniso=False)
    assert fg.basis["iso_local"].shape[:2] == tuple(np.round(np.array([L, L]) / 0.1e-6).astype(int))
    with pytest.raises(ValueError, match="periodic cell"):
        field_grid_of(pm, box=(np.zeros(2), np.ones(2)))
    with pytest.raises(TypeError, match="MyelinatedCylinder"):
        field_grid_of(d.Sphere(1e-6))
    walk = d.simulate_trajectories(200, D0, pm, 4e-3, 4e-4, seed=0, require_gpu=False)
    comps = Compartments(extra=Pool(T2=0.08), intra=Pool(T2=0.05), myelin=Pool(T2=0.01))
    pk = build_replay_pack(walk, id="test/pm-field", compartments=comps, field=fg, K=8, envelope=ENV,
                           license="x", citation="x")
    assert pk.has_field and pk.has_relaxation and pk.has_surface
    G0 = np.zeros((1, pk.n_t, 3))                                           # b = 0, gradient echo
    with pytest.raises(ValueError, match="without chi_iso"):
        pk.replay(G0, B0=3.0, b0_dir=(1, 0, 0), relaxation=False, refocus_time=None)
    with_field = pk.replay(G0, B0=3.0, b0_dir=(1, 0, 0), chi_iso=1e-6, relaxation=False, refocus_time=None)
    no_field = pk.replay(G0, relaxation=False)
    assert no_field[0] == pytest.approx(1.0) and with_field[0] < no_field[0]      # the field dephases
    lumen = pk.replay(G0, B0=3.0, b0_dir=(1, 0, 0), chi_iso=1e-6, relaxation=False, refocus_time=None, compartment=1)
    assert lumen[0] > with_field[0]                                        # zero lumen field: the lumen keeps more
