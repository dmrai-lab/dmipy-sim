"""The field basis's node spacing follows the field source's thinnest shell (dmipy-sim#304).

A rasterised field basis is only as good as its raster of the sheath: the dipole kernel does not decay with |k|,
so a sheath the grid does not resolve rings into the lumen where the intra pool sits. The spacing is therefore
derived -- ``FIELD_NODES_ACROSS`` nodes across the thinnest shell the spec records -- and the number of nodes
across is MEASURED here against the Wharton-Bowtell hollow cylinder, not assumed. A basis the host cannot afford
is refused naming the shell, never coarsened.
"""
import dataclasses
import os

import numpy as np
import pytest

import dmipy_sim as d
from dmipy_sim.fields.susceptibility_field import (FIELD_BYTES_PER_NODE, FIELD_NODES_ACROSS, assemble_field,
                                                   field_grid_of, field_node_budget, field_resolution)
from dmipy_sim.spec import SpecError, field_grid_of_spec, walk_spec, winther_spec
from dmipy_sim.spec.producers import shell_thickness

trimesh = pytest.importorskip("trimesh")


def _lumen_field_errors(res, a=0.40e-6, b=0.52e-6):
    """The hollow cylinder with the CACTUS bundle's thinnest sheath (0.120 um) rasterised at ``res``, B0 across the
    axis: the lumen's anisotropic term against Wharton-Bowtell's closed form (relative), and the isotropic lumen
    null as a fraction of chi (the exact answer is 0)."""
    dchi, chi, B0 = -1e-7, -1e-7, 3.0
    direction = (1.0, 0.0, 0.0)
    expect = 0.5 * dchi * np.log(b / a)
    lo, hi = np.array([-4e-6, -4e-6, -1e-6]), np.array([4e-6, 4e-6, 1e-6])
    fg = field_grid_of(d.MyelinatedCylinder(a, b, (0, 0, 1), 1.7e-9, 1.7e-9), res=res, box=(lo, hi))
    shape = tuple(fg.basis["shape"]); vs = np.asarray(fg.basis["voxel_size"], float)
    ax = [np.asarray(fg.origin)[k] + (np.arange(shape[k]) + 0.5) * vs[k] for k in range(2)]
    X, Y = np.meshgrid(ax[0], ax[1], indexing="ij")
    lumen = (X ** 2 + Y ** 2) < (0.6 * a) ** 2
    f_an = np.asarray(assemble_field(fg.basis, direction, B0=B0, chi_iso=0.0, chi_aniso=dchi)) / B0
    f_is = np.asarray(assemble_field(fg.basis, direction, B0=B0, chi_iso=chi, chi_aniso=0.0)) / B0
    got = np.mean([f_an[:, :, k][lumen].mean() for k in range(shape[2])])
    null = max(np.abs(f_is[:, :, k][lumen]).max() for k in range(shape[2]))
    return abs(got - expect) / abs(expect), null / abs(chi)


def test_two_nodes_across_the_thinnest_sheath_resolve_it_and_one_does_not():
    """The rule's measurement. At the derived spacing the lumen field is within half a percent of the closed form
    on both terms; at one node across the sheath it is not, so the rule is the smallest that works."""
    thin = 0.12e-6
    err_an, err_is = _lumen_field_errors(field_resolution(thin))
    assert err_an < 5e-3 and err_is < 5e-3, (err_an, err_is)
    err_an1, err_is1 = _lumen_field_errors(thin)
    assert err_an1 > 1e-2 and err_is1 > 1e-2, (err_an1, err_is1)
    assert field_resolution(thin) == pytest.approx(thin / FIELD_NODES_ACROSS)
    with pytest.raises(ValueError, match="positive"):
        field_resolution(0.0)


def test_analytic_sheaths_record_their_shell_and_the_grid_follows_it():
    """Every analytic myelinated geometry writes its sheath into the spec; its own field grid takes the spacing
    from it when none is given."""
    g = d.MyelinatedCylinder(1.0e-6, 1.15e-6, (0, 0, 1), 1.7e-9, 1.7e-9)
    assert g.spec.validity.thinnest_shell == pytest.approx(0.15e-6)
    fg = field_grid_of(g)
    assert np.asarray(fg.basis["voxel_size"])[0] == pytest.approx(0.075e-6, rel=0.05)
    pk = d.PackedMyelinatedCylinders([1.0e-6, 2.0e-6], [0.7, 0.9], [[-3e-6, 0.0], [3e-6, 0.0]], 12e-6, N_max=2,
                                     D_intra=1.7e-9, D_extra=1.7e-9)
    assert pk.spec.validity.thinnest_shell == pytest.approx(min(1.0e-6 / 0.7 - 1.0e-6, 2.0e-6 / 0.9 - 2.0e-6), rel=1e-6)


def _thin_axon(dir_path, r_in=1.88, r_out=2.0, height=4.0):
    """Concentric closed tubes in MICRONS (trimesh merges vertices at an absolute 1e-8, so a metre-scale mesh
    collapses); the producers take ``scale=1e-6``."""
    paths = []
    for tag, radius in (("inner", r_in), ("outer", r_out)):
        m = trimesh.creation.cylinder(radius=radius, height=height, sections=96)
        p = os.path.join(str(dir_path), f"{tag}.ply")
        m.export(p)
        paths.append(p)
    return paths


def test_a_mesh_pair_measures_its_shell(tmp_path):
    """The thickness of a meshed sheath, exact to the faceting; the spec records it."""
    inner, outer = _thin_axon(tmp_path)
    spec = winther_spec(inner, outer, scale=1e-6, pad=1e-6)
    assert spec.validity.thinnest_shell == pytest.approx(0.12e-6, rel=0.03)
    mi, mo = trimesh.load(inner), trimesh.load(outer)
    assert shell_thickness((mi.vertices * 1e-6, mi.faces), (mo.vertices * 1e-6, mo.faces)) == pytest.approx(0.12e-6, rel=0.03)
    assert spec.to_dict()["validity"]["thinnest_shell"] == spec.validity.thinnest_shell


def test_walk_spec_derives_the_grid_and_refuses_one_it_cannot_afford(tmp_path):
    inner, outer = _thin_axon(tmp_path)
    spec = winther_spec(inner, outer, scale=1e-6, pad=0.5e-6)
    w = walk_spec(spec, 24, 4e-4, 2e-4, seed=0, n_probe=5_000, require_gpu=False, field=True)
    vs = np.asarray(w.field_basis.basis["voxel_size"], float)
    np.testing.assert_allclose(vs, field_resolution(spec.validity.thinnest_shell), rtol=0.02)   # a whole count of nodes
    cert = w.field_basis.certificate                                                          # the raster checked
    assert cert["nodes_across_thinnest_shell"] == pytest.approx(2.0, rel=0.02)
    assert abs(cert["shell_fraction_raster"] - cert["shell_fraction_sampled"]) < 0.01 * cert["shell_fraction_sampled"] + 4 * cert["shell_fraction_se"]
    fg = field_grid_of_spec(spec)                                                             # the same raster, standalone
    np.testing.assert_array_equal(fg.basis["iso_local"], w.field_basis.basis["iso_local"])
    from dmipy_sim.replay.bank import build_replay_pack
    pk = build_replay_pack(w, id="t/raster", license="x", citation="x", K=3, susc_path_K=4, field=w.field_basis)
    assert pk.meta["compression"]["channels"]["susceptibility_grid"]["raster"]["res_m"] == cert["res_m"]
    with pytest.raises(SpecError, match="thinnest shell of 0.1"):
        walk_spec(spec, 24, 4e-4, 2e-4, seed=0, n_probe=5_000, require_gpu=False, field=True, field_budget=100)
    bare = dataclasses.replace(spec, validity=dataclasses.replace(spec.validity, thinnest_shell=None))
    with pytest.raises(SpecError, match="field_res="):
        walk_spec(bare, 24, 4e-4, 2e-4, seed=0, n_probe=5_000, require_gpu=False, field=True)
    coarse = walk_spec(bare, 24, 4e-4, 2e-4, seed=0, n_probe=5_000, require_gpu=False, field=True, field_res=0.5e-6)
    assert np.asarray(coarse.field_basis.basis["voxel_size"], float).max() > 0.4e-6         # an override is obeyed


def test_the_node_budget_follows_the_host():
    """Half the memory ceiling at the build's measured bytes per node; an explicit ceiling makes it a number."""
    assert field_node_budget(ceiling=2e9) == pytest.approx(1e9 / FIELD_BYTES_PER_NODE)
    assert field_node_budget() > 0
