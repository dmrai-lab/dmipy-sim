"""Mesh substrates are specs too: a single surface round-trips through its spec, and a multi-surface
bundle is walked pool by pool by `walk_spec` from the spec alone, into a pack that embeds it."""
import json
import os

import numpy as np
from dmipy_sim import ScannerSequence
import pytest

import dmipy_sim as d
from dmipy_sim.geometry import mesh_shapes
from dmipy_sim.spec import spec_of, geometry_from_spec, walk_spec, SubstrateSpec, SpecError
from dmipy_sim.replay.bank import build_replay_pack

D = 2e-9
ENV = dict(bvals=[0.0, 1e9], dirs=[[0, 0, 1]], ogse_periods=[2], shortd_b=1e9, shortd_deltas_frac=[0.05],
           B0_list=[3.0], theta_deg=[90], delta_frac=0.2, Delta_frac=0.5, rho_list=[1e-5])


def test_a_mesh_writes_its_spec_and_is_rebuilt_from_it(tmp_path, monkeypatch):
    V, F = mesh_shapes.icosphere(2e-6, subdivisions=2)
    m = d.Mesh(V, F, feature_radius=1e-6, permeability={"intra_to_extra": 2e-5, "extra_to_intra": 1e-5},
               compartments=d.Compartments(intra=d.Pool(T2=0.05, surface_relaxivity_t2=1e-6), extra=d.Pool(T2=0.1)),
               voxel_min=[-4e-6] * 3, voxel_max=[4e-6] * 3)
    monkeypatch.setenv("DMIPY_SIM_SURFACE_DIR", str(tmp_path / "cache"))
    cached = m.spec                                   # an in-memory mesh writes its surface to the cache, named by content
    assert cached.wall("surface").surface.file.startswith(str(tmp_path / "cache")) and m.source["file"] == cached.wall("surface").surface.file
    assert m.spec.wall("surface").surface.file == cached.wall("surface").surface.file      # written once
    spec = spec_of(m, surface_dir=tmp_path)           # an explicit directory still wins
    spec.validate()
    w = spec.wall("surface")
    assert w.surface.kind == "mesh" and os.path.exists(w.surface.file) and len(w.surface.sha256) == 64
    assert w.permeability.in_to_out == pytest.approx(2e-5, rel=1e-6) and w.permeability.out_to_in == pytest.approx(1e-5, rel=1e-6)
    assert w.surface_relaxivity.inside == pytest.approx(1e-6, rel=1e-6) and w.surface_relaxivity.outside == 0.0
    assert spec.domain.boundary == ["reflect"] * 3 and spec.seeding.pools == [1]
    assert spec.pool("intra").T2 == 0.05 and spec.validity.mesh_edge_feature_ratio > 0
    m2 = geometry_from_spec(spec)
    assert isinstance(m2, d.Mesh) and m2.source["file"] == w.surface.file
    spec2 = spec_of(m2)
    assert spec2 == spec                                     # a fixed point, the file included
    w1 = d.simulate_trajectories(40, D, m, 4e-4, 2e-4, seed=2, require_gpu=False)
    w2 = d.simulate_trajectories(40, D, m2, 4e-4, 2e-4, seed=2, require_gpu=False)
    np.testing.assert_array_equal(w1.positions, w2.positions)
    assert w2.spec == m2.spec and w2.geometry is m2               # a walk always carries its geometry's spec


def _bundle_spec(tmp_path, n_fibres=2, L=8.0e-6):
    import trimesh
    walls = []
    centres = [(-2.0e-6, 0.0), (2.0e-6, 0.0)][:n_fibres]
    um = 1e-6                                          # the files are in micrometres, as the datasets are
    for k, (cx, cy) in enumerate(centres):
        for tag, r, inside, outside in (("inner", 0.8e-6, 1, 2), ("outer", 1.2e-6, 2, 0)):
            tm = trimesh.creation.cylinder(radius=r / um, height=0.9 * L / um, sections=48)
            tm.apply_translation([cx / um, cy / um, 0.0])
            path = tmp_path / f"fibre_{k}_{tag}.ply"
            tm.export(str(path))
            walls.append({"name": f"fibre-{k}/{tag}", "surface": {"kind": "mesh", "file": str(path), "format": "ply", "scale": um},
                          "inside_pool": inside, "outside_pool": outside,
                          "permeability": {"in_to_out": 0.0, "out_to_in": 0.0},
                          "surface_relaxivity": {"inside": 1e-6 if tag == "inner" else 0.0, "outside": 0.0 if tag == "inner" else 1e-6},
                          "mt_reactivity": {"inside": 0.0, "outside": 0.0}})
    return SubstrateSpec.from_dict({
        "substrate_spec_version": "0.1", "id": "test/bundle-2",
        "domain": {"box_min": [-L / 2] * 3, "box_max": [L / 2] * 3, "boundary": ["reflect", "reflect", "reflect"]},
        "frame": {"axis": [0, 0, 1]},
        "pools": [{"id": 0, "name": "extra", "D": D, "T2": 0.08, "T1": 1.0, "water_fraction": 1.0, "susceptibility": None},
                  {"id": 1, "name": "intra", "D": D, "T2": 0.05, "T1": 1.2, "water_fraction": 1.0, "susceptibility": None},
                  {"id": 2, "name": "myelin", "D": 0.0, "T2": 0.01, "T1": 0.4, "water_fraction": 0.4,
                   "susceptibility": {"chi_iso": -1e-6, "chi_aniso": 0.0, "director": "radial"}}],
        "walls": walls,
        "seeding": {"pools": [0, 1, 2], "rule": "uniform_by_volume", "weights": "thin"},
        "validity": {"smallest_feature": 0.4e-6, "tiers": ["gradient", "relaxation", "surface", "field"]},
    }).validate()


def test_a_multi_surface_bundle_is_walked_from_the_spec_alone(tmp_path):
    spec = _bundle_spec(tmp_path)
    with pytest.raises(SpecError, match="pool by pool"):
        geometry_from_spec(spec)
    walk = walk_spec(spec, 240, 1e-3, 2.5e-4, seed=0, n_probe=20_000, field_res=0.4e-6, require_gpu=False)
    assert walk.spec is spec and walk.geometry is None and walk.weights is None      # thinned: unweighted
    ids = np.asarray(walk.compartment)[:, 0]
    assert set(np.unique(ids)) == {0, 1, 2} and (np.asarray(walk.compartment) == ids[:, None]).all()
    assert walk.has_surface and walk.diffusivity == D and walk.field_grid is not None
    pos = np.asarray(walk.positions)
    r0 = np.hypot(pos[:, 0, 0] - np.where(pos[:, 0, 0] < 0, -2e-6, 2e-6), pos[:, 0, 1])
    assert (r0[ids == 1] < 0.8e-6 + 1e-9).all() and (r0[ids == 2] > 0.8e-6 - 1e-9).all()      # pools where the walls say
    assert (pos[ids == 2] == pos[ids == 2][:, :1]).all()                                        # stuck myelin
    pk = build_replay_pack(walk, id="test/bundle", license="x", citation="x", K=8, envelope=ENV)
    assert pk.has_relaxation and pk.has_surface and pk.has_field
    assert pk.meta["substrate"]["id"] == "test/bundle-2" and SubstrateSpec.from_dict(pk.meta["substrate"]) == spec
    assert "per_comp" not in pk.meta and pk.substrate == spec
    G0 = ScannerSequence(G=np.zeros((1, pk.n_t, 3)), dt=pk.dt)      # b = 0, no pulse: a gradient echo
    e = pk.replay(G0, B0=3.0, b0_dir=(1, 0, 0), chi_iso=-1e-6)
    assert pk.replay(G0, T2={"intra": 0.05, "extra": 0.08, "myelin": 0.01})[0] < 1.0     # T2 by pool name
    assert 0 < e[0] <= 1.0


def test_walk_spec_walks_an_analytic_spec_too():
    g = d.Cylinder(3e-6, (0, 0, 1))
    w = walk_spec(g.spec, 32, 4e-4, 2e-4, diffusivity=D, seed=1, require_gpu=False)
    ref = d.simulate_trajectories(32, D, g, 4e-4, 2e-4, seed=1, require_gpu=False)
    np.testing.assert_array_equal(w.positions, ref.positions)
    assert w.spec == g.spec and type(w.geometry) is d.Cylinder
    with pytest.raises(SpecError, match="no D"):
        walk_spec(g.spec, 32, 4e-4, 2e-4, seed=1, require_gpu=False)
