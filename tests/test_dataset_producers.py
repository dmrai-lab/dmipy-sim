"""The dataset producers emit specs and construct nothing: CATERPillar tables (sphere unions, four pools), EPFL
strand lists (CACTUS / DiSCo, sphere-swept polylines). The specs validate, round-trip through JSON, walk pool by
pool through `walk_spec` and are embedded in the pack."""
import dataclasses
import os

import numpy as np
import pytest

import dmipy_sim as d
from dmipy_sim.io.caterpillar import read_caterpillar, write_caterpillar, points_inside_union
from dmipy_sim.io.strands import read_strands, write_strands
from dmipy_sim.replay.bank import build_replay_pack
from dmipy_sim.spec import (caterpillar_spec, strands_spec, disco_spec, walk_spec, geometry_from_spec, load_spec,
                            SpecError, Seeding)

ENV = dict(bvals=[0.0, 1e9], dirs=[[1, 0, 0]], ogse_periods=[2], shortd_b=1e9, shortd_deltas_frac=[0.05], B0_list=[],
           theta_deg=[0], delta_frac=0.2, Delta_frac=0.5, rho_list=[1e-5])


@pytest.fixture(scope="module")
def caterpillar_csv(tmp_path_factory):
    """Two straight axon chains (one myelinated, one bare) and one glial soma in a 12 um voxel."""
    tmp = tmp_path_factory.mktemp("cat")
    z = np.arange(-8e-6, 8e-6, 0.25e-6)
    def chain(x, y, r_in, r_out, cid):
        c = np.stack([np.full_like(z, x), np.full_like(z, y), z], 1)
        return c, np.full(len(z), r_in), np.full(len(z), r_out), np.full(len(z), "axon"), np.full(len(z), cid), np.zeros(len(z), int)
    parts = [chain(-3e-6, 0, 1.0e-6, 1.4e-6, 0), chain(3e-6, 0, 1.2e-6, 1.2e-6, 1),
             (np.array([[0, 3.5e-6, 0]]), np.array([1.5e-6]), np.array([1.5e-6]), np.array(["glial_cell"]), np.array([0]), np.array([0]))]
    cat = [np.concatenate([p[i] for p in parts]) for i in range(6)]
    csv = str(tmp / "vox.csv")
    write_caterpillar(csv, *cat)
    with open(str(tmp / "vox_growth_info.txt"), "w") as fh:
        fh.write("Small voxel min limits -6 -6 -6\nSmall voxel max limits 6 6 6\nAxon icvf 0.2\n")
    return csv


def test_caterpillar_table_is_read_by_header_name_in_metres(caterpillar_csv):
    t = read_caterpillar(caterpillar_csv)
    assert t["centers"].shape == (129, 3) and t["r_out"].max() == pytest.approx(1.5e-6)
    assert set(t["cell_type"]) == {"axon", "glial_cell"} and t["box_source"].startswith("growth_info")
    np.testing.assert_allclose(t["box_max"], 6e-6)
    ax = t["cell_type"] == "axon"
    p = np.array([[-3e-6, 0, 0], [3e-6, 0, 0], [0, 3.5e-6, 0], [0, 0, 0]])
    assert points_inside_union(t["centers"][ax], t["r_in"][ax], p).tolist() == [True, True, False, False]
    assert read_caterpillar(caterpillar_csv, cell_types=("axon",))["centers"].shape == (128, 3)


def test_caterpillar_spec_declares_four_pools_three_walls_and_a_reflecting_voxel(caterpillar_csv, tmp_path):
    spec = caterpillar_spec(caterpillar_csv, id="test/cat")
    assert [p.name for p in spec.pools] == ["extra", "intra", "myelin", "glia"]
    assert [(w.name, w.inside_pool, w.outside_pool) for w in spec.walls] == [("axolemma", 1, 2), ("sheath", 2, 0), ("glia", 3, 0)]
    assert all(w.surface.kind == "sphere_union" and w.surface.file == caterpillar_csv for w in spec.walls)
    assert spec.walls[0].surface.column == "inner_radius" and spec.walls[2].surface.cell_type == "glial_cell"
    assert spec.domain.boundary == ["reflect"] * 3 and spec.domain.box_max == [6e-6] * 3
    assert spec.seeding.pools == [0, 1, 2, 3] and spec.validity.smallest_feature == pytest.approx(1.0e-6)
    assert spec.pools[2].susceptibility.chi_iso is not None                      # the dataset producer knows chi
    spec.save(tmp_path / "cat.sub.json")
    assert load_spec(tmp_path / "cat.sub.json") == spec
    with pytest.raises(SpecError, match="pool by pool"):
        geometry_from_spec(spec)
    assert [p.name for p in caterpillar_spec(caterpillar_csv, glia=False).pools] == ["extra", "intra", "myelin"]


def test_caterpillar_spec_is_walked_pool_by_pool_and_packed(caterpillar_csv):
    spec = caterpillar_spec(caterpillar_csv, id="test/cat")
    w = walk_spec(spec, 160, 1e-3, 2.5e-4, seed=0, n_probe=20_000, field_res=0.5e-6, require_gpu=False)
    ids = np.asarray(w.compartment)[:, 0]; pos = np.asarray(w.positions)
    assert set(np.unique(ids)) == {0, 1, 2, 3} and w.spec is spec and w.weights is not None
    r_ax = np.hypot(pos[:, :, 0] - np.where(pos[:, :, 0] < 0, -3e-6, 3e-6), pos[:, :, 1])
    assert (r_ax[ids == 1] < 1.2e-6 + 2e-9).all()                                  # intra confined to its axon
    assert (np.abs(pos) <= 6e-6 + 2e-9).all()                                      # nobody left the voxel
    assert (pos[ids == 2] == pos[ids == 2][:, :1]).all()                           # myelin frozen
    assert (np.linalg.norm(pos[ids == 3] - np.array([0, 3.5e-6, 0]), axis=-1) < 1.5e-6 + 2e-9).all()   # glia in its soma
    assert w.field_grid is not None and w.diffusivity == spec.pool("intra").D
    pk = build_replay_pack(w, id="t/cat", license="x", citation="x", K=3, envelope=ENV)
    assert pk.has_relaxation and pk.has_surface and pk.has_field and pk.substrate == spec


@pytest.fixture(scope="module")
def strand_txt(tmp_path_factory):
    tmp = tmp_path_factory.mktemp("strands")
    cls_ = [np.array([[x, 0, -12e-6], [x, 0.5e-6, 0], [x, 0, 12e-6]]) for x in (-5e-6, 0, 5e-6)]
    path = str(tmp / "optimized_final.txt")
    write_strands(path, cls_, [1.5e-6, 1.0e-6, 2.0e-6], 20e-6)
    return path


def test_strand_list_round_trips_in_metres(strand_txt):
    t = read_strands(strand_txt)
    assert t["side"] == pytest.approx(20e-6) and t["n_strands"] == 3 and len(t["centerlines"][0]) == 3
    assert t["radii"][2][0] == pytest.approx(2.0e-6)
    with open(strand_txt) as fh:
        assert fh.readline().strip() == "20" and fh.readline().strip() == "3"        # micrometres in the file


def test_strands_spec_is_one_wall_of_swept_polylines_in_a_reflecting_voxel(strand_txt):
    spec = strands_spec(strand_txt, id="test/strands")
    assert [w.name for w in spec.walls] == ["cylinders"] and spec.walls[0].surface.kind == "swept_polyline"
    assert spec.walls[0].surface.instances["radii"] == pytest.approx([1.5e-6, 1.0e-6, 2.0e-6])
    assert spec.domain.box_min == pytest.approx([-10e-6] * 3) and spec.domain.boundary == ["reflect"] * 3
    assert spec.seeding.pools == [0, 1] and spec.validity.smallest_feature == pytest.approx(1.0e-6)
    assert spec.validity.tiers == ["gradient", "relaxation"]           # no surface, no field: the engine walks neither yet
    # one seeded pool is one geometry, each side of the wall
    g_e = geometry_from_spec(dataclasses.replace(spec, seeding=Seeding([0])))
    g_i = geometry_from_spec(dataclasses.replace(spec, seeding=Seeding([1])))
    assert type(g_e) is d.PackedCurvedCylinders and not g_e.interior and g_e.box is not None and g_i.interior
    w = walk_spec(spec, 120, 1e-3, 2.5e-4, seed=0, n_probe=20_000, require_gpu=False)
    ids = np.asarray(w.compartment)[:, 0]; pos = np.asarray(w.positions)
    assert set(np.unique(ids)) == {0, 1} and (np.abs(pos) <= 10e-6 + 2e-9).all()
    assert not g_e.inside_any(pos[ids == 0].reshape(-1, 3)).any()                  # extra never enters a strand
    with pytest.raises(SpecError, match="not periodic"):
        strands_spec(strand_txt, boundary="periodic")


def test_disco_spec_adds_the_sheath_at_the_phantoms_g_ratio(strand_txt):
    spec = disco_spec(strand_txt)
    assert spec.id == "disco/optimized_final" and [w.name for w in spec.walls] == ["axolemma", "sheath"]
    assert spec.walls[0].surface.instances["radii"] == pytest.approx([0.7 * r for r in (1.5e-6, 1.0e-6, 2.0e-6)])
    assert [p.name for p in spec.pools] == ["extra", "intra", "myelin"] and spec.validity.tiers == ["gradient", "relaxation"]
    w = walk_spec(spec, 90, 8e-4, 2e-4, seed=0, n_probe=20_000, field_res=0.5e-6, require_gpu=False, field=False)
    ids = np.asarray(w.compartment)[:, 0]
    assert set(np.unique(ids)) == {0, 1, 2} and w.field_grid is None            # no field route for strands yet (#76 item 3)


def test_a_strand_whose_radius_varies_is_refused(tmp_path):
    path = str(tmp_path / "beaded.txt")
    write_strands(path, [np.array([[0, 0, -5e-6], [0, 0, 0], [0, 0, 5e-6]])], [np.array([1e-6, 1.5e-6, 1e-6])], 20e-6)
    with pytest.raises(SpecError, match="varies along its length"):
        strands_spec(path)
    assert strands_spec(path, radius_tol=1.0).validity.smallest_feature > 0


def test_a_strand_pack_claims_no_tier_the_walk_did_not_record(strand_txt):
    """The curved tubes accumulate no boundary local time: the walk carries no surface channel, the pack claims no
    C2, a replay at rho is refused rather than returned unattenuated; and the field is refused at the walk."""
    from dmipy_sim.replay.bank import build_replay_pack
    spec = disco_spec(strand_txt)
    assert spec.validity.tiers == ["gradient", "relaxation"]
    w = walk_spec(spec, 120, 1e-3, 2.5e-4, seed=0, n_probe=20_000, require_gpu=False, field=False)
    assert w.boundary_local_time is None and not w.has_surface and w.has_compartments
    pk = build_replay_pack(w, id="t/strands", license="x", citation="x", K=4, field=False)
    assert pk.has_relaxation and not pk.has_surface and not pk.meta["replay_envelope"]["surface_relaxivity"]
    seq = d.set_b(d.pgse([[1, 0, 0]], 0.2e-3, 0.5e-3, gradient_strengths=0.1, n_t=pk.n_t, slew_rate=np.inf), [1e9])
    with pytest.raises(ValueError, match="no C2"):
        pk.replay(seq, rho=1e-5)
    with pytest.raises(SpecError, match="not implemented"):
        walk_spec(spec, 60, 1e-3, 2.5e-4, seed=0, n_probe=20_000, require_gpu=False, field=True)
