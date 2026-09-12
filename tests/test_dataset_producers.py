"""The dataset producers emit specs and construct nothing: CATERPillar tables (sphere unions, four pools), EPFL
strand lists (CACTUS / DiSCo, sphere-swept polylines). The specs validate, round-trip through JSON, walk pool by
pool through `walk_spec` and are embedded in the pack."""
import dataclasses
import os

import numpy as np
import pytest

import dmipy_sim as d
from dmipy_sim.io.caterpillar import read_caterpillar, write_caterpillar, points_inside_union
from dmipy_sim.io.strands import read_strands, write_strands, read_tck, write_tck, read_diameters
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
    assert w.field_basis is not None and w.diffusivity == spec.pool("intra").D
    pk = build_replay_pack(w, id="t/cat", license="x", citation="x", K=3, envelope=ENV)
    assert pk.has_relaxation and pk.has_surface and pk.has_field and pk.substrate == spec


@pytest.fixture(scope="module")
def strand_txt(tmp_path_factory):
    tmp = tmp_path_factory.mktemp("strands")
    cls_ = [np.array([[x, 0, -12e-6], [x, 0.5e-6, 0], [x, 0, 12e-6]]) for x in (-5e-6, 0, 5e-6)]
    path = str(tmp / "optimized_final.txt")
    write_strands(path, cls_, [1.5e-6, 1.0e-6, 2.0e-6], 20e-6)
    return path


@pytest.fixture(scope="module")
def disco_files(tmp_path_factory):
    """The three strands of `strand_txt` in DiSCo's release form: a .tck in 25 um voxel units over [0, 20 um]^3 and
    the INNER diameters in mm (the outer radius is inner / 0.7)."""
    tmp = tmp_path_factory.mktemp("disco")
    cls_ = [np.array([[x, 0, -12e-6], [x, 0.5e-6, 0], [x, 0, 12e-6]]) + 10e-6 for x in (-5e-6, 0, 5e-6)]
    tck, dia = str(tmp / "DiSCo_Strands_Trajectories.tck"), str(tmp / "DiSCo_Strands_Diameters.txt")
    write_tck(tck, cls_, coordinate_unit_m=25e-6)
    np.savetxt(dia, np.array([2 * 0.7 * r for r in (1.5e-6, 1.0e-6, 2.0e-6)]) / 1e-3)
    return tck, dia


def test_the_track_file_round_trips_in_metres(disco_files):
    tck, dia = disco_files
    cls_ = read_tck(tck, coordinate_unit_m=25e-6)
    assert len(cls_) == 3 and np.allclose(cls_[1], np.array([[0, 0, -12e-6], [0, 0.5e-6, 0], [0, 0, 12e-6]]) + 10e-6, atol=1e-11)
    np.testing.assert_allclose(read_diameters(dia, diameter_unit_m=1e-3), [2 * 0.7 * r for r in (1.5e-6, 1.0e-6, 2.0e-6)], rtol=1e-12)
    nib = pytest.importorskip("nibabel")
    ref = nib.streamlines.load(tck).streamlines                                   # the reference reader agrees
    assert len(ref) == 3 and np.allclose(np.asarray(ref[2]) * 25e-6, cls_[2], atol=1e-11)


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
    assert spec.validity.tiers == ["gradient", "relaxation", "surface"]
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


def test_disco_spec_adds_the_sheath_at_the_phantoms_g_ratio(disco_files):
    spec = disco_spec(*disco_files, side_m=20e-6)
    assert spec.id == "disco/rafael-patino-2021" and [w.name for w in spec.walls] == ["axolemma", "sheath"]
    assert spec.walls[0].surface.instances["radii"] == pytest.approx([0.7 * r for r in (1.5e-6, 1.0e-6, 2.0e-6)])
    assert spec.walls[1].surface.instances["radii"] == pytest.approx([1.5e-6, 1.0e-6, 2.0e-6])        # inner / g
    assert spec.domain.box_min == [0.0] * 3 and spec.domain.box_max == [20e-6] * 3 and spec.domain.boundary == ["reflect"] * 3
    assert any("25e-06" in t or "2.5e-05" in t for t in spec.provenance["transformations"])
    assert [p.name for p in spec.pools] == ["extra", "intra", "myelin"] and spec.validity.tiers == ["gradient", "relaxation", "surface", "field"]
    w = walk_spec(spec, 90, 8e-4, 2e-4, seed=0, n_probe=20_000, field_res=0.5e-6, require_gpu=False)
    ids = np.asarray(w.compartment)[:, 0]
    assert set(np.unique(ids)) == {0, 1, 2} and w.field_basis is not None       # a small voxel rasterises its sheath


def test_a_strand_whose_radius_varies_is_refused(tmp_path):
    path = str(tmp_path / "beaded.txt")
    write_strands(path, [np.array([[0, 0, -5e-6], [0, 0, 0], [0, 0, 5e-6]])], [np.array([1e-6, 1.5e-6, 1e-6])], 20e-6)
    with pytest.raises(SpecError, match="varies along its length"):
        strands_spec(path)
    assert strands_spec(path, radius_tol=1.0).validity.smallest_feature > 0


def test_a_strand_pack_claims_what_its_walk_recorded(disco_files):
    """A strand spec declares the tiers the engine walks it for; the walk records contact (the curved tubes
    accumulate it), the pack claims C2 and replays rho; a domain too large to rasterise refuses the field."""
    from dmipy_sim.replay.bank import build_replay_pack
    spec = disco_spec(*disco_files, side_m=20e-6)
    assert spec.validity.tiers == ["gradient", "relaxation", "surface", "field"]
    w = walk_spec(spec, 120, 1e-3, 2.5e-4, seed=0, n_probe=20_000, require_gpu=False, field=False)
    assert w.has_surface and w.has_compartments
    pk = build_replay_pack(w, id="t/strands", license="x", citation="x", K=4, field=False)
    assert pk.has_relaxation and pk.has_surface and pk.meta["replay_envelope"]["surface_relaxivity"]
    seq = d.set_b(d.pgse([[1, 0, 0]], 0.2e-3, 0.5e-3, gradient_strengths=0.1, n_t=pk.n_t, slew_rate=np.inf), [1e9])
    assert pk.replay(seq, tissue=False, rho=1e-5, D=1.7e-9)[0] < pk.replay(seq, tissue=False)[0]   # contact attenuates
    with pytest.raises(SpecError, match="voxel budget"):                       # the raster of a domain too large is refused
        walk_spec(spec, 60, 1e-3, 2.5e-4, seed=0, n_probe=20_000, require_gpu=False, field="grid", field_budget=1e3)
    # the default field of a strand substrate is the per-segment closed form: no grid, a certified cutoff, a C3 pack
    from dmipy_sim.fields.strand_field import StrandFieldBasis
    wf = walk_spec(spec, 60, 1e-3, 2.5e-4, seed=0, n_probe=20_000, require_gpu=False, field=True, field_budget=1e3)
    assert isinstance(wf.field_basis, StrandFieldBasis) and wf.field_basis.certificate["converged"]
    pkf = build_replay_pack(wf, id="t/strands-field", license="x", citation="x", K=4, susc_path_K=4)
    assert pkf.has_field and pkf.meta["compression"]["channels"]["susceptibility_grid"]["source"]["kind"] == "strand_superposition"
    s_off = pkf.replay(seq, tissue=False)[0]
    s_gre = pkf.replay(d.gre(0.5e-3, gradient_directions=[[1, 0, 0]], bvalues=[1e9], delta=0.2e-3, Delta=0.3e-3, n_t=pk.n_t, slew_rate=np.inf),
                       tissue=False, B0=7.0, b0_dir=(1, 0, 0), chi_iso=-0.1e-6, chi_aniso=-0.1e-6)[0]
    assert np.isfinite(s_gre) and s_gre != s_off

def test_a_straight_myelinated_curved_tube_is_the_myelinated_cylinder():
    """The straight limit of the curved myelinated tube is the myelinated cylinder: the same pools, the same walls,
    and the same rasterised field basis -- the cross-check a per-segment field along a path is measured against.
    With the strand pack's own radial director the two bases agree to 1e-6 in every term at every B0; with the
    mask-gradient director the general route falls back on (meshes, sphere unions) the anisotropic term with B0
    across the axis is 28 % off (dmipy-sim#213), pinned at its measured size so a better director flips it."""
    from dmipy_sim.fields.susceptibility_field import field_grid_of, predicate_field_basis, assemble_field
    r_in, r_out = 1.0e-6, 1.4e-6
    cl = np.array([[0.0, 0.0, -6e-6], [0.0, 0.0, 6e-6]])
    straight = d.CurvedMyelinatedCylinder(cl, r_in, r_out, pool="intra")
    pack = d.PackedCurvedCylinders([cl], [r_in], interior=True)                 # the strand pack: its radial director
    cyl = d.MyelinatedCylinder(r_in, r_out, (0, 0, 1), 1.7e-9, 1.7e-9)
    lo, hi = np.array([-3e-6, -3e-6, -2e-6]), np.array([3e-6, 3e-6, 2e-6])
    res = 0.2e-6
    inner = lambda p: np.asarray(straight.classify_positions_exact(p)) == 1
    outer = lambda p: np.asarray(straight.classify_positions_exact(p)) != 0
    exact, _, _ = predicate_field_basis(inner, outer, lo, hi, res=res, mask_supersample=4, director=pack.radial_directors)
    grad, o_curved, _ = predicate_field_basis(inner, outer, lo, hi, res=res, mask_supersample=4)
    fg = field_grid_of(cyl, res=res, box=(lo, hi), mask_supersample=4)              # stored translation-invariant: a few slabs
    assert tuple(exact["shape"][:2]) == tuple(fg.basis["shape"][:2]) and np.allclose(o_curved[:2], np.asarray(fg.origin)[:2])

    def rms(basis, direction, chi_iso, chi_aniso):
        f1 = np.asarray(assemble_field(basis, direction, B0=3.0, chi_iso=chi_iso, chi_aniso=chi_aniso))[:, :, basis["shape"][2] // 2]
        f2 = np.asarray(assemble_field(fg.basis, direction, B0=3.0, chi_iso=chi_iso, chi_aniso=chi_aniso))[:, :, fg.basis["shape"][2] // 2]
        return np.sqrt(np.mean((f1 - f2) ** 2)) / np.sqrt(np.mean(f2 ** 2))
    for direction in ((0, 0, 1.0), (1.0, 0, 0)):
        assert rms(exact, direction, -1e-7, 0.0) < 1e-6 and rms(exact, direction, 0.0, -1e-7) < 1e-6 and rms(exact, direction, -1e-7, -1e-7) < 1e-6
    assert rms(grad, (0, 0, 1.0), 0.0, -1e-7) < 1e-6                                 # the gradient director along the axis
    across = rms(grad, (1.0, 0, 0), 0.0, -1e-7)
    assert 0.2 < across < 0.35, across                                             # and across it (#213)


def test_the_curved_tubes_record_surface_time_and_the_straight_limit_is_the_cylinder():
    """The curved tubes accumulate the wall contact (#76 item 2): a straight curved tube with a surface relaxivity
    attenuates like the analytic cylinder with the same rho, walked from the same seed; a strand walk carries the
    boundary channel, its pack claims C2, and a replay at rho attenuates."""
    from dmipy_sim.replay.bank import build_replay_pack
    R, rho, D = 2e-6, 20e-6, 2e-9
    cl = np.array([[0.0, 0.0, -30e-6], [0.0, 0.0, 30e-6]])
    # a PGSE at 0.1 mT/m: b ~ 0.1 s/m^2, exp(-bD) = 1 - 2e-10, so the signal is the wall relaxation alone (a zero-G
    # sequence has no echo to read)
    seq = d.pgse([[1.0, 0.0, 0.0]], 0.2e-3, 3.8e-3, gradient_strengths=1e-4, n_t=200, slew_rate=np.inf)
    T = seq.echo_idx * seq.dt
    s_cyl = float(np.asarray(d.simulate(20_000, D, seq, d.Cylinder(radius=R, orientation=(0, 0, 1), surface_relaxivity_t2=rho),
                                        seed=3, require_gpu=False))[0])
    s_cur = float(np.asarray(d.simulate(20_000, D, seq, d.CurvedCylinder(cl, R, surface_relaxivity_t2=rho),
                                        seed=3, require_gpu=False))[0])
    T2 = R / (2 * rho)                                                                  # Brownstein-Tarr, fast regime
    assert abs(s_cyl - np.exp(-T / T2)) < 0.01 and abs(s_cur - s_cyl) < 0.01, (s_cur, s_cyl, np.exp(-T / T2))
    s_pack = float(np.asarray(d.simulate(20_000, D, seq, d.PackedCurvedCylinders([cl], [R], interior=True, surface_relaxivity_t2=rho),
                                         seed=3, require_gpu=False))[0])
    assert abs(s_pack - s_cyl) < 0.01, (s_pack, s_cyl)
    # the strand walk records it and the pack replays it
    w = d.simulate_trajectories(300, D, d.PackedCurvedCylinders([cl], [R], interior=True), 2e-3, 2.5e-4, seed=0, require_gpu=False)
    assert w.has_surface and float(np.abs(np.asarray(w.boundary_local_time)).sum()) > 0
    pk = build_replay_pack(w, id="t/curved", license="x", citation="x", K=4, blt_temporal_K=4)
    assert pk.has_surface
    wf = d.pgse([[1.0, 0.0, 0.0]], 0.1e-3, 1.9e-3, gradient_strengths=1e-4, n_t=100, slew_rate=np.inf)
    assert pk.replay(wf, tissue=False, rho=rho, D=D)[0] < 0.99 * pk.replay(wf, tissue=False)[0]


def test_stratified_seeding_fills_every_occupied_voxel_and_keeps_the_volumes(disco_files):
    """Seeded per voxel: every voxel a pool occupies holds the asked count (or all a sliver allows), the weights
    make the per-voxel pool volumes right (against a rejection census), and the pack's per-voxel certificate
    reads back on the grid."""
    from dmipy_sim.spec import StratifiedByVoxel, plan_seeding
    from dmipy_sim.phantom import Grid
    from dmipy_sim.replay.bank import build_replay_pack, voxel_fidelity_volumes
    spec = disco_spec(*disco_files, side_m=20e-6)
    grid = Grid(shape=(4, 4, 4), voxel_size_m=(5e-6,) * 3, origin_m=(2.5e-6,) * 3)
    w = walk_spec(spec, T_max=8e-4, dt_save=2e-4, seed=0, require_gpu=False, field=False,
                  seeding=StratifiedByVoxel(grid=grid, walkers_per_voxel={"extra": 12, "intra": 8, "myelin": 3}))
    with pytest.raises(TypeError, match="not both"):
        walk_spec(spec, 50, 8e-4, 2e-4, seeding=StratifiedByVoxel(grid=grid, walkers_per_voxel=4))
    ids = np.asarray(w.compartment)[:, 0]; r0 = np.asarray(w.positions)[:, 0]; wt = np.asarray(w.weights)
    ijk, inside = grid.bin(r0); assert inside.all()
    flat = np.ravel_multi_index(tuple(ijk.T), grid.shape)
    n_extra = np.bincount(flat[ids == 0], minlength=64); n_intra = np.bincount(flat[ids == 1], minlength=64)
    assert (n_extra[n_extra > 0] == 12).all() and n_intra.max() == 8 and (n_intra > 0).sum() >= 6
    # a rejection census of each pool's volume per voxel, against the sum of the weights (f x water fraction)
    rng = np.random.default_rng(3); P = rng.uniform(0, 20e-6, (400_000, 3)); v = np.ravel_multi_index(tuple(grid.bin(P)[0].T), grid.shape)
    inner = d.PackedCurvedCylinders([np.asarray(c) for c in spec.walls[0].surface.instances["centerlines"]],
                                    spec.walls[0].surface.instances["radii"], interior=True)
    f_in = np.bincount(v, weights=inner.inside_any(P), minlength=64) / np.bincount(v, minlength=64)
    wsum_in = np.bincount(flat[ids == 1], weights=wt[ids == 1], minlength=64)
    wf_in = spec.pool("intra").water_fraction
    assert np.abs(wsum_in - f_in * wf_in).max() < 0.03 * wf_in, (wsum_in, f_in * wf_in)
    pk = build_replay_pack(w, id="t/strat", license="x", citation="x", K=4, voxel_grid=grid)
    pv = pk.meta["fidelity"]["per_voxel"]
    assert pv["n_voxels"] == 64 and pv["pools"] == [0, 1, 2] and pv["floor_max"] > 0 and pk.arrays["voxel_certificate"].shape == (64, 3, 3)
    g2, floors, counts = voxel_fidelity_volumes(pk)
    assert g2.shape == grid.shape and counts["extra"].sum() == (ids == 0).sum() and (floors["extra"][counts["extra"] > 1] > 0).all()
    plan = plan_seeding({"extra": floors["extra"]}, {"extra": counts["extra"]}, target_floor=floors["extra"].max() / 2, grid=grid)
    want = plan.count_for("extra")
    assert want.max() == 4 * 12 and want.min() >= 2                     # halving the floor quadruples the count
    with pytest.raises(ValueError, match="no per-voxel certificate"):
        voxel_fidelity_volumes(build_replay_pack(w, id="t/plain", license="x", citation="x", K=4))
    # the floor on the acquisition the pack is meant for (per shell), from the pack alone
    from dmipy_sim.replay.bank import voxel_floor
    seq = d.set_b(d.pgse(np.eye(3), 0.1e-3, 0.4e-3, gradient_strengths=[0.1] * 3, n_t=100, slew_rate=np.inf), [1e9, 1e9, 2e9])
    fl, cnt = voxel_floor(pk, grid, seq, shells={"b1": [0, 1], "b2": [2]}, tissue=False)
    assert set(fl) == {"extra", "intra", "myelin"} and cnt["extra"].sum() == (ids == 0).sum()
    assert fl["extra"]["b1"].shape == grid.shape and (fl["extra"]["b1"][cnt["extra"] > 1] > 0).all() and (fl["myelin"]["b2"] < 1e-6).all()
