"""The mesh datasets enter as specs: `cactus_spec` and `winther_spec` emit a validated SubstrateSpec that
records every decision the loaders used to make, and `walk_spec` walks it into a pack."""
import os

import numpy as np
import pytest

from dmipy_sim.spec import cactus_spec, winther_spec, walk_spec, SpecError
from dmipy_sim.replay.bank import build_replay_pack

trimesh = pytest.importorskip("trimesh")


def _tube_pair(dir_path, g=0.62, r_out=2.0, height=20.0, prefix="tube"):
    """Concentric closed tubes with a known g-ratio, written as PLY (units: whatever the caller says)."""
    paths = []
    for radius in (g * r_out, r_out):
        m = trimesh.creation.cylinder(radius=radius, height=height, sections=64)
        p = os.path.join(str(dir_path), f"{prefix}_r{radius:.3f}.ply")
        m.export(p)
        paths.append(p)
    return paths[0], paths[1]


def _cactus_run_dir(tmp_path, n_strands=3, side=30.0, g=0.7):
    """A minimal CACTUS run directory: the strand-list header plus per-strand inner/outer PLYs."""
    run = tmp_path / "cactus_run"
    sim = run / "meshes" / "simulations"
    sim.mkdir(parents=True)
    for i in range(n_strands):
        for tag, radius in (("inner", g * 1.5), ("outer", 1.5)):
            m = trimesh.creation.cylinder(radius=radius, height=side, sections=48)
            m.apply_translation([3.0 * i - 3.0, 0.0, 0.0])
            m.export(sim / f"strand_{i:05d}_{tag}_erode_0.ply")
    lines = [f"{side}", f"{n_strands}"]
    for i in range(n_strands):
        lines += ["2", f"{3.0*i-3.0} 0 {-side/2} 1.5", f"{3.0*i-3.0} 0 {side/2} 1.5"]
    (run / "optimized_final.txt").write_text("\n".join(lines) + "\n")
    return str(run)

ENV = dict(bvals=[0.0, 1e9], dirs=[[0, 0, 1]], ogse_periods=[2], shortd_b=1e9, shortd_deltas_frac=[0.05],
           B0_list=[3.0], theta_deg=[90], delta_frac=0.2, Delta_frac=0.5, rho_list=[1e-5])


def test_cactus_run_directory_becomes_a_spec_that_says_everything(tmp_path):
    run = _cactus_run_dir(tmp_path, n_strands=3, side=30.0, g=0.7)
    spec = cactus_spec(run)
    assert spec.domain.boundary == ["periodic"] * 3 and spec.domain.box_max[0] - spec.domain.box_min[0] == pytest.approx(30e-6)
    assert [p.name for p in spec.pools] == ["extra", "intra", "myelin"] and spec.field_source_pools[0].name == "myelin"
    assert len(spec.walls) == 6 and {(w.inside_pool, w.outside_pool) for w in spec.walls} == {(1, 2), (2, 0)}
    assert all(w.surface.kind == "mesh" and w.surface.scale == 1e-6 and len(w.surface.sha256) == 64 for w in spec.walls)
    assert spec.seeding.pools == [0, 1, 2] and spec.seeding.weights == "thin"
    assert spec.validity.smallest_feature == pytest.approx(0.7 * 1.5e-6, rel=0.05)      # the lumen radius
    assert any("optimized_final.txt" in t for t in spec.provenance["transformations"])
    assert spec.pool("intra").T2 is not None and spec.realisation["n_objects"] == 3
    again = type(spec).from_json(spec.to_json())
    assert again == spec


def test_a_strand_without_its_inner_surface_is_dropped_and_recorded(tmp_path):
    import os, glob
    run = _cactus_run_dir(tmp_path, n_strands=3)
    os.remove(glob.glob(os.path.join(run, "meshes", "simulations", "strand_00001_inner_*.ply"))[0])
    spec = cactus_spec(run)
    assert len(spec.walls) == 4 and any("lacking an inner or outer" in t for t in spec.provenance["transformations"])


def test_cactus_spec_walks_into_a_pack(tmp_path):
    run = _cactus_run_dir(tmp_path, n_strands=2, side=12.0, g=0.7)
    spec = cactus_spec(run)
    walk = walk_spec(spec, 150, 6e-4, 2e-4, seed=0, n_probe=10_000, require_gpu=False)   # the raster at the sheath's own spacing
    ids = np.asarray(walk.compartment)[:, 0]
    assert set(np.unique(ids)) == {0, 1, 2} and walk.spec is spec
    pk = build_replay_pack(walk, id="test/cactus", license="x", citation="x", K=8, envelope=ENV)
    assert pk.has_relaxation and pk.has_surface and pk.has_field and pk.meta["substrate"]["id"] == spec.id


def test_winther_axon_is_an_open_box_with_free_water_outside(tmp_path):
    inner, outer = _tube_pair(tmp_path, g=0.62, r_out=2.0, height=20.0)
    spec = winther_spec(inner, outer, scale=1e-6, pad=1e-6)
    assert spec.domain.boundary == ["open"] * 3
    assert spec.domain.box_max[0] - spec.domain.box_min[0] == pytest.approx(6e-6, rel=1e-3)
    assert spec.pool("extra").water_fraction == 0.0 and spec.seeding.pools == [1, 2]
    assert len(spec.walls) == 2 and spec.wall("fibre-0/inner").surface_relaxivity.inside > 0
    with pytest.raises(SpecError, match="no paired strand"):
        cactus_spec(str(tmp_path / "nothing_here"), side_um=10.0)


@pytest.mark.parametrize("g", [0.45, 0.62, 0.80])
def test_the_g_ratio_is_measured_from_the_surfaces_not_assumed(tmp_path, g):
    """For a tube V ~ r^2 L, so g = sqrt(V_in / V_out); the spec records what the meshes realise (the Winther
    axons realise 0.70, CACTUS strands 0.56-0.87), never a default."""
    from dmipy_sim.spec import winther_spec, cactus_spec
    inner, outer = _tube_pair(tmp_path, g=g)
    assert winther_spec(inner, outer, scale=1e-6, pad=1e-6).realisation["g_ratio"] == pytest.approx(g, abs=0.015)
    run = _cactus_run_dir(tmp_path, g=g)
    gr = cactus_spec(run).realisation["g_ratio"]
    assert set(gr) == {"0", "1", "2"} and all(v == pytest.approx(g, abs=0.015) for v in gr.values())


def test_producers_record_the_calibration_field_and_the_datasets_chi_convention(tmp_path):
    from dmipy_sim.spec import winther_spec, cactus_spec
    inner, outer = _tube_pair(tmp_path, g=0.7)
    w = winther_spec(inner, outer, scale=1e-6, pad=1e-6, field_T=7.0)
    assert w.nominal_field_T == 7.0 and w.pool("myelin").susceptibility.chi_iso == 1.06e-6      # Winther's own convention
    c = cactus_spec(_cactus_run_dir(tmp_path))
    assert c.nominal_field_T == 3.0 and c.pool("myelin").susceptibility.chi_iso == -0.1e-6     # the catalogue's


def _pinched_tube(dir_path, r_out=2.0, height=20.0):
    """The Winther case (dmrai-lab/dmipy-sim#452): a closed inner surface with one degenerate face and one edge
    shared by four faces (two closed sheets welded along it: no boundary anywhere), and its closed outer tube."""
    inner, outer = _tube_pair(dir_path, g=0.62, r_out=r_out, height=height, prefix="pinch")
    m = trimesh.load(inner, process=False)
    V, F = np.asarray(m.vertices), np.asarray(m.faces)
    a, b, c = F[0]
    V2 = V + np.array([0.0, 0.0, 2.0 * height]); F2 = F + len(V)
    V2[a] = V[a]; V2[b] = V[b]                                    # the copy's a, b sit exactly on the original's: duplicates
    F_all = np.vstack([F, F2, [a, b, a]])                          # and one face with a repeated vertex, as the writer left it
    trimesh.Trimesh(np.vstack([V, V2]), F_all, process=False).export(inner)
    return inner, outer


def test_a_pinched_closed_surface_is_accepted_and_the_pinch_recorded(tmp_path):
    inner, outer = _pinched_tube(tmp_path)
    with pytest.warns(UserWarning, match="degenerate face"):
        spec = winther_spec(inner, outer, scale=1e-6, pad=1e-6)
    notes = [t for t in spec.provenance["transformations"] if "three or more faces" in t]
    assert len(notes) == 1 and "1 edge(s)" in notes[0] and "1 degenerate face(s)" in notes[0] and "2 duplicate vertex" in notes[0]


def test_an_open_surface_is_still_refused(tmp_path):
    inner, outer = _tube_pair(tmp_path, prefix="open")
    m = trimesh.load(inner, process=False)
    trimesh.Trimesh(np.asarray(m.vertices), np.asarray(m.faces)[1:], process=False).export(inner)     # one face removed
    with pytest.raises(SpecError, match="closed surfaces; open"):
        winther_spec(inner, outer, scale=1e-6, pad=1e-6)


def test_a_crack_of_near_duplicate_vertices_is_closed_by_the_loader(tmp_path):
    """axon08's case (dmrai-lab/dmipy-sim#452): a vertex written twice a few nanometres apart, one face on each copy,
    leaves two boundary edges; the loader merges the crack's lips and the surface is closed."""
    from dmipy_sim.geometry.mesh import load_ply, surface_topology
    inner, outer = _tube_pair(tmp_path, prefix="crack")
    m = trimesh.load(inner, process=False)
    V, F = np.asarray(m.vertices), np.asarray(m.faces).copy()
    a = F[0, 0]
    edge = float(np.median(np.linalg.norm(V[F[:, 0]] - V[F[:, 1]], axis=1)))
    V2 = np.vstack([V, V[a] + 0.1 * edge * np.array([1.0, 0.0, 0.0])])       # the copy, a tenth of an edge away
    F[0, 0] = len(V)                                                            # one face moves to the copy: a crack
    trimesh.Trimesh(V2, F, process=False).export(inner)
    raw = trimesh.load(inner, process=False)
    assert surface_topology(np.asarray(raw.vertices), np.asarray(raw.faces))["boundary_edges"] == 2
    with pytest.warns(UserWarning, match="1 duplicate vertex"):
        Vr, Fr = load_ply(inner, scale=1e-6)
    assert surface_topology(Vr, Fr)["boundary_edges"] == 0 and len(Vr) == len(V)
    winther_spec(inner, outer, scale=1e-6, pad=1e-6)                            # accepted
