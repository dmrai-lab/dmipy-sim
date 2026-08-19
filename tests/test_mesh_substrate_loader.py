"""One loader for inner/outer mesh substrates: an isolated axon and a packed bundle differ in count and box.

Two loaders drifted apart (dmipy-sim#37): into different containment predicates (#23, #33) and, in the
public port, into different volume-fraction arithmetic. The Winther path built its `trimesh` on
already-scaled vertices and then applied ``scale**3`` a second time, so at its own default ``scale=1e-6``
every fraction collapsed by 1e-18 -- ``f_intra`` 0.1218 -> 1.2e-19, ``f_extra`` -> 1.0. The only test that
touched the arithmetic ran at ``scale=1.0``, where a double unit conversion is the identity, so nothing
caught it. Hence the first test here.
"""
from __future__ import annotations

import os

import numpy as np
import pytest

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


def test_volume_fractions_are_scale_invariant(tmp_path):
    """A volume fraction is dimensionless, so changing mesh units must not change it.

    This is the regression guard for the double unit conversion described in the module docstring: it fails
    by 18 orders of magnitude on the old Winther path, and cannot be seen at scale=1.0.
    """
    from dmipy_sim.io.winther import load_winther_bundle

    inner, outer = _tube_pair(tmp_path, g=0.62)
    at_unit = load_winther_bundle(inner, outer, scale=1.0, pad=1.0)
    at_si = load_winther_bundle(inner, outer, scale=1e-6, pad=1e-6)

    for name in ("f_intra", "f_myelin", "f_extra", "g_ratio"):
        a, b = getattr(at_unit, name), getattr(at_si, name)
        assert b == pytest.approx(a, rel=1e-6), (
            f"{name} changed with mesh units: {a:.6g} at scale=1 vs {b:.6g} at scale=1e-6")
    assert at_si.f_intra > 0.01, f"f_intra collapsed to {at_si.f_intra:.3g} at SI scale"


def test_the_box_is_in_metres_at_si_scale(tmp_path):
    """The padded box must follow the scaled geometry, not the raw mesh units."""
    from dmipy_sim.io.winther import load_winther_bundle

    inner, outer = _tube_pair(tmp_path)
    b = load_winther_bundle(inner, outer, scale=1e-6, pad=1e-6)
    # r_out=2 mesh units -> 2 um, plus 1 um pad either side
    assert b.box_side[0] == pytest.approx(6e-6, rel=1e-6), b.box_side
    assert b.fibre_axis == 2, "the 20-unit-long tube axis should be the elongated one"


def test_an_isolated_axon_has_no_extra_substrate(tmp_path):
    """An isolated axon sits in free water; a bundle's extra-axonal space is substrate.

    Kept on the loaded substrate so the pool structure travels with the geometry that determines it, rather
    than being re-decided by every caller at walk time (dmipy-sim#37).
    """
    from dmipy_sim.io.winther import load_winther_bundle

    inner, outer = _tube_pair(tmp_path)
    assert load_winther_bundle(inner, outer, scale=1.0, pad=1.0).has_extra_substrate is False


# ---------------------------------------------------------------------------
# CACTUS run directory — the path public did not have at all
# ---------------------------------------------------------------------------

def _cactus_run_dir(tmp_path, n_strands=3, side=30.0, g=0.7):
    """A minimal CACTUS run directory: the config header plus per-strand inner/outer PLYs."""
    run = tmp_path / "cactus_run"
    sim = run / "meshes" / "simulations"
    sim.mkdir(parents=True)
    for i in range(n_strands):
        for tag, radius in (("inner", g * 1.5), ("outer", 1.5)):
            m = trimesh.creation.cylinder(radius=radius, height=side, sections=48)
            m.apply_translation([3.0 * i - 3.0, 0.0, 0.0])
            m.export(sim / f"strand_{i:05d}_{tag}_erode_0.ply")
    # header: side / n_fibres / n_ctrl, then one block per fibre (count, then control points)
    lines = [f"{side}", f"{n_strands}", "2"]
    for i in range(n_strands):
        lines += ["2", f"{3.0*i-3.0} 0 {-side/2}", f"{3.0*i-3.0} 0 {side/2}"]
    (run / "optimized_final.txt").write_text("\n".join(lines) + "\n")
    return str(run)


def test_public_can_load_a_cactus_run_directory(tmp_path):
    """The point of dmipy-sim#37: public shipped the container with no way to fill it."""
    from dmipy_sim.io.cactus import load_cactus_bundle

    run = _cactus_run_dir(tmp_path, n_strands=3, side=30.0, g=0.7)
    b = load_cactus_bundle(run)

    assert b.n_fibres == 3, b.summary()
    assert b.g_ratio == pytest.approx(0.7, abs=0.02), f"g measured {b.g_ratio:.4f}"
    assert b.has_extra_substrate is True, "a bundle's extra-axonal space is substrate"
    # periodic cell from the config header, in metres
    assert np.allclose(b.box_side, 30e-6), b.box_side
    assert b.inner[0].shape[1] == 3 and len(b.inner[1]) > 0
    # three non-overlapping fibres in a 30 um cell: a small but non-zero intra fraction
    assert 0.0 < b.f_intra < b.f_intra + b.f_myelin < 1.0, b.summary()
    assert b.fibre_tangents is not None and b.fibre_tangents.shape == (3, 3)
    assert np.allclose(np.abs(b.fibre_tangents[:, 2]), 1.0), "tangents should follow the +z fibre axis"


def test_a_strand_missing_its_inner_surface_is_skipped(tmp_path):
    """A fibre with an outer but no inner mesh would assign its whole interior to myelin."""
    from dmipy_sim.io.cactus import load_cactus_bundle

    run = _cactus_run_dir(tmp_path, n_strands=3)
    os.remove(os.path.join(run, "meshes", "simulations", "strand_00001_inner_erode_0.ply"))

    assert load_cactus_bundle(run).n_fibres == 2
    # ...unless the caller explicitly opts out, in which case the outer surfaces load alone
    assert load_cactus_bundle(run, require_pairs=False).n_fibres == 3


def test_both_datasets_go_through_one_loader(tmp_path):
    """Same code path, differing only in file count and box — the actual ask in dmipy-sim#37."""
    import dmipy_sim.io.cactus as cactus_mod
    import dmipy_sim.io.winther as winther_mod
    from dmipy_sim.io.mesh_substrate import load_mesh_substrate

    src = load_mesh_substrate.__module__
    assert cactus_mod.load_cactus_bundle.__doc__ and winther_mod.load_winther_bundle.__doc__
    # neither module may carry its own PLY reader any more
    assert not hasattr(winther_mod, "_load_ply"), "winther still has a private PLY reader"
    assert src == "dmipy_sim.io.mesh_substrate"


def test_an_explicit_box_is_honoured(tmp_path):
    """Reconstructing a substrate from stored metadata needs the box given outright."""
    from dmipy_sim.io.mesh_substrate import load_mesh_substrate

    inner, outer = _tube_pair(tmp_path)
    lo, hi = np.array([-5.0, -5.0, -15.0]), np.array([5.0, 5.0, 15.0])
    b = load_mesh_substrate([inner], [outer], box=(lo, hi), scale=1.0, volume_reference="box")
    assert np.allclose(b.box_min, lo) and np.allclose(b.box_max, hi)
    assert b.f_intra == pytest.approx(np.pi * (0.62 * 2.0) ** 2 * 20.0 / 3000.0, rel=0.02)
