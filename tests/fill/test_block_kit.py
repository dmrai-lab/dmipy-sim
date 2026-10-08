"""dmipy-sim#695: the block kit (``Recipe.block_kit``) -- what a node downloads for ONE block, constant in the
substrate's size rather than the whole-substrate kit's (``tests/fill/test_kit.py``). The layout and the window
arithmetic (containment, the dense renumbering's bijection) are checked here, on the CPU, against the three-
strand DiSCo-like fixture; the identity check against an actual CUDA walk is ``test_block_kit_cuda.py`` (the
L40S bench venv)."""
import json

import numpy as np

from dmipy_sim.engine.adaptive import adaptive_plan
from dmipy_sim.engine.backends import cell_window, csr_cells, window_cell_table
from dmipy_sim.fill import Recipe
from dmipy_sim.fill.kit import _block_box, _pool_geometry, _reach_margin, _sha256


def test_block_kit_layout_sha256_and_sections(field_fake, tmp_path):
    """One request per diffusing pool with its cropped strand tables, the field's tables cropped the same way
    and a far crop for the block's box, the block's own single plan row, the manifest, and ``kit.json`` naming
    the block and every file's sha256 -- the whole kit's layout (``test_kit.py``), per block."""
    hub, _ = field_fake
    rc = Recipe(hub)
    out = rc.block_kit(0, tmp_path / "bkit0")
    kit = json.loads((out / "kit.json").read_text())
    assert kit["block"] == 0 and kit["recipe_id"] == "test/fill" and kit["commit"] == "test" and kit["variant"] == "t"
    assert kit["dmipy_sim_version"]
    assert kit["field"]["present"] is True
    assert set(kit["pools"]) == {"intra", "extra"}
    for pool in ("intra", "extra"):
        pdir = out / "pools" / pool
        for name in ("request.txt", "request.json", "A.f32", "AB.f32", "AB2.f32", "rr.f32", "tube.i32", "cell_off.i32", "cell_ids.i32"):
            assert (pdir / name).is_file(), (pool, name)
    for name in ("fA.f32", "fAB.f32", "fAB2.f32", "fa.f32", "fb.f32", "fsid.i32", "fcell_off.i32", "fcell_ids.i32",
                "fbox.i16", "far_full.u16", "far_full.json"):
        assert (out / name).is_file(), name
    blocks = json.loads((out / "plan" / "blocks.json").read_text())["blocks"]
    assert len(blocks) == 1 and blocks[0]["block"] == 0
    # the row and the manifest's grid are re-based to the crop's own LOCAL index frame (review point 2):
    # i/j/k start at 0, and the counts array is cropped to exactly that local shape
    assert blocks[0]["i"][0] == 0 and blocks[0]["j"][0] == 0 and blocks[0]["k"][0] == 0
    man_local = json.loads((out / "manifest.json").read_text())
    assert man_local["grid"]["shape"] == [blocks[0]["i"][1], blocks[0]["j"][1], blocks[0]["k"][1]]
    counts = np.load(out / "plan" / "counts.npz")
    for k in counts.files:
        assert list(counts[k].shape) == man_local["grid"]["shape"]
    assert (out / "plan" / "counts.npz").is_file()
    assert (out / "manifest.json").is_file()
    assert "kit.json" not in kit["files"]
    for rel, sha in kit["files"].items():
        assert _sha256(out / rel) == sha, rel


def test_block_kit_without_a_field_variant(fake, tmp_path):
    """The field-off recipe's block kit has no field tables and no far crop, but still one request per pool."""
    hub, _ = fake
    rc = Recipe(hub)
    out = rc.block_kit(0, tmp_path / "bkit0")
    kit = json.loads((out / "kit.json").read_text())
    assert kit["field"]["present"] is False
    assert not (out / "far_full.u16").exists()
    assert "field" not in (out / "pools" / "intra" / "request.json").read_text()


def test_block_kit_is_smaller_than_the_whole_kit(field_fake, tmp_path):
    """The whole point (dmrai-lab/dmipy-sim#695): a block's own window carries far fewer segments than the
    substrate's whole table, and a far crop no bigger than the whole kit's uncropped grid."""
    hub, _ = field_fake
    rc = Recipe(hub)
    whole = rc.kit(tmp_path / "whole")
    block = rc.block_kit(0, tmp_path / "bkit0")
    for pool in ("intra", "extra"):
        n_whole = json.loads((whole / "pools" / pool / "request.json").read_text())["n_seg"]
        n_block = json.loads((block / "pools" / pool / "request.json").read_text())["n_seg"]
        assert n_block <= n_whole
    assert (block / "far_full.u16").stat().st_size <= (whole / "far_full.u16").stat().st_size


def _touched_cells(gmin, dims, cs, lo_m, hi_m, margin_m):
    """Brute force: every cell index whose physical ``[gmin + i*cs, gmin + (i+1)*cs)`` box overlaps
    ``[lo_m - margin_m, hi_m + margin_m]`` -- the ground truth :func:`cell_window` must contain."""
    out = []
    for ix in range(int(dims[0])):
        for iy in range(int(dims[1])):
            for iz in range(int(dims[2])):
                c_lo = gmin + np.array([ix, iy, iz], float) * cs
                c_hi = c_lo + cs
                if np.all(c_hi > lo_m - margin_m) and np.all(c_lo < hi_m + margin_m):
                    out.append((ix, iy, iz))
    return out


def test_block_kit_window_contains_box_plus_reach_and_renumbering_is_a_bijection(field_fake):
    """Per pool: the cell window (:func:`~dmipy_sim.engine.backends.cell_window`) contains every cell the
    block's box plus the walk's reach could touch (checked by brute force over the geometry's own grid,
    :func:`_touched_cells`), and the dense renumbering (:func:`~dmipy_sim.engine.backends.window_cell_table`)
    is a bijection on the segments the window's cells actually reference: ``uniq`` has no duplicate, and
    re-deriving the window's CSR table straight from the SAME cropped cells (:func:`csr_cells`) and mapping it
    through ``uniq`` reproduces ``cell_ids`` exactly."""
    hub, _ = field_fake
    rc = Recipe(hub)
    ctx = rc.context(); g = ctx.tests
    row = next(r for r in rc.table if r["block"] == 0)
    lo_m, hi_m = _block_box(rc, row)

    checked = 0
    for pid, pool in g.pools.items():
        if pool.D in (None, 0.0) or (bool(g.inside_w[pid]) and bool(g.outside_w[pid])):
            continue
        geom = _pool_geometry(ctx, pid)
        D = float(pool.D)
        plan = adaptive_plan(geom, D, 2e-4)
        T_max = float(rc.man["walk"]["T_max_s"])
        margin = _reach_margin(plan, geom.cell_size, D, T_max)

        gmin, dims, cs = np.asarray(geom._GMIN), np.asarray(geom._DIMS), float(geom._CS)
        win_lo, win_hi = cell_window(gmin, cs, dims, lo_m, hi_m, margin)

        for (ix, iy, iz) in _touched_cells(gmin, dims, cs, lo_m, hi_m, margin):
            assert win_lo[0] <= ix < win_hi[0] and win_lo[1] <= iy < win_hi[1] and win_lo[2] <= iz < win_hi[2], \
                (pool.name, (ix, iy, iz), win_lo, win_hi)

        cell_off, cell_ids, uniq, window_dims = window_cell_table(np.asarray(geom._CELL), dims, win_lo, win_hi)
        assert tuple(window_dims) == tuple(int(x) for x in (win_hi - win_lo))
        assert len(np.unique(uniq)) == len(uniq)                 # a bijection: no id claimed twice
        sub = np.asarray(geom._CELL).reshape(tuple(int(x) for x in dims) + (-1,))
        sub = sub[win_lo[0]:win_hi[0], win_lo[1]:win_hi[1], win_lo[2]:win_hi[2], :]
        true_off, true_ids = csr_cells(np.ascontiguousarray(sub).reshape(-1, sub.shape[-1]))
        assert np.array_equal(cell_off, true_off)
        assert np.array_equal(uniq[cell_ids], true_ids)           # uniq[new] == old: the bijection's own check
        if uniq.size:
            assert uniq.min() >= 0 and uniq.max() < int(geom._A.shape[0])
        checked += 1
    assert checked == 2   # intra and extra both walked in this fixture
