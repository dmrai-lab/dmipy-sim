"""dmipy-sim#695: the test that matters -- a batch of a block walked with the CUDA backend (the standalone
``dmipy-walk`` binary) from a request assembled from the BLOCK kit is bit-identical to the same batch walked
from the WHOLE kit's tables: positions, local time, compartment and field samples, compared with
``np.array_equal``. The kernel's 27-cell gather and the field list only ever read the window
(``dmipy_sim.engine.backends.cell_window``/``window_cell_table``), so renumbering it is invisible to them --
this is the proof. Skipped unless ``dmipy_sim_cuda`` is importable and built (the L40S bench venv)."""
import pathlib
import subprocess

import numpy as np
import pytest

pytest.importorskip("dmipy_sim_cuda")
import dmipy_sim_cuda  # noqa: E402
from dmipy_sim_cuda.cli import read_walk  # noqa: E402

from dmipy_sim.fill import Recipe  # noqa: E402
from dmipy_sim.fill.kit import _block_box, assemble_request  # noqa: E402

BIN = pathlib.Path(dmipy_sim_cuda.__file__).resolve().parents[1] / "csrc" / "bin" / "dmipy-walk"


def _starts_in_box(ctx, pid, n, seed, lo_m, hi_m):
    """``n`` valid start positions of pool ``pid`` that also lie in the block's own box -- the fill's own
    seeding is voxel-stratified within a block; a plain oversample-and-mask draw is enough for this test."""
    sampler = ctx.tests.sampler(pid)
    rng = np.random.default_rng(seed)
    pts, ok = sampler(400 * n, rng)
    pts = pts[ok]
    inside = np.all((pts >= lo_m) & (pts < hi_m), axis=1)
    pts = pts[inside]
    assert len(pts) >= n, f"only {len(pts)} of {n} wanted starts landed in pool {pid} within the block's box"
    return np.ascontiguousarray(pts[:n], np.float32)


def _run(req_dir, out_dir):
    out_dir = pathlib.Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)       # dmipy-walk does not create it and does not check the open
    p = subprocess.run([str(BIN), str(req_dir / "request.txt"), str(req_dir / "r0.f32"), str(req_dir / "keys.u32"), str(out_dir)],
                       capture_output=True, text=True)
    assert p.returncode == 0, p.stderr[-2000:]
    return read_walk(out_dir)


@pytest.mark.parametrize("pool_name", ["intra", "extra"])
def test_block_kit_walk_matches_the_whole_kit(pool_name, field_fake, tmp_path):
    hub, _ = field_fake
    rc = Recipe(hub)
    ctx = rc.context(); g = ctx.tests
    pid = next(p.id for p in g.pools.values() if p.name == pool_name)
    row = next(r for r in rc.table if r["block"] == 0)
    lo_m, hi_m = _block_box(rc, row)

    whole_dir = rc.kit(tmp_path / "whole")
    block_dir = rc.block_kit(0, tmp_path / "block0")

    r0 = _starts_in_box(ctx, pid, 64, seed=42 + pid, lo_m=lo_m, hi_m=hi_m)
    seed = 123 + pid
    keys = np.random.default_rng(99 + pid).integers(0, 2 ** 31 - 1, size=(len(r0), 2)).astype(np.uint32)

    whole_req = assemble_request(whole_dir, pool_name, r0, keys, seed, tmp_path / f"whole-req-{pool_name}")
    block_req = assemble_request(block_dir, pool_name, r0, keys, seed, tmp_path / f"block-req-{pool_name}")

    w_whole, _ = _run(whole_req, tmp_path / f"whole-out-{pool_name}")
    w_block, _ = _run(block_req, tmp_path / f"block-out-{pool_name}")

    assert np.array_equal(w_whole.positions, w_block.positions)
    assert np.array_equal(w_whole.boundary_local_time, w_block.boundary_local_time)
    assert np.array_equal(w_whole.compartment, w_block.compartment)
    assert (w_whole.field_samples is None) == (w_block.field_samples is None)
    if w_whole.field_samples is not None:
        assert np.array_equal(w_whole.field_samples, w_block.field_samples)
