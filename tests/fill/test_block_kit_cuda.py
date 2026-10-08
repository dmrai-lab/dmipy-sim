"""dmipy-sim#695: the test that matters -- a batch of a block walked with the CUDA backend (the standalone
``dmipy-walk`` binary) from a request assembled from the BLOCK kit is bit-identical to the same batch walked
from the WHOLE kit's tables: positions, local time, compartment and field samples, compared with
``np.array_equal``. The kernel's 27-cell gather and the field list only ever read the window
(``dmipy_sim.engine.backends.cell_window``/``window_cell_table``), so renumbering it is invisible to them --
this is the proof. Skipped unless ``dmipy_sim_cuda`` is importable and built (the L40S bench venv).

``test_block_kit_matches_the_whole_tables_at_production_scale`` repeats the proof at DiSCo's own scale (636,951
segments, 18k/2k walkers, 3349 saves), against a DUMPED request of the real block 0 walked from the WHOLE
tables (``r0.f32``/``keys.u32`` and all) -- skipped unless that dump and a block kit built at
``DMIPY_SIM_BLOCK_KIT_DIR`` (default ``~/tessera-work/disco/bkit0``) both exist on this machine; the margin
review (dmrai-lab/dmipy-sim#695: a walker is not confined to its block's box) could not be shown wrong on the
20 um toy fixture above, whose window was the whole domain regardless -- only production scale can."""
import os
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
#: a dumped request of the real DiSCo block 0, walked against the WHOLE tables (r0/keys/seed and all); the
#: production-scale proof below needs one per pool, named by its walker count the way the dump script left them
DISCO_DUMPS = {"extra": pathlib.Path(os.path.expanduser(os.environ.get("DMIPY_SIM_DISCO_DUMP_EXTRA",
                                                                       "~/tessera-work/disco/req/batch-18051-1791391378"))),
              "intra": pathlib.Path(os.path.expanduser(os.environ.get("DMIPY_SIM_DISCO_DUMP_INTRA",
                                                                      "~/tessera-work/disco/req/batch-1975-1791391390")))}
DISCO_BLOCK_KIT = pathlib.Path(os.path.expanduser(os.environ.get("DMIPY_SIM_BLOCK_KIT_DIR", "~/tessera-work/disco/bkit0")))


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


@pytest.mark.parametrize("pool_name", ["extra", "intra"])
def test_block_kit_matches_the_whole_tables_at_production_scale(pool_name, tmp_path):
    """DiSCo block 0, real scale: the SAME batch (its dumped ``r0``/``keys``/``seed`` against the whole
    tables) assembled from the block kit and walked with ``dmipy-walk`` (``DSC_PROFILE=1``) is bit-for-bit
    identical to the dump's own walk -- ``cmp``, not ``np.array_equal``: the raw output files, byte for byte."""
    dump_dir = DISCO_DUMPS[pool_name]
    if not (dump_dir / "request.txt").is_file():
        pytest.skip(f"no dumped production request at {dump_dir}")
    if not (DISCO_BLOCK_KIT / "kit.json").is_file():
        pytest.skip(f"no block kit at {DISCO_BLOCK_KIT} (python -m dmipy_sim.fill --block-kit 0 --out ...)")

    lines = dict(l.split("=", 1) for l in (dump_dir / "request.txt").read_text().splitlines())
    n, seed = int(lines["n"]), int(lines["seed"])
    r0 = np.fromfile(dump_dir / "r0.f32", np.float32).reshape(n, 3)
    keys = np.fromfile(dump_dir / "keys.u32", np.uint32).reshape(n, 2)

    block_req = assemble_request(DISCO_BLOCK_KIT, pool_name, r0, keys, seed, tmp_path / "block-req")

    env = dict(os.environ, DSC_PROFILE="1")
    whole_out, block_out = tmp_path / "whole-out", tmp_path / "block-out"
    whole_out.mkdir(); block_out.mkdir()
    for req_dir, out_dir in ((dump_dir, whole_out), (block_req, block_out)):
        p = subprocess.run([str(BIN), str(req_dir / "request.txt"), str(req_dir / "r0.f32"), str(req_dir / "keys.u32"), str(out_dir)],
                           env=env, capture_output=True, text=True)
        assert p.returncode == 0, p.stderr[-2000:]

    for name in ("positions.f32", "blt.f32", "comp.f32", "field.f32"):
        a, b = (whole_out / name).read_bytes(), (block_out / name).read_bytes()
        assert a == b, f"{name}: {len(a)} vs {len(b)} bytes, {pool_name} (n={n})"
