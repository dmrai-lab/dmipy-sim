"""Blocks by the excursion: the table covers the grid once, a block's shards sum exactly to its counts, and a block's
tar is the block kit with the far crop as a truncated spectrum."""
import json
import tarfile

import numpy as np

from dmipy_sim.fill import Recipe
from dmipy_sim.fill.blocks import block_table, plan_blocks, shard_counts, write_block_tar
from dmipy_sim.fill.farspec import pack_far, unpack_far
from dmipy_sim.fill.kit import _sha256, write_block_kit

PLAN_KEYS = {"repo", "variant", "pass", "scale", "block_edge_voxels", "shard_walkers", "code", "blocks"}
ROW_KEYS = {"block", "i", "j", "k", "seed", "walkers", "voxels", "n_shards", "kit"}


def test_block_table_covers_every_voxel_once(fake):
    hub, _ = fake
    rc = Recipe(hub)
    rows = block_table(rc, 1, pass_n=1, shard_walkers=5)
    seen = np.zeros(rc.grid().shape, int)
    for r in rows:
        seen[r["i"][0]:r["i"][1], r["j"][0]:r["j"][1], r["k"][0]:r["k"][1]] += 1
        want, _, _ = rc.counts(r, pass_scale=0.5)
        assert r["walkers"] == {n: int(v.sum()) for n, v in want.items()}
        assert r["n_shards"] == -(-sum(r["walkers"].values()) // 5) and r["voxels"] == 1
    assert (seen == 1).all() and [r["block"] for r in rows] == list(range(8))
    one = block_table(rc, 2, pass_n=2, shard_walkers=1000)
    assert len(one) == 1 and one[0]["voxels"] == 8 and one[0]["n_shards"] == 1
    uneven = block_table(rc, 3, pass_n=1, shard_walkers=1000)             # an edge past the grid: the one cube is the grid
    assert len(uneven) == 1 and uneven[0]["i"] == [0, 2]


def test_shards_sum_exactly_and_absent_pools_are_absent():
    rng = np.random.default_rng(1)
    want = {"extra": rng.integers(0, 40, (3, 4, 2)), "intra": rng.integers(0, 3, (3, 4, 2)), "none": np.zeros((3, 4, 2), int)}
    for n in (1, 2, 7, 50):
        shards = [shard_counts(want, i, n) for i in range(n)]
        for pool, w in want.items():
            total = sum(s[pool] for s in shards if pool in s) if any(pool in s for s in shards) else np.zeros_like(w)
            assert np.array_equal(total, w), (pool, n)
        for s in shards:
            assert "none" not in s
            assert all(v.sum() > 0 for v in s.values())
    assert "intra" not in shard_counts({"intra": np.array([1, 0])}, 0, 2)          # floor(1*1/2) - 0 = 0


def test_block_tar_layout_and_far_spectrum(field_fake, tmp_path):
    hub, _ = field_fake
    rc = Recipe(hub)
    row = block_table(rc, 1, pass_n=1, shard_walkers=100)[0]
    path, sha, nbytes = write_block_tar(rc, row, tmp_path / "b.tar")
    assert _sha256(path) == sha and nbytes == (tmp_path / "b.tar").stat().st_size
    dense = write_block_kit(rc, row, tmp_path / "dense")
    with tarfile.open(path) as tf:
        names = tf.getnames()
        assert names == sorted(names)
        kit = json.load(tf.extractfile("kit.json"))
        expect = {str(p.relative_to(dense)) for p in dense.rglob("*") if p.is_file()} - {"far_full.u16"} | {"far_spec.npz"}
        assert set(names) == expect
        assert "far_full.u16" not in kit["files"] and "far_spec.npz" in kit["files"] and "far_full.json" in names
        import hashlib
        for rel, s in kit["files"].items():
            assert hashlib.sha256(tf.extractfile(rel).read()).hexdigest() == s, rel
        tf.extract("far_spec.npz", tmp_path / "x")
    meta = json.loads((dense / "far_full.json").read_text())
    crop = np.fromfile(dense / "far_full.u16", np.uint16).view(np.float16).reshape(meta["shape"])
    ref = unpack_far(pack_far(crop, meta["spacing_m"]))
    assert np.array_equal(unpack_far(np.load(tmp_path / "x" / "far_spec.npz")), ref)
    assert _sha256(write_block_tar(rc, row, tmp_path / "b2.tar")[0]) == sha            # the same bytes every time


def test_plan_blocks_writes_the_plan_file_and_every_tar(field_fake, tmp_path):
    hub, _ = field_fake
    rc = Recipe(hub)
    plan = plan_blocks(rc, edge_voxels=2, shard_walkers=7, pass_n=1, out_dir=tmp_path / "out")
    on_disk = json.loads((tmp_path / "out" / "plan.p1.json").read_text())
    assert on_disk == plan and PLAN_KEYS <= set(plan) and plan["pass"] == 1 and plan["scale"] == 0.5 and plan["variant"] == "t"
    assert plan["code"]["dmipy_sim"]
    assert len(plan["blocks"]) == 1
    for b in plan["blocks"]:
        assert ROW_KEYS <= set(b)
        assert b["kit"]["path"] == f"kit/t/block-{b['block']:04d}.p1.tar"
        tar = tmp_path / "out" / b["kit"]["path"]
        assert tar.stat().st_size == b["kit"]["bytes"] and _sha256(tar) == b["kit"]["sha256"]


def test_cli_publishes_plan_and_tars_in_one_commit(field_fake, tmp_path, monkeypatch):
    from dmipy_sim.fill import blocks, hub as hubmod
    hub, _ = field_fake
    monkeypatch.setattr(hubmod, "Hub", lambda repo: hub)
    n0 = len(hub.log)
    blocks.main(["--repo", "x/y", "--pass", "2", "--edge", "2", "--shard-walkers", "50", "--out", str(tmp_path / "o"), "--publish"])
    assert len(hub.log) == n0 + 1
    assert set(hub.log[-1]["adds"]) == {"blocks/t/plan.p2.json", "kit/t/block-0000.p2.tar"}
