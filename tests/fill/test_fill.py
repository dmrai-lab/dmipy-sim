"""The fill module against a fake hub (`dmipy_sim.fill.FakeHub`): one commit per block, the claim protocol, the
429, the pipeline's rounds, the drain, the status."""
import json
import os
import time

import numpy as np
import pytest

from dmipy_sim.fill import (Fill, Options, Recipe, claim_next, claim_blocks, collect, draw_round, heartbeat_once, parse_shard, release_queue,
                            release_stale, render, round_seeding, shard_name, summarise)
from dmipy_sim.fill import hub as hubmod


def opts(work, **kw):
    return Options(workdir=work, host=kw.pop("host", "h1"), repo="fake", require_gpu=False, **kw)


def test_shard_names():
    assert shard_name(12, {"pass": None}) == "block-0012" and shard_name(12, {"pass": 2}) == "block-0012.p2"
    assert parse_shard("blocks/t/block-0012.p1.rpk") == (12, 1) and parse_shard("claims/t/block-0012.host.json") == (12, None)
    assert parse_shard("blocks/t/block-0007.rpk") == (7, None) and parse_shard("claims/t/block-0007.p2.some-host.json") == (7, 2)


def test_a_filled_block_costs_its_claim_and_one_commit(certified):
    """The loop over the plan's two blocks and two passes: every block leaves exactly two commits on the hub, the
    claim and the shard with its summary, its run record and the claim's release; no claim is left; the status
    reads it all back."""
    hub, work = certified
    n0 = len(hub.log)
    Fill(hub, Recipe(hub), opts(work, loop=True, claim_batch=1)).run(heartbeat_every=3600)
    log = hub.log[n0:]
    claims = [c for c in log if c["message"].startswith("claim ")]
    blocks = [c for c in log if c["message"].startswith("t block-")]
    assert len(claims) == 4 and len(blocks) == 4 and len(log) == 8, [c["message"] for c in log]
    for c in blocks:
        name = c["message"].split()[1][:-1]
        assert {f"blocks/t/{name}.rpk", f"blocks/t/{name}.json", f"blocks/t/{name}.run/manifest.json", f"blocks/t/{name}.run/events.jsonl",
                f"blocks/t/{name}.run/summary.json"} <= set(c["adds"])
        assert c["deletes"] == [f"claims/t/{name}.h1.json"]
    files = hub.files()
    assert not [f for f in files if f.startswith("claims/")]
    assert {f"blocks/t/block-000{b}.p{p}.rpk" for b in (0, 1) for p in (1, 2)} <= files
    s = json.load(open(hub.get("blocks/t/block-0000.p1.json")))
    assert s["pass"] == 1 and s["pass_scale"] == 0.5 and s["walkers"] > 0 and s["certified"] == "inherited" and s["host"] == "h1"
    assert not os.listdir(work)                                     # nothing left behind
    sm = summarise(collect(hub, "t"))
    assert sm["shards"] == 4 and sm["contributors"]["h1"]["shards"] == 4 and sm["per_pass"][1]["blocks_done"] == 2
    assert sm["commits_last_hour"]["blocks"] == 4 and sm["commits_last_hour"]["claims"] == 4
    assert "h1" in render(sm) and "| h1 |" in render(sm, markdown=True)


def test_a_429_is_retried_and_a_heartbeat_is_not(certified, monkeypatch):
    """The shard commit hit by a 429 lands after the wait; a heartbeat hit by one is skipped (its commit is not
    retried) and the claim keeps its previous content."""
    hub, work = certified
    monkeypatch.setattr(hubmod, "RATE_LIMIT_WAIT_S", 0.01)
    hub.fail_429.append("t block-0000.p1")
    Fill(hub, Recipe(hub), opts(work, block=0, only_pass=1)).run(heartbeat_every=3600)
    assert hub.exists("blocks/t/block-0000.p1.rpk") and not hub.fail_429
    rc = Recipe(hub)
    claimed = claim_next(hub, rc, "h2", claim_batch=1)
    before = open(hub.get(claimed["claim"])).read()
    hub.fail_429.append("heartbeat")
    cur = dict(block=claimed["block"], variant="t", host="h2", name=claimed["name"], claim=claimed["claim"], started="2026-01-01T00:00:00Z", commit="test",
               run_dir=os.path.join(work, "none"), stage="walking", **{"pass": 1})
    from dmipy_sim.run import report
    assert heartbeat_once(hub, {cur["name"]: cur}, report) is False and open(hub.get(claimed["claim"])).read() == before and not hub.fail_429
    other = claim_next(hub, rc, "h2", claim_batch=1); hub.queue = []
    held = {cur["name"]: cur, other["name"]: dict(cur, name=other["name"], claim=other["claim"], block=other["block"], stage="uploading", run_dir=None)}
    n = len(hub.log)
    assert heartbeat_once(hub, held, report) is True and len(hub.log) == n + 1        # every held claim in ONE commit
    assert json.load(open(hub.get(claimed["claim"])))["stage"] == "walking" and json.load(open(hub.get(other["claim"])))["stage"] == "uploading"


def test_the_claim_protocol(fake):
    """Two workers never take the same (pass, block); a whole-block shard covers both passes; a batch is one
    commit and its leftovers are released; a stale claim is released and its block taken again."""
    hub, work = fake
    rc = Recipe(hub)
    a = claim_next(hub, rc, "h1", claim_batch=1); b = claim_next(hub, rc, "h2", claim_batch=1)
    assert (a["block"], a["P"]["pass"]) == (0, 1) and (b["block"], b["P"]["pass"]) == (1, 1)
    c = claim_next(hub, rc, "h3", claim_batch=1)
    assert (c["block"], c["P"]["pass"]) == (0, 2)                    # pass 1 is taken: pass 2 begins
    os.makedirs(os.path.join(hub.root, "blocks/t"), exist_ok=True); open(os.path.join(hub.root, "blocks/t/block-0001.rpk"), "wb").write(b"x")
    d = claim_next(hub, rc, "h4", claim_batch=1)
    assert d is None                                                  # block 1 whole: its pass 2 is covered too
    for f in (a["claim"], b["claim"], c["claim"]):
        hub.delete(f, "test")
    n = len(hub.log)
    got = claim_blocks(hub, rc, [0], "h5", rc.passes[0]); assert got and len(hub.log) == n + 1
    hub.delete(got[0]["claim"], "test")
    e = claim_next(hub, rc, "h6", claim_batch=3)                      # block 0 pass 1 and pass 2 are open, block 1 is whole
    assert e["block"] == 0 and e["P"]["pass"] == 1 and hub.queue == [] and len(hub.log) == n + 3
    stale = "claims/t/block-0000.p2.dead.json"
    hub.put_json(dict(block=0, variant="t", host="dead", started="2026-01-01T00:00:00Z", **{"pass": 2}), stale, "claim block-0000.p2 (t) on dead")
    assert release_stale(hub, "t") == {stale} and not hub.exists(stale)
    fresh = "claims/t/block-0000.p2.alive.json"
    hub.put_json(dict(block=0, variant="t", host="alive", started=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), **{"pass": 2}), fresh, "claim")
    assert release_stale(hub, "t") == set() and hub.exists(fresh)
    hub.queue = [dict(name="block-0000.p2", claim=fresh)]
    release_queue(hub, "alive"); assert not hub.exists(fresh) and hub.queue == []


def test_the_rounds_of_a_block_merge_to_the_shard_walked_by_hand(certified):
    """Under ``max_walkers`` a block walks in rounds whose seeds are dealt from the plan; the shard is the rounds'
    packs merged with the union's weights, equal to the same rounds walked, packed and merged by hand."""
    from dmipy_sim.replay import read_rpk
    from dmipy_sim.replay.bank import build_replay_pack, merge_packs
    from dmipy_sim.spec import draw_seeds, walk_spec
    hub, work = certified
    rc = Recipe(hub)
    assert rc.context() is rc.context()
    Fill(hub, rc, opts(work, block=0, only_pass=1, max_walkers=10, keep=True)).run(heartbeat_every=3600)
    shard = read_rpk(hub.get("blocks/t/block-0000.p1.rpk"))
    s = json.load(open(hub.get("blocks/t/block-0000.p1.json")))
    assert s["rounds"] == 3 and shard.meta["provenance"]["shards"] and len(shard.meta["provenance"]["shards"]) == 3
    row, P = rc.table[0], rc.passes[0]; W = rc.man["walk"]; cert = json.load(open(hub.get("certificate/t.json")))
    packs = []
    for r in range(3):
        seeding, seed, n_plan, tot, scale = round_seeding(rc, row, r, 3, P, None)
        w = walk_spec(rc.spec(), T_max=W["T_max_s"], dt_save=rc.dt_save(), seeding=draw_seeds(rc.spec(), seeding, seed, context=rc.context()), seed=seed,
                      field=False, require_gpu=False, walker_batch_size=W["walker_batch_size"], adaptive_steps=True, scanner=W["scanner"],
                      floor_fraction=W["floor_fraction"], context=rc.context())
        packs.append(build_replay_pack(w, id=f"hand/{r}", license="x", citation="x", K=3, position_container="bands", blt_container="bands",
                                       voxel_grid=rc.grid(), out_path=os.path.join(work, f"hand{r}.rpk"), device="numpy", fidelity="inherited", fidelity_from=cert))
    hand = merge_packs(packs, id="hand/merged", out_path=os.path.join(work, "hand.rpk"), overlap="recertify", device="numpy")
    assert hand.n_walkers == shard.n_walkers == s["walkers"]
    for k in shard.arrays:
        np.testing.assert_array_equal(np.asarray(shard.arrays[k]), np.asarray(hand.arrays[k]), err_msg=k)


def test_drain_finishes_what_a_dead_worker_left(certified):
    """A worker that died with a walked round on disk and a second claim: ``drain`` packs and uploads the block
    (one commit, the claim released with it) and releases the other claim."""
    from dmipy_sim.fill.pipeline import save_walk, walk_round
    hub, work = certified
    rc = Recipe(hub); o = opts(work, host="dead"); f = Fill(hub, rc, o)
    a = claim_next(hub, rc, "dead", claim_batch=2)                    # block 0 pass 1 walked, block 1 pass 1 queued
    b = hub.queue[0]; hub.queue = []
    w, rd = walk_round(o, rc, a["row"], a["name"], 0, 1, a["P"], draw_round(rc, a["row"], 0, 1, a["P"], None))
    save_walk(w, rd)
    job = f.job_of(a, [rd], os.path.join(work, f"t-{a['name']}.rpk")); job.pop("claim_state")   # a job file of an earlier worker version
    json.dump(job, open(job["file"], "w"), default=float)
    n = len(hub.log)
    Fill(hub, rc, opts(work, host="dead")).drain()
    assert hub.exists(f"blocks/t/{a['name']}.rpk") and not hub.exists(a["claim"]) and not hub.exists(b["claim"])
    assert [c["message"][:12] for c in hub.log[n:]] == [f"t {a['name']}"[:12], "release clai"] and not os.listdir(work)
