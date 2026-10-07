"""The fill module against a fake hub (`dmipy_sim.fill.FakeHub`): one commit per block, the claim protocol, the
429, the pipeline's rounds, the drain, the status."""
import dataclasses
import json
import os
import time

import numpy as np
import pytest

from dmipy_sim.fill import (Fill, Options, Recipe, claim_next, claim_blocks, collect, draw_round, heartbeat_once, parse_shard, release_queue,
                            release_stale, render, round_seeding, shard_name, summarise)
from dmipy_sim.fill.claims import settle_collisions
from dmipy_sim.fill import hub as hubmod


def opts(work, **kw):
    return Options(workdir=work, host=kw.pop("host", "h1"), repo="fake", require_gpu=False, **kw)


def test_the_default_worker_token_is_never_the_hostname(tmp_path, monkeypatch):
    """`--host`'s default (`dmipy_sim.fill.__main__._worker_token`) is an opaque label cached locally per
    machine, stable across restarts, and never this machine's hostname."""
    import socket
    from dmipy_sim.fill.__main__ import _worker_token
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.delenv("DMIPY_SIM_FILL_WORKER_ID", raising=False)
    tok1 = _worker_token(); tok2 = _worker_token()
    assert tok1 == tok2 and tok1 != socket.gethostname() and tok1 != socket.gethostname().split(".")[0]
    cache = tmp_path / ".cache" / "dmipy-sim" / "fill-worker-id"
    assert cache.is_file() and cache.read_text().strip() == tok1
    monkeypatch.setenv("DMIPY_SIM_FILL_WORKER_ID", "explicit-label")
    assert _worker_token() == "explicit-label"


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


def test_the_shard_and_claim_name_hardware_class_never_the_machine(certified):
    """A worker's ``host`` is its own opaque label (never this machine's hostname, which `Options.host` must
    never default to -- `dmipy_sim.fill.__main__._worker_token`); the shard's summary and the pack's provenance
    carry the hardware CLASS (`gpu`, `cpu_arch`) a worker declares, never a hostname or an absolute path."""
    import socket
    hub, work = certified
    Fill(hub, Recipe(hub), opts(work, loop=True, claim_batch=1, gpu="L40S", cpu_arch="x86_64")).run(heartbeat_every=3600)
    s = json.load(open(hub.get("blocks/t/block-0000.p1.json")))
    assert s["host"] == "h1" and s["gpu"] == "L40S" and s["cpu_arch"] == "x86_64"
    pk_meta = json.load(open(hub.get("blocks/t/block-0000.p1.run/manifest.json")))
    blob = json.dumps(s) + json.dumps(pk_meta)
    hostname = socket.gethostname()
    if hostname and len(hostname) > 2:
        assert hostname not in blob
    assert work not in blob                          # no absolute local path
    assert "platform" not in pk_meta and pk_meta.get("cpu_arch")


def test_a_429_is_retried_and_a_heartbeat_is_not(certified, monkeypatch):
    """The shard commit hit by a 429 lands after the wait; a heartbeat hit by one is skipped (its commit is not
    retried) and the claim keeps its previous content."""
    hub, work = certified
    monkeypatch.setattr(hubmod, "RATE_LIMIT_WAIT_S", 0.01); monkeypatch.setattr(hubmod, "OUTAGE_WAIT_S", 0.01)
    hub.fail_429.append("t block-0000.p1"); hub.fail_500 += ["t block-0000.p1"] * 8      # eight 5xx in a row: an outage, beyond `tries`
    Fill(hub, Recipe(hub), opts(work, block=0, only_pass=1)).run(heartbeat_every=3600)
    assert hub.exists("blocks/t/block-0000.p1.rpk") and not hub.fail_429 and not hub.fail_500
    hub.fail_500.append("claim"); n = len(hub.log)
    with pytest.raises(Exception, match="500"):
        hub.commit({"claims/t/x.json": b"{}"}, [], "claim x", tries=1)                    # a heartbeat-style commit never retries
    assert len(hub.log) == n and not hub.fail_500
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
    hub.delete(other["claim"], "test")
    q = claim_next(hub, rc, "h2", claim_batch=2); queued = list(hub.queue); hub.queue = []   # a batch: the second claim is queued
    assert queued and json.load(open(hub.get(queued[0]["claim"]))).get("heartbeat") is None
    n = len(hub.log)
    assert heartbeat_once(hub, {cur["name"]: cur}, report, queue=queued) is True and len(hub.log) == n + 1
    d = json.load(open(hub.get(queued[0]["claim"]))); assert d["stage"] == "queued" and d["heartbeat"] and d["host"] == "h2"


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
                      require_gpu=False, walker_batch_size=W["walker_batch_size"], adaptive_steps=True, scanner=W["scanner"],
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


def test_two_claims_of_one_block_settle_to_the_first(fake):
    """Two workers that read the same open list and both claim block 0 (a released block is the lowest open one
    for everybody): the claim that started first stays, the other is released in one commit and the loser gets
    nothing; a worker whose claimed block gained a shard meanwhile skips it."""
    hub, work = fake
    rc = Recipe(hub); P = rc.passes[0]
    a = claim_blocks(hub, rc, [0], "first", P)                       # 'first' claims at t0
    time.sleep(1.1)                                                  # 'second' read the same open list and claims too (the race)
    hub.put_json({"block": 0, "pass": 1, "variant": "t", "host": "second", "started": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                  "commit": "test", "stage": "claimed"}, "claims/t/block-0000.p1.second.json", "claim block-0000.p1 (t) on second")
    got = settle_collisions(hub, "t", [dict(block=0, row=rc.table[0], name="block-0000.p1", claim="claims/t/block-0000.p1.second.json", P=P)], "second")
    assert got == [] and hub.exists(a[0]["claim"]) and not hub.exists("claims/t/block-0000.p1.second.json")
    assert hub.log[-1]["message"].startswith("release block-0000.p1: claimed first")
    # the mirror: 'first' settling against 'second' keeps its claim
    assert settle_collisions(hub, "t", a, "first") == a
    # a claimed block that gained a shard meanwhile is skipped by the loop, its claim released
    os.makedirs(os.path.join(hub.root, "blocks/t"), exist_ok=True); open(os.path.join(hub.root, "blocks/t/block-0000.p1.rpk"), "wb").write(b"x")
    os.makedirs(os.path.join(hub.root, "certificate"), exist_ok=True); open(os.path.join(hub.root, "certificate/t.json"), "w").write("{}")
    f = Fill(hub, rc, opts(work, host="first", loop=False))
    f.claim_first = lambda: a[0]
    f.run(heartbeat_every=3600)
    assert not hub.exists(a[0]["claim"]) and not os.listdir(work)


def test_a_batch_lost_entirely_is_not_the_end_of_the_loop(fake, monkeypatch):
    """A worker whose whole batch was claimed first by another (every worker takes the lowest open blocks) reads the
    listing again and claims the next open (pass, block) instead of returning None, which ended a loop with
    hundreds of blocks open."""
    hub, work = fake
    rc = Recipe(hub)
    earlier = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(time.time() - 5))
    first = [f"claims/t/block-000{b}.p1.first.json" for b in (0, 1)]
    for b, f in zip((0, 1), first):                                  # 'first' holds pass 1 of both blocks, claimed 5 s ago
        hub.put_json({"block": b, "pass": 1, "variant": "t", "host": "first", "started": earlier, "commit": "test", "stage": "claimed"}, f, "claim")
    real, calls = hub.files, []
    def stale_once():                                                # 'second' read the listing before those claims landed
        calls.append(1); fs = real()
        return fs - set(first) if len(calls) == 1 else fs
    monkeypatch.setattr(hub, "files", stale_once)
    got = claim_next(hub, rc, "second", claim_batch=2)
    assert (got["block"], got["P"]["pass"]) == (0, 2)                # pass 1 is 'first's: pass 2 of block 0 is the next open one
    assert [(q["block"], q["P"]["pass"]) for q in hub.queue] == [(1, 2)]  # and pass 2 of block 1 came in the same batch
    assert all(hub.exists(f) for f in first) and not any("second" in f and ".p1." in f for f in hub.files())
    hub.queue = []
    assert claim_next(hub, rc, "third", claim_batch=2) is None       # every (pass, block) is claimed: nothing open is None, still


def test_a_read_that_fails_is_retried_and_a_missing_file_is_not(fake, monkeypatch):
    """A listing, an exists or a download that meets a 5xx or a network error is retried with backoff (a worker's
    loop ended on one, dmipy-sim#286); a file that is not there is an answer and is raised at once."""
    hub, work = fake
    waits = []
    monkeypatch.setattr(hubmod.time, "sleep", lambda s: waits.append(s))
    hub.fail_read += ["exists", "exists", "files", "get"]
    assert hub.exists("manifest.json") and hub.files() and hub.get("manifest.json")
    assert waits == [30, 60, 30, 30]
    with pytest.raises(FileNotFoundError):
        hub.get("nothing/here.json")
    assert waits == [30, 60, 30, 30]                                     # not retried
    hub.fail_read += ["exists"] * 8                                      # beyond READ_TRIES: the error surfaces
    with pytest.raises(Exception, match="the hub failed"):
        hub.exists("manifest.json")


def test_the_duty_pauses_the_device_after_a_walk(certified, monkeypatch, tmp_path):
    """A duty below one pauses the worker after each walk for walk_time * (1 / duty - 1), read from the duty file
    before every walk so a shared box is given back by the hour without a restart."""
    hub, work = certified
    rc = Recipe(hub)
    pauses = []
    monkeypatch.setattr(hubmod.time, "sleep", lambda s: None)
    import dmipy_sim.fill.pipeline as pl
    monkeypatch.setattr(pl.time, "sleep", lambda s: pauses.append(s))
    duty = tmp_path / "duty"; duty.write_text("0.5")
    o = opts(work, host="h", block=0, loop=False); o = dataclasses.replace(o, duty_file=str(duty), no_upload=True)
    assert o.current_duty() == 0.5
    Fill(hub, rc, o).run(heartbeat_every=3600)
    assert len(pauses) >= 1 and all(p > 0 for p in pauses)
    duty.write_text("1")
    assert o.current_duty() == 1.0
    duty.write_text("nonsense")
    assert o.current_duty() == 1.0                                      # unparsable: the option's value, logged


def test_the_backend_reaches_the_walk_and_an_unknown_one_is_refused(certified, monkeypatch):
    """`--backend` is handed to `walk_spec` as given: the walk resolves it (an unknown name is refused naming what
    is installed, before any block is walked) and its record names it."""
    import pytest
    import dmipy_sim.fill.pipeline as pl
    hub, work = certified
    rc = Recipe(hub)
    seen = []
    real = pl.walk_spec if hasattr(pl, "walk_spec") else None
    from dmipy_sim import spec as specmod
    orig = specmod.walk_spec

    def spy(*a, **kw):
        seen.append((kw.get("backend"), kw.get("spool"))); return orig(*a, **kw)
    monkeypatch.setattr(specmod, "walk_spec", spy)
    o = dataclasses.replace(opts(work, host="h", block=0, loop=False), no_upload=True, backend="jax")
    Fill(hub, rc, o).run(heartbeat_every=3600)
    assert seen and all(b == ("jax", True) for b in seen)                  # the JAX walk is spooled (the resume)
    o = dataclasses.replace(opts(work, host="h2", block=0, loop=False), no_upload=True, backend="nope")
    with pytest.raises(ValueError, match="no backend 'nope' is installed"):
        Fill(hub, rc, o).run(heartbeat_every=3600)



def test_the_pack_of_one_block_overlaps_the_walk_of_the_next(certified, monkeypatch):
    """dmipy-sim#678: the pack runs in a thread of this process, from the walk's own objects, no longer a
    subprocess reloading a ``.walk`` file -- a fake walk and a fake pack, both padded with a sleep, record a
    timeline over the two blocks of one pass: the second block's walk overlaps the first block's pack (the
    device is never idle waiting on it), and the two packs themselves stay serialised (queue depth one)."""
    import dmipy_sim.fill.pipeline as pl
    hub, work = certified
    rc = Recipe(hub)
    timeline = []
    real_walk_round, real_pack = pl.walk_round, pl.pack_in_process

    def fake_walk_round(o, rc, row, name, r, k, P, seeds):
        t0 = time.time(); w, rd = real_walk_round(o, rc, row, name, r, k, P, seeds); time.sleep(0.3)
        timeline.append(("walk", name, t0, time.time())); return w, rd

    def fake_pack(job, walks):
        t0 = time.time(); time.sleep(0.5); summary = real_pack(job, walks)
        timeline.append(("pack", job["name"], t0, time.time())); return summary

    monkeypatch.setattr(pl, "walk_round", fake_walk_round)
    monkeypatch.setattr(pl, "pack_in_process", fake_pack)
    Fill(hub, rc, opts(work, loop=True, claim_batch=1, only_pass=1)).run(heartbeat_every=3600)
    walks = {n: (t0, t1) for kind, n, t0, t1 in timeline if kind == "walk"}
    packs = {n: (t0, t1) for kind, n, t0, t1 in timeline if kind == "pack"}
    assert len(walks) == 2 and len(packs) == 2
    first, second = sorted(walks, key=lambda n: walks[n][0])
    assert walks[second][0] < packs[first][1], "the second block's walk should start before the first block's pack ends"
    assert packs[second][0] >= packs[first][1], "packs stay serialised: at most one in flight"


def test_batch_auto_asks_the_backend_and_refuses_one_that_sizes_nothing(certified, monkeypatch):
    """`--batch auto` takes the backend's `suggested_batch` for the walk (n_t and the field's samples from the manifest)
    and refuses, by name, a backend without one."""
    import pytest
    from dmipy_sim.engine import backends
    from dmipy_sim.fill.pipeline import batch_size

    class _Sized(backends.Backend):
        name = "sized"
        def __init__(self): self.asked = []
        def suggested_batch(self, n_t, **kw): self.asked.append((n_t, kw)); return 777
    hub, work = certified
    rc = Recipe(hub)
    be = _Sized()
    monkeypatch.setattr(backends, "resolve", lambda name: be if name == "sized" else None)
    import dmipy_sim.fill.pipeline as pl
    monkeypatch.setattr(pl, "resolve", lambda name: be if name == "sized" else None, raising=False)
    o = dataclasses.replace(opts(work, host="h", block=0, loop=False), batch="auto", backend="sized")
    assert batch_size(o, rc) == 777 and be.asked[0][0] == int(round(rc.man["walk"]["T_max_s"] / rc.dt_save())) + 1
    o = dataclasses.replace(opts(work, host="h", block=0, loop=False), batch="auto", backend="jax")
    with pytest.raises(ValueError, match="sizes no batch"):
        batch_size(o, rc)
    assert batch_size(dataclasses.replace(o, batch=12), rc) == 12
    assert batch_size(dataclasses.replace(o, batch=None), rc) == rc.man["walk"]["walker_batch_size"]
