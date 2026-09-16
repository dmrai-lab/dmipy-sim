"""The claim-by-file protocol: any number of machines fill one plan against one repository with no scheduler.

A shard of a block is ``blocks/<variant>/block-NNNN[.pK].rpk`` (``.pK`` for pass K of a plan in passes; no tag
for the whole block, which covers every pass). A worker takes a block by writing
``claims/<variant>/block-NNNN[.pK].<host>.json``; a block whose shard or claim exists is taken. Claims are
written several per commit (:func:`claim_next`, ``claim_batch``), served from the worker's queue and released
when it stops. A claim carries a heartbeat (:func:`heartbeat_payload`, refreshed every :data:`HEARTBEAT_S` by the
worker's thread); a claim not refreshed for :data:`STALE_S` is a dead worker's and is released by the next
worker that claims (:func:`release_stale`)."""
import calendar
import json
import logging
import os
import time

from .recipe import FULL

HEARTBEAT_S = 900            # the claim's heartbeat: the walk's progress, ETA and memory, refreshed this often (a commit each)
STALE_S = 3 * HEARTBEAT_S    # a claim not refreshed for this long belongs to a dead worker
log = logging.getLogger("dmipy_sim.fill")


def stamp():
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def shard_name(block, P):
    """``block-NNNN`` for the whole block, ``block-NNNN.pK`` for pass K of it."""
    return f"block-{block:04d}" + (f".p{P['pass']}" if P.get("pass") is not None else "")


def parse_shard(f):
    """``(block, pass)`` of a shard or claim path (``.../block-0012.p1.rpk``, ``.../block-0012.host.json``);
    ``pass`` None for the whole block."""
    rest = f.split("block-")[1]; block = int(rest[:4]); tag = rest[4:].split(".")
    p = tag[1] if len(tag) > 1 else ""
    return block, (int(p[1:]) if p.startswith("p") and p[1:].isdigit() else None)


def claim_path(variant, name, host):
    return f"claims/{variant}/{name}.{host}.json"


def has_shard(hub, variant, block, P):
    """Whether the hub holds this pass's shard, or the whole block's."""
    return hub.exists(f"blocks/{variant}/{shard_name(block, P)}.rpk") or (
        P.get("pass") is not None and hub.exists(f"blocks/{variant}/{shard_name(block, FULL)}.rpk"))


def claim_next(hub, rc, host, *, only_pass=None, claim_batch=3, write=True):
    """The lowest open (pass, block), passes in the plan's order (``only_pass`` restricts to one), claimed; ``None``
    when none is open. ``claim_batch`` blocks are claimed in ONE commit and the rest queued on ``hub.queue``, served
    first by the next call; :func:`release_queue` gives them back when the worker stops. ``write=False`` claims
    nothing (a rehearsal) and returns the first open block."""
    if hub.queue:
        return hub.queue.pop(0)
    files = hub.files() - (release_stale(hub, rc.variant) if write else set()); variant = rc.variant
    held = [parse_shard(f) for f in files if f.startswith((f"blocks/{variant}/block-", f"claims/{variant}/block-"))]
    whole = {b for b, p in held if p is None}
    for P in rc.passes:
        if only_pass is not None and P.get("pass") != only_pass:
            continue
        taken = whole | {b for b, p in held if p == P.get("pass")}
        open_blocks = [r["block"] for r in rc.table if r["block"] not in taken]
        if open_blocks:
            if not write:
                return claim_block(hub, rc, open_blocks[0], host, P, write=False)
            got = claim_blocks(hub, rc, open_blocks[:max(1, int(claim_batch))], host, P)
            if got:
                hub.queue = got[1:]
                return got[0]
    return None


def claim_block(hub, rc, block, host, P, *, write=True):
    """One block claimed (one commit); ``write=False`` for a walk that claims nothing (a rehearsal, a certifying
    walk, a smoke test). ``None`` when the block already has a shard."""
    got = claim_blocks(hub, rc, [block], host, P, write=write)
    return got[0] if got else None


def claim_blocks(hub, rc, blocks, host, P, *, write=True):
    """``blocks`` claimed in one commit (a block that already has a shard is skipped); the claim dicts
    ``dict(block, row, name, claim, P)`` in order, ``claim`` None when nothing was written."""
    adds, got = {}, []
    for block in blocks:
        row = rc.table[block]; assert row["block"] == block
        name = shard_name(block, P)
        if write and has_shard(hub, rc.variant, block, P):
            log.info("%s already has a shard", name); continue
        claim = claim_path(rc.variant, name, host) if write else None
        if write:
            adds[claim] = json.dumps({"block": block, "pass": P.get("pass"), "variant": rc.variant, "host": host, "started": stamp(),
                                      "commit": rc.commit, "stage": "claimed"}, indent=1).encode()
        got.append(dict(block=block, row=row, name=name, claim=claim, P=P))
    if adds:
        hub.commit(adds, [], f"claim {', '.join(g['name'] for g in got)} ({rc.variant}) on {host}")
        log.info("claimed %s in one commit", ", ".join(adds))
        got = settle_collisions(hub, rc.variant, got, host)
    return got


def settle_collisions(hub, variant, got, host):
    """Two workers that claimed the same block within seconds of each other (a released block is the lowest open
    one for every worker at once): the claim that started first keeps the block, the other is released in one
    commit and dropped from ``got``. Reads the claims after our commit, so both see the same pair."""
    names = {g["name"]: g for g in got if g.get("claim")}
    if not names:
        return got
    lost, files = {}, hub.files()
    for f in files:
        if not f.startswith(f"claims/{variant}/") or f.endswith(f".{host}.json"):
            continue
        base = os.path.basename(f)[:-5]                    # block-NNNN[.pK].<other host>
        name = next((n for n in names if base.startswith(n + ".")), None)
        if name is None or name in lost:
            continue
        try:
            other = json.load(open(hub.get_live(f)))
            ours = json.loads(open(hub.get_live(names[name]["claim"])).read())
        except Exception as e:
            log.warning("could not compare claims of %s: %s", name, e); continue
        if (other.get("started") or "", other.get("host") or "") < (ours.get("started") or "", host):
            lost[name] = f
    if lost:
        hub.commit({}, [names[n]["claim"] for n in lost], f"release {', '.join(lost)}: claimed first by another worker")
        for n, f in lost.items():
            log.info("%s: %s claimed it first (%s); ours released", n, os.path.basename(f).split(".")[-2], f)
    return [g for g in got if g["name"] not in lost]


def release(hub, claimed, why):
    """A claim given back (one commit); nothing for a claim never written."""
    if claimed and claimed.get("claim"):
        try:
            hub.delete(claimed["claim"], f"release claim {claimed['name']}: {why}"); log.info("released %s", claimed["claim"])
        except Exception as e:
            log.warning("could not release %s: %s", claimed["claim"], e)


def release_queue(hub, host):
    """The blocks claimed ahead in a batch and never started: open again, in one commit."""
    if hub.queue:
        try:
            hub.commit({}, [c["claim"] for c in hub.queue if c.get("claim")], f"release {', '.join(c['name'] for c in hub.queue)}: {host} stopped")
            log.info("released the claim batch: %s", ", ".join(c["name"] for c in hub.queue))
        except Exception as e:
            log.warning("could not release the claim batch: %s", e)
        hub.queue = []


def claim_age(d, now=None):
    """Seconds since a claim's last sign of life (its heartbeat, else its start); None for a claim without one."""
    last = max(d.get("started") or "", d.get("heartbeat") or "")
    return ((now or time.time()) - calendar.timegm(time.strptime(last, "%Y-%m-%dT%H:%M:%SZ"))) if last else None


def release_stale(hub, variant, *, stale_s=None):
    """The claims of dead workers, released (a commit each): a claim whose heartbeat is older than ``stale_s``
    (:data:`STALE_S`). Returns the released paths."""
    stale_s = STALE_S if stale_s is None else stale_s
    gone = set()
    for f in sorted(hub.files()):
        if not f.startswith(f"claims/{variant}/"):
            continue
        try:
            age = claim_age(json.load(open(hub.get_live(f))))
        except Exception as e:                            # a claim being written or already gone: not ours to judge now
            log.warning("could not read %s: %s", f, e); continue
        if age is not None and age > stale_s:
            try:
                hub.delete(f, f"release {f}: no heartbeat for {age / 60:.0f} min (stale after {stale_s // 60})")
                log.info("released %s (stale: %.0f min)", f, age / 60); gone.add(f)
            except Exception as e:
                log.warning("could not release %s: %s", f, e)
    return gone


def heartbeat_payload(cur, rep):
    """A claim's refreshed content: the block's stage in the pipeline (``walking``, ``packing``, ``uploading``)
    and, while it walks, the run record's last progress row."""
    lp = rep.get("last_progress") or {}
    return dict(block=cur["block"], variant=cur["variant"], host=cur["host"], started=cur["started"], commit=cur["commit"],
                stage=cur.get("stage", "walking"), round=cur.get("round"), **{"pass": cur.get("pass")}, heartbeat=stamp(),
                progress=dict(done=lp.get("done"), total=lp.get("total"), unit=lp.get("unit"), rate_per_s=lp.get("rate_per_s"), eta_s=lp.get("eta_s")),
                peaks=rep.get("peaks"), status=rep.get("status"))


def heartbeat_once(hub, held, report):
    """One heartbeat: every claim the worker holds (``held``: ``{name: cur}`` -- the block walking, the ones
    packing and uploading) rewritten with :func:`heartbeat_payload` in ONE commit that is not retried (a 429 is
    logged and the next beat tries again). A block's claim stays fresh from its first round to its upload, so a
    slow device's shard is never taken for a dead worker's. ``report`` is :func:`dmipy_sim.run.report`."""
    curs = [c for c in list(held.values()) if c and c.get("claim")]
    if not curs:
        return False
    try:
        adds = {}
        for cur in curs:
            rd = cur.get("run_dir")
            rep = report(rd) if rd and os.path.isfile(os.path.join(rd, "manifest.json")) else {}
            adds[cur["claim"]] = json.dumps(heartbeat_payload(cur, rep), indent=1).encode()
        hub.commit(adds, [], f"heartbeat {', '.join(c['name'] for c in curs)} on {curs[0]['host']}", tries=1)
        return True
    except Exception as e:
        log.warning("heartbeat failed: %s", e)
        return False


def heartbeat(hub, state, stop, report, *, every=None):
    """The worker's heartbeat thread: :func:`heartbeat_once` on ``state["held"]`` every ``every`` seconds until
    ``stop`` is set. ``report`` is imported by the main thread before this one starts: a thread that imports
    while the main thread imports deadlocks on Python's import lock (measured)."""
    while not stop.wait(HEARTBEAT_S if every is None else every):
        heartbeat_once(hub, state.get("held") or {}, report)


def mine(hub, variant, host):
    """This host's claims on the hub."""
    return [f for f in hub.files() if f.startswith(f"claims/{variant}/") and f.endswith(f".{host}.json")]
