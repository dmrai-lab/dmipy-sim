"""The state of a fill, read from the repository and nothing else: every shard's summary, every claim's
heartbeat, the plan, the commit log. One row per contributor (shards, walkers, blocks per hour, the block in
flight with its progress and ETA, the heartbeat's age), the plan's progress per pass with the time to its end at
the last hour's rate, and the commit budget (128 an hour per repository: blocks, claims and heartbeats all
count). No provider is asked for credits or quotas: a contributor that stops is seen by its heartbeat.

    python -m dmipy_sim.fill.status --repo OWNER/DATASET                    # the table
    python -m dmipy_sim.fill.status --repo OWNER/DATASET --publish          # ... and STATUS.md committed (one commit)
    python -m dmipy_sim.fill.status --repo OWNER/DATASET --json status.json # ... and the numbers for a supervisor

Summaries are cached in ``--cache`` (a shard's summary never changes); the file list, the claims and the commit
log are read fresh."""
import argparse
import datetime as dt
import json
import os
import re
import time

from .claims import HEARTBEAT_S, claim_age, parse_shard

CONTRIBUTORS = [(r"^gaia$", "gaia GH200"), (r"L40S", "L40S"), (r"^modal", "Modal T4"), (r"^ip-10-", "Lightning T4"),
                (r"^[0-9a-f]{12}$", "Kaggle T4")]      # host name -> contributor, for the DiSCo fill's machines
WINDOW_S = 3600              # the rate is read over the last hour (three hours when the hour holds fewer than 3 shards)
COMMITS_PER_HOUR = 128       # the hub's cap per repository


def who(host, contributors=CONTRIBUTORS):
    for pat, name in contributors:
        if re.search(pat, host or ""):
            return name
    return host or "?"


def parse_stamp(s):
    import calendar
    return calendar.timegm(time.strptime(s[:19], "%Y-%m-%dT%H:%M:%S")) if s else None


def _read(hub, f, cache=None):
    if cache:
        local = os.path.join(cache, f)
        if os.path.isfile(local):
            return json.load(open(local))
    d = json.load(open(hub.get_live(f)))
    if cache:
        os.makedirs(os.path.dirname(local), exist_ok=True); json.dump(d, open(local, "w"))
    return d


def collect(hub, variant, cache=None, *, now=None):
    """Everything the status is computed from, read once: the plan, the shards' summaries with the time the hub
    took each (the commit log), the claims with their ages, the commits of the last hour by kind."""
    now = time.time() if now is None else now
    files = hub.files()
    man = _read(hub, "manifest.json")
    table = _read(hub, man["plan"]["blocks"])["blocks"]
    passes = man["plan"].get("passes") or [{"pass": None, "scale": 1.0}]
    scale = {p.get("pass"): float(p["scale"]) for p in passes}
    planned = {r["block"]: sum(r["walkers"].values()) for r in table}
    done_at, n_last = {}, {"blocks": 0, "claims": 0, "heartbeats": 0, "other": 0}
    for t, title in hub.commits():
        m = re.match(rf"{variant} (block-\d+(?:\.p\d+)?): ", title)
        if m:
            done_at.setdefault(m.group(1), t)
        if now - t <= 3600:
            kind = "blocks" if m else "claims" if title.startswith("claim ") else "heartbeats" if title.startswith("heartbeat") else "other"
            n_last[kind] += 1
    shards = {}
    for f in sorted(files):
        if re.fullmatch(rf"blocks/{variant}/block-\d+(\.p\d+)?\.json", f):   # a shard's summary, not a run record
            name = os.path.basename(f)[:-5]
            d = _read(hub, f, cache); b, P = parse_shard(f)
            walk = (d.get("run") or {}).get("walk") or {}
            shards[name] = dict(block=b, pass_=P, host=d.get("host"), who=who(d.get("host")), walkers=int(d.get("walkers", 0)),
                                t_walk=float(d.get("t_walk_s") or 0), t_pack=float(d.get("t_pack_s") or 0), size=int(d.get("size_bytes") or 0),
                                started=parse_stamp(walk.get("started")), done=done_at.get(name), peak_rss=walk.get("peak_rss_bytes"))
    claims = []
    for f in sorted(files):
        if f.startswith(f"claims/{variant}/"):
            try:
                d = _read(hub, f)
            except Exception:
                continue
            age = claim_age(d, now); pr = d.get("progress") or {}
            claims.append(dict(file=f, block=d.get("block"), pass_=d.get("pass"), host=d.get("host"), who=who(d.get("host")),
                               started=parse_stamp(d.get("started")), age=age, done=pr.get("done"), total=pr.get("total"),
                               eta_s=pr.get("eta_s"), round=d.get("round")))
    return dict(now=now, n_blocks=len(table), planned=planned, scale=scale, shards=shards, claims=claims, commits_last_hour=n_last,
                walkers_planned=int(man["plan"].get("walkers", sum(planned.values()))))


def summarise(st):
    """The numbers: per contributor, per pass, the rate over the window, the commit budget."""
    now, planned, scale = st["now"], st["planned"], st["scale"]
    shards = list(st["shards"].values()); stale_s = 3 * HEARTBEAT_S
    window = WINDOW_S
    recent = [s for s in shards if s["done"] and now - s["done"] <= window]
    if len(recent) < 3:
        window = 3 * WINDOW_S; recent = [s for s in shards if s["done"] and now - s["done"] <= window]
    names = [n for _, n in CONTRIBUTORS]
    for s in shards + st["claims"]:
        if s["who"] not in names:
            names.append(s["who"])
    contributors = {}
    for n in names:
        mine = [s for s in shards if s["who"] == n]; rec = [s for s in recent if s["who"] == n]
        inflight = [c for c in st["claims"] if c["who"] == n]
        live = [c for c in inflight if c["age"] is not None and c["age"] <= stale_s]
        status = "live" if live else "stale" if inflight else "idle"
        if not inflight and mine:
            status = "idle" if now - max(s["done"] or 0 for s in mine) > stale_s else "between blocks"
        contributors[n] = dict(shards=len(mine), whole=sum(s["pass_"] is None for s in mine), walkers=sum(s["walkers"] for s in mine),
                               walk_hours=sum(s["t_walk"] + s["t_pack"] for s in mine) / 3600,
                               ms_per_walker=(1e3 * sum(s["t_walk"] for s in mine) / max(sum(s["walkers"] for s in mine), 1)) if mine else None,
                               blocks_per_hour=len(rec) * 3600 / window, walkers_per_hour=sum(s["walkers"] for s in rec) * 3600 / window, status=status,
                               inflight=[dict(block=c["block"], pass_=c["pass_"], host=c["host"], age_min=(c["age"] or 0) / 60,
                                              progress=(c["done"] / c["total"]) if c.get("done") and c.get("total") else None,
                                              eta_min=(c["eta_s"] or 0) / 60 if c.get("eta_s") is not None else None, round=c["round"]) for c in inflight],
                               last_done_min=((now - max(s["done"] or 0 for s in mine)) / 60) if mine else None)
    whole = {s["block"] for s in shards if s["pass_"] is None}
    rate_w = sum(s["walkers"] for s in recent) * 3600 / window
    per_pass = {}
    for P, sc in scale.items():
        done_blocks = whole | {s["block"] for s in shards if s["pass_"] == P}
        done_w = sum(s["walkers"] for s in shards if s["pass_"] == P) + sum(planned[b] * sc for b in whole if P is not None)
        remaining_w = sum(planned[b] * sc for b in planned if b not in done_blocks)
        per_pass[P] = dict(scale=sc, blocks_done=len(done_blocks), blocks=st["n_blocks"], walkers_done=int(done_w), walkers_remaining=int(remaining_w),
                           hours_left=(remaining_w / rate_w) if rate_w > 0 else None)
    return dict(time=dt.datetime.utcfromtimestamp(now).strftime("%Y-%m-%d %H:%M UTC"), window_h=window / 3600,
                blocks_per_hour=len(recent) * 3600 / window, walkers_per_hour=rate_w, per_pass=per_pass, contributors=contributors,
                commits_last_hour=st["commits_last_hour"], commits_cap=COMMITS_PER_HOUR, shards=len(shards),
                bytes_on_hub=sum(s["size"] for s in shards), claims=st["claims"], stale_s=stale_s)


def fmt_h(h):
    return "—" if h is None else (f"{h * 60:.0f} min" if h < 1 else f"{h:.1f} h" if h < 48 else f"{h / 24:.1f} d")


def render(sm, markdown=False):
    """The summary as text (or Markdown)."""
    L = [f"# Fill status — {sm['time']}" if markdown else f"Fill status — {sm['time']}", ""]
    for P, d in sorted(sm["per_pass"].items(), key=lambda kv: (kv[0] is None, kv[0] or 0)):
        L.append(f"{'pass %s' % P if P is not None else 'the plan'} ({d['scale']:.2f} of the plan): {d['blocks_done']} / {d['blocks']} blocks, "
                 f"{d['walkers_done'] / 1e6:.1f} M walkers done, {d['walkers_remaining'] / 1e6:.1f} M to go — {fmt_h(d['hours_left'])} at the last "
                 f"{sm['window_h']:.0f} h's rate" + ("  " if markdown else ""))
    c = sm["commits_last_hour"]; tot = sum(c.values())
    L.append(f"rate: {sm['blocks_per_hour']:.1f} blocks/h, {sm['walkers_per_hour'] / 1e6:.2f} M walkers/h; hub: {sm['shards']} shards, "
             f"{sm['bytes_on_hub'] / 1e9:.1f} GB; commits in the last hour {tot} / {sm['commits_cap']} "
             f"(blocks {c['blocks']}, claims {c['claims']}, heartbeats {c['heartbeats']}, other {c['other']})")
    L.append("")
    head = ["contributor", "status", "shards", "walkers", "blocks/h", "ms/walker", "GPU h", "in flight"]
    rows = []
    for n, d in sm["contributors"].items():
        if d["shards"] == 0 and not d["inflight"]:
            continue
        fl = "; ".join(f"block {c['block']}{'.p%d' % c['pass_'] if c['pass_'] else ''}"
                       + (f" {100 * c['progress']:.0f} %" if c["progress"] is not None else "") + (f" eta {c['eta_min']:.0f} min" if c["eta_min"] else "")
                       + f" (hb {c['age_min']:.0f} min)" for c in d["inflight"]) or "—"
        rows.append([n, d["status"], str(d["shards"]) + (f" (+{d['whole']} whole)" if d["whole"] else ""), f"{d['walkers'] / 1e6:.2f} M",
                     f"{d['blocks_per_hour']:.1f}", f"{d['ms_per_walker']:.1f}" if d["ms_per_walker"] else "—", f"{d['walk_hours']:.1f}", fl])
    if markdown:
        L += ["| " + " | ".join(head) + " |", "|" + "---|" * len(head)] + ["| " + " | ".join(r) + " |" for r in rows]
    else:
        w = [max(len(x) for x in col) for col in zip(head, *rows)] if rows else [len(h) for h in head]
        L += ["  ".join(h.ljust(w[i]) for i, h in enumerate(head))] + ["  ".join(x.ljust(w[i]) for i, x in enumerate(r)) for r in rows]
    stale = [c for c in sm["claims"] if c["age"] is not None and c["age"] > sm["stale_s"]]
    if stale:
        L += ["", "stale claims (no heartbeat for over %d min): " % (sm["stale_s"] // 60)
              + ", ".join(f"{os.path.basename(c['file'])} ({c['age'] / 60:.0f} min)" for c in stale)]
    L += ["", "status: live = a claim with a fresh heartbeat; between blocks = no claim, a shard within the last %d min; idle = nothing for longer; "
          "stale = a claim without a heartbeat for over %d min (a dead worker's; released by the next worker that claims). "
          "GPU h = walk + pack time of the shards; ms/walker = walk time per walker at the block's scale." % (sm["stale_s"] // 60, sm["stale_s"] // 60)]
    return "\n".join(L)


def publish(hub, sm, path="STATUS.md"):
    """The rendered status committed to the repository (one commit)."""
    md = render(sm, markdown=True) + "\n\nWritten by `python -m dmipy_sim.fill.status --publish` from the repository's own files.\n"
    hub.commit({path: md.encode()}, [], f"status {sm['time']}: {sm['blocks_per_hour']:.1f} blocks/h")


def main(argv=None):
    from .hub import FakeHub, Hub
    ap = argparse.ArgumentParser(prog="python -m dmipy_sim.fill.status")
    ap.add_argument("--repo", default=None); ap.add_argument("--local", default=None); ap.add_argument("--variant", default=None)
    ap.add_argument("--cache", default=os.path.expanduser("~/.cache/dmipy-sim/fill-status"))
    ap.add_argument("--publish", action="store_true", help="commit STATUS.md to the repository (one commit)")
    ap.add_argument("--json", default=None, help="write the numbers here (for a supervisor)")
    ap.add_argument("--quiet", action="store_true")
    a = ap.parse_args(argv)
    if not a.repo and not a.local:
        ap.error("--repo OWNER/DATASET or --local DIR")
    hub = FakeHub(a.local) if a.local else Hub(a.repo)
    variant = a.variant or json.load(open(hub.get("manifest.json")))["default_variant"]
    sm = summarise(collect(hub, variant, None if a.local else a.cache))
    if not a.quiet:
        print(render(sm))
    if a.json:
        json.dump(sm, open(a.json, "w"), indent=1, default=float)
    if a.publish:
        publish(hub, sm)
        if not a.quiet:
            print("\nSTATUS.md published")


if __name__ == "__main__":
    main()
