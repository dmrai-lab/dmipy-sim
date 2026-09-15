"""The stages of a block and their overlap: seed -> walk -> save -> pack -> upload, so the device only ever waits
on itself.

The next round's seeds are drawn on the CPU (:func:`dmipy_sim.spec.draw_seeds`) in a thread while this round
walks, the next block claimed ahead for its first round; a walked round is written to its file in a thread while
the next round walks; the pack runs in a subprocess on the CPU over the block's walk files (its rounds merged
with ``overlap="recertify"``); the upload is one commit in a thread, the local files kept until the hub's sha256
matches. Every stage has queue depth one (at most one round saving, one pack, one upload), so a slow pack or
upload makes the walk wait rather than stacking packs on disk: :func:`Fill.settle` is the only back-pressure.
The walk context (:meth:`Recipe.context`) is built once per process. Measured on the GH200 before the seeds and
saves left the device's path (dmipy-sim#258): 19 s of seeding and 21 s of saving around a 132 s pass-1 walk, the
device busy 77 % of the time.

A plan may be filled in passes (``plan.passes`` in the manifest): a pass of a block is its own shard walked from
its own seed stream (round ``r`` of pass ``K`` of block ``b``: seed ``b_seed * 1000 + 100 (K - 1) + r``), the
first pass of every block before the second; a consumer merges a block's passes with
``merge_packs(overlap="recertify")``. A round that holds none of a sparse pool walks without it (#271)."""
import glob
import json
import logging
import os
import shutil
import subprocess
import sys
import threading
import time
from dataclasses import dataclass, field

import numpy as np

from . import claims as C
from .hub import sha256_of
from .recipe import FULL

log = logging.getLogger("dmipy_sim.fill")


@dataclass
class Options:
    """What a worker was asked to do. ``workdir`` holds the walk files, run records, job files and packs of the
    blocks in flight; ``host`` names this machine in claims and summaries."""
    workdir: str
    host: str
    repo: str = ""
    block: int = None                 # one block (else the next open one)
    loop: bool = False                # fill blocks until none is open
    hours: float = None               # claim nothing new after this many hours
    only_pass: int = None             # fill only this pass of the plan
    claim_batch: int = 3              # blocks claimed per commit
    budget: int = None                # scale the block's counts to this many walkers (rehearsals, certifying walks)
    max_walkers: int = None           # walk a block in rounds of at most this many
    smoke: bool = False               # upload under smoke/ (never claims)
    no_upload: bool = False
    keep: bool = False                # keep the local shard and run records after the upload
    certify: bool = False             # a certifying walk: measured fidelity, under certificate/
    batch: int = None                 # walker_batch_size (default: the manifest's)
    pack_device: str = "numpy"        # where the pack subprocess runs its transforms: numpy, jax, auto
    require_gpu: bool = True
    devices: list = field(default_factory=list)

    @property
    def claims(self):
        return not (self.smoke or self.certify or self.no_upload)

    @property
    def prefix(self):
        return "certificate" if self.certify else "smoke" if self.smoke else "blocks"


# ----------------------------------------------------------------------------------------------------- stages
def round_seeding(rc, row, r, k, P, budget):
    """Round ``r`` of ``k`` of pass ``P`` of the block as a seeding: the plan's counts at the pass's scale dealt
    into rounds (every voxel's share exact), and the round's seed of the pass's own stream."""
    from ..spec import StratifiedByVoxel
    W = rc.man["walk"]; spec = rc.spec(); grid = rc.grid()
    want, tot, scale = rc.counts(row, budget, P["scale"])
    for p in spec.pools:                                   # a pool the plan does not name (the myelin water) gets none
        if p.id in spec.seeding.pools:
            want.setdefault(p.name, np.zeros(grid.shape, np.int64))
    if k > 1:
        want = {n_: (np.floor(v * (r + 1) / k) - np.floor(v * r / k)).astype(np.int64) for n_, v in want.items()}
    seeding = StratifiedByVoxel(grid=grid, walkers_per_voxel=want, census_draws=int(W.get("census_draws", 200)))
    seed = row["seed"] * 1000 + 100 * ((P.get("pass") or 1) - 1) + r
    return seeding, seed, {n: int(v.sum()) for n, v in want.items()}, tot, scale


def draw_round(rc, row, r, k, P, budget):
    """The round's seeds drawn on the CPU: what the worker does for the next round in a thread while the device
    walks this one."""
    from ..spec import draw_seeds
    seeding, seed, n_plan, tot, scale = round_seeding(rc, row, r, k, P, budget)
    return dict(drawn=draw_seeds(rc.spec(), seeding, seed, context=rc.context()), seed=seed, n_plan=n_plan, tot=tot, scale=scale)


def round_paths(workdir, name, r, k):
    tag = f".round{r}" if k > 1 else ""
    return os.path.join(workdir, f"{name}{tag}.run"), os.path.join(workdir, f"{name}{tag}.walk")


def walk_round(o, rc, row, name, r, k, P, seeds):
    """The walk stage (the device): round ``r`` of ``k`` of pass ``P`` of the block from its drawn seeds, walked
    with the manifest's walk, spooled into its run record. Returns ``(walk, record)``; :func:`save_walk` writes
    the walk (a thread, off the device's path)."""
    from ..spec import walk_spec
    W = rc.man["walk"]; spec = rc.spec()
    log.info("%s round %d/%d: voxels %s, plan %d walkers, scale %.4f -> %s", name, r + 1, k, row["voxels"], seeds["tot"], seeds["scale"], seeds["n_plan"])
    run_dir, out = round_paths(o.workdir, name, r, k)
    t0 = time.time()
    w = walk_spec(spec, T_max=W["T_max_s"], dt_save=rc.dt_save(), seeding=seeds["drawn"], seed=seeds["seed"], field=bool(spec.field_source_pools),
                  require_gpu=o.require_gpu, walker_batch_size=o.batch or W["walker_batch_size"], adaptive_steps=W["adaptive_steps"], scanner=W["scanner"],
                  floor_fraction=W["floor_fraction"], field_sample_every=int(W.get("field_sample_every", 1)),
                  context=rc.context(), field_gather_every=int(W.get("field_gather_every", 4)), run_dir=run_dir, spool=True)
    t_walk = time.time() - t0
    n_r, n_t = w.positions.shape[:2]
    pools = np.bincount(np.asarray(w.compartment)[:, 0].astype(int)).tolist()
    log.info("%s round %d/%d: walked %d walkers x %.0f ms (n_t %d, dt_save %.3g s) in %.0f s; pools %s", name, r + 1, k, n_r,
             W["T_max_s"] * 1e3, n_t, float(w.dt), t_walk, pools)
    rd = dict(walk=out, run_dir=run_dir, n_walkers=int(n_r), n_t=int(n_t), dt_save=float(rc.dt_save()), t_walk=t_walk, pools=pools,
              scale=seeds["scale"], plan=seeds["tot"], round=r)
    return w, rd


def save_walk(w, rd):
    """The walk written as one file (the handoff to the pack stage); the spool dropped once the file is whole."""
    t0 = time.time(); w.save(rd["walk"])
    shutil.rmtree(os.path.join(rd["run_dir"], "spool"), ignore_errors=True)
    log.info("%s saved in %.0f s", os.path.basename(rd["walk"]), time.time() - t0)


def pack_job(job):
    """The pack stage (a subprocess on the CPU): the block's walk files packed, the rounds merged, the summary
    and the certificate written beside the shard. ``job`` is the JSON the walk stage wrote."""
    from ..persistent_walk import PersistentWalk
    from ..replay.bank import build_replay_pack, merge_packs, voxel_fidelity_volumes
    from ..phantom import Grid
    man, P = job["manifest"], job["manifest"]["pack"]
    G = man["grid"]; grid = Grid(shape=tuple(G["shape"]), voxel_size_m=tuple(G["voxel_size_m"]), origin_m=tuple(G["origin_m"]))
    fid = dict(fidelity="measured", blt_temporal_K=P["blt_K"]) if job["certify"] else dict(fidelity="inherited", fidelity_from=job["certificate"])
    rounds = job["rounds"]; k = len(rounds); out = job["out"]; t0 = time.time(); packs = []
    for r, rd in enumerate(rounds):
        w = PersistentWalk.load(rd["walk"])
        out_r = out if k == 1 else out[:-4] + f".round{r}.rpk"
        pk = build_replay_pack(w, id=f"{man['id']}/{job['variant']}/{job['name']}{'-certifying' if job['certify'] else ''}" + (f"/round-{r}" if k > 1 else ""),
                               license=man["license"], citation=man["citation"], K=P["K"], position_container=P["position_container"],
                               blt_container=P["blt_container"], susc_path_K=(P["K_path"] if w.field_samples is not None else None), voxel_grid=grid,
                               out_path=out_r, device=job["device"], provenance=dict(job["provenance"], round=(r if k > 1 else None), rounds=k), **fid)
        packs.append(out_r); del w
    if k > 1:
        pk = merge_packs(packs, id=f"{man['id']}/{job['variant']}/{job['name']}", out_path=out, overlap="recertify", device=job["device"])
        for f_ in packs:
            os.remove(f_)
    t_pack = time.time() - t0
    g_, floors, cnts = voxel_fidelity_volumes(pk)
    cert = {n: dict(voxels=int((cnts[n] > 0).sum()), walkers=int(cnts[n].sum()),
                    floor_median=float(np.median(floors[n][cnts[n] > 1])) if (cnts[n] > 1).any() else None,
                    floor_max=float(floors[n][cnts[n] > 1].max()) if (cnts[n] > 1).any() else None) for n in floors}
    n_w = sum(rd["n_walkers"] for rd in rounds)
    pools = [sum(x) for x in zip(*[rd["pools"] + [0] * (max(len(q["pools"]) for q in rounds) - len(rd["pools"])) for rd in rounds])]
    summary = dict(block=job["block"], variant=job["variant"], host=job["host"], devices=job["devices"], commit=man["code"]["commit"],
                   **{"pass": job.get("pass"), "pass_scale": job.get("pass_scale", 1.0)},
                   certified=pk.meta["fidelity"]["certified"], floor_max=float(pk.meta["fidelity"]["floor_max"]),
                   box=job["box"], seed=job["seed"], budget=job["budget"], scale=rounds[0]["scale"], walkers=int(n_w), pools=pools,
                   n_t=rounds[0]["n_t"], dt_save_s=rounds[0]["dt_save"], t_walk_s=sum(rd["t_walk"] for rd in rounds), t_pack_s=t_pack,
                   size_bytes=os.path.getsize(out), sha256=sha256_of(out), certificate=cert, rounds=k,
                   run=pk.meta.get("provenance", {}).get("run"), finished=C.stamp())
    json.dump(summary, open(out[:-4] + ".json", "w"), indent=1)
    if job["certify"]:
        json.dump(pk.meta, open(out[:-4] + ".certificate.json", "w"), default=float)
    for rd in rounds:                                      # the walk files were the handoff; the shard is the artifact
        os.remove(rd["walk"])
    log.info("%s: packed in %.0f s -> %s (%.0f MB, %.2f kB/walker); certificate %s", job["name"], t_pack, out, os.path.getsize(out) / 1e6,
             os.path.getsize(out) / 1e3 / max(n_w, 1), cert)


def upload_block(hub, o, job):
    """The upload stage (a thread): the shard, its summary, the certificate of a certifying block, the walk's run
    records and the claim's release in ONE commit; every file verified by sha256, then the local files removed."""
    out, name, variant = job["out"], job["name"], job["variant"]
    prefix = job["prefix"]; tag = f"{variant}-" if job["certify"] else ""
    adds = {}
    if job["certify"]:
        adds[f"certificate/{variant}.json"] = out[:-4] + ".certificate.json"
    for ext in (".rpk", ".json"):
        adds[f"{prefix}/{tag}{name}{ext}"] = out[:-4] + ext
    for rd in job["rounds"]:                               # the record of the walk that made it (#257)
        tag_r = f"/round-{rd['round']}" if len(job["rounds"]) > 1 else ""
        for f_ in ("manifest.json", "events.jsonl", "summary.json"):
            p = os.path.join(rd["run_dir"], f_)
            if os.path.isfile(p):
                adds[f"{prefix}/{tag}{name}.run{tag_r}/{f_}"] = p
    kind = "certificate " if job["certify"] else "smoke " if prefix.startswith("smoke") else ""
    hub.commit(adds, [job["claim"]] if job.get("claim") else [], f"{kind}{variant} {name}: {job['n_walkers']} walkers from {job['host']}")
    log.info("uploaded %s/%s to %s (verified, one commit)", prefix, name, hub.repo)
    if not o.keep:
        for ext in (".rpk", ".json", ".certificate.json"):
            if os.path.isfile(out[:-4] + ext):
                os.remove(out[:-4] + ext)
        for rd in job["rounds"]:
            shutil.rmtree(rd["run_dir"], ignore_errors=True)
    os.remove(job["file"])


# ------------------------------------------------------------------------------------------------- the worker
class Fill:
    """A worker: the pipeline's state over the blocks it fills. :meth:`run` fills; :meth:`drain` finishes what
    an earlier worker on this host left in the work directory."""

    def __init__(self, hub, rc, o):
        self.hub, self.rc, self.o = hub, rc, o
        os.makedirs(o.workdir, exist_ok=True)
        self.packing = {"proc": None, "job": None}; self.uploading = {"thread": None, "error": None}
        self.post = {"threads": [], "error": None}; self.post_lock = threading.RLock()   # the pack/upload handoff: one thread at a time
        self.prefetch = {"thread": None, "key": None, "seeds": None, "error": None}
        self.state = {"walking": None, "held": {}}; self.stop = threading.Event()   # held: every claim in the pipeline, by name
        self.cert = None
        if not o.certify:                                  # the fill's certificate: measured once, inherited by every block
            try:
                self.cert = json.load(open(hub.get(f"certificate/{rc.variant}.json")))
            except Exception as e:
                raise SystemExit(f"no certificate at certificate/{rc.variant}.json for variant {rc.variant}: fill one block with certify first ({e})")
        if o.certify and o.budget is None:
            raise SystemExit("a certifying walk takes a budget: it is a scaled block (its pack holds the dense oracle in host memory)")

    # ---- pack + upload, depth one each
    def start_pack(self, job):
        """The pack subprocess for a block whose rounds are all saved; a waiter thread hands its shard to the upload
        the moment it ends (on a slow device a block's walk is an hour, and its shard must not wait for the next)."""
        jf = job["file"]; json.dump(job, open(jf, "w"), indent=1, default=float)
        env = dict(os.environ, JAX_PLATFORMS="cpu") if self.o.pack_device == "numpy" else dict(os.environ)
        pkg = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))   # the same dmipy_sim as this process
        env["PYTHONPATH"] = pkg + (os.pathsep + env["PYTHONPATH"] if env.get("PYTHONPATH") else "")
        proc = subprocess.Popen([sys.executable, "-m", "dmipy_sim.fill", "--pack", jf], env=env)
        cs = job.get("claim_state") or dict(block=job["block"], variant=job["variant"], host=job["host"], name=job["name"], claim=job.get("claim"),
                                            round=None, **{"pass": job.get("pass")}, run_dir=None, commit=job["manifest"]["code"]["commit"], started=C.stamp())
        self.packing["proc"] = proc; self.packing["job"] = job; self.state["held"][job["name"]] = dict(cs, stage="packing")
        log.info("%s: pack started (pid %d, device %s)", job["name"], proc.pid, self.o.pack_device)

        def waiter():
            proc.wait()
            try:
                with self.post_lock:
                    self.finish_pack()
            except BaseException as e:
                self.post["error"] = e
        threading.Thread(target=waiter, daemon=True).start()

    def finish_pack(self):
        """The pack in flight, waited for; its shard handed to the upload thread (after the one before). Called by
        the waiter thread, by the post thread that starts the next pack, and by the loop's end: the lock keeps
        one handoff at a time."""
        with self.post_lock:
            self._finish_pack()

    def _finish_pack(self):
        if self.packing["proc"] is None:
            return
        rc_ = self.packing["proc"].wait(); job = self.packing["job"]; self.packing["proc"] = None; self.packing["job"] = None
        if rc_ != 0:
            raise RuntimeError(f"{job['name']}: the pack subprocess failed (exit {rc_}); its walk files are kept in {self.o.workdir}")
        if self.o.no_upload:
            log.info("%s: packed, not uploaded", job["name"]); os.remove(job["file"]); self.state["held"].pop(job["name"], None); return
        self.join_upload()
        if job["name"] in self.state["held"]:
            self.state["held"][job["name"]]["stage"] = "uploading"

        def go():
            try:
                upload_block(self.hub, self.o, job)
            except BaseException as e:
                self.uploading["error"] = e
            finally:
                self.state["held"].pop(job["name"], None)
        self.uploading["thread"] = threading.Thread(target=go, daemon=False); self.uploading["thread"].start()

    def join_upload(self):
        with self.post_lock:
            t = self.uploading["thread"]
        if t is not None:
            t.join()
            with self.post_lock:
                if self.uploading["thread"] is t:
                    self.uploading["thread"] = None
            if self.uploading["error"] is not None:
                raise self.uploading["error"]

    # ---- the save thread per round, then the pack
    def post_round(self, w, rd, job):
        """Save the walk in a thread, then (``job`` given: the block's last round) pack it once every earlier
        round's file is whole; the pack stages keep their depth of one under the lock."""
        def go():
            try:
                save_walk(w, rd)
                if job is not None:
                    for t_ in list(self.post["threads"]):
                        if t_ is not threading.current_thread():
                            t_.join()
                    with self.post_lock:
                        self.finish_pack(); self.start_pack(job)
            except BaseException as e:
                self.post["error"] = e
        t = threading.Thread(target=go, daemon=False); t.start(); self.post["threads"].append(t)

    def settle(self, keep=1):
        """Wait until at most ``keep`` post threads are still running (the device's queue depth), and raise what
        one of them raised."""
        while len([t for t in self.post["threads"] if t.is_alive()]) > keep:
            time.sleep(0.2)
        self.post["threads"] = [t for t in self.post["threads"] if t.is_alive()]
        if self.post["error"] is not None:
            raise self.post["error"]

    # ---- the seeds drawn ahead
    def prefetch_start(self, row, r, k, P, key):
        def go():
            try:
                self.prefetch["seeds"] = draw_round(self.rc, row, r, k, P, self.o.budget)
            except BaseException as e:
                self.prefetch["error"] = e
        self.prefetch.update(thread=threading.Thread(target=go, daemon=True), key=key, seeds=None, error=None); self.prefetch["thread"].start()

    def prefetch_take(self, row, r, k, P, key):
        if self.prefetch["thread"] is not None and self.prefetch["key"] == key:
            self.prefetch["thread"].join(); self.prefetch["thread"] = None
            if self.prefetch["error"] is not None:
                raise self.prefetch["error"]
            return self.prefetch["seeds"]
        return draw_round(self.rc, row, r, k, P, self.o.budget)   # nothing drawn ahead for this round: drawn now

    # ---- the claims
    def claim_first(self):
        o, rc = self.o, self.rc
        if o.block is not None:
            P = FULL if o.certify else next((P_ for P_ in rc.passes if P_.get("pass") == o.only_pass), rc.passes[0])
            return C.claim_block(self.hub, rc, o.block, o.host, P, write=o.claims)
        return C.claim_next(self.hub, rc, o.host, only_pass=o.only_pass, claim_batch=o.claim_batch, write=o.claims)

    def job_of(self, claimed, rounds, out):
        o, rc, row, P = self.o, self.rc, claimed["row"], claimed["P"]
        return dict(file=os.path.join(o.workdir, f"{claimed['name']}.job.json"), block=claimed["block"], name=claimed["name"], variant=rc.variant,
                    host=o.host, devices=o.devices, certify=bool(o.certify), certificate=self.cert, manifest=rc.man, rounds=list(rounds), out=out,
                    device=o.pack_device, prefix=(o.prefix if o.certify else f"{o.prefix}/{rc.variant}"), claim=claimed["claim"], budget=o.budget,
                    seed=row["seed"], box=dict(i=row["i"], j=row["j"], k=row["k"]), n_walkers=sum(rd["n_walkers"] for rd in rounds),
                    **{"pass": P.get("pass"), "pass_scale": P["scale"]},
                    provenance=dict(fill=dict(dataset=o.repo, variant=rc.variant, block=claimed["block"], host=o.host, budget=o.budget, code=rc.man["code"],
                                              **{"pass": P.get("pass"), "pass_scale": P["scale"]}, plan=rc.man["plan"]["file"], blocks=rc.man["plan"]["blocks"])),
                    claim_state=dict(block=claimed["block"], variant=rc.variant, host=o.host, name=claimed["name"], claim=claimed["claim"], round=None,
                                     **{"pass": P.get("pass")}, run_dir=None, commit=rc.commit, started=C.stamp()))

    # ---- the loop
    def run(self, *, heartbeat_every=None):
        """Fill: the first block (claimed), then, with ``loop``, every next open block until none is open or
        ``hours`` have passed; the pipeline drained at the end. On an error every claim this worker holds is
        released and the error re-raised."""
        o, rc, hub = self.o, self.rc, self.hub
        from ..run import report                           # every dmipy_sim import on the main thread, before any thread
        rc.context()                                       # the spec, geometries and field basis: once, before any thread
        hb = threading.Thread(target=C.heartbeat, args=(hub, self.state, self.stop, report), kwargs=dict(every=heartbeat_every), daemon=True)
        if o.claims:
            hb.start()
        t_start = time.time(); claimed = self.claim_first(); pending = None
        try:
            while claimed is not None:
                row, name, block, P = claimed["row"], claimed["name"], claimed["block"], claimed["P"]
                k = rc.rounds(row, o.budget, P, o.max_walkers)
                if k > 1 and o.certify:
                    raise SystemExit("a certifying walk is one round: lower the budget instead of max_walkers")
                out = os.path.join(o.workdir, f"{rc.variant}-{name}{'-certifying' if o.certify else ''}.rpk")
                rounds = []; nxt = None
                for r in range(k):
                    self.settle(keep=1)                    # the device's queue: at most one round still saving
                    seeds = self.prefetch_take(row, r, k, P, (name, r))
                    if r + 1 < k:                          # the next round's seeds, drawn while this one walks
                        self.prefetch_start(row, r + 1, k, P, (name, r + 1))
                    elif o.loop and not o.certify and not o.smoke and (o.hours is None or (time.time() - t_start) / 3600 < o.hours):
                        nxt = pending = C.claim_next(hub, rc, o.host, only_pass=o.only_pass, claim_batch=o.claim_batch, write=o.claims)
                        if nxt is not None:                # the next block's first round, likewise
                            self.prefetch_start(nxt["row"], 0, rc.rounds(nxt["row"], o.budget, nxt["P"], o.max_walkers), nxt["P"], (nxt["name"], 0))
                    cur = dict(block=block, variant=rc.variant, host=o.host, name=name, claim=claimed["claim"], round=(r if k > 1 else None), stage="walking",
                               **{"pass": P.get("pass")}, run_dir=round_paths(o.workdir, name, r, k)[0], commit=rc.commit, started=C.stamp())
                    self.state["walking"] = cur; self.state["held"][name] = cur
                    w, rd = walk_round(o, rc, row, name, r, k, P, seeds); rounds.append(rd)
                    self.state["walking"] = None
                    if r + 1 < k:
                        self.state["held"].pop(name, None)  # between rounds: covered again by the next round's walk
                    self.post_round(w, rd, self.job_of(claimed, rounds, out) if r + 1 == k else None)
                    del w
                claimed = nxt; pending = None
                if claimed is None and o.loop and not o.certify and not o.smoke:
                    log.info("no open block for variant %s", rc.variant)
            self.settle(keep=0)
            self.finish_pack()
            self.join_upload()
        except BaseException:
            C.release(hub, claimed, "the worker failed")
            if pending is not None and pending is not claimed:
                C.release(hub, pending, "the worker failed")
            for t in self.post["threads"]:
                t.join()
            if self.packing["job"] is not None:
                C.release(hub, dict(claim=self.packing["job"].get("claim"), name=self.packing["job"]["name"]), "the worker failed")
            raise
        finally:
            C.release_queue(hub, o.host)
            self.stop.set()

    def drain(self):
        """What an earlier worker on this host left in the work directory: finished walk files packed, finished
        packs uploaded, this host's other claims released."""
        o, rc, hub = self.o, self.rc, self.hub
        for jf in sorted(glob.glob(os.path.join(o.workdir, "*.job.json"))):
            job = json.load(open(jf))
            if not os.path.isfile(job["out"]) and all(os.path.isfile(rd["walk"]) for rd in job["rounds"]):
                self.start_pack(job); self.finish_pack()
            elif os.path.isfile(job["out"]) and os.path.isfile(job["out"][:-4] + ".json"):
                self.join_upload(); upload_block(hub, o, job)
        self.join_upload()
        for f in C.mine(hub, rc.variant, o.host):         # a claim of this host with nothing here to finish: open again
            hub.delete(f, f"release {f}: drained on {o.host}"); log.info("released %s", f)
