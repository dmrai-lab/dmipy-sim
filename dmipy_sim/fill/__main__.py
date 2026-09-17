"""The fill worker as a command:

    python -m dmipy_sim.fill --repo OWNER/DATASET --next --loop            # claim, fill and upload blocks until none is open
    python -m dmipy_sim.fill --repo OWNER/DATASET --next --loop --pass 1   # only the first pass of the plan
    python -m dmipy_sim.fill --repo OWNER/DATASET --block 17
    python -m dmipy_sim.fill --repo OWNER/DATASET --block 0 --budget 30000 --smoke     # a scaled block into smoke/, for a new machine
    python -m dmipy_sim.fill --repo OWNER/DATASET --block 0 --budget 200000 --certify  # the fill's certificate, measured once
    python -m dmipy_sim.fill --repo OWNER/DATASET --drain                  # finish what an earlier worker here left, release the rest
    python -m dmipy_sim.fill --local DIR --block 0 --budget 3000 --cpu     # a rehearsal against a directory, nothing uploaded

Needs dmipy-sim at the manifest's commit (with a CUDA jaxlib unless ``--cpu``) and, against the hub, a login with
write access to the repository."""
import argparse
import json
import logging
import os
import socket

from .hub import Hub, FakeHub
from .pipeline import Fill, Options, pack_job
from .recipe import Recipe


def main(argv=None):
    ap = argparse.ArgumentParser(prog="python -m dmipy_sim.fill", description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--block", type=int, help="the block index in the plan's table")
    g.add_argument("--next", action="store_true", help="claim the lowest block with no shard and no claim")
    g.add_argument("--drain", action="store_true", help="upload what an earlier worker on this host left finished in --workdir "
                   "(walk files are packed first), release this host's other claims, then stop")
    g.add_argument("--pack", default=None, help=argparse.SUPPRESS)          # the pack stage: a job file (the worker's own subprocess)
    ap.add_argument("--repo", default=None, help="the dataset repository (owner/name)")
    ap.add_argument("--local", default=None, help="read the recipe from this directory instead of the hub (implies --no-upload)")
    ap.add_argument("--loop", action="store_true", help="with --next: fill blocks until none is open, pipelined")
    ap.add_argument("--hours", type=float, default=None, help="with --loop: claim no new block once this many hours have passed "
                    "(a session with a time limit; the block in flight is finished and uploaded)")
    ap.add_argument("--variant", default=None, help="a key of manifest['variants'] (default: manifest['default_variant'])")
    ap.add_argument("--claim-batch", type=int, default=3, help="with --next: claim this many blocks in one commit and fill them in turn")
    ap.add_argument("--pass", dest="only_pass", type=int, default=None, help="with a plan in passes: fill only this pass")
    ap.add_argument("--budget", type=int, default=None, help="scale the block's planned counts to this many walkers")
    ap.add_argument("--duty", type=float, default=1.0, help="the device's duty (0-1]: after each walk the worker pauses walk_time * (1/duty - 1)")
    ap.add_argument("--duty-file", default=None, help="a file holding the duty, read before every walk (change it without a restart)")
    ap.add_argument("--max-walkers", type=int, default=None, help="walk a block whose plan exceeds this in rounds of at most this many "
                    "walkers and merge the rounds (a walk file holds ~85 kB per walker; the pack of a round holds it in host memory)")
    ap.add_argument("--smoke", action="store_true", help="upload under smoke/ instead of blocks/ (never claims)")
    ap.add_argument("--no-upload", action="store_true")
    ap.add_argument("--keep", action="store_true", help="keep the local shard and run records after the upload")
    ap.add_argument("--certify", action="store_true", help="a CERTIFYING walk: the block's voxels at --budget walkers, the full fidelity "
                    "battery measured (the dense oracle holds ~300 bytes per walker-save on the host), the pack under "
                    "certificate/<variant>-block-NNNN.rpk and its meta as certificate/<variant>.json, which every block then inherits")
    ap.add_argument("--workdir", default=os.path.join(os.getcwd(), "fill_work"))
    ap.add_argument("--batch", type=int, default=None, help="walker_batch_size (default: the manifest's; lower it on a small card)")
    ap.add_argument("--pack-device", default="numpy", help="where the pack subprocess runs its transforms: numpy (the CPU, so the walk "
                    "keeps the GPU), jax, auto")
    ap.add_argument("--cpu", action="store_true", help="walk on the CPU (a rehearsal)")
    ap.add_argument("--host", default=None, help="this machine's name in claims and summaries (default: the hostname)")
    a = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    logging.getLogger("jax").setLevel(logging.WARNING)
    if a.pack:
        return pack_job(json.load(open(a.pack)))
    if not a.repo and not a.local:
        ap.error("--repo OWNER/DATASET or --local DIR")
    hub = FakeHub(a.local) if a.local else Hub(a.repo)
    rc = Recipe(hub, a.variant)
    import jax                                             # every dmipy_sim import on the main thread, before any thread
    from ..spec import walk_spec, StratifiedByVoxel, disco_spec        # noqa: F401  (what the walk stage imports)
    from ..acquisition.scanners import save_interval                    # noqa: F401
    from ..phantom import Grid                                          # noqa: F401
    devices = [str(d) for d in jax.devices()]; logging.getLogger("dmipy_sim.fill").info("devices %s", devices)
    o = Options(workdir=a.workdir, host=a.host or socket.gethostname(), repo=a.repo or f"local:{a.local}", block=a.block, loop=a.loop,
                hours=a.hours, only_pass=a.only_pass, claim_batch=a.claim_batch, budget=a.budget, max_walkers=a.max_walkers, smoke=a.smoke,
                no_upload=a.no_upload or bool(a.local), keep=a.keep, certify=a.certify, batch=a.batch, pack_device=a.pack_device,
                duty=a.duty, duty_file=a.duty_file, require_gpu=not a.cpu, devices=devices)
    f = Fill(hub, rc, o)
    if a.drain:
        return f.drain()
    return f.run()


if __name__ == "__main__":
    main()
