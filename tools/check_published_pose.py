"""Replay the published packs through the pose route once, at the stamp step: a check run by hand, not a test.

The brain page replays two published packs on a GPU, the white-matter bundle (with its field channel) and the grey-matter
spheres. Whether a published pack replays is a property of the published bytes and, once stamped, of the envelope the stamp
states; the suite never reads one (it holds the route on two committed fixture packs, ``tests/test_pose_series.py``). For each
pack, scanner (the Swoop, the Prisma, the Terra) and echo (spin, stimulated) this plays the brain page's default timing (delta
25, Delta 55, TE 100 ms; a stimulated echo storing for TM = 30 ms; a b = 0 row and 30 directions at 1000 and 3000 s/mm^2
at the scanner's slew) over 24 voxels of a head up to 5.4 cm off isocentre, each its own class, and for the classes at the
first, middle and last voxel holds what the suite holds on the fixtures: the closed form and the series, under torch on the
device, each within the two misfits (and ``FLOAT32``) of the numpy closed form, the series' misfit below a tenth of the pack's
floor, and the series at three rotations within its misfit of the direct posed replay. A pack with a field channel is read at
each scanner's own field with its nominal tissue; one without, with its nominal tissue alone.

    python tools/check_published_pose.py                           # the brain page's two packs, read from the Hub
    python tools/check_published_pose.py PACK [PACK ...]           # packs by path or hf:// URI, e.g. the stamped copies
                                                                   # that dmipy-replay's tools/stamp_envelope.py --out DIR wrote
    python tools/check_published_pose.py --machine siemens_magnetom_prisma_3T --kind "spin echo" PACK
    python tools/check_published_pose.py --device cpu PACK         # a host without a CUDA device (slow)

Run from the repository root. It reads only: a Hub pack goes to the local download cache, and nothing is written to the Hub
or beside a pack. It prints, per pack, scanner and echo, the largest share of each bound the comparison used (1 is the
bound), and exits 1 when a comparison fails or a pack does not replay."""
import argparse
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]

WHITE_MATTER = "hf://SubstrateCommons/cactus-axons/packs/single_bundle_1s_c3_seg125ms.rpk"
GREY_MATTER = "hf://SubstrateCommons/grey-matter-spheres/packs/packed_spheres_leaky_250ms_c2_seg125ms.rpk"
MACHINES = ("hyperfine_swoop_64mT", "siemens_magnetom_prisma_3T", "siemens_magnetom_terra_7T")
KINDS = ("spin echo", "stimulated echo")
#: the closed form refuses an encoding that is not one direction per measurement; such a class is read by quadrature
NOT_ONE_DIRECTION = "the closed-form pose expansion needs a single-direction encoding"


def head_classes(machine, kind):
    """The classes ``machine`` plays the brain page's default timing as over 24 voxels of a head, one per voxel."""
    from dmipy_replay import sequences
    from dmipy_replay.acquisition.scanners import ScannerLimits
    from dmipy_replay.phantom.bore import encoding_classes
    from dmipy_replay.phantom.grid import Grid
    rng = np.random.default_rng(5)
    dirs = rng.normal(size=(30, 3))
    dirs /= np.linalg.norm(dirs, axis=1, keepdims=True)
    dirs = np.vstack([[0.0, 0.0, 1.0], dirs, dirs])
    b = np.array([0.0] + [1e9] * 30 + [3e9] * 30)
    slew = float(ScannerLimits.of(machine).slew_max)
    if kind == "spin echo":
        seq = sequences.pgse(dirs, 0.025, 0.055, bvalues=b, TE=0.100, n_t=1000, slew_rate=slew)
    else:
        seq = sequences.pgste(dirs, 0.025, 0.030, bvalues=b, n_t=1000, slew_rate=slew, ste_flip_angles=(90.0, 90.0, 90.0))
    grid = Grid(shape=(4, 3, 2), voxel_size_m=(2e-2,) * 3, origin_m=(-0.03, -0.02, -0.01), isocenter_m=(0.0, 0.0, 0.0))
    idx = np.array([[i, j, k] for i in range(4) for j in range(3) for k in range(2)])
    _cls, played = encoding_classes(ScannerLimits.of(machine), grid, seq, idx, tolerance=None)
    return played


def check_pack(uri, machines, kinds, device):
    """Check the pack at ``uri`` on every scanner and echo; the list of the comparisons that failed."""
    from dmipy_replay import ReplayPack
    from dmipy_replay.acquisition.scanners import ScannerLimits
    from dmipy_replay.replay.hub import is_hub_uri
    from tests.test_pose_series import _series_vs_direct_replay, _torch_route_vs_closed_form
    pack = ReplayPack.load(uri, windows=range(1)) if is_hub_uri(uri) else ReplayPack.load(uri)   # the echo is within window 0
    field = pack.susceptibility_field == "present"
    keep = (8, 0) if field else (0, 0)                              # the ODF band the page composes with
    print(f"{uri}: {pack.n_walkers} walkers, window {pack.T_window * 1e3:g} ms, field {'present' if field else 'absent'}")
    failed = []
    for machine in machines:
        kw = dict(tissue=pack.nominal)
        if field:
            kw["scanner"] = float(ScannerLimits.of(machine).field_T)
        for kind in kinds:
            t0 = time.time()
            played = head_classes(machine, kind)
            last = len(played) - 1
            used, skipped = {}, []
            for k in sorted({0, len(played) // 2, last}):
                try:
                    r = _torch_route_vs_closed_form(pack, played, k, device=device, keep=keep, **kw)
                    used = {n: max(used.get(n, 0.0), v) for n, v in r.items()}
                except ValueError as e:
                    if not str(e).startswith(NOT_ONE_DIRECTION):
                        failed.append(f"{uri} {machine} {kind} class {k}: {e}")
                        continue
                    skipped.append(k)
                except AssertionError as e:
                    failed.append(f"{uri} {machine} {kind} class {k}: {e}")
            try:
                used["direct"] = _series_vs_direct_replay(pack, played, last, device=device, **kw)
            except (AssertionError, ValueError) as e:
                failed.append(f"{uri} {machine} {kind} class {last} against the direct replay: {e}")
            shares = ", ".join(f"{n} {v:.3f}" for n, v in used.items())
            print(f"  {machine:28s} {kind:16s} {shares}" + (f" (closed form refused classes {skipped})" if skipped else "")
                  + f"  [{time.time() - t0:.0f} s]", flush=True)
    return failed


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("packs", nargs="*", help="packs by path or hf:// URI (default: the brain page's two)")
    ap.add_argument("--device", default="cuda", help="the torch device (default cuda)")
    ap.add_argument("--machine", action="append", choices=MACHINES, help="a scanner (default: all three); repeatable")
    ap.add_argument("--kind", action="append", choices=KINDS, help="an echo (default: both); repeatable")
    a = ap.parse_args(argv)
    sys.path.insert(0, str(ROOT))
    failed = []
    for uri in a.packs or [WHITE_MATTER, GREY_MATTER]:
        try:
            failed += check_pack(uri, a.machine or MACHINES, a.kind or KINDS, a.device)
        except Exception as e:                                       # a pack that does not load or replay is a result
            failed.append(f"{uri}: {type(e).__name__}: {e}")
    for f in failed:
        print("FAILED", f)
    print("all comparisons hold" if not failed else f"{len(failed)} failed")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
