"""Build the two replay packs the CUDA pose test reads, under ``tests/data/``: ``cylinder_k12.rpk`` (a cylinder, no field
channel) and ``strands_field_k4.rpk`` (three sheathed strands with the susceptibility path channel). They are the packs the
CPU pose tests walk in their fixtures (``walk_pack`` of ``tests/test_pose_series.py`` and ``tests/test_pose_torch.py``,
``field_pack`` of ``tests/conftest.py``), written to files, so that
``tests/test_pose_series.py::test_the_torch_route_on_a_cuda_device_is_the_closed_form_within_the_misfits`` reads a pack on the
device and walks nothing.

    python tools/build_pose_fixtures.py             # writes the two files under tests/data/ and prints their sizes
    python tools/build_pose_fixtures.py --out DIR   # writes them under DIR instead

Run from the repository root. The packs are synthetic (no third-party data), 74 KiB in all, and are rebuilt, and committed
again, when the pack format or the walk's certificate changes."""
import argparse
import os
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "tests" / "data"

CYLINDER = "cylinder_k12.rpk"
STRANDS = "strands_field_k4.rpk"
LICENSE = "CC-BY-4.0"
CITATION = "dmipy-sim test fixture (tools/build_pose_fixtures.py)"


def cylinder(out_path):
    """400 walkers in a cylinder of radius 2 um (a one-cylinder periodic pack of 8 um), 12 ms in blocks of 0.5 ms,
    stored at K = 12."""
    import dmipy_sim as d
    from dmipy_sim.engine.core import _walk_geometry
    from dmipy_sim.replay.bank import build_replay_pack
    from tests._envelope import stated
    walk = _walk_geometry(400, 2e-9, d.PackedCylinders([2e-6], [[0.0, 0.0]], 8e-6), 12e-3, 5e-4, seed=4, require_gpu=False)
    return build_replay_pack(walk, id="fixture/cylinder_k12", license=LICENSE, citation=CITATION, K=12,
                             out_path=str(out_path), **stated(of=walk))


def strands(out_path):
    """90 walkers among three sheathed strands, 0.8 ms in blocks of 0.2 ms, the field sampled in the walk (adaptive
    steps), stored at K = 4 with the path channel at 4 modes."""
    from dmipy_sim import walk
    from dmipy_sim.replay.bank import build_replay_pack
    from tests._envelope import stated, walk_envelope
    from tests.conftest import _three_strands
    with tempfile.TemporaryDirectory(prefix="dmipy-pose-fixture-") as tmp:
        spec = _three_strands(Path(tmp))
        w = walk(spec, 90, 8e-4, 2e-4, seed=0, n_probe=20_000, require_gpu=False, adaptive_steps=True, K=4,
                 envelope=walk_envelope(of=spec))
        return build_replay_pack(w, id="fixture/strands_field_k4", license=LICENSE, citation=CITATION, K=4,
                                 susc_path_K=4, out_path=str(out_path), **stated(of=w))


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--out", type=Path, default=DATA, help="the directory to write the packs to (default: tests/data)")
    a = ap.parse_args(argv)
    sys.path.insert(0, str(ROOT))
    a.out.mkdir(parents=True, exist_ok=True)
    for name, build in ((CYLINDER, cylinder), (STRANDS, strands)):
        path = a.out / name
        build(path)
        print(f"{path}: {os.path.getsize(path) / 1024:.0f} KiB")
    return 0


if __name__ == "__main__":
    sys.exit(main())
