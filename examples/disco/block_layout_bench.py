"""Read-locality benchmark for the block layout (dmipy-sim#505): opens a columnar layout and its block re-tiling
at the same directory-local paths and reads the z-slab (default z = 20) at K = 32 with :meth:`~dmipy_sim.replay.
columnar.ColumnarPack.view`, counting the column-file reads and their bytes (:attr:`Source.requests` /
:attr:`Source.bytes_read`, which every :class:`ColumnarPack` already keeps) and the wall time.

The present columnar layout sorts rows by voxel ijk lexicographically (x-major) and stores every column in its
own file, so a z-slab is hundreds of separate row ranges PER COLUMN and no tier can be fetched on its own; the
block layout groups a slab's voxels by 4x4x4 block and stores every tier's columns of one block in one
safetensors file, so the same slab is few, large, tier-scoped reads.

    python examples/disco/block_layout_bench.py SOURCE_DIR BLOCKS_DIR [--z 20] [--k 32]
"""
from __future__ import annotations
import argparse
import time

from dmipy_sim.replay.columnar import open_columnar


def bench(uri, z, k, label):
    pack = open_columnar(uri)
    voxels = sorted({tuple(int(x) for x in r["ijk"]) for r in pack.index["rows"] if int(r["ijk"][2]) == z})
    pack.src.requests = 0; pack.src.bytes_read = 0
    t0 = time.time()
    view = pack.view(K=k, voxels=voxels)
    dt = time.time() - t0
    print(f"{label:>10}: {len(voxels):5d} voxels, {view.n_walkers:8d} walkers, "
          f"{pack.src.requests:6d} reads, {pack.src.bytes_read / 1e6:9.3f} MB, {dt:7.3f} s")
    return pack.src.requests, pack.src.bytes_read, dt


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("source_dir", help="the columnar layout (a local directory)")
    ap.add_argument("blocks_dir", help="the block layout of the same pack (a local directory)")
    ap.add_argument("--z", type=int, default=20, help="the grid z-index of the slab")
    ap.add_argument("--k", type=int, default=32, help="the number of bands read")
    a = ap.parse_args(argv)
    reads_c, bytes_c, s_c = bench(a.source_dir, a.z, a.k, "columnar")
    reads_b, bytes_b, s_b = bench(a.blocks_dir, a.z, a.k, "blocks")
    print(f"reads  {reads_c / max(reads_b, 1):7.1f}x fewer")
    print(f"bytes  {bytes_c / max(bytes_b, 1):7.2f}x")
    print(f"time   {s_c / max(s_b, 1e-9):7.2f}x")


if __name__ == "__main__":
    main()
