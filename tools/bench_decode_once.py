"""Time N ``walker_signals`` calls on one pack with its decoded cache (the default) and without (``decode_budget = 0``):
the same results bit for bit, the decodes counted. ``python tools/bench_decode_once.py PACK.rpk [--calls 44]``."""
import argparse
import collections
import time

import numpy as np

from dmipy_sim.replay.decoded import DecodedCache
from dmipy_sim.replay.replay import read_rpk
from dmipy_sim.sequences.builders import pgse
from dmipy_sim.spec import Tissue


def sequences(pk, n):
    """``n`` PGSE rows on the pack's grid, the gradient along x, ``b`` and ``Delta`` varied."""
    T = (pk.n_t - 1) * pk.dt
    rows = [(0.005, 0.020), (0.010, 0.040), (0.010, 0.060), (0.025, 0.055)]
    out = []
    for i in range(n):
        delta, Delta = rows[i % len(rows)]
        out.append(pgse([[1.0, 0.0, 0.0]], delta, Delta, bvalues=[(1.0 + i % 3) * 1e9], TE=T, n_t=pk.n_t, slew_rate=200.0))
    return out


def run(pk, seqs, tissue):
    t0 = time.perf_counter()
    out = [pk.walker_signals(s, tissue=tissue) for s in seqs]
    return time.perf_counter() - t0, out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("pack")
    ap.add_argument("--calls", type=int, default=44)
    a = ap.parse_args()
    pk = read_rpk(a.pack)
    seqs, tissue = sequences(pk, a.calls), Tissue(rho2=1e-6)
    misses = collections.Counter()
    get = DecodedCache.get

    def counting(self, kind, selection, tensors, decode):
        def counted():
            misses[kind] += 1
            return decode()
        return get(self, kind, selection, tensors, counted)
    DecodedCache.get = counting
    pk.decode_budget = 0
    t_bare, bare = run(pk, seqs, tissue)
    n_bare = dict(misses); misses.clear()
    pk.decode_budget = 1 << 30
    t_held, held = run(pk, seqs, tissue)
    same = all(np.array_equal(x, y) for r, s in zip(bare, held) for x, y in zip(r, s))
    print(f"{a.calls} calls on {pk.n_walkers} walkers, K = {pk.K}")
    print(f"decode per call : {t_bare:7.1f} s  {n_bare}")
    print(f"decoded once    : {t_held:7.1f} s  {dict(misses)}  holding {pk.decoded_bytes / 2**20:.0f} MiB")
    print(f"results bit-identical: {same}")


if __name__ == "__main__":
    main()
