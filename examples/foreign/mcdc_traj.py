"""An MC/DC trajectory file as a C0-only replay pack.

MC/DC writes, per configuration ``<prefix>``, ``<prefix>_0.bhdr`` (float32: [_, n_walkers, n_saves]) and
``<prefix>_0.traj`` (float32 positions in MILLIMETRES, C order ``(n_walkers, n_saves, 3)``), on a uniform save grid
from t = 0. The file is read here, into an array; the library reads no file.

    python examples/foreign/mcdc_traj.py PREFIX SPEC.json OUT.rpk --pool intra --D 1.7e-9 --T 0.1 --K 64 \
        --version 1.50 [--seed 4000]

``PREFIX``'s positions must be unwrapped. The pack holds the position channel and the static pool label only.
"""
import argparse

import numpy as np
from dmipy_replay.spec.substrate import load_spec

from dmipy_sim.foreign import walk_from_positions
from dmipy_sim.replay.bank import build_replay_pack

ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
ap.add_argument("prefix"); ap.add_argument("spec"); ap.add_argument("out")
ap.add_argument("--pool", required=True); ap.add_argument("--D", type=float, required=True, help="m^2/s")
ap.add_argument("--T", type=float, required=True, help="duration of the walk (s): the last save")
ap.add_argument("--K", type=int, required=True); ap.add_argument("--version", required=True)
ap.add_argument("--seed", type=int, default=None)
ap.add_argument("--license", default="CC-BY-4.0"); ap.add_argument("--citation", default="")
a = ap.parse_args()

hdr = np.fromfile(f"{a.prefix}_0.bhdr", dtype=np.float32)
n, n_saves = int(hdr[1]), int(hdr[2])
positions = np.fromfile(f"{a.prefix}_0.traj", dtype=np.float32).reshape(n, n_saves, 3) * np.float32(1e-3)   # mm -> m
walk = walk_from_positions(positions, sample_interval=a.T / (n_saves - 1), K=a.K, pool=a.pool, spec=load_spec(a.spec), D=a.D,
                           producer=dict(engine="MC/DC", version=a.version, trajectory_file=f"{a.prefix}_0.traj"), seed=a.seed)
pack = build_replay_pack(walk, id=f"foreign/mcdc/{a.prefix.rsplit('/', 1)[-1]}", license=a.license, citation=a.citation,
                         out_path=a.out)
print(f"{a.out}: {n} walkers, {n_saves} saves, K = {pack.K}, err_max = {pack.fidelity['err_max']:.3g}")
