"""An MCMRSimulator.jl float32 export as a C0-only replay pack.

The export is one flat float32 file of positions in METRES, C order ``(n_walkers, n_saves, 3)`` (Julia's column-major
``Array{Float32}(3, n_saves, n_walkers)`` written with ``write``), on a uniform save grid from t = 0 that includes the
first and last save. The file is read here, into an array; the library reads no file.

    python examples/foreign/mcmrsimulator_f32.py FILE.f32 SPEC.json OUT.rpk --n-saves 2715 --pool intra --D 1.7e-9 \
        --T 0.1 --K 64 --version 1.1 [--seed 3000]

The positions must be unwrapped. The pack holds the position channel and the static pool label only.
"""
import argparse

import numpy as np
from dmipy_replay.spec.substrate import load_spec

from dmipy_sim.foreign import walk_from_positions
from dmipy_sim.replay.bank import build_replay_pack

ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
ap.add_argument("file"); ap.add_argument("spec"); ap.add_argument("out")
ap.add_argument("--n-saves", type=int, required=True); ap.add_argument("--pool", required=True)
ap.add_argument("--D", type=float, required=True, help="m^2/s")
ap.add_argument("--T", type=float, required=True, help="duration of the walk (s): the last save")
ap.add_argument("--K", type=int, required=True); ap.add_argument("--version", required=True)
ap.add_argument("--seed", type=int, default=None)
ap.add_argument("--license", default="CC-BY-4.0"); ap.add_argument("--citation", default="")
a = ap.parse_args()

positions = np.fromfile(a.file, dtype=np.float32).reshape(-1, a.n_saves, 3)
walk = walk_from_positions(positions, dt_save=a.T / (a.n_saves - 1), K=a.K, pool=a.pool, spec=load_spec(a.spec), D=a.D,
                           producer=dict(engine="MCMRSimulator.jl", version=a.version, export_file=a.file.rsplit("/", 1)[-1]),
                           seed=a.seed)
pack = build_replay_pack(walk, id=f"foreign/mcmrsimulator/{a.file.rsplit('/', 1)[-1].rsplit('.', 1)[0]}", license=a.license,
                         citation=a.citation, out_path=a.out)
print(f"{a.out}: {positions.shape[0]} walkers, {a.n_saves} saves, K = {pack.K}, err_max = {pack.fidelity['err_max']:.3g}")
