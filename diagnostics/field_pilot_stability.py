"""CPU raster pilot stability measurements for issue 935."""
import json
import numpy as np
from dmipy_sim.geometry.analytic import PermeableSlab1D
from dmipy_sim.fields.susceptibility_field import FieldGrid
from dmipy_sim.engine.mt_walk import simulate_mt_trajectories
from dmipy_sim.replay import bank

ENV = dict(B0_list=[7.0], theta_deg=[0, 90], chi_max=1.06e-6, max_refocus_pulses=None)

def capture(*, seed=43, pilot_walkers=1024, block_h=None):
    """The diagnosed slab's CPU pilot on a 400-node step raster."""
    mask = np.zeros((400, 1, 1), np.float32)
    mask[200:] = 1
    p = np.zeros((6, 400, 1, 1), np.float32)
    p[0] = mask - mask.mean()
    grid = FieldGrid(dict(iso_local=(mask-mask.mean())/3, iso_P=p,
                         shape=mask.shape, voxel_size=np.full(3, 1e-8)), np.zeros(3))
    class Captured(Exception):
        pass
    from dmipy_sim.engine import accumulate
    original_count = accumulate.PILOT_WALKERS
    accumulate.PILOT_WALKERS = pilot_walkers
    original = bank.derive_field_band_per_pool
    result = {}
    def intercept(series, pools, names, h, env, **kwargs):
        result.update(series=series, names=names, h=h, env=env, kwargs=kwargs)
        raise Captured
    bank.derive_field_band_per_pool = intercept
    try:
        simulate_mt_trajectories(20000, 2e-9, PermeableSlab1D(4e-6, 1e-5), .01,
                       seed=seed, K=32, block_h=block_h, sub_steps=4, tiers='all', field_basis=grid,
                       envelope=ENV, require_gpu=False)
    except Captured:
        pass
    finally:
        bank.derive_field_band_per_pool = original
        accumulate.PILOT_WALKERS = original_count
    if not result:
        raise RuntimeError("the field pilot did not reach band selection")
    return result

def measure(data, baseline):
    """Twenty seeded bootstrap pilots for smooth, diagnosed and demanding field cases."""
    from scipy.fft import dct, idct
    series = data['series']
    modes = dct(series, axis=1, norm='ortho')
    modes[:, 2:] = 0
    smooth = idct(modes, axis=1, norm='ortho')
    results = {}
    for case, values, chi, production in [('easy', smooth, 1.06e-6, 20000),
                                          ('raster43', series, 1.06e-6, 20000),
                                          ('hard', series, 4.24e-6, 2000000)]:
        rows = []
        kwargs = dict(data['kwargs'], chi=chi, floor_scale=np.sqrt(len(series)/production))
        for seed in range(93500, 93520):
            sample = values[np.random.default_rng(seed).integers(0, len(series), len(series))]
            old_k, old = baseline(sample, data['names'], data['h'], data['env'], **kwargs)
            k, new = bank.derive_field_band(sample, data['names'], data['h'], data['env'],
                                            evaluate_all=True, **kwargs)
            rows.append(dict(seed=seed, old_K=old_k, K=k,
                             old=[dict(K=r['K'], ratio=r['worst']['ratio']) for r in old['ladder']],
                             new=[dict(K=r['K'], ratio=r['worst_ratio'], upper_bound=r['upper_bound'])
                                  for r in new['ladder']]))
        results[case] = rows
    return results


if __name__ == '__main__':
    import argparse
    import subprocess
    import types
    from pathlib import Path
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--baseline-ref', required=True)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    # Only the historical selector's early stop changes, so every rung is measured.
    source = subprocess.check_output(['git', 'show', args.baseline_ref + ':dmipy_sim/replay/bank.py'], text=True)
    if 'chosen = k; break' not in source:
        raise ValueError('baseline-ref must identify the selector before issue 935')
    source = source.replace('chosen = k; break', 'chosen = k if chosen is None else chosen')
    historical = types.ModuleType('dmipy_sim.replay.baseline935')
    historical.__package__ = 'dmipy_sim.replay'
    exec(compile(source, 'baseline935', 'exec'), historical.__dict__)
    commit = subprocess.check_output(['git', 'rev-parse', args.baseline_ref], text=True).strip()
    data = capture()
    output = dict(baseline_commit=commit, pilot_seed=43, bootstrap_seeds=[93500, 93519],
                  pilot_walkers=1024, confidence_level=.99, h=data['h'], n_blocks=126,
                  sampling='independent bootstrap samples of one CPU walker bank; not independent physical pilots',
                  cases=measure(data, historical.derive_field_band))
    Path(args.output).write_text(json.dumps(output, indent=2) + '\n')
