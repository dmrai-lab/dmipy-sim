"""Fast CPU native-shell validation; full registration: docs/permeable-shell-walk.md.

The registered low-G sd envelope is 0.5: N >= (0.5/0.002)^2 = 62500,
rounded to two native batches (65536); require measured signal SE <= 0.002.
The public simulate check resolves SE <= 0.012 with sd <= 1: N >= 6945,
rounded to 8192. Placement resolves correlation SE <= 0.003: N >= 111112,
rounded to 131072 per geometry. Seeds are fixed independent batch seeds.
The registered 396-comparison family budget and bias envelopes are unchanged.
"""
from __future__ import annotations
import json
import jax
import numpy as np
import pytest
from scipy.stats import kstest, pearsonr
from dmipy_sim import simulate
from dmipy_sim.geometry import PermeableShell
from dmipy_replay.sequences.builders import from_waveform
from diagnostics.permeable_shell_walk import (
    R, R_OUT, D, DELTA, BIG_DELTA, T, DTS, BATCH, Z_FAMILY, MODAL_SLOPE,
    references, walk, record, next_seed,
)

N_FAST = BATCH * int(np.ceil((0.5 / 0.002)**2 / BATCH))


@pytest.fixture(scope="module")
def study():
    return dict(signals=[], exchange=[], api=[], placement=[], runs=[],
                seed_iterator=iter((3526169395, 965206861, 2500257597,
                                    3834533282, 3024068679)))


def test_fast_native_shell_and_public_pgse(study, monkeypatch):
    """One low-G sphere point, including the public simulate acquisition route."""
    monkeypatch.setenv("DMIPY_SIM_RUN_DIR", "")
    refs = references("sphere", 1e-4, (0.8,))
    result = walk(study, "sphere", 1e-4, DTS[0], [refs[0][0]], N_FAST)
    assert np.max(result[1]) <= 0.002
    failures = record(study, "sphere", 1e-4, DTS[0], refs, result, N_FAST, "fast")
    # The last waveform node is the readout; append it after 500 held samples.
    gradients = np.zeros((1, int(round(T / DTS[0])) + 1, 3))
    nd, nD = int(round(DELTA / DTS[0])), int(round(BIG_DELTA / DTS[0]))
    gradients[0, :nd, 0] = refs[0][0]
    gradients[0, nD:nD + nd, 0] = -refs[0][0]
    sequence = from_waveform(gradients, DTS[0], [[1., 0., 0.]])
    master_seed = next_seed(study)
    n_api = 2 ** int(np.ceil(np.log2(1 / 0.012**2)))
    got = float(simulate(n_api, D, sequence, PermeableShell(R, R_OUT, 1e-4),
                         seed=master_seed, sub_steps=1, require_gpu=False)[0])
    ref = refs[0][1]
    tolerance = Z_FAMILY / np.sqrt(n_api) + ref.error_total + (MODAL_SLOPE + 1.8 / T) * DTS[0]
    study["api"].append(dict(seed=master_seed, n=n_api, walk=got,
                             reference=ref.total.real, tolerance=tolerance))
    assert abs(got - ref.total.real) <= tolerance
    assert not failures, json.dumps(failures, indent=2)


@pytest.mark.parametrize("kind, master_seed", (("sphere", 3834533282), ("cylinder", 3024068679)))
def test_initial_volume_distribution_and_direction_correlation(study, kind, master_seed):
    """Resolve correlations with SE <= 0.003 and check uniform volume starts."""
    n = 2 ** int(np.ceil(np.log2(1 / 0.003**2)))
    positions = np.asarray(PermeableShell(R, R_OUT, 1e-4, kind=kind).init_positions(
        n, jax.random.PRNGKey(master_seed)), dtype=np.float64)
    dimension = 3 if kind == "sphere" else 2
    radius = np.linalg.norm(positions[:, :dimension], axis=1)
    volume_fraction = (radius / R) ** dimension
    ks = kstest(volume_fraction, "uniform")
    direction = positions[:, :dimension] / radius[:, None]
    features = [direction[:, axis] for axis in range(dimension)]
    features += [direction[:, axis] ** 2 for axis in range(dimension)]
    if kind == "cylinder":
        features += [positions[:, 2] / R, (positions[:, 2] / R) ** 2]
    correlations = [pearsonr(volume_fraction, f) for f in features]
    row = dict(kind=kind, n=n, seed=master_seed, ks=float(ks.statistic), ks_p=float(ks.pvalue),
               correlations=[dict(r=float(c.statistic), p=float(c.pvalue)) for c in correlations])
    study["placement"].append(row)
    assert ks.pvalue > 0.005, row
    assert min(c.pvalue for c in correlations) > 0.01 / 12, row
