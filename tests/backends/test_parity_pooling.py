"""The backend parity harness pools only independent walks: every rate case of ``parity_report`` walks at seeds of
its own (``case_seeds``), the spread it judges a backend against excludes the stream the backend walks, and the
pooled hit-count statistic (Stouffer over Student-t normal scores) is ``N(0, 1)`` on independent inputs."""
import numpy as np
import pytest
from scipy.stats import norm

from dmipy_sim.engine import backends
from dmipy_sim.engine.backends import _normal_scores, case_seeds, pooled_bias


def _rate_case_names():
    return [n for n, c in backends.parity_cases().items() if c.starts is None]


def test_case_seeds_are_distinct_and_deterministic():
    """For report seeds 0 .. 99 and 11, the 19 default rate cases' 9 streams each are 171 distinct integers in
    [0, 2^31), the same on a second call; one value is pinned."""
    names = _rate_case_names()
    assert len(names) == 19
    for seed in list(range(100)) + [11]:
        got = [s for n in names for s in case_seeds(seed, n, 9)]
        assert len(got) == 171 and len(set(got)) == 171, seed
        assert all(0 <= s < 2 ** 31 for s in got)
        assert got == [s for n in names for s in case_seeds(seed, n, 9)]
    assert case_seeds(11, "sphere", 1)[0] == 1503219651                     # golden: sha256("11/sphere/0")[:4]
    with pytest.raises(ValueError, match="at least one stream"):
        case_seeds(11, "sphere", 0)


def _label_cases():
    cases = backends.parity_cases()
    return {"label_slab": cases["label_slab"], "label_slab_perm": cases["label_slab_perm"]}


def test_a_seed_collision_is_refused_by_name_before_any_walk(monkeypatch):
    """Two rate cases handed the same seeds are refused, naming both, before a single walk runs."""
    from dmipy_sim.engine import core
    calls = []
    monkeypatch.setattr(backends, "case_seeds", lambda seed, name, streams: tuple(range(7, 7 + int(streams))))
    monkeypatch.setattr(core, "_walk_geometry", lambda *a, **k: calls.append(1))
    with pytest.raises(ValueError) as e:
        backends.parity_report(None, cases=_label_cases(), n_walkers=64, sweep=2)
    assert "'label_slab'" in str(e.value) and "'label_slab_perm'" in str(e.value)
    assert calls == []


def test_a_sweep_of_one_stream_is_refused():
    with pytest.raises(ValueError, match="sweep >= 2"):
        backends.parity_report(None, cases=_label_cases(), n_walkers=64, sweep=1)


def test_the_formerly_shared_pair_walks_apart():
    """``label_slab`` and ``label_slab_perm`` were one walk until a first crossing under the old seeding; now each has
    its own starts and streams, the sweep excludes the walk's stream, and the pooled record states the per-class Z."""
    report = backends.parity_report(None, cases=_label_cases(), n_walkers=64, sweep=2, seed=11)
    recs = {r["case"]: r for r in report}
    assert set(recs) == {"label_slab", "label_slab_perm", "pooled"}
    a, b = recs["label_slab"], recs["label_slab_perm"]
    assert a["seeds"]["walk"] != b["seeds"]["walk"]
    sa, sb = set(a["seeds"]["sweep"]), set(b["seeds"]["sweep"])
    assert len(sa) == len(sb) == 2 and not sa & sb
    walks = {a["seeds"]["walk"], b["seeds"]["walk"]}
    assert not (sa | sb) & walks                                            # the spread excludes stream 0
    assert a["starts_sha256"] != b["starts_sha256"]
    for r in (a, b):
        assert r["seeds"]["walk"] == case_seeds(11, r["case"], 3)[0]
        assert r["checks"]["n_hits"]["dof"] == 1 and r["checks"]["n_hits"]["sweep"]["streams"] == 2
        for key in backends.RARE_COUNTS:
            assert r["checks"][key]["reference"]["streams"] == 2
    per_class = recs["pooled"]["checks"]["n_hits_bias"]["per_class"]
    assert set(per_class) == {"LabelVolume"} and set(per_class["LabelVolume"]) == {"Z", "cases"}
    assert per_class["LabelVolume"]["cases"] == 2 and np.isfinite(per_class["LabelVolume"]["Z"])


def _two_sided_rate(t, dof):
    Z = _normal_scores(t, dof).sum(axis=1) / np.sqrt(t.shape[1])
    return float(np.mean(np.abs(Z) > norm.isf(5e-3)))


def test_the_pooled_statistic_is_calibrated_on_independent_inputs_and_detects_shared_walks():
    """20 000 replicates of 19 independent Student t on 7 dof: the two-sided 1 % rejection rate of the Stouffer Z lies
    within 4.3 binomial sd of 0.01. Six columns copied onto six others (two cases sharing a walk) inflate Var(Z) to
    31/19 and the rate to ~0.044: the same band rejects it."""
    rng = np.random.default_rng(0)
    t = rng.standard_t(7, size=(20_000, 19))
    assert 0.0070 <= _two_sided_rate(t, 7) <= 0.0130
    shared = t.copy()
    shared[:, 13:19] = shared[:, 0:6]
    rate = _two_sided_rate(shared, 7)
    assert not 0.0070 <= rate <= 0.0130 and rate > 0.03


def test_pooled_bias_is_the_stouffer_sum_and_refuses_a_shared_walk_seed():
    rng = np.random.default_rng(1)
    tv = rng.standard_t(7, size=19)
    hits = [(f"c{i}", float(tv[i]), 7, 0.001 * i, 100 + i, "Sphere" if i < 10 else "Mesh") for i in range(19)]
    got = pooled_bias(hits, 1e-2)
    z = _normal_scores(tv, 7)
    assert got["Z"] == pytest.approx(z.sum() / np.sqrt(19), rel=1e-12)
    assert got["per_class"]["Sphere"] == {"Z": pytest.approx(z[:10].sum() / np.sqrt(10), rel=1e-12), "cases": 10}
    assert got["per_class"]["Mesh"]["cases"] == 9 and "ok" not in got["per_class"]["Mesh"]
    dup = hits[:3] + [("twin", 0.3, 7, 0.0, hits[1][4], "Sphere")]
    with pytest.raises(ValueError) as e:
        pooled_bias(dup, 1e-2)
    assert "'c1'" in str(e.value) and "'twin'" in str(e.value)
