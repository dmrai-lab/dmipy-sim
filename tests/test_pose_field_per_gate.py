"""The closed-form pose expansion's field factor is one per gate (dmrai-lab/dmipy-sim#575).

The field phase ``a_w + u^T A_w u`` is the path channel integrated through the acquisition's own gate, so two
acquisitions whose field accrues through different gates -- a spin echo and a stimulated echo, or two spin echoes
of different echo time -- have different field factors. A batch of them is expanded in one pass per factor: every
member is the same numbers as its own call, and a batch whose members share a gate is still one pass.
"""
import numpy as np
import pytest

from dmipy_sim import sequences
from dmipy_sim.replay import so3
from dmipy_sim.spec.tissue import Tissue

KW = dict(scanner=7.0, tissue=Tissue(chi_iso=-1e-7, chi_aniso=-5e-8))


def _acquisitions(pk):
    """A spin echo of two directions (TE 0.8 ms), a stimulated echo (TE 0.8 ms, TM 0.56 ms) and a spin echo of
    TE 0.6 ms on the pack's walk: three gates."""
    n = 4 * pk.n_t + 1
    se = sequences.pgse([[1, 0, 0], [0.6, 0.8, 0.0]], 1e-4, 4e-4, bvalues=[5e8] * 2, TE=8e-4, n_t=n, slew_rate=np.inf)
    ste = sequences.pgste([[1, 0, 0]], 1.2e-4, 5.6e-4, bvalues=[5e8], TE=8e-4, n_t=n, slew_rate=np.inf,
                          ste_flip_angles=(90.0, 90.0, 90.0))
    se_short = sequences.pgse([[0, 0, 1]], 1e-4, 3e-4, bvalues=[5e8], TE=6e-4, n_t=n, slew_rate=np.inf)
    return [se, ste, se_short]


@pytest.mark.parametrize("method", ["closed", "quadrature"])
def test_a_batch_of_different_gates_is_each_member_alone(field_pack, method):
    """Field on, a spin echo, a stimulated echo and a shorter spin echo in one call: each member's coefficients and
    misfit are its own call's to the bit, and the expansion is the direct posed replay at every rotation."""
    pk = field_pack
    batch = _acquisitions(pk)
    got = pk.pose_responses(batch, method=method, **KW)
    for wf, pr in zip(batch, got):
        alone = pk.pose_response(wf, method=method, **KW)
        assert pr.route == alone.route == method
        np.testing.assert_array_equal(pr.coeffs, alone.coeffs)
        np.testing.assert_array_equal(pr.misfit, alone.misfit)
        if method == "closed":
            for R in so3.haar_rotations(4, 5):
                np.testing.assert_allclose(pr.at(R), pk.replay(wf, orientation=R, complex_signal=True, **KW), atol=2e-6)


def test_one_gate_is_one_pass_and_one_field_factor(field_pack, monkeypatch):
    """The encoding classes of one timing share a gate: a single acquisition, and a batch of acquisitions of one
    timing with different directions and b-values, are expanded in one pass with the field factor formed once.
    A batch of different gates takes one pass per gate."""
    import dmipy_sim.replay.replay as rr
    pk = field_pack
    se, ste, se_short = _acquisitions(pk)
    se_b = sequences.pgse([[0, 0, 1]], 1e-4, 4e-4, bvalues=[1e9], TE=8e-4, n_t=4 * pk.n_t + 1, slew_rate=np.inf)
    calls = {"pass": 0, "factor": 0}
    many, factor = rr.ReplayPack._pose_coeffs_closed_many, rr.ReplayPack._field_harmonics_of

    def counted_many(self, *a, **k):
        calls["pass"] += 1
        return many(self, *a, **k)

    def counted_factor(self, *a, **k):
        calls["factor"] += 1
        return factor(self, *a, **k)

    monkeypatch.setattr(rr.ReplayPack, "_pose_coeffs_closed_many", counted_many)
    monkeypatch.setattr(rr.ReplayPack, "_field_harmonics_of", counted_factor)
    for batch, n_pass in (([se], 1), ([se, se_b], 1), ([se, ste, se_short, se_b], 1 + 3)):
        pk.__dict__.pop("_field_factor_cache", None)
        calls.update(dict.fromkeys(calls, 0))
        pk.pose_responses(batch, method="closed", **KW)
        assert calls["pass"] == n_pass                    # the partition's call, then one per gate
        assert calls["factor"] == (1 if n_pass == 1 else n_pass - 1)
