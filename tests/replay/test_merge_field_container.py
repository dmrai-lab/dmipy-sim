"""The field container's term of a merged pack is each block's own term, the largest over the blocks; a pack states its
worst block's. A block's coefficients are decoded with that block's own steps, so neither the first shard's term nor the
term of the elementwise widest step over blocks (which mixes the widest mode of different blocks) is a charge."""
import numpy as np
import pytest

from dmipy_replay.replay.field import CHANNEL_NAMES
from dmipy_sim.replay import bank
from dmipy_sim.replay import compression as _cx
from dmipy_sim.replay.bank import merge_packs
from tests.test_field_read_interval import _float_pack

KW = dict(B0=7.0, chi_iso=1e-7, chi_aniso=1e-7)


def _meta(K=16):
    return dict(K=K, channels=list(CHANNEL_NAMES), max_refocus_pulses=K // 2)


def _table(K=16, hot=()):
    """One (n_ch, K) block of steps: a flat floor, with the modes in ``hot`` ten times wider."""
    t = np.full((len(CHANNEL_NAMES), K), 1e-4)
    t[:, list(hot)] *= 10.0
    return t


def _term(table):
    return bank.field_quantisation({"susc_path_scale": table}, _meta(), _cx.default_battery(), T=0.1, h=5e-4, **KW)


def test_a_merged_table_is_the_worst_blocks_own_term_not_the_widest_step_of_each_mode():
    a, b = _table(hot=(1,)), _table(hot=(9,))              # each block is wide in a different mode
    own = [_term(a), _term(b)]
    merged = _term(np.stack([a, b]))
    assert merged.eps == max(o.eps for o in own) and merged.block == int(np.argmax([o.eps for o in own]))
    assert merged.eps == own[0].eps or merged.eps == own[1].eps
    mixed = _term(np.maximum(a, b))                         # the elementwise widest step: describes no stored block
    assert mixed.eps > merged.eps * 1.001
    assert _term(np.stack([b, a])).eps == merged.eps and _term(np.stack([b, a])).block == 1 - merged.block


def test_a_single_block_table_is_block_zero():
    t = _term(_table(hot=(3,)))
    assert t.block == 0 and t.eps == _term(_table(hot=(3,))[None]).eps


def test_the_float_container_has_no_term():
    assert bank.field_quantisation({}, _meta(), _cx.default_battery(), T=0.1, h=5e-4, **KW) == (0.0, 0)


@pytest.fixture(scope="module")
def shards(read_walks, read_coarse):
    """Two walks of one spec, each packed in the derived field container: each shard's blocks have their own steps."""
    return [_float_pack(w, id=f"t/q-{i}", license="x", citation="x", K=4, susc_path_K=4, device="numpy", field_container="derived")
            for i, w in enumerate((read_walks[1], read_coarse))]


@pytest.mark.parametrize("order", [(0, 1), (1, 0)])
def test_a_merged_pack_states_the_largest_shard_term_in_any_order(shards, order):
    own = [s.meta["fidelity"]["err_quantisation_field"] for s in shards]
    assert own[0] != own[1]
    m = merge_packs([shards[i] for i in order], id="t/merged", overlap="recertify")
    f = m.meta["fidelity"]
    assert f["err_quantisation_field"] == pytest.approx(max(own), rel=1e-5)
    rest = [s.meta["fidelity"]["err_max"] - s.meta["fidelity"]["err_quantisation_field"] for s in shards]
    assert f["err_max"] == pytest.approx(max(rest) + max(own), rel=1e-5)
    assert f["err_max"] >= f["terms"]["truncation"] + f["err_quantisation_field"] - 1e-15
