"""The field read of shards that read it at different intervals (dmipy-sim#755): a pool's read snaps to a whole number
of its own rounds per block, so the shards of one block carry different ``walk_params.field_read_interval``. The merge
treats the interval as the producer's pool join does: not an identity key, the merged pack states the COARSEST and the
term charged at it, and each shard's own interval stays in the provenance record."""
import numpy as np
import pytest

from dmipy_sim.replay.bank import merge_packs
from dmipy_sim.spec import walk_spec
from tests.test_field_read_interval import BLOCK, _float_pack, _walk, spec  # noqa: F401  (the fixture)


@pytest.fixture(scope="module")
def shards(spec, tmp_path_factory):
    """A walk that reads the field at the derived interval (the step) and one that reads it once per block, as packs
    carrying no per-voxel certificate."""
    tmp = tmp_path_factory.mktemp("readmerge")
    fine, coarse = _walk(spec, None, seed=3), _walk(spec, BLOCK, seed=4)
    assert fine.field_read_interval < coarse.field_read_interval == pytest.approx(BLOCK)
    return [_float_pack(w, id=f"t/read-{name}", license="x", citation="x", K=4, susc_path_K=4, device="numpy",
                        out_path=str(tmp / f"{name}.rpk")) for name, w in (("fine", fine), ("coarse", coarse))]


@pytest.mark.parametrize("order", [(0, 1), (1, 0)])
def test_the_merged_pack_states_the_coarsest_read_and_its_term(shards, order):
    fine, coarse = shards
    m = merge_packs([shards[i] for i in order], id="t/merged", overlap="recertify")
    cr = coarse.meta
    assert m.meta["walk_params"]["field_read_interval"] == cr["walk_params"]["field_read_interval"] == pytest.approx(BLOCK)
    assert m.meta["replay_envelope"]["field_read"] == cr["replay_envelope"]["field_read"]
    pm, pc = (x.meta["compression"]["channels"]["susc_field_basis"] for x in (m, coarse))
    assert pm["read"] == pc["read"] and pm["read_interval"] == pc["read_interval"]
    terms = m.meta["fidelity"]["terms"]
    assert terms["field_read"] == cr["fidelity"]["terms"]["field_read"] > fine.meta["fidelity"]["terms"]["field_read"]
    t = terms
    assert m.meta["fidelity"]["err_max"] >= t["truncation"] + sum(t["quadrature"].values()) + t["field_read"] + t["rounding"] - 1e-15
    own = [s["field_read_interval"] for s in m.meta["provenance"]["shards"]]       # each shard's own read stays on record
    assert own == [shards[i].meta["walk_params"]["field_read_interval"] for i in order]
    assert m.n_walkers == fine.n_walkers + coarse.n_walkers


def test_the_shards_are_not_changed_by_the_merge(shards):
    fine, coarse = shards
    before = (fine.meta["fidelity"]["terms"]["field_read"], fine.meta["walk_params"]["field_read_interval"],
              dict(fine.meta["replay_envelope"]["field_read"]))
    merge_packs([fine, coarse], id="t/merged")
    assert (fine.meta["fidelity"]["terms"]["field_read"], fine.meta["walk_params"]["field_read_interval"],
            dict(fine.meta["replay_envelope"]["field_read"])) == before


def test_a_different_block_is_still_refused(spec, shards):
    """Only the read is a pool's own: shards walked on another block are not one walk."""
    other = walk_spec(spec, 60, 8e-4, 2 * BLOCK, seed=5, n_probe=20_000, require_gpu=False, adaptive_steps=True, K=4,
                      field_cutoff_max_m=25e-6, field_read_interval=2 * BLOCK)
    pk = _float_pack(other, id="t/other", license="x", citation="x", K=4, susc_path_K=4, device="numpy")
    with pytest.raises(ValueError, match="the shards differ in walk_params"):
        merge_packs([shards[1], pk], id="t/refused")


def test_shards_that_do_not_all_read_the_field_are_refused(shards):
    from dmipy_sim.replay.bank import charge_field_read, field_read_record
    rec = field_read_record(shards[0].meta)
    assert rec is not None and field_read_record({"walk_params": {}}) is None
    with pytest.raises(ValueError, match="some of the shards read the field"):
        charge_field_read(dict(shards[0].meta), [rec, None])
