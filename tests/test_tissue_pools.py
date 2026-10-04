"""A pack's per-pool T2 / T1 is one ``{pool name: seconds}`` mapping over every pool of its embedded spec, and the
``.rph`` stores one form, a list by pool id (dmipy-sim#440).

Before this rule a partial mapping was zero-filled and the kernels read a zero time as "no decay", so
``T2={"intra": 0.05}`` on a two-pool pack silently switched relaxation off in the extra pool; a scalar was
replicated, a list by id was taken unchecked (an over-long one accepted), and the file held whichever spelling
the caller used. Now a missing pool, an unknown name, a scalar, a list or a zero is refused by name, ``inf`` is
no decay, ``replace`` merges pool by pool, and completeness is judged on the spec rather than on the walkers.
"""
import numpy as np
import pytest

import dmipy_sim as d
from dmipy_sim import sequences
from dmipy_sim.phantom import PackSubstrate
from dmipy_sim.replay.bank import build_replay_pack
from dmipy_sim.replay.phantom import _check_tissue_entry
from dmipy_sim.spec.substrate import SubstrateSpec
from dmipy_sim.spec.tissue import Tissue

INF = float("inf")


@pytest.fixture(scope="module")
def extra_only():
    """A packed cylinder walked in its extra pool alone: the spec names two pools, the walkers label one."""
    g = d.PackedCylinders([1e-6], [[0.0, 0.0]], 10e-6, pool="extra")
    walk = d.simulate_trajectories(120, 2e-9, g, 3e-3, 5e-4, seed=0, require_gpu=False)
    return build_replay_pack(walk, id="t/extra-only", license="x", citation="x", K=6)


def _b0(pack, TE=2e-3):
    return sequences.pgse([[0, 0, 1]], 0.4e-3, 1.0e-3, bvalues=[0.0], TE=TE, n_t=pack.n_t, slew_rate=np.inf)


def test_a_partial_mapping_is_refused_naming_the_missing_pool(pack):
    seq = _b0(pack)
    with pytest.raises(ValueError, match=r"no value for the pool\(s\) \['intra'\]"):
        pack.replay(seq, tissue=Tissue(T2={"extra": 0.05}))
    with pytest.raises(ValueError, match=r"\['extra'\]"):
        pack.walker_primitives(seq).rates(Tissue(T1={"intra": 1.0}))
    with pytest.raises(ValueError, match=r"\['intra'\]"):
        pack.replay_bloch(seq, tissue=Tissue(T2={"extra": 0.05}))
    with pytest.raises(ValueError, match="does not have"):
        pack.replay(seq, tissue=Tissue(T2={"extra": 0.05, "intra": 0.05, "csf": 2.0}))


def test_replace_merges_pool_by_pool():
    t = Tissue(T2={"extra": 0.05, "intra": 0.03}, T1={"extra": 1.0, "intra": 1.2}, rho2=1e-6)
    u = t.replace(T2={"intra": 0.08})
    assert u.T2 == {"extra": 0.05, "intra": 0.08} and u.T1 == t.T1 and u.rho2 == t.rho2
    assert Tissue().replace(T2={"intra": 0.08}).T2 == {"intra": 0.08}          # nothing to merge onto: as given
    assert t.replace(T2=None).T2 is None                                        # the tier switched off


def test_zero_is_refused_and_inf_decays_nothing(pack):
    with pytest.raises(ValueError, match="must be positive"):
        Tissue(T2={"extra": 0.0, "intra": 0.05})
    with pytest.raises(ValueError, match="must be positive"):
        Tissue(T2=0.0)
    seq = _b0(pack)
    bare = pack.replay(seq)
    np.testing.assert_allclose(pack.replay(seq, tissue=Tissue(T2={"extra": INF, "intra": INF}, T1={"extra": INF, "intra": INF})), bare, rtol=1e-12)
    assert pack.replay(seq, tissue=Tissue(T2={"extra": 0.05, "intra": 0.05}))[0] < bare[0]


def test_a_scalar_or_a_list_on_a_pack_is_refused(pack):
    seq = _b0(pack)
    with pytest.raises(ValueError, match="one number is the closed form's"):
        pack.replay(seq, tissue=Tissue(T2=0.05))
    with pytest.raises(TypeError, match="list by pool id"):
        Tissue(T2=[0.05, 0.05])
    with pytest.raises(TypeError, match="list by pool id"):
        Tissue(T1=np.array([1.0, 1.2]))


def test_the_file_form_is_a_list_by_pool_id_strict_both_ways(pack):
    sub = PackSubstrate(pack, m0=0.7, name="wm", tissue=Tissue(T2={"intra": 0.03, "extra": INF}, T1={"extra": 1.0, "intra": 1.2}, rho2=1e-6))
    entry = sub.to_meta()["tissue"]
    assert entry == {"T2": [None, 0.03], "T1": [1.0, 1.2], "rho2": 1e-6}         # by id, null for no decay, whatever the dict's order
    back = PackSubstrate.from_meta({**sub.to_meta(), "kind": "pack"}, pack=pack)
    assert back.tissue.T2 == {"extra": INF, "intra": 0.03} and back.tissue.T1 == {"extra": 1.0, "intra": 1.2}
    spec = pack.substrate
    for bad in ({"T2": {"intra": 0.03, "extra": 0.05}}, {"T2": 0.05}, {"T2": [0.05]}, {"T2": [0.05, 0.05, 0.05]}):
        with pytest.raises(ValueError):
            Tissue.from_meta(bad, spec=spec)
    with pytest.raises(ValueError, match="list by pool id"):
        Tissue.from_meta({"T2": [0.05, 0.05]})                                  # no spec: a list cannot be read
    with pytest.raises(ValueError, match="list by pool id"):
        _check_tissue_entry({"kind": "pack", "id": "wm", "tissue": {"T2": {"intra": 0.03}}})
    with pytest.raises(ValueError, match="positive seconds or null"):
        _check_tissue_entry({"kind": "pack", "id": "wm", "tissue": {"T2": [0.0, 0.03]}})
    with pytest.raises(ValueError, match="2 pools"):
        _check_tissue_entry({"kind": "pack", "id": "wm", "tissue": {"T2": [0.03]}}, pack)
    _check_tissue_entry({"kind": "pack", "id": "wm", "tissue": {"T2": [None, 0.03]}}, pack)
    with pytest.raises(ValueError, match="one number"):
        Tissue(T2=0.05).to_meta(spec=spec)                                      # a closed form's spelling has no file form on a pack


def test_an_unknown_tissue_key_is_refused_naming_what_is_accepted():
    """A tissue entry naming a key outside ``Tissue.KNOBS`` is refused, naming both the key and the
    accepted ones; there is no migration or version-gated acceptance of any other spelling."""
    for key in ("rho", "rho_2", "rho_1", "foo"):
        with pytest.raises(ValueError, match=rf"declares \['{key}'\]"):
            Tissue.from_meta({key: 4e-6})
        with pytest.raises(ValueError, match=rf"declares \['{key}'\]"):
            Tissue.from_meta({key: 4e-6, "D": 2e-9})
    assert Tissue.from_meta({"rho2": 4e-6}).rho2 == 4e-6
    assert Tissue.from_meta({"rho1": 4e-6}).rho1 == 4e-6


def test_packsubstrate_from_meta_refuses_an_unknown_tissue_key(pack):
    """``PackSubstrate.from_meta`` / ``substrate_from_meta`` refuse an unknown tissue-entry key outright --
    for a scalar entry (resolved at once) and a per-pool one (resolved lazily on first ``.tissue`` read)
    alike."""
    for key in ("rho", "rho_2", "rho_1", "foo"):
        scalar = {"id": "wm", "kind": "pack", "m0": 1.0, "tissue": {key: 4e-6}}
        with pytest.raises(ValueError, match=rf"declares \['{key}'\]"):
            PackSubstrate.from_meta(scalar, pack=pack)

        per_pool = {"id": "wm2", "kind": "pack", "m0": 1.0, "tissue": {"T2": [None, 0.03], key: 4e-6}}
        with pytest.raises(ValueError, match=rf"declares \['{key}'\]"):
            PackSubstrate.from_meta(per_pool, pack=pack)


def test_completeness_is_judged_on_the_spec_not_the_walkers(extra_only):
    pk = extra_only
    assert [p.name for p in pk.substrate.pools] == ["extra", "intra"]
    seq = _b0(pk)
    with pytest.raises(ValueError, match=r"\['intra'\]"):
        pk.replay(seq, tissue=Tissue(T2={"extra": 0.05}))                       # no walker is intra, the spec still names it
    S = pk.replay(seq, tissue=Tissue(T2={"extra": 0.05, "intra": INF}))
    np.testing.assert_allclose(S, pk.replay(seq, tissue=Tissue(T2={"extra": 0.05, "intra": 0.001})), rtol=1e-12)   # an empty pool's value changes nothing
    assert S[0] == pytest.approx(np.exp(-2e-3 / 0.05), rel=1e-6)


def test_nominal_is_the_mapping_of_the_pools_that_declare_a_value(pack):
    assert pack.nominal.T2 is None                                              # a bare cylinder's spec declares none
    dd = pack.substrate.to_dict()
    dd["pools"][1]["T2"] = 0.05
    partial = Tissue.from_spec(SubstrateSpec.from_dict(dd))
    assert partial.T2 == {"intra": 0.05}
    with pytest.raises(ValueError, match=r"\['extra'\]"):
        pack.replay(_b0(pack), tissue=partial)                                  # incomplete, refused by name
    assert partial.replace(T2={"extra": INF}).T2 == {"intra": 0.05, "extra": INF}
