"""A phantom's substrates relax consistently under a readout, or the replay refuses (dmipy-sim#238): a pack pool
with no T2 anywhere would replay as if T2 were infinite, and beside a substrate that relaxes that is a tissue
contrast that looks right and is not. A scalar T2 on a pack substrate is every pool's."""
import numpy as np
import pytest

from dmipy_sim.phantom import FreeWater, Grid, Inert, PackSubstrate, Phantom, Watson
from dmipy_sim.replay import read_rpk

from tests.test_rph_constructor import _acq, _annulus, _pack, _tangential, GRID8


@pytest.fixture(scope="module")
def pack_path(tmp_path_factory):
    return _pack(tmp_path_factory.mktemp("pk"))


def _phantom(wm, csf):
    f_wm, f_csf = _annulus()
    return Phantom.compose(GRID8, fractions={wm: f_wm, csf: f_csf}, orientation=Watson(mu=_tangential(), kappa=12.0, lmax=8),
                           remainder=Inert())


def _seq(pack_path):
    return _acq(read_rpk(pack_path), [[1, 0, 0], [0, 0, 1]], [0.0, 1e9])


def test_a_relaxing_pack_beside_free_water_with_no_t2_is_refused(pack_path):
    ph = _phantom(PackSubstrate(pack_path, m0=0.7, name="wm", T2_s=[0.06, 0.06, 0.06]), FreeWater(D_m2_s=3e-9, m0=1.0, name="csf"))
    with pytest.raises(ValueError, match="'wm' relaxes while 'csf' would not"):
        ph.replay(_seq(pack_path))


def test_a_pack_with_no_t2_beside_a_relaxing_one_is_refused_by_name_and_pool(pack_path):
    """The brain example's case: a pack whose pools declare no T2, and none given, beside a white matter that relaxes."""
    wm = PackSubstrate(pack_path, m0=0.7, name="wm", T2_s=[0.06, 0.06, 0.06])
    gm = PackSubstrate(pack_path, m0=0.8, name="gm")
    with pytest.raises(ValueError, match="'gm' would not .pools extra, intra declare no T2"):
        _phantom(wm, gm).replay(_seq(pack_path))
    # T2= on the call resolves every pack, so the same phantom replays
    S = _phantom(wm, gm).replay(_seq(pack_path), T2_s=[0.06, 0.06, 0.06])
    assert np.isfinite(S[~np.isnan(S)]).all()


def test_no_substrate_relaxing_is_a_consistent_phantom(pack_path):
    """No T2 anywhere is a diffusion phantom and replays; tissue=False resolves no nominal value but does not
    silence a T2 declared on a substrate, so a mixed phantom stays refused under it."""
    seq = _seq(pack_path)
    bare = _phantom(PackSubstrate(pack_path, m0=0.7, name="wm"), FreeWater(D_m2_s=3e-9, m0=1.0, name="csf"))
    S0 = bare.replay(seq)
    np.testing.assert_allclose(np.nan_to_num(bare.replay(seq, tissue=False)), np.nan_to_num(S0))
    mixed = _phantom(PackSubstrate(pack_path, m0=0.7, name="wm", T2_s=[0.06, 0.06, 0.06]), FreeWater(D_m2_s=3e-9, m0=1.0, name="csf"))
    with pytest.raises(ValueError, match="'wm' relaxes while 'csf' would not"):
        mixed.replay(seq, tissue=False)
    both = _phantom(PackSubstrate(pack_path, m0=0.7, name="wm", T2_s=[0.06, 0.06, 0.06]),
                    FreeWater(D_m2_s=3e-9, m0=1.0, name="csf", T2_s=2.0))
    assert not np.allclose(np.nan_to_num(both.replay(seq)), np.nan_to_num(S0))


def test_a_scalar_t2_is_every_pools_value(pack_path):
    pk = read_rpk(pack_path)
    seq = _seq(pack_path)
    np.testing.assert_allclose(pk.replay(seq, T2=0.06), pk.replay(seq, T2=[0.06, 0.06, 0.06]))
    a = _phantom(PackSubstrate(pack_path, m0=0.7, name="wm", T2_s=0.06), FreeWater(D_m2_s=3e-9, m0=1.0, T2_s=2.0)).replay(seq)
    b = _phantom(PackSubstrate(pack_path, m0=0.7, name="wm", T2_s=[0.06] * 3), FreeWater(D_m2_s=3e-9, m0=1.0, T2_s=2.0)).replay(seq)
    np.testing.assert_allclose(np.nan_to_num(a), np.nan_to_num(b))
