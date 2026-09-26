"""A phantom's substrates relax consistently under a readout, or the replay refuses (dmipy-sim#238): a pack pool
with no T2 anywhere would replay as if T2 were infinite, and beside a substrate that relaxes that is a tissue
contrast that looks right and is not. A pack substrate's T2 is a mapping over every pool of its spec, never one
number; nothing on the call resolves a value, every substrate declares its own tissue."""
import numpy as np
import pytest

from dmipy_sim.phantom import FreeWater, Grid, Inert, PackSubstrate, Phantom, Watson
from dmipy_sim.replay import read_rpk
from dmipy_sim.spec.tissue import Tissue

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
    ph = _phantom(PackSubstrate(pack_path, m0=0.7, name="wm", tissue=Tissue(T2={"extra": 0.06, "intra": 0.06})), FreeWater(m0=1.0, name="csf", tissue=Tissue(D=3e-9)))
    with pytest.raises(ValueError, match="'wm' relaxes while 'csf' would not"):
        ph.replay(_seq(pack_path))


def test_a_pack_with_no_t2_beside_a_relaxing_one_is_refused_by_name(pack_path):
    """The brain example's case: a pack substrate that declares no tissue beside a white matter that relaxes; the
    same pack with its T2 declared replays. The pack's own nominal values are never read for it: a substrate
    says what it replays at."""
    wm = PackSubstrate(pack_path, m0=0.7, name="wm", tissue=Tissue(T2={"extra": 0.06, "intra": 0.06}))
    gm = PackSubstrate(pack_path, m0=0.8, name="gm")
    with pytest.raises(ValueError, match="'gm' would not"):
        _phantom(wm, gm).replay(_seq(pack_path))
    S = _phantom(wm, PackSubstrate(pack_path, m0=0.8, name="gm", tissue=Tissue(T2={"extra": 0.06, "intra": 0.06}))).replay(_seq(pack_path))
    assert np.isfinite(S[~np.isnan(S)]).all()


def test_no_substrate_relaxing_is_a_consistent_phantom(pack_path):
    """No T2 anywhere is a diffusion phantom and replays; a T2 declared on one substrate and not the other is refused."""
    seq = _seq(pack_path)
    bare = _phantom(PackSubstrate(pack_path, m0=0.7, name="wm"), FreeWater(m0=1.0, name="csf", tissue=Tissue(D=3e-9)))
    S0 = bare.replay(seq)
    assert np.isfinite(S0[~np.isnan(S0)]).all()
    mixed = _phantom(PackSubstrate(pack_path, m0=0.7, name="wm", tissue=Tissue(T2={"extra": 0.06, "intra": 0.06})), FreeWater(m0=1.0, name="csf", tissue=Tissue(D=3e-9)))
    with pytest.raises(ValueError, match="'wm' relaxes while 'csf' would not"):
        mixed.replay(seq)
    both = _phantom(PackSubstrate(pack_path, m0=0.7, name="wm", tissue=Tissue(T2={"extra": 0.06, "intra": 0.06})),
                    FreeWater(m0=1.0, name="csf", tissue=Tissue(D=3e-9, T2=2.0)))
    assert not np.allclose(np.nan_to_num(both.replay(seq)), np.nan_to_num(S0))


def test_a_scalar_t2_on_a_pack_substrate_is_refused(pack_path):
    """A pack has named pools, a closed form has one unnamed pool: one number is the closed form's spelling
    (dmipy-sim#440), so on a pack it is refused rather than replicated."""
    pk = read_rpk(pack_path)
    seq = _seq(pack_path)
    with pytest.raises(ValueError, match="one number is the closed form's"):
        pk.replay(seq, tissue=Tissue(T2=0.06))
    with pytest.raises(ValueError, match="one number is the closed form's"):
        PackSubstrate(pk, m0=0.7, name="wm", tissue=Tissue(T2=0.06))
    with pytest.raises(ValueError, match="one number"):
        _phantom(PackSubstrate(pack_path, m0=0.7, name="wm", tissue=Tissue(T2=0.06)), FreeWater(m0=1.0, tissue=Tissue(D=3e-9, T2=2.0))).replay(seq)
