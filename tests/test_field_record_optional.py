"""A pack's field channel names the basis it stores; the field record beside it (``susceptibility_grid``) is not
needed to read it. A field read off the strands in the walk (a fill's shards) carries the channel and no record."""
import copy

import numpy as np

from dmipy_sim import sequences
from dmipy_sim.replay import ReplayPack
from dmipy_sim.spec.tissue import Tissue

TISSUE = Tissue(chi_iso=-1e-7, chi_aniso=-5e-8)


def _without_record(pk):
    meta = copy.deepcopy(pk.meta)
    meta["compression"]["channels"].pop("susceptibility_grid")
    return ReplayPack(pk.arrays, meta)


def test_a_field_channel_without_its_record_replays_as_with_it(field_pack):
    """The same signal at 3 T, the anisotropic term included: the basis's channel names say it is stored."""
    pk = field_pack
    assert pk.meta["compression"]["channels"]["susceptibility_grid"]["has_aniso"]
    assert "aniso_G_xx" in pk.meta["compression"]["channels"]["susc_field_basis"]["channels"]
    seq = sequences.pgse([[1, 0, 0], [0, 0, 1]], 1e-4, 4e-4, bvalues=[5e8, 5e8], TE=8e-4, n_t=4 * pk.n_t + 1, slew_rate=np.inf)
    bare = _without_record(pk)
    np.testing.assert_array_equal(bare.replay(seq, scanner=3.0, tissue=TISSUE, tol=None), pk.replay(seq, scanner=3.0, tissue=TISSUE, tol=None))
    iso = Tissue(chi_iso=-1e-7, chi_aniso=0.0)
    assert not np.array_equal(bare.replay(seq, scanner=3.0, tissue=TISSUE, tol=None), bare.replay(seq, scanner=3.0, tissue=iso, tol=None))
