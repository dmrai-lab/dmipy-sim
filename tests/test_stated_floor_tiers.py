"""A pack's precision tiers read the floor its certificate states: ``floor_at_full_n`` IS ``fidelity.floor_max``, and
the tiers' walker counts follow from that stated value, so two builds of the same walkers write the same bytes
whatever the last digits of the floor's reduction on their hardware."""
from __future__ import annotations

from dmipy_sim import Cylinder
from dmipy_sim.engine.core import _walk_geometry
from dmipy_sim.replay.bank import build_replay_pack, window_of_pack
from tests._envelope import TEST_BAND_HZ, battery_reach, walk_envelope


def _pack():
    walk = _walk_geometry(240, 2e-9, Cylinder(4e-6, (0, 0, 1)), T_max=0.02, block_h=2e-4, seed=613, require_gpu=False)
    return build_replay_pack(walk, id="t/tiers", license="x", citation="x", K=16, position_container=None, blt_container=None,
                             field_container=None, envelope=walk_envelope(G_max=battery_reach(walk.T_window), band_hz=TEST_BAND_HZ, of=walk))


def _stated(pack):
    return pack.meta["compression"]["precision_tiers"]["floor_at_full_n"], pack.meta["fidelity"]["floor_max"]


def test_the_tiers_read_the_stated_floor_of_a_built_pack():
    tier, stated = _stated(_pack())
    assert tier == stated and float(f"{stated:.5e}") == stated            # the stated floor, at its 6 digits


def test_the_tiers_read_the_stated_floor_of_a_window_of_a_pack():
    pk = _pack()
    tier, stated = _stated(window_of_pack(pk, 0.0, 0.01))
    assert tier == stated and float(f"{stated:.5e}") == stated
