"""A single ``.rpk`` pack on the Hub read by plan, the reader being :class:`dmipy_replay.replay.remote.RemotePack`:
:class:`RemotePack` is that reader with this package's :class:`~dmipy_sim.replay.replay.ReplayPack` as the packs its
views and its ``load`` build.

    pack = ReplayPack.open("hf://SubstrateCommons/canonical-pores/packs/cylinder/d05.00um.rpk")
    plan = pack.plan(seq)                        # bands, modes, error and bytes, from the header alone
    S = pack.replay(seq)                         # fetches the plan's tensors; S.plan is the plan, pack.bytes_fetched the cost
"""
from dmipy_replay.replay.remote import RemotePack as _RemotePack

from .replay import ReplayPack

__all__ = ["RemotePack", "open_remote"]


class RemotePack(_RemotePack):
    """:class:`dmipy_replay.replay.remote.RemotePack` building this package's :class:`~dmipy_sim.replay.replay.ReplayPack`."""
    pack_class = ReplayPack


def open_remote(uri, **kw):
    """:class:`RemotePack` for ``uri``, a ``.rpk`` on the Hub (``hf://owner/name/path.rpk``)."""
    return RemotePack(uri, **kw)
