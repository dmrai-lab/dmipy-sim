"""``dmipy_sim.replay``: the producer's side of a replay pack -- building it from a walk (:mod:`~dmipy_sim.replay.bank`),
its container (:mod:`~dmipy_sim.replay.container`), the operations that re-encode it (:mod:`~dmipy_sim.replay.reencode`,
:mod:`~dmipy_sim.replay.continuation`), the columnar and time-segment layouts' writers, publishing. Reading and
replaying a pack is dmipy-replay's (:class:`dmipy_replay.ReplayPack`); a name that moved there is refused by name."""


def __getattr__(name):
    from .._moved import refuse
    raise refuse(__name__, name)
