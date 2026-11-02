"""The names dmipy-sim does not define because dmipy-replay does: each is refused by name with its one spelling.

A package of dmipy-sim whose names moved (``dmipy_sim``, ``dmipy_sim.replay``) defines ``__getattr__`` through
:func:`refuse`, so ``dmipy_sim.ReplayPack`` raises an ``AttributeError`` that says ``from dmipy_replay import
ReplayPack``; a name that was never one is the ordinary ``AttributeError``. A module that moved whole
(``dmipy_sim.replay.so3``) is a ``ModuleNotFoundError``: there is no module stub."""

MOVED = {
    "dmipy_sim": {
        "ReplayPack": "dmipy_replay", "read_rpk": "dmipy_replay", "FOD": "dmipy_replay", "Distribution": "dmipy_replay",
        "replay_signal_jax": "dmipy_replay.replay.jax_routes", "cap_cpu_threads": "dmipy_replay",
        **{n: "dmipy_replay.phantom" for n in ("Phantom", "PartitionPhantom", "Pose", "Grid", "PackSubstrate", "FreeWater",
                                               "Inert", "Peaks", "ODF", "Watson", "Frames", "Fan")},
    },
    "dmipy_sim.replay": {
        "ReplayPack": "dmipy_replay", "read_rpk": "dmipy_replay", "FOD": "dmipy_replay",
        "PoseResponse": "dmipy_replay", "analytic_pose_response": "dmipy_replay.replay.pose",
        "replay_signal_jax": "dmipy_replay.replay.jax_routes", "replay_batch_jax": "dmipy_replay.replay.jax_routes",
        "write_rpk": "dmipy_sim.replay.container",
    },
}
"""``{package: {name: the module that defines it}}``."""


def refuse(package, name):
    """The ``AttributeError`` of ``package.name``: naming the import of a name that moved, plain otherwise."""
    where = MOVED.get(package, {}).get(name)
    if where is None:
        return AttributeError(f"module {package!r} has no attribute {name!r}")
    return AttributeError(f"{package}.{name} is not dmipy-sim's: it is {where}.{name} (from {where} import {name})")
