"""The names dmipy-sim does not define because dmipy-replay does (the pack's reader and everything a replay reads: the
acquisitions, the sequence builders, the scanner catalogue, the substrate spec, the tissue): each is refused by name
with its one spelling.

A package of dmipy-sim that names dmipy-replay's code (``dmipy_sim``, ``dmipy_sim.replay``, ``dmipy_sim.sequences``,
``dmipy_sim.math``, ``dmipy_sim.spec``) defines ``__getattr__`` through
:func:`refuse`, so ``dmipy_sim.ReplayPack`` raises an ``AttributeError`` that says ``from dmipy_replay import
ReplayPack``; a name that was never one is the ordinary ``AttributeError``. A module that moved whole
(``dmipy_sim.replay.so3``) is a ``ModuleNotFoundError``: there is no module stub."""

MOVED = {
    "dmipy_sim": {
        "ReplayPack": "dmipy_replay", "read_rpk": "dmipy_replay", "FOD": "dmipy_replay", "Distribution": "dmipy_replay",
        "replay_signal_jax": "dmipy_replay.replay.jax_routes", "cap_cpu_threads": "dmipy_replay",
        **{n: "dmipy_replay" for n in ("ScannerSequence", "Encoding", "Protocol", "Prescription", "B1Pulse", "RFEvent",
                                       "RFSchedule", "SequenceTiming", "ScannerLimits", "set_b", "calc_b", "calc_btensor",
                                       "btensor_invariants", "rotate_waveform", "tile_waveform", "pgse", "pgste", "ogse",
                                       "cpmg", "splice", "gre", "ste", "pte", "GAMMA", "compile_scheme")},
        "SCANNERS": "dmipy_replay.acquisition.scanners", "bloch_simulate": "dmipy_replay.acquisition.rf",
        "slice_profile": "dmipy_replay.acquisition.rf", "replay_signal": "dmipy_replay.replay.replay",
        "surface_logweight": "dmipy_replay.replay.replay", "frame_from_axis": "dmipy_replay.replay.field",
        **{n: "dmipy_replay.phantom" for n in ("Phantom", "PartitionPhantom", "Pose", "Grid", "PackSubstrate", "FreeWater",
                                               "Inert", "Peaks", "ODF", "Watson", "Frames", "Fan")},
    },
    "dmipy_sim.sequences": {
        **{n: "dmipy_replay.sequences" for n in ("pgse", "pgste", "gre", "cpmg", "splice", "ogse", "ste", "pte", "from_waveform",
                                                 "from_btensor_waveform", "from_pgste_waveform", "instantaneous",
                                                 "to_gradient_array")},
        "from_pulseq": "dmipy_replay.sequences.pulseq",
    },
    "dmipy_sim.math": {n: "dmipy_replay.math.gradient_conversions" for n in ("q_from_b", "b_from_q", "q_from_g", "g_from_q",
                                                                            "b_from_g", "g_from_b")},
    "dmipy_sim.spec": {
        **{n: "dmipy_replay.spec.substrate" for n in ("SubstrateSpec", "Domain", "Frame", "Pool", "Susceptibility", "Surface",
                                                      "Wall", "Directional", "Sided", "Seeding", "Validity", "SpecError",
                                                      "validate", "load_spec", "susceptibility_field_of", "SCHEMA_PATH",
                                                      "SPEC_VERSION")},
        "Tissue": "dmipy_replay.spec.tissue",
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
