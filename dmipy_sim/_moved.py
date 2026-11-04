"""The names dmipy-sim does not define because dmipy-replay does (the pack's reader and everything a replay reads: the
acquisitions, the sequence builders, the scanner catalogue, the substrate spec, the tissue) or dmipy-analytic does (the
forms: the physical constants, the matrix method, the two-pool MT forms, the field forms): each is refused by name
with its one spelling.

A package of dmipy-sim that names dmipy-replay's code (``dmipy_sim``, ``dmipy_sim.replay``, ``dmipy_sim.sequences``,
``dmipy_sim.math``, ``dmipy_sim.spec``) defines ``__getattr__`` through
:func:`refuse`, so ``dmipy_sim.ReplayPack`` raises an ``AttributeError`` that says ``from dmipy_replay import
ReplayPack``, and ``from dmipy_sim import ReplayPack`` raises an ``ImportError`` that says it too (CPython drops the
``AttributeError`` of a module's ``__getattr__`` inside a ``from`` import and reports only that the name cannot be
imported); a name that was never one is the ordinary ``AttributeError``. A module that moved whole
(``dmipy_sim.replay.so3``) is a ``ModuleNotFoundError``: there is no module stub."""
import dis
import sys

MOVED = {
    "dmipy_sim": {
        "ReplayPack": "dmipy_replay", "read_rpk": "dmipy_replay", "FOD": "dmipy_replay", "Distribution": "dmipy_replay",
        "replay_signal_jax": "dmipy_replay.replay.jax_routes", "cap_cpu_threads": "dmipy_replay",
        **{n: "dmipy_replay" for n in ("ScannerSequence", "Encoding", "Protocol", "Prescription", "B1Pulse", "RFEvent",
                                       "RFSchedule", "SequenceTiming", "ScannerLimits", "set_b",
                                       "btensor_invariants", "rotate_waveform", "tile_waveform", "pgse", "pgste", "fexi",
                                       "ogse", "cpmg", "splice", "gre", "ste", "pte", "compile_scheme")},
        "GAMMA": "dmipy_analytic.constants",
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
    "dmipy_sim.math": {**{n: "dmipy_replay.math.gradient_conversions" for n in ("q_from_b", "b_from_q", "q_from_g",
                                                                               "g_from_q", "b_from_g", "g_from_b")},
                       "matrix_method": "dmipy_analytic.restricted"},
    "dmipy_sim.engine.mt": {n: "dmipy_analytic.relaxation" for n in (
        "forward_rate", "backward_rate", "bound_fraction", "kappa_MT_from_forward_rate", "dwell_time_from_fraction",
        "two_pool_generator", "evolve_two_pool", "bloch_mcconnell_transverse", "bloch_mcconnell_longitudinal",
        "mt_z_spectrum")},
    "dmipy_sim.fields.hollow_cylinder": {"hollow_cylinder_basis": "dmipy_analytic.jax.fields",
                                         "segment_basis": "dmipy_analytic.jax.fields",
                                         "annulus_mean_log": "dmipy_analytic.fields"},
    "dmipy_sim.acquisition.coil": {"MU0": "dmipy_analytic.constants", "segment_field": "dmipy_analytic.fields"},
    "dmipy_sim.spec": {
        **{n: "dmipy_replay.spec.substrate" for n in ("SubstrateSpec", "Domain", "Frame", "Susceptibility", "Surface",
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

SPELLINGS = {
    "dmipy_sim": {"calc_b": "seq.b()", "calc_btensor": "seq.btensor()",
                  "fexi_b_detect": "dmipy_replay.fexi_bvalues(seq) (the filter's and the detection's b-values)"},
    "dmipy_sim.spec": {"Pool": "PoolSpec (from dmipy_replay.spec.substrate import PoolSpec)"},
}
"""``{package: {refused name: its one spelling}}``: a name no package defines, spelled another way (a sequence's b and
B-tensor are its methods)."""


def _from_import():
    """Whether the lookup that ``refuse`` answers is the one of a ``from package import name`` statement: the
    statement's frame is three frames up (``refuse`` is called by a package's ``__getattr__``, which the statement
    runs), standing at its ``IMPORT_FROM`` instruction."""
    try:
        frame = sys._getframe(3)
    except ValueError:
        return False
    return frame.f_code.co_code[frame.f_lasti] == dis.opmap["IMPORT_FROM"]


def refuse(package, name):
    """The exception of ``package.name``: the refusal that says the spelling of a name that moved or has another
    spelling (an ``ImportError`` in a ``from package import name`` statement, an ``AttributeError`` otherwise, so
    ``hasattr`` stays false), the ordinary ``AttributeError`` of any other name. Called as ``raise refuse(__name__,
    name)`` by a package's ``__getattr__``."""
    where = MOVED.get(package, {}).get(name)
    spelling = SPELLINGS.get(package, {}).get(name)
    if where is None and spelling is None:
        return AttributeError(f"module {package!r} has no attribute {name!r}")
    if spelling is not None:
        message = f"{package}.{name} does not exist: use {spelling}"
    else:
        message = f"{package}.{name} is not dmipy-sim's: it is {where}.{name} (from {where} import {name})"
    return ImportError(message, name=package) if _from_import() else AttributeError(message)
