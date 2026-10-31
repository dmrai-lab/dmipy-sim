"""dmipy-sim: JAX Monte Carlo diffusion MRI replay engine.

A Monte Carlo walk's natural output is neither an acquisition-specific signal nor a trajectory history but an
acquisition-independent set of sufficient projections, accumulated while the physics is simulated. The engine
encodes as it goes: ``walk(spec, ...)`` returns the :class:`CoefficientWalk` (per walker the exact endpoints and
the sine-band projections of its path, its contact, its pool events, its field modes), ``build_replay_pack`` makes it
a ``.rpk`` pack, and ``ReplayPack.replay`` contracts a pack with any waveform. ``simulate`` is the walk and the
contraction in one call. Surface relaxivity and membrane permeability are baked into the walk. Shares one
pulse-sequence and substrate interface with ``dmipy-fit``.
"""

# Apply the GPU memory cap (DMIPY_GPU_MEM_GB) BEFORE any submodule imports JAX.
from .engine._gpu_config import apply_gpu_mem_cap as _apply_gpu_mem_cap, configure  # noqa: E402
_apply_gpu_mem_cap()

# Own the CPU thread count (DMIPY_SIM_CPU_THREADS) BEFORE any submodule imports JAX: see dmipy_sim._blas.
from ._blas import cap_cpu_threads  # noqa: E402
cap_cpu_threads()

from .engine.core import simulate, simulate_mixture, simulate_cpmg, walk_cost, certify_bounce_budget
from .spec.walk import walk
from .engine.mt_walk import simulate_mt_trajectories
from .coefficient_walk import CoefficientWalk
from .compartments import Compartments, Pool, pool_id
from .engine.pulse_sequence import (bare_gradient_echo, bare_spin_echo, fexi, fexi_b_detect,
                             saturation_pulse, prepend_mt_prep, emergent_z_spectrum)
from .engine.gpu import (gpu_available, check_gpu, free_gpu_memory, gpu_session,
                  list_gpu_processes)
from .viz.viz import (plot_waveform, plot_sequence_comparison,
                  plot_mesh_section, plot_walkers_3d, plot_cell_surface, plot_mesh_3d,
                  seed_in_cell, walk_paths, plot_trajectories, save_rotation)
from dmipy_replay.acquisition.scanner_sequence import ScannerSequence, Encoding, Protocol, Prescription
from dmipy_replay.sequences.builders import pgse, pgste, ogse, cpmg, splice, gre, ste, pte
from dmipy_replay.acquisition.waveforms import (
    set_b,
    calc_b,
    calc_btensor,
    btensor_invariants,
    rotate_waveform,
    tile_waveform,
)
from .geometry import (FreeDiffusion, Box1D, Sphere, Cylinder, MyelinatedCylinder,
                         Ellipsoid,
                         PackedCylinders, pack_cylinders,
                         PackedSpheres, pack_spheres,
                         PackedMyelinatedCylinders,
                         pack_myelinated_cylinders,
                         PermeableSlab1D)
from .geometry.curved_cylinder import CurvedCylinder, CurvedMyelinatedCylinder, PackedCurvedCylinders
from .geometry.sphere_union import SphereUnion
from .geometry.label_volume import LabelVolume
from .geometry.mesh import Mesh, load_ply
from .fields.susceptibility import SusceptibilitySources, MyelinSusceptibility, GridSusceptibility
from .fields.susceptibility_field import dipole_field, myelin_susceptibility_tensor, radial_from_sdf, sample_grid
from .geometry import mesh_shapes
from dmipy_replay.constants import GAMMA
from .acquisition.noise import add_rician_noise, add_rician_noise_batch, add_nc_chi_noise, estimate_sigma
from .replay.fod import FOD
from .replay.so3 import Distribution
from .engine import mt
from dmipy_replay.acquisition.rf import B1Pulse, RFEvent, RFSchedule, bloch_simulate, slice_profile
from dmipy_replay.acquisition.timing import SequenceTiming
from dmipy_replay.acquisition.scanners import ScannerLimits, SCANNERS
from .replay.replay import ReplayPack, read_rpk, write_rpk, replay_signal_jax
from dmipy_replay.replay.replay import compile_scheme, replay_signal, surface_logweight
from .replay import bank
from .replay.bank import build_replay_pack, build_to_floor, frame_from_bundles
from dmipy_replay.replay.field import frame_from_axis
from .phantom import Phantom, PartitionPhantom, Pose, Grid, PackSubstrate, FreeWater, Inert, Peaks, ODF, Watson, Frames, Fan

__all__ = [
    # the walk and its contraction: the producers return a CoefficientWalk
    "simulate", "simulate_mixture", "simulate_cpmg", "walk_cost", "walk",
    "certify_bounce_budget", "simulate_mt_trajectories", "CoefficientWalk",
    # replay pack assembler (producer side of the substrate bank)
    "build_replay_pack", "build_to_floor", "frame_from_axis", "frame_from_bundles",
    "Compartments", "Pool", "pool_id",
    "ScannerSequence", "Encoding", "Protocol",
    "bare_gradient_echo", "bare_spin_echo", "fexi", "fexi_b_detect", "saturation_pulse", "prepend_mt_prep",
    "emergent_z_spectrum",
    "gpu_available", "check_gpu", "free_gpu_memory", "gpu_session", "list_gpu_processes",
    "ScannerLimits", "SCANNERS",
    "pgse", "pgste", "ogse", "cpmg", "splice", "gre", "ste", "pte",
    "set_b", "calc_b", "calc_btensor", "btensor_invariants",
    "rotate_waveform", "tile_waveform",
    "FreeDiffusion", "Box1D", "Sphere", "Cylinder", "MyelinatedCylinder",
    "Ellipsoid",
    "PackedCylinders", "pack_cylinders",
    "PackedSpheres", "pack_spheres",
    "PackedMyelinatedCylinders", "pack_myelinated_cylinders",
    "PermeableSlab1D",
    "CurvedCylinder", "CurvedMyelinatedCylinder", "PackedCurvedCylinders", "SphereUnion", "LabelVolume",
    "Mesh", "load_ply",
    # susceptibility off-resonance fields (forward Bloch)
    "SusceptibilitySources", "MyelinSusceptibility", "GridSusceptibility",
    "dipole_field", "myelin_susceptibility_tensor", "radial_from_sdf", "sample_grid",
    "mesh_shapes",
    "GAMMA",
    "add_rician_noise", "add_rician_noise_batch", "add_nc_chi_noise", "estimate_sigma",
    # SH convolution
    "FOD",
    "Distribution",
    # magnetization transfer (physics + analytic oracle)
    "mt",
    # continuous RF pulses: complex B1(t) envelope + Bloch forward
    "Prescription", "Phantom", "PartitionPhantom", "Pose", "Grid", "PackSubstrate", "FreeWater", "Inert", "Peaks", "ODF", "Watson", "Frames", "Fan",
    "B1Pulse", "RFEvent", "RFSchedule", "SequenceTiming", "ReplayPack", "read_rpk", "write_rpk", "compile_scheme", "replay_signal", "replay_signal_jax", "bloch_simulate", "slice_profile",
]
