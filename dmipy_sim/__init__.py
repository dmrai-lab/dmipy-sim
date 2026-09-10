"""dmipy-sim: JAX Monte Carlo diffusion MRI simulator.

Forward Monte-Carlo of spins diffusing under arbitrary free gradient waveforms
``G(t)``, with surface relaxivity and membrane permeability baked into the walk.
Shares one pulse-sequence and substrate interface with ``dmipy-fit``.
"""

# Apply the GPU memory cap (DMIPY_GPU_MEM_GB) BEFORE any submodule imports JAX.
from .engine._gpu_config import apply_gpu_mem_cap as _apply_gpu_mem_cap, configure  # noqa: E402
_apply_gpu_mem_cap()

from .engine.core import simulate, simulate_mixture, simulate_cpmg, simulate_trajectories
# NB: the scalar trajectory-replay entrypoint is `dmipy_sim.trajectories.replay`, NOT a bare
# top-level `replay` — the name `dmipy_sim.replay.replay` is the .rpk pack-forward module (see below).
from .replay.trajectories import (unwrap_periodic, replay_jax,
                           replay_bloch, replay_bloch_jax,
                           finite_180_longitudinal_dwell, pre_pulse_gradient_phase,
                           pathway_sign_se)
from .engine.mt_walk import simulate_mt_trajectories
from .persistent_walk import PersistentWalk
from .compartments import Compartments, Pool, pool_id
from .engine.bloch import simulate_bloch
from .engine.pulse_sequence import (bare_gradient_echo, bare_spin_echo, fexi, fexi_b_detect,
                             saturation_pulse, prepend_mt_prep, emergent_z_spectrum)
from .engine.gpu import (gpu_available, check_gpu, free_gpu_memory, gpu_session,
                  list_gpu_processes)
from .viz.viz import (plot_waveform, plot_sequence_comparison,
                  plot_mesh_section, plot_walkers_3d, plot_cell_surface, plot_mesh_3d,
                  seed_in_cell, walk_paths, plot_trajectories, save_rotation)
from .acquisition.scanner_sequence import ScannerSequence, Encoding, Protocol
from .sequences.builders import pgse, pgste, ogse, cpmg, gre, ste, pte
from .acquisition.waveforms import (set_b, calc_b, calc_btensor, btensor_invariants, rotate_waveform, tile_waveform)
from .geometry import (FreeDiffusion, Box1D, Sphere, Cylinder, MyelinatedCylinder,
                         Ellipsoid,
                         PackedCylinders, pack_cylinders,
                         PackedSpheres, pack_spheres,
                         PackedMyelinatedCylinders,
                         pack_myelinated_cylinders,
                         PermeableSlab1D)
from .geometry.curved_cylinder import CurvedCylinder, CurvedMyelinatedCylinder, PackedCurvedCylinders
from .geometry.sphere_union import SphereUnion
from .geometry.mesh import Mesh, load_ply
from .fields.susceptibility import (SusceptibilitySources, MyelinSusceptibility,
                             GridSusceptibility, dipole_field,
                             myelin_susceptibility_tensor, radial_from_sdf, sample_grid)
from .geometry import mesh_shapes
from .constants import GAMMA
from .acquisition.noise import add_rician_noise, add_rician_noise_batch, add_nc_chi_noise, estimate_sigma
from .replay.fod import FOD
from .replay.so3 import Distribution
from .engine import mt
from .acquisition.rf import B1Pulse, RFEvent, RFSchedule, bloch_simulate, slice_profile
from .acquisition.timing import SequenceTiming
from .acquisition.scanners import ScannerLimits, SCANNERS
from .replay.replay import (ReplayPack, read_rpk, write_rpk, compile_scheme, replay_signal,
                     replay_signal_jax, surface_logweight)
from .replay import bank
from .replay.bank import build_replay_pack, build_to_floor, frame_from_axis, frame_from_bundles

__all__ = [
    "simulate", "simulate_mixture", "simulate_cpmg", "simulate_trajectories",
    # replay path: walk-once producer + scalar replay operators
    "unwrap_periodic", "replay_jax",
    # replay pack assembler (producer side of the substrate bank)
    "build_replay_pack", "build_to_floor", "frame_from_axis", "frame_from_bundles",
    # replay path: vector-Bloch + susceptibility + MT + refocusing helpers
    "replay_bloch", "replay_bloch_jax",
    "finite_180_longitudinal_dwell", "pre_pulse_gradient_phase", "pathway_sign_se",
    "simulate_mt_trajectories", "PersistentWalk", "Compartments", "Pool", "pool_id",
    "simulate_bloch",
    "ScannerSequence", "Encoding", "Protocol",
    "bare_gradient_echo", "bare_spin_echo", "fexi", "fexi_b_detect", "saturation_pulse", "prepend_mt_prep",
    "emergent_z_spectrum",
    "gpu_available", "check_gpu", "free_gpu_memory", "gpu_session", "list_gpu_processes",
    "ScannerLimits", "SCANNERS",
    "pgse", "pgste", "ogse", "cpmg", "gre", "ste", "pte",
    "set_b", "calc_b", "calc_btensor", "btensor_invariants",
    "rotate_waveform", "tile_waveform",
    "FreeDiffusion", "Box1D", "Sphere", "Cylinder", "MyelinatedCylinder",
    "Ellipsoid",
    "PackedCylinders", "pack_cylinders",
    "PackedSpheres", "pack_spheres",
    "PackedMyelinatedCylinders", "pack_myelinated_cylinders",
    "PermeableSlab1D",
    "CurvedCylinder", "CurvedMyelinatedCylinder", "PackedCurvedCylinders", "SphereUnion",
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
    "B1Pulse", "RFEvent", "RFSchedule", "SequenceTiming", "ReplayPack", "read_rpk", "write_rpk", "compile_scheme", "replay_signal", "replay_signal_jax", "bloch_simulate", "slice_profile",
]
