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

# Own the CPU thread count (DMIPY_REPLAY_CPU_THREADS) BEFORE any submodule imports JAX: see dmipy_replay._blas.
from dmipy_replay._blas import cap_cpu_threads as _cap_cpu_threads  # noqa: E402
_cap_cpu_threads()

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
from .acquisition.noise import add_rician_noise, add_rician_noise_batch, add_nc_chi_noise, estimate_sigma
from .engine import mt
from .replay.container import write_rpk
from .replay import bank
from .replay.bank import build_replay_pack, build_to_floor, frame_from_bundles

__all__ = [
    # the walk and its contraction: the producers return a CoefficientWalk
    "simulate", "simulate_mixture", "simulate_cpmg", "walk_cost", "walk",
    "certify_bounce_budget", "simulate_mt_trajectories", "CoefficientWalk",
    # replay pack assembler (producer side of the substrate bank)
    "build_replay_pack", "build_to_floor", "frame_from_bundles",
    "Compartments", "Pool", "pool_id",
    "bare_gradient_echo", "bare_spin_echo", "fexi", "fexi_b_detect", "saturation_pulse", "prepend_mt_prep",
    "emergent_z_spectrum",
    "gpu_available", "check_gpu", "free_gpu_memory", "gpu_session", "list_gpu_processes",
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
    "add_rician_noise", "add_rician_noise_batch", "add_nc_chi_noise", "estimate_sigma",
    # magnetization transfer (physics + analytic oracle)
    "mt",
    # the pack's container on the producer's side
    "write_rpk",
]


def __getattr__(name):
    from ._moved import refuse
    raise refuse(__name__, name)
