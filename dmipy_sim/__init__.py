"""dmipy-sim: JAX Monte Carlo diffusion MRI simulator.

Forward Monte-Carlo of spins diffusing under arbitrary free gradient waveforms
``G(t)``, with surface relaxivity and membrane permeability baked into the walk.
Shares one pulse-sequence and substrate interface with ``dmipy-fit``.
"""

# Apply the GPU memory cap (DMIPY_GPU_MEM_GB) BEFORE any submodule imports JAX.
from ._gpu_config import apply_gpu_mem_cap as _apply_gpu_mem_cap, configure  # noqa: E402
_apply_gpu_mem_cap()

from .core import simulate, simulate_mixture, simulate_cpmg, simulate_trajectories
from .trajectories import (unwrap_periodic, replay, replay_jax,
                           replay_bloch, replay_bloch_jax,
                           finite_180_longitudinal_dwell, pre_pulse_gradient_phase,
                           pathway_sign_se)
from .mt_walk import simulate_mt_trajectories
from .bloch import simulate_bloch
from .pulse_sequence import (BlochSequence, gradient_echo, spin_echo,
                             prepend_mt_prep, run_bloch_sequence, emergent_z_spectrum)
from .gpu import (gpu_available, check_gpu, free_gpu_memory, gpu_session,
                  list_gpu_processes)
from .viz import (plot_waveform, plot_sequence_comparison,
                  plot_mesh_section, plot_walkers_3d, plot_cell_surface, plot_mesh_3d,
                  seed_in_cell, walk_paths, plot_trajectories, save_rotation)
from .waveforms import (Waveform, pgse, pgste, ogse, cpmg, trapezoidal_ogse, b_trapezoidal_ogse,
                        set_b, calc_b, calc_btensor, btensor_invariants, ste, pte,
                        rotate_waveform, tile_waveform)
from .geometries import (FreeDiffusion, Box1D, Sphere, Cylinder, MyelinatedCylinder,
                         Ellipsoid,
                         PackedCylinders, pack_cylinders,
                         PackedSpheres, pack_spheres,
                         PackedMyelinatedCylinders,
                         pack_myelinated_cylinders,
                         PermeableSlab1D)
from .mesh import Mesh, load_ply
from .susceptibility import (SusceptibilitySources, MyelinSusceptibility,
                             GridSusceptibility, dipole_field,
                             myelin_susceptibility_tensor, radial_from_sdf, sample_grid)
from . import mesh_shapes
from .constants import GAMMA
from .noise import add_rician_noise, add_rician_noise_batch, add_nc_chi_noise, estimate_sigma
from .sh_convolution import (
    compute_fiber_response,
    apply_odf,
    watson_odf_sh,
    isotropic_odf_sh,
)
from . import mt
from .rf import B1Pulse, bloch_simulate, slice_profile

__all__ = [
    "simulate", "simulate_mixture", "simulate_cpmg", "simulate_trajectories",
    # replay path: walk-once producer + scalar replay operators
    "unwrap_periodic", "replay", "replay_jax",
    # replay path: vector-Bloch + susceptibility + MT + refocusing helpers
    "replay_bloch", "replay_bloch_jax",
    "finite_180_longitudinal_dwell", "pre_pulse_gradient_phase", "pathway_sign_se",
    "simulate_mt_trajectories",
    "simulate_bloch",
    "BlochSequence", "gradient_echo", "spin_echo", "prepend_mt_prep",
    "run_bloch_sequence", "emergent_z_spectrum",
    "gpu_available", "check_gpu", "free_gpu_memory", "gpu_session", "list_gpu_processes",
    "Waveform", "pgse", "pgste", "ogse", "cpmg", "trapezoidal_ogse", "b_trapezoidal_ogse",
    "set_b", "calc_b", "calc_btensor", "btensor_invariants", "ste", "pte",
    "rotate_waveform", "tile_waveform",
    "FreeDiffusion", "Box1D", "Sphere", "Cylinder", "MyelinatedCylinder",
    "Ellipsoid",
    "PackedCylinders", "pack_cylinders",
    "PackedSpheres", "pack_spheres",
    "PackedMyelinatedCylinders", "pack_myelinated_cylinders",
    "PermeableSlab1D",
    "Mesh", "load_ply",
    # susceptibility off-resonance fields (forward Bloch)
    "SusceptibilitySources", "MyelinSusceptibility", "GridSusceptibility",
    "dipole_field", "myelin_susceptibility_tensor", "radial_from_sdf", "sample_grid",
    "mesh_shapes",
    "GAMMA",
    "add_rician_noise", "add_rician_noise_batch", "add_nc_chi_noise", "estimate_sigma",
    # SH convolution
    "compute_fiber_response",
    "apply_odf",
    "watson_odf_sh",
    "isotropic_odf_sh",
    # magnetization transfer (physics + analytic oracle)
    "mt",
    # continuous RF pulses: complex B1(t) envelope + Bloch forward
    "B1Pulse", "bloch_simulate", "slice_profile",
]
