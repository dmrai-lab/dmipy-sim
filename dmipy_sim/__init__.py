"""dmipy-sim: JAX Monte Carlo diffusion MRI simulator.

Forward Monte-Carlo of spins diffusing under arbitrary free gradient waveforms
``G(t)``, with surface relaxivity and membrane permeability baked into the walk.
Shares one pulse-sequence and substrate interface with ``dmipy-fit``.
"""

# Apply the GPU memory cap (DMIPY_GPU_MEM_GB) BEFORE any submodule imports JAX.
from ._gpu_config import apply_gpu_mem_cap as _apply_gpu_mem_cap, configure  # noqa: E402
_apply_gpu_mem_cap()

from .core import simulate, simulate_mixture, simulate_cpmg
from .gpu import (gpu_available, check_gpu, free_gpu_memory, gpu_session,
                  list_gpu_processes)
from .viz import plot_waveform, plot_sequence_comparison
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
from .constants import GAMMA
from .noise import add_rician_noise, add_rician_noise_batch, add_nc_chi_noise, estimate_sigma
from .sh_convolution import (
    compute_fiber_response,
    apply_odf,
    watson_odf_sh,
    isotropic_odf_sh,
)

__all__ = [
    "simulate", "simulate_mixture", "simulate_cpmg",
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
    "GAMMA",
    "add_rician_noise", "add_rician_noise_batch", "add_nc_chi_noise", "estimate_sigma",
    # SH convolution
    "compute_fiber_response",
    "apply_odf",
    "watson_odf_sh",
    "isotropic_odf_sh",
]
