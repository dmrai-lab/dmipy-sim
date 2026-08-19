"""Winther et al. (DRCMR MAP susceptibility-and-axon-morphology) single-axon meshes.

A Winther axon is two triangulated surface PLYs per axon per morphological config --
``axonNN-...inner.ply`` (axonal) and ``axonNN-...outer.ply`` (myelin) -- in micrometres, aligned to +z
(Winther et al., Sci Rep 2024, doi:10.1038/s41598-024-79043-5; data:
resources.drcmr.dk/MAPdata/susceptibility-and-axon-morphology-dataset). Coaxial cylinders use a g-ratio of
0.7; chi is isotropic (chi_myelin-chi_water ~ +1.06e-6), B0 = 7 T, D0 = 0.6e-9 m^2/s.

This module is name resolution: the geometry itself is loaded by
:func:`dmipy_sim.io.mesh_substrate.load_mesh_substrate`, shared with the CACTUS bundles
(:mod:`dmipy_sim.io.cactus`). The one substantive difference is the box -- an isolated axon gets a padded
bounding box, a packed bundle a periodic cell -- and the fact that an isolated axon's surroundings are free
water rather than substrate, which the loaded substrate records as ``has_extra_substrate=False``.
"""
from __future__ import annotations

import glob
import os
import re

import numpy as np

from .cactus import CactusBundle  # re-exported: callers import the container from either module

_UM = 1e-6
# 29 XNH-segmented C5 axons selected by Winther's production/1a white_list (bounding-box feasible).
WINTHER_C5_AXONS = ("axon06", "axon08", "axon12", "axon13", "axon14", "axon15", "axon18", "axon22",
                    "axon24", "axon25", "axon26", "axon27", "axon31", "axon32", "axon34", "axon38",
                    "axon40", "axon41", "axon43", "axon45", "axon46", "axon47", "axon48", "axon49",
                    "axon50", "axon51", "axon52", "axon53", "axon54")


def load_winther_bundle(inner_ply, outer_ply, *, scale=_UM, g_ratio=None, pad=1.0e-6,
                        fibre_tangent=(0.0, 0.0, 1.0), axon_id=None):
    """Build a single-fibre :class:`CactusBundle` from one axon's inner + outer surface PLYs.

    ``scale`` converts mesh units to metres (Winther meshes are micrometres -> ``1e-6``). ``pad`` grows the
    box around the outer surface so the exterior region is resolved. The axon is loaded at its FULL length:
    the longitudinal diameter variation and tortuosity along the whole axon are the morphology under study,
    so the mesh is never truncated.

    ``g_ratio`` is a cross-check only -- the stored value is the meshes' own ``sqrt(V_in/V_out)``. The
    shipped default of 0.7 happens to be true for this dataset, whose outer surfaces are constructed from
    the inner at a fixed g (measured 0.695-0.704 across all 29), which is exactly why accepting it as an
    input was able to go unnoticed.
    """
    from .mesh_substrate import load_mesh_substrate
    return load_mesh_substrate(
        [inner_ply], [outer_ply], box=("pad", float(pad)), scale=scale, g_ratio=g_ratio,
        volume_reference="box",
        fibre_tangents=np.asarray([fibre_tangent], float),
        # An isolated axon sits in free water: its exterior carries no substrate information, so a walk
        # should not build an extra-axonal pool from it.
        has_extra_substrate=False,
        run_dir=str(os.path.dirname(str(inner_ply))),
        label=f"winther:{axon_id}" if axon_id else "winther")


def _discover_axon_plys(mesh_dir, axon_id):
    """Find the ('...inner.ply', '...outer.ply') pair for one axon id in ``mesh_dir``."""
    inner = sorted(p for p in glob.glob(os.path.join(mesh_dir, f"*{axon_id}*inner*.ply")))
    outer = sorted(p for p in glob.glob(os.path.join(mesh_dir, f"*{axon_id}*outer*.ply")))
    if not inner or not outer:
        raise FileNotFoundError(f"no inner/outer PLY pair for {axon_id!r} in {mesh_dir} "
                                f"(inner={inner}, outer={outer})")
    return inner[0], outer[0]


def load_winther_axon(mesh_dir, axon_id, **kw):
    """Convenience: discover an axon's inner/outer PLY pair in ``mesh_dir`` and load the bundle."""
    inner, outer = _discover_axon_plys(mesh_dir, axon_id)
    return load_winther_bundle(inner, outer, axon_id=axon_id, **kw)


def available_axons(mesh_dir):
    """List axon ids (``axonNN``) that have BOTH an inner and an outer PLY in ``mesh_dir``."""
    have = {}
    for p in glob.glob(os.path.join(mesh_dir, "*.ply")):
        m = re.search(r"(axon\d+)", os.path.basename(p))
        if not m:
            continue
        tag = m.group(1); low = os.path.basename(p).lower()
        have.setdefault(tag, set()).add("inner" if "inner" in low else "outer" if "outer" in low else "?")
    return sorted(t for t, s in have.items() if {"inner", "outer"} <= s)
