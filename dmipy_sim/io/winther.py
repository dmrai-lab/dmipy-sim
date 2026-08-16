"""Load Winther et al. (DRCMR MAP susceptibility-and-axon-morphology) axon meshes into a
:class:`dmipy_sim.io.cactus.CactusBundle`, so the CACTUS mesh->replay-pack pipeline
(:func:`dmipy_sim.bank_cactus.cactus_master`) applies unchanged.

A Winther axon is provided as two triangulated surface PLYs per axon per morphological config:
``axonNN-...inner.ply`` (the axonal/inner surface) and ``axonNN-...outer.ply`` (the myelin/outer
surface), in micrometres, aligned to +z (Winther et al., Sci Rep 2024, doi:10.1038/s41598-024-79043-5;
data: resources.drcmr.dk/MAPdata/susceptibility-and-axon-morphology-dataset). Coaxial cylinders use
a g-ratio of 0.7; χ is isotropic (χ_myelin−χ_water ≈ +1.06e-6), B0 = 7 T, D0 = 0.6e-9 m²/s.

Unlike CACTUS, an isolated axon has no meaningful extra-axonal *substrate* (the box is free water),
so build packs with ``cactus_master(..., include_extra=False)`` (intra + frozen myelin only).
"""
from __future__ import annotations

import glob
import os
import re

import numpy as np

from .cactus import CactusBundle

_UM = 1e-6
# 29 XNH-segmented C5 axons selected by Winther's production/1a white_list (bounding-box feasible).
WINTHER_C5_AXONS = ("axon06", "axon08", "axon12", "axon13", "axon14", "axon15", "axon18", "axon22",
                    "axon24", "axon25", "axon26", "axon27", "axon31", "axon32", "axon34", "axon38",
                    "axon40", "axon41", "axon43", "axon45", "axon46", "axon47", "axon48", "axon49",
                    "axon50", "axon51", "axon52", "axon53", "axon54")


def _load_ply(path, scale):
    """Read an ASCII PLY surface -> (V (n,3) float64 metres, F (m,3) int64, trimesh.Trimesh).

    These meshes are written by MATLAB ``plywrite`` with a **malformed header**: the face list
    declares a 16-bit index type (``property list uchar ushort vertex_indices``) while the meshes
    carry more than 65536 vertices (e.g. 79564). The file is ASCII so the text holds the correct
    indices, but a reader that honours the declared type (trimesh/pymesh) clamps them at 65535 and
    silently produces faces that span the whole axon — which destroys the geometry (and explodes any
    spatial acceleration structure). So parse the numbers directly, as the source study's own reader
    does, and validate the index range; trimesh is then used only for geometry (volume/watertight).
    """
    import trimesh
    with open(path, "rb") as fb:                      # header is ASCII even in a binary PLY
        raw = []
        for bline in fb:
            raw.append(bline.decode("ascii", "replace"))
            if raw[-1].strip() == "end_header":
                break
    if not any(l.startswith("format ascii") for l in raw):
        # A binary PLY states its index type truthfully, so the mis-declaration this function guards
        # against cannot bite; use the standard reader.
        mesh = trimesh.load(str(path), process=False, force="mesh")
        V = np.asarray(mesh.vertices, np.float64) * float(scale)
        return V, np.asarray(mesh.faces, np.int64), trimesh.Trimesh(V, mesh.faces, process=False)
    with open(path) as fh:
        header = []
        for line in fh:
            header.append(line)
            if line.strip() == "end_header":
                break
    n_header = len(header)
    n_v = int(next(l for l in header if l.startswith("element vertex")).split()[-1])
    n_f = int(next(l for l in header if l.startswith("element face")).split()[-1])
    V = np.loadtxt(path, skiprows=n_header, max_rows=n_v, usecols=(0, 1, 2)).astype(np.float64)
    F = np.loadtxt(path, skiprows=n_header + n_v, max_rows=n_f, usecols=(1, 2, 3)).astype(np.int64)
    if F.min() < 0 or F.max() >= n_v:
        raise ValueError(f"{path}: face index out of range [0,{n_v - 1}] (got [{F.min()},{F.max()}])")
    V = V * float(scale)
    return V, F, trimesh.Trimesh(V, F, process=False)


def load_winther_bundle(inner_ply, outer_ply, *, scale=_UM, g_ratio=None, pad=1.0e-6,
                        fibre_tangent=(0.0, 0.0, 1.0), axon_id=None):
    """Build a :class:`CactusBundle` (single fibre) from one axon's inner + outer surface PLYs.

    ``scale`` converts mesh units to metres (Winther meshes are in µm -> 1e-6). Volume fractions are
    from the mesh volumes over the padded outer bounding box; ``fibre_axis`` is the elongated bbox
    axis (Winther axons are +z-aligned). ``pad`` grows the box around the outer surface so the
    field/extra region is resolved. The axon is loaded at its FULL length — the longitudinal diameter
    variation and tortuosity along the whole axon are the morphology under study, so the mesh is
    never truncated.
    """
    Vi, Fi, mi = _load_ply(inner_ply, scale)
    Vo, Fo, mo = _load_ply(outer_ply, scale)

    box_min = Vo.min(0) - pad
    box_max = Vo.max(0) + pad
    box_vol = float(np.prod(box_max - box_min))
    fibre_axis = int(np.argmax(box_max - box_min))                 # elongated axis (=2 for +z axons)

    vol_in = abs(float(mi.volume)) * scale ** 3
    vol_out = abs(float(mo.volume)) * scale ** 3
    if not (mi.is_watertight and mo.is_watertight):
        import warnings
        warnings.warn(f"winther axon {axon_id or ''}: mesh not watertight "
                      f"(inner={mi.is_watertight}, outer={mo.is_watertight}); volume fractions "
                      f"use |mesh.volume| and may be approximate.", stacklevel=2)
    # g-ratio is a PROPERTY OF THE MESHES, not an input. Both volumes are already in hand, and for a tube
    # V ∝ r²L, so g = r_in/r_out = sqrt(V_in/V_out). Taking it as an argument let a caller's default be
    # written into pack provenance and substrate cards while the geometry said otherwise, with nothing to
    # notice: the shipped default 0.7 happens to match this dataset (whose outer surfaces are constructed
    # from the inner at a fixed g), but is meaningless for a bundle, where g varies fibre to fibre.
    g_measured = float(np.sqrt(vol_in / vol_out)) if vol_out > 0 else float("nan")
    if g_ratio is not None and np.isfinite(g_measured) and abs(float(g_ratio) - g_measured) > 0.02:
        import warnings
        warnings.warn(
            f"winther axon {axon_id or ''}: supplied g_ratio={float(g_ratio):.3f} disagrees with the "
            f"meshes' own {g_measured:.3f} (from V_in/V_out). Using the measured value; drop the argument "
            f"to silence this.", stacklevel=2)
    g_out = g_measured if np.isfinite(g_measured) else (0.7 if g_ratio is None else float(g_ratio))

    f_intra = vol_in / box_vol
    f_myelin = max(0.0, (vol_out - vol_in) / box_vol)
    f_extra = max(0.0, 1.0 - vol_out / box_vol)

    return CactusBundle(
        inner=(Vi, Fi), outer=(Vo, Fo), box_min=box_min, box_max=box_max,
        g_ratio=float(g_out), f_intra=float(f_intra), f_myelin=float(f_myelin),
        f_extra=float(f_extra), n_fibres=1, fibre_axis=fibre_axis,
        run_dir=str(os.path.dirname(str(inner_ply))),
        fibre_tangents=np.asarray([fibre_tangent], float))


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
