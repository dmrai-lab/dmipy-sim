"""One loader for inner/outer surface-mesh substrates, whatever generator wrote them.

An isolated axon and a packed bundle are the same object at different counts: both are inner (axonal) and
outer (myelin) triangulated surfaces in micrometres, both build the same :class:`CactusBundle`, and
everything downstream -- containment, seeding, walking, the field basis -- is identical. The two genuine
differences are *how many files* there are and *what the box is*, so those are the arguments; the rest is
name resolution, which belongs in the per-dataset helpers (:mod:`dmipy_sim.io.winther`,
:mod:`dmipy_sim.io.cactus`).

Keeping them separate cost real correctness. The paths drifted into different containment predicates
(dmipy-sim#23, #33) and, in the public port, into different volume-fraction arithmetic: the Winther path
built its `trimesh` on already-scaled vertices and then multiplied the volume by ``scale**3`` again, so at
the default ``scale=1e-6`` every fraction collapsed by 1e-18 -- ``f_intra`` 0.1218 -> 1.2e-19 and
``f_extra`` -> 1.0. Nothing caught it because the only test that exercised the arithmetic ran at
``scale=1.0``, where the double conversion is the identity. Volumes are therefore computed **once, from the
scaled vertices**, and a scale-invariance test pins it.

Facts that the meshes already carry are measured, never accepted (dmipy-sim#36): the g-ratio follows from
the two volumes, since for a tube ``V ~ r^2 L`` gives ``g = r_in/r_out = sqrt(V_in/V_out)``.
"""
from __future__ import annotations

import warnings

import numpy as np

from ..mesh import load_ply

_UM = 1e-6


def _concat(meshes):
    """Concatenate ``[(V, F), ...]`` into one ``(V, F)``, offsetting face indices."""
    Vs, Fs, off = [], [], 0
    for V, F in meshes:
        Vs.append(V)
        Fs.append(F + off)
        off += len(V)
    if not Vs:
        return np.zeros((0, 3)), np.zeros((0, 3), np.int64)
    return np.vstack(Vs), np.vstack(Fs).astype(np.int64)


def _volume_and_watertight(V, F):
    """Enclosed volume (in the units of ``V``) and watertightness of one surface.

    ``V`` is already in metres, so the volume is already in m^3 -- there is no second unit conversion to
    apply here. That is the whole content of the bug this module's docstring describes.
    """
    import trimesh
    m = trimesh.Trimesh(np.asarray(V, np.float64), np.asarray(F, np.int64), process=False)
    return abs(float(m.volume)), bool(m.is_watertight)


def resolve_box(spec, outer_V, *, scale=1.0):
    """Turn a box specification into ``(box_min, box_max)`` in metres.

    The box is explicit because it is a real physical difference between substrates, not a file-format one:

    * ``('pad', p)`` -- the outer surfaces' bounding box grown by ``p`` metres. An isolated axon sits in
      free water, and the pad is what leaves room for the exterior region to be resolved.
    * ``('periodic', side)`` -- the centred cube ``[-side/2, side/2]^3``, ``side`` in metres. A packed
      bundle's simulation volume is the periodic cell, which the fibres over-run along their own axis by
      their end caps.
    * ``(box_min, box_max)`` -- given outright, as when reconstructing a substrate from stored metadata.
    """
    if isinstance(spec, (tuple, list)) and len(spec) == 2 and isinstance(spec[0], str):
        kind, value = spec
        if kind == "pad":
            p = float(value)
            return outer_V.min(0) - p, outer_V.max(0) + p
        if kind == "periodic":
            half = 0.5 * float(value)
            return np.array([-half, -half, -half]), np.array([half, half, half])
        raise ValueError(f"box kind must be 'pad' or 'periodic', got {kind!r}")
    if isinstance(spec, (tuple, list)) and len(spec) == 2:
        return np.asarray(spec[0], float), np.asarray(spec[1], float)
    raise ValueError("box must be ('pad', p), ('periodic', side) or (box_min, box_max); "
                     f"got {spec!r}")


def load_mesh_substrate(inner_plys, outer_plys, *, box, scale=_UM, g_ratio=None,
                        volume_reference=None, fibre_tangents=None, has_extra_substrate=None,
                        run_dir="", label=""):
    """Load paired inner/outer surface PLYs into a :class:`CactusBundle`.

    Parameters
    ----------
    inner_plys, outer_plys : sequence of str
        Paths to the axonal (inner) and myelin (outer) surfaces. One entry each for a single axon, one per
        strand for a bundle. Read through :func:`dmipy_sim.mesh.load_ply`, which carries the one genuinely
        format-specific quirk in these datasets (MATLAB ``plywrite`` declaring a 16-bit face-index type past
        65536 vertices, dmipy-sim#18).
    box : see :func:`resolve_box`
        ``('pad', p)``, ``('periodic', side)``, or an explicit ``(box_min, box_max)``, in metres.
    scale : float
        Mesh units -> metres (these datasets store micrometres, so ``1e-6``).
    g_ratio : float, optional
        Cross-check only. The stored value is always the meshes' own ``sqrt(V_in/V_out)``; a disagreeing
        argument warns and loses, because the alternative is a caller's default being written into
        provenance while the geometry says otherwise, with nothing comparing them (dmipy-sim#36).
    volume_reference : {'box', 'fibre_extent'}, optional
        What the volume fractions are fractions *of*. ``'box'`` uses the whole box. ``'fibre_extent'`` uses
        the box cross-section times the meshes' extent along the fibre axis, which removes end-cap overhang
        from the ratio -- the right choice for a bundle whose fibres run past the periodic cell. Defaults to
        ``'fibre_extent'`` for a periodic box and ``'box'`` otherwise.
    has_extra_substrate : bool, optional
        Whether the space outside the outer surfaces is *substrate* rather than free water. True for a
        packed bundle (extra-axonal space between strands carries structure), False for an isolated axon
        (its surroundings are free water carrying no substrate information). Defaults to whether more than
        one fibre was loaded. Kept on the substrate rather than passed at walk time, so the pool structure
        travels with the geometry that determines it.
    """
    inner_plys = [str(p) for p in ([inner_plys] if isinstance(inner_plys, (str, bytes)) else inner_plys)]
    outer_plys = [str(p) for p in ([outer_plys] if isinstance(outer_plys, (str, bytes)) else outer_plys)]

    inner_meshes, outer_meshes = [], []
    vol_inner = vol_outer = 0.0        # m^3
    open_surfaces = []
    for paths, meshes, tag in ((inner_plys, inner_meshes, "inner"),
                               (outer_plys, outer_meshes, "outer")):
        for p in paths:
            V, F = load_ply(p, scale=scale)
            meshes.append((V, F))
            vol, watertight = _volume_and_watertight(V, F)
            if tag == "inner":
                vol_inner += vol
            else:
                vol_outer += vol
            if not watertight:
                open_surfaces.append(p)

    if not outer_meshes:
        raise ValueError("no outer surfaces loaded; an outer (myelin) surface defines the fibre volume")
    if open_surfaces:
        warnings.warn(
            f"{label or 'substrate'}: {len(open_surfaces)} of {len(inner_plys) + len(outer_plys)} surfaces "
            f"are not watertight (e.g. {open_surfaces[0]}); volume fractions use |mesh.volume| and "
            f"containment by ray parity is undefined through the rims (dmipy-sim#50).", stacklevel=2)

    inner = _concat(inner_meshes)
    outer = _concat(outer_meshes)

    box_min, box_max = resolve_box(box, outer[0], scale=scale)
    L = box_max - box_min
    ext = outer[0].max(0) - outer[0].min(0)
    fibre_axis = int(np.argmax(ext))            # the elongated axis

    periodic = isinstance(box, (tuple, list)) and len(box) == 2 and box[0] == "periodic"
    if volume_reference is None:
        volume_reference = "fibre_extent" if periodic else "box"
    if volume_reference == "fibre_extent":
        cross = float(np.prod([L[a] for a in range(3) if a != fibre_axis]))
        v_ref = cross * float(ext[fibre_axis])
    elif volume_reference == "box":
        v_ref = float(np.prod(L))
    else:
        raise ValueError(f"volume_reference must be 'box' or 'fibre_extent', got {volume_reference!r}")
    if v_ref <= 0:
        raise ValueError(f"{label or 'substrate'}: reference volume is {v_ref}; check the box")

    # g is the geometry's, measured from the two volumes it already gave us.
    g_measured = float(np.sqrt(vol_inner / vol_outer)) if vol_outer > 0 else float("nan")
    if g_ratio is not None and np.isfinite(g_measured) and abs(float(g_ratio) - g_measured) > 0.02:
        warnings.warn(
            f"{label or 'substrate'}: supplied g_ratio={float(g_ratio):.3f} disagrees with the meshes' own "
            f"{g_measured:.3f} (from V_in/V_out). Using the measured value; drop the argument to silence "
            f"this.", stacklevel=2)
    g_out = g_measured if np.isfinite(g_measured) else (0.7 if g_ratio is None else float(g_ratio))

    f_axon = vol_outer / v_ref
    f_intra = vol_inner / v_ref
    f_myelin = max(0.0, f_axon - f_intra)
    f_extra = max(0.0, 1.0 - f_axon)

    n_fibres = max(len(inner_meshes), len(outer_meshes))
    if has_extra_substrate is None:
        has_extra_substrate = n_fibres > 1

    from .cactus import CactusBundle
    return CactusBundle(
        inner=inner, outer=outer, box_min=box_min, box_max=box_max,
        g_ratio=float(g_out), f_intra=float(f_intra), f_myelin=float(f_myelin),
        f_extra=float(f_extra), n_fibres=int(n_fibres), fibre_axis=fibre_axis,
        run_dir=str(run_dir),
        fibre_tangents=(np.asarray(fibre_tangents, float) if fibre_tangents is not None else None),
        has_extra_substrate=bool(has_extra_substrate))
