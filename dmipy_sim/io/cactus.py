"""CACTUS axon-bundle substrates: the :class:`CactusBundle` container and the run-directory loader.

CACTUS (Villarreal-Haro et al., https://github.com/Juanitovh/CACTUS) grows realistic, dispersed,
variable-caliber axons and emits, **per fibre**, an OUTER surface of radius ``r_outer`` and -- via a
mesh-time g-ratio -- an INNER surface of radius ``g * r_outer``. The three white-matter compartments follow
directly:

    intra   = inside the inner surface
    myelin  = the inner -> outer shell
    extra   = the periodic box minus the outer surfaces

Myelin water is effectively stuck (``D ~= 0``), so it is a frozen short-T2 pool that needs no walk. The two
concatenated multi-surface meshes feed two independent :class:`dmipy_sim.mesh.Mesh` walks -- intra restricted
by the inner wall, extra hindered by the outer -- assembled by
:func:`dmipy_sim.mesh_axon.mesh_axon_master`.

Geometry loading is shared with every other inner/outer mesh dataset; see
:mod:`dmipy_sim.io.mesh_substrate` for why, and :mod:`dmipy_sim.io.winther` for the single-axon sibling.
"""
from __future__ import annotations

import glob
import os
import re
from dataclasses import dataclass

import numpy as np

_UM = 1e-6  # CACTUS stores micrometres; the simulator works in metres.


@dataclass
class CactusBundle:
    """Two multi-surface meshes (metres) + box + volume fractions."""
    inner: tuple           # (V, F) metres — axonal (inner) surface(s)
    outer: tuple           # (V, F) metres — myelin (outer) surface(s)
    box_min: np.ndarray    # (3,) metres
    box_max: np.ndarray    # (3,) metres
    g_ratio: float
    f_intra: float
    f_myelin: float
    f_extra: float
    n_fibres: int
    fibre_axis: int        # 0/1/2 — the elongated (fibre) axis in mesh coords
    run_dir: str = ""
    fibre_tangents: np.ndarray = None   # (n_fibres, 3) unit per-fibre mean tangents
    # Is the space outside the outer surfaces substrate, or free water? True for a packed bundle, where the
    # extra-axonal space between strands carries structure; False for an isolated axon, whose surroundings
    # carry no substrate information. A property of the geometry, so it travels with it rather than being
    # re-decided by each caller at walk time.
    has_extra_substrate: bool = True

    @property
    def box_side(self):
        return self.box_max - self.box_min

    def summary(self):
        return (f"CactusBundle({self.n_fibres} fibre(s), g={self.g_ratio:.2f}, "
                f"box={np.round(self.box_side / _UM, 1)} um, axis={'xyz'[self.fibre_axis]}, "
                f"f=[intra {self.f_intra:.3f}, myelin {self.f_myelin:.3f}, extra {self.f_extra:.3f}]"
                f"{'' if self.has_extra_substrate else ', extra=free water'})")


def _box_side_um(run_dir):
    """Voxel side (um) from the ``optimized_final.txt`` header (first line)."""
    f = os.path.join(run_dir, "optimized_final.txt")
    if not os.path.isfile(f):
        return None
    with open(f) as fh:
        return float(fh.readline().strip())


def _discover(sim_dir):
    """Map strand id -> {'inner': path, 'outer': path} for every meshed fibre."""
    pat = re.compile(r"strand_(\d+)_(inner|outer)_erode_\d+\.ply$")
    out = {}
    for p in sorted(glob.glob(os.path.join(sim_dir, "strand_*_erode_*.ply"))):
        m = pat.search(os.path.basename(p))
        if m:
            out.setdefault(int(m.group(1)), {})[m.group(2)] = p
    return out


def fibre_mean_tangents(run_dir):
    """Per-fibre unit mean tangent ``(n_fibres, 3)`` from the ``optimized_final.txt`` centrelines.

    The endpoint-to-endpoint direction of each fibre, which is what an orientation frame should be built
    from per bundle rather than as one global average. Sign is arbitrary (orientation is axial). Returns
    None if the centreline file is absent.
    """
    path = os.path.join(run_dir, "optimized_final.txt")
    if not os.path.isfile(path):
        return None
    lines = open(path).read().splitlines()
    i, tang = 3, []                                  # skip the 3-line header (side / n_fibres / n_ctrl)
    while i < len(lines):
        tok = lines[i].split()
        if len(tok) == 1:                            # a fibre block: count, then that many control points
            n = int(float(tok[0])); i += 1
            blk = np.array([[float(x) for x in lines[i + j].split()[:3]] for j in range(n)])
            d = blk[-1] - blk[0]; nn = np.linalg.norm(d)
            if nn > 0:
                tang.append(d / nn)
            i += n
        else:
            i += 1
    return np.asarray(tang) if tang else None


def load_cactus_bundle(run_dir, *, g_ratio=None, scale=_UM, side_um=None,
                       require_pairs=True, volume_reference=None):
    """Load a CACTUS run directory into a :class:`CactusBundle`.

    Parameters
    ----------
    run_dir : str
        The per-repetition experiment folder, containing ``optimized_final.txt`` and
        ``meshes/simulations/*.ply``.
    g_ratio : float, optional
        Cross-check only; the stored value is measured from the meshes. Meaningless as an *input* for a
        bundle in any case, where g varies fibre to fibre (measured 0.560-0.872 across 80 strands).
    scale : float
        Mesh units -> metres (default ``1e-6``: CACTUS micrometres).
    side_um : float, optional
        Periodic cell side (um). Defaults to the ``optimized_final.txt`` header, and failing that to the
        largest outer-mesh extent.
    require_pairs : bool
        Use only fibres having BOTH surfaces. A fibre with an outer but no inner mesh would have its whole
        interior assigned to myelin, which is why this defaults to True.
    volume_reference : {'box', 'fibre_extent'}, optional
        See :func:`dmipy_sim.io.mesh_substrate.load_mesh_substrate`. Defaults to ``'fibre_extent'`` here:
        CACTUS fibres are finite and their end caps over-run the periodic cell along the fibre axis, so
        dividing by the whole box would understate the fractions.
    """
    from .mesh_substrate import load_mesh_substrate

    sim_dir = os.path.join(run_dir, "meshes", "simulations")
    if not os.path.isdir(sim_dir):
        raise FileNotFoundError(f"no meshes/simulations under {run_dir}")
    found = _discover(sim_dir)
    if not found:
        raise FileNotFoundError(f"no strand PLYs in {sim_dir}")

    inner_plys, outer_plys = [], []
    for sid in sorted(found):
        paths = found[sid]
        if require_pairs and not ("inner" in paths and "outer" in paths):
            continue
        if "inner" in paths:
            inner_plys.append(paths["inner"])
        if "outer" in paths:
            outer_plys.append(paths["outer"])
    if not outer_plys:
        raise FileNotFoundError(
            f"no strand has both surfaces in {sim_dir} (found {len(found)} strands); run the inner mesh "
            f"pass, or pass require_pairs=False to load the outer surfaces alone")

    side = side_um if side_um is not None else _box_side_um(run_dir)
    if side is None:
        # No config header: fall back to the meshes' own largest extent, in mesh units.
        from ..mesh import load_ply
        V0, _ = load_ply(outer_plys[0], scale=1.0)
        side = float((V0.max(0) - V0.min(0)).max())

    return load_mesh_substrate(
        inner_plys, outer_plys, box=("periodic", float(side) * scale), scale=scale,
        g_ratio=g_ratio, volume_reference=volume_reference,
        fibre_tangents=fibre_mean_tangents(run_dir),
        has_extra_substrate=True, run_dir=run_dir,
        label=f"cactus:{os.path.basename(os.path.normpath(run_dir))}")
