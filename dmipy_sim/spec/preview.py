"""What is inside a substrate, as a picture: three orthogonal cross-sections through the centre of a spec's domain.

:func:`preview` is the one renderer every dataset card calls (dmipy-sim#480). It dispatches on the kinds of the
spec's walls and nothing else, and in every case the picture is drawn from the SAME membership test the walk
uses, so a preview cannot show a substrate the walk does not have:

* a ``label_volume`` is already a grid, so the three planes are three slices of it, cropped as the spec says;
* ``mesh``, ``sphere_union`` and ``swept_polyline`` walls go through :class:`~dmipy_sim.spec.walk._Boundary`,
  the one implementation of "inside these closed surfaces" that seeding and walking share -- a mesh cut, packed
  spheres in section, a strand bundle in section;
* an analytic wall (a cylinder, a sphere, an ellipsoid, a plane, one or many) goes through
  ``geometry_from_spec(spec).classify_position``, the one compartment map of an analytic substrate.

The returned record is what the card embeds beside the picture: which plane is where, the extent in metres, the
scale bar, and the area fraction of each pool in the section -- a number a reader can compare with the spec's
own realisation.
"""
from __future__ import annotations

import os

import numpy as np

from .substrate import SpecError

#: Pixels across the longest edge of a section. The picture is a provenance thumbnail, not a figure.
PREVIEW_PIXELS = 360

#: Colours by pool id, id 0 (the free / extra-cellular pool) first -- the `.rpk` compartment convention.
POOL_COLOURS = ("#ffffff", "#2b2b2b", "#b03a2e", "#2874a6", "#239b56", "#b7950b")

__all__ = ["preview", "PREVIEW_PIXELS"]


def _scale_bar(extent_m):
    """A round bar under a fifth of the section's extent, in metres."""
    target = extent_m / 5.0
    decade = 10.0 ** np.floor(np.log10(target))
    for mult in (5.0, 2.0, 1.0):
        if mult * decade <= target:
            return float(mult * decade)
    return float(decade)


def _planes(lo, hi):
    """The three centre planes as ``(normal axis, the two in-plane axes, the coordinate of the cut)``."""
    mid = 0.5 * (np.asarray(lo, float) + np.asarray(hi, float))
    return [(k, tuple(a for a in (0, 1, 2) if a != k), float(mid[k])) for k in (0, 1, 2)]


def _section_from_labels(wall):
    """The three slices of a ``label_volume``: its own grid, read and cropped by the one reader a walk uses
    (:func:`~dmipy_sim.spec.build.label_volume_arrays`, which checks the cited sha256), with the pool of each
    voxel. A label grid already IS the wall, so there is nothing to rasterise."""
    from .build import label_volume_arrays
    labels, voxel_size, _origin, names = label_volume_arrays(wall.surface)
    labels = np.asarray(labels)
    ids = np.full(labels.shape, -1, np.int8)
    for pid, value in enumerate(names):                                # insertion order IS the pool id
        ids[labels == value] = pid
    h = float(np.asarray(voxel_size, float)[0])
    c = [n // 2 for n in ids.shape]
    out = []
    for k, (a1, a2), _ in _planes([0, 0, 0], [n * h for n in ids.shape]):
        plane = np.take(ids, c[k], axis=k)
        out.append(dict(normal=k, in_plane=(a1, a2), at_m=c[k] * h, ids=plane,
                        extent_m=(plane.shape[0] * h, plane.shape[1] * h)))
    return out, list(names.values()), h          # insertion order IS the pool id


def _inside_fn(spec):
    """``(pool id) -> (points -> bool)``, from whichever one membership implementation this spec's walls have."""
    kinds = {w.surface.kind for w in spec.walls}
    if kinds <= {"mesh", "sphere_union", "swept_polyline"}:
        from .walk import _PoolTests
        tests = _PoolTests(spec)
        return {p.id: tests.member(p.id) for p in spec.pools}
    from .build import geometry_from_spec
    geom = geometry_from_spec(spec)
    def member(pid):
        def pred(pts):
            comp = geom.classify_positions_exact(np.asarray(pts, float))
            # a packed geometry's classifier returns the OBJECT (or an encoded lumen/sheath id); `pool_of` is the
            # one map to the pool the spec names, and is the identity where the classifier already gives one
            comp = np.asarray(geom.pool_of(comp) if hasattr(geom, "pool_of") else comp)
            return comp == pid
        return pred
    return {p.id: member(p.id) for p in spec.pools}


def _section_by_membership(spec):
    """The three centre planes rasterised by the spec's own membership tests."""
    lo, hi = np.asarray(spec.domain.box_min, float), np.asarray(spec.domain.box_max, float)
    members = _inside_fn(spec)
    step = float(np.max(hi - lo)) / PREVIEW_PIXELS
    out = []
    for k, (a1, a2), at in _planes(lo, hi):
        n1 = max(2, int(round((hi[a1] - lo[a1]) / step)))
        n2 = max(2, int(round((hi[a2] - lo[a2]) / step)))
        u = lo[a1] + (np.arange(n1) + 0.5) * (hi[a1] - lo[a1]) / n1
        v = lo[a2] + (np.arange(n2) + 0.5) * (hi[a2] - lo[a2]) / n2
        U, V = np.meshgrid(u, v, indexing="ij")
        pts = np.empty((U.size, 3)); pts[:, k] = at; pts[:, a1] = U.ravel(); pts[:, a2] = V.ravel()
        ids = np.full(U.size, -1, np.int8)
        for p in sorted(spec.pools, key=lambda q: -q.id):              # an enclosed pool wins over the free one
            ids[np.asarray(members[p.id](pts), bool)] = p.id
        out.append(dict(normal=k, in_plane=(a1, a2), at_m=at, ids=ids.reshape(U.shape),
                        extent_m=(float(hi[a1] - lo[a1]), float(hi[a2] - lo[a2]))))
    return out, [p.name for p in spec.pools], float(step)


def preview(spec, path=None, *, dpi=110, title=None):
    """Three orthogonal cross-sections through the centre of ``spec``'s domain, written to ``path`` as one PNG.

    Returns the record the card embeds: the file, the surface kinds it dispatched on, the pixel size, the scale
    bar, and per plane its normal axis, where it cuts, its extent in metres and the area fraction of every pool.
    A spec with no wall has nothing to show and is refused.
    """
    if not spec.walls:
        raise SpecError("preview: this spec has no wall, so it has no cross-section to show")
    kinds = sorted({w.surface.kind for w in spec.walls})
    if kinds == ["label_volume"]:
        sections, names, pixel = _section_from_labels(spec.walls[0])
    else:
        sections, names, pixel = _section_by_membership(spec)
    bar = _scale_bar(max(max(s["extent_m"]) for s in sections))

    planes = []
    for s in sections:
        ids = s["ids"]
        frac = {names[p]: float(np.mean(ids == p)) for p in range(len(names))}
        planes.append(dict(normal="xyz"[s["normal"]], at_m=s["at_m"], extent_m=list(s["extent_m"]),
                           pool_area_fraction=frac))

    if path is not None:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from matplotlib.colors import ListedColormap
        cmap = ListedColormap(["#d9d9d9"] + [POOL_COLOURS[i % len(POOL_COLOURS)]
                                             for i in range(len(names))])
        fig, axes = plt.subplots(1, 3, figsize=(7.2, 2.6), dpi=dpi)
        for ax, s in zip(axes, sections):
            a1, a2 = s["in_plane"]
            ax.imshow(np.asarray(s["ids"]).T + 1, cmap=cmap, vmin=0, vmax=len(names),
                      interpolation="nearest", origin="lower")
            ax.set_title(f"{'xyz'[s['normal']]} = centre", fontsize=8)
            ax.set_xlabel(f"{'xyz'[a1]}, {'xyz'[a2]}", fontsize=7)
            ax.set_xticks([]); ax.set_yticks([])
            px = bar / (s["extent_m"][0] / s["ids"].shape[0])
            x0, y0 = 0.06 * s["ids"].shape[0], 0.055 * s["ids"].shape[1]
            ax.plot([x0, x0 + px], [y0, y0], lw=2.5, color="tab:red")
            ax.text(x0, y0 + 0.03 * s["ids"].shape[1], f"{bar * 1e6:g} um", color="tab:red", fontsize=7)
        fig.suptitle(title or f"{spec.id}: {', '.join(names)} ({', '.join(kinds)})", fontsize=9)
        fig.tight_layout(rect=(0, 0, 1, 0.94))
        os.makedirs(os.path.dirname(os.path.abspath(path)) or ".", exist_ok=True)
        fig.savefig(path)
        plt.close(fig)
    return dict(path=(None if path is None else os.path.basename(path)), kinds=kinds, pools=names,
                pixel_m=pixel, scale_bar_m=bar, planes=planes)
