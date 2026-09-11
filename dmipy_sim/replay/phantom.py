"""Replay phantoms (``.rph``) — solved packs arranged in space, and replayed with the physics
layered on one channel at a time.

A phantom owns no walkers. It cites substrates per voxel with a geometric fraction each and an
orientation, so one solved pack serves every voxel and every orientation that cites it. The
format is specified in RPH.md of the replay-pack-spec; this module is the reference reader,
writer, and replay path.

The point of the replay side is that the physics is *separable*: the exponent of the signal is
a sum of independent channel terms (gradient, field, occupancy-weighted relaxation, surface
contact), so a caller adds a physics by layering a term rather than by re-simulating. That is
what :meth:`ReplayPhantom.replay` exposes -- each keyword turns on exactly one channel, and a
pack that does not carry the channel refuses rather than silently returning the signal without
it.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .so3 import n_sh_coeffs

__all__ = ["ReplayPhantom", "read_rph", "write_rph", "build_rph", "Grid", "ANALYTIC_MODELS", "pack_substrate",
           "analytic_substrate", "inert_substrate", "ODFField", "PeakField", "WatsonField", "FrameField",
           "BinghamField",
           "SUBSTRATE_KINDS", "SCALAR_REGISTRY"]

SUBSTRATE_KINDS = ("pack", "analytic", "inert")
RPH_SCHEMA_VERSION = "0.4.0"

#: The closed forms an ``analytic`` substrate may name (RPH.md 3.1), each a callable ``(params, b_values) ->
#: signal``. Orientation-independent by construction: a closed form has no pose.
ANALYTIC_MODELS = {}


def analytic_model(name):
    def reg(f):
        ANALYTIC_MODELS[name] = f
        return f
    return reg


@analytic_model("free_water")
def _free_water(params, b_values):
    """``exp(-b D)``: isotropic Gaussian diffusion, in closed form.

    Free water is the case where storing walkers is not merely wasteful but unusable: its signal decays
    exponentially in ``b`` while a Monte-Carlo floor decays only as ``1/sqrt(N)``, so at ``b = 3000 s/mm^2`` a
    4000-walker pack carries ~113x more noise than signal. The closed form has no such error, and no pose.
    """
    return np.exp(-np.atleast_1d(np.asarray(b_values, np.float64)) * float(params["diffusivity"]))


#: The macroscopic layers a phantom may declare per voxel (RPH.md 5.1). A name outside this registry is
#: refused rather than ignored: a layer silently dropped is a phantom that replays wrong while looking right.
SCALAR_REGISTRY = ("kappa_B1", "delta_B0_T", "m0_scale")


# ------------------------------------------------------------------ the grid in the scanner
@dataclass(frozen=True)
class Grid:
    """Where the voxels are in the bore (RPH.md 3, 7).

    ``shape`` voxels of ``voxel_size_m``, with ``origin_m`` the scanner coordinates of the centre of voxel
    ``(0, 0, 0)`` and ``isocenter_m`` the point the scanner is focused on. ``frame`` names the scanner axes the
    indices run along, and is what the acquisition's gradient and B0 directions are expressed in. Both
    positions default to a grid centred on the isocenter at the origin, which is what a synthetic phantom
    wants and what a macroscopic layer needs to be a function of position.
    """

    shape: tuple
    voxel_size_m: tuple
    origin_m: tuple = None
    isocenter_m: tuple = None
    frame: str = "RAS"

    def __post_init__(self):
        sh = tuple(int(v) for v in self.shape)
        vs = tuple(float(v) for v in self.voxel_size_m)
        if len(sh) != 3 or len(vs) != 3:
            raise ValueError(f"a grid is three-dimensional: got shape {self.shape}, voxel_size_m {self.voxel_size_m}")
        if min(sh) < 1 or min(vs) <= 0:
            raise ValueError(f"shape must be positive integers and voxel_size_m positive lengths: {sh}, {vs}")
        org = tuple(-0.5 * (n - 1) * d for n, d in zip(sh, vs)) if self.origin_m is None \
            else tuple(float(v) for v in self.origin_m)
        iso = tuple(o + 0.5 * (n - 1) * d for o, n, d in zip(org, sh, vs)) if self.isocenter_m is None \
            else tuple(float(v) for v in self.isocenter_m)
        object.__setattr__(self, "shape", sh); object.__setattr__(self, "voxel_size_m", vs)
        object.__setattr__(self, "origin_m", org); object.__setattr__(self, "isocenter_m", iso)

    @classmethod
    def from_meta(cls, meta):
        g = meta["grid"] if "grid" in meta else meta
        return cls(g["shape"], g["voxel_size_m"], g.get("origin_m"), g.get("isocenter_m"), g.get("frame", "RAS"))

    def to_meta(self):
        return {"shape": list(self.shape), "voxel_size_m": list(self.voxel_size_m),
                "origin_m": list(self.origin_m), "isocenter_m": list(self.isocenter_m), "frame": self.frame}

    def positions_m(self, voxel_index):
        """Scanner coordinates of the centres of the given voxels, ``(N, 3)``."""
        return np.asarray(self.origin_m) + np.asarray(voxel_index, np.float64) * np.asarray(self.voxel_size_m)

    def radius_m(self, voxel_index):
        """Distance of each voxel centre from the isocenter: what a macroscopic layer varies over."""
        return np.linalg.norm(self.positions_m(voxel_index) - np.asarray(self.isocenter_m), axis=-1)


# ------------------------------------------------------------------ substrate declarations
def pack_substrate(id, pack, m0, *, tissue=None):
    """A solved pack as a substrate: ``pack`` is a ``.rpk`` path or a :class:`ReplayPack`, ``m0`` its proton
    density, ``tissue`` the knobs it replays at (RPH.md 3.2) -- anything not named takes the pack's nominal
    value from its embedded specification."""
    return {"id": str(id), "kind": "pack", "m0": float(m0), "_pack": pack,
            **({"tissue": dict(tissue)} if tissue else {})}


def analytic_substrate(id, model, params, m0=1.0):
    """A closed-form substrate (RPH.md 3.1), e.g. ``analytic_substrate("csf", "free_water", {"diffusivity": 3e-9})``.

    Free water is analytic rather than a pack because its signal decays exponentially in b while a Monte-Carlo
    floor does not: at b = 3000 s/mm^2 a few-thousand-walker free-water pack carries orders of magnitude more
    noise than signal."""
    return {"id": str(id), "kind": "analytic", "m0": float(m0), "model": str(model), "params": dict(params)}


def inert_substrate(id="background/inert", m0=0.0):
    """Volume that occupies a voxel and emits nothing. Inert is not air: an air interface would dominate the
    field of its neighbouring voxels, which a voxel-by-voxel phantom has no mechanism for (RPH.md 3.1)."""
    return {"id": str(id), "kind": "inert", "m0": float(m0)}


# ------------------------------------------------------------------ orientation fields
class _OrientationField:
    """A pose per voxel for the substrates that have one. Subclasses declare the phantom's orientation mode and
    materialise, per voxel, the population weights and their orientation payload."""

    mode = None

    def _volume(self, a, trailing, name):
        a = np.asarray(a, np.float64)
        if a.ndim == len(trailing) + 3 and a.shape[3:] == tuple(trailing):
            return a[:, :, :, None, ...]                                    # one population
        if a.ndim == len(trailing) + 4 and a.shape[4:] == tuple(trailing):
            return a
        raise ValueError(f"{name} must be a volume of shape grid + {tuple(trailing)} (one population per voxel) or "
                         f"grid + (K,) + {tuple(trailing)} (K populations); got {a.shape}")


class ODFField(_OrientationField):
    """Orientation distributions over the grid, in a **named** basis: ``basis`` and ``legacy`` are passed to
    :meth:`~dmipy_sim.replay.fod.FOD.from_sh`, which converts to the orthonormal basis RPH.md 4.1 requires.

    The conversion is not cosmetic. The MRtrix / legacy-tournier basis differs by a scale on the ``m != 0``
    coefficients and breaks the addition theorem the composition goes through, and it does so by an amount that
    vanishes exactly when the gradient is parallel to B0 -- the one geometry a cursory check would test. Hence
    no default: a producer that cannot say which convention its volume is in cannot declare one.
    """

    mode = "odf_sh"

    def __init__(self, coeffs, *, basis, legacy=False, weights=None, normalize=True):
        self.coeffs = self._volume(coeffs, (np.asarray(coeffs).shape[-1],), "odf coefficients")
        self.basis, self.legacy, self.normalize = basis, legacy, normalize
        self.weights = None if weights is None else np.asarray(weights, np.float64)
        self.lmax = _lmax_of_n_coeffs(self.coeffs.shape[-1])

    def at(self, ijk):
        from .fod import FOD
        c = self.coeffs[ijk]                                                # (K, n_c)
        w = np.ones(c.shape[0]) / c.shape[0] if self.weights is None else self.weights[ijk]
        out = []
        for k in range(c.shape[0]):
            if w[k] <= 0.0 or not np.any(c[k]):
                continue                                    # an all-zero block is "no orientation here", not a density
            f = FOD.from_sh(c[k], basis=self.basis, legacy=self.legacy, normalize=self.normalize)
            out.append((float(w[k]), f.coeffs))
        return out


class PeakField(_OrientationField):
    """Discrete orientations over the grid (RPH.md 4): ``directions`` of shape grid + ``(3,)``, or
    grid + ``(K, 3)`` with ``weights`` for a crossing -- one slot per population, each with its own fraction."""

    mode = "peaks"

    def __init__(self, directions, *, weights=None):
        self.directions = self._volume(directions, (3,), "peak directions")
        self.weights = None if weights is None else np.asarray(weights, np.float64)
        self.lmax = 0

    def at(self, ijk):
        d = self.directions[ijk]                                            # (K, 3)
        w = np.ones(d.shape[0]) / d.shape[0] if self.weights is None else self.weights[ijk]
        out = []
        for k in range(d.shape[0]):
            n = np.linalg.norm(d[k])
            if w[k] <= 0.0 or n == 0.0:
                continue
            out.append((float(w[k]), d[k] / n))
        return out


class FrameField(_OrientationField):
    """A whole rotation per voxel (RPH.md 4, frames mode): ``rotations`` of shape grid + ``(3, 3)``, or
    grid + ``(K, 3, 3)`` with ``weights`` for several populations.

    What a substrate whose response is not axially symmetric needs in order to be placed unambiguously, and what
    an operation on the magnetisation vector needs: a pulse is applied to a pose, not to an axis.
    """

    mode = "frames"

    def __init__(self, rotations, *, weights=None):
        self.rotations = self._volume(rotations, (3, 3), "frame rotations")
        self.weights = None if weights is None else np.asarray(weights, np.float64)
        self.lmax = 0

    def at(self, ijk):
        R = self.rotations[ijk]                                             # (K, 3, 3)
        w = np.ones(R.shape[0]) / R.shape[0] if self.weights is None else self.weights[ijk]
        out = []
        for k in range(R.shape[0]):
            if w[k] <= 0.0:
                continue
            if not np.allclose(R[k] @ R[k].T, np.eye(3), atol=1e-5) or np.linalg.det(R[k]) < 0:
                raise ValueError(f"the frame at {ijk} population {k} is not a proper rotation")
            out.append((float(w[k]), R[k]))
        return out


class BinghamField(FrameField):
    """A fan per voxel (RPH.md 4, bingham mode): a frame, and a concentration about each of its first two axes.

    ``kappa`` is a volume of ``(k1, k2)`` (or a pair, for a uniform field): larger is tighter, equal values are
    the Watson cone of that width, and unequal ones are a population spread in one plane and narrow in the
    other. ``roll_kappa`` ties the substrate's own azimuth to the frame; zero leaves it free.
    """

    mode = "bingham"

    def __init__(self, rotations, kappa, *, roll_kappa=0.0, weights=None):
        super().__init__(rotations, weights=weights)
        shape = self.rotations.shape[:4]                                    # grid + (K,)
        self.kappa = _with_population_axis(kappa, shape, (2,), "bingham kappa")
        self.roll_kappa = _with_population_axis(roll_kappa, shape, (), "roll_kappa")

    def at(self, ijk):
        return [(w, (R, tuple(self.kappa[ijk][k]), float(self.roll_kappa[ijk][k])))
                for k, (w, R) in enumerate(super().at(ijk))]


class WatsonField(ODFField):
    """A Watson distribution per voxel: ``kappa`` (a scalar or a volume) about ``mu``, a direction volume. The
    coefficients are generated in the required basis, so there is no convention to declare."""

    def __init__(self, kappa, mu, *, lmax=8, weights=None):
        from .fod import FOD
        mu = self._volume(mu, (3,), "watson mu")
        kap = np.broadcast_to(np.asarray(kappa, np.float64), mu.shape[:4])
        c = np.zeros(mu.shape[:4] + (n_sh_coeffs(lmax),))
        for ijk in np.ndindex(mu.shape[:4]):
            n = np.linalg.norm(mu[ijk])
            if n == 0.0:
                continue
            c[ijk] = FOD.watson(float(kap[ijk]), mu=mu[ijk] / n, lmax=lmax).coeffs
        super().__init__(c, basis="tournier07", legacy=False, weights=weights, normalize=False)
        self.coeffs = c                                                     # already grid + (K, n_c)


def _lmax_of_n_coeffs(n_c):
    for l in range(0, 33, 2):
        if n_sh_coeffs(l) == n_c:
            return l
    raise ValueError(f"{n_c} coefficients is not an even-order real SH block")


# ------------------------------------------------------------------ the constructor
def build_rph(path, *, grid, substrates, occupancy, orientation, remainder=None, scalars=None,
              id, license, citation, embed=True, extra_meta=None):
    """Build a ``.rph`` from volumes: **the** way a phantom is made (RPH.md, issue #151).

    * ``grid`` -- a :class:`Grid`, or its keyword dict.
    * ``substrates`` -- the declarations of :func:`pack_substrate`, :func:`analytic_substrate`,
      :func:`inert_substrate`, in the order the file cites them.
    * ``occupancy`` -- either a mapping ``{substrate id or index: volume of volume fractions}``, or an integer
      **label** volume (one substrate per voxel, ``-1`` for none).
    * ``remainder`` -- the substrate that takes ``1 - sum`` where the given fractions leave a gap. A voxel is
      always full, so without it a short row is an error rather than a normalised guess.
    * ``orientation`` -- an :class:`ODFField`, :class:`PeakField` or :class:`WatsonField`, or a mapping
      ``{substrate id: field}`` when pack substrates are oriented differently. A field with ``K`` populations
      splits that substrate's slot into ``K``, each carrying its share of the fraction (RPH.md 4).
    * ``scalars`` -- optional ``{name: volume}`` macroscopic layers, names from :data:`SCALAR_REGISTRY`.

    Voxels with no signal-bearing content are dropped: the file is sparse, so an anatomy costs what it
    occupies. Returns the written metadata.
    """
    g = grid if isinstance(grid, Grid) else Grid(**grid)
    subs = [dict(s) for s in substrates]
    for s in subs:
        if s.get("kind") not in SUBSTRATE_KINDS:
            raise ValueError(f"substrate {s.get('id')!r} has kind {s.get('kind')!r}, not one of {SUBSTRATE_KINDS}")
    ids = [s["id"] for s in subs]
    if len(set(ids)) != len(ids):
        raise ValueError(f"substrate ids must be unique: {ids}")

    def sid_of(key):
        if isinstance(key, (int, np.integer)):
            return int(key)
        if key not in ids:
            raise ValueError(f"occupancy names substrate {key!r}, which is not declared: {ids}")
        return ids.index(key)

    F = _fraction_volumes(g, subs, occupancy, remainder, sid_of)
    fields = _orientation_fields(orientation, subs, ids)
    mode = {f.mode for f in fields.values()}
    if len(mode) != 1:
        raise ValueError(f"a phantom declares exactly one orientation mode (RPH.md 4); got {sorted(mode)}")
    mode = mode.pop()
    lmax = max((f.lmax for f in fields.values()), default=0)
    n_c = n_sh_coeffs(lmax)

    bearing = np.array([s["kind"] != "inert" for s in subs])
    live = (F[bearing] > 0).any(axis=0)
    idx = np.argwhere(live)
    if idx.size == 0:
        raise ValueError("no voxel carries a signal-bearing substrate: the occupancy volumes are empty")

    rows = []
    for ijk in map(tuple, idx):
        slots = []
        for i, s in enumerate(subs):
            f = float(F[(i,) + ijk])
            if f <= 0.0:
                continue
            fld = fields.get(i)
            if fld is None:
                slots.append((i, f, None))
                continue
            pops = fld.at(ijk)
            if not pops:
                raise ValueError(f"voxel {ijk} has {f:.3g} of substrate {ids[i]!r} but no orientation there; an "
                                 f"oriented substrate needs a pose in every voxel it occupies")
            wsum = sum(w for w, _ in pops)
            for w, payload in pops:
                slots.append((i, f * w / wsum, payload))
        rows.append((ijk, slots))
    P = max(len(sl) for _, sl in rows)

    N = len(rows)
    voxel_index = np.zeros((N, 3), np.int32)
    substrate_id = np.full((N, P), -1, np.int16)
    frac = np.zeros((N, P), np.float32)
    # one array per mode, since each states a different amount of the pose (RPH.md 4)
    width = {"odf_sh": n_c, "peaks": 3, "frames": 4, "bingham": 4}[mode]
    ori = np.zeros((N, P, width), np.float32)
    kap = np.zeros((N, P, 2), np.float32) if mode == "bingham" else None
    rollk = np.zeros((N, P), np.float32) if mode == "bingham" else None
    for v, (ijk, slots) in enumerate(rows):
        voxel_index[v] = ijk
        for p, (i, f, payload) in enumerate(slots):
            substrate_id[v, p], frac[v, p] = i, f
            if payload is None:
                if mode == "peaks":
                    ori[v, p] = (0.0, 0.0, 1.0)                # unoriented: an isotropic substrate has no axis
                elif mode in ("frames", "bingham"):
                    ori[v, p] = (0.0, 0.0, 0.0, 1.0)           # the identity rotation
                continue
            if mode == "bingham":
                R, (k1, k2), rk = payload
                ori[v, p] = _quat_of(R)
                kap[v, p], rollk[v, p] = (k1, k2), rk
            elif mode == "frames":
                ori[v, p] = _quat_of(payload)
            else:
                ori[v, p, :len(payload)] = payload

    sc, names = None, ()
    if scalars:
        names = tuple(scalars)
        bad = [n for n in names if n not in SCALAR_REGISTRY]
        if bad:
            raise ValueError(f"unknown macroscopic layer(s) {bad}: the registry is {list(SCALAR_REGISTRY)} "
                             f"(RPH.md 5.1). A replayer refuses a layer it does not know rather than dropping it, "
                             f"so a phantom may not declare one either.")
        sc = np.stack([_as_volume(scalars[n], g, f"scalar {n!r}")[tuple(voxel_index.T)] for n in names], axis=1).astype(np.float32)

    embed_packs = {}
    for i, s in enumerate(subs):
        pk = s.pop("_pack", None)
        if s["kind"] == "pack":
            if pk is None:
                raise ValueError(f"pack substrate {s['id']!r} was declared without a pack")
            if embed:
                embed_packs[i] = pk
            else:
                if not isinstance(pk, (str, Path)):
                    raise ValueError(f"substrate {s['id']!r} is referenced rather than embedded, which needs a "
                                     f"path to the .rpk, not an in-memory pack")
                s["uri"] = str(pk)
    return write_rph(path, voxel_index=voxel_index, substrate_id=substrate_id, geometric_fraction=frac,
                     substrates=subs, grid=g, lmax=lmax, scalars=sc, scalar_names=names,
                     id=id, license=license, citation=citation, embed_packs=embed_packs,
                     extra_meta=extra_meta, bingham_kappa=kap, roll_kappa=rollk,
                     **{{"odf_sh": "odf_sh", "peaks": "peak_dir", "frames": "pose_quat",
                         "bingham": "pose_quat"}[mode]: ori})


def _with_population_axis(a, shape, trailing, name):
    """A per-voxel value broadcast over the population axis, whether or not the caller wrote one."""
    a = np.asarray(a, np.float64)
    for cand in (shape + trailing, shape[:3] + trailing):
        try:
            out = np.broadcast_to(a, cand)
        except ValueError:
            continue
        return out if cand == shape + trailing else np.broadcast_to(out[:, :, :, None, ...], shape + trailing)
    raise ValueError(f"{name} has shape {a.shape}, which is neither grid + {trailing} nor grid + (K,) + "
                     f"{trailing} for a {shape[3]}-population field")


def _quat_of(R):
    """``(x, y, z, w)`` of a proper rotation, sign fixed by ``w >= 0`` so a pose has one spelling."""
    from scipy.spatial.transform import Rotation
    q = Rotation.from_matrix(np.asarray(R, np.float64).reshape(3, 3)).as_quat()
    return q if q[3] >= 0 else -q


def _as_volume(a, g, name):
    a = np.asarray(a, np.float64)
    if a.shape != tuple(g.shape):
        raise ValueError(f"{name} has shape {a.shape}, not the grid's {tuple(g.shape)}")
    return a


def _fraction_volumes(g, subs, occupancy, remainder, sid_of):
    """``(n_sub,) + grid`` volume fractions, with the remainder placed and every row checked to sum to one."""
    F = np.zeros((len(subs),) + tuple(g.shape))
    if isinstance(occupancy, dict):
        for key, vol in occupancy.items():
            v = _as_volume(vol, g, f"occupancy of {key!r}")
            if v.min() < 0.0:
                raise ValueError(f"occupancy of {key!r} has negative fractions")
            F[sid_of(key)] += v
    else:
        lab = np.asarray(occupancy)
        if lab.shape != tuple(g.shape) or not np.issubdtype(lab.dtype, np.integer):
            raise ValueError(f"a label volume is an integer volume of the grid's shape {tuple(g.shape)}; got "
                             f"{lab.shape} of {lab.dtype}. A volume of fractions goes in a mapping instead.")
        for i in range(len(subs)):
            F[i] = lab == i
    tot = F.sum(axis=0)
    if tot.max() > 1.0 + 1e-4:
        raise ValueError(f"the occupancy of {int((tot > 1.0 + 1e-4).sum())} voxel(s) sums to more than one "
                         f"(max {tot.max():.4g}): fractions are shares of the voxel volume")
    short = tot < 1.0 - 1e-4
    if short.any():
        if remainder is None:
            raise ValueError(
                f"{int(short.sum())} voxel(s) have occupancy summing to less than one (min {tot.min():.4g}) and no "
                f"remainder= was given. A voxel is always full: what is not tissue is a substrate -- inert "
                f"background, or free water -- not slack in the sum, because composing a short row returns a "
                f"signal that is quietly too low (RPH.md 3).")
        i = sid_of(remainder)
        F[i] = F[i] + np.where(short | (tot < 1.0), 1.0 - tot, 0.0)
    return F


def _orientation_fields(orientation, subs, ids):
    """Which substrate each orientation field applies to: every pack substrate by default, or per id."""
    oriented = [i for i, s in enumerate(subs) if s["kind"] == "pack"]
    if isinstance(orientation, dict):
        out = {}
        for key, f in orientation.items():
            i = ids.index(key) if not isinstance(key, (int, np.integer)) else int(key)
            if subs[i]["kind"] != "pack":
                raise ValueError(f"substrate {ids[i]!r} is {subs[i]['kind']!r} and has no orientation: an analytic "
                                 f"or inert substrate is orientation-independent")
            out[i] = f
        missing = [ids[i] for i in oriented if i not in out]
        if missing:
            raise ValueError(f"no orientation given for pack substrate(s) {missing}")
        return out
    if len(oriented) > 1 and getattr(orientation, "weights", None) is not None:
        raise ValueError(f"one orientation field with populations cannot serve {len(oriented)} pack substrates "
                         f"unambiguously: pass a mapping {{substrate id: field}}")
    return {i: orientation for i in oriented}


# ------------------------------------------------------------------ writing
def write_rph(path, *, voxel_index, substrate_id, geometric_fraction, substrates, grid, id, license, citation,
              odf_sh=None, peak_dir=None, pose_quat=None, bingham_kappa=None, roll_kappa=None, lmax=None,
              scalars=None, scalar_names=(), embed_packs=None, extra_meta=None):
    """Write a ``.rph``. Prefer :func:`build_rph`, which derives every array from volumes; this is the writer
    it calls and the level to reach for only when the sparse arrays already exist.

    ``embed_packs`` maps a substrate index to a ``.rpk`` path or an in-memory pack, whose arrays are copied in
    under ``substrate{i}/`` -- making the phantom a standalone artifact, with nothing to resolve and nothing to
    go missing. Substrates not embedded carry a ``uri``.

    Fractions MUST sum to one per voxel (RPH.md 3), which is checked rather than trusted.
    """
    from safetensors.numpy import save_file

    g = grid if isinstance(grid, Grid) else Grid(**grid)
    gf = np.asarray(geometric_fraction, np.float32)
    bad = np.abs(gf.sum(axis=1) - 1.0) > 1e-4
    if bad.any():
        raise ValueError(
            f"{int(bad.sum())} voxel(s) have geometric fractions summing to "
            f"{gf.sum(axis=1)[bad][:3]} rather than 1. A voxel is always full; give the "
            f"remainder to an 'inert' substrate rather than leaving it unmodelled.")
    for s in substrates:
        if s.get("kind") not in SUBSTRATE_KINDS:
            raise ValueError(f"substrate kind {s.get('kind')!r} not in {SUBSTRATE_KINDS}")
    given = [k for k, v in (("odf_sh", odf_sh), ("peak_dir", peak_dir), ("pose_quat", pose_quat)) if v is not None]
    if len(given) != 1:
        raise ValueError(f"give exactly one of odf_sh=, peak_dir= or pose_quat=: a phantom declares one "
                         f"orientation mode (RPH.md 4), and they differ in how much of the pose they pin down "
                         f"(got {given})")

    tensors = {"voxel_index": np.asarray(voxel_index, np.int32),
               "substrate_id": np.asarray(substrate_id, np.int16),
               "geometric_fraction": gf}
    if odf_sh is not None:
        tensors["odf_sh"] = np.asarray(odf_sh, np.float32)
        ori_meta = {"mode": "odf_sh", "lmax": int(lmax if lmax is not None else _lmax_of_n_coeffs(tensors["odf_sh"].shape[-1])),
                    "basis": "real", "convention": "orthonormal"}
    elif peak_dir is not None:
        tensors["peak_dir"] = np.asarray(peak_dir, np.float32)
        ori_meta = {"mode": "peaks", "max_peaks": int(tensors["peak_dir"].shape[1])}
    else:
        tensors["pose_quat"] = np.asarray(pose_quat, np.float32)
        ori_meta = {"mode": "frames", "max_peaks": int(tensors["pose_quat"].shape[1])}
        if bingham_kappa is not None:
            tensors["bingham_kappa"] = np.asarray(bingham_kappa, np.float32)
            ori_meta["mode"] = "bingham"
            if roll_kappa is not None:
                tensors["roll_kappa"] = np.asarray(roll_kappa, np.float32)
    if scalars is not None:
        names = list(scalar_names)
        sc = np.asarray(scalars, np.float32)
        if sc.ndim != 2 or sc.shape[1] != len(names) or sc.shape[0] != tensors["voxel_index"].shape[0]:
            raise ValueError(f"scalars must be (n_voxels, n_names) = ({tensors['voxel_index'].shape[0]}, "
                             f"{len(names)}); got {sc.shape}")
        bad = [n for n in names if n not in SCALAR_REGISTRY]
        if bad:
            raise ValueError(f"unknown macroscopic layer(s) {bad}; the registry is {list(SCALAR_REGISTRY)} (RPH.md 5.1)")
        tensors["scalars"] = sc

    subs = [dict(s) for s in substrates]
    for i, rpk in (embed_packs or {}).items():
        from .replay import read_rpk
        import hashlib
        if isinstance(rpk, (str, Path)):
            pk = read_rpk(rpk)
            subs[i]["sha256"] = hashlib.sha256(open(rpk, "rb").read()).hexdigest()
        else:
            pk = rpk
            subs[i]["sha256"] = hashlib.sha256(
                b"".join(np.ascontiguousarray(v).tobytes() for _, v in sorted(pk.arrays.items()))).hexdigest()
        for k, v in pk.arrays.items():
            tensors[f"substrate{i}/{k}"] = np.ascontiguousarray(v)
        subs[i]["embedded"] = True
        subs[i]["pack_meta"] = pk.meta

    meta = {"rph_schema_version": RPH_SCHEMA_VERSION, "id": id, "grid": g.to_meta(),
            "orientation": ori_meta, "substrates": subs, "license": license, "citation": citation}
    if scalars is not None:
        meta["scalars"] = list(scalar_names)
    meta.update(extra_meta or {})
    save_file(tensors, str(path), metadata={"rph": json.dumps(meta)})
    return meta


# ------------------------------------------------------------------ reading
def read_rph(path):
    """Read a ``.rph`` into a :class:`ReplayPhantom`."""
    from safetensors import safe_open
    with safe_open(str(path), framework="numpy") as f:
        meta = json.loads(f.metadata()["rph"])
        arrays = {k: f.get_tensor(k) for k in f.keys()}
    return ReplayPhantom(arrays, meta, source=str(path))


class ReplayPhantom:
    """A voxel grid citing solved substrates. Owns no walkers."""

    def __init__(self, arrays, meta, source=None):
        self.arrays, self.meta, self.source = arrays, meta, source

    # ---- structure
    @property
    def substrates(self):
        return self.meta["substrates"]

    @property
    def grid(self):
        """The voxel grid placed in the scanner (:class:`Grid`)."""
        return Grid.from_meta(self.meta)

    @property
    def mode(self):
        """``"peaks"`` or ``"odf_sh"``: how the phantom states its orientations (RPH.md 4)."""
        return self.meta["orientation"]["mode"]

    @property
    def peak_dir(self):
        return self.arrays["peak_dir"]

    @property
    def pose_quat(self):
        return self.arrays["pose_quat"]

    @property
    def bingham_kappa(self):
        return self.arrays["bingham_kappa"]

    @property
    def roll_kappa(self):
        return self.arrays.get("roll_kappa")

    @property
    def m0(self):
        return np.array([float(s["m0"]) for s in self.substrates])

    @property
    def scalar_names(self):
        """The macroscopic layers this phantom declares (RPH.md 5.1), in the columns of :attr:`scalars`."""
        return tuple(self.meta.get("scalars", ()))

    @property
    def scalars(self):
        return self.arrays.get("scalars")

    def scalar(self, name):
        """One declared layer as ``(n_voxels,)``; raises for a layer the phantom does not carry."""
        names = self.scalar_names
        if name not in names:
            raise ValueError(f"the phantom declares no {name!r} layer; it carries {list(names)}")
        return np.asarray(self.arrays["scalars"])[:, names.index(name)]

    def fraction(self, substrate):
        """Per-voxel total volume fraction of one substrate, ``(n_voxels,)``. Slots are packed, so a column of
        :attr:`geometric_fraction` is not a fixed substrate; this sums the slots that cite it."""
        i = substrate if isinstance(substrate, (int, np.integer)) else self.index_of(substrate)
        return (self.geometric_fraction * (self.substrate_id == i)).sum(axis=1)

    def index_of(self, substrate_id):
        for i, s in enumerate(self.substrates):
            if s.get("id") == substrate_id:
                return i
        raise ValueError(f"no substrate {substrate_id!r}; the phantom cites "
                         f"{[s.get('id') for s in self.substrates]}")

    @property
    def lmax(self):
        return int(self.meta["orientation"].get("lmax", 0))

    @property
    def voxel_index(self):
        return self.arrays["voxel_index"]

    @property
    def substrate_id(self):
        return self.arrays["substrate_id"]

    @property
    def geometric_fraction(self):
        return self.arrays["geometric_fraction"]

    @property
    def odf_sh(self):
        return self.arrays["odf_sh"]

    @property
    def n_voxels(self):
        return int(self.arrays["voxel_index"].shape[0])

    def __repr__(self):
        kinds = ", ".join(f"{s['kind']}:{s.get('id','?')}" for s in self.substrates)
        layers = f", layers={list(self.scalar_names)}" if self.scalar_names else ""
        return (f"ReplayPhantom(id={self.meta.get('id')!r}, voxels={self.n_voxels}, "
                f"grid={list(self.grid.shape)}, {self.mode}, substrates=[{kinds}]{layers})")

    def is_embedded(self, i):
        return bool(self.substrates[i].get("embedded"))

    def pack(self, i):
        """The embedded pack for substrate ``i`` as a :class:`~dmipy_sim.replay.replay.ReplayPack`.

        Raises for a substrate that is not an embedded pack rather than silently returning
        something else -- a phantom citing a pack by ``uri`` needs that file resolved, and an
        analytic or inert substrate has no walkers at all.
        """
        from .replay import ReplayPack
        s = self.substrates[i]
        if s.get("kind") != "pack":
            raise ValueError(f"substrate {i} is {s.get('kind')!r}, not a pack")
        if not s.get("embedded"):
            raise ValueError(
                f"substrate {i} ({s.get('id')}) is referenced by uri {s.get('uri')!r}, not "
                f"embedded; read that .rpk and pass it explicitly")
        pre = f"substrate{i}/"
        arrays = {k[len(pre):]: v for k, v in self.arrays.items() if k.startswith(pre)}
        if not arrays:
            raise ValueError(f"substrate {i} is declared embedded but carries no arrays")
        return ReplayPack(arrays, s["pack_meta"])

    # ---- capability
    def tiers(self, i=None):
        """Which replay tiers the phantom can serve: the INTERSECTION over its packs.

        A phantom adds no capability of its own, so a channel missing from any cited pack is
        missing from the phantom (RPH.md). Analytic and inert substrates are closed forms and
        do not constrain the intersection.
        """
        idxs = [i] if i is not None else [j for j, s in enumerate(self.substrates)
                                          if s.get("kind") == "pack"]
        out = None
        for j in idxs:
            env = (self.substrates[j].get("pack_meta") or {}).get("replay_envelope", {})
            have = {k for k, v in env.items() if v is True}
            out = have if out is None else (out & have)
        return out or set()

    def require(self, *tiers):
        """Raise unless every named tier is available across the cited packs."""
        have = self.tiers()
        missing = [t for t in tiers if t not in have]
        if missing:
            raise ValueError(
                f"phantom cannot serve {missing}: its packs declare {sorted(have)}. A replayer "
                f"refuses a tier a pack does not carry rather than returning the signal "
                f"without it.")

    # ---- replay
    def replay(self, waveform, *, B0=None, b0_dir=(0.0, 0.0, 1.0), tissue="nominal", packs=None,
               n_check=256, complex_signal=False, T2=None, T1=None, rho=None, D=None,
               chi_iso=None, chi_aniso=0.0):
        """Replay the whole phantom: ``(voxel_index, S)`` with ``S`` of shape ``(n_voxels, n_measurements)``.

        Each cited pack is replayed **once** into its response over poses (:meth:`ReplayPack.pose_response`),
        and every voxel is then an inner product of those SO(3) coefficients with its own orientation
        distribution. That is the whole economy of a replay phantom: the expensive object is the walk, and it is
        shared by every voxel and every pose that cites it.

        The acquisition's gradient and B0 directions are in the scanner frame of ``grid.frame``; the substrates
        rotate under it. Knobs given here apply to every pack; a substrate's own ``tissue`` (RPH.md 3.2) wins
        over them, and anything neither names takes the pack's nominal value. ``packs`` supplies the packs of
        substrates cited by ``uri`` as ``{id or index: path or ReplayPack}``.

        No band is passed. Each pack projects its response at the band that response needs (its phase
        amplitude, :meth:`ReplayPack.pose_response`), and the composition retains only what this phantom's
        orientations can reach: an ODF of order ``L`` reaches ``l <= L`` and, saying nothing about the
        substrate's own azimuth, only ``n = 0``. At order 8 that is 45 coefficients per measurement rather than
        the 969 an unrestricted expansion carries.

        Declared macroscopic layers (RPH.md 5.1) are applied per voxel, and one that this acquisition cannot
        carry raises rather than being dropped.
        """
        knobs = dict(T2=T2, T1=T1, rho=rho, D=D, chi_iso=chi_iso, chi_aniso=chi_aniso)
        pose, analytic, m0, ref = self._responses(waveform, B0, b0_dir, tissue, packs, n_check, knobs,
                                                  keep=self.retained_band())
        sid, frac = self.substrate_id, self.geometric_fraction
        n_meas = next(iter(pose.values())).n_meas if pose else len(np.atleast_1d(next(iter(analytic.values()))))
        S = np.zeros((self.n_voxels, n_meas), np.complex128)
        keep_l, keep_n = self._resolve_band(pose)
        vp, F = self.slot_coefficients(keep_l, keep_n)
        ids = sid[vp[:, 0], vp[:, 1]].astype(int)
        weight = frac[vp[:, 0], vp[:, 1]].astype(np.float64) * m0[vp[:, 0], ids]
        for i in set(ids.tolist()):
            m = ids == i
            if i in analytic:                                          # a closed form has no pose
                np.add.at(S, vp[m, 0], weight[m][:, None] * np.atleast_1d(analytic[i])[None, :])
            elif i in pose:                                            # one product for every slot citing it
                np.add.at(S, vp[m, 0], weight[m][:, None] * (F[m] @ pose[i].retained(keep_l, keep_n).T))
        S = self._apply_layers(S, waveform, ref)
        return self.voxel_index, (S if complex_signal else np.abs(S))

    def slot_coefficients(self, lmax, nmax):
        """Every slot's orientation distribution as SO(3) coefficients: ``(n_live, n_features)``, with the
        ``(voxel, slot)`` index of each row.

        One batched build per mode rather than a quadrature per voxel, which is what makes a phantom of many
        voxels cost a matrix product (RPH.md 4):

        * **peaks** -- a direction with its azimuth unstated, mapped through the cached axis map;
        * **odf_sh** -- an axis density in the required basis, the same map applied to its coefficients;
        * **frames** -- a rotation, so the coefficients are the basis evaluated there;
        * **bingham** -- a canonical fan per distinct concentration pair, rotated into each slot's frame.
        """
        from .fod import FOD
        from . import so3
        sid, frac = self.substrate_id, self.geometric_fraction
        rows = [(v, p) for v in range(self.n_voxels) for p in range(sid.shape[1])
                if sid[v, p] >= 0 and frac[v, p] > 0.0]
        vp = np.array(rows, np.int64).reshape(-1, 2)
        n = vp.shape[0]
        F = np.zeros((n, so3.n_so3_coeffs(lmax, nmax)))
        # only a pack has a pose to compose: an analytic substrate is a closed form and an inert one emits
        # nothing, so their slots keep the zero row rather than being read as an orientation
        posed = np.array([self.substrates[int(sid[v, p])]["kind"] == "pack" for v, p in vp], bool)
        if not posed.any():
            return vp, F
        idx = vp[posed]
        mode = self.mode
        if mode in ("peaks", "odf_sh"):
            T = so3._axis_map(int(lmax), int(nmax))
            if mode == "peaks":
                d = self.peak_dir[idx[:, 0], idx[:, 1]].astype(np.float64)
                d = d / np.maximum(np.linalg.norm(d, axis=1, keepdims=True), 1e-30)
                sh = so3.real_sh(lmax, d, full=True)
            else:
                sh = np.stack([so3._embed_sh(FOD.native(self.odf_sh[v, p].astype(np.float64)).coeffs, lmax)
                               for v, p in idx])
            F[posed] = sh @ T.T
        elif mode in ("frames", "bingham"):
            R = so3.rotations_from_quaternions(self.pose_quat[idx[:, 0], idx[:, 1]])
            if mode == "frames":
                F[posed] = so3.so3_design(lmax, R, nmax)
            else:
                kap = self.bingham_kappa[idx[:, 0], idx[:, 1]].astype(np.float64)
                rk = self.roll_kappa
                rk = np.zeros(idx.shape[0]) if rk is None else rk[idx[:, 0], idx[:, 1]].astype(np.float64)
                key = np.stack([kap[:, 0], kap[:, 1], rk], axis=1)
                out = np.zeros((idx.shape[0], F.shape[1]))
                for u in np.unique(key, axis=0):                       # one canonical fan per distinct pair
                    m = (key == u).all(axis=1)
                    can = so3.bingham_coeffs(np.eye(3), (u[0], u[1]), lmax, nmax, roll_kappa=u[2])
                    out[m] = so3.rotate_coeffs(can, R[m], lmax, nmax)
                F[posed] = out
        else:
            raise ValueError(f"unknown orientation mode {mode!r}")
        return vp, F

    def _resolve_band(self, pose):
        """The band to retain, once the packs have said what they projected at.

        What this phantom's orientations can reach, capped by what its packs carry. The cap is safe rather than
        lossy: a pack's projection band was chosen so that its response is reproduced to within the pack's own
        Monte-Carlo floor, so the response has no content above it to multiply, and an orientation distribution
        stated at a higher order contributes nothing there.
        """
        want_l, want_n = self.retained_band()
        if not pose:
            return 0, 0
        have_l = min(p.lmax for p in pose.values())
        have_n = min(p.nmax for p in pose.values())
        return (have_l if want_l is None else min(want_l, have_l)), \
               (have_n if want_n is None else min(want_n, have_n))

    def retained_band(self):
        """The band a composition of this phantom's orientations can reach: ``(lmax, nmax)``.

        Everything above it is annihilated by the inner product, so retaining it would be arithmetic on numbers
        that cannot matter (RPH.md 4). An ODF states an order and no azimuth; a peak states a direction and no
        azimuth, so it is not band-limited in ``l`` and takes whatever the response was projected at; a frame
        states a whole rotation and reaches everything; a Bingham with a free azimuth keeps ``n = 0``.
        """
        mode = self.mode
        if mode == "odf_sh":
            return int(self.lmax), 0
        if mode == "peaks":
            return None, 0                                  # l: whatever the response carries; n: only 0
        if mode == "bingham":
            return None, (None if self.roll_kappa is not None and np.any(self.roll_kappa) else 0)
        return None, None                                   # frames: a point mass reaches every coefficient

    def _responses(self, waveform, B0, b0_dir, tissue, packs, n_check, knobs, keep=None):
        """One response per substrate: a :class:`PoseResponse` for a pack, a closed form for an analytic
        substrate, nothing for an inert one. Plus the per-voxel ``m0`` with the ``m0_scale`` layer applied."""
        from .replay import ReplayPack, read_rpk
        given = {}
        for key, pk in (packs or {}).items():
            given[key if isinstance(key, (int, np.integer)) else self.index_of(key)] = pk
        pose, analytic, ref = {}, {}, None
        for i, sub in enumerate(self.substrates):
            if sub["kind"] == "inert":
                continue
            if sub["kind"] == "analytic":
                model = ANALYTIC_MODELS.get(sub.get("model"))
                if model is None:
                    raise ValueError(f"substrate {sub['id']!r} names the closed form {sub.get('model')!r}, which this "
                                     f"replayer does not implement; it knows {sorted(ANALYTIC_MODELS)}")
                analytic[i] = model(sub.get("params", {}), _b_values(waveform))
                continue
            pk = given.get(i)
            if pk is None:
                pk = self.pack(i)                                   # raises for a uri substrate: say which file
            elif not isinstance(pk, ReplayPack):
                pk = read_rpk(pk)
            kw = dict(knobs)
            kw.update({k: v for k, v in (sub.get("tissue") or {}).items()})
            ref = pk
            pose[i] = pk.pose_response(waveform, tissue=tissue, B0=B0, b0_dir=b0_dir, n_check=n_check,
                                       keep=keep, **kw)
        if not pose and not analytic:
            raise ValueError("the phantom cites no signal-bearing substrate")
        m0 = np.broadcast_to(self.m0, (self.n_voxels, len(self.substrates))).astype(np.float64)
        if "m0_scale" in self.scalar_names:
            m0 = m0 * self.scalar("m0_scale")[:, None]
        return pose, analytic, m0, ref

    def _apply_layers(self, S, waveform, ref):
        """The macroscopic layers of RPH.md 5.1, each on the voxel's whole signal."""
        from ._replay_kernel import se_gate
        from ..constants import GAMMA
        for name in self.scalar_names:
            if name == "m0_scale":
                continue                                            # already in m0
            if name == "delta_B0_T":
                if ref is None:
                    raise ValueError("the delta_B0_T layer needs a pack's save grid to integrate the coherence gate "
                                     "over; the phantom cites no pack")
                n_t, dt = ref.n_t, ref.dt
                t_ref = waveform.rf.refocus_time                     # the sequence's own 180, or none
                gate = float(dt * se_gate(n_t, dt, t_ref).sum())     # zero for a 180 at TE/2: the layer refocuses
                S = S * np.exp(1j * GAMMA * self.scalar("delta_B0_T")[:, None] * gate)
            elif name == "kappa_B1":
                raise ValueError(
                    "this phantom declares a kappa_B1 layer, which scales every RF flip angle and so needs the "
                    "RF-aware (vector-Bloch) replay of each pack at the voxel's pose. A magnitude gradient replay "
                    "cannot carry it, and dropping it would return a signal that looks right and is not. Use "
                    "ReplayPhantom.replay_bloch, which propagates the magnetisation per pose.")
            else:
                raise ValueError(f"the phantom declares the layer {name!r}, which this replayer does not apply; "
                                 f"a layer silently dropped is a phantom that replays wrong (RPH.md 5.1)")
        return S

    def replay_bloch(self, waveform, *, B0=None, b0_dir=(0.0, 0.0, 1.0), tissue="nominal",
                     packs=None, complex_signal=False, T2=None, T1=None, rho=None, D=None, chi_iso=None,
                     chi_aniso=0.0, decimals=3):
        """Replay the phantom through the RF-aware route: ``(voxel_index, S)``, one magnetisation propagation
        per distinct pose rather than one contraction per voxel.

        This is what a layer acting on the magnetisation vector needs. ``kappa_B1`` scales every flip angle of
        the acquisition, and a flip angle is not something the pose expansion of :meth:`replay` carries: that
        route reads the signal as a phase sum over an ideal-pulse echo, so an RF scale has nowhere to enter.
        Here each pack is propagated at the voxel's own pose and transmit scale
        (:meth:`ReplayPack.replay_bloch`), and analytic substrates take the same RF train on a static spin
        times their closed form.

        **Frames mode only.** A propagation happens at a pose, so the phantom has to state one: a distribution
        of poses under a scaled RF pulse is not the composition of one propagation, and a mode that leaves the
        substrate's azimuth unstated leaves the propagation undefined rather than merely dispersed. Both are
        refused instead of approximated. Distinct ``(substrate, rotation, scale)`` triples are propagated once
        each and reused, ``decimals`` setting how finely they are distinguished; the cost is that count, not the
        voxel count.
        """
        from .replay import ReplayPack, read_rpk
        if self.mode != "frames":
            raise ValueError(f"replay_bloch propagates the magnetisation at a pose, so it needs a frames-mode "
                             f"phantom, which states one rotation per slot (RPH.md 4); this one is {self.mode!r}. "
                             f"A mode that leaves the substrate's azimuth unstated leaves the propagation "
                             f"undefined, and a distribution of poses under a scaled RF pulse is not the "
                             f"composition of one propagation, so neither is approximated here.")
        rf = waveform.rf
        if not rf:
            raise ValueError("the Bloch route replays an RF schedule and this sequence carries none")
        given = {}
        for key, pk in (packs or {}).items():
            given[key if isinstance(key, (int, np.integer)) else self.index_of(key)] = pk
        loaded = {}
        for i, sub in enumerate(self.substrates):
            if sub["kind"] != "pack":
                continue
            pk = given.get(i)
            loaded[i] = pk if isinstance(pk, ReplayPack) else (read_rpk(pk) if pk is not None else self.pack(i))
        kappa = self.scalar("kappa_B1") if "kappa_B1" in self.scalar_names else np.ones(self.n_voxels)
        knobs = dict(T2=T2, T1=T1, rho=rho, D=D, chi_iso=chi_iso, chi_aniso=chi_aniso)
        b = _b_values(waveform)
        m0 = np.broadcast_to(self.m0, (self.n_voxels, len(self.substrates))).astype(np.float64)
        if "m0_scale" in self.scalar_names:
            m0 = m0 * self.scalar("m0_scale")[:, None]
        from .so3 import rotations_from_quaternions
        sid, frac = self.substrate_id, self.geometric_fraction
        poses = rotations_from_quaternions(self.pose_quat.reshape(-1, 4)).reshape(self.pose_quat.shape[:2] + (3, 3))
        S, cache = None, {}
        for v in range(self.n_voxels):
            for p, i in enumerate(sid[v]):
                i, f = int(i), float(frac[v, p])
                if i < 0 or f == 0.0 or self.substrates[i]["kind"] == "inert":
                    continue
                kap = round(float(kappa[v]), int(decimals))
                key = (i, kap, tuple(np.round(poses[v, p].reshape(-1), int(decimals))))
                resp = cache.get(key)
                if resp is None:
                    sub = self.substrates[i]
                    if sub["kind"] == "analytic":
                        model = ANALYTIC_MODELS.get(sub.get("model"))
                        if model is None:
                            raise ValueError(f"substrate {sub['id']!r} names the closed form {sub.get('model')!r}, "
                                             f"which this replayer does not implement")
                        resp = model(sub.get("params", {}), b) * _static_spin_rf(waveform, kap)
                    else:
                        kw = dict(knobs)
                        kw.update({k: val for k, val in (sub.get("tissue") or {}).items()})
                        resp = loaded[i].replay_bloch(waveform, b1_scale=kap, tissue=tissue, B0=B0,
                                                      b0_dir=b0_dir, orientation=poses[v, p], complex_signal=True,
                                                      **kw)
                    cache[key] = resp
                resp = np.atleast_1d(resp)
                if S is None:
                    S = np.zeros((self.n_voxels, resp.shape[0]), np.complex128)
                S[v] += f * m0[v, i] * resp
        if S is None:
            raise ValueError("the phantom cites no signal-bearing substrate")
        return self.voxel_index, (S if complex_signal else np.abs(S))

    def to_volume(self, values, fill=np.nan):
        """Scatter per-voxel values back onto the dense grid: ``(nx, ny, nz) + values.shape[1:]``, with ``fill``
        where the sparse phantom has no voxel."""
        v = np.asarray(values)
        out = np.full(tuple(self.grid.shape) + v.shape[1:], fill, dtype=np.result_type(v.dtype, type(fill)))
        out[tuple(self.voxel_index.T)] = v
        return out


def _b_values(waveform):
    """The b-values an analytic closed form is evaluated at: the sequence's declared encoding, else the integral
    of its effective gradient (``b = int |q|^2 dt``, ``q = gamma int G_eff``)."""
    enc = waveform.encoding
    return np.asarray(enc.bvalues if enc is not None else waveform.b(), np.float64)


def _static_spin_rf(waveform, b1_scale):
    """The transverse magnetisation a single static spin at the origin keeps through this sequence's RF schedule
    at this transmit scale: what multiplies an analytic substrate's closed form, since a closed form has diffusion
    attenuation but no magnetisation of its own. Zero gradient, so the schedule alone acts."""
    from .trajectories import replay_bloch
    n_t, dt = waveform.n_t, float(waveform.dt)
    out = replay_bloch(np.zeros((1, n_t, 3)), dt, np.zeros((1, n_t, 3)), dt, waveform.rf, b1_scale=float(b1_scale))
    return complex(np.atleast_1d(np.asarray(out).ravel())[-1])
