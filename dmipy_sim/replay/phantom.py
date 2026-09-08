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

from .gaunt import n_sh_coeffs

__all__ = ["ReplayPhantom", "read_rph", "write_rph", "build_rph", "Grid", "ANALYTIC_MODELS", "pack_substrate",
           "analytic_substrate", "inert_substrate", "ODFField", "PeakField", "WatsonField",
           "SUBSTRATE_KINDS", "SCALAR_REGISTRY"]

SUBSTRATE_KINDS = ("pack", "analytic", "inert")
RPH_SCHEMA_VERSION = "0.3.0"

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
    from .sh_convolution import free_water_response
    return np.array([free_water_response(b, params["diffusivity"]) for b in np.atleast_1d(b_values)])


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
    ori = np.zeros((N, P, n_c if mode == "odf_sh" else 3), np.float32)
    for v, (ijk, slots) in enumerate(rows):
        voxel_index[v] = ijk
        for p, (i, f, payload) in enumerate(slots):
            substrate_id[v, p], frac[v, p] = i, f
            if payload is not None:
                ori[v, p, :len(payload)] = payload
            elif mode == "peaks":
                ori[v, p] = (0.0, 0.0, 1.0)                    # unoriented: an isotropic substrate has no axis

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
                     extra_meta=extra_meta, **{("odf_sh" if mode == "odf_sh" else "peak_dir"): ori})


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
              odf_sh=None, peak_dir=None, lmax=None, scalars=None, scalar_names=(), embed_packs=None,
              extra_meta=None):
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
    if (odf_sh is None) == (peak_dir is None):
        raise ValueError("give exactly one of odf_sh= or peak_dir=: a phantom declares one orientation mode "
                         "(RPH.md 4), and the two are the same physics read two ways")

    tensors = {"voxel_index": np.asarray(voxel_index, np.int32),
               "substrate_id": np.asarray(substrate_id, np.int16),
               "geometric_fraction": gf}
    if odf_sh is not None:
        tensors["odf_sh"] = np.asarray(odf_sh, np.float32)
        ori_meta = {"mode": "odf_sh", "lmax": int(lmax if lmax is not None else _lmax_of_n_coeffs(tensors["odf_sh"].shape[-1])),
                    "basis": "real", "convention": "orthonormal"}
    else:
        tensors["peak_dir"] = np.asarray(peak_dir, np.float32)
        ori_meta = {"mode": "peaks", "max_peaks": int(tensors["peak_dir"].shape[1])}
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
               l_g=8, l_b=6, n_theta=32, n_phi=64, complex_signal=False, T2=None, T1=None, rho=None, D=None,
               chi_iso=None, chi_aniso=0.0, refocus_time="auto"):
        """Replay the whole phantom: ``(voxel_index, S)`` with ``S`` of shape ``(n_voxels, n_measurements)``.

        Each cited pack is replayed **once** into its response over poses (:meth:`ReplayPack.pose_spectra`), and
        every voxel is then a contraction of those spectra against its own orientation -- an ODF through the
        Gaunt route, a peak read at its direction. That is the whole economy of a replay phantom: the expensive
        object is the walk, and it is shared by every voxel and every pose that cites it.

        The acquisition's gradient and B0 directions are in the scanner frame of ``grid.frame``; the substrates
        rotate under it. Knobs given here apply to every pack; a substrate's own ``tissue`` (RPH.md 3.2) wins
        over them, and anything neither names takes the pack's nominal value. ``packs`` supplies the packs of
        substrates cited by ``uri`` as ``{id or index: path or ReplayPack}``.

        ``n_theta`` / ``n_phi`` set the quadrature the pose response is expanded on: the default resolves the
        ``l_g``, ``l_b`` truncation comfortably, and a coarser one is cheaper for a sweep of many frames.

        Declared macroscopic layers (RPH.md 5.1) are applied per voxel, and one that this acquisition cannot
        carry raises rather than being dropped.
        """
        from .fod import FOD
        pose, analytic, m0, ref = self._responses(waveform, B0, b0_dir, tissue, packs, l_g, l_b, n_theta, n_phi,
                                            dict(T2=T2, T1=T1, rho=rho, D=D, chi_iso=chi_iso, chi_aniso=chi_aniso,
                                                 refocus_time=refocus_time))
        sid, frac = self.substrate_id, self.geometric_fraction
        n_meas = next(iter(pose.values())).n_meas if pose else len(np.atleast_1d(next(iter(analytic.values()))))
        S = np.zeros((self.n_voxels, n_meas), np.complex128)
        peaks = self.mode == "peaks"
        ori = self.peak_dir if peaks else self.odf_sh
        cache = {}
        for v in range(self.n_voxels):
            for p, i in enumerate(sid[v]):
                i = int(i)
                f = float(frac[v, p])
                if i < 0 or f == 0.0:
                    continue
                if i in analytic:
                    resp = analytic[i]
                elif i in pose:
                    key = (i, ori[v, p].tobytes())
                    resp = cache.get(key)
                    if resp is None:
                        resp = cache[key] = (pose[i].at(ori[v, p]) if peaks
                                             else pose[i].compose(FOD.native(ori[v, p].astype(np.float64))))
                else:
                    continue                                        # inert: occupies the volume, emits nothing
                S[v] += f * m0[v, i] * resp
        S = self._apply_layers(S, waveform, refocus_time, ref)
        return self.voxel_index, (S if complex_signal else np.abs(S))

    def _responses(self, waveform, B0, b0_dir, tissue, packs, l_g, l_b, n_theta, n_phi, knobs):
        """One response per substrate: a :class:`PoseSpectra` for a pack, a closed form for an analytic
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
            pose[i] = pk.pose_spectra(waveform, tissue=tissue, B0=B0, b0_dir=b0_dir, l_g=l_g, l_b=l_b,
                                      n_theta=n_theta, n_phi=n_phi, **kw)
        if not pose and not analytic:
            raise ValueError("the phantom cites no signal-bearing substrate")
        m0 = np.broadcast_to(self.m0, (self.n_voxels, len(self.substrates))).astype(np.float64)
        if "m0_scale" in self.scalar_names:
            m0 = m0 * self.scalar("m0_scale")[:, None]
        return pose, analytic, m0, ref

    def _apply_layers(self, S, waveform, refocus_time, ref):
        """The macroscopic layers of RPH.md 5.1, each on the voxel's whole signal."""
        from ._replay_kernel import se_gate
        from .replay import _refocus_time_of
        from ..constants import GAMMA
        for name in self.scalar_names:
            if name == "m0_scale":
                continue                                            # already in m0
            if name == "delta_B0_T":
                if ref is None:
                    raise ValueError("the delta_B0_T layer needs a pack's save grid to integrate the coherence gate "
                                     "over; the phantom cites no pack")
                n_t, dt = ref.n_t, ref.dt
                t_ref = _refocus_time_of(waveform) if refocus_time == "auto" else refocus_time
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

    def replay_bloch(self, waveform, *, rf_events=None, B0=None, b0_dir=(0.0, 0.0, 1.0), tissue="nominal",
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

        **Peaks mode only.** A pose here is one rotation of the substrate, and a distribution of poses under a
        scaled RF pulse is not the pose-averaged response of one propagation, so an ODF phantom is refused
        rather than approximated. Distinct ``(substrate, direction, scale)`` triples are propagated once each
        and reused, ``decimals`` setting how finely they are distinguished; the cost is that count, not the
        voxel count.
        """
        from .replay import ReplayPack, read_rpk
        if self.mode != "peaks":
            raise ValueError("replay_bloch composes one rotation per slot, so it needs a peaks-mode phantom "
                             "(RPH.md 4). Under a scaled RF pulse a distribution of poses is not the average of "
                             "one propagation, so an ODF phantom would be an approximation with no error bound.")
        rf = rf_events if rf_events is not None else (getattr(waveform, "rf_events", None) or [])
        if not rf:
            raise ValueError("the Bloch route replays an RF schedule: give rf_events= or a waveform carrying them")
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
        sid, frac, peak = self.substrate_id, self.geometric_fraction, self.peak_dir
        S, cache = None, {}
        for v in range(self.n_voxels):
            for p, i in enumerate(sid[v]):
                i, f = int(i), float(frac[v, p])
                if i < 0 or f == 0.0 or self.substrates[i]["kind"] == "inert":
                    continue
                kap = round(float(kappa[v]), int(decimals))
                key = (i, kap, tuple(np.round(peak[v, p], int(decimals))))
                resp = cache.get(key)
                if resp is None:
                    sub = self.substrates[i]
                    if sub["kind"] == "analytic":
                        model = ANALYTIC_MODELS.get(sub.get("model"))
                        if model is None:
                            raise ValueError(f"substrate {sub['id']!r} names the closed form {sub.get('model')!r}, "
                                             f"which this replayer does not implement")
                        resp = model(sub.get("params", {}), b) * _static_spin_rf(waveform, rf, kap)
                    else:
                        kw = dict(knobs)
                        kw.update({k: val for k, val in (sub.get("tissue") or {}).items()})
                        resp = loaded[i].replay_bloch(waveform, rf_events=rf, b1_scale=kap, tissue=tissue, B0=B0,
                                                      b0_dir=b0_dir, orientation=np.asarray(peak[v, p], float),
                                                      complex_signal=True, **kw)
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
    """The b-values of the waveform, from its own declaration or from the gradient itself: ``b = int |q|^2 dt``
    with ``q = gamma int G``, which is what an analytic closed form is evaluated at."""
    from ..constants import GAMMA
    b = getattr(waveform, "bvalues", None)
    if b is not None:
        return np.asarray(b, np.float64)
    G = np.asarray(getattr(waveform, "G", waveform), np.float64)
    if G.ndim == 2:
        G = G[None]
    dt = float(getattr(waveform, "dt"))
    q = GAMMA * np.cumsum(G, axis=1) * dt
    return (q * q).sum(axis=2).sum(axis=1) * dt


def _static_spin_rf(waveform, rf_events, b1_scale):
    """The transverse magnetisation a single static spin at the origin keeps through this RF schedule at this
    transmit scale: what multiplies an analytic substrate's closed form, since a closed form has diffusion
    attenuation but no magnetisation of its own. Zero gradient, so the schedule alone acts."""
    from .trajectories import replay_bloch
    n_t = int(np.asarray(getattr(waveform, "G", waveform)).shape[-2])
    dt = float(getattr(waveform, "dt"))
    out = replay_bloch(np.zeros((1, n_t, 3)), dt, np.zeros((1, n_t, 3)), dt, rf_events, b1_scale=float(b1_scale))
    return complex(np.atleast_1d(np.asarray(out).ravel())[-1])
