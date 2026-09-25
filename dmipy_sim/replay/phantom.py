"""The ``.rph`` file: reader, writer, and the reference replay of a phantom read from disk (RPH.md).

A phantom owns no walkers. It cites substrates per voxel with a geometric fraction each and an orientation, so
one solved pack serves every voxel and every orientation that cites it. The **construction** of a phantom from
volumes lives in :mod:`dmipy_sim.phantom` (:meth:`~dmipy_sim.phantom.Phantom.compose`); this module is the
format: :func:`write_rph`, :func:`read_rph`, and :class:`ReplayPhantom`, the arrays-and-metadata object the
file decodes to, with the replay operation of RPH.md 6 on it.

The point of the replay side is that the physics is *separable*: the exponent of the signal is a sum of
independent channel terms (gradient, field, occupancy-weighted relaxation, surface contact), so a caller adds
a physics by layering a term rather than by re-simulating. Each keyword of :meth:`ReplayPhantom.replay` turns
on exactly one channel, and a pack that does not carry the channel refuses rather than silently returning the
signal without it.
"""
from __future__ import annotations

import json
import logging
import time
from pathlib import Path

import numpy as np

from ..phantom.grid import Grid
from ..constants import GAMMA, GAMMA_BAR
from ..phantom.substrates import substrate_from_meta, _echo_time
from .so3 import n_sh_coeffs, lmax_of
from ..run import Run, current

log = logging.getLogger(__name__)


__all__ = ["ReplayPhantom", "read_rph", "write_rph", "Grid", "SUBSTRATE_KINDS", "SCALAR_REGISTRY", "RPH_SCHEMA_VERSION"]

SUBSTRATE_KINDS = ("pack", "analytic", "inert")
RPH_SCHEMA_VERSION = "0.4.0"

#: The macroscopic layers a phantom may declare per voxel (RPH.md 5.1). A name outside this registry is
#: refused rather than ignored: a layer silently dropped is a phantom that replays wrong while looking right.
SCALAR_REGISTRY = ("kappa_B1", "delta_B0_T", "m0_scale")


def quantise(values, tolerance):
    """``values`` binned to multiples of ``tolerance``, or untouched for ``None``: what decides how many distinct
    propagations a phantom costs.

    A route that propagates once per distinct value of something -- a transmit scale, a field offset, a pose
    -- pays one propagation per distinct value, and a SMOOTH map, which is what a machine's own profile is,
    has as many distinct values as voxels. Binning to a tolerance bounds that count, and the error it
    introduces is of the tolerance's own order: a flip angle reaches the signal through a sine, an offset
    through a phase linear in it. ``None`` bins nothing, so every distinct value is its own class.
    """
    v = np.asarray(values, np.float64)
    if tolerance is None:
        return v
    tol = float(tolerance)
    if not tol > 0:
        raise ValueError(f"a tolerance is a positive width to bin to, or None for no binning; got {tolerance}")
    return np.round(v / tol) * tol


def _gather(S, vp, F, weight, which, coeff):
    """Every slot's contribution into its voxel, in place: ``S[v] += weight * F[slot] . coeff[which[slot]]``.

    The one contraction a replay phantom is -- a slot's orientation features ``F`` against the SO(3)
    coefficients of the response it cites -- shared by every route that expands over poses. A slot whose
    ``which`` is negative cites nothing here and is skipped.
    """
    live = which >= 0
    if np.any(live):
        part = np.einsum("sf,smf->sm", np.asarray(F[live], np.complex128), coeff[which[live]])
        np.add.at(S, vp[live, 0], weight[live][:, None] * part)
    return S


def _gather_jax(S, vp, F, weight, which, coeff):
    """:func:`_gather` on an accelerator: a gather of each slot's coefficients, one contraction over the
    orientation features, then a segment sum into voxels. Complex is carried as a real pair because a segment
    sum over complex is not uniformly supported."""
    import jax, jax.numpy as jnp
    live = which >= 0
    Fj = jnp.asarray(np.ascontiguousarray(F[live]), jnp.float32)
    Cr = jnp.asarray(np.ascontiguousarray(coeff.real), jnp.float32)
    Ci = jnp.asarray(np.ascontiguousarray(coeff.imag), jnp.float32)
    wj = jnp.asarray(np.ascontiguousarray(weight[live]), jnp.float32)
    idx = jnp.asarray(np.ascontiguousarray(which[live]), jnp.int32)
    vox = jnp.asarray(np.ascontiguousarray(vp[live, 0]), jnp.int32)
    n_voxels = S.shape[0]

    @jax.jit
    def go(Fj, Cr, Ci, wj, idx, vox):
        gr, gi = Cr[idx], Ci[idx]                                  # (n_slots, n_meas, n_feat)
        re = jnp.einsum("sf,smf->sm", Fj, gr) * wj[:, None]
        im = jnp.einsum("sf,smf->sm", Fj, gi) * wj[:, None]
        return (jax.ops.segment_sum(re, vox, num_segments=n_voxels),
                jax.ops.segment_sum(im, vox, num_segments=n_voxels))

    re, im = go(Fj, Cr, Ci, wj, idx, vox)
    S += np.asarray(re, np.float64) + 1j * np.asarray(im, np.float64)
    return S



# ------------------------------------------------------------------ writing
def write_rph(path, *, voxel_index, substrate_id, geometric_fraction, substrates, grid, id, license, citation,
              odf_sh=None, peak_dir=None, pose_quat=None, bingham_kappa=None, roll_kappa=None, lmax=None,
              scalars=None, scalar_names=(), embed_packs=None, extra_meta=None):
    """Write a ``.rph`` from its sparse arrays. :meth:`dmipy_sim.phantom.Phantom.compose` derives them from
    volumes and :meth:`~dmipy_sim.phantom.Phantom.write` calls this; reach for it directly only when the arrays
    already exist.

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
        ori_meta = {"mode": "odf_sh", "lmax": int(lmax if lmax is not None else lmax_of(tensors["odf_sh"].shape[-1])),
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
        embed_pack(subs, tensors, i, rpk)

    meta = {"rph_schema_version": RPH_SCHEMA_VERSION, "id": id, "grid": g.to_meta(),
            "orientation": ori_meta, "substrates": subs, "license": license, "citation": citation}
    if scalars is not None:
        meta["scalars"] = list(scalar_names)
    meta.update(extra_meta or {})
    tensors = {k: np.ascontiguousarray(v) for k, v in tensors.items()}      # safetensors writes a buffer as it lies
    save_file(tensors, str(path), metadata={"rph": json.dumps(meta)})
    return meta


def embed_pack(subs, tensors, i, rpk):
    """Substrate ``i`` carried inside the phantom file: the pack's arrays under ``substrate{i}/``, its meta on the
    row, and its identity -- the file's sha256 for a pack given by path, :attr:`ReplayPack.digest` for one in memory."""
    from .replay import read_rpk
    if isinstance(rpk, (str, Path)):
        from ..fill.hub import sha256_of
        pk, subs[i]["sha256"] = read_rpk(rpk), sha256_of(rpk)
    else:
        pk, subs[i]["sha256"] = rpk, rpk.digest
    for k, v in pk.arrays.items():
        tensors[f"substrate{i}/{k}"] = np.ascontiguousarray(v)
    subs[i]["embedded"] = True
    subs[i]["pack_meta"] = pk.meta


def scatter_volume(shape, voxel_index, values, fill=np.nan):
    """Per-voxel values back on the dense grid, ``shape + values.shape[1:]``, ``fill`` where there is no voxel."""
    v = np.asarray(values)
    out = np.full(tuple(shape) + v.shape[1:], fill, dtype=np.result_type(v.dtype, type(fill)))
    out[tuple(np.asarray(voxel_index).T)] = v
    return out


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
        """How the phantom states its orientations (RPH.md 4): ``"peaks"``, ``"odf_sh"``, ``"frames"`` or
        ``"bingham"``."""
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
    def _concomitant_phase(self, waveform, scanner, echo=None):
        """The order-0 concomitant phase of every voxel, ``(n_voxels, n_meas)`` at one readout (the last, or
        ``echo``), for a catalogued machine with a field strength; ``None`` otherwise
        (:func:`~dmipy_sim.phantom.bore.concomitant_phase_map`, dmipy-sim#394)."""
        from ..acquisition.scanners import ScannerLimits
        from ..phantom.bore import concomitant_phase_map
        if not isinstance(scanner, ScannerLimits):
            return None
        ph = concomitant_phase_map(scanner, self.grid, waveform, voxels=self.voxel_index)
        if ph is None:
            return None
        return ph[:, :, -1 if echo is None else int(echo)]

    def _encoding_classes(self, waveform, scanner, tolerance, report=None):
        """The acquisition as the machine plays it per voxel, binned (:func:`~dmipy_sim.phantom.bore.encoding_classes`),
        or ``None`` for a scanner that brings no gradient-side term -- a bare field strength, or a machine with
        no catalogued shape, tensor or field."""
        from ..acquisition.scanners import ScannerLimits
        from ..phantom.bore import encoding_classes
        out = None
        if isinstance(scanner, ScannerLimits):
            out = encoding_classes(scanner, self.grid, waveform, self.voxel_index, tolerance=tolerance)
        if report is not None:
            report.update(n_encoding_classes=1 if out is None else len(out[1]))
        return out

    def replay(self, waveform, *, scanner=None, pose=None, packs=None, complex_signal=False,
               off_resonance=None, proton_density=None, cache=None, forms=None, encoding_tolerance=1e-3,
               report=None):
        """Replay the whole phantom through the pose expansion: ``(voxel_index, S)`` with ``S`` of shape
        ``(n_voxels, n_measurements)``.

        Each cited pack is replayed **once** into its response over poses (:meth:`ReplayPack.pose_response`),
        and every voxel is then an inner product of those SO(3) coefficients with its own orientation
        distribution. That is the whole economy of a replay phantom: the expensive object is the walk, and it is
        shared by every voxel and every pose that cites it.

        The acquisition's gradient and the field are in the scanner frame of the grid; the substrates rotate
        under it, and ``pose`` (a 3x3 rotation or a :class:`~dmipy_sim.phantom.partition.Pose`) is the specimen's
        rigid rotation in the bore. Each substrate replays at its own declared ``tissue`` (RPH.md 3.2; a pack
        substrate with none is the bare diffusion signal) and at the one ``scanner`` field (a
        :class:`~dmipy_sim.acquisition.scanners.ScannerLimits` or tesla, ``None`` for no field). ``packs``
        supplies the packs of substrates cited by ``uri`` as ``{id or index: path or ReplayPack}``.

        No band is passed: each pack projects its response at the band that response needs, and the composition
        retains only what this phantom's orientations can reach.

        The macroscopic layers (RPH.md 5.1) come from the file **and** from this call, per voxel ``(n_voxels,)``:
        ``off_resonance`` (T) adds to a ``delta_B0_T`` layer and ``proton_density`` multiplies an ``m0_scale``
        layer. A ``kappa_B1`` layer cannot be carried here -- it acts on the magnetisation, not on a phase sum --
        and is refused with the route that can (:meth:`replay_bloch`), never dropped.

        **The machine's gradient side** (dmipy-sim#377). A :class:`~dmipy_sim.acquisition.scanners.ScannerLimits`
        that catalogues a gradient-nonlinearity tensor, a field shape or a field strength delivers a different
        gradient at every voxel -- the tensor's tilt and scale, the magnet's own background, the coils' Maxwell
        term -- and each pack is replayed once per distinct delivered gradient rather than once. The voxels are
        binned to ``encoding_tolerance``, a fraction of ``b`` (:func:`~dmipy_sim.phantom.bore.encoding_classes`;
        ``None`` for exact, one class per distinct voxel), and ``report`` receives ``n_encoding_classes``, which
        is the cost.
        """
        if "kappa_B1" in self.scalar_names:
            raise ValueError(
                "this phantom declares a kappa_B1 layer, which scales every RF flip angle and so needs the "
                "RF-aware (vector-Bloch) replay of each pack at the voxel's pose. A magnitude gradient replay "
                "cannot carry it, and dropping it would return a signal that looks right and is not. Use "
                "ReplayPhantom.replay_bloch, which propagates the magnetisation per pose.")
        with Run("phantom.replay", params=dict(n_voxels=int(self.n_voxels), n_meas=int(waveform.n_meas), encoding_tolerance=encoding_tolerance,
                                         scanner=(scanner if scanner is None or isinstance(scanner, (int, float)) else type(scanner).__name__))) as run:
            classes = self._encoding_classes(waveform, scanner, encoding_tolerance, report)
            waveforms = [waveform] if classes is None else classes[1]
            cls_of_voxel = np.zeros(self.n_voxels, int) if classes is None else classes[0]
            run.phase("classes", n_classes=len(waveforms))
            pose, analytic, m0 = self._responses(waveforms, scanner, pose, packs, keep=self.retained_band(),
                                                 proton_density=proton_density, cache=cache, forms=forms, report=report)
            run.phase("gather")
            sid, frac = self.substrate_id, self.geometric_fraction
            n_meas = next(iter(pose.values())).n_meas if pose else len(np.atleast_1d(next(iter(analytic.values()))))
            S = np.zeros((self.n_voxels, n_meas), np.complex128)
            keep_l, keep_n = self._resolve_band(pose)
            vp, F = self.slot_coefficients(keep_l, keep_n)
            ids = sid[vp[:, 0], vp[:, 1]].astype(int)
            cv = cls_of_voxel[vp[:, 0]]
            weight = frac[vp[:, 0], vp[:, 1]].astype(np.float64) * m0[vp[:, 0], ids]
            for (i, c), resp in analytic.items():                          # a closed form has no pose
                m = (ids == i) & (cv == c)
                np.add.at(S, vp[m, 0], weight[m][:, None] * np.atleast_1d(resp)[None, :])
            if pose:                                                       # one product for every slot citing a pack
                order = sorted(pose)
                coeff = np.stack([np.asarray(pose[k].retained(keep_l, keep_n), np.complex128) for k in order])
                lookup = {k: j for j, k in enumerate(order)}
                which = np.array([lookup.get((int(i), int(c)), -1) for i, c in zip(ids, cv)])
                _gather(S, vp, F, weight, which, coeff)
            run.phase("layers")
            dB0 = self.layer_values("delta_B0_T", off_resonance)
            if dB0 is not None:
                S = S * np.exp(1j * GAMMA * dB0[:, None] * self.gate_integral(waveform))
            ph = self._concomitant_phase(waveform, scanner)
            if ph is not None:
                S = S * np.exp(1j * ph)
            if report is not None:
                report["seconds"] = run.phase_seconds()
            return self.voxel_index, (S if complex_signal else np.abs(S))

    def layer_values(self, name, extra=None, combine="add"):
        """One macroscopic layer per voxel, ``(n_voxels,)``: the file's column (if declared) combined with a
        value given at replay time (``extra``: a scalar or ``(n_voxels,)``), added for a field offset and
        multiplied for a scale. ``None`` when neither is present."""
        have = self.scalar(name) if name in self.scalar_names else None
        if extra is None:
            return None if have is None else np.asarray(have, np.float64)
        e = np.asarray(extra, np.float64)
        e = np.broadcast_to(e, (self.n_voxels,)) if e.ndim == 0 else e.reshape(-1)
        if e.shape != (self.n_voxels,):
            raise ValueError(f"{name} at replay time must be a scalar or one value per voxel ({self.n_voxels}); got {e.shape}")
        if have is None:
            return e.copy()
        return (have + e) if combine == "add" else (have * e)

    @staticmethod
    def gate_integral(waveform):
        """``int s(t) dt`` of the acquisition's coherence gate over **its own** grid (s): what a uniform
        off-resonance dephases through. Zero for a 180 at TE/2 (the layer refocuses), TE for a gradient echo."""
        from ._replay_kernel import se_gate
        n_t, dt = int(waveform.n_t), float(waveform.dt)
        return float(dt * se_gate(n_t, dt, waveform.rf.refocus_time if waveform.rf else None).sum())

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
        sid, frac = np.asarray(self.substrate_id), np.asarray(self.geometric_fraction)
        vp = np.argwhere((sid >= 0) & (frac > 0.0)).astype(np.int64)           # (n_live, 2): every live slot
        n = vp.shape[0]
        F = np.zeros((n, so3.n_so3_coeffs(lmax, nmax)))
        # only a pack has a pose to compose: an analytic substrate is a closed form and an inert one emits
        # nothing, so their slots keep the zero row rather than being read as an orientation
        has_pose = np.array([s_["kind"] == "pack" or (s_["kind"] == "analytic" and bool(s_.get("oriented", False)))
                             for s_ in self.substrates], bool)                # a pack, or a closed form with an axis
        posed = has_pose[sid[vp[:, 0], vp[:, 1]]]
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
                from .fod import _C00
                c = self.odf_sh[idx[:, 0], idx[:, 1]].astype(np.float64)            # (n_posed, n_c), compact even
                if np.any(np.abs(c[:, 0] - _C00) > 1e-6 * _C00):
                    raise ValueError("an ODF slot is not a unit-integral density in the required basis")
                l_odf = lmax_of(c.shape[1])
                sh = np.zeros((c.shape[0], so3.n_sh_coeffs(int(lmax), full=True)))
                for l in range(0, min(l_odf, int(lmax)) + 1, 2):                    # compact even -> full layout
                    sh[:, so3.sh_block(l, True)] = c[:, so3.sh_block(l, False)]
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

    def _loaded_packs(self, packs):
        """``{substrate index: ReplayPack}`` for every pack substrate: given, or embedded, or refused by uri."""
        from .replay import ReplayPack, read_rpk
        given = {}
        for key, pk in (packs or {}).items():
            given[key if isinstance(key, (int, np.integer)) else self.index_of(key)] = pk
        out = {}
        for i, sub in enumerate(self.substrates):
            if sub["kind"] != "pack":
                continue
            pk = given.get(i)
            if pk is None:
                pk = self.pack(i)                                   # raises for a uri substrate: say which file
            elif not isinstance(pk, ReplayPack):
                pk = read_rpk(pk)
            out[i] = pk
        return out

    def _m0(self, proton_density=None):
        """Per voxel and substrate, with the ``m0_scale`` layer and a replay-time proton density applied."""
        m0 = np.broadcast_to(self.m0, (self.n_voxels, len(self.substrates))).astype(np.float64)
        pd = self.layer_values("m0_scale", proton_density, combine="mul")
        return m0 if pd is None else m0 * pd[:, None]

    @staticmethod
    def _form(i, sub, forms):
        """The closed form of substrate ``i``: the live object when the caller holds one (``forms``, an in-memory
        phantom's own declarations), else the one its ``model`` names, read back from the file's record."""
        if forms and i in forms:
            return forms[i]
        return substrate_from_meta(sub)

    def _check_relaxation(self, waveform, loaded, forms):
        """Refuse a phantom whose substrates would relax inconsistently under a readout (dmipy-sim#238).

        Every acquisition reads out at an echo time, so a substrate whose tissue declares no T2 / T1 replays as if
        T2 were infinite; composed beside a substrate that does relax, it returns a tissue contrast that looks
        right and is not (the brain example's grey matter at its proton density against a white matter at 0.06).
        A phantom in which no substrate relaxes is a consistent diffusion phantom and is not refused."""
        from ..spec.tissue import Tissue
        relaxes, missing = [], []
        for i, sub in enumerate(self.substrates):
            if sub["kind"] == "inert":
                continue
            t = self._form(i, sub, forms).tissue if sub["kind"] == "analytic" else Tissue.from_meta(sub.get("tissue"))
            (relaxes if (t is not None and t.relaxes) else missing).append(sub["id"])
        if relaxes and missing:
            TE = _echo_time(waveform)
            raise ValueError(
                f"under a readout at TE = {TE * 1e3:.3g} ms, "
                + ", ".join(f"substrate {n!r} relaxes" for n in relaxes) + " while "
                + ", ".join(f"{n!r} would not" for n in missing)
                + ": its tissue declares no T2, so it would replay as if T2 were infinite and the composed contrast "
                  "would look right and be wrong. Declare it on the substrate (PackSubstrate(..., tissue=Tissue(T2=...)) "
                  "by pool, or pack.nominal; FreeWater(tissue=Tissue(D=..., T2=...))), or declare none anywhere for "
                  "a phantom with no relaxation (RPH.md 3.2).")

    def _responses(self, waveforms, scanner, specimen, packs, keep=None, proton_density=None, cache=None, forms=None,
                   report=None):
        """One response per (substrate, encoding class): a :class:`PoseResponse` for a pack, a closed form for
        an analytic substrate, nothing for an inert one -- keyed ``(i, c)`` with ``c`` indexing ``waveforms``,
        the acquisition as played in each class. Plus the per-voxel ``m0``. ``specimen`` is the specimen's
        rotation in the bore: a pack's expansion runs in that frame, and an analytic form sees the acquisition
        turned into it."""
        from ..spec.tissue import Tissue
        from ..acquisition.waveforms import rotate_waveform
        from .replay import _pose_matrix
        pose, analytic = {}, {}
        loaded = self._loaded_packs(packs)
        self._check_relaxation(waveforms[0], loaded, forms)
        R_s = _pose_matrix(specimen)
        run, t_start, rows = current(), time.time(), []
        for c, waveform in enumerate(waveforms):
            t_c = time.time()
            turned = waveform if R_s is None else rotate_waveform(waveform, R_s.T)    # G @ R_s: the acquisition in the specimen frame
            for i, sub in enumerate(self.substrates):
                if sub["kind"] == "inert":
                    continue
                if sub["kind"] == "analytic":
                    form = self._form(i, sub, forms)                               # refuses an unknown closed form
                    if sub.get("oriented", False) or getattr(form, "oriented", False):   # a form with an axis: expanded over
                        from .replay import analytic_pose_response               # SO(3) like a pack, then contracted
                        pose[(i, c)] = analytic_pose_response(form, turned, keep)
                    else:
                        analytic[(i, c)] = form.response(turned)
                    continue
                pose[(i, c)] = resp = loaded[i].pose_response(waveform, tissue=Tissue.from_meta(sub.get("tissue")),
                                                              scanner=scanner, pose=R_s, keep=keep, cache=cache)
                rows.append(dict(cls=c, substrate=i, route=resp.route, lmax=resp.lmax, nmax=resp.nmax,
                                 field_lmax=int(resp.field_lmax), n_bodies=resp.n_bodies, seconds=time.time() - t_c))
            if len(waveforms) > 1:                                                # the classes of a machine pass
                per = (time.time() - t_start) / (c + 1)
                log.info("phantom.replay: class %d/%d in %.0f s (%.0f s/class, ETA %.0f min)", c + 1, len(waveforms),
                         time.time() - t_c, per, per * (len(waveforms) - c - 1) / 60.0)
            if run is not None:
                run.progress(c + 1, len(waveforms), unit="classes")
        if not pose and not analytic:
            raise ValueError("the phantom cites no signal-bearing substrate")
        if report is not None:
            report["responses"] = rows
        return pose, analytic, self._m0(proton_density)

    def replay_bloch(self, waveform, *, scanner=None, pose=None, packs=None, complex_signal=False,
                     transmit=None, off_resonance=None, proton_density=None, transmit_tolerance=1e-3,
                     pose_tolerance=1e-3, off_resonance_tolerance=1e-4, forms=None, encoding_tolerance=1e-3,
                     report=None):
        """Replay the phantom through the RF-aware route: ``(voxel_index, S)``, one magnetisation propagation
        per distinct pose rather than one contraction per voxel.

        This is what a layer acting on the magnetisation vector needs. ``kappa_B1`` scales every flip angle of
        the acquisition, and a flip angle is not something the pose expansion of :meth:`replay` carries: that
        route reads the signal as a phase sum over an ideal-pulse echo, so an RF scale has nowhere to enter.
        Here each pack is propagated at the voxel's own pose, transmit scale and off-resonance
        (:meth:`ReplayPack.replay_bloch`), and analytic substrates take the same RF train on a static spin
        times their closed form.

        The layers come from the file and from the call: ``transmit`` multiplies a ``kappa_B1`` layer,
        ``off_resonance`` (T) adds to ``delta_B0_T``, ``proton_density`` multiplies ``m0_scale``; each a scalar
        or ``(n_voxels,)``.

        **Frames mode only.** A propagation happens at a pose, so the phantom has to state one: a distribution
        of poses under a scaled RF pulse is not the composition of one propagation, and a mode that leaves the
        substrate's azimuth unstated leaves the propagation undefined rather than merely dispersed. Both are
        refused instead of approximated. Distinct ``(substrate, rotation, transmit, off-resonance)`` tuples are
        propagated once each and scattered to every slot that shares them; the cost is that count, not the
        voxel count.

        The three tolerances set how finely those tuples are told apart, each in the unit of the thing it bins
        (:func:`quantise`): ``transmit_tolerance`` on the flip-angle scale, ``pose_tolerance`` on the entries
        of the rotation, ``off_resonance_tolerance`` in hertz. That is how a SMOOTH map is afforded -- a
        transmit profile a machine produces is continuous, and unbinned it costs one propagation per voxel.
        Binning is exact to the tolerance in the quantity binned, and each reaches the signal through a
        function of slope at most one there, so the signal error is of the same order and never larger.
        ``None`` bins nothing. ``encoding_tolerance`` bins the machine's gradient-side terms per voxel as
        :meth:`replay` does, and each class propagates the acquisition as played there.
        """
        if self.mode != "frames":
            raise ValueError(f"replay_bloch propagates the magnetisation at a pose, so it needs a frames-mode "
                             f"phantom, which states one rotation per slot (RPH.md 4); this one is {self.mode!r}. "
                             f"A mode that leaves the substrate's azimuth unstated leaves the propagation "
                             f"undefined, and a distribution of poses under a scaled RF pulse is not the "
                             f"composition of one propagation, so neither is approximated here.")
        rf = waveform.rf
        if not rf:
            raise ValueError("the Bloch route replays an RF schedule and this sequence carries none")
        with Run("phantom.replay_bloch", params=dict(n_voxels=int(self.n_voxels), n_meas=int(waveform.n_meas),
                                               encoding_tolerance=encoding_tolerance,
                                               scanner=(scanner if scanner is None or isinstance(scanner, (int, float)) else type(scanner).__name__))) as run:
            loaded = self._loaded_packs(packs)
            kappa = self.layer_values("kappa_B1", transmit, combine="mul")
            kappa = np.ones(self.n_voxels) if kappa is None else kappa
            dB0 = self.layer_values("delta_B0_T", off_resonance)
            dB0 = np.zeros(self.n_voxels) if dB0 is None else dB0
            self._check_relaxation(waveform, loaded, forms)
            m0 = self._m0(proton_density)
            from .so3 import rotations_from_quaternions
            from .replay import _pose_matrix
            from ..spec.tissue import Tissue
            R_s = _pose_matrix(pose)
            sid, frac = self.substrate_id, self.geometric_fraction
            live = [(v, p) for v in range(self.n_voxels) for p in range(sid.shape[1])
                    if sid[v, p] >= 0 and frac[v, p] > 0.0 and self.substrates[int(sid[v, p])]["kind"] != "inert"]
            if not live:
                raise ValueError("the phantom cites no signal-bearing substrate")
            vp = np.array(live, np.int64)
            v_idx, p_idx = vp[:, 0], vp[:, 1]
            ids = sid[v_idx, p_idx].astype(int)
            R = rotations_from_quaternions(self.pose_quat[v_idx, p_idx])
            if R_s is not None:
                R = np.einsum("ij,njk->nik", R_s, R)                       # substrate -> specimen -> lab
            R = R.reshape(-1, 9)
            classes = self._encoding_classes(waveform, scanner, encoding_tolerance, report)
            waveforms = [waveform] if classes is None else classes[1]
            cls_of_voxel = np.zeros(self.n_voxels, int) if classes is None else classes[0]
            # every slot's propagation key: substrate, encoding class, rounded pose, rounded transmit scale, rounded
            # field offset
            keys = np.concatenate([ids[:, None].astype(np.float64), cls_of_voxel[v_idx][:, None].astype(np.float64),
                                   quantise(R, pose_tolerance),
                                   quantise(kappa[v_idx], transmit_tolerance)[:, None],
                                   quantise(dB0[v_idx], None if off_resonance_tolerance is None
                                            else float(off_resonance_tolerance) / GAMMA_BAR)[:, None]], axis=1)
            uniq, inverse = np.unique(keys, axis=0, return_inverse=True)
            inverse = np.asarray(inverse).reshape(-1)
            gate = self.gate_integral(waveform)
            S = None
            run.phase("poses", n_propagations=int(uniq.shape[0]), n_classes=len(waveforms))
            for u in range(uniq.shape[0]):
                run.progress(u, uniq.shape[0], unit="propagations")
                first = int(np.flatnonzero(inverse == u)[0])
                i, kap, off = int(ids[first]), float(kappa[v_idx[first]]), float(dB0[v_idx[first]])
                played = waveforms[int(cls_of_voxel[v_idx[first]])]
                sub = self.substrates[i]
                if sub["kind"] == "analytic":
                    form = self._form(i, sub, forms)
                    pose_R = R[first].reshape(3, 3) if (sub.get("oriented", False) or getattr(form, "oriented", False)) else None
                    resp = form.response(played, pose=pose_R) * _static_spin_rf(played, kap)
                    if off != 0.0:
                        resp = resp * np.exp(1j * GAMMA * off * gate)
                else:
                    resp = loaded[i].replay_bloch(played, b1_scale=kap, off_resonance_T=(off or None),
                                                  tissue=Tissue.from_meta(sub.get("tissue")), scanner=scanner,
                                                  orientation=R[first].reshape(3, 3), complex_signal=True)
                resp = np.atleast_1d(np.asarray(resp, np.complex128))
                if S is None:
                    S = np.zeros((self.n_voxels, resp.shape[0]), np.complex128)
                members = np.flatnonzero(inverse == u)
                w = frac[v_idx[members], p_idx[members]].astype(np.float64) * m0[v_idx[members], ids[members]]
                np.add.at(S, v_idx[members], w[:, None] * resp[None, :])
            ph = self._concomitant_phase(waveform, scanner)
            if ph is not None:
                S = S * np.exp(1j * ph)
            run.progress(uniq.shape[0], uniq.shape[0], unit="propagations")
            if report is not None:
                report["seconds"] = run.phase_seconds()
            return self.voxel_index, (S if complex_signal else np.abs(S))

    def replay_train(self, waveform, *, echo=-1, transmit=None, transmit_tolerance=1e-2, off_resonance=None,
                     off_resonance_tolerance=2.0, scanner=None, pose=None, packs=None, proton_density=None,
                     keep=None, complex_signal=False, jax=None, forms=None, report=None):
        """A diffusion-prepared RF train over every voxel, at one echo: ``(voxel_index, S)``.

        This is what :mod:`dmipy_sim.replay.pathways` is for. :meth:`replay_bloch` needs one rotation per slot
        and so refuses an orientation distribution (dmipy-sim#338); a train decomposed into microscopic gates
        does not, because each gate is an ordinary phase sum and the pose expansion already carries those. The
        gate expansions are built ONCE per substrate and are the expensive part; the transmit scale then
        enters as a re-weighting of coefficients already built, and a uniform field offset is carried THROUGH
        the train rather than applied at the end, because off-resonance is gated like the gradient -- a
        pathway that spent an interval along z accrues none of it.

        ``transmit`` and ``off_resonance`` are per voxel as :meth:`replay_bloch` takes them, each binned by
        its tolerance (:func:`quantise`; ``off_resonance_tolerance`` in hertz), so a smooth map costs one
        state propagation per distinct value and a drifting magnet, uniform in space, costs one.

        A closed form takes the same pathway sum with its own response to each gate's gradient in place of a
        pose expansion (:func:`~dmipy_sim.replay.pathways.closed_form_train`): free water under a train is
        ``exp(-b_gate D)`` per pathway with its bulk relaxation over the waveform, the reading a pack's walkers
        get for the same gate. An oriented closed form is refused, since its response depends on a pose the
        distribution does not state.
        """
        from .pathways import train_response, closed_form_train
        from .replay import _pose_matrix
        from ..spec.tissue import Tissue

        if self.mode not in ("odf_sh", "peaks", "frames", "bingham"):
            raise ValueError(f"unknown orientation mode {self.mode!r}")
        if self._encoding_classes(waveform, scanner, None) is not None:
            raise ValueError(
                "a train replay builds one set of gate expansions per substrate from the waveform as prescribed, and "
                "this scanner delivers a different gradient at every voxel (a nonlinearity tensor, a field shape or "
                "the coils' Maxwell term; dmipy-sim#377). Carrying that here means one gate set per encoding class, "
                "which is not done; replay at the field alone (scanner=<tesla>), or use replay / replay_bloch")
        R_s = _pose_matrix(pose)
        with Run("phantom.replay_train", params=dict(n_voxels=int(self.n_voxels), n_meas=int(waveform.n_meas), echo=echo,
                                               scanner=(scanner if scanner is None or isinstance(scanner, (int, float)) else type(scanner).__name__))) as run:
            loaded = self._loaded_packs(packs)
            kappa = self.layer_values("kappa_B1", transmit, combine="mul")
            kappa = np.ones(self.n_voxels) if kappa is None else np.broadcast_to(np.asarray(kappa, np.float64), (self.n_voxels,))
            binned = quantise(kappa, transmit_tolerance)
            dB0 = self.layer_values("delta_B0_T", off_resonance)
            dw_binned = (np.zeros(self.n_voxels) if dB0 is None else
                         quantise(2.0 * np.pi * GAMMA_BAR * np.asarray(dB0, np.float64),
                                  None if off_resonance_tolerance is None else 2.0 * np.pi * float(off_resonance_tolerance)))
            scales, offsets = np.unique(binned), np.unique(dw_binned)
            m0 = self._m0(proton_density)

            # The band to expand at is the one the DISTRIBUTION retains, not the one the response reaches: composing
            # is an inner product, so an ODF of order 8 cannot see a response's order 38, and for a brain that is
            # the difference between seconds and minutes.
            if keep is None:
                keep = (int(self.meta["orientation"].get("lmax", 8)), 0)

            trains, forms_ = {}, {}
            run.phase("gates", n_substrates=len(self.substrates))
            for i, sub in enumerate(self.substrates):
                if sub["kind"] == "inert":
                    continue
                if sub["kind"] == "pack":
                    trains[i] = train_response(loaded[i], waveform, keep=keep,
                                               tissue=Tissue.from_meta(sub.get("tissue")), scanner=scanner, pose=R_s)
                else:
                    forms_[i] = closed_form_train(self._form(i, sub, forms), waveform)
                run.progress(i + 1, len(self.substrates), unit="substrates")
            if not trains and not forms_:
                raise ValueError("the phantom cites no signal-bearing substrate")
            run.phase("gather")

            sid, frac = self.substrate_id, self.geometric_fraction
            n_meas = int(waveform.n_meas)
            S = np.zeros((self.n_voxels, n_meas), np.complex128)
            gates = None
            if trains:
                from .so3 import rebanded
                first = next(iter(trains.values()))
                probes = {i: tr.at(1.0, echo=echo) for i, tr in trains.items()}
                # each pack's train is expanded at the band its own response needs (a b = 0 gate reaches order two,
                # a diffusion preparation higher); the composition reads them all at the widest, zeros above a
                # narrower one's own band being exact, and no wider than the distribution can use
                keep_l = min(int(keep[0]), max(pr.lmax for pr in probes.values()))
                keep_n = max(pr.nmax for pr in probes.values())
                gates, readouts = first.n_gates, first.readouts
                vp, F = self.slot_coefficients(keep_l, keep_n)
                ids = sid[vp[:, 0], vp[:, 1]].astype(int)
                weight = frac[vp[:, 0], vp[:, 1]].astype(np.float64) * m0[vp[:, 0], ids]
                # every (substrate, transmit scale, offset) triple has its own coefficients and every slot belongs
                # to exactly one, so the whole phantom is one gather and one contraction
                pairs, coeff = {}, []
                for i, tr in trains.items():
                    for scale in scales:
                        for dwv in offsets:
                            pairs[(i, float(scale), float(dwv))] = len(coeff)
                            resp = tr.at(float(scale), echo=echo, dw=float(dwv))
                            coeff.append(np.asarray(rebanded(resp.coeffs, resp.lmax, resp.nmax, keep_l, keep_n), np.complex128))
                coeff = np.stack(coeff)                                   # (n_classes, n_meas, n_feat)
                which = np.array([pairs.get((int(i), float(sc), float(dv)), -1)
                                  for i, sc, dv in zip(ids, binned[vp[:, 0]], dw_binned[vp[:, 0]])])
                (_gather_jax if jax else _gather)(S, vp, F, weight, which, coeff)
            else:
                pairs, keep_l = {}, None
            # a closed form has no pose: its slots take its train amplitude at their own transmit scale and offset
            n_form_pairs = 0
            for i, cf in forms_.items():
                if gates is None:
                    gates, readouts = cf.n_gates, cf.readouts
                v_idx, p_idx = np.nonzero((sid == i) & (frac > 0.0))
                w = frac[v_idx, p_idx].astype(np.float64) * m0[v_idx, i]
                key = np.stack([binned[v_idx], dw_binned[v_idx]], axis=1)
                for sc, dv in np.unique(key, axis=0):
                    m = (key[:, 0] == sc) & (key[:, 1] == dv)
                    amp = cf.at(float(sc), echo=echo, dw=float(dv))
                    np.add.at(S, v_idx[m], w[m][:, None] * amp[None, :])
                    n_form_pairs += 1

            if report is not None:
                report.update(n_scales=len(scales), n_offsets=len(offsets), n_gates=gates,
                              lmax=keep_l, n_echoes=len(readouts), n_pairs=len(pairs) + n_form_pairs,
                              n_closed_forms=len(forms_))
            ph = self._concomitant_phase(waveform, scanner, echo=echo)
            if ph is not None:
                S = S * np.exp(1j * ph)
            if report is not None:
                report["seconds"] = run.phase_seconds()
            return self.voxel_index, (S if complex_signal else np.abs(S))

    def to_volume(self, values, fill=np.nan):
        """Scatter per-voxel values back onto the dense grid: ``(nx, ny, nz) + values.shape[1:]``, with ``fill``
        where the sparse phantom has no voxel."""
        return scatter_volume(self.grid.shape, self.voxel_index, values, fill)


def _static_spin_rf(waveform, b1_scale):
    """The transverse magnetisation a single static spin at the origin keeps through this sequence's RF schedule
    at this transmit scale: what multiplies an analytic substrate's closed form, since a closed form has diffusion
    attenuation but no magnetisation of its own. Zero gradient, so the schedule alone acts."""
    from .trajectories import replay_bloch
    n_t, dt = waveform.n_t, float(waveform.dt)
    out = replay_bloch(np.zeros((1, n_t, 3)), dt, np.zeros((1, n_t, 3)), dt, waveform.rf, b1_scale=float(b1_scale))
    return complex(np.atleast_1d(np.asarray(out).ravel())[-1])
