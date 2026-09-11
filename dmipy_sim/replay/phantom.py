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
from pathlib import Path

import numpy as np

from ..phantom.grid import Grid
from ..constants import GAMMA
from ..phantom.substrates import substrate_from_meta
from .so3 import n_sh_coeffs

__all__ = ["ReplayPhantom", "read_rph", "write_rph", "Grid", "SUBSTRATE_KINDS", "SCALAR_REGISTRY", "RPH_SCHEMA_VERSION"]

SUBSTRATE_KINDS = ("pack", "analytic", "inert")
RPH_SCHEMA_VERSION = "0.4.0"

#: The macroscopic layers a phantom may declare per voxel (RPH.md 5.1). A name outside this registry is
#: refused rather than ignored: a layer silently dropped is a phantom that replays wrong while looking right.
SCALAR_REGISTRY = ("kappa_B1", "delta_B0_T", "m0_scale")


def _lmax_of_n_coeffs(n_c):
    for l in range(0, 33, 2):
        if n_sh_coeffs(l) == n_c:
            return l
    raise ValueError(f"{n_c} coefficients is not an even-order real SH block")



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
    def replay(self, waveform, *, B0=None, b0_dir=(0.0, 0.0, 1.0), tissue="nominal", packs=None,
               n_check=256, complex_signal=False, T2=None, T1=None, rho=None, D=None,
               chi_iso=None, chi_aniso=0.0, off_resonance=None, proton_density=None):
        """Replay the whole phantom through the pose expansion: ``(voxel_index, S)`` with ``S`` of shape
        ``(n_voxels, n_measurements)``.

        Each cited pack is replayed **once** into its response over poses (:meth:`ReplayPack.pose_response`),
        and every voxel is then an inner product of those SO(3) coefficients with its own orientation
        distribution. That is the whole economy of a replay phantom: the expensive object is the walk, and it is
        shared by every voxel and every pose that cites it.

        The acquisition's gradient and B0 directions are in the scanner frame of the grid; the substrates rotate
        under it. Knobs given here apply to every pack; a substrate's own ``tissue`` (RPH.md 3.2) wins over them,
        and anything neither names takes the pack's nominal value. ``packs`` supplies the packs of substrates
        cited by ``uri`` as ``{id or index: path or ReplayPack}``.

        No band is passed: each pack projects its response at the band that response needs, and the composition
        retains only what this phantom's orientations can reach.

        The macroscopic layers (RPH.md 5.1) come from the file **and** from this call, per voxel ``(n_voxels,)``:
        ``off_resonance`` (T) adds to a ``delta_B0_T`` layer and ``proton_density`` multiplies an ``m0_scale``
        layer. A ``kappa_B1`` layer cannot be carried here -- it acts on the magnetisation, not on a phase sum --
        and is refused with the route that can (:meth:`replay_bloch`), never dropped.
        """
        if "kappa_B1" in self.scalar_names:
            raise ValueError(
                "this phantom declares a kappa_B1 layer, which scales every RF flip angle and so needs the "
                "RF-aware (vector-Bloch) replay of each pack at the voxel's pose. A magnitude gradient replay "
                "cannot carry it, and dropping it would return a signal that looks right and is not. Use "
                "ReplayPhantom.replay_bloch, which propagates the magnetisation per pose.")
        knobs = dict(T2=T2, T1=T1, rho=rho, D=D, chi_iso=chi_iso, chi_aniso=chi_aniso)
        pose, analytic, m0 = self._responses(waveform, B0, b0_dir, tissue, packs, n_check, knobs,
                                             keep=self.retained_band(), proton_density=proton_density)
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
        dB0 = self.layer_values("delta_B0_T", off_resonance)
        if dB0 is not None:
            S = S * np.exp(1j * GAMMA * dB0[:, None] * self.gate_integral(waveform))
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

    def _responses(self, waveform, B0, b0_dir, tissue, packs, n_check, knobs, keep=None, proton_density=None):
        """One response per substrate: a :class:`PoseResponse` for a pack, a closed form for an analytic
        substrate, nothing for an inert one. Plus the per-voxel ``m0``."""
        pose, analytic = {}, {}
        loaded = self._loaded_packs(packs)
        for i, sub in enumerate(self.substrates):
            if sub["kind"] == "inert":
                continue
            if sub["kind"] == "analytic":
                analytic[i] = substrate_from_meta(sub).response(waveform)      # refuses an unknown closed form
                continue
            kw = dict(knobs)
            kw.update({k: v for k, v in (sub.get("tissue") or {}).items()})
            pose[i] = loaded[i].pose_response(waveform, tissue=tissue, B0=B0, b0_dir=b0_dir, n_check=n_check,
                                              keep=keep, **kw)
        if not pose and not analytic:
            raise ValueError("the phantom cites no signal-bearing substrate")
        return pose, analytic, self._m0(proton_density)

    def replay_bloch(self, waveform, *, B0=None, b0_dir=(0.0, 0.0, 1.0), tissue="nominal",
                     packs=None, complex_signal=False, T2=None, T1=None, rho=None, D=None, chi_iso=None,
                     chi_aniso=0.0, transmit=None, off_resonance=None, proton_density=None, decimals=3):
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
        propagated once each and scattered to every slot that shares them, ``decimals`` setting how finely
        they are distinguished; the cost is that count, not the voxel count.
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
        loaded = self._loaded_packs(packs)
        kappa = self.layer_values("kappa_B1", transmit, combine="mul")
        kappa = np.ones(self.n_voxels) if kappa is None else kappa
        dB0 = self.layer_values("delta_B0_T", off_resonance)
        dB0 = np.zeros(self.n_voxels) if dB0 is None else dB0
        knobs = dict(T2=T2, T1=T1, rho=rho, D=D, chi_iso=chi_iso, chi_aniso=chi_aniso)
        m0 = self._m0(proton_density)
        from .so3 import rotations_from_quaternions
        sid, frac = self.substrate_id, self.geometric_fraction
        live = [(v, p) for v in range(self.n_voxels) for p in range(sid.shape[1])
                if sid[v, p] >= 0 and frac[v, p] > 0.0 and self.substrates[int(sid[v, p])]["kind"] != "inert"]
        if not live:
            raise ValueError("the phantom cites no signal-bearing substrate")
        vp = np.array(live, np.int64)
        v_idx, p_idx = vp[:, 0], vp[:, 1]
        ids = sid[v_idx, p_idx].astype(int)
        R = rotations_from_quaternions(self.pose_quat[v_idx, p_idx]).reshape(-1, 9)
        # every slot's propagation key: substrate, rounded pose, rounded transmit scale, rounded field offset
        keys = np.concatenate([ids[:, None].astype(np.float64), np.round(R, int(decimals)),
                               np.round(kappa[v_idx], int(decimals))[:, None],
                               np.round(dB0[v_idx], int(decimals) + 9)[:, None]], axis=1)
        uniq, inverse = np.unique(keys, axis=0, return_inverse=True)
        inverse = np.asarray(inverse).reshape(-1)
        gate = self.gate_integral(waveform)
        S = None
        for u in range(uniq.shape[0]):
            first = int(np.flatnonzero(inverse == u)[0])
            i, kap, off = int(ids[first]), float(kappa[v_idx[first]]), float(dB0[v_idx[first]])
            sub = self.substrates[i]
            if sub["kind"] == "analytic":
                resp = substrate_from_meta(sub).response(waveform) * _static_spin_rf(waveform, kap)
                if off != 0.0:
                    resp = resp * np.exp(1j * GAMMA * off * gate)
            else:
                kw = dict(knobs)
                kw.update({k: val for k, val in (sub.get("tissue") or {}).items()})
                resp = loaded[i].replay_bloch(waveform, b1_scale=kap, off_resonance_T=(off or None), tissue=tissue,
                                              B0=B0, b0_dir=b0_dir, orientation=R[first].reshape(3, 3),
                                              complex_signal=True, **kw)
            resp = np.atleast_1d(np.asarray(resp, np.complex128))
            if S is None:
                S = np.zeros((self.n_voxels, resp.shape[0]), np.complex128)
            members = np.flatnonzero(inverse == u)
            w = frac[v_idx[members], p_idx[members]].astype(np.float64) * m0[v_idx[members], ids[members]]
            np.add.at(S, v_idx[members], w[:, None] * resp[None, :])
        return self.voxel_index, (S if complex_signal else np.abs(S))

    def to_volume(self, values, fill=np.nan):
        """Scatter per-voxel values back onto the dense grid: ``(nx, ny, nz) + values.shape[1:]``, with ``fill``
        where the sparse phantom has no voxel."""
        v = np.asarray(values)
        out = np.full(tuple(self.grid.shape) + v.shape[1:], fill, dtype=np.result_type(v.dtype, type(fill)))
        out[tuple(self.voxel_index.T)] = v
        return out


def _static_spin_rf(waveform, b1_scale):
    """The transverse magnetisation a single static spin at the origin keeps through this sequence's RF schedule
    at this transmit scale: what multiplies an analytic substrate's closed form, since a closed form has diffusion
    attenuation but no magnetisation of its own. Zero gradient, so the schedule alone acts."""
    from .trajectories import replay_bloch
    n_t, dt = waveform.n_t, float(waveform.dt)
    out = replay_bloch(np.zeros((1, n_t, 3)), dt, np.zeros((1, n_t, 3)), dt, waveform.rf, b1_scale=float(b1_scale))
    return complex(np.atleast_1d(np.asarray(out).ravel())[-1])
