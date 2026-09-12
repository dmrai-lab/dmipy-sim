"""``Phantom``: solved substrates arranged in voxels, built from volumes and replayed as a volume.

A phantom is keyed by its substrate **objects** (:class:`~dmipy_sim.phantom.substrates.PackSubstrate`,
:class:`~dmipy_sim.phantom.substrates.FreeWater`, :class:`~dmipy_sim.phantom.substrates.Inert`): fractions,
orientations and replay-time overrides all name the object, so an id is typed once, when the object is made.
Provenance (id, license, citation) is asked for when a file is **written**, never when a phantom is built.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

from .grid import Grid
from .substrates import Inert, PackSubstrate, substrate_from_meta

__all__ = ["Phantom"]


class Phantom:
    """A voxel grid citing substrates: which is where, in what fraction, at what pose (RPH.md).

    Build one with :meth:`compose` (volumes in, sparse phantom out) or :meth:`read`; replay with :meth:`replay`,
    which returns a **dense volume** over the grid; write with :meth:`write`. The file-level object
    (:class:`~dmipy_sim.replay.phantom.ReplayPhantom`) is :attr:`file`.
    """

    def __init__(self, file, substrates):
        self.file = file                                    # the .rph object of a composed phantom; None for a partition
        self.substrates = list(substrates)
        ids = [s.name for s in self.substrates]
        if len(set(ids)) != len(ids):
            raise ValueError(f"substrate names must be unique: {ids}")

    # ---- construction --------------------------------------------------------------------------------------
    @classmethod
    def compose(cls, grid, *, fractions, orientation, remainder=None, labels=None, layers=None):
        """Place substrates into voxels by **volume fraction**.

        * ``grid`` -- a :class:`Grid`.
        * ``fractions`` -- ``{substrate: volume}`` of volume fractions over the grid, **or** an integer label
          volume (one substrate per voxel) with ``labels={value: substrate}``.
        * ``remainder`` -- the substrate that takes ``1 - sum`` where the fractions leave a gap. A voxel is always
          full, so without it a short row is an error rather than a normalised guess (RPH.md 3).
        * ``orientation`` -- one field (:class:`~dmipy_sim.phantom.orientation.Peaks`, ``ODF``, ``Watson``,
          ``Frames``, ``Fan``) when one pack substrate is cited, else ``{pack substrate: field}``. A field with
          ``K`` populations splits that substrate's slot into ``K``, each carrying its share of the fraction.
        * ``layers`` -- optional ``{name: volume}`` macroscopic layers, names from
          :data:`~dmipy_sim.replay.phantom.SCALAR_REGISTRY` (``kappa_B1``, ``delta_B0_T``, ``m0_scale``).

        Voxels with no signal-bearing content are dropped: the phantom is sparse, an anatomy costs what it occupies.
        """
        from ..replay.phantom import ReplayPhantom, SCALAR_REGISTRY, RPH_SCHEMA_VERSION
        if not isinstance(grid, Grid):
            raise TypeError(f"grid is a Grid; got {type(grid).__name__}")
        subs, F = _fraction_volumes(grid, fractions, remainder, labels)
        fields = _orientation_fields(orientation, subs)
        modes = {f.mode for f in fields.values()}
        if len(modes) > 1:
            raise ValueError(f"a phantom declares exactly one orientation mode (RPH.md 4); got {sorted(modes)}")
        mode = modes.pop() if modes else "peaks"
        lmax = max((getattr(f, "lmax", 0) for f in fields.values()), default=0)
        n_c = _n_sh(lmax)

        bearing = np.array([s.kind != "inert" for s in subs])
        live = (F[bearing] > 0).any(axis=0) if bearing.any() else np.zeros(grid.shape, bool)
        idx = np.argwhere(live)
        if idx.size == 0:
            raise ValueError("no voxel carries a signal-bearing substrate: the fraction volumes are empty")

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
                    raise ValueError(f"voxel {ijk} holds {f:.3g} of {s!r} but its orientation field has no pose there; "
                                     f"an oriented substrate needs a pose in every voxel it occupies")
                wsum = sum(w for w, _ in pops)
                for w, payload in pops:
                    slots.append((i, f * w / wsum, payload))
            rows.append((ijk, slots))
        P = max(len(sl) for _, sl in rows)
        N = len(rows)
        voxel_index = np.zeros((N, 3), np.int32)
        substrate_id = np.full((N, P), -1, np.int16)
        frac = np.zeros((N, P), np.float32)
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

        arrays = {"voxel_index": voxel_index, "substrate_id": substrate_id, "geometric_fraction": frac}
        key = {"odf_sh": "odf_sh", "peaks": "peak_dir", "frames": "pose_quat", "bingham": "pose_quat"}[mode]
        arrays[key] = ori
        ori_meta = ({"mode": "odf_sh", "lmax": int(lmax), "basis": "real", "convention": "orthonormal"} if mode == "odf_sh"
                    else {"mode": mode, "max_peaks": int(P)})
        if mode == "bingham":
            arrays["bingham_kappa"] = kap
            if np.any(rollk):
                arrays["roll_kappa"] = rollk
        meta = {"rph_schema_version": RPH_SCHEMA_VERSION, "id": None, "grid": grid.to_meta(), "orientation": ori_meta,
                "substrates": [s.to_meta() for s in subs], "license": None, "citation": None}
        for i, s in enumerate(subs):
            if s.kind == "pack" and s.uri is not None:
                meta["substrates"][i]["uri"] = s.uri
        if layers:
            names = tuple(layers)
            bad = [n for n in names if n not in SCALAR_REGISTRY]
            if bad:
                raise ValueError(f"unknown macroscopic layer(s) {bad}: the registry is {list(SCALAR_REGISTRY)} (RPH.md 5.1). "
                                 f"A replayer refuses a layer it does not know rather than dropping it, so a phantom may "
                                 f"not declare one either.")
            arrays["scalars"] = np.stack([grid.check_volume(layers[n], f"layer {n!r}")[tuple(voxel_index.T)] for n in names],
                                         axis=1).astype(np.float32)
            meta["scalars"] = list(names)
        return cls(ReplayPhantom(arrays, meta), subs)

    @classmethod
    def partition(cls, pack, grid=None, *, declared=None, outside=None, pose=None):
        """Cut **one walk** into voxels by where each walker started (RPH.md partition addressing; #76).

        * ``pack`` -- a :class:`~dmipy_sim.phantom.substrates.PackSubstrate`, or a list of them when the walk
          was packed per compartment (each cites the same walk; their weighted counts share every voxel).
        * ``grid`` -- the voxels, with ``attach="substrate"`` (the grid follows the tissue) or ``"lab"`` (the
          bore's); ``None`` leaves the voxels to the acquisition's ``Prescription`` at replay time.
        * ``declared`` -- ``{Inert or FreeWater: volume}``: fractions of walker-less slots (the myelin the walk
          excluded), declared because nothing weighs them.
        * ``outside`` -- what a voxel with no walkers is made of (``FreeWater(...)`` beyond the strands, or
          ``Inert()``); required when the grid reaches beyond the walk.
        * ``pose`` -- a :class:`~dmipy_sim.phantom.partition.Pose`, the specimen's rotation in the bore.

        Membership is derived from :attr:`ReplayPack.r0`, never stored; fractions of pack slots are emergent from
        ``spin_weights``; the grid is free (:meth:`~dmipy_sim.phantom.partition.PartitionPhantom.regrid`).
        """
        from .partition import PartitionPhantom
        packs = list(pack) if isinstance(pack, (list, tuple)) else [pack]
        return PartitionPhantom(packs, grid, declared=declared, outside=outside, pose=pose)

    @classmethod
    def read(cls, path, *, packs=None):
        """A phantom from a ``.rph``. ``packs`` -- ``{substrate name: pack or path}`` -- supplies packs for
        substrates the file cites by ``uri``; otherwise the recorded path is read on first use, and a missing file
        is reported by name. A partition file (``addressing: "partition"``) comes back as a
        :class:`~dmipy_sim.phantom.partition.PartitionPhantom`."""
        from ..replay.phantom import read_rph
        f = read_rph(path)
        if f.meta.get("addressing") == "partition":
            from .partition import PartitionPhantom
            return PartitionPhantom._read(path, f.meta, f.arrays, packs=packs)
        given = dict(packs or {})
        subs = []
        for i, m in enumerate(f.substrates):
            pk = given.pop(m.get("id"), None)
            if pk is None and m.get("kind") == "pack" and m.get("embedded"):
                pk = f.pack(i)
            subs.append(substrate_from_meta(m, pack=pk))
        if given:
            raise ValueError(f"packs= names substrates the file does not cite: {sorted(given)}; it cites "
                             f"{[m.get('id') for m in f.substrates]}")
        return cls(f, subs)

    def write(self, path, *, id, license, citation, embed=False, extra_meta=None):
        """Write the ``.rph``. Provenance is required here, because a file is a published artifact; ``embed=True``
        copies every cited pack into the file so it stands alone (RPH.md 2), else packs are cited by path and an
        in-memory pack without one is refused."""
        from ..replay.phantom import write_rph
        f = self.file
        a = f.arrays
        embed_packs, subs = {}, []
        for i, s in enumerate(self.substrates):
            m = s.to_meta()
            if s.kind == "pack":
                if embed:
                    embed_packs[i] = s.pack
                elif s.uri is not None:
                    m["uri"] = s.uri
                else:
                    raise ValueError(f"substrate {s.name!r} is an in-memory pack with no path: write with embed=True, "
                                     f"or declare it from its .rpk path so the file can cite it")
            subs.append(m)
        ori = f.meta["orientation"]
        kw = {}
        if ori["mode"] == "odf_sh":
            kw["odf_sh"], kw["lmax"] = a["odf_sh"], ori.get("lmax")
        elif ori["mode"] == "peaks":
            kw["peak_dir"] = a["peak_dir"]
        else:
            kw["pose_quat"] = a["pose_quat"]
            if ori["mode"] == "bingham":
                kw["bingham_kappa"], kw["roll_kappa"] = a["bingham_kappa"], a.get("roll_kappa")
        names = tuple(f.meta.get("scalars", ()))
        meta = write_rph(path, voxel_index=a["voxel_index"], substrate_id=a["substrate_id"],
                         geometric_fraction=a["geometric_fraction"], substrates=subs, grid=self.grid,
                         id=id, license=license, citation=citation, embed_packs=embed_packs,
                         scalars=a.get("scalars"), scalar_names=names, extra_meta=extra_meta, **kw)
        self.file.meta.update(id=id, license=license, citation=citation)
        return meta

    # ---- structure -----------------------------------------------------------------------------------------
    @property
    def grid(self):
        return self.file.grid

    @property
    def mode(self):
        return self.file.mode

    @property
    def n_voxels(self):
        return self.file.n_voxels

    @property
    def voxel_index(self):
        """``(n_voxels, 3)``: which grid voxels the phantom occupies."""
        return self.file.voxel_index

    @property
    def layers(self):
        """The macroscopic layers this phantom declares (RPH.md 5.1)."""
        return self.file.scalar_names

    def index_of(self, substrate):
        if isinstance(substrate, (int, np.integer)):
            return int(substrate)
        if isinstance(substrate, str):
            names = [s.name for s in self.substrates]
            if substrate not in names:
                raise ValueError(f"no substrate named {substrate!r}; the phantom cites {names}")
            return names.index(substrate)
        for i, s in enumerate(self.substrates):
            if s is substrate:
                return i
        raise ValueError(f"{substrate!r} is not one of this phantom's substrates: {self.substrates}")

    def fraction(self, substrate):
        """The volume fraction of one substrate over the grid: a dense volume, 0 where the phantom has no voxel."""
        i = self.index_of(substrate)
        return self.to_volume(self.file.fraction(i), fill=0.0)

    def to_volume(self, values, fill=np.nan):
        """Scatter per-voxel values onto the dense grid, ``grid.shape + values.shape[1:]``."""
        return self.file.to_volume(values, fill=fill)

    def sparse(self, volume):
        """The occupied rows of a dense volume: ``(voxel_index, values)`` with ``values`` of shape
        ``(n_voxels,) + volume.shape[3:]``."""
        v = np.asarray(volume)
        if v.shape[:3] != tuple(self.grid.shape):
            raise ValueError(f"volume has shape {v.shape}, not the grid's {tuple(self.grid.shape)} + ...")
        return self.voxel_index, v[tuple(self.voxel_index.T)]

    def __repr__(self):
        return (f"Phantom(voxels={self.n_voxels}, grid={list(self.grid.shape)}, {self.mode}, "
                f"substrates={self.substrates}" + (f", layers={list(self.layers)}" if self.layers else "") + ")")

    # ---- replay ---------------------------------------------------------------------------------------------
    def _packs(self, packs):
        out = {}
        for i, s in enumerate(self.substrates):
            if s.kind == "pack":
                out[i] = s.pack
        for key, pk in (packs or {}).items():
            out[self.index_of(key)] = pk
        return out

    def replay(self, seq, *, B0_T=None, b0_dir=(0.0, 0.0, 1.0), tissue="nominal", packs=None, complex_signal=False,
               T2_s=None, T1_s=None, rho_m_s=None, D_m2_s=None, chi_iso=None, chi_aniso=0.0,
               transmit=None, off_resonance=None, proton_density=None, cache=None):
        """The signal of every voxel under ``seq``: a dense volume ``grid.shape + (n_measurements,)``, NaN where
        the phantom has no voxel (:meth:`sparse` gives the rows).

        Each cited pack is replayed once into its response over poses and every voxel is an inner product of
        that with its own orientation distribution (RPH.md 6). Knobs apply to every pack; a substrate's own
        declared tissue values win over them, and anything neither names is the pack's nominal value.

        **Maps at replay time**, each a scalar, a volume of the grid's shape, or a callable of scanner
        coordinates ``f(xyz_m (N, 3)) -> (N,)`` evaluated at the voxel centres:

        * ``transmit`` -- the B1+ scale, 1 nominal; multiplies a ``kappa_B1`` layer in the file. A transmit scale
          acts on the magnetisation, so giving one (or carrying the layer) sends the replay through the RF-aware
          route -- one propagation per distinct pose -- which needs a frames-mode phantom and refuses the others
          rather than approximating them.
        * ``off_resonance`` -- a field-map value in T, added to a ``delta_B0_T`` layer: a uniform precession over
          the voxel through the acquisition's own coherence gate (zero for a 180 at TE/2).
        * ``proton_density`` -- multiplies every slot's ``m0`` in the voxel (and an ``m0_scale`` layer).

        ``cache`` (a directory, or ``True``) keeps each pack's expansion on disk under the acquisition and the knobs,
        so a phantom replayed twice under the same acquisition pays the expansion once
        (:meth:`ReplayPack.pose_response`). A declared layer this route cannot carry raises rather than being dropped.
        """
        f = self.file
        self._check_prescription(seq)
        maps = dict(transmit=self._map(transmit, "transmit"), off_resonance=self._map(off_resonance, "off_resonance"),
                    proton_density=self._map(proton_density, "proton_density"))
        common = dict(B0=B0_T, b0_dir=b0_dir, tissue=tissue, packs=self._packs(packs), complex_signal=complex_signal,
                      T2=T2_s, T1=T1_s, rho=rho_m_s, D=D_m2_s, chi_iso=chi_iso, chi_aniso=chi_aniso,
                      off_resonance=maps["off_resonance"], proton_density=maps["proton_density"])
        if maps["transmit"] is not None or "kappa_B1" in f.scalar_names:
            _, S = f.replay_bloch(seq, transmit=maps["transmit"], **common)
        else:
            _, S = f.replay(seq, cache=cache, **common)
        return self.to_volume(S)

    def _check_prescription(self, seq):
        """A prescribed acquisition and this grid must agree on the scanner axes: the gradient and B0
        directions are given in them (ACQUISITION.md 4.1), so a mismatch is refused rather than rotated."""
        p = getattr(seq, "prescription", None)
        if p is not None and p.axes != self.grid.axes:
            raise ValueError(f"the acquisition is prescribed on axes {p.axes!r} and the phantom's grid on {self.grid.axes!r}: "
                             f"the gradient and B0 directions are given in the scanner frame, so the two must agree; "
                             f"build the grid with Grid.from_prescription(seq.prescription) or re-prescribe the sequence")

    def _map(self, value, name):
        """A replay-time map as one value per occupied voxel, or None."""
        if value is None:
            return None
        if callable(value):
            out = np.asarray(value(self.grid.positions_m(self.voxel_index)), np.float64).reshape(-1)
            if out.shape != (self.n_voxels,):
                raise ValueError(f"{name}(xyz) must return one value per voxel ({self.n_voxels}); got {out.shape}")
            return out
        v = np.asarray(value, np.float64)
        if v.ndim == 0:
            return v
        if v.shape == tuple(self.grid.shape):
            return v[tuple(self.voxel_index.T)]
        raise ValueError(f"{name} is a scalar, a volume of the grid's shape {tuple(self.grid.shape)}, or a callable of "
                         f"scanner coordinates; got an array of shape {v.shape}")


# ------------------------------------------------------------------ helpers
def _fraction_volumes(grid, fractions, remainder, labels):
    """The substrate list (in slot order) and ``(n_sub,) + grid`` fractions, rows checked to sum to one."""
    subs = []

    def add(s):
        if not hasattr(s, "kind") or s.kind not in ("pack", "analytic", "inert"):
            raise TypeError(f"a substrate is a PackSubstrate, FreeWater or Inert object; got {s!r}")
        for t in subs:
            if t is s:
                return subs.index(t)
        subs.append(s)
        return len(subs) - 1

    if isinstance(fractions, dict):
        if labels is not None:
            raise ValueError("labels= goes with an integer label volume, not with a fractions mapping")
        vols = []
        for s, vol in fractions.items():
            i = add(s)
            v = grid.check_volume(vol, f"fractions of {s!r}")
            if v.min() < 0.0:
                raise ValueError(f"fractions of {s!r} have negative values")
            vols.append((i, v))
    else:
        lab = np.asarray(fractions)
        if lab.shape != tuple(grid.shape) or not np.issubdtype(lab.dtype, np.integer):
            raise ValueError(f"a label volume is an integer volume of the grid's shape {tuple(grid.shape)}; got "
                             f"{lab.shape} of {lab.dtype}. Volumes of fractions go in a mapping instead.")
        if not labels:
            raise ValueError("a label volume needs labels={value: substrate} to say which substrate each value is")
        present = set(np.unique(lab).tolist()) - {-1}
        missing = present - set(int(k) for k in labels)
        if missing:
            raise ValueError(f"label values {sorted(missing)} have no substrate in labels=")
        vols = [(add(s), (lab == int(val)).astype(np.float64)) for val, s in labels.items()]
    if remainder is not None:
        i_rem = add(remainder)
    F = np.zeros((len(subs),) + tuple(grid.shape))
    for i, v in vols:
        F[i] += v
    tot = F.sum(axis=0)
    if tot.max() > 1.0 + 1e-4:
        raise ValueError(f"the fractions of {int((tot > 1.0 + 1e-4).sum())} voxel(s) sum to more than one "
                         f"(max {tot.max():.4g}): fractions are shares of the voxel volume")
    short = tot < 1.0 - 1e-4
    if short.any():
        if remainder is None:
            raise ValueError(
                f"{int(short.sum())} voxel(s) have fractions summing to less than one (min {tot.min():.4g}) and no "
                f"remainder= was given. A voxel is always full: what is not tissue is a substrate -- Inert() or "
                f"FreeWater(...) -- not slack in the sum, because composing a short row returns a signal that is "
                f"quietly too low (RPH.md 3).")
        F[i_rem] = F[i_rem] + np.where(tot < 1.0, 1.0 - tot, 0.0)
    return subs, F


def _orientation_fields(orientation, subs):
    """Which substrate each orientation field applies to: every pack substrate by default, or per object."""
    oriented = [i for i, s in enumerate(subs) if s.kind == "pack"]
    if isinstance(orientation, dict):
        out = {}
        for s, f in orientation.items():
            i = next((j for j, t in enumerate(subs) if t is s), None)
            if i is None:
                raise ValueError(f"orientation names {s!r}, which is not among the substrates {subs}")
            if subs[i].kind != "pack":
                raise ValueError(f"{s!r} is {subs[i].kind!r} and has no orientation: an analytic or inert substrate "
                                 f"is orientation-independent")
            out[i] = f
        missing = [subs[i] for i in oriented if i not in out]
        if missing:
            raise ValueError(f"no orientation given for pack substrate(s) {missing}")
        return out
    if orientation is None:
        if oriented:
            raise ValueError(f"pack substrate(s) {[subs[i] for i in oriented]} need an orientation field")
        return {}
    if not hasattr(orientation, "at") or not hasattr(orientation, "mode"):
        raise TypeError(f"orientation is a Peaks, ODF, Watson, Frames or Fan field, or a mapping of them; got {type(orientation).__name__}")
    if len(oriented) > 1 and getattr(orientation, "weights", None) is not None:
        raise ValueError(f"one orientation field with populations cannot serve {len(oriented)} pack substrates "
                         f"unambiguously: pass a mapping {{substrate: field}}")
    return {i: orientation for i in oriented}


def _quat_of(R):
    """``(x, y, z, w)`` of a proper rotation, sign fixed by ``w >= 0`` so a pose has one spelling."""
    from scipy.spatial.transform import Rotation
    q = Rotation.from_matrix(np.asarray(R, np.float64).reshape(3, 3)).as_quat()
    return q if q[3] >= 0 else -q


def _n_sh(lmax):
    from ..replay.so3 import n_sh_coeffs
    return n_sh_coeffs(lmax)
