"""Partition: one walk of a substrate larger than a voxel, cut into voxels by where each walker started.

The other route to a replay phantom (RPH.md; dmipy-sim#76). Nothing per walker is stored: membership is
derived from the pack's exact start positions (:attr:`~dmipy_sim.replay.replay.ReplayPack.r0`), so the grid is
free to change -- coarsen, refine, shift -- without a re-walk or a re-encode, and fractions are **emergent**
from the walkers' statistical weights rather than declared. The anatomy is one physical object, so orientation
has rigid scope: it rotates as a block through a :class:`Pose`, never voxel by voxel.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .grid import Grid
from .phantom import Phantom
from .substrates import FreeWater, Inert, PackSubstrate

__all__ = ["Pose", "PartitionPhantom"]


@dataclass(frozen=True)
class Pose:
    """Where the specimen sits in the bore: the rotation taking the substrate frame (the frame the walk was
    stored in) to the scanner frame. Translation is the grid's (its origin), so a pose is a rotation only."""

    rotation: np.ndarray

    def __init__(self, rotation=None):
        R = np.eye(3) if rotation is None else np.asarray(rotation, np.float64).reshape(3, 3)
        if not np.allclose(R @ R.T, np.eye(3), atol=1e-8) or np.linalg.det(R) < 0:
            raise ValueError("a pose is a proper rotation (R R^T = I, det +1)")
        object.__setattr__(self, "rotation", R)

    @property
    def is_identity(self):
        return bool(np.allclose(self.rotation, np.eye(3), atol=1e-12))

    def to_meta(self):
        return {"rotation": self.rotation.tolist()}

    @classmethod
    def from_meta(cls, m):
        return cls(None if m is None else m.get("rotation"))


class PartitionPhantom(Phantom):
    """One walk, cut into voxels (RPH.md partition addressing). Built by :meth:`Phantom.partition`.

    * ``packs`` -- the :class:`~dmipy_sim.phantom.substrates.PackSubstrate` objects, one per pool of the walk
      cited (a whole-walk pack, or one pack per compartment of the same walk); each is addressed by
      partition: its walkers are binned by ``r0``.
    * ``grid`` -- the voxels, with ``attach``; ``None`` binds the voxels to the acquisition's prescription at
      replay time (the scanner decides).
    * ``declared`` -- ``{Inert or FreeWater: volume}`` fractions of walker-less slots (e.g. the myelin the walk
      excluded), declared because nothing weighs them.
    * ``outside`` -- the substrate a voxel with no walkers is made of (e.g. free water beyond the strands). Without
      it such voxels are simply not part of the phantom, as a composed phantom drops voxels with nothing in them.
    * ``pose`` -- the specimen's rotation in the bore.
    """

    def __init__(self, packs, grid, *, declared=None, outside=None, pose=None):
        packs = list(packs)
        for s in packs:
            if not isinstance(s, PackSubstrate):
                raise TypeError(f"a partition cites packs; got {s!r}")
        subs = list(packs)
        self.packs = packs
        self.declared = {}
        for s, vol in (declared or {}).items():
            if isinstance(s, PackSubstrate):
                raise ValueError(f"{s!r} is a pack: its fraction is emergent from its walkers, not declared")
            subs.append(s); self.declared[s] = np.asarray(vol, np.float64)
        self.outside = outside
        if outside is not None:
            if isinstance(outside, PackSubstrate):
                raise ValueError("outside is what a voxel WITHOUT walkers is made of: Inert() or FreeWater(...)")
            subs.append(outside)
        Phantom.__init__(self, None, subs)
        self.pose = Pose() if pose is None else pose
        self._grid = grid
        if grid is not None:
            for s, vol in self.declared.items():
                grid.check_volume(vol, f"declared fraction of {s!r}")
        self._cache = {}

    # ---- structure ------------------------------------------------------------------------------------------
    @property
    def grid(self):
        return self._grid

    @property
    def mode(self):
        return "rigid"

    @property
    def layers(self):
        return ()

    def _need_grid(self, seq=None):
        if self._grid is not None:
            return self._grid
        p = getattr(seq, "prescription", None) if seq is not None else None
        if p is None:
            raise ValueError("this partition declares no grid of its own, so its voxels are the scanner's: replay it with "
                             "a sequence that carries a Prescription (seq.with_prescription(...)), or give grid= to bind them")
        return Grid.from_prescription(p, attach="lab")

    def _r0_lab(self, pack):
        """Every walker's start in the frame the grid is attached to."""
        r0 = pack.r0
        return r0 if (self.grid_attach == "substrate" or self.pose.is_identity) else r0 @ self.pose.rotation.T

    @property
    def grid_attach(self):
        return "lab" if self._grid is None else self._grid.attach

    def membership(self, substrate, grid=None):
        """``(voxel_ijk (n_w, 3), inside (n_w,))`` of one pack's walkers on ``grid`` (default: the phantom's).
        Derived, never stored (RPH.md): the bore's grid bins the posed starts, the tissue's grid the stored ones."""
        g = self._need_grid() if grid is None else grid
        s = self.substrates[self.index_of(substrate)]
        if not isinstance(s, PackSubstrate):
            raise ValueError(f"{s!r} has no walkers to bin")
        return g.bin(self._r0_lab(s.pack))

    def _voxels(self, grid):
        """The occupied voxel set and, per pack, each walker's row into it (-1 outside the grid)."""
        key = ("voxels", grid, self.grid_attach, self.pose.rotation.tobytes())
        if key in self._cache:
            return self._cache[key]
        rows_of = {}
        occupied = set()
        for s in self.packs:
            ijk, inside = grid.bin(self._r0_lab(s.pack))
            rows_of[s.name] = (ijk, inside)
            occupied.update(map(tuple, ijk[inside]))
        if self.outside is not None:
            occupied.update(map(tuple, np.argwhere(np.ones(grid.shape, bool))))
        for vol in self.declared.values():
            occupied.update(map(tuple, np.argwhere(np.asarray(vol) > 0)))
        vi = np.array(sorted(occupied), np.int64).reshape(-1, 3)
        flat = {tuple(v): i for i, v in enumerate(vi)}
        walker_row = {}
        for name, (ijk, inside) in rows_of.items():
            row = np.full(ijk.shape[0], -1, np.int64)
            idx = np.flatnonzero(inside)
            row[idx] = [flat[tuple(v)] for v in ijk[idx]]
            walker_row[name] = row
        out = (vi, walker_row)
        self._cache[key] = out
        return out

    def _weights(self, grid):
        """Per voxel: the weighted count of each pack's walkers, the declared fractions, and the emergent pack
        fractions ``f_p = (1 - declared) * W_p / sum_q W_q`` (RPH.md, weights are the fractions)."""
        vi, walker_row = self._voxels(grid)
        n = vi.shape[0]
        W = {}
        for s in self.packs:
            row = walker_row[s.name]
            w = np.asarray(s.pack.spin_weights, np.float64)
            acc = np.zeros(n)
            m = row >= 0
            np.add.at(acc, row[m], w[m])
            W[s.name] = acc
        Wtot = sum(W.values()) if W else np.zeros(n)
        decl = np.zeros(n)
        decl_of = {}
        for s, vol in self.declared.items():
            d = np.asarray(vol)[tuple(vi.T)]
            decl_of[s.name] = d
            decl += d
        if (decl > 1.0 + 1e-6).any():
            raise ValueError("declared fractions exceed one in some voxel")
        frac = {}
        for s in self.packs:
            frac[s.name] = np.where(Wtot > 0, (1.0 - decl) * W[s.name] / np.where(Wtot > 0, Wtot, 1.0), 0.0)
        if self.outside is not None:
            frac[self.outside.name] = np.where(Wtot > 0, 0.0, 1.0 - decl)
        else:
            short = (Wtot <= 0) & (decl > 0) & (decl < 1.0 - 1e-6)
            if short.any():
                raise ValueError(f"{int(short.sum())} voxel(s) declare a fraction but hold no walkers and nothing fills the "
                                 f"rest: a voxel is always full, so give outside= (Inert() or FreeWater(...)) to say what "
                                 f"the space beyond the walk is made of")
        frac.update(decl_of)
        return vi, walker_row, W, frac

    @property
    def n_voxels(self):
        return int(self._voxels(self._need_grid())[0].shape[0])

    @property
    def voxel_index(self):
        return self._voxels(self._need_grid())[0]

    def fraction(self, substrate):
        """The volume fraction of one substrate over the grid: emergent from the walkers' weights for a pack,
        declared for the others. A dense volume, 0 where the phantom has no voxel."""
        g = self._need_grid()
        vi, _, _, frac = self._weights(g)
        name = self.substrates[self.index_of(substrate)].name
        out = np.zeros(g.shape)
        out[tuple(vi.T)] = frac.get(name, np.zeros(vi.shape[0]))
        return out

    def walkers_per_voxel(self, substrate):
        """How many of a pack's walkers back each voxel (unweighted): the per-voxel Monte-Carlo floor a reader
        should know about. A dense volume."""
        g = self._need_grid()
        vi, walker_row = self._voxels(g)
        name = self.substrates[self.index_of(substrate)].name
        cnt = np.zeros(vi.shape[0])
        row = walker_row[name]
        np.add.at(cnt, row[row >= 0], 1.0)
        out = np.zeros(g.shape)
        out[tuple(vi.T)] = cnt
        return out

    def to_volume(self, values, fill=np.nan):
        v = np.asarray(values)
        g = self._need_grid()
        out = np.full(tuple(g.shape) + v.shape[1:], fill, dtype=np.result_type(v.dtype, type(fill)))
        out[tuple(self.voxel_index.T)] = v
        return out

    def __repr__(self):
        g = "scanner-prescribed" if self._grid is None else f"{list(self._grid.shape)} attach={self._grid.attach!r}"
        return (f"PartitionPhantom(packs={self.packs}, grid={g}, declared={list(self.declared)}, "
                f"outside={self.outside!r}, pose={'identity' if self.pose.is_identity else 'rotated'})")

    # ---- the free grid --------------------------------------------------------------------------------------
    def regrid(self, *, voxel_size_m=None, grid=None):
        """The same walk on other voxels: an exact rebin of ``r0``, no re-walk, no re-encode. Give a voxel size
        (the field of view is kept) or a whole :class:`Grid`. Declared volumes are resampled by nearest voxel
        centre, since they are the caller's drawing and not a measurement."""
        old = self._need_grid()
        new = grid if grid is not None else old.with_voxel_size(voxel_size_m)
        declared = {s: _resample_nearest(vol, old, new) for s, vol in self.declared.items()}
        return PartitionPhantom(self.packs, new, declared=declared, outside=self.outside, pose=self.pose)

    def with_pose(self, pose):
        """The specimen turned in the bore. With ``grid.attach == "substrate"`` only the physics changes (the
        acquisition is rotated into the tissue) and every walker keeps its voxel; with ``"lab"`` the posed starts
        are rebinned on the bore's grid and the fractions re-emerge."""
        if not isinstance(pose, Pose):
            pose = Pose(pose)
        return PartitionPhantom(self.packs, self._grid, declared=self.declared, outside=self.outside, pose=pose)

    # ---- replay ---------------------------------------------------------------------------------------------
    def replay(self, seq, *, B0_T=None, b0_dir=(0.0, 0.0, 1.0), tissue="nominal", packs=None, complex_signal=False,
               T2_s=None, T1_s=None, rho_m_s=None, D_m2_s=None, chi_iso=None, chi_aniso=0.0,
               transmit=None, off_resonance=None, proton_density=None):
        """The signal of every voxel: each pack's walkers replayed **once** (:meth:`ReplayPack.walker_signals`),
        summed by the voxel they started in, normalised by the weights in that voxel, and weighted by the
        emergent fraction and the pack's ``m0``. Declared and outside substrates add their closed form (or
        nothing). ``transmit`` is looked up per voxel and applied **per walker**, so the RF-aware route runs once for
        the whole walk rather than once per voxel; ``off_resonance`` goes the same way when a transmit scale is
        given, and otherwise is the uniform precession through the acquisition's gate, as for a composed
        phantom. A dense volume, as :meth:`Phantom.replay`.
        """
        g = self._need_grid(seq)
        if getattr(seq, "prescription", None) is not None and seq.prescription.axes != g.axes:
            raise ValueError(f"the acquisition is prescribed on axes {seq.prescription.axes!r} and the grid on {g.axes!r}")
        vi, walker_row, W, frac = self._weights(g)
        n = vi.shape[0]
        maps = {k: self._map_on(v, k, g, vi) for k, v in (("transmit", transmit), ("off_resonance", off_resonance),
                                                          ("proton_density", proton_density))}
        R = self.pose.rotation
        orientation = None if self.pose.is_identity else R
        knobs = dict(tissue=tissue, T2=T2_s, T1=T1_s, rho=rho_m_s, D=D_m2_s, B0=B0_T, b0_dir=b0_dir,
                     chi_iso=chi_iso, chi_aniso=chi_aniso)
        S = None
        given = {self.index_of(k): v for k, v in (packs or {}).items()}
        bloch = maps["transmit"] is not None
        for i, s in enumerate(self.substrates):
            if not isinstance(s, PackSubstrate):
                continue
            pk = given.get(i, s.pack)
            row = walker_row[s.name]
            m = row >= 0
            kw = dict(knobs)
            kw.update(s.tissue)
            per = {}
            if bloch:
                # the RF-aware route, once for the whole walk: every walker at its own voxel's scale and offset
                per["b1_scale"] = np.where(m, maps["transmit"][np.where(m, row, 0)], 1.0)
                if maps["off_resonance"] is not None:
                    per["off_resonance_T"] = np.where(m, maps["off_resonance"][np.where(m, row, 0)], 0.0)
            w, ew, E = pk.walker_signals(seq, orientation=orientation, **kw, **per)
            if S is None:
                S = np.zeros((n, E.shape[1]), np.complex128)
            acc = np.zeros((n, E.shape[1]), np.complex128)
            np.add.at(acc, row[m], ew[m, None] * E[m])
            norm = np.where(W[s.name] > 0, W[s.name], 1.0)
            S += (s.m0 * frac[s.name])[:, None] * acc / norm[:, None]
        for s in self.substrates:
            if isinstance(s, PackSubstrate) or s.name not in frac:
                continue
            if isinstance(s, Inert):
                continue
            resp = np.atleast_1d(np.asarray(s.response(seq), np.complex128))
            if S is None:
                S = np.zeros((n, resp.shape[0]), np.complex128)
            S += (s.m0 * frac[s.name])[:, None] * resp[None, :]
        if S is None:
            raise ValueError("the partition cites no signal-bearing substrate")
        if maps["off_resonance"] is not None and not bloch:
            # the phase route: a uniform precession over the voxel through the acquisition's own coherence gate
            from ..constants import GAMMA
            from ..replay.phantom import ReplayPhantom
            S = S * np.exp(1j * GAMMA * maps["off_resonance"][:, None] * ReplayPhantom.gate_integral(seq))
        if maps["proton_density"] is not None:
            S = S * maps["proton_density"][:, None]
        out = np.full(tuple(g.shape) + (S.shape[1],), np.nan, dtype=np.complex128 if complex_signal else np.float64)
        out[tuple(vi.T)] = S if complex_signal else np.abs(S)
        return out

    def _map_on(self, value, name, grid, vi):
        if value is None:
            return None
        if callable(value):
            out = np.asarray(value(grid.positions_m(vi)), np.float64).reshape(-1)
            if out.shape != (vi.shape[0],):
                raise ValueError(f"{name}(xyz) must return one value per voxel ({vi.shape[0]}); got {out.shape}")
            return out
        v = np.asarray(value, np.float64)
        if v.ndim == 0:
            return np.full(vi.shape[0], float(v))
        if v.shape == tuple(grid.shape):
            return v[tuple(vi.T)]
        raise ValueError(f"{name} is a scalar, a volume of the grid's shape {tuple(grid.shape)}, or a callable of scanner "
                         f"coordinates; got an array of shape {v.shape}")

    # ---- the file --------------------------------------------------------------------------------------------
    def write(self, path, *, id, license, citation, embed=False, extra_meta=None):
        """Write the partition as a ``.rph`` (RPH.md 0.5 draft, ``addressing: "partition"``): the packs (embedded
        or by path), the grid with its attachment, the pose, the declared fractions and the outside substrate.
        Membership and the pack fractions are **not** written: a reader derives them from ``r0``."""
        import json
        from ..replay.phantom import RPH_SCHEMA_VERSION
        from safetensors.numpy import save_file
        tensors = {}
        subs = []
        embed_packs = {}
        for i, s in enumerate(self.substrates):
            m = s.to_meta()
            if isinstance(s, PackSubstrate):
                m["addressing"] = "partition"
                if embed:
                    embed_packs[i] = s.pack
                elif s.uri is not None:
                    m["uri"] = s.uri
                else:
                    raise ValueError(f"substrate {s.name!r} is an in-memory pack with no path: write with embed=True")
            if s in self.declared:
                m["declared"] = True
            if s is self.outside:
                m["outside"] = True
            subs.append(m)
        for s, vol in self.declared.items():
            idx = np.argwhere(np.asarray(vol) > 0)
            # argwhere returns a Fortran-ordered view; safetensors writes the buffer as it lies
            tensors[f"declared{self.index_of(s)}/voxel_index"] = np.ascontiguousarray(idx.astype(np.int32))
            tensors[f"declared{self.index_of(s)}/fraction"] = np.ascontiguousarray(np.asarray(vol, np.float32)[tuple(idx.T)])
        for i, pk in embed_packs.items():
            import hashlib
            subs[i]["sha256"] = hashlib.sha256(b"".join(np.ascontiguousarray(v).tobytes()
                                                          for _, v in sorted(pk.arrays.items()))).hexdigest()
            for k, v in pk.arrays.items():
                tensors[f"substrate{i}/{k}"] = np.ascontiguousarray(v)
            subs[i]["embedded"] = True
            subs[i]["pack_meta"] = pk.meta
        if not tensors:
            tensors["_empty"] = np.zeros(1, np.int8)
        meta = {"rph_schema_version": "0.5.0-draft", "id": id, "addressing": "partition",
                "grid": None if self._grid is None else self._grid.to_meta(), "pose": self.pose.to_meta(),
                "orientation": {"mode": "rigid", "scope": "rigid"}, "substrates": subs,
                "license": license, "citation": citation}
        meta.update(extra_meta or {})
        save_file(tensors, str(path), metadata={"rph": json.dumps(meta)})
        return meta

    @classmethod
    def _read(cls, path, meta, arrays, packs=None):
        """A partition from a file's metadata and tensors (:meth:`Phantom.read` dispatches here)."""
        from ..replay.replay import ReplayPack
        from .substrates import substrate_from_meta
        given = dict(packs or {})
        pack_subs, declared, outside = [], {}, None
        grid = None if meta.get("grid") is None else Grid.from_meta(meta)
        for i, m in enumerate(meta["substrates"]):
            pk = given.pop(m.get("id"), None)
            if pk is None and m.get("kind") == "pack" and m.get("embedded"):
                pre = f"substrate{i}/"
                pk = ReplayPack({k[len(pre):]: v for k, v in arrays.items() if k.startswith(pre)}, m["pack_meta"])
            s = substrate_from_meta(m, pack=pk)
            if isinstance(s, PackSubstrate):
                pack_subs.append(s)
            elif m.get("declared"):
                if grid is None:
                    raise ValueError("a declared fraction needs the phantom's own grid")
                vol = np.zeros(grid.shape)
                idx = arrays[f"declared{i}/voxel_index"]
                vol[tuple(idx.T)] = arrays[f"declared{i}/fraction"]
                declared[s] = vol
            elif m.get("outside"):
                outside = s
        if given:
            raise ValueError(f"packs= names substrates the file does not cite: {sorted(given)}")
        return cls(pack_subs, grid, declared=declared, outside=outside, pose=Pose.from_meta(meta.get("pose")))


def _resample_nearest(vol, old, new):
    """A declared volume carried to another grid by the voxel centre it falls in."""
    vol = np.asarray(vol, np.float64)
    ijk = np.argwhere(np.ones(new.shape, bool))
    pos = new.positions_m(ijk)
    src, inside = old.bin(pos)
    out = np.zeros(new.shape)
    out[tuple(ijk[inside].T)] = vol[tuple(src[inside].T)]
    return out
