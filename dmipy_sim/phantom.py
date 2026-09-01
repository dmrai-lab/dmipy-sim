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
import os

import numpy as np

__all__ = ["ReplayPhantom", "read_rph", "write_rph", "SUBSTRATE_KINDS"]

SUBSTRATE_KINDS = ("pack", "analytic", "inert")
RPH_SCHEMA_VERSION = "0.2.0"


# ------------------------------------------------------------------ writing
def write_rph(path, *, voxel_index, substrate_id, geometric_fraction, odf_sh, substrates,
              grid, lmax, id, license, citation, embed_packs=None, extra_meta=None):
    """Write a ``.rph``.

    ``embed_packs`` maps a substrate index to a ``.rpk`` path whose arrays are copied in under
    ``substrate{i}/``, making the phantom a standalone artifact -- nothing to resolve and
    nothing to go missing. Substrates not embedded carry a ``uri`` instead.

    Fractions MUST sum to one per voxel: a voxel is always full, so an unmodelled remainder is
    a substrate (``inert``), not an absence. This is checked rather than trusted.
    """
    from safetensors.numpy import save_file

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

    tensors = {"voxel_index": np.asarray(voxel_index, np.int32),
               "substrate_id": np.asarray(substrate_id, np.int16),
               "geometric_fraction": gf,
               "odf_sh": np.asarray(odf_sh, np.float32)}
    subs = [dict(s) for s in substrates]
    for i, rpk in (embed_packs or {}).items():
        from .replay import read_rpk
        import hashlib
        pk = read_rpk(rpk)
        for k, v in pk.arrays.items():
            tensors[f"substrate{i}/{k}"] = np.ascontiguousarray(v)
        subs[i]["embedded"] = True
        subs[i]["pack_meta"] = pk.meta
        subs[i]["sha256"] = hashlib.sha256(open(rpk, "rb").read()).hexdigest()

    meta = {"rph_schema_version": RPH_SCHEMA_VERSION, "id": id, "grid": grid,
            "orientation": {"mode": "odf_sh", "lmax": int(lmax), "basis": "real",
                            "convention": "orthonormal"},
            "substrates": subs, "license": license, "citation": citation}
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
        return self.meta["grid"]

    @property
    def lmax(self):
        return int(self.meta["orientation"]["lmax"])

    @property
    def voxel_index(self):
        return self.arrays["voxel_index"]

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
        return (f"ReplayPhantom(id={self.meta.get('id')!r}, voxels={self.n_voxels}, "
                f"grid={self.grid.get('shape')}, substrates=[{kinds}])")

    def is_embedded(self, i):
        return bool(self.substrates[i].get("embedded"))

    def pack(self, i):
        """The embedded pack for substrate ``i`` as a :class:`~dmipy_sim.replay.ReplayPack`.

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
