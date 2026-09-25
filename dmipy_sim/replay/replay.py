"""Replay packs (``.rpk``) and the compiled-scheme forward — the shared replay primitive.

A **replay pack** stores the state of one converged Monte-Carlo walk (walker trajectories in the bridge form:
per axis the two endpoints and the lowest ``K`` sine bands of the path pinned at both, plus a spin weight and
optional channels) so the
diffusion-weighted signal for *any* gradient waveform can be reconstructed without re-simulating. It is a
single ``safetensors`` file: the arrays are the tensors, the JSON metadata sits in the ``"rpk"`` header
key. Producers (e.g. the substrate generator) write them; consumers (dmipy-fit compartments, dmipy-design
waveform optimization) replay them.

The forward is exact and cheap. The signal is

    E = < w_i exp(i phi_i) > / < w_i > ,   phi_i(m) = gamma * dt * sum_t G_m(t) . r_i(t)

and because the phase is linear in position it is a contraction of the stored coefficients ``C_i`` (the two
endpoints and the ``K`` bands, :func:`~dmipy_sim.replay.compression.encode_bridge_dst`) with the waveform's
projection on the same basis,

    phi_i(m) = sum_{k,c} C_{i,k,c} * W_{m,k,c},

exact for every waveform inside the stored band. The projection ``W`` (= :func:`compile_scheme`) is independent of the walkers; each forward is
then one dense matmul ``C @ W`` + a weighted complex mean (:func:`replay_signal`). This is the SAME math
whether the acquisition is fixed and the substrate varies (fitting) or the substrate is fixed and the
waveform varies (design) — in the latter it is differentiable in ``G``, so it drives gradient-based
waveform/B1 optimization. A JAX twin (:func:`replay_signal_jax`) supplies the autodiff/GPU path.

Surface relaxivity is exact, via the stored boundary local time (the C2 channel, bridge form): a
per-walker reweight by ``exp((rho/D) * sum_t chi(t) ell_i(t))``, optionally coherence-gated by an
occupancy schedule ``chi`` (:func:`surface_logweight`).
"""
import json
from dataclasses import dataclass
from functools import cached_property

import numpy as np
from .compression import c2_bands_K as _cx_bands_K

from ..constants import GAMMA
from ..run import Run, current
from ..acquisition.rf import RFSchedule
from ..acquisition.scanner_sequence import Protocol, ScannerSequence

__all__ = ["ReplayPack", "PoseResponse", "read_rpk", "write_rpk", "analytic_pose_response",
           "compile_scheme", "replay_signal", "replay_coefficients", "replay_signal_jax", "replay_batch_jax",
           "surface_logweight"]


# ------------------------------- .rpk container I/O -------------------------------
def write_rpk(path, arrays, metadata):
    """Write a replay pack: ``arrays`` (name -> ndarray) as safetensors tensors, ``metadata`` (a dict)
    serialised to JSON under the ``"rpk"`` header key."""
    from safetensors.numpy import save_file
    meta = dict(metadata)
    tens = {k: np.ascontiguousarray(v) for k, v in arrays.items()}
    save_file(tens, str(path), metadata={"rpk": json.dumps(meta)})


def read_rpk(path):
    """Read a replay pack into a :class:`ReplayPack` (arrays + metadata)."""
    from safetensors import safe_open
    arrays = {}
    with safe_open(str(path), framework="numpy") as f:
        hdr = f.metadata() or {}
        meta = json.loads(hdr.get("rpk") or hdr.get("json") or "{}")
        for k in f.keys():
            arrays[k] = f.get_tensor(k)
    return ReplayPack(arrays, meta, source=str(path))


class PoseResponse:
    """A pack's response over every pose of its substrate, for one acquisition, as SO(3) coefficients.

    The pose of a substrate is a rotation, so the response of one measurement is a function on SO(3), and this
    holds its expansion in the real Wigner basis (:mod:`dmipy_sim.replay.so3`). Two bands are involved and they
    are not the same:

    * the band this was **projected** at, ``(lmax, nmax)``, which follows the response's own sharpness -- the
      accumulated phase amplitude :attr:`phase_amplitude`, in radians. It sets what had to be sampled, since a
      rule that resolves only the retained band folds everything above it into the coefficients kept.
    * the band a composition **retains**, which follows the orientation distribution: composing is an inner
      product, so a distribution has no reach above its own order and none at all outside ``n = 0`` when it
      leaves the substrate's azimuth unstated. :meth:`compose` restricts to that band, exactly.

    ``misfit`` is the worst case, over rotations off the projection's own grid, of what the expansion could not
    reproduce -- in signal units, to be read against the pack's Monte-Carlo floor. A worst case rather than a
    spread, because a composition is exposed to individual poses and a spread bounds nothing.
    """

    def __init__(self, coeffs, lmax, nmax, misfit, floor, phase_amplitude, n_samples):
        self.coeffs = np.asarray(coeffs)
        self.lmax, self.nmax = int(lmax), int(nmax)
        self.misfit = np.asarray(misfit, float)
        self.floor, self.n_samples = float(floor), int(n_samples)
        self.phase_amplitude = float(phase_amplitude)
        self.route = "quadrature" if n_samples > 0 else "closed"      # which route built it (#197)
        self.n_bodies = None                                          # distinct waveforms contracted (closed form)
        self.field_lmax = 0

    @property
    def n_meas(self):
        return self.coeffs.shape[0]

    def save(self, path):
        """Write the expansion as one ``.npz``: what a cache keeps between two replays of the same pack under the
        same acquisition and knobs."""
        np.savez(str(path), coeffs=self.coeffs, lmax=self.lmax, nmax=self.nmax, misfit=self.misfit, floor=self.floor,
                 phase_amplitude=self.phase_amplitude, n_samples=self.n_samples, route=self.route,
                 n_bodies=-1 if self.n_bodies is None else int(self.n_bodies), field_lmax=int(self.field_lmax))

    @classmethod
    def load(cls, path):
        z = np.load(str(path), allow_pickle=False)
        out = cls(z["coeffs"], int(z["lmax"]), int(z["nmax"]), z["misfit"], float(z["floor"]), float(z["phase_amplitude"]),
                  int(z["n_samples"]))
        out.route = str(z["route"]); nb = int(z["n_bodies"]); out.n_bodies = None if nb < 0 else nb
        out.field_lmax = int(z["field_lmax"])
        return out

    @property
    def asymmetry(self):
        """The share of the response's energy that depends on the substrate's own azimuth.

        Zero for an axially symmetric substrate, and the reason a pose has to be a rotation rather than an axis
        when it is not: an axis-only representation cannot carry that share, and a rule that samples over
        directions alone folds it into the ``n = 0`` coefficients a composition does use.
        """
        from . import so3
        num = den = 0.0
        for c in self.coeffs:
            _per_l, per_n = so3.energy(c, self.lmax, self.nmax)
            num += float(per_n[1:].sum())
            den += float(per_n.sum())
        return num / max(den, 1e-300)

    def retained(self, lmax, nmax):
        """The coefficients a distribution of this band reaches: ``(n_meas, n_features)``.

        Dropping the rest is lossless, not an approximation -- the distribution has no coefficients there to
        multiply. For an ODF of order 8 this is 45 numbers per measurement out of the 969 an unrestricted
        expansion carries, which is what makes a per-voxel composition a short dot product.
        """
        from .so3 import truncate_coeffs
        return truncate_coeffs(self.coeffs, self.lmax, self.nmax, lmax, nmax)

    def compose(self, distribution):
        """``(n_meas,)`` complex: the signal of a voxel holding this distribution of poses."""
        from .so3 import Distribution
        if not isinstance(distribution, Distribution):
            raise TypeError("compose takes a dmipy_sim.replay.so3.Distribution (Distribution.pose / axis / "
                            "axis_density / watson / bingham / uniform), so that what the coefficients mean is "
                            "carried with them")
        from .so3 import truncate_coeffs
        # both sides come down to the band they share. Dropping the distribution's higher terms is safe for the
        # same reason dropping the response's is: the response was certified to reproduce itself at its own
        # band, so it has nothing above it for those terms to multiply.
        keep_l = min(distribution.lmax, self.lmax)
        keep_n = min(distribution.nmax, self.nmax)
        f = distribution.coeffs
        if (distribution.lmax, distribution.nmax) != (keep_l, keep_n):
            f = truncate_coeffs(f, distribution.lmax, distribution.nmax, keep_l, keep_n)
        return self.retained(keep_l, keep_n) @ f

    def at(self, R):
        """``(n_meas,)`` complex: the response at one stated pose, evaluated from the expansion.

        A single pose is a distribution of unbounded band, so this reads the whole expansion, and it is the one
        case whose accuracy is the pointwise :attr:`misfit`. When one pose is all that is wanted,
        :meth:`ReplayPack.replay` with ``orientation=R`` answers it exactly and costs nothing to certify.
        """
        from .so3 import so3_design
        A = so3_design(self.lmax, np.asarray(R, np.float64).reshape(1, 3, 3), self.nmax)
        return self.coeffs @ A[0]


@dataclass(frozen=True)
class ScannerField:
    """What a replay reads from its one ``scanner=`` knob: the static field's strength in tesla (``None``
    for no field) and the patient-frame unit vector it points along, plus the machine's name for provenance.

    The direction is not a labelling detail. A susceptibility field is not isotropic and neither is the
    phase it produces: an anisotropic susceptibility depends on the angle between the source and B0, so the
    field's direction enters every contraction. A bare strength carries no direction and keeps the
    conventional bore geometry -- B0 along the bore, the patient's head-foot axis -- which is right for every
    cylindrical magnet and wrong for a bi-planar one, where B0 runs across the patient; such a machine is
    given as a :class:`~dmipy_sim.acquisition.scanners.ScannerLimits` and declares ``b0_axis``.
    """
    B0: object = None
    axis: tuple = (0.0, 0.0, 1.0)
    name: str = None


def scanner_field(scanner):
    """Resolve ``scanner=`` -- ``None``, a field strength in tesla, or a
    :class:`~dmipy_sim.acquisition.scanners.ScannerLimits` -- into a :class:`ScannerField`. A name is refused:
    resolving it is the catalogue's job (``ScannerLimits.of``), not a replay's."""
    if scanner is None:
        return ScannerField()
    from ..acquisition.scanners import ScannerLimits
    if isinstance(scanner, ScannerLimits):
        if scanner.field_T is None:
            raise ValueError(f"the catalogue knows no field strength for {scanner.name!r}; give the field in tesla")
        axis = (0.0, 0.0, 1.0)
        if scanner.b0_axis is not None:
            v = np.asarray(scanner.b0_axis, dtype=np.float64)
            axis = tuple(float(x) for x in v / np.linalg.norm(v))
        return ScannerField(B0=float(scanner.field_T), axis=axis, name=scanner.name)
    if isinstance(scanner, str):
        raise TypeError("scanner is a ScannerLimits (ScannerLimits.of('connectom')) or a field strength in tesla, not a name")
    return ScannerField(B0=float(scanner))


def _pose_matrix(pose):
    """A specimen pose as a 3x3 rotation, from a matrix or an object with ``.rotation``; ``None`` stays ``None``."""
    if pose is None:
        return None
    R = getattr(pose, "rotation", pose)
    return np.asarray(R, np.float64).reshape(3, 3)


class ReplayPack:
    """A replay pack: the channel ``arrays`` plus ``meta``, with ``load`` / ``save`` and the one
    consume path, :meth:`replay`. Accessors mirror the walk parameters (``n_t``, ``dt``, ``K``,
    ``n_walkers``) and the tiers carried (``has_relaxation``, ``has_surface``, ``has_field``)."""

    @staticmethod
    def open(uri, workers=8):
        """The columnar layout at ``uri`` (a directory, or ``hf://owner/name/prefix`` on the Hub), open by reference:
        a :class:`~dmipy_sim.replay.columnar.ColumnarPack`, whose ``view`` gives a :class:`ReplayPack` of the rows
        and bands an acquisition needs and whose ``image`` replays a whole grid in one pass over the rows."""
        from .columnar import open_columnar
        return open_columnar(uri, workers=workers)

    def __init__(self, arrays, meta, source=None):
        self.arrays = dict(arrays)
        self.meta = dict(meta)
        self.source = source

    @classmethod
    def load(cls, path):
        """Read a ``.rpk`` file: a local path, or ``hf://owner/name/path/to/file.rpk`` fetched from the hub into the
        local cache and checked against the dataset's ``manifest.json`` (:func:`~dmipy_sim.replay.publish.fetch`)."""
        from .publish import fetch, is_hub_uri
        return read_rpk(fetch(path) if is_hub_uri(path) else path)

    def save(self, path):
        """Write this pack to a ``.rpk`` file."""
        write_rpk(path, {k: v for k, v in self.arrays.items() if v is not None}, self.meta)
        self.source = str(path)
        return path

    def publish(self, repo, **kw):
        """Put this pack in the dataset ``repo`` (``owner/name``) and return its ``hf://`` URI:
        :func:`dmipy_sim.replay.publish.publish`, which takes ``path=``, ``hub=`` and ``message=``."""
        from .publish import publish
        return publish(self, repo, **kw)

    # ---- segments (RPK.md 4.3): the walk stored in windows of one duration ----
    @property
    def segments(self):
        """The segment table ``walk_params.segments``, ``{n, n_t, T, walks}``: the walk is stored in ``n`` windows
        of ``n_t`` saves (``T`` seconds) each, consecutive windows sharing their boundary save; segment 0's tensors
        sit under the channel names and segment ``i`` under the key prefix ``s{i}/``; ``walks`` lists the walks
        that produced them, ``{first, last, seed}`` in segments. Every pack declares it, a walk within one window as
        ``n = 1``."""
        seg = (self.meta.get("walk_params") or {}).get("segments")
        if not seg:
            raise ValueError("this pack declares no walk_params.segments (RPK.md 4.3): a pack is its walk in windows of "
                             "one duration, one window for a walk within that duration. It was written before the "
                             "table existed; rebuild it with build_replay_pack, which declares the table for every pack")
        return seg

    @property
    def n_segments(self):
        return int(self.segments["n"])

    def _segment_arrays(self, i):
        """The tensors of window ``i`` under the channel names: segment 0's are the unprefixed ones; a later
        segment's are its ``s{i}/`` tensors beside what every segment shares -- the weights, a static label, the
        field grid, the voxel tables, whatever has no per-segment counterpart."""
        arrays = self.arrays
        if int(i) == 0:
            return {k: v for k, v in arrays.items() if "/" not in k}
        p = f"s{int(i)}/"
        own = {k[len(p):]: v for k, v in arrays.items() if k.startswith(p)}
        if not own:
            raise IndexError(f"segment {i}: the pack stores no tensors under {p!r}")
        per_segment = {k[3:] for k in arrays if k.startswith("s1/")}
        shared = {k: v for k, v in arrays.items() if "/" not in k and k not in per_segment}
        return dict(shared, **own)

    def segment(self, i):
        """Window ``i`` of the walk as a pack of its own: its ``n_t`` saves from the window's start, its own
        certificate (``fidelity.segments[i]``), the same walkers, weights and spec."""
        import copy
        seg = self.segments; n = int(seg["n"]); i = int(i)
        if i < 0 or i >= n:
            raise IndexError(f"segment {i} of a pack that stores {n}")
        if n == 1:
            return self
        meta = copy.deepcopy(self.meta)
        wp = meta["walk_params"]
        n_seg, T_seg = int(seg["n_t"]), float(seg["T"])
        walk = next((w for w in seg.get("walks") or [] if int(w["first"]) <= i <= int(w["last"])), None)
        seed = walk["seed"] if walk is not None else wp.get("seed")
        wp.update(n_t=n_seg, T_max=T_seg, seed=seed, segments=dict(n=1, n_t=n_seg, T=T_seg, walks=[dict(first=0, last=0, seed=seed)]))
        meta["compression"]["n_t"] = n_seg
        fid = dict(meta.get("fidelity") or {})
        per = fid.pop("segments", None)
        meta["fidelity"] = dict(per[i]) if per else fid
        meta.setdefault("provenance", {})["segment"] = dict(index=i, of=n, parent_id=self.meta.get("id"))
        return ReplayPack(self._segment_arrays(i), meta, source=self.source)

    def _windows(self):
        """``[(pack, t0, n_t), ...]``: every window of the walk as a pack, the time its first save sits at on the
        walk's clock, and its saves; ``[(self, 0.0, n_t)]`` for a single-window pack."""
        n = self.n_segments
        if n == 1:
            return [(self, 0.0, int(self.n_t))]
        n_seg, dt = int(self.segments["n_t"]), float(self.dt)
        return [(self.segment(i), i * (n_seg - 1) * dt, n_seg) for i in range(n)]

    def truncate(self, n_keep, *, id=None, out_path=None):
        """The first ``n_keep`` segments as a pack: a prefix of whole windows is the range of their tensors and
        nothing is re-encoded (RPK.md 8.3). The segments keep their certificates and the whole's is the bound
        over them (:func:`~dmipy_sim.replay.bank.combine_segment_fidelity`)."""
        import copy
        from .bank import combine_segment_fidelity
        seg = self.segments; n = int(seg["n"]); n_keep = int(n_keep)
        if n_keep < 1 or n_keep > n:
            raise ValueError(f"keep between 1 and {n} segments; got {n_keep}")
        if n_keep == n:
            return self
        keep = {k: v for k, v in self.arrays.items()
                if "/" not in k or int(k[1:k.index("/")]) < n_keep}
        meta = copy.deepcopy(self.meta)
        wp = meta["walk_params"]; n_seg, T_seg = int(seg["n_t"]), float(seg["T"])
        walks = [dict(w, last=min(int(w["last"]), n_keep - 1)) for w in seg.get("walks") or [] if int(w["first"]) < n_keep]
        wp.update(n_t=n_keep * (n_seg - 1) + 1, T_max=n_keep * T_seg, segments=dict(n=n_keep, n_t=n_seg, T=T_seg, walks=walks))
        fid = dict(meta.get("fidelity") or {})
        per = fid.get("segments")
        if per:
            meta["fidelity"] = combine_segment_fidelity(per[:n_keep])
        meta.setdefault("provenance", {})["truncated"] = dict(parent_id=self.meta.get("id"), segments_kept=n_keep, of=n)
        if id is not None:
            meta["id"] = id
        out = ReplayPack(keep, meta, source=None)
        if out_path is not None:
            out.save(out_path)
        return out

    def end_state(self):
        """``(r_end, pool)``: where every walker is at the walk's last save and the pool it is in
        (:func:`~dmipy_sim.replay.continuation.end_state`)."""
        from .continuation import end_state
        return end_state(self)

    def extend(self, T_add, *, seed, out_path=None, require_gpu=None, field=True, envelope=None, device="auto"):
        """This pack lengthened by ``T_add`` seconds of new segments: the walk continued from its end on its own
        substrate with a fresh seed and appended (:func:`~dmipy_sim.replay.continuation.extend_pack`)."""
        from .continuation import extend_pack
        return extend_pack(self, T_add, seed=seed, out_path=out_path, require_gpu=require_gpu, field=field, envelope=envelope, device=device)

    def _decoded_channels(self):
        """The per-save channels of the whole walk decoded window by window and joined on the shared saves:
        ``comp`` / ``bound`` tracks ``(n_w, n_t)``, the contact increments ``ell`` ``(n_w, n_t)`` and the path
        series ``(series (n_w, n_ch, n_tf), names)`` -- each ``None`` when the pack lacks the channel."""
        from .bank import susc_path_decode
        from .compression import decode_occupancy, decode_boundary_bridge, decode_boundary_local_time
        ch = dict(self.meta.get("compression", {}).get("channels", {}) or {})
        out = dict(comp=None, bound=None, ell=None, path=None)
        comps, bounds, ells, series = [], [], [], []
        names = None
        for j, (w, _, n_seg) in enumerate(self._windows()):
            cut = 0 if j == 0 else 1                                   # the shared save is the previous window's
            if "compartment" in ch:
                occ = decode_occupancy(w.arrays, ch["compartment"])
                comp = np.asarray(occ["comp"])
                comps.append(comp[:, cut:] if comp.ndim == 2 else comp)
                if "bound" in occ:
                    b = np.asarray(occ["bound"]); bounds.append(b[:, cut:] if b.ndim == 2 else b)
            if "boundary_local_time" in ch:
                bm = dict(ch["boundary_local_time"]); bm.setdefault("n_t", n_seg)
                if w.has_surface:
                    bm.setdefault("K", _cx_bands_K(w.arrays, bm))
                    ell = decode_boundary_bridge(w.arrays, bm)
                else:
                    ell = decode_boundary_local_time(w.arrays, bm)
                ells.append(np.asarray(ell)[:, cut:])
            if "susceptibility_path" in ch:
                ser, names = susc_path_decode(w.arrays, ch["susceptibility_path"], n_w=self.n_walkers)
                series.append(ser[:, :, cut:])
        if comps:
            out["comp"] = np.concatenate(comps, axis=1) if comps[0].ndim == 2 else comps[0]
        if bounds:
            out["bound"] = np.concatenate(bounds, axis=1) if bounds[0].ndim == 2 else bounds[0]
        if ells:
            out["ell"] = np.concatenate(ells, axis=1)
        if series:
            out["path"] = (np.concatenate(series, axis=2), names)
        return out

    # ---- tiers carried ----
    @property
    def has_relaxation(self):
        """C1: a compartment channel, so per-pool T2 / T1 given at replay can be applied."""
        from .compression import has_c1
        return has_c1(self.arrays)

    @property
    def substrate(self):
        """The :class:`~dmipy_sim.spec.SubstrateSpec` the walk was driven by, when the pack embeds one."""
        d = self.meta.get("substrate")
        if d is None:
            return None
        from ..spec.substrate import SubstrateSpec
        return SubstrateSpec.from_dict(d)

    @property
    def has_surface(self):
        """C2: the boundary local time in the bridge form."""
        from .compression import has_c2
        return has_c2(self.arrays)

    @property
    def has_field(self):
        """C3: a susceptibility path channel or the static field grid."""
        ch = (self.meta.get("compression", {}).get("channels", {}) or {})
        return ch.get("susceptibility_path") is not None or "susc_grid_iso_local" in self.arrays

    @property
    def field_is_zero(self):
        """The substrate declares no field source: its embedded spec names no pool susceptibility, so its
        off-resonance field is zero everywhere and a replay at any ``B0`` is its gradient-only replay. Such a
        pack is C3-capable with a field of zero (a grey-matter sphere packing, a free walk); a pack without a
        spec, or with a susceptible pool and no field channel, still refuses ``B0``."""
        spec = self.substrate
        if spec is None or not getattr(spec, "pools", None):
            return False
        return all(getattr(p, "susceptibility", None) is None for p in spec.pools)

    def _field_active(self, B0):
        """Whether a replay at ``B0`` has a field term to evaluate: a field was asked and the substrate has one.
        Refuses a field asked of a substrate that has one but carries no channel for it."""
        if B0 is None:
            return False
        if self.has_field:
            return True
        if self.field_is_zero:
            return False
        raise ValueError("a scanner field was given but the pack carries no field tier (C3) and its substrate declares a "
                         "susceptibility, or no spec at all; build it with field=FieldGrid(...)")

    @property
    def diffusivity(self):
        """The walk's diffusivity (m^2/s) when the producer recorded it."""
        return self.meta.get("walk_params", {}).get("diffusivity")

    @property
    def permeability(self):
        """The permeability (m/s) the walk realised across its walls, from the embedded spec: one number when
        every permeable wall and direction shares it, ``None`` when no wall is permeable, and a dict
        ``{"wall_i.side": kappa}`` when they differ."""
        spec = self.substrate
        if spec is None:
            return None
        found = {}
        for i, w in enumerate(getattr(spec, "walls", ()) or ()):
            perm = getattr(w, "permeability", None)
            for side in ("in_to_out", "out_to_in"):
                v = getattr(perm, side, None) if perm is not None else None
                if v:
                    found[f"wall_{i}.{side}"] = float(v)
        if not found:
            return None
        vals = set(found.values())
        return found[next(iter(found))] if len(vals) == 1 else found

    def at_permeability(self, kappa):
        """This pack read at the wall permeability ``kappa`` (m/s): the view at ``a = kappa / permeability``,
        which is also the view at ``a`` times the walked diffusivity, since the same path is the walk at
        ``(a D, a kappa)`` and at no other pair (:meth:`at_diffusivity`). Needs a walk whose permeable walls
        share one permeability; a spec with several is read by its diffusivity instead."""
        k0 = self.permeability
        if k0 is None:
            raise ValueError("this pack's walls are impermeable (or it embeds no spec), so a permeability has no "
                             "walked value to be read against; give the diffusivity instead")
        if isinstance(k0, dict):
            raise ValueError(f"this pack's walls have different permeabilities, {k0}, which one number cannot "
                             f"rescale; read it by diffusivity (at_diffusivity), which scales them all together")
        return self.at_diffusivity(float(self.diffusivity) * float(kappa) / float(k0))

    def at_diffusivity(self, D):
        """This pack read at the bulk diffusivity ``D``: the same arrays on the save grid divided by
        ``a = D / diffusivity`` (dmipy-sim#289).

        THE PAIR, AND THE COMPUTE IT SAVES. Across a permeable wall the walk realised a crossing probability
        per encounter of about ``kappa sqrt(dt / D)``, which the rescale leaves unchanged only when the
        permeability scales with the diffusivity: the view is the walk at ``(a D, a kappa)`` -- the pair with
        the walked ratio ``kappa / D`` -- and at no other. So one walk at the SLOWEST diffusivity a study needs,
        over the LONGEST time, serves every faster setting on that line at its shorter time as a view, and a
        setting off the line needs its own walk. :attr:`~dmipy_sim.spec.Tissue.kappa` states the permeability
        wanted and :attr:`~dmipy_sim.spec.Tissue.D` the diffusivity; given both, they must sit on the line.

        A path walked at ``D0`` and read on a grid divided by ``a`` IS a path walked at ``a D0``: the stored
        coefficients are duration-agnostic, so nothing is decoded and every channel follows on the new grid in
        its own space -- the gradient's per-save weights, the relaxation gates, the surface term (whose weight
        divides by the new ``D``), the field's gate. This is what a change of the substrate's temperature does,
        about 2.5 per cent per kelvin for water. The same path is the walk at ``(a D0, a kappa)`` across a
        permeable wall and at no other pair, and the pair is recorded in ``provenance.diffusivity_scaled``.

        Only faster than walked is served (``a >= 1``). The certificate was measured at the walked grid: the
        split-half floor and the codec error are the same statistics on the same coefficients, but the in-step
        integration bias goes as ``D dt^2``, which is ``D0 dt0^2 / a`` -- it falls for ``a > 1`` and grows for
        ``a < 1``, past what the walk's save interval was chosen to keep below the floor. A published pack does
        not carry what would re-certify it, so a slower replay is refused rather than returned uncertified;
        walk the pack at the slower diffusivity. The acquisition must still fit the shortened walk, ``T / a``,
        which the replay checks as it checks every waveform.

        The result is an ordinary :class:`ReplayPack` sharing this one's arrays, so :meth:`prefix`, the pose
        cache and every route read it as they read any pack; :attr:`~dmipy_sim.spec.Tissue.D` on a replay
        resolves to it.
        """
        import copy
        D0 = self.diffusivity
        if D0 is None:
            raise ValueError("this pack records no diffusivity (walk_params.diffusivity), so a replay at another "
                             "one has no ratio to read the grid by")
        a = float(D) / float(D0)
        if not np.isfinite(a) or a <= 0.0:
            raise ValueError(f"a diffusivity is a positive number; got {D!r} against the walk's {D0!r}")
        if a == 1.0:
            return self
        if a < 1.0:
            raise ValueError(
                f"a replay at D = {float(D):.3g} m^2/s is slower than the walk's {float(D0):.3g} (a = {a:.3f}), "
                f"which stretches the save grid and grows the in-step integration bias as 1/a beyond what the "
                f"pack's certificate covers. Faster than walked is free; slower needs a walk at that "
                f"diffusivity, or a pack that recorded enough to re-certify a stretched grid, which this one did "
                f"not (dmipy-sim#289)")
        meta = copy.deepcopy(self.meta)
        wp = meta.setdefault("walk_params", {})
        wp["diffusivity"] = float(D)
        for k in ("dt", "dt_traj", "T_max"):
            if wp.get(k) is not None:
                wp[k] = float(wp[k]) / a
        if wp.get("segments"):
            wp["segments"] = dict(wp["segments"], T=float(wp["segments"]["T"]) / a)
        if meta.get("dt") is not None:
            meta["dt"] = float(meta["dt"]) / a
        cx = meta.get("compression") or {}
        if cx.get("temporal_bandwidth_hz") is not None:
            cx["temporal_bandwidth_hz"] = float(cx["temporal_bandwidth_hz"]) * a
        pm = (cx.get("channels") or {}).get("susceptibility_path")
        if pm is not None and pm.get("dt") is not None:
            pm["dt"] = float(pm["dt"]) / a                   # the path channel's own grid follows too
        # the view's situation IS the walk at (a D, a kappa): the embedded spec says so, wall by wall and pool by
        # pool, so that the view's own permeability and diffusivity read as what it realises
        kappa = {}
        sd = meta.get("substrate")
        if isinstance(sd, dict):
            for i, w in enumerate(sd.get("walls") or []):
                perm = w.get("permeability") if isinstance(w, dict) else None
                if isinstance(perm, dict):
                    for side in ("in_to_out", "out_to_in"):
                        v = perm.get(side)
                        if v:
                            kappa[f"wall_{i}.{side}"] = {"walked": float(v), "replayed": float(v) * a}
                            perm[side] = float(v) * a
            for pool in sd.get("pools") or []:
                if isinstance(pool, dict):
                    for key in ("diffusivity", "D"):
                        if isinstance(pool.get(key), (int, float)) and pool[key]:
                            pool[key] = float(pool[key]) * a
        prov = meta.setdefault("provenance", {})
        prov["diffusivity_scaled"] = dict(walked=float(D0), replayed=float(D), a=a,
                                          grid_dt=dict(walked=float(self.dt), replayed=float(self.dt) / a),
                                          permeability=kappa,
                                          certificate="measured at the walked grid; the in-step bias falls as 1/a")
        out = ReplayPack(self.arrays, meta, source=self.source)
        return out

    def _at_tissue(self, tissue):
        """The pack a replay with ``tissue`` reads: this one, or its view at the tissue's diffusivity and
        permeability. Both stated, they must sit on the walk's line ``kappa / D = kappa_walk / D_walk``."""
        D = getattr(tissue, "D", None)
        kappa = getattr(tissue, "kappa", None)
        if kappa is None and (D is None or self.diffusivity is None or float(D) == float(self.diffusivity)):
            return self
        if kappa is None:
            return self.at_diffusivity(D)
        k0 = self.permeability
        if k0 is None or isinstance(k0, dict):
            raise ValueError(
                "a tissue states a wall permeability, and this pack "
                + ("has no permeable wall to read it against" if k0 is None else f"has walls of different permeabilities {k0}")
                + ": a replay cannot make a wall permeable or change one wall alone; walk the spec wanted (dmipy-sim#289)")
        a_k = float(kappa) / float(k0)
        if D is not None and self.diffusivity is not None:
            a_d = float(D) / float(self.diffusivity)
            if abs(a_d - a_k) > 1e-9 * max(a_d, a_k):
                raise ValueError(
                    f"the pair (D = {float(D):.3g} m^2/s, kappa = {float(kappa):.3g} m/s) is not on this walk's line: "
                    f"the walk realised kappa / D = {float(k0) / float(self.diffusivity):.3g} s/m^2 and a rescale "
                    f"keeps that ratio, so it serves (a D_walk, a kappa_walk) for any a >= 1 and no other pair. "
                    f"D asks for a = {a_d:.4g}, kappa for a = {a_k:.4g}; a setting off the line needs its own "
                    f"walk (dmipy-sim#289)")
        return self.at_permeability(kappa)

    @property
    def nominal(self):
        """The embedded spec's values as a :class:`~dmipy_sim.spec.Tissue`: what a paper's replay applies,
        ``replay(seq, tissue=pack.nominal, scanner=pack.nominal_field_T)``; ``None`` for a pack without a spec."""
        spec = self.substrate
        if spec is None:
            return None
        from ..spec.tissue import Tissue
        return Tissue.from_spec(spec)

    @property
    def nominal_field_T(self):
        """The spec's calibration field (T), the ``scanner`` of the nominal replay; ``None`` when it names none."""
        spec = self.substrate
        return None if spec is None else spec.nominal_field_T

    def positions(self):
        """The ``(n_walkers, n_t, 3)`` trajectory decoded from the position codec (float64); a pack of several
        segments decodes each window and joins them on the shared saves."""
        from .compression import decode, is_walker_preserving, require_position_method
        if self.n_segments > 1:
            return np.concatenate([w.positions()[:, (0 if j == 0 else 1):] for j, (w, _, _) in enumerate(self._windows())], axis=1)
        cx = self.meta.get("compression", {})
        meta = {"method": require_position_method(cx.get("method")), "K": int(cx.get("K", 0)),
                "n_t": int(cx.get("n_t") or self.n_t)}
        wp = is_walker_preserving(meta["method"])
        return np.asarray(decode(self.arrays, meta, n_walkers=(self.n_walkers if wp else None)), np.float64)

    def _by_pool(self, values, what, n=None):
        """Per-pool values as a list by id; a ``{name: value}`` dict resolves through the embedded spec, and a
        scalar is every pool's value (``n`` pools, else the spec's). A list is by id and must cover every id."""
        if values is None:
            return None
        if isinstance(values, dict):
            spec = self.substrate
            if spec is None:
                raise ValueError(f"{what} given by pool name but the pack embeds no substrate spec; give a list by id")
            out = [0.0] * len(spec.pools)
            for name, v in values.items():
                out[spec.pool(name).id] = float(v)
            return out
        if np.ndim(values) == 0:                                   # one value is every pool's
            spec = self.substrate
            return [float(values)] * (int(n) if n is not None else (len(spec.pools) if spec is not None else 1))
        return [float(v) for v in np.asarray(values, float).reshape(-1)]

    def replay(self, waveform, *, tissue=None, scanner=None, orientation=None, compartment=None, complex_signal=False):
        """The signal of ``waveform`` on this pack: the gradient always, and every other tier whose inputs are given
        and which the pack carries.

        ``waveform`` is a :class:`~dmipy_sim.acquisition.scanner_sequence.ScannerSequence`: its ``G_eff``, the
        effective gradient, is what this route integrates, and its RF schedule says where the 180 is (a schedule
        without one is a gradient echo). It is read on the pack grid (``n_t`` samples of ``dt``, zero outside
        the waveform).

        Three things describe a replay setting, each stated once:

        * ``tissue`` -- **what the material is**: a :class:`~dmipy_sim.spec.Tissue` (pool T2 / T1, the walls'
          rho, the bulk D, the field source's chi) or ``None``, the bare diffusion signal. ``pack.nominal`` is
          the embedded spec's values, so a paper's replay is ``replay(seq, tissue=pack.nominal,
          scanner=pack.nominal_field_T)``; ``pack.nominal.replace(T2={"intra": 0.08})`` changes one.
        * ``orientation`` -- **where the substrate sits**: its pose. **One pose** -- a 3x3 rotation (substrate
          frame -> lab), or the lab direction its axis (``spec.frame.axis``, default z) points along -- is exact by
          pose covariance: the gradient and the field are rotated into the substrate frame together, and no
          expansion is involved. **A distribution of poses** -- a :class:`~dmipy_sim.replay.so3.Distribution`
          (one pose, an axis density, a Watson cone, a Bingham fan) or an :class:`~dmipy_sim.replay.fod.FOD`,
          which is read as an axis density with no statement about the substrate's own azimuth -- goes through
          :meth:`pose_response` and is composed on SO(3). An FOD's basis must be declared (``FOD.from_sh``,
          ``FOD.native``); a bare coefficient array is refused, since the convention cannot be inferred.
        * ``scanner`` -- **what the scanner is**: its static field, as a
          :class:`~dmipy_sim.acquisition.scanners.ScannerLimits` (the catalogue's ``field_T``) or a number in
          tesla, or ``None`` for no field. Its DIRECTION comes from the machine too -- a bi-planar
          magnet's field runs across the patient, not along the bore -- and the pose turns it.

        The tiers follow from those: **gradient** (C0) always, in mode space from the position coefficients;
        **bulk relaxation** (C1) with a T2 / T1 in the tissue, under the waveform's coherence gate, on the
        occupancy channel; **surface relaxivity** (C2) with a rho, scaled by the walk's D (the tissue's, else the
        pack's recorded one), on the boundary local time; **field** (C3) with a chi in the tissue and a field on
        the scanner, on the path channel (or the stored basis sampled along the decoded path), the 180 the
        waveform's own. A tier whose inputs are given but which the pack does not carry raises rather than
        returning a plausible number.

        ``compartment`` restricts the ensemble mean to one pool id (or a boolean walker mask).
        """
        from .compression import read_position_coeffs
        from ._replay_kernel import gradient_phase, field_gate
        waveform = waveform.waveform if hasattr(waveform, "waveform") else waveform
        if isinstance(waveform, Protocol):                # a multi-TE scheme: each sequence replayed, placed at its rows
            kw = dict(tissue=tissue, scanner=scanner, orientation=orientation, compartment=compartment, complex_signal=complex_signal)
            return waveform.scatter([self.replay(seq, **kw) for seq in waveform])
        view = self._at_tissue(tissue)
        if view is not self:
            return view.replay(waveform, tissue=tissue, scanner=scanner, orientation=orientation,
                               compartment=compartment, complex_signal=complex_signal)
        dist = _as_distribution(orientation)
        if dist is not None:
            S = self.pose_response(waveform, tissue=tissue, scanner=scanner, compartment=compartment).compose(dist)
            return S if complex_signal else np.abs(S)
        P = self._prepare(waveform, tissue=tissue, scanner=scanner, orientation=orientation, compartment=compartment)
        phi = self._walker_phases(P, waveform)
        S = P["pathway"] * P["voxel"] * (P["ew"][:, None] * np.exp(1j * phi)).sum(0) / P["norm"]
        return S if complex_signal else np.abs(S)

    def walker_signals(self, waveform, *, tissue=None, scanner=None, orientation=None, compartment=None,
                       b1_scale=None, off_resonance_T=None):
        """The replay **before the ensemble mean**: ``(w, ew, E)`` with ``w`` the walkers' statistical weights
        ``(n_w,)``, ``ew`` those weights with the relaxation and surface terms applied, and ``E`` the complex
        signal factor of every walker at every measurement, ``(n_w, n_meas)``.

        :meth:`replay` is ``(ew[:, None] * E).sum(0) / w.sum()``; any other grouping of the walkers -- by the
        voxel they started in, which is what partitions one walk into a phantom (RPH.md) -- is the same sum
        over its members with its own normaliser. Knobs resolve as in :meth:`replay`. An unbalanced encoding's
        voxel factor (:meth:`ScannerSequence.voxel_factor`) is carried in ``E`` as a per-measurement scale,
        so the sum above is the voxel's signal. With ``b1_scale`` (a scalar or per walker) or
        ``off_resonance_T`` (a scalar or per walker) given, ``E`` comes from the RF-aware route
        (:meth:`replay_bloch`): the transverse magnetisation of each walker at the readout, with the relaxation
        the route applied itself, so ``ew`` is then ``w``.
        """
        waveform = waveform.waveform if hasattr(waveform, "waveform") else waveform
        view = self._at_tissue(tissue)
        if view is not self:
            return view.walker_signals(waveform, tissue=tissue, scanner=scanner, orientation=orientation,
                                       compartment=compartment, b1_scale=b1_scale, off_resonance_T=off_resonance_T)
        if b1_scale is None and off_resonance_T is None:
            P = self._prepare(waveform, tissue=tissue, scanner=scanner, orientation=orientation, compartment=compartment)
            w = np.asarray(self.spin_weights, np.float64)
            E = np.exp(1j * self._walker_phases(P, waveform)) * P["voxel"][None, :]
            return w, P["pathway"] * P["ew"], E
        E = self.replay_bloch(waveform, b1_scale=b1_scale, off_resonance_T=off_resonance_T, tissue=tissue, scanner=scanner,
                              orientation=orientation, compartment=compartment, complex_signal=True, per_walker=True)
        w = np.asarray(self.spin_weights, np.float64)
        return w, w, E

    def walker_phases(self, waveform, *, tissue=None, scanner=None, orientation=None, compartment=None,
                      weights=None):
        """:meth:`walker_signals` before the complex exponential: ``(w, ew, phi)`` with ``phi`` the accumulated
        phase of every walker at every measurement, ``(n_w, n_meas)`` real, so that ``E = exp(1j * phi)``. A
        consumer that reduces many walkers over its own groups (an image: the walkers of each voxel) takes the
        phase and forms the exponential and the sums where it accumulates them, on its device; the exponential
        over ``(n_w, n_meas)`` is the one host operation of a replay that does not amortise. Knobs resolve as in
        :meth:`replay`; the RF-aware route has no phase (its signal is the magnetisation vector itself) and is
        :meth:`walker_signals` with ``b1_scale`` or ``off_resonance_T``. An unbalanced encoding is refused
        here: its voxel factor is an amplitude per measurement, which a phase cannot carry, and a consumer
        that formed its own sums would silently omit it -- :meth:`walker_signals` carries it in ``E``."""
        waveform = waveform.waveform if hasattr(waveform, "waveform") else waveform
        view = self._at_tissue(tissue)
        if view is not self:
            return view.walker_phases(waveform, tissue=tissue, scanner=scanner, orientation=orientation,
                                      compartment=compartment, weights=weights)
        P = self._prepare(waveform, tissue=tissue, scanner=scanner, orientation=orientation, compartment=compartment)
        if np.any(P["voxel"] != 1.0):
            raise ValueError(
                "this encoding leaves a net moment at the readout, so the voxel's extent multiplies the signal by "
                "a factor per measurement (ScannerSequence.voxel_factor). A phase cannot carry an amplitude, and a "
                "consumer forming its own sums from these phases would omit it silently; use walker_signals, "
                "whose E carries it, or replay")
        if weights is not None:
            P["W"] = self._check_weights(weights, P, orientation)
        w = np.asarray(self.spin_weights, np.float64)
        return w, P["pathway"] * P["ew"], self._walker_phases(P, waveform)

    def walker_primitives(self, acquisition):
        """What ``acquisition`` (a :class:`~dmipy_sim.replay.study.Acquisition`, or a sequence) leaves of every
        walker before any tissue or scanner is applied -- the bands and the path channel contracted once, the
        exposures and the contact read once: a :class:`~dmipy_sim.replay.study.Primitives`, whose ``signals(tissue,
        scanner)`` is :meth:`walker_signals` for every pair without touching the bands again (dmipy-sim#297)."""
        from .study import walker_primitives
        return walker_primitives(self, acquisition)

    def study(self, study):
        """The signal for every pair of a :class:`~dmipy_sim.replay.study.Study`, ``(pairs, n_meas)``: the
        protocol's acquisitions contracted once each, every tissue and scanner applied elementwise."""
        from .study import study_signals
        return study_signals(self, study)

    def _check_weights(self, weights, P, orientation):
        """Validate a caller's per-position replay weights, and refuse the cases this route cannot serve.

        Shape alone is not enough to make weights the right ones. It is identical for every save grid once
        ``K`` and ``n_meas`` match, so weights built on the SEQUENCE's grid rather than the pack's pass a
        shape test and encode a 45 per cent error in b. The dtype matters too: an integer array is silently
        truncated and a complex one silently loses its imaginary part.
        """
        if len(P["windows"]) > 1:
            raise ValueError("weights= are the per-position replay weights of ONE window, and this pack stores its walk in "
                             f"{len(P['windows'])} segments; give the waveform and let the replay read every window")
        W = np.asarray(weights)
        if not np.issubdtype(W.dtype, np.floating):
            raise ValueError(
                f"weights must be real floating point; got {W.dtype}. An integer array is truncated and a "
                f"complex one loses its imaginary part, both silently")
        W = W.astype(np.float64)
        want = (self.n_coeffs * 3, P["Geff"].shape[0])
        if W.shape != want:
            raise ValueError(
                f"weights are this pack's replay weights for ONE position, {want} -- (n_coeffs * 3, "
                f"n_meas). Got {W.shape}. dmipy_sim.phantom.bore.delivered_weights builds them per voxel, "
                f"and must be given THIS pack's save grid: K={self.K}, n_t={self.n_t}, dt={self.dt!r}")
        if not np.all(np.isfinite(W)):
            raise ValueError("weights are not finite, so the phase they produce is not a signal")
        if orientation is not None:
            raise ValueError(
                "weights= and orientation= cannot both be given. A pose is carried by ROTATING the gradient "
                "before it is projected, and supplied weights replace that projection wholesale -- so the "
                "pose would be silently discarded (measured: 48 per cent of the signal). Build the weights "
                "for the posed gradient instead, which reproduces the pose exactly")
        if self._field_active(P["B0"]):
            raise ValueError(
                "weights= cannot be combined with an active susceptibility field yet. The field branches of "
                "this contraction rebuild the gradient from the nominal sequence and do not read supplied "
                "weights, so the result would be bit-identical to passing none -- including for weights of "
                "zero. That is silently wrong in exactly the low-field case this exists for (dmipy-sim#369)")
        return W

    def _walker_phases(self, P, waveform):
        """``(n_w, n_meas)`` accumulated phase of every walker under the prepared acquisition ``P``: the gradient
        as the bridge coefficients against the effective gradient's projection, and with a field the path channel's
        cosine modes against the gate's DCT (the grid route, a pack without the path channel, samples the field
        along the decoded path)."""
        from .compression import read_position_coeffs
        from ._replay_kernel import gradient_phase, field_gate, effective_gradient
        n_w, dt, n_t, Geff = P["n_w"], P["dt"], P["n_t"], P["Geff"]
        windows = P["windows"]
        if self.n_segments > 1:                                                      # the windows' phases sum (RPK.md 4.3)
            phi = None
            for seg, t0, n_s in windows:
                P_s = dict(P, n_t=n_s, Geff=effective_gradient(P["G_eff_wf"], P["dt_wf"], n_s, dt, t0=t0), t0=t0, W=None,
                           windows=[(seg, t0, n_s)])
                phi_s = seg._walker_phases(P_s, waveform)
                phi = phi_s if phi is None else phi + phi_s
            return phi
        t0 = P.get("t0")
        if not self._field_active(P["B0"]):                                          # no field, or a field of zero
            C = read_position_coeffs(self.arrays, dtype=np.float64)
            W = P.get("W")
            if W is None:
                W = _compile_effective(Geff, dt, self.K, n_t)
            phi = C.reshape(n_w, self.n_coeffs * 3) @ W                              # (n_w, n_meas)
        else:
            from .bank import susc_path_decode, susc_path_field
            from ..fields.susceptibility_field import assemble_field, sample_grid
            ch, b0_dir, B0, chi_aniso = P["ch"], P["b0_dir"], P["B0"], P["chi_aniso"]
            gm = ch["susceptibility_grid"]
            if P["chi_iso"] is None:
                raise ValueError("a scanner field was given without a chi_iso in the tissue: the pack carries the substrate's field basis, "
                                 "not a susceptibility; give a tissue with chi_iso (and chi_aniso)")
            chi_i = float(P["chi_iso"])
            pm = ch.get("susceptibility_path")
            if pm is not None:                                                        # the path route: every term a contraction
                from .bank import path_field_integral
                from ..fields.hollow_cylinder import contract
                C = read_position_coeffs(self.arrays, dtype=np.float64)
                phi = C.reshape(n_w, self.n_coeffs * 3) @ _compile_effective(Geff, dt, self.K, n_t)     # the gradient, as without a field
                Psi, names = path_field_integral(self.arrays, pm, waveform, n_t, dt, t0=t0, n_w=n_w)
                aniso = chi_aniso if (bool(gm.get("has_aniso")) and chi_aniso and "aniso_G_xx" in names) else 0.0
                phi_x = contract(Psi, b0_dir, B0=float(B0), chi_iso=chi_i, chi_aniso=aniso)
                return phi + phi_x[:, None]
            pos = self.positions()                                                    # the grid route samples the field along the path
            if True:
                from .bank import grid_basis_of
                dB = sample_grid(assemble_field(grid_basis_of(self.arrays, gm), b0_dir, B0=float(B0), chi_iso=chi_i, chi_aniso=chi_aniso),
                                 pos, np.asarray(gm["origin"], float), gm["voxel_size"], periodic=False)
            phi_x = GAMMA * dt * (dB * field_gate(waveform, n_t, dt, t0=t0)[None, :]).sum(1)    # (n_w,)
            phi = gradient_phase(Geff, pos, dt).T + phi_x[:, None]                             # (n_w, n_meas)
        return phi

    def replay_bloch(self, waveform, *, b1_scale=None, off_resonance_T=None, tissue=None, scanner=None,
                     orientation=None, compartment=None, jax=False, complex_signal=False, per_walker=False,
                     crusher_seed=0):
        """The RF-aware replay: each walker's magnetisation vector propagated through the actual sequence
        operators on this pack's walk (:func:`~dmipy_sim.replay.trajectories.replay_bloch`).

        This route reads the waveform's PHYSICAL ``G`` and applies its pulses; :meth:`replay` reads ``G_eff``
        with the pulses folded in. The magnitude route of :meth:`replay` assumes ideal pulses and reads the
        signal as a phase sum, so it cannot carry anything that acts on the magnetisation vector: a flip angle that is not nominal
        (``b1_scale``), a finite pulse, a pulse train's coherence pathways. Those are what this route is for,
        and it costs a propagation per piece instead of a projection.

        Knobs are the same as :meth:`replay` and resolve the same way; the pose rotates the
        acquisition and the field direction as it does there. ``b1_scale`` scales every flip angle, as a scalar
        or per walker. ``off_resonance_T`` is a **uniform** static field offset (a field-map value, in T) every
        walker precesses in through the actual pulses -- what a macroscopic layer of a phantom is (RPH.md 5.1).
        ``per_walker`` returns every walker's transverse magnetisation at the readout, ``(n_w, n_meas)`` complex,
        unweighted, instead of the ensemble mean (single-readout sequences).

        A sequence's ``crusher`` is applied here (dmipy-sim#305), and so is the voxel itself for an encoding
        that leaves a net moment at the readout (dmipy-sim#375): each walker is placed at its own drawn
        offset in the prescribed voxel, which is what a played spoiler winds across; a declared ``crusher`` is
        for a winding the waveform does not play in ``G``. The declared one is the voxel-scale spoiler, which a
        micron cell cannot produce geometrically -- a gradient cannot wind much beyond 2 pi across it -- so it
        is modelled as the forward engine models it (:func:`~dmipy_sim.engine.bloch._build_crusher`): each
        walker carries a macroscopic coordinate ``u`` in [0, 1) and accrues ``2 pi n_cycles u`` over each
        crusher window, so the ensemble dephases over ``n_cycles`` turns. Without it every coherence pathway
        of a pulse train stays degenerate and recombines, and the echo barely depends on the refocusing flip
        angle, which is not what a train does. ``crusher_seed`` draws those coordinates, so a replay of one
        pack and one sequence is reproducible.
        """
        view = self._at_tissue(tissue)
        if view is not self:
            return view.replay_bloch(waveform, b1_scale=b1_scale, off_resonance_T=off_resonance_T, tissue=tissue, scanner=scanner, orientation=orientation, compartment=compartment, jax=jax, complex_signal=complex_signal, per_walker=per_walker, crusher_seed=crusher_seed)
        from .trajectories import replay_bloch as _rb, replay_bloch_jax as _rbj, _bloch_timeline
        from .compression import decode_occupancy, decode_boundary_bridge
        from ._replay_kernel import gate_weights
        P = self._prepare(waveform, tissue=tissue, scanner=scanner, orientation=orientation, compartment=compartment,
                          relaxation=False, surface=False, pathway=False)
        rf = waveform.rf
        if not rf:
            raise ValueError("the Bloch route replays an RF schedule and this sequence carries none. Without a "
                             "pulse there is nothing this route adds over replay().")
        ch, dt, n_t = P["ch"], P["dt"], P["n_t"]
        _bloch_timeline(rf, n_t, dt)                                    # the whole schedule against the whole walk, once
        # the readouts on the pack grid: the sequence's echoes, else its last sample -- the acquisition's own end,
        # never the walk's, so a shorter acquisition on a longer pack is read at its readout
        echoes = _echo_saves(waveform, dt)
        single = echoes is None
        if single:
            echoes = [int(round((int(waveform.n_t) - 1) * float(waveform.dt) / dt))]
        if per_walker and not single:
            raise ValueError("per_walker reads each walker at the readout of a single-echo sequence")
        T2v, T1v = P["T2"], P["T1"]
        relax = None
        if T2v is not None or T1v is not None:
            col = next(d for d in ch["compartment"]["columns"] if d["name"] == "comp")
            n_ids = 2 if col["kind"] == "fraction" else self._n_pool_ids(col)   # a fractional occupancy is two pools

            # the Bloch route reads a rate as 1/T, so "no decay in this pool" is an infinite time, not a zero
            # one; a zero would make the rate infinite and return an identically dark signal
            def per_pool(v, what):
                out = self._by_pool(v, what, n=n_ids)
                if out is None:
                    return [np.inf] * n_ids
                return [np.inf if t is None or float(t) <= 0.0 else float(t) for t in out]
            relax = dict(T2_per_comp=per_pool(T2v, "T2"), T1_per_comp=per_pool(T1v, "T1"))
        surface = None
        if P["rho"] is not None and float(P["rho"]) != 0.0:
            D_walk = self.diffusivity if P["D"] is None else P["D"]
            if D_walk is None:
                raise ValueError("rho needs the walk's diffusivity: the pack did not record it, pass D=")
            if not self.has_surface:
                raise ValueError("surface relaxivity was requested but this pack carries no C2 channel")
            surface = dict(surface_relaxivity=float(P["rho"]), D=float(D_walk))
        n_w = P["n_w"]
        off = None
        if off_resonance_T is not None and np.any(np.asarray(off_resonance_T, np.float64) != 0.0):
            off = np.asarray(off_resonance_T, np.float64).reshape(-1)
            if off.size not in (1, n_w):
                raise ValueError(f"off_resonance_T is a scalar or one value per walker ({n_w}); got {off.shape}")
        per_save_all = None
        if getattr(waveform, "crusher", None) is not None:
            from ..engine.bloch import _build_crusher
            rate, has = _build_crusher(waveform.crusher, P["dt_wf"], P["G"].shape[1])   # rad/step, waveform grid
            if has and np.any(rate):
                # the macroscopic coordinate is STRATIFIED over the ensemble, (i + 1/2) / n_w dealt to the walkers
                # by the seed: a declared winding of a whole number of turns then cancels the crushed pathways
                # exactly, where n_w uniform draws leave them at 1/sqrt(n_w) (dmipy-sim#393: a 0.2 % offset on a
                # crushed spin echo against the enumeration)
                # the rate as a density (rad/s) carried onto the pack's save grid, then back to radians per save: the
                # phase of each window is preserved however the two grids differ. The propagator reads these as SAMPLES
                # at the saves (a sampled rate, integrated with the path interpolant), so a window takes the whole
                # grid's samples sliced, never a window's own weights, whose shared save would be halved
                per_save_all = np.asarray(gate_weights(rate / P["dt_wf"], P["dt_wf"], n_t, dt), np.float64).reshape(-1, n_t)[0] * dt
                u = np.random.default_rng(int(crusher_seed)).permutation((np.arange(n_w) + 0.5) / n_w)
        r_v = None
        if waveform.unbalanced and not waveform.voxel_declared:
            # dmipy-sim#375: an encoding that leaves a net moment at the readout winds across the VOXEL, and a
            # micron-scale substrate cannot. Each walker is put at its own drawn place in the voxel -- the walk
            # translated by r_v, which adds exactly gamma int G . r_v dt to its phase, per measurement, with the
            # RF applied by the propagator as to every other phase. The scalar routes take the same average
            # analytically (ScannerSequence.voxel_factor); this is it per walker. The voxel is the scanner's, so
            # r_v is drawn in the lab and turned into the stored frame with the gradient.
            L = np.asarray(waveform.prescription.voxel_size_m, np.float64)     # _prepare refused without one
            r_v = (np.random.default_rng(int(crusher_seed) + 1).random((n_w, 3)) - 0.5) * L
            if orientation is not None:
                r_v = r_v @ np.asarray(self.pose_rotation(orientation), np.float64)
        # the windows of the walk in turn (RPK.md 4.3): each propagated from the state the previous one left every
        # walker in, its own positions decoded and its own channels read, the readouts it holds recorded
        M = None; recs = []
        for seg, t0, n_s in P["windows"]:
            k0 = int(round(t0 / dt))
            pos = seg.positions()
            local = [e - k0 for e in echoes if (k0 < e <= k0 + n_s - 1) or (k0 == 0 and e == 0)]
            kw = dict(weights=P["ew"] / P["norm"], t0=t0, M_init=M, return_state=True,
                      echo_steps=(local or None), echo_per_walker=bool(per_walker and local))
            if b1_scale is not None:
                kw["b1_scale"] = b1_scale
            if relax is not None:
                kw.update(comp_traj=decode_occupancy(seg.arrays, ch["compartment"])["comp"], **relax)
            if surface is not None:
                meta = dict(ch.get("boundary_local_time") or {})
                meta.setdefault("n_t", n_s)
                meta.setdefault("K", _cx_bands_K(seg.arrays, meta))
                kw.update(dlog_boundary_unit=decode_boundary_bridge(seg.arrays, meta), **surface)
            extra = None
            if P["B0"] is not None:
                extra = GAMMA * dt * seg._field_along_walk(P, pos)
            if off is not None:
                uniform = GAMMA * dt * np.broadcast_to(off[:, None], (n_w, n_s))
                extra = uniform if extra is None else extra + uniform
            if per_save_all is not None:
                crush = u[:, None] * np.broadcast_to(per_save_all[k0:k0 + n_s], (n_w, n_s))
                extra = crush if extra is None else extra + crush
            if extra is not None:
                kw["extra_phase_per_step"] = extra
            if r_v is not None:
                pos = pos + r_v[:, None, :]
            M, out = (_rbj if jax else _rb)(pos, dt, P["G"], P["dt_wf"], rf, **kw)
            if local:
                recs.append(np.asarray(out))
        E = np.concatenate(recs, axis=1)                                          # (n_meas, n_echo[, n_w])
        if per_walker:
            E = E[:, 0, :].T                                                     # (n_w, n_meas)
            return E if complex_signal else np.abs(E)
        S = E[:, 0] if single else E
        return S if complex_signal else np.abs(S)

    def _field_along_walk(self, P, pos):
        """The susceptibility off-resonance each walker sees at each save, from whichever C3 route the pack
        carries: the compressed path coefficients, or the stored field basis sampled along the walk."""
        from .bank import susc_path_decode, susc_path_field
        from ..fields.susceptibility_field import assemble_field, sample_grid
        if not self._field_active(P["B0"]):
            return np.zeros((P["n_w"], self.n_t))                                     # a declared zero field
        if P["chi_iso"] is None:
            raise ValueError("a scanner field was given without a chi_iso in the tissue: the pack carries the substrate's field basis, "
                             "not a susceptibility; give a tissue with chi_iso (and chi_aniso)")
        gm = P["ch"]["susceptibility_grid"]
        pm = P["ch"].get("susceptibility_path")
        if pm is not None:
            b, _ = susc_path_decode(self.arrays, pm, n_w=P["n_w"])
            return susc_path_field(b, P["b0_dir"], B0=float(P["B0"]), chi_iso=float(P["chi_iso"]),
                                   chi_aniso=P["chi_aniso"], has_aniso=bool(gm.get("has_aniso")))
        from .bank import grid_basis_of
        return sample_grid(assemble_field(grid_basis_of(self.arrays, gm), P["b0_dir"], B0=float(P["B0"]), chi_iso=float(P["chi_iso"]),
                                          chi_aniso=P["chi_aniso"]),
                           pos, np.asarray(gm["origin"], float), gm["voxel_size"], periodic=False)

    def _prepare(self, waveform, *, tissue, scanner, orientation, compartment, relaxation=True, surface=True,
                 pathway=True):
        """Everything a replay resolves before it reads positions: the waveform's exact per-save weights (rotated
        into the substrate frame when a pose is given), the tissue's values (none for ``None``), the scanner's
        field along the machine's own B0 axis turned by the pose, the per-walker weights with the relaxation and surface terms
        applied, and the compartment selection.

``pathway`` asks for the amplitude of the coherence pathway the sequence's readout IS
        (:func:`~dmipy_sim.acquisition.epg.pathway_weight`): 1 for a refocused echo, and a stimulated echo's
        ``0.5 sin a1 sin a2 sin a3`` for a store-and-recall schedule. It is returned BESIDE the weights and
        not folded into them, because ``ew`` means the relaxation and surface terms and a codec oracle checks
        it means only that; each route that forms a signal applies it once. The vector-Bloch route does not:
        it propagates the magnetisation through the actual pulses, so that amplitude is already in its
        answer, and it passes ``pathway=False``. That is not only to avoid applying the factor twice: a
        readout no SINGLE amplitude describes -- a train at any flip but 180 -- is refused by
        :func:`~dmipy_sim.acquisition.epg.pathway_weight`, and the vector route is precisely the one that
        does not need one, since it carries every pathway itself."""
        from .compression import require_position_method, decode_occupancy, relaxation_logweight
        from ._replay_kernel import effective_gradient, bin_gate
        require_position_method(self.method)
        # two gradients: the PHYSICAL one (``G``, for the vector-Bloch route, which applies the pulses itself)
        # and the EFFECTIVE one (``G_eff``, for the scalar routes, the pulses folded in)
        waveform = waveform.waveform if hasattr(waveform, "waveform") else waveform
        if not isinstance(waveform, ScannerSequence):
            raise TypeError(f"a replay takes a ScannerSequence (a bare gradient array says nothing about its pulses); "
                            f"got {type(waveform).__name__}")
        G = np.asarray(waveform.G, np.float64)
        G_eff = np.asarray(waveform.G_eff, np.float64)
        dt_wf = float(waveform.dt)
        n_t, dt = self.n_t, self.dt
        chi = waveform.chi_perp                          # the gate on the occupancy / contact channels: the sequence's
        if chi is None:                                  # coherence gate, or 1 over the sequence -- and 0 after its
            chi = np.ones(G.shape[1])                    # echo, so a short acquisition relaxes to ITS echo, never to
        chi = np.asarray(chi, np.float64).reshape(-1)    # the end of a longer walk (the TE-prefix property, #199)
        on_wf = chi.shape[0] == G.shape[1]
        active = bin_gate(np.ones(chi.shape[0]), dt_wf if on_wf else dt, n_t, dt)[0]  # the acquisition's own extent
        chi = bin_gate(chi, dt_wf if on_wf else dt, n_t, dt)[0]                       # averaged over each save's
        ch = (self.meta.get("compression", {}).get("channels", {}) or {})
        n_w = self.n_walkers
        w = np.asarray(self.spin_weights, np.float64)
        from ..spec.tissue import Tissue
        if tissue is not None and not isinstance(tissue, Tissue):
            raise TypeError(f"tissue is a Tissue (pack.nominal, Tissue(...)) or None for the bare diffusion signal; "
                            f"got {type(tissue).__name__}")
        t = tissue if tissue is not None else Tissue()
        T2, T1, rho, D, chi_iso, chi_aniso = t.T2, t.T1, t.rho, t.D, t.chi_iso, t.chi_aniso
        field = scanner_field(scanner)
        B0, b0_dir = field.B0, field.axis                            # the MACHINE's field; the pose turns it
        if orientation is not None:
            R = self.pose_rotation(orientation)
            G, G_eff = G @ R, G_eff @ R                                   # R^T g per sample: stored coordinates
            b0_dir = tuple(np.asarray(R, float).T @ np.asarray(b0_dir, float))
        Geff = effective_gradient(G_eff, dt_wf, n_t, dt)                 # exact per-save weights of the effective gradient
        # the windows of the walk (RPK.md 4.3): every tier is a sum over them, each window reading the acquisition
        # from where it sits on the walk's clock; a single window is the walk itself
        # only the windows the acquisition reaches are read: a window whose first save sits at or beyond the readout
        # contributes nothing, and is not touched -- a short acquisition on a long pack reads its first windows alone
        T_acq = (G.shape[1] - 1) * dt_wf
        windows = [w for w in self._windows() if w[1] < T_acq * (1.0 - 1e-12)] or self._windows()[:1]
        dt_chi = dt_wf if on_wf else dt
        chi_wf = np.asarray(waveform.chi_perp if waveform.chi_perp is not None else np.ones(G.shape[1]), np.float64).reshape(-1)
        window_gates = [(bin_gate(chi_wf, dt_chi, n_s, dt, t0=t0)[0], bin_gate(np.ones(chi_wf.shape[0]), dt_chi, n_s, dt, t0=t0)[0])
                        for _, t0, n_s in windows]
        logw = np.zeros(n_w)
        if (T2 is not None or T1 is not None) and relaxation:
            if not self.has_relaxation:
                raise ValueError("T2 / T1 were given but the pack carries no compartment channel (C1); build it "
                                 "from a walk with tiers='all'")
            from .compression import relaxation_logweight_runs, is_current_c1
            if not is_current_c1(ch["compartment"]):
                decode_occupancy(self.arrays, ch["compartment"])              # raises with the re-encode message
            col = next(d for d in ch["compartment"]["columns"] if d["name"] == "comp")
            n_ids = self._n_pool_ids(col)
            T2v = self._by_pool(T2, "T2", n=n_ids); T1v = self._by_pool(T1, "T1", n=n_ids)
            if T2v is None:
                T2v = [0.0] * n_ids                                   # no T2 decay, T1 only
            if T1v is None:
                T1v = [0.0] * n_ids                                   # no T1 term
            if len(T2v) < n_ids or (T1v is not None and len(T1v) < n_ids):
                raise ValueError(f"the compartment channel uses pool ids up to {n_ids - 1}; T2 / T1 must be given "
                                 f"for every id (got {len(T2v)}{'' if T1v is None else f' / {len(T1v)}'})")
            for (seg, _, _), (chi_s, act_s) in zip(windows, window_gates):
                logw = logw + relaxation_logweight_runs(seg.arrays, col, T2v, T1v, dt, chi_s, act_s)   # on the runs, never a track
        if rho is not None and float(rho) != 0.0 and surface:
            D_walk = self.diffusivity if D is None else D
            if D_walk is None:
                raise ValueError("rho needs the walk's diffusivity: the pack did not record it, pass D=")
            for (seg, _, _), (chi_s, _) in zip(windows, window_gates):
                logw = logw + surface_logweight(seg.arrays, float(rho) / float(D_walk),
                                                ch.get("boundary_local_time"), chi_s)      # raises without C2
        ew = w * np.exp(logw)
        norm = w.sum()
        ew, norm = self._select(compartment, ew, norm, w, ch, n_w)
        from ..acquisition.epg import pathway_weight
        # the voxel's factor for an unbalanced encoding (dmipy-sim#375), or the refusal without a voxel size;
        # read on the LAB waveform, since the voxel is the scanner's and the dot product k . r_v is the same in
        # every frame. The vector-Bloch route does not apply this scale: it places each walker in the voxel.
        voxel = np.asarray(waveform.voxel_factor(), np.float64)
        return dict(G=G, Geff=Geff, dt=dt, n_t=n_t, dt_wf=dt_wf, ch=ch, n_w=n_w, w=w, ew=ew, norm=norm, B0=B0,
                    pathway=(pathway_weight(waveform) if pathway else 1.0), voxel=voxel,
                    b0_dir=b0_dir, chi_iso=chi_iso, chi_aniso=chi_aniso, T2=T2, T1=T1, rho=rho, D=D, chi=chi, active=active,
                    G_eff_wf=G_eff, windows=windows, window_gates=window_gates)

    def _n_pool_ids(self, col):
        """How many pool ids the ``comp`` column addresses, read over every window."""
        if col["kind"] == "fraction":
            return 2
        key = "comp_static" if col["kind"] == "static" else "comp_rle_vals"
        return int(max(int(np.max(w.arrays[key])) for w, _, _ in self._windows())) + 1

    def pose_response(self, waveform, *, tissue=None, scanner=None, pose=None, compartment=None,
                      method="auto", keep=None, cache=None):
        """The pack's response over every pose of its substrate, for one acquisition: a :class:`PoseResponse` whose
        coefficients a voxel's orientation distribution contracts against (RPH.md 6).

        **Closed form, by default (#197).** For a single-direction encoding the response is a sum of plane waves in
        each walker's rotated moment, and its harmonics are the Rayleigh expansion, computed per walker with no
        rotation ever evaluated; with a field the response is that expansion times the field factor's, coupled
        with the real Clebsch-Gordan tables. The band follows the walkers' phase amplitudes by construction and
        nothing is chosen. ``keep = (lmax, nmax)`` restricts what is computed to what the composition retains: an
        ODF or peaks composition keeps ``n = 0``, a frame keeps everything; ``None`` in either slot means the
        response's own band.

        **The quadrature, by name.** An encoding whose moment matrix is not rank one -- a b-tensor or multi-axis
        waveform -- has no plane-wave expansion, and takes the sampled route: the response evaluated on an SO(3)
        quadrature sized to its phase amplitude, projected, and certified off the grid (``misfit``, a worst case).
        ``method="quadrature"`` asks for it explicitly; ``method="closed"`` refuses what the closed form cannot
        take rather than falling back. :attr:`PoseResponse.route` says which was used.

        ``tissue`` and ``scanner`` are :meth:`replay`'s. The substrate's pose is not a knob here, since every pose
        of it is what is being expanded; ``pose`` is the SPECIMEN's rigid rotation in the bore (a 3x3 rotation, or
        a phantom's :class:`~dmipy_sim.phantom.partition.Pose`): the acquisition and the field turn into that
        frame and the expansion runs there.

        ``cache`` -- a directory (or ``True`` for ``$DMIPY_SIM_CACHE``, else ``~/.cache/dmipy_sim/pose``): the
        expansion is written there under a key of the pack's digest, the acquisition on the pack's grid, the
        resolved knobs, the frame, the method and the band, and read back instead of recomputed the next time
        the same pack meets the same acquisition. Off unless asked for.
        """
        view = self._at_tissue(tissue)
        if view is not self:
            return view.pose_response(waveform, tissue=tissue, scanner=scanner, pose=pose, compartment=compartment, method=method, keep=keep, cache=cache)
        R_s = _pose_matrix(pose)
        if R_s is not None:                                       # the acquisition in the specimen frame: what the
            from ..acquisition.waveforms import rotate_waveform   # expansion reads, in P and from the waveform itself
            waveform = rotate_waveform(waveform, R_s.T)
        with Run("pose_response", params=dict(id=self.id, n_meas=int(waveform.n_meas), method=method,
                                        keep=(None if keep is None else [None if k is None else int(k) for k in keep]))):
            P = self._prepare(waveform, tissue=tissue, scanner=scanner, orientation=None, compartment=compartment)
            if R_s is not None:
                P["b0_dir"] = tuple(R_s.T @ np.array([0.0, 0.0, 1.0]))   # the bore's field, seen from the specimen
            if method not in ("auto", "closed", "quadrature"):
                raise ValueError("method is 'auto', 'closed' (the per-walker Rayleigh expansion) or 'quadrature'")
            path = None
            if cache is not None and cache is not False:
                path = self._pose_cache_path(cache, P, waveform, method, keep)
                if path.exists():
                    return PoseResponse.load(path)
            out = None
            if method != "quadrature":
                out = self._pose_coeffs_closed(P, waveform, keep=keep)
                if out is None and method == "closed":
                    raise ValueError("the closed-form pose expansion needs a single-direction encoding on every measurement: "
                                     "this acquisition has a b-tensor or multi-axis waveform, so use method='quadrature'")
            if out is None:
                out = self._pose_coeffs(P, waveform, keep=keep)
            if np.any(P["voxel"] != 1.0):
                # an unbalanced encoding: the voxel's factor scales each measurement's expansion, and its misfit with
                # it. The pose does not enter -- k . r_v is the same dot product in the lab and in the substrate.
                f = P["voxel"]
                out.coeffs = out.coeffs * f[:, None]
                m = np.asarray(out.misfit, float)
                out.misfit = m * np.abs(f) if m.shape == f.shape else m * float(np.abs(f).max())
            if path is not None:
                path.parent.mkdir(parents=True, exist_ok=True)
                out.save(path)
            return out

    def _pose_cache_path(self, cache, P, waveform, method, keep):
        """``<dir>/<key>.npz`` with the key over everything the expansion depends on."""
        import hashlib, os
        from pathlib import Path
        if cache is True:
            root = Path(os.environ.get("DMIPY_SIM_CACHE") or (Path.home() / ".cache" / "dmipy_sim")) / "pose"
        else:
            root = Path(cache)
        h = hashlib.sha256()
        h.update(self.digest.encode())
        h.update(np.ascontiguousarray(P["Geff"], np.float64).tobytes())      # the acquisition on the pack's grid
        h.update(np.ascontiguousarray(P["ew"], np.float64).tobytes())        # the weights with every tissue knob applied
        h.update(np.ascontiguousarray(self.substrate_frame).tobytes())
        rf = waveform.rf.refocus_time if waveform.rf else None
        gate = None if getattr(waveform, "gate", None) is None else np.asarray(waveform.gate, np.float32).tobytes()
        h.update(repr((float(P["norm"]), P["B0"], tuple(np.round(np.asarray(P["b0_dir"], float), 12)), P["chi_iso"],
                       P["chi_aniso"], rf, gate, method, None if keep is None else tuple(keep),
                       tuple(np.round(np.asarray(P["voxel"], float), 12)))).encode())
        return root / (h.hexdigest() + ".npz")

    def _select(self, compartment, ew, norm, w, ch, n_w):
        """Restrict the ensemble mean to ``compartment`` (a pool id or a walker mask)."""
        from .compression import decode_occupancy
        if compartment is None:
            return ew, norm
        sel = np.asarray(compartment)
        if sel.dtype != bool:
            if "compartment" not in ch:
                raise ValueError("compartment= by id needs the pack's compartment channel")
            ids = np.asarray(decode_occupancy(self.arrays, ch["compartment"])["comp"])
            ids = ids[:, 0] if ids.ndim == 2 else ids
            sel = ids.astype(int) == int(sel)
        if sel.shape[0] != n_w:
            raise ValueError(f"compartment mask has {sel.shape[0]} entries for {n_w} walkers")
        if not sel.any():
            raise ValueError("compartment selection matched no walkers")
        return np.where(sel, ew, 0.0), w[sel].sum()

    @property
    def temporal_bandwidth_hz(self):
        """The highest frequency the position bands resolve, ``K / (2 T)`` (#199): the pack's temporal band stated
        as a frequency, which is what a scanner envelope is checked against and what a prefix keeps."""
        return float(self.K) / (2.0 * (self.n_t - 1) * self.dt)

    def prefix(self, TE, *, K=None, out_path=None, tol=2.0, id=None, provenance=None):
        """This walk re-encoded to the shorter echo time ``TE`` (s), at the same bands per second (#199).

        The TE-prefix property (RPK.md 3) makes the first ``ceil(TE / dt)`` saves a valid walk to ``TE``. Every
        channel is decoded, cut there and re-encoded with its own codec through
        :func:`~dmipy_sim.replay.bank.build_replay_pack`, so the result is a pack like any other with its own
        fidelity certificate -- and that certificate is what licenses the prefix: re-pinning the bridge at ``TE``
        is a new truncation, measured against the parent's decoded prefix on the fidelity battery, and a prefix
        whose replay error exceeds ``tol`` times its Monte-Carlo floor is refused rather than written. The band
        starts at the same bands per second, ``K' = ceil(K TE / T)``, and is doubled (up to the parent's ``K``)
        until the certificate passes: a short prefix needs more than its share, since the pinned residual's
        truncation error does not scale with the duration (#199: 100 ms at K = 48 prefixes to 25 ms at 12 but
        needs 12, not 6, at 12.5 ms). ``K=`` fixes the band instead and is refused as it stands.
        ``provenance.prefix`` records the parent (digest, id, T, K), the cut and every band tried.
        """
        from .bank import build_replay_pack, seed_value, susc_path_decode, susc_path_encode_series, susc_path_series_fidelity
        from .compression import decode_occupancy, decode_boundary_bridge, decode_boundary_local_time
        dt, n_t = float(self.dt), int(self.n_t)
        T = (n_t - 1) * dt
        n_cut = int(round(float(TE) / dt)) + 1
        if n_cut < 3 or n_cut > n_t:
            raise ValueError(f"TE = {TE * 1e3:.3f} ms is not a prefix of this {T * 1e3:.3f} ms walk (dt {dt * 1e6:.1f} us): "
                             f"it needs at least three saves and at most the walk's {n_t}")
        T_cut = (n_cut - 1) * dt
        if self.n_segments > 1:
            steps = int(self.segments["n_t"]) - 1
            whole = (n_cut - 1) // steps + (1 if (n_cut - 1) % steps else 0)      # the windows the prefix lies in
            if (n_cut - 1) % steps == 0:
                out = self.truncate(whole, id=id, out_path=out_path)             # a prefix of whole windows is a range read
                return out
            return self.truncate(whole)._prefix_within(TE, K=K, out_path=out_path, tol=tol, id=id, provenance=provenance)
        return self._prefix_within(TE, K=K, out_path=out_path, tol=tol, id=id, provenance=provenance)

    def _prefix_within(self, TE, *, K=None, out_path=None, tol=2.0, id=None, provenance=None):
        """:meth:`prefix` by re-encoding: every channel decoded over the windows the prefix spans, cut and
        re-encoded as one window through :func:`~dmipy_sim.replay.bank.build_replay_pack`."""
        from .bank import build_replay_pack, seed_value, susc_path_decode, susc_path_encode_series, susc_path_series_fidelity
        from .compression import decode_occupancy, decode_boundary_bridge, decode_boundary_local_time
        dt, n_t = float(self.dt), int(self.n_t)
        T = (n_t - 1) * dt
        n_cut = int(round(float(TE) / dt)) + 1
        T_cut = (n_cut - 1) * dt
        K_new = int(K) if K is not None else max(2, int(np.ceil(self.K * T_cut / T)))
        ch = dict(self.meta.get("compression", {}).get("channels", {}) or {})
        decoded = self._decoded_channels()
        wp = dict(self.meta.get("walk_params", {}) or {})
        m = dict(traj=self.positions()[:, :n_cut, :], dt_traj=dt, T_max=T_cut,
                 walkers_shuffled=bool(self.meta.get("compression", {}).get("precision_tiers", {}).get("walkers_shuffled", False)),
                 seed=seed_value(wp.get("seed", 0)))
        if "spin_weights" in self.arrays:
            m["w"] = np.asarray(self.arrays["spin_weights"], np.float64)
        if self.meta.get("substrate") is not None:
            m["substrate"] = self.meta["substrate"]
        if wp.get("diffusivity") is not None:
            m["D_intra"] = float(wp["diffusivity"])
        m["substrate_frame"] = self.substrate_frame
        if "compartment" in ch:
            comp = np.asarray(decoded["comp"])
            m["comp"] = comp[:, :n_cut] if comp.ndim == 2 else comp
            if decoded["bound"] is not None:
                b = np.asarray(decoded["bound"]); m["bfrac"] = b[:, :n_cut] if b.ndim == 2 else b
        blt_K = None
        if "boundary_local_time" in ch:
            bm = dict(ch["boundary_local_time"])
            if self.has_surface:
                bm.setdefault("K", _cx_bands_K(self.segment(0).arrays, bm))
                blt_K = max(2, int(np.ceil(bm["K"] * T_cut / T)))
            m["dlog_b"] = np.asarray(decoded["ell"])[:, :n_cut]
        path_series = None
        if "susceptibility_path" in ch:
            series, names = decoded["path"]
            _n_tf, _dt_f = _path_grid(ch["susceptibility_path"], int(self.segments["n_t"]), dt)
            _every = max(1, int(round(_dt_f / dt)))
            n_cut_f = len(range(0, n_cut, _every))                                    # the channel's own prefix
            path_series = (series[:, :, :n_cut_f], names,
                           max(2, int(np.ceil(int(ch["susceptibility_path"]["K"]) * T_cut / T))),
                           ch["susceptibility_path"].get("bits", 8))
        tried = []
        while True:
            prov = dict(provenance or {})
            prov["prefix"] = dict(parent_digest=self.digest, parent_id=self.meta.get("id"), parent_T_s=T, parent_K=int(self.K),
                                  TE_s=T_cut, K=K_new, K_tried=tried + [K_new],
                                  note="re-encoded from the parent's decoded prefix; the band starts at the parent's bands "
                                       "per second and doubles until the certificate passes")
            pk = build_replay_pack(m, id=id or f"{self.meta.get('id')}/prefix-{T_cut * 1e3:.0f}ms", license=self.license,
                                   citation=self.citation, K=K_new, tol=tol, field=False, blt_temporal_K=blt_K, provenance=prov)
            fid = pk.meta.get("fidelity", {})
            if K is not None or fid.get("within_2x_floor", True) or K_new >= int(self.K):
                break
            tried.append(K_new)
            K_new = min(2 * K_new, int(self.K))
        if path_series is not None:
            series, names, Kp, bits = path_series
            a, pm = susc_path_encode_series(series, names, K=Kp, bits=bits, dt=_dt_f,
                                            max_refocus_pulses=self.meta["compression"]["channels"]["susceptibility_path"].get("max_refocus_pulses"))
            pk.arrays.update(a)
            pk.meta["compression"]["channels"]["susceptibility_path"] = pm
            g = dict(ch.get("susceptibility_grid", {})); g.update(arrays_in_pack=False, replay_route="path")
            pk.meta["compression"]["channels"]["susceptibility_grid"] = g
            pk.meta["replay_envelope"]["field"] = True
            pk.meta["replay_envelope"].setdefault("acquisition", {})["max_refocusing_pulses"] = int(pm["max_refocus_pulses"])
            # the field tier is certified as the producer certifies it (GRE, SE, the CPMG train it advertises),
            # against the parent's decoded prefix, which is the reference this child has
            spec = self.substrate
            chi_i = None
            if spec is not None:
                from ..spec.tissue import Tissue
                chi_i = Tissue.from_spec(spec).chi_iso
            cf = susc_path_series_fidelity(series, pk.arrays, pm, g, w=self.spin_weights, dt=_dt_f,
                                           env=self.meta.get("replay_envelope", {}).get("acquisition"),
                                           chi_iso=float(chi_i or 1.06e-6))
            fid = pk.meta.setdefault("fidelity", {})
            fid.update(err_susc_path=cf["err"], floor_susc_path=cf["floor"], susc_path_pulses_certified=cf["n_pulses_certified"],
                       err_max=max(float(fid.get("err_max", 0.0)), cf["err"]),
                       floor_max=max(float(fid.get("floor_max", 0.0)), cf["floor"]))
            fid["within_2x_floor"] = bool(fid["err_max"] <= tol * fid["floor_max"])
        fid = pk.meta.get("fidelity", {})
        if not fid.get("within_2x_floor", True):
            raise ValueError(f"the prefix at TE = {T_cut * 1e3:.1f} ms with K' = {K_new} does not reproduce the parent's decoded "
                             f"prefix within {tol:g} x its Monte-Carlo floor (error {fid.get('err_max'):.3g}, floor "
                             f"{fid.get('floor_max'):.3g}); give a larger K= or keep the parent")
        if out_path is not None:
            write_rpk(out_path, pk.arrays, pk.meta)
        return pk

    @property
    def digest(self):
        """A sha256 of the pack's arrays and metadata: its identity for a cache. Computed once per instance."""
        d = getattr(self, "_digest", None)
        if d is None:
            import hashlib, json
            h = hashlib.sha256()
            for k in sorted(self.arrays):
                h.update(k.encode()); h.update(np.ascontiguousarray(self.arrays[k]).tobytes())
            h.update(json.dumps(self.meta, sort_keys=True, default=str).encode())
            d = self._digest = h.hexdigest()
        return d

    @property
    def substrate_frame(self):
        """The substrate's intrinsic frame (RPK.md 4.2): a ``(3, 3)`` right-handed basis in stored coordinates, column
        3 the primary structural axis, ``d_stored = F d_canonical``. From ``walk_params.substrate_frame`` when the
        producer declared it, else from the embedded spec's ``frame`` (axis, optional in-plane), else the identity --
        the pack's positions are then taken to be in the canonical frame already."""
        from .bank import frame_from_axis
        wp = self.meta.get("walk_params", {}) or {}
        F = wp.get("substrate_frame")
        if F is None:
            F = self.meta.get("substrate_frame")
        if F is not None:
            F = np.asarray(F, np.float64).reshape(3, 3)
        else:
            spec = self.substrate
            fr = getattr(spec, "frame", None) if spec is not None else None
            if fr is None:
                return np.eye(3)
            F = np.asarray(frame_from_axis(fr.axis, in_plane=getattr(fr, "in_plane", None)), np.float64).reshape(3, 3)
        if not np.allclose(F @ F.T, np.eye(3), atol=1e-6) or np.linalg.det(F) < 0:
            raise ValueError("the pack's substrate_frame is not a proper rotation")
        return F

    @property
    def frame_axis(self):
        """The substrate's own axis in stored coordinates: column 3 of :attr:`substrate_frame`."""
        return self.substrate_frame[:, 2].copy()

    def pose_rotation(self, orientation):
        """The rotation taking STORED coordinates to the lab for one pose: from a 3x3 pose of the canonical
        substrate frame (``R``, so stored -> lab is ``R F^T`` with ``F`` the pack's :attr:`substrate_frame`), or
        from the lab direction the substrate's axis points along (the azimuth left as the frame's). A lab
        waveform reads in stored coordinates as ``G @ pose_rotation(orientation)``: what a consumer compiling
        its own scheme (a fit's compartment model) rotates by."""
        from .so3 import rotation_of
        o = np.asarray(orientation, float)
        if o.shape == (3, 3):
            if not np.allclose(o @ o.T, np.eye(3), atol=1e-6) or np.linalg.det(o) < 0:
                raise ValueError("orientation must be a proper rotation matrix (R R^T = I, det +1)")
            return o @ self.substrate_frame.T
        if o.shape == (3,):
            return rotation_of(o, self.frame_axis)
        raise ValueError("orientation is a (3, 3) rotation, a (3,) axis direction, or a distribution of poses "
                         "(dmipy_sim.replay.so3.Distribution, or an FOD read as an axis density)")

    def _pose_coeffs_closed(self, P, waveform, keep=None, tol=1e-8, l_cap=64):
        """The pose expansion in closed form (#197): the response of a single-direction encoding is a sum of plane
        waves in the rotated moment of each walker, and a plane wave's harmonics are the Rayleigh expansion.

        Measurement ``i`` plays ``G_i(t) = g_i s_i(t)``; walker ``w``'s phase at pose ``R`` is ``kappa g^ . R m^``
        with ``m_w = gamma sum_t s_i(t) r_w(t) dt`` (the moment, in the canonical frame) and ``kappa = |g||m|``. Then

            exp(i kappa g^.R m^) = 4 pi sum_l i^l j_l(kappa) sum_m Y_lm(g^) sum_n M^l(R)[m, n] Y_ln(m^)

        so in the basis ``sqrt(2l+1) M^l(R)[m, n]`` the ``(l, m, n)`` coefficient of the ensemble is

            c^l_mn = 4 pi i^l / sqrt(2l+1) * Y_lm(g^_i) * sum_w ew_w j_l(kappa_w) Y_ln(m^_w) / norm

        -- one contraction over walkers per order, no rotation ever evaluated. ``j_l(kappa)`` dies above
        ``l ~ kappa``, so the band is the largest phase amplitude and nothing is chosen: orders are added until
        their weighted Bessel tail is below ``tol``. ``keep`` restrains the band as before: an ODF or peaks
        composition keeps ``n = 0`` only, which here is the Legendre polynomial of the moment's angle to the
        substrate axis and never a roll quadrature.

        **The field.** The susceptibility phase of a walker is ``a_w + u^T A_w u`` in the field direction
        ``u = R^T b`` (:meth:`_field_quadratic`), a function on the sphere whose harmonics ``a_l'm'`` are read
        off a product quadrature exact to its own band (the band follows the phase amplitude ``|A_w|``, orders
        added until the energy above them is below ``tol``). The response is then the product of two expansions,
        contracted with the real coupling tables (:func:`so3.coupling`) on the lab index (``Y_lm(g^)`` with
        ``Y_l'n'(b^)``) and on the body index (``j_l Y_ln(m^)`` with ``a_l'm'``); no ``g x B0`` frame exists.
        Returns None when a measurement is not single-direction (a b-tensor encoding): that takes the quadrature.
        """
        from . import so3
        from .compression import read_position_coeffs
        Geff, dt, n_t, ew, norm = P["Geff"], P["dt"], P["n_t"], P["pathway"] * P["ew"], P["norm"]
        n_meas, n_w = Geff.shape[0], ew.shape[0]
        field = self._field_quadratic(P, waveform) if self._field_active(P["B0"]) else None   # (a_w, A_w) or None
        run = current()
        _ph = (lambda name, **f: run.phase(name, **f)) if run is not None else (lambda name, **f: None)
        _ph("moments", n_meas=int(n_meas), n_w=int(n_w))
        G = np.asarray(Geff, np.float64)
        g_hat = np.zeros((n_meas, 3)); s_wave = np.zeros((n_meas, n_t))
        for i in range(n_meas):
            Gi = G[i]
            if not np.any(Gi):
                g_hat[i] = (0.0, 0.0, 1.0)                                     # a b = 0 row: no phase at any pose
                continue
            _u, sv, vt = np.linalg.svd(Gi, full_matrices=False)
            if sv[1] > 1e-6 * sv[0]:                                    # G is stored float32; a direction is one to that
                return None                                                    # rank > 1: not a single direction
            g, sw = vt[0], Gi @ vt[0]
            lead = int(np.flatnonzero(np.abs(sw) > 1e-6 * np.abs(sw).max())[0])
            if sw[lead] < 0:                       # one spelling of (direction, waveform): the first lobe positive
                g, sw = -g, -sw
            g_hat[i] = g; s_wave[i] = sw
        # the body of a coefficient depends on the waveform's shape and amplitude only, never on its direction:
        # measurements that play the same s_i(t) -- a shell -- share one body, and their directions enter as
        # harmonics afterwards. Group by the played waveform, exactly, and contract once per group.
        group, first = _group_waveforms(s_wave, rtol=1e-5)                   # float32 G: 1e-5 is the same waveform
        n_grp = len(first)
        s_grp = s_wave[first]                                                  # (n_grp, n_t)
        from ._replay_kernel import effective_gradient
        e = np.eye(3)
        m = np.zeros((n_w, n_grp, 3))
        for seg, t0, n_s in P["windows"]:                                      # the windows' moments sum (RPK.md 4.3)
            G_s = effective_gradient(P["G_eff_wf"], P["dt_wf"], n_s, dt, t0=t0) if self.n_segments > 1 else G
            s_s = np.einsum("mtc,mc->mt", G_s[first], g_hat[first])            # each group's profile over this window
            C = read_position_coeffs(seg.arrays, dtype=np.float64).reshape(n_w, -1)
            for b_ in range(3):                                                # m_w[b] = gamma sum_t s(t) r_w(t)_b dt
                W = _compile_effective(s_s[:, :, None] * e[b_][None, None, :], dt, self.K, n_s)
                m[:, :, b_] += C @ W
        m = m @ self.substrate_frame                                           # stored -> canonical: F^T m, per walker
        kappa = np.linalg.norm(m, axis=2)                                      # (n_w, n_grp), radians
        safe = np.where(kappa > 0, kappa, 1.0)
        m_hat = m / safe[:, :, None]
        m_hat[kappa == 0] = (0.0, 0.0, 1.0)
        w = np.asarray(ew, np.float64) / float(norm)
        # the band: orders until the weighted Bessel tail is below tol for the worst group, every order from one
        # downward recurrence
        k_max = float(kappa.max()) if kappa.size else 0.0
        _ph("bessel", n_grp=int(n_grp), phase_amplitude=k_max)
        L = int(np.ceil(k_max)) + 2
        J_all = _spherical_jn_all(min(l_cap, L + 12), kappa)                  # (L_hi+1, n_w, n_grp)
        while L < l_cap:
            if L + 1 >= J_all.shape[0]:
                J_all = _spherical_jn_all(min(l_cap, J_all.shape[0] + 12), kappa)
            tail = (2 * (L + 1) + 1) * (np.abs(w)[:, None] * np.abs(J_all[L + 1])).sum(0).max()
            if tail < tol:
                break
            L += 1
        # the field factor's harmonics per walker, and the band the product reaches
        if field is None:
            L_f, F_sh = 0, None
        else:
            _ph("field")
            F_sh, L_f = self._field_harmonics(field, tol=tol, l_cap=l_cap)      # (n_w, (L_f+1)^2) complex
        L_tot = L + L_f
        want_l, want_n = (None, None) if keep is None else (keep[0], keep[1])
        keep_l = L_tot if want_l is None else min(int(want_l), L_tot)
        keep_n = L_tot if want_n is None else min(int(want_n), L_tot)
        n_feat = so3.n_so3_coeffs(keep_l, keep_n)
        _ph("harmonics", L=int(L), L_f=int(L_f), keep_l=int(keep_l), keep_n=int(keep_n), n_feat=int(n_feat))
        coeffs = np.zeros((n_meas, n_feat), np.complex128)
        cos_z = m_hat[:, :, 2]
        J = [J_all[l] for l in range(L + 1)]                                    # (n_w, n_grp) per order
        Yg = so3.real_sh(L, g_hat, full=True)                                   # (n_meas, (L+1)^2): the lab side
        if field is None:
            # ---- gradient only: one body per order and group, outer product with the direction harmonics
            bodies = [None] * (keep_l + 1)                                      # per order: (n_grp, 2k+1)
            if keep_n == 0:                                                     # n = 0 only: the Legendre of the angle to the axis
                for l in range(keep_l + 1):
                    bodies[l] = np.sqrt((2 * l + 1) / (4 * np.pi)) * ((w[:, None] * J[l]) * _legendre(l, cos_z)).sum(0)[:, None]
            else:
                # the body side: every walker's moment direction's harmonics, all orders in one recurrence pass,
                # in chunks of groups sized to ~256 MB
                for l in range(keep_l + 1):
                    bodies[l] = np.empty((n_grp, 2 * (so3._n_cols(l, keep_n) // 2) + 1))
                n_cols = (keep_l + 1) ** 2
                step = max(1, int(2.5e8 / (8 * n_w * n_cols)))
                for lo in range(0, n_grp, step):
                    sl = slice(lo, min(lo + step, n_grp))
                    nc = sl.stop - sl.start
                    if run is not None:
                        run.progress(lo, n_grp, unit="groups")
                    Y = so3.real_sh(keep_l, m_hat[:, sl, :].reshape(-1, 3), full=True).reshape(n_w, nc, n_cols)
                    for l in range(keep_l + 1):
                        k = so3._n_cols(l, keep_n) // 2
                        blk = so3.sh_block(l, True)
                        Yl = Y[:, :, blk.start + l - k:blk.start + l + k + 1]                 # (n_w, nc, 2k+1)
                        bodies[l][sl] = np.einsum("wi,wim->im", w[:, None] * J[l][:, sl], Yl)
            off = 0
            for l in range(keep_l + 1):
                k = so3._n_cols(l, keep_n) // 2
                blk = so3.sh_block(l, True)
                body_i = bodies[l][group]                                                      # (n_meas, 2k+1)
                block = (4 * np.pi * (1j ** l) / np.sqrt(2 * l + 1)) * Yg[:, blk][:, :, None] * body_i[:, None, :]
                coeffs[:, off:off + (2 * l + 1) * (2 * k + 1)] = block.reshape(n_meas, -1)
                off += (2 * l + 1) * (2 * k + 1)
        else:
            # ---- gradient x field: the outer product of the two body expansions per walker, summed over the
            # walkers per group, then coupled on both indices into the total order L_tot
            b_lab = np.asarray(P["b0_dir"], np.float64); b_lab = b_lab / np.linalg.norm(b_lab)
            Yb = so3.real_sh(L_f, b_lab[None, :], full=True)[0]                 # ((L_f+1)^2,): the field direction, lab side
            # the field factor as two contiguous real blocks: a .real view of a complex array strides 16 bytes on
            # its last axis, which BLAS does not take, and numpy's fallback loop is twenty times slower per product
            F_re, F_im = np.ascontiguousarray(F_sh.real), np.ascontiguousarray(F_sh.imag)
            n_cols = (L + 1) ** 2
            Ym = np.empty((n_w, n_grp, n_cols))                                # the moment harmonics, all groups
            step = max(1, int(2.5e8 / (8 * n_w * n_cols)))
            for lo in range(0, n_grp, step):
                sl = slice(lo, min(lo + step, n_grp)); nc = sl.stop - sl.start
                if run is not None:
                    run.progress(lo, n_grp, unit="groups")
                Ym[:, sl, :] = so3.real_sh(L, m_hat[:, sl, :].reshape(-1, 3), full=True).reshape(n_w, nc, n_cols)
            _ph("couplings", L=int(L), L_f=int(L_f))
            offs = {}
            off = 0
            for Lc in range(keep_l + 1):
                offs[Lc] = off; off += (2 * Lc + 1) * (2 * (so3._n_cols(Lc, keep_n) // 2) + 1)
            for l in range(L + 1):
                bl = so3.sh_block(l, True)
                if l > keep_l + L_f:
                    break
                # body for every field order at once: B_all[(g, n), (l', m')] = sum_w w j_l(kappa) Y_ln(m^) a_l'm'(w),
                # one matrix product over the walkers per gradient order
                X = ((w[:, None] * J[l])[:, :, None] * Ym[:, :, bl]).reshape(n_w, -1)       # (n_w, n_grp (2l+1))
                B_all = (X.T @ F_re + 1j * (X.T @ F_im)).reshape(n_grp, 2 * l + 1, -1)   # (n_grp, 2l+1, (L_f+1)^2); real products
                for lp in range(L_f + 1):
                    blp = so3.sh_block(lp, True)
                    Ls = [Lc for Lc in range(abs(l - lp), min(l + lp, keep_l) + 1)]
                    if not Ls:
                        continue
                    B = B_all[:, :, blp].reshape(n_grp, -1)                                 # (n_grp, (2l+1)(2l'+1))
                    # lab: Lam[i, (m, n')] = Y_lm(g^_i) Y_l'n'(b^)
                    Lam = (Yg[:, bl][:, :, None] * Yb[blp][None, None, :]).reshape(n_meas, -1)
                    K = so3.coupling(l, lp)
                    for Lc in Ls:
                        KL = K[Lc]                                              # ((2l+1)(2l'+1), 2Lc+1)
                        kk = so3._n_cols(Lc, keep_n) // 2
                        lab = Lam @ KL                                          # (n_meas, 2Lc+1)
                        body = B @ KL.conj()[:, Lc - kk:Lc + kk + 1]            # (n_grp, 2kk+1)
                        block = (4 * np.pi * (1j ** l) / np.sqrt(2 * Lc + 1)) * lab[:, :, None] * body[group][:, None, :]
                        o = offs[Lc]
                        coeffs[:, o:o + (2 * Lc + 1) * (2 * kk + 1)] += block.reshape(n_meas, -1)
        # what the expansion cannot hold pointwise: the orders above the band it was built to, as a bound from
        # |P_l| <= 1 -- below tol by construction
        tail = np.zeros(n_grp)
        for l in range(L + 1, min(L + 4, J_all.shape[0])):
            tail += (2 * l + 1) * (np.abs(w)[:, None] * np.abs(J_all[l])).sum(0)
        out = PoseResponse(coeffs, keep_l, keep_n, misfit=tail[group], floor=1.0 / np.sqrt(n_w), phase_amplitude=k_max,
                           n_samples=0)
        out.n_bodies = n_grp                                                   # the distinct waveforms contracted
        out.field_lmax = L_f
        out.route = "closed"
        return out

    def _field_quadratic(self, P, waveform):
        """The susceptibility phase of every walker as ``a_w + u^T A_w u`` in the field direction ``u`` expressed in
        the CANONICAL frame: ``(a (n_w,), A (n_w, 3, 3))``, the gate-integrated path field basis of the pack (C3 path
        route) scaled by ``B0``, ``chi_iso``, ``chi_aniso``. Raises, as the quadrature route does, when the pack
        cannot supply it."""
        from scipy.fft import dct
        from .bank import path_field_integral
        B0, chi_iso, chi_aniso = P["B0"], P["chi_iso"], P["chi_aniso"]
        self._field_active(B0)
        pm = self.meta.get("compression", {}).get("channels", {}).get("susceptibility_path")
        if pm is None:
            raise ValueError("the pose expansion with a field needs the pack's susc_path channel (C3 path route)")
        if chi_iso is None:
            raise ValueError("a scanner field was given without a chi_iso in the tissue; give chi_iso (and chi_aniso)")
        dt = P["dt"]
        Psi = None
        for seg, t0, n_s in P["windows"]:                                        # the windows' path integrals sum
            Psi_s, names = path_field_integral(seg.arrays, pm, waveform, n_s, dt, t0=t0)          # (n_w, n_ch)
            Psi = Psi_s if Psi is None else Psi + Psi_s
        i_p = names.index("iso_P_xx")
        i_a = names.index("aniso_G_xx") if "aniso_G_xx" in names else None
        a = float(chi_iso) * float(B0) * Psi[:, names.index("iso_local")]
        six = -float(chi_iso) * float(B0) * Psi[:, i_p:i_p + 6]
        if chi_aniso and i_a is not None:
            six = six + float(chi_aniso) * float(B0) * Psi[:, i_a:i_a + 6]
        xx, yy, zz, xy, xz, yz = six.T                                         # u^T A u = xx x^2 + ... + 2 xy x y + ...
        A = np.empty((six.shape[0], 3, 3))
        A[:, 0, 0], A[:, 1, 1], A[:, 2, 2] = xx, yy, zz
        A[:, 0, 1] = A[:, 1, 0] = xy; A[:, 0, 2] = A[:, 2, 0] = xz; A[:, 1, 2] = A[:, 2, 1] = yz
        F = self.substrate_frame                                               # stored -> canonical: A_c = F^T A F
        A = np.einsum("ab,wbc,cd->wad", F.T, A, F)
        return a, A

    def _field_harmonics(self, field, tol=1e-8, l_cap=64):
        """The harmonics of ``exp(i (a_w + u^T A_w u))`` over the sphere per walker, ``(n_w, (L'+1)^2)``, by a product
        quadrature exact to the band ``L'`` chosen from the phase amplitude: orders are added until the energy in
        the last one is below ``tol`` of the total (``4 pi`` per walker, the phase having unit modulus). A phase
        whose band lies past ``l_cap`` is refused: the expansion is not the route for it, a replay per pose is."""
        from . import so3
        a, A = field
        amp = float(np.abs(np.linalg.eigvalsh(A)).max()) if A.size else 0.0
        Lp = int(np.ceil(2.0 * amp)) + 4
        if Lp > l_cap:
            raise ValueError(
                f"the scanner's field sweeps {amp:.1f} radians of phase on this pack, so its pose response reaches "
                f"order ~{Lp}, beyond the cap of {l_cap}. That is a real cost, not a setting: the expansion is "
                f"worth building to share one walk over many poses, and at this sharpness a direct replay per pose "
                f"(orientation=R) is the exact route for it.")
        n_w = a.shape[0]
        run = current()
        while True:
            dirs, wq = so3.sphere_quadrature(Lp + 2, 2 * Lp + 2)
            Y = so3.real_sh(Lp, dirs, full=True)                               # (n_q, (Lp+1)^2)
            Yw = Y * wq[:, None]
            F = np.empty((n_w, Y.shape[1]), np.complex128)
            step = max(1, int(2.5e8 / (16 * dirs.shape[0])))                   # walkers per ~256 MB of phase
            for lo in range(0, n_w, step):
                sl = slice(lo, min(lo + step, n_w))
                if run is not None:
                    run.progress(lo, n_w, unit="walkers")
                q = np.einsum("qa,wab,qb->wq", dirs, A[sl], dirs)              # (n_c, n_q)
                f = np.exp(1j * (a[sl, None] + q))
                F[sl] = (f.real @ Yw) + 1j * (f.imag @ Yw)                     # real products
            top = so3.sh_block(Lp, True)
            if (np.abs(F[:, top]) ** 2).sum(1).max() <= tol * 4.0 * np.pi or Lp + 4 > l_cap:
                return F, Lp
            Lp += 4

    def _pose_coeffs(self, P, waveform, band=None, keep=None, margin=2, n_check=256, seed=0,
                     chunk=256, over=2, band_cap=12, strict=True):
        """Sample the response over rotations, project it, and check it off the grid.

        The gradient term is exact for any waveform, single- or multi-axis: the phase of walker ``w`` at pose
        ``R`` is ``<R, M_w>`` with ``M_w[a, b] = sum_t Geff[t, a] r_w[t, b]``, so the walk is contracted once
        per measurement and every pose is then a 3x3 inner product. The field term is the same second-rank form
        in the field direction carried into the substrate frame, reached through the six quadratic products of
        the rotated field direction.

        The band follows from those contractions rather than from a setting: ``||M_w||`` is the phase the pose
        modulates, in radians, and the harmonic content of ``exp(i <U, M_w>)`` reaches about that order -- the
        cutoff that makes ``j_l(x)`` negligible for ``l`` beyond ``x``. The quadrature is then oversampled past
        the projection band, because a rule exact only for the band being kept folds everything above it into
        those coefficients, and does so differently at different frames. The misfit is measured at rotations off
        that grid, where such folding cannot hide, and as a worst case rather than a spread.
        """
        from scipy.fft import dct
        from . import so3
        from .compression import read_position_coeffs
        from ._replay_kernel import field_gate
        Geff, dt, n_t, ew, norm, B0 = P["Geff"], P["dt"], P["n_t"], P["pathway"] * P["ew"], P["norm"], P["B0"]
        b0_dir, chi_iso, chi_aniso = P["b0_dir"], P["chi_iso"], P["chi_aniso"]
        n_meas, n_w = Geff.shape[0], ew.shape[0]

        C = read_position_coeffs(self.arrays, dtype=np.float64).reshape(n_w, -1)
        # The gradient term at pose R is <R, M_w> with M_w[a, b] = sum_t Geff[i, t, a] r_w[t, b]: the walk is
        # contracted against the waveform once, and every pose after that is a 3x3 inner product. Exact for a
        # multi-axis waveform too, which an expansion in one gradient direction could not take at all.
        Q = np.empty((n_w, n_meas, 3, 3))
        e = np.eye(3)
        for a in range(3):
            for b_ in range(3):
                W = _compile_effective(Geff[:, :, a][:, :, None] * e[b_][None, None, :], dt, self.K, n_t)
                Q[:, :, a, b_] = C @ W

        Psi = names = i_p = i_a = None
        if self._field_active(B0):
            pm = self.meta.get("compression", {}).get("channels", {}).get("susceptibility_path")
            if pm is None:
                raise ValueError("the pose expansion with a field needs the pack's susc_path channel (C3 path route)")
            if chi_iso is None:
                raise ValueError("a scanner field was given without a chi_iso in the tissue; give chi_iso (and chi_aniso)")
            from .bank import path_field_integral
            Psi, names = path_field_integral(self.arrays, pm, waveform, n_t, dt)         # (n_w, n_ch)
            i_p = names.index("iso_P_xx")
            i_a = names.index("aniso_G_xx") if "aniso_G_xx" in names else None
        b = np.asarray(b0_dir, float); b = b / np.linalg.norm(b)

        # a pose R is a rotation of the CANONICAL substrate frame: the stored walk is first turned into it
        # (F^T, RPK.md 4.2) and then by R -- the identity for a pack whose frame is the identity
        Qf_axis = self.substrate_frame.T

        def response(R):
            """The ensemble signal of every measurement at every one of these poses."""
            E = np.empty((R.shape[0], n_meas), np.complex128)
            for lo in range(0, R.shape[0], int(chunk)):
                sl = slice(lo, min(lo + int(chunk), R.shape[0]))
                Rc = R[sl] @ Qf_axis
                if Psi is None:
                    Ew = np.broadcast_to(ew[None, :].astype(np.complex128), (Rc.shape[0], n_w))
                else:
                    bs = np.einsum("nji,j->ni", Rc, b)                              # the field in the substrate frame
                    Qf = np.stack([bs[:, 0] ** 2, bs[:, 1] ** 2, bs[:, 2] ** 2, 2 * bs[:, 0] * bs[:, 1],
                                   2 * bs[:, 0] * bs[:, 2], 2 * bs[:, 1] * bs[:, 2]], axis=1)
                    phi_chi = float(chi_iso) * float(B0) * (Psi[:, names.index("iso_local")][None, :]
                                                            - Qf @ Psi[:, i_p:i_p + 6].T)
                    if chi_aniso and i_a is not None:
                        phi_chi = phi_chi + float(chi_aniso) * float(B0) * (Qf @ Psi[:, i_a:i_a + 6].T)
                    Ew = np.exp(1j * phi_chi) * ew[None, :]
                for i in range(n_meas):
                    E[sl, i] = (Ew * np.exp(1j * np.einsum("nab,wab->nw", Rc, Q[:, i]))).sum(1) / norm
            return E

        # the phase the pose modulates, per measurement: the sampling band follows this, not a setting
        amp = np.linalg.norm(Q.reshape(n_w, n_meas, 9), axis=2)
        phi_amp = float(np.percentile(amp, 95)) if amp.size else 0.0
        if band is None:
            S_L = int(max(int(np.ceil(phi_amp)) + int(margin), 2))
        else:
            S_L = int(band[0]) if np.ndim(band) else int(band)
        if S_L > int(band_cap):
            raise ValueError(
                f"this acquisition sweeps {phi_amp:.1f} radians of phase on this pack, so its pose response "
                f"reaches order ~{S_L}, beyond the cap of {band_cap}. That is a real cost, not a setting: the "
                f"expansion is worth building to share one walk over many poses, and at this sharpness a direct "
                f"replay per pose (orientation=R) is the exact route for it.")
        S_N = S_L
        # `keep` may leave either index open with None, meaning "whatever the response carries"
        want_l, want_n = (None, None) if keep is None else (keep[0], keep[1])
        keep_l = S_L if want_l is None else min(int(want_l), S_L)
        keep_n = S_N if want_n is None else min(int(want_n), S_N)

        # sample beyond the retained band and project onto it, in blocks: a rule exact only for what is kept
        # folds everything above it into those coefficients (and differently at different frames)
        Rq, wq, _dirs, _rolls = so3.so3_quadrature(S_L + int(over), S_N + int(over))
        n_feat = so3.n_so3_coeffs(keep_l, keep_n)
        coeffs = np.zeros((n_feat, n_meas), np.complex128)
        for lo in range(0, Rq.shape[0], int(chunk)):
            sl = slice(lo, min(lo + int(chunk), Rq.shape[0]))
            A = so3.so3_design(keep_l, Rq[sl], keep_n)
            coeffs += (A * wq[sl, None]).T @ response(Rq[sl])
        floor = 1.0 / np.sqrt(n_w)

        Rc, Ac = so3.haar_design(keep_l, keep_n, int(n_check), int(seed))
        if (keep_l, keep_n) == (S_L, S_N):
            # the retained band is the whole response: certify pointwise, worst case, off the grid
            misfit = np.abs(Ac @ coeffs - response(Rc)).max(axis=0)
        else:
            # only part of the response is retained, so a pointwise comparison is not the question. What has to
            # hold is that the retained coefficients are alias-free: refine the grid and require them to stand.
            Rf, wf, _d, _r = so3.so3_quadrature(S_L + 2 * int(over) + 1, S_N + 2 * int(over) + 1)
            fine = np.zeros_like(coeffs)
            for lo in range(0, Rf.shape[0], int(chunk)):
                sl = slice(lo, min(lo + int(chunk), Rf.shape[0]))
                A = so3.so3_design(keep_l, Rf[sl], keep_n)
                fine += (A * wf[sl, None]).T @ response(Rf[sl])
            misfit = np.abs(Ac @ (coeffs - fine)).max(axis=0)
        if misfit.max() > floor:
            msg = (
                f"this pack's pose response is not represented at (lmax, nmax) = ({keep_l}, {keep_n}): the worst "
                f"case away from the projection's own grid is {misfit.max():.4f} in signal units, against the "
                f"pack's Monte-Carlo floor of {floor:.4f}. The response's phase amplitude is {phi_amp:.1f} rad, "
                f"so it reaches about that order; retain more of it. Composing "
                f"here would return a plausible wrong number rather than a wrong-looking one.")
            if strict:
                raise ValueError(msg)
            import warnings
            warnings.warn(msg, UserWarning, stacklevel=3)
        return PoseResponse(coeffs.T, keep_l, keep_n, misfit, floor, phi_amp, Rq.shape[0] + Rc.shape[0])


    @cached_property
    def position_coeffs(self):
        """``(n_walkers, K+2, n_axes)``: two exact endpoints, then ``K`` sine bands per axis.

        The leading two entries are NOT bands -- they are ``r(0)`` and ``r(T)-r(0)`` -- so this
        array must never be handed to a band-basis contraction.  Raises on a pre-layout pack
        rather than guessing: see compression.read_position_coeffs.

        Stacked from the per-axis tensors once; a pack's arrays do not change after loading, and
        ``K``, ``n_walkers`` and ``n_coeffs`` all read this block.
        """
        from .compression import read_position_coeffs, require_position_method
        require_position_method(self.method)
        return read_position_coeffs(self.arrays, dtype=np.float32)

    @property
    def r0(self):
        """``(n_walkers, 3)``: where every walker **started**, in metres, read from the stored coefficients.

        The position codec keeps ``r(0)`` as an exact entry rather than a reconstructed band, so this is the
        stored value to float32 (of order 0.1 nm on a millimetre coordinate), identical at every ``K``, and it
        costs one slice of the coefficient block -- no path is decoded. It is what partitions a walk into voxels
        (RPH.md: ``voxel = floor((r0 - origin) / voxel_size)``), which is why it must not depend on the band.
        """
        from .compression import read_position_coeffs, require_position_method
        require_position_method(self.method)
        return np.ascontiguousarray(read_position_coeffs(self.arrays, dtype=np.float64)[:, 0, :])

    @property
    def dct_coeffs(self):
        raise AttributeError(
            "ReplayPack.dct_coeffs is gone. Positions are stored as bridge_dst -- two exact "
            "endpoints followed by sine bands -- so the name described the wrong basis and the "
            "array it returned would be contracted against the wrong one. Use "
            "ReplayPack.position_coeffs, and compile the scheme with replay.compile_scheme, "
            "which emits the matching layout.")

    @property
    def spin_weights(self):
        w = self.arrays.get("spin_weights")
        return np.ones(self.position_coeffs.shape[0]) if w is None else w

    @property
    def blt_dct(self):
        raise AttributeError(
            "blt_dct is retired: C2 is stored in the bridge form as 'blt_bridge_dst' plus the "
            "exact endpoints 'blt_start'/'blt_endpoint'. Use pack.arrays for the tensors, or "
            "compression.decode_boundary_bridge for the per-save series.")

    @property
    def boundary_local_time(self):
        """The C2 channel's exact cumulative total per walker, or None if the pack has no C2."""
        return self.arrays.get("blt_endpoint")

    @property
    def K(self):
        """Retained sine bands -- NOT the stored coefficient count, which is ``K + 2``.

        Reading the width off the array instead would hand ``K + 2`` to compile_scheme, whose
        output would then be the right shape to multiply and the wrong thing to multiply by.
        Declared and stored are cross-checked so they cannot drift apart.
        """
        k = int(self.meta.get("compression", {}).get("K", -1))
        stored = int(self.position_coeffs.shape[1])
        if k < 0:
            raise ValueError("pack declares no compression.K; refusing to infer it from the "
                             "array width, which counts the two endpoints as well")
        if stored != k + 2:
            raise ValueError(f"pack declares K={k} but stores {stored} coefficients per axis; "
                             f"expected {k + 2} (two endpoints + K bands)")
        return k

    @property
    def n_coeffs(self):
        """Stored coefficients per axis, ``K + 2``."""
        return int(self.position_coeffs.shape[1])

    @property
    def n_walkers(self):
        return int(self.position_coeffs.shape[0])

    @property
    def n_t(self):
        wp = self.meta.get("walk_params", {})
        return int(self.meta.get("n_t") or wp.get("n_t"))

    @property
    def dt(self):
        wp = self.meta.get("walk_params", {})
        return float(self.meta.get("dt") or wp.get("dt_traj") or wp.get("dt"))

    # ---- producer metadata (present on packs written by dmipy_sim.replay.bank.build_replay_pack) ----
    @property
    def id(self):
        return self.meta.get("id")

    @property
    def method(self):
        return self.meta.get("compression", {}).get("method")

    license = property(lambda self: self.meta.get("license"))
    citation = property(lambda self: self.meta.get("citation"))
    fidelity = property(lambda self: self.meta.get("fidelity"))
    replay_envelope = property(lambda self: self.meta.get("replay_envelope"))
    provenance = property(lambda self: self.meta.get("provenance"))



def _as_distribution(orientation):
    """``orientation=`` as a distribution of poses, or None when it names a single pose.

    An :class:`~dmipy_sim.replay.fod.FOD` is read as an axis density with no statement about the substrate's own
    azimuth, which is what an ODF says.
    """
    from .fod import FOD
    from .so3 import Distribution
    if isinstance(orientation, Distribution):
        return orientation
    if isinstance(orientation, FOD):
        return Distribution.axis_density(orientation)
    return None


def _echo_saves(waveform, dt_pack):
    """The pack-grid save indices of a multi-echo readout, or ``None`` when the sequence is read at its last
    sample (the Bloch replay then returns the final magnetisation)."""
    readout = tuple(int(i) for i in waveform.readout)
    if readout == (waveform.n_t - 1,):
        return None
    return [int(round(i * float(waveform.dt) / float(dt_pack))) for i in readout]


# ------------------------------- compiled-scheme forward -------------------------------
def analytic_pose_response(form, waveform, keep=None, *, over=8, tol=1e-8, l_cap=48):
    """The :class:`PoseResponse` of a closed form with an axis: ``form.response(waveform, pose=R)`` expanded in the
    SO(3) basis (RPH.md 6), so a voxel's orientation distribution contracts it exactly as it does a pack's.

    A form with an axis and no azimuth of its own is a function of where its axis points, ``g(R z)``, so its
    expansion lives in the ``n = 0`` coefficients and is the harmonic expansion of ``g`` over the sphere carried
    through the same map an axis density is (:func:`so3.axis_density_coeffs`): one sphere quadrature of the
    form, never a quadrature over rotations. ``keep = (lmax, nmax)`` is what the composition retains; ``None``
    in the first slot means the form's own band, found by raising ``lmax`` (4, 8, 12, 16, 24, 32, 48) until the
    top two orders carry less than ``tol`` of the energy. The sphere rule is exact ``over`` orders past the
    band because the form is not band-limited; ``misfit`` is the largest difference, per measurement, between
    that expansion and one exact eight orders further, relative to the largest response, and the route is
    refused when it exceeds ten times ``tol``. A closed form has no Monte-Carlo floor: ``floor = 0``."""
    from . import so3
    keep_l, keep_n = (None, None) if keep is None else keep
    nmax = 0 if keep_n is None else int(keep_n)
    evaluated = {}

    def sh(lmax, ov):
        """The form's harmonic coefficients over the sphere, ``(n_sh, n_meas)``, by a rule exact to ``lmax + ov``."""
        L = int(lmax) + int(ov)
        dirs, w = so3.sphere_quadrature(L + 1, 2 * L + 2)
        E = np.stack([_evaluate(form, waveform, so3.rotation_of(u), evaluated) for u in dirs])   # (n_q, n_meas)
        Y = so3.real_sh(int(lmax), dirs, full=True)
        return (Y * w[:, None]).T @ E

    if keep_l is None:
        ladder = [l for l in (4, 8, 12, 16, 24, 32, 48) if l <= l_cap] or [int(l_cap)]
        for lmax in ladder:
            g = sh(lmax, over)
            per_l = np.array([float((np.abs(g[so3.sh_block(l, True)]) ** 2).sum()) for l in range(lmax + 1)])
            tot = float(per_l.sum())
            if tot > 0 and per_l[-2:].sum() <= tol * tot:
                break
    else:
        lmax = int(keep_l)
    M = _axis_map_cached(lmax, nmax)                                 # (n_feat, n_sh): a scaled isometry per order
    scale = np.einsum("ij,ij->j", M, M)                              # so the response's coefficients are M g / |M_l|^2
    scale = np.where(scale > 0, scale, 1.0)
    f1 = M @ (sh(lmax, over) / scale[:, None])
    f2 = M @ (sh(lmax, over + 8) / scale[:, None])
    ref = float(np.abs(_evaluate(form, waveform, np.eye(3), evaluated)).max()) or 1.0
    misfit = np.abs(f1 - f2).max(axis=0) / ref
    if misfit.max() > 10 * tol:
        raise ValueError(f"the closed form {form!r} expands to lmax = {lmax} with a sphere rule that has not converged "
                         f"(misfit {misfit.max():.3g} of the largest response between over = {over} and {over + 8}): "
                         f"raise the orientation field's order or pass over=")
    out = PoseResponse(f2.T, lmax, nmax, misfit, 0.0, 0.0, 0)
    out.route = "analytic"
    return out


_AXIS_MAPS = {}


def _axis_map_cached(lmax, nmax):
    from . import so3
    key = (int(lmax), int(nmax))
    if key not in _AXIS_MAPS:
        _AXIS_MAPS[key] = np.asarray(so3._axis_map(int(lmax), int(nmax)), np.float64)
    return _AXIS_MAPS[key]


def _evaluate(form, waveform, R, cache):
    key = tuple(np.round(np.asarray(R, np.float64).reshape(-1), 12))
    if key not in cache:
        cache[key] = np.asarray(form.response(waveform, pose=np.asarray(R, np.float64).reshape(3, 3)), np.complex128).reshape(-1)
    return cache[key]


def compile_scheme(G, dt, K, gyromagnetic_ratio=GAMMA, *, n_t=None, method=None, dt_pack=None):
    """Compile an acquisition into its temporal-basis projection ``W``: the exact integral of the waveform
    against the stored path, in mode space.

    ``G`` is the gradient waveform ``(n_meas, n_wf, 3)`` [T/m] on ITS OWN grid ``dt`` [s]; ``dt_pack`` the pack's
    save interval (default: the waveform is on the pack grid, ``dt_pack = dt``); ``n_t`` the pack's save count
    (default ``G.shape[1]`` on the pack grid); ``K`` the pack's retained-mode count; ``method`` the pack's
    position codec, accepted only to let a caller assert it. Shape is ``(n_c (K+2), n_meas)`` for a waveform of
    ``n_c`` components (three for a pack's positions; fewer when a consumer reads only some stored axes and
    supplies the waveform's matching components, in the same order): per component the two gradient moments
    ``M0`` and ``M1``, which a motion-compensated waveform makes vanish, then the sine bands.
    Nothing is resampled: the waveform enters through its exact per-save weights
    (:func:`_replay_kernel.effective_gradient`), so an edge between saves carries exactly its b. Reusable across
    every pack on one grid (fitting) and every fit iteration; in design it is recomputed per candidate."""
    from .compression import require_position_method
    from ._replay_kernel import effective_gradient
    require_position_method(method or "bridge_dst")
    G = np.asarray(G, np.float64)
    if dt_pack is None:
        dt_pack = dt
        n_t = int(n_t or G.shape[1])
    elif n_t is None:
        raise ValueError("a waveform on its own grid needs the pack's n_t")
    return _compile_effective(effective_gradient(G, dt, int(n_t), dt_pack), dt_pack, K, int(n_t), gyromagnetic_ratio)


def _group_waveforms(s, rtol=1e-5):
    """Group rows of ``s (n, n_t)`` that are the same waveform to ``rtol`` of the largest amplitude: ``(group (n,),
    first (n_grp,))``. A tolerance, not a rounding, so two rows a rounding boundary apart stay together."""
    s = np.asarray(s, np.float64)
    scale = float(np.abs(s).max()) or 1.0
    tol = rtol * scale
    group = np.full(s.shape[0], -1, np.int64)
    first = []
    for i in range(s.shape[0]):
        if group[i] >= 0:
            continue
        same = np.flatnonzero((group < 0) & (np.abs(s - s[i]).max(axis=1) <= tol))
        group[same] = len(first)
        first.append(i)
    return group, np.asarray(first, np.int64)


def _spherical_jn_all(L, x, extra=24):
    """``j_0(x) .. j_L(x)`` for every entry of ``x``, ``(L+1,) + x.shape``, by the downward (Miller) recurrence
    started ``extra`` orders above ``L`` and normalised to ``j_0 = sin x / x``: stable for every order and
    argument, exact to 1e-12 against scipy, and one pass instead of one call per order."""
    x = np.asarray(x, np.float64)
    L = int(L)
    N = L + int(extra) + int(np.ceil(np.abs(x).max())) if x.size else L + int(extra)
    small = np.abs(x) < 1e-12
    xs = np.where(small, 1.0, x)
    hi, lo = np.zeros_like(xs), np.full_like(xs, 1e-30)           # j_{N+1} := 0, j_N := tiny, then downward
    out = np.empty((L + 1,) + x.shape, np.float64)
    for l in range(N, -1, -1):
        cur = (2 * l + 3) / xs * lo - hi                            # j_l = (2l+3)/x j_{l+1} - j_{l+2}
        hi, lo = lo, cur
        if l <= L:
            out[l] = cur
        m = np.abs(cur) > 1e200                                     # rescale before it overflows; the ratio is what matters
        if m.any():
            hi = np.where(m, hi * 1e-200, hi); lo = np.where(m, lo * 1e-200, lo)
            if l <= L:
                out[l:] = np.where(m[None, ...], out[l:] * 1e-200, out[l:])
    j0 = np.where(small, 1.0, np.sin(xs) / xs)
    scale = j0 / np.where(out[0] == 0, 1.0, out[0])
    out = out * scale[None, ...]
    if small.any():
        out[1:, small] = 0.0; out[0, small] = 1.0
    return out


def _legendre(l, x):
    """``P_l(x)`` by the three-term recurrence, vectorised over ``x``."""
    x = np.asarray(x, np.float64)
    if l == 0:
        return np.ones_like(x)
    p0, p1 = np.ones_like(x), x.copy()
    for k in range(1, l):
        p0, p1 = p1, ((2 * k + 1) * x * p1 - k * p0) / (k + 1)
    return p1


def _path_grid(pm, n_t, dt):
    """The path channel's own save grid ``(n_t, dt)``: the walk's when the field was read at every save, else the
    coarser grid the producer recorded (``field_sample_every`` saves per field sample)."""
    return int(pm.get("n_t", n_t)), float(pm.get("dt", dt))


def _compile_effective(Geff, dt_pack, K, n_t, gyromagnetic_ratio=GAMMA):
    """``W`` from per-save weights already on the pack grid (:func:`_replay_kernel.effective_gradient`)."""
    from .compression import bridge_projection
    W = bridge_projection(np.asarray(Geff, np.float64), int(n_t), K)              # (n_meas, K+2, n_c)
    return (gyromagnetic_ratio * float(dt_pack) * W).reshape(W.shape[0], (K + 2) * W.shape[2]).T


def surface_logweight(arrays, rho_over_D, chan_meta=None, chi_hat=None):
    """Per-walker surface log-weight ``(rho/D) * sum_t chi(t) ell_i(t)`` from the C2 channel.

    C2 is stored in the bridge form (``blt_bridge_dst`` + the two exact endpoints), so the
    UNGATED total contact is ``blt_endpoint`` read directly -- it is the exact cumulative
    ``L(T)``, not something reconstructed from bands, which is the whole reason the endpoint is
    held exactly. A coherence gate contracts the bridge with the gate's sine transform
    (:func:`~dmipy_sim.replay.compression.surface_logweight_bridge`): no per-save series is built.

    Takes the pack's ``arrays`` rather than one tensor: the channel is three tensors now, and a
    signature that accepted just the coefficient block invited passing the wrong one.
    """
    from .compression import surface_logweight_bridge, has_c2
    if not has_c2(arrays):
        raise ValueError(
            "surface relaxivity was requested but this pack carries no C2 channel "
            "(no 'blt_bridge_dst'). A pack written before the C2 bridge form stored "
            "'blt_dct_coeffs', which is retired -- re-encode it. Returning the signal without "
            "the requested attenuation would be a plausible wrong number.")
    if chi_hat is None:
        return float(rho_over_D) * np.asarray(arrays["blt_endpoint"], np.float64)
    meta = dict(chan_meta or {})
    meta.setdefault("n_t", int(np.asarray(chi_hat).shape[0]))
    meta.setdefault("K", _cx_bands_K(arrays, meta))
    return surface_logweight_bridge(arrays, meta, rho_over_D, chi_hat)          # the bridge contracted, never decoded


def replay_signal(pack, W, *, rho_over_D=0.0, chi_hat=None, complex_signal=False):
    """Replay a compiled scheme ``W`` (from :func:`compile_scheme`) against ``pack`` (a :class:`ReplayPack`
    or a plain arrays dict). Returns ``E`` per measurement (magnitude unless ``complex_signal``).

    ``rho_over_D`` > 0 activates the exact surface-relaxivity replay via the pack's boundary local time
    (a per-walker signal loss decaying the whole signal, including ``b=0``); ``chi_hat`` coherence-gates it.
    """
    a = pack.arrays if isinstance(pack, ReplayPack) else pack
    from .compression import read_position_coeffs
    from .compression import require_position_method
    if isinstance(pack, ReplayPack):
        require_position_method(pack.method)
    C = read_position_coeffs(a, dtype=np.float64)
    N_w = C.shape[0]
    w0 = np.asarray(a.get("spin_weights", np.ones(N_w)), np.float64)
    surface_logw = None
    if rho_over_D:
        # asked for, so it must happen: a missing C2 channel raises inside surface_logweight
        # rather than being skipped. The previous form looked up a key the bridge rename
        # retired, so `rho_over_D` was silently ignored and callers got an unattenuated signal.
        cm = ((pack.meta.get("compression", {}).get("channels", {}) or {}).get("boundary_local_time")
              if isinstance(pack, ReplayPack) else None)
        surface_logw = surface_logweight(a, rho_over_D, cm, chi_hat)
    return replay_coefficients(C, w0, W, surface_logw=surface_logw, complex_signal=complex_signal)


def replay_coefficients(position_coeffs, spin_weights, W, *, surface_logw=None, complex_signal=False):
    """The compiled replay on host arrays: ``E = <w_eff exp(i C W)> / <w>`` for the position coefficients
    ``C`` ``(n_w, K+2, n_c)`` (:func:`compression.read_position_coeffs`, any subset of the stored axes so long
    as ``W`` was compiled from the matching waveform components), the walkers' weights ``w`` and the compiled
    scheme ``W`` ``(n_c (K+2), n_meas)`` from :func:`compile_scheme`. ``surface_logw`` is the per-walker
    surface-relaxivity log-weight from :func:`surface_logweight`: a signal LOSS, so it weights the numerator
    while the denominator stays ``<w>`` (``E(b=0) < 1``). The one host kernel every consumer -- the pack's
    ``replay``, a fit's per-voxel forward, a lookup-table build -- evaluates; :func:`replay_signal_jax` is its
    traced twin."""
    C = np.asarray(position_coeffs, np.float64)
    N_w, K, n_c = C.shape
    W = np.asarray(W)
    if W.shape[0] != K * n_c:
        raise ValueError(
            f"compiled scheme has {W.shape[0]} rows for {K * n_c} stored coefficients "
            f"({K} per axis = 2 endpoints + {K - 2} bands, {n_c} axes). Compile with "
            f"compile_scheme(G, dt, pack.K, n_t=pack.n_t) from the same components -- passing the stored "
            f"width instead of pack.K produces a scheme that multiplies cleanly and means nothing.")
    w0 = np.asarray(spin_weights, np.float64)
    phi = C.reshape(N_w, K * n_c) @ W                          # (N_w, n_meas)
    w_eff = w0 if surface_logw is None else w0 * np.exp(np.asarray(surface_logw, np.float64))
    S = (w_eff[:, None] * np.exp(1j * phi)).sum(0) / w0.sum()
    return S if complex_signal else np.abs(S)


def replay_signal_jax(position_coeffs, spin_weights, W, *, surface_logw=None):
    """JAX/autodiff twin of :func:`replay_signal` -- differentiable in the compiled scheme ``W``
    (hence in the waveform ``G`` that produced it) and jittable. ``position_coeffs`` is
    ``(n_walkers, K+2, n_axes)`` -- two endpoints then sine bands -- and ``W`` must come from
    :func:`compile_scheme`, which emits the matching row order.

    ``surface_logw`` is the per-walker surface log-weight from :func:`surface_logweight` (it does
    not depend on ``W``, so it is computed once on the host and passed in). Returns the complex
    signal (take ``abs`` for magnitude). This is the forward a gradient-based waveform/B1
    optimizer differentiates through."""
    import jax.numpy as jnp
    C = jnp.asarray(position_coeffs)
    N_w, K, _ = C.shape
    phi = C.reshape(N_w, K * C.shape[2]) @ jnp.asarray(W)
    w0 = jnp.asarray(spin_weights)
    w_eff = w0 if surface_logw is None else w0 * jnp.exp(jnp.asarray(surface_logw))
    return (w_eff[:, None] * jnp.exp(1j * phi)).sum(0) / w0.sum()


def replay_batch_jax(position_coeffs, spin_weights, W_batch, *, surface_logw=None):
    """:func:`replay_signal_jax` over a batch of compiled schemes ``W_batch`` ``(n_batch, n_c (K+2), n_meas)``
    -- one candidate orientation or knob setting per entry -- as one traced call: ``(n_batch, n_meas)``
    magnitudes. What a fit evaluates over a grid of poses in one device call."""
    import jax
    import jax.numpy as jnp
    fn = lambda W: jnp.abs(replay_signal_jax(position_coeffs, spin_weights, W, surface_logw=surface_logw))
    return jax.vmap(fn)(jnp.asarray(W_batch))
