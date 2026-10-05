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
per-walker reweight by ``exp((rho2/D) * sum_t chi(t) ell_i(t))``, optionally coherence-gated by an
occupancy schedule ``chi`` (:func:`surface_logweight`). A tissue's ``rho1`` adds the longitudinal
term on the same channel, gated by the complement ``chi_parallel = active - chi`` (on while the
magnetisation is stored along B0, e.g. a stimulated echo's mixing time): the full contact weight is
``exp((rho2/D) * <chi, ell> + (rho1/D) * <chi_parallel, ell>)``, two calls of the same bridge
contraction summed -- no new channel, no new walk (dmipy-sim#574).
"""
import json
from dataclasses import dataclass
from functools import cached_property

import numpy as np
from .compression import c2_bands_K as _cx_bands_K

from ..constants import GAMMA
from ..run import Run, current
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
    """Read a replay pack into a :class:`ReplayPack` (arrays + metadata). A pack that embeds no substrate spec, or
    whose spec's pools, tiers and stored field channel disagree (:func:`declared_susceptibility_field`), is
    refused."""
    from safetensors import safe_open
    arrays = {}
    with safe_open(str(path), framework="numpy") as f:
        hdr = f.metadata() or {}
        meta = json.loads(hdr.get("rpk") or hdr.get("json") or "{}")
        for k in f.keys():
            arrays[k] = f.get_tensor(k)
    pack = ReplayPack(arrays, meta, source=str(path))
    pack.susceptibility_field
    return pack


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
        out.field_lmax = int(z["field_lmax"]) if "field_lmax" in z.files else 0   # a cache written before the field tier
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


def declared_susceptibility_field(meta, *, has_field):
    """The susceptibility field (``"present"`` / ``"absent"``) a pack's (or a layout's) metadata ``meta`` declares
    through its embedded substrate spec's pools, with ``has_field`` whether it stores the field channel (C3):
    :func:`~dmipy_sim.spec.substrate.susceptibility_field_of`, which refuses a pack without an embedded spec and
    every disagreement between magnetic pools, the ``"field"`` tier and the stored channel."""
    from ..spec.substrate import susceptibility_field_of
    return susceptibility_field_of(meta.get("substrate"), stores_field=has_field, name=meta.get("id"))


@dataclass(frozen=True)
class FieldTerm:
    """The susceptibility term of one replay, as :func:`field_term` resolves it: the static field ``B0`` (tesla) and
    its direction ``axis`` in the machine's frame, with the tissue's ``chi_iso`` / ``chi_aniso``. ``B0`` is
    ``None`` when the term is identically zero -- a substrate with nothing magnetic, a tissue with no
    susceptibility, or a field of zero tesla -- and every route then reads no field channel at all."""
    B0: object = None
    axis: tuple = (0.0, 0.0, 1.0)
    chi_iso: float = 0.0
    chi_aniso: float = 0.0

    @property
    def active(self):
        return self.B0 is not None


def field_term(declared, tissue, scanner):
    """The susceptibility term a substrate's field, a tissue and a scanner give together: the one place a tissue's
    ``chi_iso`` / ``chi_aniso`` meets a substrate's susceptibility field ``declared`` (``"present"`` / ``"absent"``:
    a pack's from its spec's pools, :func:`declared_susceptibility_field`; an analytic or inert phantom substrate's
    ``"absent"`` by kind, :data:`~dmipy_sim.phantom.substrates.FIELD_BY_KIND`) and a scanner's ``B0``. Every replay
    route resolves its field here -- :meth:`ReplayPack._prepare` for the routes that read a tissue with the walk,
    :meth:`~dmipy_sim.replay.study.Primitives.reduction_terms` for those that apply it to contracted primitives (the
    study, the columnar image, the shape-moment layout) -- so the rule holds on all of them:

    * ``"absent"``: nothing in the substrate is magnetic and its field is identically zero. A non-zero tissue
      ``chi_iso`` or ``chi_aniso`` is refused by name; any ``B0``, of any strength along any axis, is accepted
      and has no effect.
    * ``"present"``: a non-zero tissue chi needs a field to act in, so one given with ``scanner=None`` is refused.
      A zero or absent chi at any ``B0`` is a field term of zero.

    Returns a :class:`FieldTerm`, inactive (``B0`` None) whenever the term is zero."""
    from ..spec.substrate import SUSCEPTIBILITY_FIELD
    if declared not in SUSCEPTIBILITY_FIELD:
        raise ValueError(f"a substrate's susceptibility field is one of {SUSCEPTIBILITY_FIELD}; got {declared!r}")
    f = scanner_field(scanner)
    chi = {k: getattr(tissue, k, None) for k in ("chi_iso", "chi_aniso")}
    given = {k: float(v) for k, v in chi.items() if v is not None and float(v) != 0.0}
    if declared == "absent":
        if given:
            raise ValueError(
                f"the tissue gives {', '.join(f'{k} = {v:.4g}' for k, v in given.items())}, and this substrate "
                "declares its susceptibility field absent: nothing in it is magnetic, so there is no source for a "
                "susceptibility to act on. Replay it with a tissue without chi_iso / chi_aniso (a B0 is accepted and "
                "has no effect)")
        return FieldTerm(axis=f.axis)
    if not given:
        return FieldTerm(axis=f.axis)
    if f.B0 is None:
        raise ValueError(
            f"the tissue gives {', '.join(f'{k} = {v:.4g}' for k, v in given.items())} and no scanner field "
            "(scanner=None): a susceptibility acts only in a B0, and replaying it without one drops the substrate's "
            "field from the signal. Give scanner= (a ScannerLimits or a field strength in tesla), or a tissue without "
            "chi_iso / chi_aniso for the field-free signal")
    if float(f.B0) == 0.0:
        return FieldTerm(axis=f.axis)
    return FieldTerm(B0=float(f.B0), axis=f.axis, chi_iso=given.get("chi_iso", 0.0), chi_aniso=given.get("chi_aniso", 0.0))


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
    def load(cls, path, *, windows=None, **kw):
        """Read a ``.rpk`` file: a local path, or ``hf://owner/name/path/to/file.rpk`` fetched from the hub into the
        local cache and checked against the dataset's ``manifest.json`` (:func:`~dmipy_sim.replay.publish.fetch`).

        ``windows`` reads only the pack's leading windows (RPK.md 4.3: a prefix ``range(k)`` of window indices)
        from the hub, by HTTP byte range, through :func:`~dmipy_sim.replay.publish.fetch_windows` -- only ``hf://``
        URIs take it, since a local file is already whole and cheap to read entire. The pack this returns carries
        :attr:`windows_present`; :meth:`_windows` serves only what it holds, and a replay that reaches further is
        refused by name. Extra keywords (``url=``, ``revision=``) are :func:`~dmipy_sim.replay.publish.fetch_windows`'s."""
        from .publish import fetch, fetch_windows, is_hub_uri
        if windows is not None:
            if not is_hub_uri(path):
                raise ValueError(f"windows= reads a pack's leading windows from the hub by byte range; "
                                 f"{path!r} is not an hf:// URI")
            return read_rpk(fetch_windows(path, windows, **kw))
        if kw:
            raise TypeError(f"load() takes {sorted(kw)} only together with windows=")
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

    @property
    def windows_present(self):
        """How many of this walk's leading windows (``0..k-1``) this pack's arrays actually hold -- ``None`` for
        a whole pack, an ``int`` ``k`` for one built by :meth:`load`'s ``windows=range(k)`` (RPK.md 4.3). The
        walk's own window count, :attr:`n_segments`, stays the parent's regardless: it says how long the walk
        IS, not how much of it this pack was fetched to hold."""
        k = self.meta.get("windows_present")
        return None if k is None else int(k)

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
        """``[(pack, t0, n_t), ...]``: the windows of the walk this pack actually holds, each as a pack, the time
        its first save sits at on the walk's clock, and its saves; ``[(self, 0.0, n_t)]`` for a single-window
        pack. Bounded by :attr:`windows_present` for a pack loaded with ``windows=`` -- a caller that needs a
        window beyond it is refused by name before it is asked for here (:meth:`_compile`)."""
        n = self.n_segments
        if n == 1:
            return [(self, 0.0, int(self.n_t))]
        n_seg, dt = int(self.segments["n_t"]), float(self.dt)
        present = self.windows_present
        upto = n if present is None else min(int(present), n)
        return [(self.segment(i), i * (n_seg - 1) * dt, n_seg) for i in range(upto)]

    def _assert_windows_reach(self, T_acq):
        """Refuse, by name, an acquisition of duration ``T_acq`` (s, from the walk's start) that needs a window
        this pack was not loaded with (:attr:`windows_present`)."""
        present = self.windows_present
        if present is None or self.n_segments <= present:
            return
        reach = self._span(T_acq)[1]
        if reach > present:
            raise ValueError(f"this acquisition reaches window {reach - 1}; load windows=range({reach})")

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

    def extend(self, T_add, *, seed, out_path=None, require_gpu=None, envelope=None, device="auto"):
        """This pack lengthened by ``T_add`` seconds of new segments: the walk continued from its end on its own
        substrate with a fresh seed and appended (:func:`~dmipy_sim.replay.continuation.extend_pack`)."""
        from .continuation import extend_pack
        return extend_pack(self, T_add, seed=seed, out_path=out_path, require_gpu=require_gpu, envelope=envelope, device=device)

    def _decoded_channels(self, start=0, stop=None):
        """The per-save channels of the walk at saves ``start .. stop - 1`` (the whole walk by default), decoded window
        by window and joined on the shared saves: ``comp`` / ``bound`` tracks ``(n_w, n_t)``, the contact increments
        ``ell`` ``(n_w, n_t)`` and the path series ``(series (n_w, n_ch, n_tf), names)`` -- each ``None`` when the pack
        lacks the channel. The windows outside the saves are not read, and the bridge and path channels are
        evaluated at those saves alone; the path series holds the field samples its own grid places there."""
        from .bank import susc_path_decode
        from .compression import decode_occupancy, decode_boundary_bridge, decode_boundary_local_time
        ch = dict(self.meta.get("compression", {}).get("channels", {}) or {})
        out = dict(comp=None, bound=None, ell=None, path=None)
        comps, bounds, ells, series = [], [], [], []
        names = None
        start = int(start); stop = int(self.n_t) if stop is None else int(stop)
        for j, (w, _, n_seg) in enumerate(self._windows()):
            off = j * (n_seg - 1)                                      # the window's first save on the walk
            if off >= stop:
                break
            lo = max(start - off, 0 if j == 0 else 1)                  # the shared save is the previous window's
            hi = min(stop - off, n_seg)
            if lo >= hi:
                continue
            if "compartment" in ch:
                occ = decode_occupancy(w.arrays, ch["compartment"])
                comp = np.asarray(occ["comp"])
                comps.append(comp[:, lo:hi] if comp.ndim == 2 else comp)
                if "bound" in occ:
                    b = np.asarray(occ["bound"]); bounds.append(b[:, lo:hi] if b.ndim == 2 else b)
            if "boundary_local_time" in ch:
                bm = dict(ch["boundary_local_time"]); bm.setdefault("n_t", n_seg)
                if w.has_surface:
                    bm.setdefault("K", _cx_bands_K(w.arrays, bm))
                    ell = decode_boundary_bridge(w.arrays, bm, start=lo, n_cut=hi)
                else:
                    ell = np.asarray(decode_boundary_local_time(w.arrays, bm))[:, lo:hi]
                ells.append(np.asarray(ell))
            if "susceptibility_path" in ch:
                pm = ch["susceptibility_path"]
                n_tf, dt_f = _path_grid(pm, n_seg, float(self.dt))
                every = max(1, int(round(dt_f / float(self.dt))))
                f_lo, f_hi = -(-lo // every), -(-hi // every)          # the field samples at saves lo .. hi - 1
                if f_lo < f_hi:
                    ser, names = susc_path_decode(w.arrays, pm, n_w=self.n_walkers, start=f_lo, n_cut=f_hi)
                    series.append(ser)
        if comps:
            out["comp"] = np.concatenate(comps, axis=1) if comps[0].ndim == 2 else comps[0]
        if bounds:
            out["bound"] = np.concatenate(bounds, axis=1) if bounds[0].ndim == 2 else bounds[0]
        if ells:
            out["ell"] = np.concatenate(ells, axis=1)
        if series:
            out["path"] = (np.concatenate(series, axis=2), names)
        return out

    def _window_walk(self, start, stop):
        """Saves ``start .. stop - 1`` of this walk as the bank's master dict a fresh walk of that stretch would have
        recorded, and the path channel's decoded series there (``(series (n_w, n_ch, n_tf), names)`` or ``None``):
        the positions a :class:`~dmipy_sim.replay.compression.LazyWalk` decoded per walker range from the
        coefficients of the windows the stretch spans, never held whole (#449 item 3); the occupancy and the
        contact at those saves (the contact of the first save 0 when the stretch starts past the walk's first: it
        ends no step of the stretch); the weights, spec, frame, diffusivity and seed."""
        from .bank import seed_value
        from .compression import LazyWalk, decode_saves, read_position_coeffs
        start, stop = int(start), int(stop)
        dt = float(self.dt)
        decoded = self._decoded_channels(start=start, stop=stop)
        wp = dict(self.meta.get("walk_params", {}) or {})
        spans = []                                                     # (coefficients, saves of the window, first save read, end)
        for j, (w, _, n_seg) in enumerate(self._windows()):
            off = j * (n_seg - 1)
            if off >= stop:
                break
            lo = max(start - off, 0 if j == 0 else 1)
            hi = min(stop - off, n_seg)
            if lo < hi:
                spans.append((read_position_coeffs(w.arrays, dtype=np.float64), n_seg, lo, hi))
        n_w = spans[0][0].shape[0]

        def _positions(a, b):
            return np.concatenate([decode_saves(C[a:b], n_seg, lo, hi) for C, n_seg, lo, hi in spans], axis=1)

        m = dict(traj=LazyWalk(_positions, (n_w, stop - start, 3)), dt_traj=dt, T_max=(stop - start - 1) * dt,
                 walkers_shuffled=bool(self.meta.get("compression", {}).get("precision_tiers", {}).get("walkers_shuffled", False)),
                 seed=seed_value(wp.get("seed", 0)), substrate_frame=self.substrate_frame,
                 susceptibility_field=self.susceptibility_field)
        if "spin_weights" in self.arrays:
            m["w"] = np.asarray(self.arrays["spin_weights"], np.float64)
        if self.meta.get("substrate") is not None:
            m["substrate"] = self.meta["substrate"]
        if wp.get("diffusivity") is not None:
            m["D_intra"] = float(wp["diffusivity"])
        if decoded["comp"] is not None:
            m["comp"] = decoded["comp"]
            if decoded["bound"] is not None:
                m["bfrac"] = decoded["bound"]
        if decoded["ell"] is not None:
            ell = np.array(decoded["ell"], copy=True)
            if start > 0:
                ell[:, 0] = 0.0
            m["dlog_b"] = ell
        return m, decoded["path"]

    def resegment(self, T_seg, *, out_path, id=None, max_dropped_fraction=1e-3, envelope=None, tol=2.0):
        """This one-window walk stored in windows of ``T_seg`` seconds (RPK.md 4.3), so that a consumer holds only
        the windows its acquisitions reach (#525).

        The windows are a whole number of saves and share their boundary saves: ``n_seg - 1`` is the largest save
        count at most ``round(T_seg / dt)`` whose windows cover the walk up to a trailing remainder of at most
        ``max_dropped_fraction`` of its saves; the remainder is dropped (``saves_dropped``), and a walk that would
        leave more is refused. Every window is the parent's channels decoded on its saves and re-encoded by
        :func:`~dmipy_sim.replay.bank.build_replay_pack` as one window of its own (:meth:`_window_walk`): the positions
        per walker range from the parent's coefficients at the parent's bands per second, ``K_seg = round(K T_seg /
        T)``, doubled until window 0's certificate passes as :meth:`prefix` does; the occupancy runs cut at the
        boundary save; the contact as the window's own cumulative local time at the band window 0's surface
        certificate chooses; the path channel at the band and container
        :func:`~dmipy_sim.replay.bank.derive_susc_path_K` derives on window 0 from the decoded series. Window 0's
        bands are every window's. Each window's certificate (``fidelity.segments[i]``) is measured against the
        parent's decoded saves on it -- the re-encode error -- and the whole's is the bound over them
        (:func:`~dmipy_sim.replay.bank.combine_segment_fidelity`, ``certified = "bounded"``); a window outside
        ``tol`` times its floor is refused. ``provenance.resegmented`` records the parent (id, sha256 of its file,
        certificate), ``T_seg``, ``n_seg``, the bands and ``saves_dropped``.

        ``envelope`` is the battery every window is certified over: by default
        :func:`~dmipy_sim.replay.compression.default_envelope` with the parent's field strengths (``B0_list``) and
        the refocusing depth its path channel serves (``max_refocus_pulses``)."""
        import copy
        import hashlib
        from .bank import build_replay_pack, combine_segment_fidelity, join_segments
        from .compression import default_envelope
        from ..fields.strand_field import StrandFieldRecord
        if self.n_segments != 1:
            raise ValueError(f"this pack already stores its walk in {self.n_segments} windows of "
                             f"{float(self.segments['T']) * 1e3:.3f} ms; resegment reads a one-window walk")
        dt, n_t = float(self.dt), int(self.n_t)
        T = (n_t - 1) * dt
        steps = None
        for s_ in range(int(round(float(T_seg) / dt)), 1, -1):
            if (n_t - 1) // s_ >= 2 and (n_t - 1) % s_ <= float(max_dropped_fraction) * n_t:
                steps = s_
                break
        if steps is None:
            raise ValueError(f"a {T * 1e3:.3f} ms walk of {n_t} saves (dt {dt * 1e6:.2f} us) has no division into windows of at "
                             f"most {float(T_seg) * 1e3:.3f} ms that drops at most {max_dropped_fraction:g} of its saves; "
                             "give a T_seg that divides it, or a larger max_dropped_fraction")
        n = (n_t - 1) // steps
        dropped = (n_t - 1) - n * steps
        ch = dict(self.meta.get("compression", {}).get("channels", {}) or {})
        pm = ch.get("susceptibility_path")
        env = dict(envelope) if envelope is not None else default_envelope()
        if envelope is None:
            B0s = ((self.meta.get("replay_envelope") or {}).get("acquisition") or {}).get("B0_list")
            if B0s:
                env["B0_list"] = list(B0s)
            if pm is not None and pm.get("max_refocus_pulses"):
                env["max_refocus_pulses"] = int(pm["max_refocus_pulses"])
        if (ch.get("susceptibility_grid") or {}).get("arrays_in_pack"):
            raise ValueError("this pack's field tier samples its stored grid at the decoded positions, which holds only while "
                             "the positions are lossless; re-encoded windows are not, so it is not resegmented")
        containers = dict(position_container=_container_of(self.meta["compression"].get("container")))
        c2 = ch.get("boundary_local_time")                            # the parent's contact channel, or None
        if c2 is not None:
            containers.update(blt_container=_container_of(c2.get("container")),
                              blt_dtype=(np.float16 if c2.get("dtype") in (None, "bands") else np.dtype(c2["dtype"])))
        col = next((c for c in (ch.get("compartment") or {}).get("columns", []) if c["name"] == "comp"), None)
        crosses = col is not None and col["kind"] != "static"
        K = max(2, int(round(self.K * steps / (n_t - 1))))
        K_cap = min(int(self.K), steps - 1)                           # the bridge's interior: n_seg - 2 bands
        blt_K, path_K, path_bits = None, ("auto" if pm is not None else None), (int(pm.get("bits") or 16) if pm is not None else 8)
        spec = self.substrate
        chi_iso = None
        if spec is not None:
            from ..spec.tissue import Tissue
            chi_iso = Tissue.from_spec(spec).chi_iso
        stamp = id or f"{self.meta.get('id')}/seg{steps * dt * 1e3:.0f}ms"
        packs, tried = [], []
        for i in range(n):
            m, path = self._window_walk(i * steps, (i + 1) * steps + 1)
            if path is not None:
                series, names = path
                m["susc_field_samples"] = np.ascontiguousarray(np.transpose(series, (0, 2, 1)), np.float32)   # (n_w, n_t, n_ch)
                del series, path
                m["susc_field_sampler"] = StrandFieldRecord(dict(ch.get("susceptibility_grid") or {}, channels=list(names)))
                m["susc_chi_iso"] = chi_iso
            while True:
                pk = build_replay_pack(m, id=stamp, license=self.license, citation=self.citation, K=K, blt_temporal_K=blt_K,
                                       susc_path_K=path_K, susc_path_bits=path_bits, envelope=env, tol=tol,
                                       segment_T=steps * dt, _occupancy_runs=crosses, _window_of_plan=True, **containers)
                fid = pk.meta["fidelity"]
                if i > 0 or fid["err_max"] <= tol * fid["floor_max"] or K >= K_cap:
                    break
                tried.append(K)
                K = min(2 * K, K_cap)
            if fid["err_max"] > tol * fid["floor_max"]:
                raise ValueError(f"window {i} of {n} re-encoded at K = {K} does not reproduce the parent's decoded saves within "
                                 f"{tol:g} x its Monte-Carlo floor (error {fid['err_max']:.3g}, floor {fid['floor_max']:.3g})")
            if i == 0:
                c2 = (pk.meta["compression"].get("channels") or {}).get("boundary_local_time")
                if c2 is not None:
                    if c2.get("mode") != "bridge_dst":
                        raise ValueError(f"window 0's contact channel chose {c2.get('mode')!r}; a pack in windows keeps the bridge form")
                    blt_K = int(c2["K"])
                pw = (pk.meta["compression"].get("channels") or {}).get("susceptibility_path")
                if pw is not None:
                    path_K, path_bits = int(pw["K"]), int(pw["bits"])
            (pk.meta["compression"].get("channels") or {}).pop("susceptibility_grid", None)
            packs.append(pk)
            del m
        fid = combine_segment_fidelity([pk.meta["fidelity"] for pk in packs])
        wp = self.meta.get("walk_params", {}) or {}
        arrays, meta = join_segments(packs, fidelity=fid, walks=[dict(first=0, last=n - 1, seed=wp.get("seed"))],
                                     walkers_shuffled=bool(self.meta.get("compression", {}).get("precision_tiers", {}).get("walkers_shuffled", False)))
        if "susceptibility_grid" in ch:
            meta["compression"]["channels"]["susceptibility_grid"] = copy.deepcopy(ch["susceptibility_grid"])
        if pm is not None:
            meta["compression"]["channels"]["susceptibility_path"]["sampling"] = "the parent's path channel decoded on the window"
        src = self.source
        if src is not None and not str(src).startswith("hf://"):
            h = hashlib.sha256()
            with open(src, "rb") as f:
                for blk in iter(lambda: f.read(1 << 24), b""):
                    h.update(blk)
            parent_sha = h.hexdigest()
        else:
            parent_sha = None
        pf = self.meta.get("fidelity") or {}
        meta["provenance"] = dict(copy.deepcopy(self.meta.get("provenance") or {}), resegmented=dict(
            parent_id=self.meta.get("id"), parent_sha256=parent_sha, parent_digest=(self.digest if parent_sha is None else None),
            parent_fidelity=dict(err_max=pf.get("err_max"), floor_max=pf.get("floor_max"), certified=pf.get("certified")),
            T_seg=float(steps * dt), n_seg=int(n), saves_per_window=int(steps + 1), saves_dropped=int(dropped),
            parent_K=int(self.K), K=int(K), K_tried=tried + [int(K)], blt_K=blt_K,
            path_K=(None if pm is None else path_K), path_bits=(None if pm is None else path_bits),
            contact=(None if c2 is None else dict(
                inherited=True, parent_K=int(c2["K"]), parent_T=float((n_t - 1) * dt),
                note="the parent's contact channel is band-limited over its whole walk; a window's cumulative local "
                     "time is the parent's decoded one cut there, so the parent's truncation (a windowed increment can "
                     "exceed zero, which the true one never does) is inherited, not re-measured: the pack's "
                     "replay_envelope.tissue.rho2_over_D_max states up to which rho2 / D the windowed contact lies within "
                     "the floor (#528). A build from the walk (bank._build_segmented) encodes each window's own.")),
            note="every window the parent's channels decoded on its saves and re-encoded; its certificate is the re-encode "
                 "error against the parent's decoded saves, on top of the parent's own certificate"))
        if id is not None:
            meta["id"] = id
        else:
            meta["id"] = stamp
        meta["license"], meta["citation"] = self.license, self.citation
        out = ReplayPack(arrays, meta, source=None)
        if out.has_surface:
            out.restate_surface_envelope(reference=self, tol=tol)
        out.save(out_path)
        return out

    @property
    def rho2_over_D_max(self):
        """The largest ``rho2 / D`` the contact tier serves (``replay_envelope.tissue.rho2_over_D_max``, RPK.md 8.7), or
        ``None`` when the pack states no bound. ``replay_envelope.tissue`` knows only this one key; any other
        key there is refused by name, naming what is accepted."""
        tissue = (self.meta.get("replay_envelope") or {}).get("tissue") or {}
        unknown = set(tissue) - {"rho2_over_D_max"}
        if unknown:
            raise ValueError(f"this pack's replay_envelope.tissue declares {sorted(unknown)}; it takes only rho2_over_D_max")
        return tissue.get("rho2_over_D_max")

    def _check_rho2(self, rho2_over_D):
        """Refuse a relaxivity beyond the contact tier's envelope (:attr:`rho2_over_D_max`), naming both."""
        top = self.rho2_over_D_max
        if top is not None and float(rho2_over_D) > float(top) * (1.0 + 1e-12):
            D = self.diffusivity
            at = "" if D is None else f" (rho2 = {float(top) * float(D):.3g} m/s at the walk's D = {float(D):.3g} m^2/s)"
            raise ValueError(f"rho2 / D = {float(rho2_over_D):.4g} 1/m is beyond this pack's contact envelope, rho2 / D <= "
                             f"{float(top):.4g} 1/m{at}: past it the stored contact band no longer gives a physical "
                             "attenuation within its certificate (replay_envelope.tissue.rho2_over_D_max)")

    def restate_surface_envelope(self, *, reference=None, tol=2.0, out_path=None):
        """State the contact tier at the envelope it serves (RPK.md 8.7): per window the largest ``rho2 / D`` at which
        the decoded contact gives a physical attenuation within ``tol`` floors
        (:func:`~dmipy_sim.replay.bank.surface_envelope`), the pack's ``replay_envelope.tissue.rho2_over_D_max`` the
        smallest over the windows, and every window's surface rows (``err_surface``, ``floor_surface``) restated up
        to that edge rather than at the battery's largest ``rho2``, its maxima re-read and the whole's certificate
        rebuilt from them. ``reference`` is the one-window pack the windows were encoded from, whose decoded contact
        is then the error's reference; without it the error is the unphysical gain alone. Only metadata changes;
        ``out_path`` writes the pack with its tensors as they are. ``provenance.surface_envelope_restated`` records
        the edges, the battery's top and the reference."""
        import copy
        from .bank import surface_envelope, restate_maxima, combine_segment_fidelity
        from .compression import decode_boundary_bridge, default_envelope
        if not self.has_surface:
            raise ValueError("this pack carries no contact channel (C2): there is no surface envelope to state")
        D = self.diffusivity
        if D is None:
            raise ValueError("the pack records no diffusivity, so rho2 / D cannot be stated")
        if reference is not None and reference.n_segments != 1:
            raise ValueError("the reference is the one-window pack the windows were encoded from")
        rho2_list = default_envelope().get("rho2_list") or [1e-5, 3e-5, 1e-4]
        hi = max(rho2_list) / float(D)
        w = np.asarray(self.spin_weights, np.float64)
        n_seg = int(self.segments["n_t"])
        reads = []                                                   # (decoded contact, reference sums) per window
        for i, (win, _, n_s) in enumerate(self._windows()):
            cm = dict((win.meta["compression"].get("channels") or {})["boundary_local_time"])
            cm.setdefault("n_t", n_s)
            cm.setdefault("K", _cx_bands_K(win.arrays, cm))
            ref = None
            if reference is not None:
                k0 = i * (n_seg - 1)
                ref = np.asarray(reference._decoded_channels(start=k0 + 1, stop=k0 + n_s)["ell"], np.float64).sum(axis=1)
            reads.append((decode_boundary_bridge(win.arrays, cm), ref))
        rows = [surface_envelope(ell, w, rho2_over_D_hi=hi, reference=ref, tol=tol) for ell, ref in reads]
        edge = float(min(r[0] for r in rows))
        restated = [surface_envelope(ell, w, rho2_over_D_hi=edge, reference=ref, tol=np.inf)[1:] if edge > 0 else (0.0, 0.0)
                    for ell, ref in reads]
        del reads
        fid = copy.deepcopy(self.meta.get("fidelity") or {})
        segs = fid.get("segments")
        if segs:
            for f_i, (e, f) in zip(segs, restated):
                f_i.update(err_surface=e, floor_surface=f, surface_rho2_over_D_max=edge)
                restate_maxima(f_i)
            if fid.get("certified") == "bounded":
                fid = combine_segment_fidelity(segs)
            else:
                fid.update(err_surface=float(sum(e for e, _ in restated)), floor_surface=float(max(f for _, f in restated)), segments=segs)
                restate_maxima(fid)
        else:
            fid.update(err_surface=restated[0][0], floor_surface=restated[0][1])
            restate_maxima(fid)
        fid["surface_rho2_over_D_max"] = edge
        self.meta["fidelity"] = fid
        env = self.meta.setdefault("replay_envelope", {})
        env.setdefault("tissue", {})["rho2_over_D_max"] = edge
        self.meta.setdefault("provenance", {})["surface_envelope_restated"] = dict(
            rho2_over_D_max=edge, rho2_max_m_per_s=edge * float(D), D=float(D), per_window=[r[0] for r in rows],
            battery_rho2_over_D_max=hi, tol=float(tol), grid_points=len(np.geomspace(1.0, hi, 256)),
            reference=(None if reference is None else reference.meta.get("id")))
        self._digest = None
        if out_path is not None:
            self.save(out_path)
        return edge

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
    def susceptibility_field(self):
        """The pack's susceptibility field: ``"present"`` -- some pool of its embedded spec is magnetic, the spec
        lists the ``"field"`` tier and the pack carries the field channel (C3) -- or ``"absent"`` -- nothing in its
        substrate is magnetic and the field is identically zero. Read through :func:`declared_susceptibility_field`,
        which refuses a pack without an embedded spec and every disagreement; it is the first thing every replay
        route reads. A view of a layout (:meth:`~dmipy_sim.replay.columnar.ColumnarPack.view`) that leaves the
        layout's field rows unread holds the channel all the same (``meta["view"]["field"] == "unread"``); a
        replay that needs the rows then refuses by name."""
        unread = (self.meta.get("view") or {}).get("field") == "unread"
        return declared_susceptibility_field(self.meta, has_field=self.has_field or unread)

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
        ``replay(seq, tissue=pack.nominal, scanner=pack.nominal_field_T)``; ``None`` for a pack without a spec.
        Its ``T2`` / ``T1`` map the pools that declare one by name (``None`` when none does), so a spec that
        declares a T2 in some pools only gives a tissue the replay refuses, naming the rest."""
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
        from .compression import decode, require_position_method
        if self.n_segments > 1:
            return np.concatenate([w.positions()[:, (0 if j == 0 else 1):] for j, (w, _, _) in enumerate(self._windows())], axis=1)
        cx = self.meta.get("compression", {})
        meta = {"method": require_position_method(cx.get("method")), "K": int(cx.get("K", 0)),
                "n_t": int(cx.get("n_t") or self.n_t)}
        return np.asarray(decode(self.arrays, meta), np.float64)

    def _by_pool(self, values, what):
        """A tissue's per-pool ``T2`` / ``T1`` as the list by pool id over every pool of the embedded spec:
        ``None`` for none, else the ``{pool name: seconds}`` mapping resolved through the spec, which refuses a
        missing pool, an unknown name, a number for every pool, and any per-pool value on a pack without a spec
        (RPK.md 8.5: such a pack replays the gradient alone)."""
        if values is None:
            return None
        spec = self.substrate
        if spec is None:
            raise ValueError(f"{what} was given by pool but this pack embeds no substrate spec, so it has no pool "
                             f"names: it replays the gradient alone (RPK.md 8.5)")
        if not isinstance(values, dict):
            raise ValueError(f"{what} on a pack is {{pool name: seconds}} over every pool of its spec "
                             f"{[p.name for p in spec.pools]}; one number is the closed form's (got {values!r})")
        from ..spec.tissue import _by_pool_id
        return [float(v) for v in _by_pool_id(spec, values, what)]

    def replay(self, waveform, *, tissue=None, scanner=None, orientation=None, compartment=None, complex_signal=False):
        """The signal of ``waveform`` on this pack: the gradient always, and every other tier whose inputs are given
        and which the pack carries.

        ``waveform`` is a :class:`~dmipy_sim.acquisition.scanner_sequence.ScannerSequence`: its ``G_eff``, the
        effective gradient, is what this route integrates, and its RF schedule says where the 180 is (a schedule
        without one is a gradient echo). It is read on the pack grid (``n_t`` samples of ``dt``, zero outside
        the waveform).

        Three things describe a replay setting, each stated once:

        * ``tissue`` -- **what the material is**: a :class:`~dmipy_sim.spec.Tissue` (pool T2 / T1, the walls'
          transverse ``rho2`` and longitudinal ``rho1``, the bulk D, the field source's chi) or ``None``, the
          bare diffusion signal. T2 / T1 are
          ``{pool name: seconds}`` over every pool of the embedded spec, ``inf`` for no decay. ``pack.nominal``
          is the embedded spec's values, so a paper's replay is ``replay(seq, tissue=pack.nominal,
          scanner=pack.nominal_field_T)``; ``pack.nominal.replace(T2={"intra": 0.08})`` changes one pool.
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
        occupancy channel; **surface relaxivity** (C2) with a rho2 and/or a rho1, scaled by the walk's D (the
        tissue's, else the pack's recorded one), on the boundary local time -- rho2 gated by the coherence
        (transverse) and rho1 by its complement (stored along B0, e.g. a stimulated echo's mixing time); **field** (C3) with a chi in the tissue and a field on
        the scanner, on the path channel (or the stored basis sampled along the decoded path), the 180 the
        waveform's own. A tier whose inputs are given but which the pack does not carry raises rather than
        returning a plausible number.

        ``compartment`` restricts the ensemble mean to one pool id (or a boolean walker mask).
        """
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
        E = _signal_factor(self._walker_phases(P, waveform), P["voxel"])
        S = (P["pathway"] * P["ew"][:, None] * E).sum(0) / P["norm"]
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
            return w, P["pathway"] * P["ew"], _signal_factor(self._walker_phases(P, waveform), P["voxel"])
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
        if P["B0"] is not None:
            raise ValueError(
                "weights= cannot be combined with an active susceptibility field yet. The field branches of "
                "this contraction rebuild the gradient from the nominal sequence and do not read supplied "
                "weights, so the result would be bit-identical to passing none -- including for weights of "
                "zero. That is silently wrong in exactly the low-field case this exists for (dmipy-sim#369)")
        return W

    def _walker_phases(self, P, waveform):
        """``(n_w, n_meas)`` accumulated phase of every walker under the prepared acquisition ``P``: the gradient
        from :func:`_band_phase`, and with a field the path channel contracted under the pose's field direction
        (the grid route, a pack without the path channel, samples the stored basis along the decoded path).

        The gradient term is :func:`_band_phase` and nothing else, so it is the same number
        :func:`~dmipy_sim.replay.study.walker_primitives` contracts, on every route and at every ``b``
        (dmipy-sim#484)."""
        from ._replay_kernel import field_gate
        n_w, dt = P["n_w"], P["dt"]
        phi = _band_phase(P, W=P.get("W"))                                            # (n_w, n_meas)
        if P["B0"] is None:                                                          # a field term of zero (field_term)
            return phi
        from ..fields.susceptibility_field import assemble_field, sample_grid
        ch, b0_dir, B0, chi_aniso = P["ch"], P["b0_dir"], P["B0"], P["chi_aniso"]
        gm = ch["susceptibility_grid"]
        chi_i = float(P["chi_iso"])
        if ch.get("susceptibility_path") is not None:                                # the path route: every term a contraction
            from ..fields.hollow_cylinder import contract
            Psi, names = _path_field_channels(P, waveform)
            aniso = chi_aniso if (_has_aniso(gm, names) and chi_aniso) else 0.0
            return phi + contract(Psi, b0_dir, B0=float(B0), chi_iso=chi_i, chi_aniso=aniso)[:, None]
        from .bank import grid_basis_of, grid_periodic_of                            # the grid route samples the field along the path
        phi_x = np.zeros(n_w)
        for seg, t0, n_s in P["windows"]:
            dB = sample_grid(assemble_field(grid_basis_of(seg.arrays, gm), b0_dir, B0=float(B0), chi_iso=chi_i, chi_aniso=chi_aniso),
                             seg.positions(), np.asarray(gm["origin"], float), gm["voxel_size"], periodic=grid_periodic_of(gm))
            phi_x = phi_x + GAMMA * dt * (dB * field_gate(waveform, n_s, dt, t0=t0)[None, :]).sum(1)
        return phi + phi_x[:, None]

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

            # the Bloch route reads a rate as 1/T: "no decay in this pool" is an infinite time
            def per_pool(v, what):
                out = self._by_pool(v, what)
                if out is None:
                    return [np.inf] * n_ids
                if len(out) < n_ids:
                    raise ValueError(f"the compartment channel addresses pool id {n_ids - 1} and the embedded spec "
                                     f"declares {len(out)} pools: the pack is inconsistent")
                return out
            relax = dict(T2_per_comp=per_pool(T2v, "T2"), T1_per_comp=per_pool(T1v, "T1"))
        if P["rho1"] is not None and float(P["rho1"]) != 0.0:
            # the vector-Bloch route propagates the actual M = (Mx, My, Mz) through the real pulses, so Mxy and
            # Mz are already separate at every step; this route's `surface_relaxivity` (below) attenuates Mxy
            # only, the same transverse-only mechanism the forward engine uses (engine/bloch.py). Extending it to
            # a longitudinal wall term means a parallel attenuation on the Mz deviation, a distinct forward-style
            # mechanism from the C2 scalar/closed-form routes this knob otherwise reaches -- refused rather than
            # silently dropped (dmipy-sim#574).
            raise ValueError("replay_bloch does not apply rho1 (longitudinal surface relaxivity): it propagates "
                             "the magnetisation vector through the real pulses and attenuates Mxy only, like the "
                             "forward engine; use replay() for the C2 scalar route, which applies rho1")
        surface = None
        if P["rho2"] is not None and float(P["rho2"]) != 0.0:
            D_walk = self.diffusivity if P["D"] is None else P["D"]
            if D_walk is None:
                raise ValueError("rho2 needs the walk's diffusivity: the pack did not record it, pass D=")
            if not self.has_surface:
                raise ValueError("surface relaxivity was requested but this pack carries no C2 channel")
            self._check_rho2(float(P["rho2"]) / float(D_walk))
            surface = dict(surface_relaxivity=float(P["rho2"]), D=float(D_walk))
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
        if P["B0"] is None:
            return np.zeros((P["n_w"], self.n_t))                                     # a field term of zero (field_term)
        gm = P["ch"]["susceptibility_grid"]
        pm = P["ch"].get("susceptibility_path")
        if pm is not None:
            b, _ = susc_path_decode(self.arrays, pm, n_w=P["n_w"])
            return susc_path_field(b, P["b0_dir"], B0=float(P["B0"]), chi_iso=float(P["chi_iso"]),
                                   chi_aniso=P["chi_aniso"], has_aniso=bool(gm.get("has_aniso")))
        from .bank import grid_basis_of, grid_periodic_of
        return sample_grid(assemble_field(grid_basis_of(self.arrays, gm), P["b0_dir"], B0=float(P["B0"]), chi_iso=float(P["chi_iso"]),
                                          chi_aniso=P["chi_aniso"]),
                           pos, np.asarray(gm["origin"], float), gm["voxel_size"], periodic=grid_periodic_of(gm))

    def _prepare(self, waveform, *, tissue, scanner, orientation, compartment, relaxation=True, surface=True,
                 pathway=True):
        """Everything a replay resolves before it reads positions: the waveform's exact per-save weights (rotated
        into the substrate frame when a pose is given), the tissue's values (none for ``None``), the susceptibility
        term :func:`field_term` resolves from the pack's declared field, the tissue's chi and the scanner's B0 (``B0``
        None when that term is zero; its direction the machine's B0 axis turned by the pose), the per-walker weights
        with the relaxation and surface terms applied, and the compartment selection.

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
        from .compression import require_position_method, decode_occupancy
        from ._replay_kernel import effective_gradient, bin_gate
        require_position_method(self.method)
        # two gradients: the PHYSICAL one (``G``, for the vector-Bloch route, which applies the pulses itself)
        # and the EFFECTIVE one (``G_eff``, for the scalar routes, the pulses folded in)
        waveform = waveform.waveform if hasattr(waveform, "waveform") else waveform
        if not isinstance(waveform, ScannerSequence):
            raise TypeError(f"a replay takes a ScannerSequence (a bare gradient array says nothing about its pulses); "
                            f"got {type(waveform).__name__}")
        self._check_band(waveform)
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
        T2, T1, rho2, rho1, D = t.T2, t.T1, t.rho2, t.rho1, t.D
        field = field_term(self.susceptibility_field, tissue, scanner)   # the one rule of tissue chi x pack field x B0
        if field.active and not self.has_field:
            raise ValueError(f"pack {self.meta.get('id')!r} declares its susceptibility field present and this replay "
                             "has a field term, but the pack holds no field channel (C3) here: read it with its field "
                             "modes (ColumnarPack.view(modes=...)) or replay the pack whole")
        B0, b0_dir = field.B0, field.axis                            # the MACHINE's field; the pose turns it
        chi_iso, chi_aniso = field.chi_iso, field.chi_aniso
        if orientation is not None:
            R = self.pose_rotation(orientation)
            G, G_eff = G @ R, G_eff @ R                                   # R^T g per sample: stored coordinates
            b0_dir = tuple(np.asarray(R, float).T @ np.asarray(b0_dir, float))
        Geff = effective_gradient(G_eff, dt_wf, n_t, dt)                 # exact per-save weights of the effective gradient
        # the windows of the walk (RPK.md 4.3): every tier is a sum over them, each window reading the acquisition
        # from where it sits on the walk's clock; a single window is the walk itself
        # only the windows the acquisition reaches are read: a window whose first save sits at or beyond the readout
        # contributes nothing, and is not touched -- a short acquisition on a long pack reads its first windows alone
        T_acq = _duration(waveform)
        self._assert_windows_reach(T_acq)
        windows = self._windows()[:self._span(T_acq)[1]]
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
            T2v = self._by_pool(T2, "T2"); T1v = self._by_pool(T1, "T1")
            if T2v is None:
                T2v = [np.inf] * n_ids                                # no T2 decay, T1 only
            if T1v is None:
                T1v = [np.inf] * n_ids                                # no T1 term
            if len(T2v) < n_ids or len(T1v) < n_ids:
                raise ValueError(f"the compartment channel addresses pool id {n_ids - 1} and the embedded spec "
                                 f"declares {min(len(T2v), len(T1v))} pools: the pack is inconsistent")
            for (seg, _, _), (chi_s, act_s) in zip(windows, window_gates):
                logw = logw + relaxation_logweight_runs(seg.arrays, col, T2v, T1v, dt, chi_s, act_s)   # on the runs, never a track
        if rho2 is not None and float(rho2) != 0.0 and surface:
            D_walk = self.diffusivity if D is None else D
            if D_walk is None:
                raise ValueError("rho2 needs the walk's diffusivity: the pack did not record it, pass D=")
            self._check_rho2(float(rho2) / float(D_walk))
            for (seg, _, _), (chi_s, _) in zip(windows, window_gates):
                logw = logw + surface_logweight(seg.arrays, float(rho2) / float(D_walk),
                                                ch.get("boundary_local_time"), chi_s)      # raises without C2
        if rho1 is not None and float(rho1) != 0.0 and surface:
            # the SAME bridge contraction (surface_logweight), gated by the complement chi_parallel = active - chi:
            # the contact channel is one series, and the C2 gate rho2 * chi + rho1 * chi_parallel is linear in it, so
            # the longitudinal term is a second call summed in rather than a second channel (dmipy-sim#574).
            D_walk = self.diffusivity if D is None else D
            if D_walk is None:
                raise ValueError("rho1 needs the walk's diffusivity: the pack did not record it, pass D=")
            self._check_rho2(float(rho1) / float(D_walk))
            for (seg, _, _), (chi_s, act_s) in zip(windows, window_gates):
                chi_parallel_s = np.clip(act_s - chi_s, 0.0, None)
                logw = logw + surface_logweight(seg.arrays, float(rho1) / float(D_walk),
                                                ch.get("boundary_local_time"), chi_parallel_s)    # raises without C2
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
                    b0_dir=b0_dir, chi_iso=chi_iso, chi_aniso=chi_aniso, T2=T2, T1=T1, rho2=rho2, rho1=rho1, D=D,
                    chi=chi, active=active, G_eff_wf=G_eff, windows=windows, window_gates=window_gates)

    def _n_pool_ids(self, col):
        """How many pool ids the ``comp`` column addresses, read over every window."""
        if col["kind"] == "fraction":
            return 2
        key = "comp_static" if col["kind"] == "static" else "comp_rle_vals"
        return int(max(int(np.max(w.arrays[key])) for w, _, _ in self._windows())) + 1

    def pose_response(self, waveform, *, tissue=None, scanner=None, pose=None, compartment=None,
                      method="auto", keep=None, cache=None, backend="jax", device=None):
        """The pack's response over every pose of its substrate, for one acquisition: a :class:`PoseResponse` whose
        coefficients a voxel's orientation distribution contracts against (RPH.md 6).

        **Closed form, by default (#197).** For a single-direction encoding the response is a sum of plane waves in
        each walker's rotated moment, and its harmonics are the Rayleigh expansion, computed per walker with no
        rotation ever evaluated; with a field the response is that expansion times the field factor's, coupled
        with the real Clebsch-Gordan tables. The band follows the walkers' phase amplitudes by construction and
        nothing is chosen. ``keep = (lmax, nmax)`` restricts what is computed to what the composition retains: an
        ODF or peaks composition keeps ``n = 0``, a frame keeps everything; ``None`` in either slot means the
        response's own band.

        **A magnet's own gradient** (``ScannerSequence.with_background_gradient``) is a second plane wave in each
        walker's background moment, coupled to the first as the field factor is; the encoding beside it is what
        must be one direction (dmrai-lab/dmipy-sim#565).

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

        ``backend`` -- where the closed form's walker work runs: ``"jax"`` (JAX's device when it is a GPU, else
        numpy on the host) or ``"torch"`` on ``device`` (a torch device; CUDA when torch sees one, else its CPU),
        for a host that has PyTorch and no JAX device (dmrai-lab/dmipy-sim#603). Both compute the same expansion,
        its band and its misfit, to float32 rounding; the quadrature route is the same on both.
        """
        return self.pose_responses([waveform], tissue=tissue, scanner=scanner, pose=pose, compartment=compartment,
                                   method=method, keep=keep, cache=cache, backend=backend, device=device)[0]

    def pose_responses(self, waveforms, *, tissue=None, scanner=None, pose=None, compartment=None,
                       method="auto", keep=None, cache=None, backend="jax", device=None):
        """:meth:`pose_response` for a batch of acquisitions on this pack -- the encoding classes of a machine pass,
        or one per voxel -- as one pass over the walkers per field factor: the closed form takes the single-direction
        acquisitions of the batch that share a gate together (:meth:`_pose_coeffs_closed_many`), the others take the
        quadrature one by one. The knobs, the pose, the cache and the backend are as for :meth:`pose_response`;
        returns one :class:`PoseResponse` per acquisition, in order (dmrai-lab/dmipy-sim#449)."""
        if backend not in ("jax", "torch"):
            raise ValueError(f"backend is 'jax' or 'torch', got {backend!r}")
        kernels = "auto" if backend == "jax" else ("torch" if device is None else f"torch:{device}")
        view = self._at_tissue(tissue)
        if view is not self:
            return view.pose_responses(waveforms, tissue=tissue, scanner=scanner, pose=pose, compartment=compartment,
                                       method=method, keep=keep, cache=cache, backend=backend, device=device)
        waveforms = list(waveforms)
        R_s = _pose_matrix(pose)
        if R_s is not None:                                       # the acquisition in the specimen frame: what the
            from ..acquisition.waveforms import rotate_waveform   # expansion reads, in P and from the waveform itself
            waveforms = [rotate_waveform(wf, R_s.T) for wf in waveforms]
        if method not in ("auto", "closed", "quadrature"):
            raise ValueError("method is 'auto', 'closed' (the per-walker Rayleigh expansion) or 'quadrature'")
        with Run("pose_response", params=dict(id=self.id, n_acq=len(waveforms), n_meas=int(waveforms[0].n_meas), method=method,
                                        keep=(None if keep is None else [None if k is None else int(k) for k in keep]))):
            Ps, paths, out = [], [], [None] * len(waveforms)
            for c, wf in enumerate(waveforms):
                P = self._prepare(wf, tissue=tissue, scanner=scanner, orientation=None, compartment=compartment)
                if R_s is not None:
                    P["b0_dir"] = tuple(R_s.T @ np.array([0.0, 0.0, 1.0]))   # the bore's field, seen from the specimen
                path = None
                if cache is not None and cache is not False:
                    path = self._pose_cache_path(cache, P, wf, method, keep)
                    if path.exists():
                        out[c] = PoseResponse.load(path)
                Ps.append(P); paths.append(path)
            todo = [c for c in range(len(waveforms)) if out[c] is None]
            if todo and method != "quadrature":
                closed = self._pose_coeffs_closed_many([Ps[c] for c in todo], [waveforms[c] for c in todo], keep=keep,
                                                       device=kernels)
                for c, resp in zip(todo, closed):
                    if resp is None and method == "closed":
                        raise ValueError("the closed-form pose expansion needs a single-direction encoding on every measurement: "
                                         "this acquisition has a b-tensor or multi-axis waveform, so use method='quadrature'")
                    out[c] = resp
            for c in todo:
                if out[c] is None:
                    out[c] = self._pose_coeffs(Ps[c], waveforms[c], keep=keep)
                P = Ps[c]
                if np.any(P["voxel"] != 1.0):
                    # an unbalanced encoding: the voxel's factor scales each measurement's expansion, and its misfit with
                    # it. The pose does not enter -- k . r_v is the same dot product in the lab and in the substrate.
                    f = P["voxel"]
                    out[c].coeffs = out[c].coeffs * f[:, None]
                    m = np.asarray(out[c].misfit, float)
                    out[c].misfit = m * np.abs(f) if m.shape == f.shape else m * float(np.abs(f).max())
                if paths[c] is not None:
                    paths[c].parent.mkdir(parents=True, exist_ok=True)
                    out[c].save(paths[c])
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
        gate = waveform.effective_gate if waveform.gate is None else waveform.gate   # what the field accrues through
        h.update(np.ascontiguousarray(gate, np.float32).tobytes())
        h.update(repr((float(P["norm"]), P["B0"], tuple(np.round(np.asarray(P["b0_dir"], float), 12)), P["chi_iso"],
                       P["chi_aniso"], method, None if keep is None else tuple(keep),
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
        """The highest frequency the position bands resolve, ``K / (2 T)`` with ``T`` a window's duration (#199; a
        one-window pack's walk): the pack's temporal band stated as a frequency, which is what a scanner envelope is
        checked against and what a prefix and a resegmented pack keep."""
        return float(self.K) / (2.0 * float(self.segments["T"]))

    def waveform_band(self, waveform, *, tol=None):
        """The temporal band ``waveform`` needs on this pack, as ``(hz, bands, err)``: the fewest sine bands whose
        dropped phase moves the signal by no more than ``tol`` on every measurement, stated as the frequency
        ``bands / (2 T)`` beside the pack's :attr:`temporal_bandwidth_hz` (#277).

        The gradient phase of a walker is the waveform's projection on the bridge basis contracted with the
        walker's bands (:func:`~dmipy_sim.replay.compression.bridge_projection`); the bands beyond the pack's
        ``K`` are not stored, so the phase they would carry is lost silently. For a free path the bridge bands
        are independent with the Dirichlet Laplacian's variances, ``D dt / (2 sin^2(pi k / 2 N))``, the tail every
        restricted walk shares at short times, so the dropped phase variance per measurement is
        ``(gamma dt)^2 sum_{k > K} sum_d W_{m,k,d}^2 var_k`` at the walk's diffusivity, and a dropped independent
        phase under-attenuates the magnitude by half of it. ``tol`` defaults to the pack's certified Monte-Carlo
        floor, the unit of every other error the certificate states; a pack whose certificate states no floor
        (walkers that never moved) is held to the largest floor its ``n_walkers`` could have, ``1 / sqrt(n)``.
        The band is judged over the saves the waveform spans (:meth:`_span`): the windows its duration reaches
        on a pack stored in windows (RPK.md 4.3), the whole grid on a one-window pack, so a short acquisition on a
        long walk costs the projection of its own duration, not the walk's. ``bands`` is at most the span's own
        ``n - 2``; a waveform beyond that returns ``inf``.
        """
        from .compression import bridge_projection
        from ._replay_kernel import effective_gradient
        waveform = waveform.waveform if hasattr(waveform, "waveform") else waveform
        dt = float(self.dt)
        n_t, windows = self._span(_duration(waveform))
        T = (n_t - 1) * dt
        D = self.diffusivity
        if D is None:
            raise ValueError("the pack records no diffusivity, so the band a waveform needs cannot be judged")
        if tol is None:
            tol = float(self.meta.get("fidelity", {}).get("floor_max") or 0.0) or 1.0 / np.sqrt(float(self.n_walkers))
        K_big = n_t - 2
        Geff = effective_gradient(np.asarray(waveform.G_eff, np.float64), float(waveform.dt), n_t, dt)
        W = bridge_projection(Geff, n_t, K_big)[:, 2:, :]                          # (n_meas, K_big, 3)
        k = np.arange(1, K_big + 1)
        var = float(D) * dt / (2.0 * np.sin(np.pi * k / (2.0 * (n_t - 1))) ** 2)   # the free bridge's band variances
        per_band = (GAMMA * dt) ** 2 * (W ** 2).sum(axis=2) * var[None, :]           # (n_meas, K_big) phase variance
        tail = np.cumsum(per_band[:, ::-1], axis=1)[:, ::-1]                        # dropped variance beyond band k-1
        err = 0.5 * np.concatenate([tail, np.zeros((tail.shape[0], 1))], axis=1).max(axis=0)   # err[j]: keeping j bands
        ok = np.flatnonzero(err <= float(tol))
        if len(ok) == 0:
            return np.inf, np.inf, float(err[-1])
        bands = int(ok[0])
        return bands / (2.0 * T), bands, float(err[min(self.K * windows, K_big)])

    def _span(self, T_acq):
        """``(n, windows)``: the saves of this walk an acquisition of duration ``T_acq`` (s, from the walk's start)
        spans and the windows they make -- on a pack stored in windows (RPK.md 4.3) the whole windows whose start
        lies before ``T_acq`` (sharing their boundary saves), whether loaded or not; on a one-window pack the grid,
        ``(n_t, 1)``. An acquisition longer than the walk spans the walk."""
        n_t, dt = int(self.n_t), float(self.dt)
        if self.n_segments == 1:
            return n_t, 1
        steps = int(self.segments["n_t"]) - 1
        windows = sum(1 for i in range(int(self.n_segments)) if i * steps * dt < float(T_acq) * (1.0 - 1e-12)) or 1
        return windows * steps + 1, windows

    def _check_band(self, waveform):
        """Refuse a waveform whose gradient needs more temporal band than this pack stores over the saves it spans
        (#277; :meth:`waveform_band`): the replay would otherwise return a smooth, plausible, wrong signal. A pack
        that records no diffusivity (a synthetic master) cannot be judged and is not refused."""
        if self.diffusivity is None:
            return
        hz, bands, err = self.waveform_band(waveform)
        n_t, windows = self._span(_duration(waveform))
        if bands <= self.K * windows:
            return
        dt = float(self.dt)
        T = (n_t - 1) * dt
        need = (f"more than {(n_t - 2) / (2.0 * T):.4g} Hz, the limit of the pack's grid over the {T * 1e3:.4g} ms it spans"
                if not np.isfinite(hz) else f"{hz:.4g} Hz ({bands} bands over the {T * 1e3:.4g} ms it spans)")
        raise ValueError(f"the waveform's gradient reaches beyond this pack's temporal band: it needs {need} and the "
                         f"pack resolves {self.temporal_bandwidth_hz:.4g} Hz (K = {self.K} bands over {float(self.segments['T']) * 1e3:.4g} ms), "
                         f"which would drop {err:.2e} of the signal into phase the pack does not carry. Build a pack at "
                         f"this band or above (build_replay_pack(temporal_bandwidth_hz=...)), or replay a waveform "
                         f"within this one")

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
        dt, n_t = float(self.dt), int(self.n_t)
        T = (n_t - 1) * dt
        n_cut = int(round(float(TE) / dt)) + 1
        if n_cut < 3 or n_cut > n_t:
            raise ValueError(f"TE = {TE * 1e3:.3f} ms is not a prefix of this {T * 1e3:.3f} ms walk (dt {dt * 1e6:.1f} us): "
                             f"it needs at least three saves and at most the walk's {n_t}")
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
        from .bank import build_replay_pack, susc_path_encode_series, susc_path_series_fidelity
        dt, n_t = float(self.dt), int(self.n_t)
        T = (n_t - 1) * dt
        n_cut = int(round(float(TE) / dt)) + 1
        T_cut = (n_cut - 1) * dt
        K_new = int(K) if K is not None else max(2, int(np.ceil(self.K * T_cut / T)))
        ch = dict(self.meta.get("compression", {}).get("channels", {}) or {})
        # the prefix's positions are never held whole: the builder, its encoder and its certificate read them per
        # walker range, decoded from the parent's coefficients where they are read, window by window (#449 item 3)
        m, path = self._window_walk(0, n_cut)
        if path is None and self.susceptibility_field == "present":
            raise ValueError("this pack carries its field channel (C3) as the stored grid read at the decoded positions, "
                             "which holds only while the positions are lossless; a re-encoded prefix is not, so it is "
                             "not prefixed")
        blt_K = None
        if "boundary_local_time" in ch and self.has_surface:
            bm = dict(ch["boundary_local_time"])
            bm.setdefault("K", _cx_bands_K(self.segment(0).arrays, bm))
            blt_K = max(2, int(np.ceil(bm["K"] * T_cut / T)))
        path_series = None
        if path is not None:
            series, names = path
            _n_tf, _dt_f = _path_grid(ch["susceptibility_path"], int(self.segments["n_t"]), dt)
            path_series = (series, names,
                           max(2, int(np.ceil(int(ch["susceptibility_path"]["K"]) * T_cut / T))),
                           ch["susceptibility_path"].get("bits", 8))
        tried = []
        while True:
            prov = dict(provenance or {})
            prov["prefix"] = dict(parent_digest=self.digest, parent_id=self.meta.get("id"), parent_T_s=T, parent_K=int(self.K),
                                  TE_s=T_cut, K=K_new, K_tried=tried + [K_new],
                                  note="re-encoded from the parent's decoded prefix; the band starts at the parent's bands "
                                       "per second and doubles until the certificate passes")
            # m (a windowed master dict, not a PersistentWalk) carries no susc_field_basis key, so this build has no
            # field tier; the path series is added back below, re-encoded from the parent's decoded prefix
            pk = build_replay_pack(m, id=id or f"{self.meta.get('id')}/prefix-{T_cut * 1e3:.0f}ms", license=self.license,
                                   citation=self.citation, K=K_new, tol=tol, blt_temporal_K=blt_K, provenance=prov,
                                   segment_T=T_cut, _window_of_plan=True)    # a prefix is one window of its own duration
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

    def _pose_coeffs_closed_many(self, Ps, waveforms, keep=None, tol=1e-8, l_cap=64, direction_tol=None, device="auto"):
        """The pose expansion in closed form (#197) for a batch of acquisitions on this pack -- the encoding classes
        of a machine pass, or one class per voxel -- in ONE pass over the walkers: every acquisition's waveform
        groups lie along one group axis, so the moments, the Bessel values, the moment harmonics and the products
        against the field factor run once for all of them, in chunks of groups, and the lab-side assembly is per
        acquisition. The field factor is one per pass: the acquisitions of a pass share the inputs of the field factor
        (:func:`_field_factor_inputs`: the gate the field accrues through, the field and the tissue's chi), and a
        batch whose acquisitions do not is expanded in one pass per distinct factor. Returns one
        :class:`PoseResponse` per acquisition, ``None`` where an acquisition is not single-direction
        (dmrai-lab/dmipy-sim#449).

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

        **Nearly one direction.** A measurement whose waveform is one direction up to a residual -- a machine's
        Maxwell gradient through the ramps of a trapezoid, which goes as the square of the ramp where the encoding
        goes as the ramp -- is expanded on its principal direction when the residual's phase, contracted on every
        walker exactly as the moments are and bounded over every pose, moves the ensemble's signal by at most
        ``direction_tol`` (default a tenth of the ensemble's floor ``1 / sqrt(n_w)``): since ``|exp(i a) - exp(i b)|
        <= |a - b|``, the walkers' bounds weighted as the signal weighs them bound it, and that is added to the
        measurement's misfit. Beyond it the acquisition takes the quadrature.

        **A magnet's own gradient.** An acquisition played in a magnet with its own gradient ``g0`` (through
        every pulse and dead time) plays two directions with two time courses; ``g0`` through the effective gate
        is taken out of the encoding before the direction is judged, and enters as a second plane-wave factor in
        the walkers' background moments (:meth:`_closed_with_background`, dmrai-lab/dmipy-sim#565).

        **Where it runs.** ``device`` is the kernels' word (:func:`pose_device.route`): ``"auto"`` the JAX device
        when it is a GPU, else numpy; ``"torch"`` / ``"torch:<device>"`` torch (dmrai-lab/dmipy-sim#603), which keeps
        the walkers' moments, their directions and the bodies on its device, takes every route the host takes (the
        shell series where the host takes it, so the misfit is the host's), and returns only what the lab side reads.
        """
        from . import so3
        from .compression import read_position_coeffs
        from ._replay_kernel import effective_gradient
        from .pose_device import bessel_tails, host_bodies, route, spherical_jn_all as _jn_all
        kind, t_dev = route(device)
        n_acq = len(Ps)
        backgrounds = [_background_of(wf) for wf in waveforms]
        # one pass per (kind, field factor): a batch that mixes acquisitions with and without a magnet's gradient, or
        # acquisitions whose field accrues through different gates, takes one pass per kind and factor, so every
        # acquisition is the same numbers whichever batch it is expanded in (dmrai-lab/dmipy-sim#575)
        kinds = [(backgrounds[c] is not None, _field_factor_inputs(Ps[c], waveforms[c])) for c in range(n_acq)]
        passes = list(dict.fromkeys(kinds))
        if len(passes) > 1:
            out = [None] * n_acq
            for one in passes:
                sub = [c for c in range(n_acq) if kinds[c] == one]
                got = self._pose_coeffs_closed_many([Ps[c] for c in sub], [waveforms[c] for c in sub], keep=keep, tol=tol,
                                                    l_cap=l_cap, direction_tol=direction_tol, device=device)
                for c, r in zip(sub, got):
                    out[c] = r
            return out
        P0 = Ps[0]
        dt, n_t, ew, norm = P0["dt"], P0["n_t"], P0["pathway"] * P0["ew"], P0["norm"]
        n_w = ew.shape[0]
        field = self._field_quadratic(P0, waveforms[0]) if P0["B0"] is not None else None   # (a_w, A_w) or None
        run = current()
        _ph = (lambda name, **f: run.phase(name, **f)) if run is not None else (lambda name, **f: None)
        _ph("moments", n_acq=int(n_acq), n_w=int(n_w))
        # every acquisition's directions and grouped profiles; an acquisition with a multi-axis measurement is left out
        per = []
        gates = []                                                                 # the distinct gates a background accrues through
        for P, wf, bg in zip(Ps, waveforms, backgrounds):
            if bg is None:
                G_wf, G = P["G_eff_wf"], np.asarray(P["Geff"], np.float64)
            else:
                # the magnet's own gradient leaves the encoding: g0 through the effective gate is the second plane wave
                g0, gate = bg
                G_wf = P["G_eff_wf"] - g0[:, None, :] * gate[None, :, None]
                G = effective_gradient(G_wf, P["dt_wf"], n_t, dt)
                key = (gate.tobytes(), float(P["dt_wf"]))
                gate_id = next((j for j, gk in enumerate(gates) if gk[0] == key), None)
                if gate_id is None:
                    gate_id = len(gates); gates.append((key, gate, float(P["dt_wf"])))
            n_meas = G.shape[0]
            g_hat = np.zeros((n_meas, 3)); s_wave = np.zeros((n_meas, n_t)); zero = np.zeros(n_meas, bool)
            for i in range(n_meas):
                Gi = G[i]
                if not np.any(Gi) or (bg is not None and np.abs(Gi).max() <= 1e-6 * np.linalg.norm(g0[i])):
                    g_hat[i] = (0.0, 0.0, 1.0)                                     # a b = 0 row: no phase at any pose
                    s_wave[i] = 0.0                                                # (beside the background, to its rounding)
                    zero[i] = True
                    continue
                _u, sv, vt = np.linalg.svd(Gi, full_matrices=False)
                g, sw = vt[0], Gi @ vt[0]
                lead = int(np.flatnonzero(np.abs(sw) > 1e-6 * np.abs(sw).max())[0])
                if sw[lead] < 0:                       # one spelling of (direction, waveform): the first lobe positive
                    g, sw = -g, -sw
                g_hat[i] = g; s_wave[i] = sw
            # the body of a coefficient depends on the waveform's shape and amplitude only, never on its direction:
            # measurements that play the same s_i(t) -- a shell -- share one body, and their directions enter as
            # harmonics afterwards. Group by the played waveform, exactly, and contract once per group.
            if bg is None:
                group, first = _group_waveforms(s_wave, rtol=1e-5)               # float32 G: 1e-5 is the same waveform
                extra = {}
            else:
                # a body is also its background's magnitude (the background's Bessel factor is per walker): |g0| is
                # one more column of what is grouped, in the same unit as the waveform
                beta = np.linalg.norm(g0, axis=1)
                group, first = _group_waveforms(np.concatenate([s_wave, beta[:, None]], axis=1), rtol=1e-5)
                extra = dict(g0=g0, beta=beta, gate_id=gate_id, s_first=s_wave[first], gate=gate)
            # what the expansion plays for each row is its group's waveform along the row's own direction (and, with a
            # magnet's gradient, the group's |g0| along the row's own g0 direction); whatever the row plays beyond that
            # -- its part off its principal direction, its difference from its group's waveform within the grouping's
            # tolerance, a b = 0 row's rounding about zero -- is its residual, bounded below (#617)
            ref = first[group]
            played = np.where(zero[:, None], 0.0, s_wave[ref])[:, :, None] * g_hat[:, None, :]
            off = np.any(G != played, axis=(1, 2))
            if bg is not None:
                off |= beta != beta[ref]
            residual = np.flatnonzero(off)
            per.append(dict(P=P, G=G, G_wf=G_wf, g_hat=g_hat, group=group, first=first, n_meas=n_meas, ref=ref, zero=zero,
                            residual=residual, bound=np.zeros(n_meas), **extra))
        limit = (0.1 / np.sqrt(n_w)) if direction_tol is None else float(direction_tol)
        self._residual_bounds(per, P0, dt, n_w, ew, norm, device=device)
        for c, q in enumerate(per):
            if q is not None and q["residual"].size and float(q["bound"].max()) > limit:
                per[c] = None                                                      # not one direction to the bound: quadrature
        live = [c for c in range(n_acq) if per[c] is not None]
        if not live:
            return [None] * n_acq
        g_off, n_grp = {}, 0                                                      # every live acquisition's groups on one axis
        for c in live:
            g_off[c] = n_grp; n_grp += len(per[c]["first"])
        e = np.eye(3)
        n_gates = len(gates)
        if kind == "torch":
            import torch
            m = torch.zeros((n_w, n_grp + n_gates, 3), dtype=torch.float64, device=t_dev)
        else:
            m = np.zeros((n_w, n_grp + n_gates, 3))
        for seg, t0, n_s in P0["windows"]:                                     # the windows' moments sum (RPK.md 4.3)
            C = read_position_coeffs(seg.arrays, dtype=np.float64).reshape(n_w, -1)
            if kind == "torch":
                C = torch.as_tensor(C, device=t_dev)
            s_all = np.zeros((n_grp + n_gates, n_s))
            for c in live:
                q = per[c]
                G_s = effective_gradient(q["G_wf"], q["P"]["dt_wf"], n_s, dt, t0=t0) if self.n_segments > 1 else q["G"]
                s_all[g_off[c]:g_off[c] + len(q["first"])] = np.einsum("mtc,mc->mt", G_s[q["first"]], q["g_hat"][q["first"]])
            for j, (_key, gate, dt_g) in enumerate(gates):                     # a background's moment: its gate's, per walker
                s_all[n_grp + j] = effective_gradient(gate[None, :, None], dt_g, n_s, dt, t0=t0 if self.n_segments > 1 else None)[0, :, 0]
            if kind == "torch":
                # the profiles projected once, as one axis; axis b of the moments reads its own coefficients against it
                W1 = torch.as_tensor(_compile_effective(s_all[:, :, None], dt, self.K, n_s), device=t_dev)
                for b_ in range(3):
                    m[:, :, b_] += C[:, b_::3] @ W1
            for b_ in range(3 if kind != "torch" else 0):                      # m_w[b] = gamma sum_t s(t) r_w(t)_b dt
                W = _compile_effective(s_all[:, :, None] * e[b_][None, None, :], dt, self.K, n_s)
                m[:, :, b_] += C @ W
        if kind == "torch":
            del C
            m = m @ torch.as_tensor(self.substrate_frame, device=t_dev)       # stored -> canonical: F^T m, per walker
            n_bg = m[:, n_grp:, :].cpu().numpy()                               # (n_w, n_gates, 3): the background moments
            m = m[:, :n_grp, :]
            kappa = torch.linalg.vector_norm(m, dim=2)                         # (n_w, n_grp), radians
            m_hat = m / torch.where(kappa > 0, kappa, torch.ones_like(kappa))[:, :, None]
            m_hat[kappa == 0] = torch.tensor([0.0, 0.0, 1.0], dtype=m.dtype, device=t_dev)
            del m
        else:
            m = m @ self.substrate_frame                                       # stored -> canonical: F^T m, per walker
            n_bg = m[:, n_grp:, :]                                             # (n_w, n_gates, 3): the background moments
            m = m[:, :n_grp, :]
            kappa = np.linalg.norm(m, axis=2)                                  # (n_w, n_grp), radians
            safe = np.where(kappa > 0, kappa, 1.0)
            m_hat = m / safe[:, :, None]
            m_hat[kappa == 0] = (0.0, 0.0, 1.0)
        w = np.asarray(ew, np.float64) / float(norm)
        # the band: orders until the weighted Bessel tail is below tol for the worst group, every order from one
        # downward recurrence
        k_max = float(kappa.max()) if n_w * n_grp else 0.0
        _ph("bessel", n_grp=int(n_grp), phase_amplitude=k_max)
        L = int(np.ceil(k_max)) + 2
        # the weighted Bessel magnitudes per order and group, from the device it can use: the values themselves
        # stay where the bodies are formed (pose_device.field_bodies); only these sums come back
        T_ab = bessel_tails(kappa, w, min(l_cap, L + 12), device=device)        # (L_hi+1, n_grp)
        while L < l_cap:
            if L + 1 >= T_ab.shape[0]:
                T_ab = bessel_tails(kappa, w, min(l_cap, T_ab.shape[0] + 12), device=device)
            tail = (2 * (L + 1) + 1) * T_ab[L + 1].max()
            if tail < tol:
                break
            L += 1
        # the field factor's harmonics per walker, and the band the product reaches
        if field is None:
            L_f, F_sh = 0, None
        else:
            _ph("field")
            F_sh, L_f = self._field_harmonics(field, tol=tol, l_cap=l_cap, device=device)   # (n_w, (L_f+1)^2) complex
        if n_gates:
            tail_all = np.zeros(n_grp)
            for l in range(L + 1, min(L + 4, T_ab.shape[0])):
                tail_all += (2 * l + 1) * T_ab[l]
            return self._closed_with_background(
                per, live, g_off, n_grp, kappa, m_hat, w, n_bg, L, k_max, tail_all, F_sh, L_f,
                None if field is None else P0["b0_dir"], keep, tol, l_cap, n_acq, _ph, run, limit, device)
        L_tot = L + L_f
        want_l, want_n = (None, None) if keep is None else (keep[0], keep[1])
        keep_l = L_tot if want_l is None else min(int(want_l), L_tot)
        keep_n = L_tot if want_n is None else min(int(want_n), L_tot)
        n_feat = so3.n_so3_coeffs(keep_l, keep_n)
        _ph("harmonics", L=int(L), L_f=int(L_f), keep_l=int(keep_l), keep_n=int(keep_n), n_feat=int(n_feat))
        if field is None and kind == "torch":
            # every order's bodies from the device's (nc, R, 1) products against a unit factor; n within keep_n
            from .pose_device import field_bodies_torch
            one = torch.ones((n_w, 1), dtype=torch.float32, device=t_dev); nil = torch.zeros_like(one)
            bodies = [np.empty((n_grp, 2 * (so3._n_cols(l, keep_n) // 2) + 1)) for l in range(keep_l + 1)]
            off_l = np.cumsum([0] + [2 * l + 1 for l in range(keep_l + 1)])
            for lo in range(0, n_grp, TORCH_GROUPS):
                sl = slice(lo, min(lo + TORCH_GROUPS, n_grp))
                if run is not None:
                    run.progress(lo, n_grp, unit="groups")
                B = field_bodies_torch(kappa[:, sl], m_hat[:, sl, :], w, one, nil, keep_l, range(keep_l + 1),
                                       n_bessel=keep_l + 24 + int(np.ceil(k_max)), device=t_dev).real[:, :, 0].cpu().numpy()
                for l in range(keep_l + 1):
                    k = so3._n_cols(l, keep_n) // 2
                    bodies[l][sl] = B[:, off_l[l] + l - k:off_l[l] + l + k + 1]
        elif field is None:
            cos_z = m_hat[:, :, 2]
            J_all = _jn_all(L, kappa)                                          # (L+1, n_w, n_grp): the bodies read them here
            J = [J_all[l] for l in range(L + 1)]
            # ---- gradient only: one body per order and group, outer product with the direction harmonics
            bodies = [None] * (keep_l + 1)                                      # per order: (n_grp, 2k+1)
            if keep_n == 0:                                                     # n = 0 only: the Legendre of the angle to the axis
                for l in range(keep_l + 1):
                    bodies[l] = np.sqrt((2 * l + 1) / (4 * np.pi)) * ((w[:, None] * J[l]) * so3.legendre(l, cos_z)).sum(0)[:, None]
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
                    X = host_bodies(kappa[:, sl], m_hat[:, sl, :], w, keep_l, range(keep_l + 1), keep_n, J=J_all[:, :, sl])
                    for l in range(keep_l + 1):
                        bodies[l][sl] = X[l].sum(0)                                          # (nc, 2k+1)
        else:
            # ---- gradient x field: the outer product of the two body expansions per walker, summed over the
            # walkers per group, then coupled on both indices into the total order L_tot
            b_lab = np.asarray(P0["b0_dir"], np.float64); b_lab = b_lab / np.linalg.norm(b_lab)
            Yb = so3.real_sh(L_f, b_lab[None, :], full=True)[0]                 # ((L_f+1)^2,): the field direction, lab side
            # the field factor as two contiguous real blocks, for field_bodies' host route (BLAS does not take a
            # .real view's 16-byte stride) and for the float32 copies its device route makes
            F_re, F_im = np.ascontiguousarray(F_sh.real), np.ascontiguousarray(F_sh.imag)
            n_cols = (L + 1) ** 2
            l_used = [l for l in range(L + 1) if l <= keep_l + L_f]
            l_off = {}; n_rows = 0
            for l in l_used:
                l_off[l] = n_rows; n_rows += 2 * l + 1
            # the bodies of every gradient order against every field order in ONE product over the walkers per chunk
            # of groups: B[g, (l, n), (l', m')] = sum_w w j_l(kappa) Y_ln(m^) a_l'm'(w)
            B_full = np.empty((n_grp, n_rows, F_sh.shape[1]), np.complex128)
            from .pose_device import field_bodies, field_bodies_torch
            n_bessel = max(l_used) + 24 + int(np.ceil(k_max))                   # the Miller recurrence's start order
            step = max(1, int(2.5e8 / (8 * n_w * n_cols)))                       # groups per ~256 MB of host harmonics (the numpy route)
            if kind == "torch":                                                 # the device's own layout is B_full's
                F_re, F_im = (torch.as_tensor(F_re, dtype=torch.float32, device=t_dev),
                              torch.as_tensor(F_im, dtype=torch.float32, device=t_dev))
                for lo in range(0, n_grp, TORCH_GROUPS):
                    sl = slice(lo, min(lo + TORCH_GROUPS, n_grp))
                    if run is not None:
                        run.progress(lo, n_grp, unit="groups")
                    B_full[sl] = field_bodies_torch(kappa[:, sl], m_hat[:, sl, :], w, F_re, F_im, L, l_used,
                                                    n_bessel=n_bessel, device=t_dev).cpu().numpy()
            for lo in range(0, n_grp if kind != "torch" else 0, step):
                sl = slice(lo, min(lo + step, n_grp)); nc = sl.stop - sl.start
                if run is not None:
                    run.progress(lo, n_grp, unit="groups")
                # the Bessel values, the harmonics and the stacked bodies of this chunk of groups against the field
                # factor, formed where the walkers are; only the (rows, (L_f+1)^2) result returns
                B_c = field_bodies(kappa[:, sl], m_hat[:, sl, :], w, F_re, F_im, L, l_used, n_bessel=n_bessel)
                row = 0
                for l in l_used:
                    B_full[sl, l_off[l]:l_off[l] + 2 * l + 1] = B_c[row:row + nc * (2 * l + 1)].reshape(nc, 2 * l + 1, -1)
                    row += nc * (2 * l + 1)
            _ph("couplings", L=int(L), L_f=int(L_f))
            offs = {}
            off = 0
            for Lc in range(keep_l + 1):
                offs[Lc] = off; off += (2 * Lc + 1) * (2 * (so3._n_cols(Lc, keep_n) // 2) + 1)
            # the body side of every coupling once for all groups; the lab side per acquisition below
            # the coupling tables of every (l, l') pair, their coupled orders side by side: one product per pair on the
            # body index (all groups) and one per pair and acquisition on the lab index, instead of one per order
            tables = {}                                                         # (l, lp) -> (Ls, K_lab, K_body, lab slices, body slices)
            for l in l_used:
                for lp in range(L_f + 1):
                    Ls = [Lc for Lc in range(abs(l - lp), min(l + lp, keep_l) + 1)]
                    if not Ls:
                        continue
                    K = so3.coupling(l, lp)
                    K_lab = np.concatenate([K[Lc] for Lc in Ls], axis=1)                     # ((2l+1)(2l'+1), sum 2Lc+1)
                    K_body = np.concatenate([K[Lc].conj()[:, Lc - so3._n_cols(Lc, keep_n) // 2:Lc + so3._n_cols(Lc, keep_n) // 2 + 1]
                                             for Lc in Ls], axis=1)                            # ((2l+1)(2l'+1), sum 2kk+1)
                    lab_sl, body_sl, o1, o2 = {}, {}, 0, 0
                    for Lc in Ls:
                        kk = so3._n_cols(Lc, keep_n) // 2
                        lab_sl[Lc] = slice(o1, o1 + 2 * Lc + 1); o1 += 2 * Lc + 1
                        body_sl[Lc] = slice(o2, o2 + 2 * kk + 1); o2 += 2 * kk + 1
                    tables[(l, lp)] = (Ls, K_lab, K_body, lab_sl, body_sl)
            bodies_c = {}                                                       # (l, lp) -> (n_grp, sum 2kk+1)
            for l in l_used:
                B_all = B_full[:, l_off[l]:l_off[l] + 2 * l + 1, :]              # (n_grp, 2l+1, (L_f+1)^2)
                for lp in range(L_f + 1):
                    if (l, lp) not in tables:
                        continue
                    blp = so3.sh_block(lp, True)
                    B = B_all[:, :, blp].reshape(n_grp, -1)                                 # (n_grp, (2l+1)(2l'+1))
                    bodies_c[(l, lp)] = B @ tables[(l, lp)][2]
        # what the expansion cannot hold pointwise: the orders above the band it was built to, as a bound from
        # |P_l| <= 1 -- below tol by construction
        tail_all = np.zeros(n_grp)
        for l in range(L + 1, min(L + 4, T_ab.shape[0])):
            tail_all += (2 * l + 1) * T_ab[l]
        # ---- the lab side and the assembly, per acquisition
        out = [None] * n_acq
        for c in live:
            q = per[c]; n_meas = q["n_meas"]; group = q["group"] + g_off[c]     # the measurements' groups on the axis
            g_sl = slice(g_off[c], g_off[c] + len(q["first"]))
            Yg = so3.real_sh(L, q["g_hat"], full=True)                             # (n_meas, (L+1)^2): the lab side
            coeffs = np.zeros((n_meas, n_feat), np.complex128)
            if field is None:
                off = 0
                for l in range(keep_l + 1):
                    k = so3._n_cols(l, keep_n) // 2
                    blk = so3.sh_block(l, True)
                    body_i = bodies[l][group]                                                      # (n_meas, 2k+1)
                    block = (4 * np.pi * (1j ** l) / np.sqrt(2 * l + 1)) * Yg[:, blk][:, :, None] * body_i[:, None, :]
                    coeffs[:, off:off + (2 * l + 1) * (2 * k + 1)] = block.reshape(n_meas, -1)
                    off += (2 * l + 1) * (2 * k + 1)
            else:
                for l in l_used:
                    bl = so3.sh_block(l, True)
                    for lp in range(L_f + 1):
                        if (l, lp) not in tables:
                            continue
                        Ls, K_lab, _K_body, lab_sl, body_sl = tables[(l, lp)]
                        blp = so3.sh_block(lp, True)
                        # lab: Lam[i, (m, n')] = Y_lm(g^_i) Y_l'n'(b^), against every coupled order at once
                        Lam = (Yg[:, bl][:, :, None] * Yb[blp][None, None, :]).reshape(n_meas, -1)
                        lab_all = Lam @ K_lab                                       # (n_meas, sum 2Lc+1)
                        body_all = bodies_c[(l, lp)][group]                         # (n_meas, sum 2kk+1)
                        for Lc in Ls:
                            kk = so3._n_cols(Lc, keep_n) // 2
                            lab = lab_all[:, lab_sl[Lc]]                           # (n_meas, 2Lc+1)
                            body = body_all[:, body_sl[Lc]]                        # (n_meas, 2kk+1)
                            block = (4 * np.pi * (1j ** l) / np.sqrt(2 * Lc + 1)) * lab[:, :, None] * body[:, None, :]
                            o = offs[Lc]
                            coeffs[:, o:o + (2 * Lc + 1) * (2 * kk + 1)] += block.reshape(n_meas, -1)
            resp = PoseResponse(coeffs, keep_l, keep_n, misfit=tail_all[group] + q["bound"], floor=1.0 / np.sqrt(n_w),
                                phase_amplitude=float(kappa[:, g_sl].max()) if n_w * n_grp else 0.0, n_samples=0)
            resp.n_bodies = len(q["first"])                                        # the distinct waveforms contracted
            resp.field_lmax = L_f
            resp.route = "closed"
            out[c] = resp
        return out

    def _closed_with_background(self, per, live, g_off, n_grp, kappa, m_hat, w, n_bg, L, k_max, tail_all, F_sh, L_f,
                                b0_dir, keep, tol, l_cap, n_acq, _ph, run, limit, device="auto"):
        """The closed form of :meth:`_pose_coeffs_closed_many` for acquisitions played in a magnet with its own
        gradient (dmrai-lab/dmipy-sim#565): the background as a second plane-wave factor.

        **The phase.** A magnet that is not uniform adds a constant ``g0`` to the physical gradient through every
        pulse and dead time (:func:`_background_of`), and it accrues through the effective gate ``e(t)``
        (:attr:`~dmipy_sim.acquisition.scanner_sequence.ScannerSequence.effective_gate`) like everything else. So
        measurement ``i`` plays two directions with two time courses, ``G_i(t) = q_i s_i(t) + g0_i e(t)``, and
        walker ``w``'s phase at pose ``R`` is

            phi_iw(R) = q_i . R m_iw + g0_i . R n_w,      n_w = gamma int e(t) r_w(t) dt,

        ``m`` the encoding's moment and ``n_w`` the walker's BACKGROUND moment (the column the shape-moment layout
        stores per sequence group), both contracted through the windows from the stored bands.

        **The second factor.** ``exp(i g0 . R n) = sum_l 4 pi i^l j_l(|g0| |n|) sum_k Y_lk(n^) Y_lk(R^T g0^)`` is
        a function of the rotated background direction exactly as the field factor is of the rotated field
        direction: its body side is ``4 pi i^l j_l(|g0| |n_w|) Y_l(n^_w)`` per walker, its lab side ``Y_l(g0^_i)``.
        It is coupled to the gradient's Rayleigh expansion with the real Clebsch-Gordan tables (:func:`so3.coupling`)
        on the body index and on the lab index, as the field factor is. With the field on too, the field and
        background factors are first coupled walker by walker into one factor of order ``Lambda`` (body side
        ``sum a_l'k' c_l''k'' conj K``, lab side ``sum Y_l'(b^) Y_l''(g0^) K``), which is then coupled with the
        gradient: the product of three Wigner blocks, two couplings.

        **What makes it exact.** The encoding left once ``g0 e(t)`` is taken out is one direction per measurement
        (or one within #561's residual bound, which is then added to the misfit); ``g0`` is constant over the
        acquisition; and it accrues through the acquisition's own effective gate. A body depends on ``|g0|``
        (through ``j_l``) and on the gate (through ``n_w``), never on ``g0``'s direction, so the bodies are per
        (waveform, ``|g0|``) group and the factor per (``|g0|``, gate) -- one per encoding class of a machine --
        while each measurement's ``g0^`` enters on the lab side alone.

        **The band and the bound.** The background's band ``L_b`` is the first order whose weighted Bessel tail
        ``(2l+1) sum_w |w_w| |j_l(|g0| |n_w|)|`` is below ``tol`` for every factor. ``|g0| |n_w|`` is a few
        tenths of a radian at the Swoop's 1.4 mT/m, so ``L_b`` is a handful of orders. Walker by walker
        ``|e^{ia} e^{ib} - T_a T_b| <= r_b + r_a (1 + r_b)`` for truncations ``T`` with remainders ``r``, so the
        misfit is the gradient's tail plus the background's times ``1 +`` the gradient's largest per-walker
        remainder, plus the residual bound. An acquisition whose encoding is not one direction to that bound has
        already been sent to the quadrature.

        ``device`` as in :meth:`_pose_coeffs_closed_many`: on torch the factor, the bodies and their coupling on the
        body index stay on its device, and the lab side reads the coupled bodies (``(groups, a few)`` per order and
        channel) in one transfer per background factor."""
        from . import so3
        from .pose_device import bessel_tails, route
        kind, t_dev = route(device)
        if kind == "torch":
            import torch
            put = lambda a: torch.as_tensor(a, device=t_dev)
            new = lambda shape: torch.empty(shape, dtype=torch.complex128, device=t_dev)
        else:
            put = lambda a: a
            new = lambda shape: np.empty(shape, np.complex128)
        n_w = w.shape[0]
        # one background factor per (|g0|, gate): a bucket
        beta_g = np.zeros(n_grp); gate_g = np.zeros(n_grp, np.int64)
        for c in live:
            q = per[c]; sl = slice(g_off[c], g_off[c] + len(q["first"]))
            beta_g[sl] = q["beta"][q["first"]]; gate_g[sl] = q["gate_id"]
        keys, bucket = np.unique(np.stack([beta_g, gate_g.astype(np.float64)], axis=1), axis=0, return_inverse=True)
        bucket = np.asarray(bucket).reshape(-1)
        k_gate = keys[:, 1].astype(np.int64)
        r_n = np.linalg.norm(n_bg, axis=2)                                    # (n_w, n_gates)
        n_hat = n_bg / np.where(r_n > 0, r_n, 1.0)[:, :, None]
        n_hat[r_n == 0] = (0.0, 0.0, 1.0)
        x = keys[:, 0][None, :] * r_n[:, k_gate]                              # (n_w, n_buckets): |g0| |n_w|, radians
        x_max = float(x.max()) if x.size else 0.0
        _ph("background", n_buckets=int(len(keys)), phase_amplitude=x_max)
        L_b = 0
        T_bg = bessel_tails(x, w, min(l_cap, int(np.ceil(x_max)) + 12), device=device)
        while L_b < l_cap:
            if L_b + 1 >= T_bg.shape[0]:
                T_bg = bessel_tails(x, w, min(l_cap, T_bg.shape[0] + 12), device=device)
            if (2 * (L_b + 1) + 1) * T_bg[L_b + 1].max() < tol:
                break
            L_b += 1
        tail_bg = np.zeros(len(keys))
        for l in range(L_b + 1, min(L_b + 4, T_bg.shape[0])):
            tail_bg += (2 * l + 1) * T_bg[l]
        J_top = so3.spherical_jn_all(L + 3, np.array([k_max]))[:, 0]         # the gradient's largest per-walker remainder
        r_a = float(sum((2 * l + 1) * abs(J_top[l]) for l in range(L + 1, L + 4)))
        # the factor's channels: (order Lambda, its body columns, how its lab side is formed)
        Yb = None
        if F_sh is None:
            channels = [(l2, so3.sh_block(l2, True), ("b", l2)) for l2 in range(L_b + 1)]
            n_cols = (L_b + 1) ** 2
        else:
            b_lab = np.asarray(b0_dir, np.float64); b_lab = b_lab / np.linalg.norm(b_lab)
            Yb = so3.real_sh(L_f, b_lab[None, :], full=True)[0]
            channels, n_cols, K_fb = [], 0, {}
            for lp in range(L_f + 1):
                for l2 in range(L_b + 1):
                    K_fb[(lp, l2)] = so3.coupling(lp, l2)
                    for Lam in range(abs(lp - l2), lp + l2 + 1):
                        channels.append((Lam, slice(n_cols, n_cols + 2 * Lam + 1), ("fb", lp, l2)))
                        n_cols += 2 * Lam + 1
        Lam_max = L_f + L_b
        Y_n = [put(so3.real_sh(L_b, n_hat[:, j, :], full=True)) for j in range(n_hat.shape[1])]
        F_sh_k = None if F_sh is None else put(F_sh)
        K_fb_k = {} if F_sh is None else {pair: {Lam: put(K.conj()) for Lam, K in Kp.items()} for pair, Kp in K_fb.items()}

        def factor(k):
            """The per-walker factor of bucket ``k``, ``(n_w, n_cols)`` complex, where the route keeps its walkers."""
            J = put(so3.spherical_jn_all(L_b, x[:, k]))                       # (L_b+1, n_w)
            Fb = new((n_w, (L_b + 1) ** 2))
            for l2 in range(L_b + 1):
                b2 = so3.sh_block(l2, True)
                Fb[:, b2] = (4 * np.pi * (1j ** l2)) * J[l2][:, None] * Y_n[k_gate[k]][:, b2]
            if F_sh is None:
                return Fb
            F = new((n_w, n_cols))
            for (lp, l2), Kc in K_fb_k.items():
                prod = (F_sh_k[:, so3.sh_block(lp, True)][:, :, None] * Fb[:, so3.sh_block(l2, True)][:, None, :]).reshape(n_w, -1)
                for Lam, cols, ch in channels:
                    if ch[1:] == (lp, l2):
                        F[:, cols] = prod @ Kc[Lam]
            return F

        L_tot = L + Lam_max
        want_l, want_n = (None, None) if keep is None else (keep[0], keep[1])
        keep_l = L_tot if want_l is None else min(int(want_l), L_tot)
        keep_n = L_tot if want_n is None else min(int(want_n), L_tot)
        n_feat = so3.n_so3_coeffs(keep_l, keep_n)
        _ph("harmonics", L=int(L), L_f=int(L_f), L_b=int(L_b), keep_l=int(keep_l), keep_n=int(keep_n), n_feat=int(n_feat))
        l_used = [l for l in range(L + 1) if l <= keep_l + Lam_max]
        offs, off = {}, 0
        for Lc in range(keep_l + 1):
            offs[Lc] = off; off += (2 * Lc + 1) * (2 * (so3._n_cols(Lc, keep_n) // 2) + 1)
        tables = {}                                                           # (l, Lambda) -> (Ls, K_lab3, K_body, slices)
        for l in l_used:
            for Lam in range(Lam_max + 1):
                Ls = [Lc for Lc in range(abs(l - Lam), min(l + Lam, keep_l) + 1)]
                if not Ls:
                    continue
                K = so3.coupling(l, Lam)
                K_lab = np.concatenate([K[Lc] for Lc in Ls], axis=1)                        # ((2l+1)(2Lam+1), sum 2Lc+1)
                K_body = np.concatenate([K[Lc].conj()[:, Lc - so3._n_cols(Lc, keep_n) // 2:Lc + so3._n_cols(Lc, keep_n) // 2 + 1]
                                         for Lc in Ls], axis=1)                               # ((2l+1)(2Lam+1), sum 2kk+1)
                lab_sl, body_sl, o1, o2 = {}, {}, 0, 0
                for Lc in Ls:
                    kk = so3._n_cols(Lc, keep_n) // 2
                    lab_sl[Lc] = slice(o1, o1 + 2 * Lc + 1); o1 += 2 * Lc + 1
                    body_sl[Lc] = slice(o2, o2 + 2 * kk + 1); o2 += 2 * kk + 1
                tables[(l, Lam)] = (Ls, K_lab.reshape(2 * l + 1, 2 * Lam + 1, -1), K_body, lab_sl, body_sl)
        n_bessel = max(l_used) + 24 + int(np.ceil(k_max))                     # the Miller recurrence's start order
        step = max(1, int(2.5e8 / (8 * n_w * (L + 1) ** 2)))                   # groups per ~256 MB of host harmonics
        # on the host, only the products the retained azimuthal band couples (pose_device.paired_bodies); the device
        # forms every product in one fused pass
        col_n = np.zeros(n_cols, np.int64)
        for Lam, cols, _kind in channels:
            col_n[cols] = np.abs(np.arange(-Lam, Lam + 1))
        host = kind == "numpy"
        paired = host and keep_n < Lam_max
        K_body_k = {}                                                         # the body-side tables on the torch device
        l_off = dict(zip(l_used, np.cumsum([0] + [2 * l + 1 for l in l_used[:-1]])))   # each order's rows in a body
        perm = np.argsort(col_n, kind="stable") if paired else np.arange(n_cols)   # the factor's columns by |N|
        at = np.empty_like(perm); at[perm] = np.arange(n_cols)                      # a column's place after the sort
        if paired:
            step = max(1, int(5e8 / (8 * n_w * sum(2 * l + 1 for l in l_used))))    # groups per ~0.5 GB of bodies
        out = [None] * n_acq
        done = 0
        shell_dev = np.zeros(n_grp)                                           # a shell row's departure from its shape
        for c in live:
            q = per[c]; n_meas = q["n_meas"]; group = q["group"] + g_off[c]
            g_sl = slice(g_off[c], g_off[c] + len(q["first"]))
            Yg = so3.real_sh(L, q["g_hat"], full=True)                        # (n_meas, (L+1)^2): the gradient's lab side
            g0 = q["g0"]
            beta = np.linalg.norm(g0, axis=1)
            u_dirs, inv = np.unique(np.where(beta[:, None] > 0, g0 / np.where(beta > 0, beta, 1.0)[:, None], (0.0, 0.0, 1.0)),
                                    axis=0, return_inverse=True)
            inv = np.asarray(inv).reshape(-1)
            Y_u = so3.real_sh(L_b, u_dirs, full=True)                         # (n_u, (L_b+1)^2): the background's lab side
            lam = []                                                          # per channel: (n_u, 2 Lambda + 1)
            for Lam, cols, ch in channels:
                if ch[0] == "b":
                    lam.append(Y_u[:, so3.sh_block(ch[1], True)])
                else:
                    _, lp, l2 = ch
                    pair = (Yb[so3.sh_block(lp, True)][None, :, None] * Y_u[:, None, so3.sh_block(l2, True)]).reshape(len(u_dirs), -1)
                    lam.append(pair @ K_fb[(lp, l2)][Lam])
            coeffs = np.zeros((n_meas, n_feat), np.complex128)
            M_lab = {}                                                        # (l, channel) -> (n_u, 2l+1, sum 2Lc+1)
            for k in np.unique(bucket[g_sl]):
                if kind == "torch":
                    F = factor(int(k))                                       # the torch route forms every pair: no sort
                    F_re, F_im = F.real.contiguous(), F.imag.contiguous()
                else:
                    F = factor(int(k))[:, perm]
                    F_re, F_im = np.ascontiguousarray(F.real), np.ascontiguousarray(F.imag)
                del F
                grp = np.flatnonzero(bucket[g_sl] == k) + g_off[c]              # this acquisition's groups of the bucket
                def assemble(idx, B3, coupled):
                    """Add the bodies of the groups ``idx`` to their measurements' coefficients: from ``B3`` (the
                    host's bodies, coupled here on the body index) or from ``coupled`` (the torch route's, coupled
                    on its device), on the lab side per order and channel."""
                    nc = len(idx)
                    where = np.full(n_grp, -1, np.int64); where[idx] = np.arange(nc)
                    sel = np.flatnonzero(where[group] >= 0)                    # the measurements whose bodies these are
                    pos = where[group[sel]]
                    for l in l_used:
                        bl = so3.sh_block(l, True)
                        B_l = None if coupled is not None else B3[:, l_off[l]:l_off[l] + 2 * l + 1, :]
                        Yg_l = Yg[sel][:, bl]                                              # (n_sel, 2l+1)
                        for ci, (Lam, cols, _ch) in enumerate(channels):
                            if (l, Lam) not in tables:
                                continue
                            Ls, K_lab3, K_body, lab_sl, body_sl = tables[(l, Lam)]
                            body_all = (coupled[(l, ci)] if coupled is not None
                                        else B_l[:, :, at[cols]].reshape(nc, -1) @ K_body)  # (nc, sum 2kk+1)
                            if (l, ci) not in M_lab:                                       # the lab side's coupling, per direction of g0
                                M_lab[(l, ci)] = np.einsum("uM,iMs->uis", lam[ci], K_lab3)
                            M_lam = M_lab[(l, ci)]
                            lab_all = (Yg_l @ M_lam[0]) if len(u_dirs) == 1 else np.einsum("ni,nis->ns", Yg_l, M_lam[inv[sel]])
                            body_all = body_all[pos]
                            for Lc in Ls:
                                kk = so3._n_cols(Lc, keep_n) // 2
                                block = (4 * np.pi * (1j ** l) / np.sqrt(2 * Lc + 1)) * lab_all[:, lab_sl[Lc]][:, :, None] \
                                    * body_all[:, body_sl[Lc]][:, None, :]
                                o = offs[Lc]
                                coeffs[sel, o:o + (2 * Lc + 1) * (2 * kk + 1)] += block.reshape(len(sel), -1)

                got_idx, got = [], []                                         # the torch route's bodies, kept on its device
                for idx, B3, dev in self._background_bodies(grp, q["s_first"][grp - g_off[c]], kappa, m_hat, w, F_re, F_im,
                                                       L, l_used, l_off, col_n[perm], keep_n, paired, host or kind == "torch",
                                                       step, n_bessel, limit, device=device):
                    nc = len(idx)
                    shell_dev[idx] = dev
                    if run is not None:
                        run.progress(done, n_grp, unit="groups")
                    done += nc
                    if kind != "torch":
                        assemble(idx, B3, None)
                        continue
                    got_idx.append(idx); got.append(B3)
                if got:
                    # the bucket's bodies side by side on the device, every (order, channel)'s body side coupled there
                    # and read back in one transfer; the lab side then runs once for the bucket's groups
                    idx = np.concatenate(got_idx); B3 = torch.cat(got) if len(got) > 1 else got[0]
                    del got
                    nc = len(idx)
                    keys_ = [(l, ci) for l in l_used for ci, (Lam, _c, _k) in enumerate(channels) if (l, Lam) in tables]
                    outs = []
                    for l in l_used:
                        B_l = B3[:, l_off[l]:l_off[l] + 2 * l + 1]
                        for ci, (Lam, cols, _k) in enumerate(channels):
                            if (l, Lam) not in tables:
                                continue
                            if (l, Lam) not in K_body_k:
                                K_body_k[(l, Lam)] = put(tables[(l, Lam)][2])
                            outs.append((B_l[:, :, cols].reshape(nc, -1) @ K_body_k[(l, Lam)]).reshape(-1))
                    flat = torch.cat(outs).cpu().numpy() if outs else np.zeros(0, np.complex128)
                    coupled, o = {}, 0
                    for key_, t_ in zip(keys_, outs):
                        coupled[key_] = flat[o:o + t_.numel()].reshape(nc, -1); o += t_.numel()
                    del outs, B3
                    assemble(idx, None, coupled)
            misfit = tail_all[group] + q["bound"] + tail_bg[bucket[group]] * (1.0 + r_a) + shell_dev[group]
            resp = PoseResponse(coeffs, keep_l, keep_n, misfit=misfit, floor=1.0 / np.sqrt(n_w),
                                phase_amplitude=float(kappa[:, g_sl].max()) if n_w * n_grp else 0.0, n_samples=0)
            resp.n_bodies = len(q["first"])
            resp.field_lmax = L_f
            resp.background_lmax = L_b
            resp.route = "closed"
            out[c] = resp
        return out

    def _background_bodies(self, grp, s_first, kappa, m_hat, w, F_re, F_im, L, l_used, l_off, col_n, keep_n, paired, host,
                           step, n_bessel, limit, device="auto"):
        """The bodies of the groups ``grp`` (one acquisition's, one background factor) against the factor ``F``, in
        chunks: yields ``(idx, B, dev)`` with ``B[g, (l, n), j] = sum_w w_w j_l(kappa_wg) Y_ln(m^_wg) F_wj``
        ``(nc, R, n_f)``, the orders ``l_used`` at the offsets ``l_off``, and ``dev`` ``(nc,)`` what that costs the
        misfit (below).

        **A shell is one body in powers of its amplitude.** A machine's class plays every row at its own delivered
        amplitude, so a shell's rows are as many groups; but they play one SHAPE, ``s_g(t) = a_g u(t)``, so their
        moments are ``a_g mu_w`` with one ``mu_w`` per walker, and ``j_l(a r) = sum_k c_lk (a r)^{l+2k}`` with
        ``c_lk = (-1)^k / (2^k k! (2l+2k+1)!!)`` makes every row's body a polynomial in its amplitude:

            B_g = sum_k (a_g / A)^{l+2k} M_k,     M_k = c_lk sum_w w_w (A |mu_w|)^{l+2k} Y_ln(mu^_w) F_w,

        ``A`` the shell's largest amplitude. The walkers are contracted once per power instead of once per row. A
        row's waveform is its shape times its amplitude only nearly -- a machine's Maxwell term along the encoding
        goes as the square of the amplitude, and its float32 rounding -- so its own moment ``m_gw`` differs from
        ``(a_g / A) m_refw`` by ``dm_gw``; its phase at any pose by at most ``|dm_gw|``, and ``dev_g = sum_w |w_w|
        |dm_gw|`` is added to its misfit. Rows are one shape when their waveforms over their amplitudes agree to
        ``SHELL_RTOL``, and a shell takes the series only when every row's ``dev_g`` is within ``limit`` (the closed
        form's ``direction_tol``: a tenth of the floor by default), as a nearly single-direction waveform does. The
        series is cut where its next term is below ``SHELL_SERIES_TOL`` at the shell's largest phase, and is taken on
        the host (and on torch, in float64 on its device: :meth:`_background_bodies_torch`) when that phase is at most ``SHELL_SERIES_MAX_PHASE`` (its terms then cancel to within a few digits
        of float64) and the shell has more rows than powers; otherwise each group is contracted on its own."""
        from . import so3
        from .pose_device import field_bodies, paired_bodies, paired_products, route
        kind, t_dev = route(device)
        if kind == "torch":
            import torch
            w_abs = torch.as_tensor(np.abs(w), device=t_dev)
        n_w = w.shape[0]
        R = sum(2 * l + 1 for l in l_used)
        amp = np.abs(s_first).max(axis=1) if s_first.size else np.zeros(0)
        shells = []                                                             # (group positions, use the series)
        live = np.flatnonzero(amp > 0)
        if host and live.size:
            shape = s_first[live] / amp[live][:, None]
            sh_group, sh_first = _group_waveforms(shape, rtol=SHELL_RTOL)
            for h in range(len(sh_first)):
                shells.append(live[sh_group == h])
        rest = np.setdiff1d(np.arange(len(grp)), np.concatenate(shells) if shells else np.zeros(0, np.int64))
        series, single = [], [rest]
        for members in shells:
            g_ref = members[np.argmax(amp[members])]
            x = kappa[:, grp[g_ref]]
            x_max = float(x.max())
            K = 0
            c = {l: [1.0 / float(np.prod(np.arange(2 * l + 1, 0, -2, dtype=np.float64)))] for l in l_used}
            while max(abs(c[l][K]) * x_max ** (l + 2 * K) for l in l_used) > SHELL_SERIES_TOL:
                for l in l_used:
                    c[l].append(-c[l][K] / (2.0 * (K + 1) * (2 * l + 2 * K + 3)))
                K += 1
            dev = np.zeros(len(members))
            if x_max <= SHELL_SERIES_MAX_PHASE and len(members) > K + 1:
                t = amp[members] / amp[g_ref]
                for lo in range(0, len(members), 64):                         # each row's departure from the shell's shape
                    gg = grp[members[lo:lo + 64]]
                    if kind == "torch":
                        tt = torch.as_tensor(t[lo:lo + 64], device=t_dev)
                        dm = kappa[:, gg, None] * m_hat[:, gg, :] - (tt[None, :, None] * x[:, None, None]) * m_hat[:, grp[g_ref], None, :]
                        dev[lo:lo + 64] = (w_abs @ torch.linalg.vector_norm(dm, dim=2)).cpu().numpy()
                        continue
                    dm = kappa[:, gg, None] * m_hat[:, gg, :] - (t[lo:lo + 64][None, :, None] * x[:, None, None]) \
                        * m_hat[:, grp[g_ref], None, :]
                    dev[lo:lo + 64] = np.abs(w) @ np.linalg.norm(dm, axis=2)
            if x_max <= SHELL_SERIES_MAX_PHASE and len(members) > K + 1 and float(dev.max()) <= limit:
                series.append((members, g_ref, x, K, c, dev))
            else:
                single.append(members)
        row_n = np.concatenate([np.abs(np.arange(-l, l + 1)) for l in l_used])
        if kind == "torch":
            yield from self._background_bodies_torch(grp, series, single, amp, kappa, m_hat, w, F_re, F_im, L, l_used, l_off,
                                                     n_bessel, step, t_dev)
            return
        for members, g_ref, x, K, c, dev in series:
            Y = so3.real_sh(L, m_hat[:, grp[g_ref], :], full=True)             # (n_w, (L+1)^2): the shell's moment directions
            M = np.empty((K + 1, R, F_re.shape[1]), np.complex128)
            for k in range(K + 1):
                X = np.empty((n_w, R))
                for l in l_used:
                    X[:, l_off[l]:l_off[l] + 2 * l + 1] = (w * c[l][k] * x ** (l + 2 * k))[:, None] * Y[:, so3.sh_block(l, True)]
                M[k] = paired_products(X, row_n, F_re, F_im, col_n, keep_n) if paired else (X.T @ F_re) + 1j * (X.T @ F_im)
            t = amp[members] / amp[g_ref]                                      # (n_m,): each row's amplitude in the shell's
            l_row = np.concatenate([np.full(2 * l + 1, l) for l in l_used])
            chunk = max(1, int(2.5e8 / (16 * R * F_re.shape[1])))
            for lo in range(0, len(members), chunk):
                mm = members[lo:lo + chunk]
                powers = t[lo:lo + chunk][:, None, None] ** (l_row[None, None, :] + 2 * np.arange(K + 1)[None, :, None])
                yield grp[mm], np.einsum("gkr,krj->grj", powers, M), dev[lo:lo + chunk]
        for members in single:
            for lo in range(0, len(members), step):
                idx = grp[members[lo:lo + step]]; nc = len(idx)
                if not nc:
                    continue
                if paired:
                    B_c = paired_bodies(kappa[:, idx], m_hat[:, idx, :], w, F_re, F_im, L, l_used, col_n, keep_n)
                else:
                    B_c = field_bodies(kappa[:, idx], m_hat[:, idx, :], w, F_re, F_im, L, l_used, n_bessel=n_bessel)
                B3 = np.empty((nc, R, F_re.shape[1]), np.complex128)
                row = 0
                for l in l_used:
                    B3[:, l_off[l]:l_off[l] + 2 * l + 1] = B_c[row:row + nc * (2 * l + 1)].reshape(nc, 2 * l + 1, -1)
                    row += nc * (2 * l + 1)
                yield idx, B3, np.zeros(nc)

    def _background_bodies_torch(self, grp, series, single, amp, kappa, m_hat, w, F_re, F_im, L, l_used, l_off, n_bessel,
                                 step, t_dev):
        """:meth:`_background_bodies`' contractions on the torch device ``t_dev``, for the shells and the single groups
        the host's choice made: a shell's powers ``M_k`` in float64 (its terms cancel, as on the host), a single
        group's bodies through :func:`pose_device.field_bodies_torch` in float32. Yields ``(idx, B, dev)`` with ``B``
        on the device, ``(nc, R, n_f)`` complex128 in the host's layout, and ``dev`` on the host."""
        import torch
        from .pose_device import field_bodies_torch, real_sh_torch
        n_w = w.shape[0]
        R = sum(2 * l + 1 for l in l_used)
        w_d = torch.as_tensor(w, device=t_dev)
        n_f = F_re.shape[1]
        l_row = np.concatenate([np.full(2 * l + 1, l) for l in l_used])
        sh_cols = torch.as_tensor(np.concatenate([np.arange(l * l, l * l + 2 * l + 1) for l in l_used]), device=t_dev)
        l_row_d = torch.as_tensor(l_row, device=t_dev, dtype=torch.float64)
        for members, g_ref, x, K, c, dev in series:
            Y = real_sh_torch(L, m_hat[:, grp[g_ref], :])[:, sh_cols]            # (n_w, R) float64
            c_k = torch.as_tensor(np.array([np.concatenate([np.full(2 * l + 1, c[l][k]) for l in l_used]) for k in range(K + 1)]),
                                  device=t_dev)                                 # (K+1, R): c_lk per row
            M = torch.empty((K + 1, R, n_f), dtype=torch.complex128, device=t_dev)
            for k in range(K + 1):
                X = (w_d[:, None] * c_k[k][None, :] * x[:, None] ** (l_row_d[None, :] + 2 * k)) * Y          # (n_w, R)
                M[k] = torch.complex(X.T @ F_re, X.T @ F_im)
            t = torch.as_tensor(amp[members] / amp[g_ref], device=t_dev)
            chunk = TORCH_GROUPS
            for lo in range(0, len(members), chunk):
                mm = members[lo:lo + chunk]
                powers = t[lo:lo + chunk][:, None, None] ** (l_row_d[None, None, :] + 2 * torch.arange(K + 1, device=t_dev,
                                                                                                         dtype=torch.float64)[None, :, None])
                yield grp[mm], torch.einsum("gkr,krj->grj", powers.to(torch.complex128), M), dev[lo:lo + chunk]
        F32 = (F_re.to(torch.float32), F_im.to(torch.float32))
        for members in single:
            for lo in range(0, len(members), TORCH_GROUPS):
                idx = grp[members[lo:lo + TORCH_GROUPS]]
                if len(idx):
                    yield idx, field_bodies_torch(kappa[:, idx], m_hat[:, idx, :], w, *F32, L, l_used, n_bessel=n_bessel,
                                                  device=t_dev), np.zeros(len(idx))

    def _residual_bounds(self, per, P0, dt, n_w, ew, norm, device="auto"):
        """Into each acquisition's ``bound``: for every measurement whose waveform departs from what the expansion plays
        for it -- its group's waveform along its own principal direction, and its group's ``|g0|`` along its own
        background direction (#617) -- a bound on how far that residual can move the ensemble's signal at ANY pose: its
        component off the principal direction, its difference from its group's waveform within the grouping's
        tolerance, a ``b = 0`` row's rounding about zero, its ``|g0|``'s difference from its group's. The residuals
        of an acquisition's measurements are written on their common time courses (an SVD over the measurements and
        axes, to its numerical rank: a machine's Maxwell residual is one ramp-shaped course per shell),
        ``res_i(t) = sum_k u_k(t) c_ik``; a walker's phase at pose ``R`` is then ``sum_k (R^T c_ik) . mu_kw`` with
        ``mu_kw = sum_t u_k(t) r_w(t)`` contracted through the windows as the moments are, and is bounded over the
        rotations by ``sum_k |c_ik| |mu_kw|`` (:meth:`_pose_coeffs_closed_many`). Windows add. The bound is the
        ENSEMBLE's, which is what the misfit bounds: ``|sum_w w_w (e^{i(phi_w + d_w)} - e^{i phi_w})| <= sum_w |w_w|
        |d_w|``, the walkers' bounds weighted as the signal weighs them. ``device`` as in
        :meth:`_pose_coeffs_closed_many`: on torch the walkers' contractions run on its device (the SVD of the
        residuals, a few thousand by the window's saves, on the host)."""
        from .compression import read_position_coeffs
        from ._replay_kernel import effective_gradient
        from .._blas import lapack_threads
        from .pose_device import route
        todo = [c for c, q in enumerate(per) if q is not None and q["residual"].size]
        if not todo:
            return
        kind, t_dev = route(device)
        e = np.eye(3)
        live = np.asarray(ew) != 0
        w_live = np.abs(np.asarray(ew, np.float64)[live]) / float(norm)
        per_walker = {c: 0.0 for c in todo}                                    # (n_live, n_r) per acquisition, windows summed
        for seg, t0, n_s in P0["windows"]:
            C = read_position_coeffs(seg.arrays, dtype=np.float64).reshape(n_w, -1)[live]
            if kind == "torch":
                import torch
                C = torch.as_tensor(C, device=t_dev)
            for c in todo:
                q = per[c]; rows = q["residual"]; ref = q["ref"][rows]
                G_s = effective_gradient(q["G_wf"], q["P"]["dt_wf"], n_s, dt, t0=t0) if self.n_segments > 1 else q["G"]
                G_s = np.asarray(G_s, np.float64)
                g = q["g_hat"][rows]
                s_ref = np.where(q["zero"][rows][:, None], 0.0, np.einsum("mtc,mc->mt", G_s[ref], q["g_hat"][ref]))
                res = G_s[rows] - s_ref[:, :, None] * g[:, None, :]                              # (n_r, n_s, 3)
                if "g0" in q:                                       # the group's |g0| played for the row's own
                    e_s = effective_gradient(q["gate"][None, :, None], q["P"]["dt_wf"], n_s, dt,
                                             t0=t0 if self.n_segments > 1 else None)[0, :, 0]
                    d_beta = (q["beta"][rows] - q["beta"][ref]) / np.where(q["beta"][rows] > 0, q["beta"][rows], 1.0)
                    res = res + (d_beta[:, None] * q["g0"][rows])[:, None, :] * e_s[None, :, None]
                X = res.transpose(0, 2, 1).reshape(-1, n_s)                                       # (n_r * 3, n_s)
                with lapack_threads():                                                            # #564: this shape hangs a known build
                    U, S, Vt = np.linalg.svd(X, full_matrices=False)
                k = int(np.sum(S > 1e-12 * S[0])) if S.size and S[0] > 0 else 0
                if k == 0:
                    continue
                coef = (U[:, :k] * S[:k]).reshape(len(rows), 3, k)                                # c_ik per axis
                if kind == "torch":
                    # every course in one projection, each axis reading its own rows of the coefficients
                    Wk = torch.as_tensor(_compile_effective(Vt[:k, :, None], dt, self.K, n_s), device=t_dev)   # (K+2, k)
                    mu = torch.stack([C[:, b_::3] @ Wk for b_ in range(3)], dim=2)                 # (n_live, k, 3)
                    nrm = torch.linalg.vector_norm(mu, dim=2) @ torch.as_tensor(np.linalg.norm(coef, axis=1).T, device=t_dev)
                    per_walker[c] = per_walker[c] + nrm                                       # (n_live, n_r) on the device
                    continue
                mu = np.stack([np.stack([C @ _compile_effective(Vt[j][None, :, None] * e[b_][None, None, :], dt, self.K, n_s)[:, 0]
                                         for b_ in range(3)], axis=1) for j in range(k)], axis=1)  # (n_live, k, 3)
                per_walker[c] = per_walker[c] + np.linalg.norm(mu, axis=2) @ np.linalg.norm(coef, axis=1).T
        for c in todo:
            if kind == "torch" and not isinstance(per_walker[c], float):
                per[c]["bound"][per[c]["residual"]] += (torch.as_tensor(w_live, device=t_dev) @ per_walker[c]).cpu().numpy()
            elif np.ndim(per_walker[c]):
                per[c]["bound"][per[c]["residual"]] += w_live @ per_walker[c]

    def _field_quadratic(self, P, waveform):
        """The susceptibility phase of every walker as ``a_w + u^T A_w u`` in the field direction ``u`` expressed in
        the CANONICAL frame: ``(a (n_w,), A (n_w, 3, 3))``, the gate-integrated path field basis of the pack (C3 path
        route) scaled by ``B0``, ``chi_iso``, ``chi_aniso``. Raises, as the quadrature route does, when the pack
        cannot supply it."""
        B0, chi_iso, chi_aniso = P["B0"], P["chi_iso"], P["chi_aniso"]
        if P["ch"].get("susceptibility_path") is None:
            raise ValueError("the pose expansion with a field needs the pack's susc_path channel (C3 path route)")
        Psi, names = _path_field_channels(P, waveform)                           # the one read of C3, windows summed
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

    def _field_harmonics(self, field, tol=1e-8, l_cap=64, device="auto"):
        """The harmonics of ``exp(i (a_w + u^T A_w u))`` over the sphere per walker, ``(n_w, (L'+1)^2)``, by a product
        quadrature exact to the band ``L'`` chosen from the phase amplitude: orders are added until the energy in
        the last one is below ``tol`` of the total (``4 pi`` per walker, the phase having unit modulus). A phase
        whose band lies past ``l_cap`` is refused: the expansion is not the route for it, a replay per pose is."""
        import hashlib
        a, A = field
        # the factor depends on the gate and the field, not on the gradient: the encoding classes of a machine pass
        # and the passes of one acquisition share it, keyed by its inputs' bytes (dmrai-lab/dmipy-sim#449)
        key = (hashlib.sha1(np.ascontiguousarray(a, np.float64).tobytes()).hexdigest(),
               hashlib.sha1(np.ascontiguousarray(A, np.float64).tobytes()).hexdigest(), float(tol), int(l_cap), str(device))
        cache = self.__dict__.setdefault("_field_factor_cache", {})
        if key in cache:
            return cache[key]
        out = self._field_harmonics_of(a, A, tol=tol, l_cap=l_cap, device=device)
        if len(cache) >= 2:
            cache.pop(next(iter(cache)))
        cache[key] = out
        return out

    def _field_harmonics_of(self, a, A, tol=1e-8, l_cap=64, device="auto"):
        """:meth:`_field_harmonics` computed: the quadrature and its band, the walkers' factor on the device
        (:func:`pose_device.field_factor`) or in numpy when there is none."""
        from . import so3
        amp = float(np.abs(np.linalg.eigvalsh(A)).max()) if A.size else 0.0
        Lp = int(np.ceil(2.0 * amp)) + 4
        if Lp > l_cap:
            raise ValueError(
                f"the scanner's field sweeps {amp:.1f} radians of phase on this pack, so its pose response reaches "
                f"order ~{Lp}, beyond the cap of {l_cap}. That is a real cost, not a setting: the expansion is "
                f"worth building to share one walk over many poses, and at this sharpness a direct replay per pose "
                f"(orientation=R) is the exact route for it.")
        from .pose_device import field_factor
        while True:
            dirs, wq = so3.sphere_quadrature(Lp + 2, 2 * Lp + 2)
            Y = so3.real_sh(Lp, dirs, full=True)                               # (n_q, (Lp+1)^2)
            F = field_factor(a, A, dirs, Y * wq[:, None], device=device)       # (n_w, (Lp+1)^2), on the device it can use
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

        The samples themselves -- the phasor sum over the walkers at every rotation of the design, with the field
        term's quadratic form in the rotated field direction -- are :func:`pose_device.pose_samples`, on the device
        in chunks of rotations with the numpy route as its oracle; this method prepares the contractions, projects
        the samples and checks the projection.
        """
        from . import so3
        from .compression import read_position_coeffs
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
        if B0 is not None:
            pm = self.meta.get("compression", {}).get("channels", {}).get("susceptibility_path")
            if pm is None:
                raise ValueError("the pose expansion with a field needs the pack's susc_path channel (C3 path route)")
            from .bank import path_field_integral
            Psi, names = path_field_integral(self.arrays, pm, waveform, n_t, dt)         # (n_w, n_ch)
            i_p = names.index("iso_P_xx")
            i_a = names.index("aniso_G_xx") if "aniso_G_xx" in names else None
        b = np.asarray(b0_dir, float); b = b / np.linalg.norm(b)

        # a pose R is a rotation of the CANONICAL substrate frame: the stored walk is first turned into it
        # (F^T, RPK.md 4.2) and then by R -- the identity for a pack whose frame is the identity
        Qf_axis = self.substrate_frame.T

        from .pose_device import pose_samples
        field_terms = None
        if Psi is not None:
            field_terms = (float(chi_iso) * float(B0), (float(chi_aniso) * float(B0) if (chi_aniso and i_a is not None) else 0.0), b,
                           Psi[:, names.index("iso_local")], Psi[:, i_p:i_p + 6], (Psi[:, i_a:i_a + 6] if i_a is not None else None))
        run = current()
        _ph = (lambda name, **f: run.phase(name, **f)) if run is not None else (lambda name, **f: None)

        def response(R):
            """The ensemble signal of every measurement at every one of these poses: the sampling on the device it can
            use (:func:`pose_device.pose_samples`), the poses turned into the canonical frame first."""
            return pose_samples(R @ Qf_axis, Q, ew, norm, field=field_terms)

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
        _ph("quadrature", n_nodes=int(Rq.shape[0]), S_L=int(S_L), keep_l=int(keep_l), keep_n=int(keep_n), n_feat=int(n_feat),
            phase_amplitude=phi_amp)
        coeffs = np.zeros((n_feat, n_meas), np.complex128)
        E_q = response(Rq)                                                     # every node's signal, one sampling
        for lo in range(0, Rq.shape[0], int(chunk)):                           # the projection, in blocks of nodes
            sl = slice(lo, min(lo + int(chunk), Rq.shape[0]))
            A = so3.so3_design(keep_l, Rq[sl], keep_n)
            coeffs += (A * wq[sl, None]).T @ E_q[sl]
        floor = 1.0 / np.sqrt(n_w)
        _ph("misfit")

        Rc, Ac = so3.haar_design(keep_l, keep_n, int(n_check), int(seed))
        if (keep_l, keep_n) == (S_L, S_N):
            # the retained band is the whole response: certify pointwise, worst case, off the grid
            misfit = np.abs(Ac @ coeffs - response(Rc)).max(axis=0)
        else:
            # only part of the response is retained, so a pointwise comparison is not the question. What has to
            # hold is that the retained coefficients are alias-free: refine the grid and require them to stand.
            Rf, wf, _d, _r = so3.so3_quadrature(S_L + 2 * int(over) + 1, S_N + 2 * int(over) + 1)
            fine = np.zeros_like(coeffs)
            E_f = response(Rf)
            for lo in range(0, Rf.shape[0], int(chunk)):
                sl = slice(lo, min(lo + int(chunk), Rf.shape[0]))
                A = so3.so3_design(keep_l, Rf[sl], keep_n)
                fine += (A * wf[sl, None]).T @ E_f[sl]
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

    def contact(self):
        """The C2 channel as a per-save series, ``(n_walkers, n_t)``: the wall contact each step accrued.

        The windows of a segmented walk share a save and the first save of an accumulated channel ends no step
        (#225), so joining them is a rule and not a concatenation; this is the one place it is applied for a
        consumer. The cumulative boundary local time is ``cumsum`` along the saves, which is what a surface
        relaxivity weights: ``S(t) = <exp((rho2/D) L(t))>``. ``None`` when the pack carries no C2 channel.
        """
        return self._decoded_channels()["ell"]

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


def _duration(waveform):
    """An acquisition's duration in seconds from the walk's start: its last gradient sample's time."""
    waveform = waveform.waveform if hasattr(waveform, "waveform") else waveform
    return (int(np.asarray(waveform.G_eff).shape[1]) - 1) * float(waveform.dt)


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


TORCH_GROUPS = 128
"""Groups the torch route of the closed form contracts per pass over the walkers (dmrai-lab/dmipy-sim#603): each pass's
bodies (``(groups, R, n_f)`` complex128, a gigabyte at the brain's orders) stay on the device until the lab side
reads their couplings, and the host's lab-side loop runs once per pass."""

SHELL_RTOL = 1e-3
"""Rows are one shape when their waveforms over their amplitudes agree to this fraction of the largest
(:meth:`ReplayPack._background_bodies`); how far each departs is then bounded and added to its misfit."""

SHELL_SERIES_TOL = 1e-14
"""Where a shell's amplitude series (:meth:`ReplayPack._background_bodies`) is cut: the largest next term
``|c_lk| x^{l+2k}`` at the shell's largest phase ``x``, far below the closed form's own band tolerance."""

SHELL_SERIES_MAX_PHASE = 10.0
"""The largest phase (radians) a shell's amplitude series is summed at: its terms grow to about ``e^x / 2 x``
before they cancel, about a thousand at 10, so the sum keeps some thirteen of float64's sixteen digits."""


def _background_of(waveform):
    """The magnet's own gradient in an acquisition, as the pose expansion separates it: ``(g0 (n_meas, 3), gate
    (n_t,))`` float64 on the waveform's grid, or ``None`` when the acquisition carries none (no
    :meth:`~dmipy_sim.acquisition.scanner_sequence.ScannerSequence.with_background_gradient`, or a zero one).

    ``g0`` is what the physical gradient is wherever the coils play nothing: the magnet's gradient together with
    the concomitant field that gradient carries on its own, which is constant in time as the magnet's is (a
    machine's Maxwell term is quadratic in the whole gradient, so its ``g0``-only part rides with ``g0``). A
    measurement whose coils are on at every sample keeps the recorded ``background_gradient``, and the constant
    Maxwell part is then left to the encoding's residual. ``gate`` is the acquisition's effective gate, through
    which ``g0`` accrues: the effective gradient is the encoding plus ``g0 gate``."""
    if waveform.background_gradient is None:
        return None
    g0 = np.broadcast_to(np.asarray(waveform.background_gradient, np.float64).reshape(-1, 3), (waveform.n_meas, 3)).copy()
    G = np.asarray(waveform.G, np.float64)
    played = np.asarray(waveform.played_gradient, np.float64)
    off = np.all(played == 0.0, axis=2)                                    # (n_meas, n_t): the coils play nothing
    for i in range(waveform.n_meas):
        if off[i].any():
            g0[i] = G[i, off[i]].mean(axis=0)
    if not np.any(g0):
        return None
    return g0, np.asarray(waveform.effective_gate, np.float64)


def _group_waveforms(s, rtol=1e-5):
    """Group rows of ``s (n, n_t)`` that are the same waveform to ``rtol`` of the largest amplitude: ``(group (n,),
    first (n_grp,))``. A tolerance, not a rounding, so two rows a rounding boundary apart stay together. Greedy in
    row order: the first ungrouped row opens a group and takes every ungrouped row within the tolerance of it.

    A row within ``tol`` of row ``i`` everywhere has its projection on a fixed vector ``v`` within ``tol |v|_1`` of
    row ``i``'s, so the rows are sorted once by such a projection and each row's exact test runs on the window of
    the sorted order that condition leaves (#605): the same groups, at the cost of the candidates instead of every
    row per group. ``v`` is a fixed pseudo-random vector with ``|v|_1 = 1`` -- a constant one would not separate
    waveforms of zero mean, which a refocused encoding is -- and the window is widened by the projections' own
    rounding."""
    s = np.asarray(s, np.float64)
    scale = float(np.abs(s).max()) or 1.0
    tol = rtol * scale
    n, n_t = s.shape
    v = np.random.default_rng(0).uniform(-1.0, 1.0, n_t)
    v /= np.abs(v).sum()
    p = s @ v
    order = np.argsort(p, kind="stable"); p_sorted = p[order]
    win = tol + 4.0 * n_t * np.finfo(np.float64).eps * scale               # the projection's rounding, generously
    group = np.full(n, -1, np.int64)
    first = []
    for i in range(n):
        if group[i] >= 0:
            continue
        cand = order[np.searchsorted(p_sorted, p[i] - win, side="left"):np.searchsorted(p_sorted, p[i] + win, side="right")]
        cand = cand[group[cand] < 0]
        same = cand[np.abs(s[cand] - s[i]).max(axis=1) <= tol]
        group[same] = len(first)
        first.append(i)
    return group, np.asarray(first, np.int64)


def _container_of(ranges):
    """The builder's band-container knob (``((upto, bits), ...)``, or ``None`` for the float container) from the
    ranges a pack's metadata records (``[{"bands": [k0, k1], "bits": b}, ...]``)."""
    if not ranges:
        return None
    return tuple((None if i == len(ranges) - 1 else int(r["bands"][1]), int(r["bits"])) for i, r in enumerate(ranges))


def _path_grid(pm, n_t, dt):
    """The path channel's own save grid ``(n_t, dt)``: the walk's when the field was read at every save, else the
    coarser grid the producer recorded (``field_sample_every`` saves per field sample)."""
    return int(pm.get("n_t", n_t)), float(pm.get("dt", dt))


def _compile_effective(Geff, dt_pack, K, n_t, gyromagnetic_ratio=GAMMA):
    """``W`` from per-save weights already on the pack grid (:func:`_replay_kernel.effective_gradient`)."""
    from .compression import bridge_projection
    W = bridge_projection(np.asarray(Geff, np.float64), int(n_t), K)              # (n_meas, K+2, n_c)
    return (gyromagnetic_ratio * float(dt_pack) * W).reshape(W.shape[0], (K + 2) * W.shape[2]).T


def _band_phase(P, W=None):
    """``(n_w, n_meas)`` gradient phase of every walker under the prepared acquisition ``P``: each window's stored
    coefficients against the waveform's projection on that window's own save grid, summed (RPK.md 4.3).

    The ONE band contraction of a replay. :meth:`ReplayPack._walker_phases` -- and with it
    :meth:`ReplayPack.replay`, :meth:`ReplayPack.walker_signals` and :meth:`ReplayPack.walker_phases` -- and
    :func:`~dmipy_sim.replay.study.walker_primitives` read it, so the per-walker depth and the ensemble depth are
    the same number by construction rather than by agreement (dmipy-sim#484). ``W`` replaces the projection with a
    caller's own per-position weights (:meth:`ReplayPack._check_weights`, single-window only).
    """
    from .compression import read_position_coeffs
    from ._replay_kernel import effective_gradient
    dt, n_w = P["dt"], P["n_w"]
    phi = None
    for seg, t0, n_s in P["windows"]:
        W_s = W if W is not None else _compile_effective(
            effective_gradient(P["G_eff_wf"], P["dt_wf"], n_s, dt, t0=t0), dt, seg.K, n_s)
        C = read_position_coeffs(seg.arrays, dtype=np.float64)
        phi_s = C.reshape(n_w, seg.n_coeffs * 3) @ W_s
        phi = phi_s if phi is None else phi + phi_s
    return phi


def _field_factor_inputs(P, waveform):
    """What the closed-form pose expansion's field factor (:meth:`ReplayPack._field_quadratic`) is a function of,
    as a hashable key: ``None`` with no field, else the field's strength and direction, the tissue's chi, the gate
    the field accrues through (:func:`~dmipy_sim.replay._replay_kernel.field_gate`'s input) on its grid, and the
    pack grid and windows it is read onto. Two acquisitions with equal keys have one field factor."""
    if P["B0"] is None:
        return None
    gate = waveform.effective_gate if waveform.gate is None else waveform.gate
    return (float(P["B0"]), tuple(float(x) for x in np.asarray(P["b0_dir"], np.float64)), float(P["chi_iso"]),
            float(P["chi_aniso"] or 0.0), np.ascontiguousarray(gate, np.float64).tobytes(), float(waveform.dt),
            int(P["n_t"]), float(P["dt"]), tuple((float(t0), int(n_s)) for _seg, t0, n_s in P["windows"]))


def _path_field_channels(P, waveform):
    """``(Psi, names)``: the C3 path channel's gated field integrals per walker under the prepared acquisition,
    summed over the windows -- the ONE read of that channel, which
    :func:`~dmipy_sim.fields.hollow_cylinder.field_terms` turns into the isotropic and anisotropic terms for
    :meth:`ReplayPack._walker_phases` and :func:`~dmipy_sim.replay.study.walker_primitives` alike."""
    from .bank import path_field_integral
    pm = P["ch"]["susceptibility_path"]
    Psi = names = None
    for seg, t0, n_s in P["windows"]:
        Psi_s, names = path_field_integral(seg.arrays, pm, waveform, n_s, P["dt"], t0=t0, n_w=P["n_w"])
        Psi = Psi_s if Psi is None else Psi + Psi_s
    return Psi, names


def _has_aniso(grid_meta, names):
    """Whether the pack's field channels carry the anisotropic basis: the grid meta says so AND the channel names
    hold it."""
    return bool((grid_meta or {}).get("has_aniso")) and names is not None and "aniso_G_xx" in names


def _signal_factor(phi, voxel):
    """``E`` of a replay from the walkers' phases: ``exp(i phi)`` times the voxel's factor per measurement
    (:meth:`~dmipy_sim.acquisition.scanner_sequence.ScannerSequence.voxel_factor`, 1 for a refocused encoding and
    the spoiler for an unbalanced one, dmipy-sim#375). Every per-walker route forms ``E`` here, so the weighted
    ensemble mean of any of them is :meth:`ReplayPack.replay`."""
    return np.exp(1j * np.asarray(phi)) * np.asarray(voxel, np.float64)[None, :]


def surface_logweight(arrays, rho2_over_D, chan_meta=None, chi_hat=None):
    """Per-walker surface log-weight ``(rho2/D) * sum_t chi(t) ell_i(t)`` from the C2 channel.

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
        return float(rho2_over_D) * np.asarray(arrays["blt_endpoint"], np.float64)
    meta = dict(chan_meta or {})
    meta.setdefault("n_t", int(np.asarray(chi_hat).shape[0]))
    meta.setdefault("K", _cx_bands_K(arrays, meta))
    return surface_logweight_bridge(arrays, meta, rho2_over_D, chi_hat)          # the bridge contracted, never decoded


def replay_signal(pack, W, *, rho2_over_D=0.0, chi_hat=None, complex_signal=False):
    """Replay a compiled scheme ``W`` (from :func:`compile_scheme`) against ``pack`` (a :class:`ReplayPack`
    or a plain arrays dict). Returns ``E`` per measurement (magnitude unless ``complex_signal``).

    ``rho2_over_D`` > 0 activates the exact surface-relaxivity replay via the pack's boundary local time
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
    if rho2_over_D:
        # asked for, so it must happen: a missing C2 channel raises inside surface_logweight
        # rather than being skipped, so a caller never gets a silently unattenuated signal
        # when rho2_over_D was asked for and the pack carries no boundary channel.
        cm = ((pack.meta.get("compression", {}).get("channels", {}) or {}).get("boundary_local_time")
              if isinstance(pack, ReplayPack) else None)
        surface_logw = surface_logweight(a, rho2_over_D, cm, chi_hat)
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
