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
endpoints and the ``K`` bands the walk accumulated, :mod:`dmipy_sim.engine.accumulate`) with the waveform's
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
from dmipy_replay.replay.replay import ReplayPack as _ReplayPack, read_rpk as _read_rpk
import dmipy_replay.replay.so3 as _r_so3
import json
from dataclasses import dataclass
from functools import cached_property

import numpy as np
from dmipy_replay.replay.compression import c2_bands_K as _cx_bands_K
from dmipy_replay.replay.decoded import PackArrays
from dmipy_replay.replay.planning import PLAN_TOL, ReadPlan, planned

from dmipy_replay.constants import GAMMA
from ..run import Run, current
from dmipy_replay.acquisition.scanner_sequence import Protocol, ScannerSequence
from dmipy_replay.replay.replay import _compile_continuous, _path_field_channels, _pose_matrix, _refuse_predating

__all__ = ["ReplayPack", "PoseResponse", "read_rpk", "write_rpk", "analytic_pose_response", "replay_signal_jax", "replay_batch_jax",]

# ------------------------------- .rpk container I/O -------------------------------
#: the safetensors dtype names of the numpy dtypes a pack's tensors are in
ST_DTYPES = {"bool": "BOOL", "uint8": "U8", "int8": "I8", "uint16": "U16", "int16": "I16", "float16": "F16", "uint32": "U32",
             "int32": "I32", "float32": "F32", "uint64": "U64", "int64": "I64", "float64": "F64"}
#: the order safetensors lays tensors out in, by dtype (then by name): the reference writer's (``safetensors.serialize``)
_ST_ORDER = ("U64", "I64", "F64", "F32", "U32", "I32", "F16", "U16", "I16", "I8", "U8", "BOOL")
_WRITE_BYTES = 1 << 26                  #: the largest piece of a tensor written at once (bytes)


def write_rpk(path, arrays, metadata):
    """Write a replay pack: ``arrays`` (name -> ndarray, or a list of ndarrays stored as their concatenation along the
    first axis, as a merge stores its shards' tensors) as safetensors tensors, ``metadata`` (a dict) serialised to JSON
    under the ``"rpk"`` header key. The file is byte for byte what ``safetensors.numpy.save_file`` writes for the
    contiguous arrays (a 0-d array stored as one element, as ``np.ascontiguousarray`` makes it): the header's
    ``__metadata__`` first and the tensors in their data order -- by dtype as :data:`_ST_ORDER`, then by name --
    compact, padded with spaces to eight bytes, the data little-endian. The tensors are streamed in pieces of at
    most :data:`_WRITE_BYTES` into a sibling file that replaces ``path`` once complete: a pack mapped from ``path``
    (:func:`read_rpk`) keeps reading the file it was mapped from."""
    import os
    import struct
    parts = {k: [np.ascontiguousarray(x) for x in (v if isinstance(v, (list, tuple)) else [v])] for k, v in arrays.items()}
    head = {}
    for k, ps in parts.items():
        dt = ps[0].dtype
        if any(x.dtype.name != dt.name or x.shape[1:] != ps[0].shape[1:] for x in ps):
            raise ValueError(f"the pieces of tensor {k!r} differ in dtype or row shape")
        if dt.name not in ST_DTYPES:
            raise TypeError(f"tensor {k!r} is {dt}, which a pack does not store")
        head[k] = dict(dtype=ST_DTYPES[dt.name], shape=[sum(int(x.shape[0]) for x in ps)] + [int(d) for d in ps[0].shape[1:]])
    order = sorted(head, key=lambda k: (_ST_ORDER.index(head[k]["dtype"]), k))
    off = 0
    for k in order:
        nb = sum(int(x.nbytes) for x in parts[k])
        head[k]["data_offsets"] = [off, off + nb]; off += nb
    h = json.dumps(dict(__metadata__={"rpk": json.dumps(dict(metadata))}, **{k: head[k] for k in order}),
                   separators=(",", ":"), ensure_ascii=False).encode()
    h += b" " * (-len(h) % 8)
    tmp = f"{path}.{os.getpid()}.part"
    try:
        with open(tmp, "wb") as f:
            f.write(struct.pack("<Q", len(h))); f.write(h)
            for k in order:
                for x in parts[k]:
                    step = max(1, _WRITE_BYTES // max(1, x.nbytes // max(1, len(x))))
                    for i in range(0, len(x), step):
                        f.write(np.ascontiguousarray(x[i:i + step], x.dtype.newbyteorder("<")).tobytes())
        os.replace(tmp, path)
    finally:
        if os.path.exists(tmp):
            os.remove(tmp)


def read_rpk(path):
    """Read a replay pack into this package's :class:`ReplayPack`: :func:`dmipy_replay.replay.replay.read_rpk`, which
    maps the file copy-on-write and refuses a pack this revision does not read and one whose embedded spec disagrees
    with its channels."""
    return _read_rpk(path, cls=ReplayPack)






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














class ReplayPack(_ReplayPack):
    """A replay pack: :class:`dmipy_replay.ReplayPack` (the reader: every replay route, the plan, the band check) with
    the producer's routes -- :meth:`save` / :meth:`publish`, the walk's continuation (:meth:`end_state`,
    :meth:`extend`), :meth:`truncate`, :meth:`rewindow`, the band-limited :meth:`prefix`, the surface envelope's
    restatement, the pose expansion of a distribution of poses (:meth:`pose_response`, :meth:`pose_series`), the study
    primitives -- :meth:`open` building this package's columnar and remote readers, and the JAX vector-Bloch
    propagator (``replay_bloch(backend="jax")``)."""

    BLOCH_BACKENDS = dict(_ReplayPack.BLOCH_BACKENDS, jax=lambda: _jax_bloch())

    @staticmethod
    def _as_distribution(orientation):
        """``orientation=`` as a distribution of poses (a :class:`~dmipy_sim.replay.so3.Distribution`, or an
        :class:`~dmipy_sim.replay.fod.FOD` read as an axis density), or None when it names a single pose."""
        return _as_distribution(orientation)

    @staticmethod
    def open(uri, workers=8):
        """A pack open by reference. ``hf://owner/name/path.rpk`` is a single pack on the Hub, a
        :class:`~dmipy_sim.replay.remote.RemotePack`: it plans an acquisition from the file's header and fetches by HTTP
        range only the tensors the plan names (``load()`` downloads the whole file). Any other ``uri`` is the columnar layout
        (a directory, or ``hf://owner/name/prefix`` on the Hub), open by reference:
        a :class:`~dmipy_sim.replay.columnar.ColumnarPack`, whose ``view`` gives a :class:`ReplayPack` of the rows
        and bands an acquisition needs and whose ``image`` replays a whole grid in one pass over the rows."""
        from dmipy_replay.replay.hub import is_hub_uri
        if is_hub_uri(uri) and uri.endswith(".rpk"):
            from .remote import RemotePack
            return RemotePack(uri)
        from .columnar import open_columnar
        return open_columnar(uri, workers=workers)

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

    def truncate(self, n_keep, *, id=None, out_path=None):
        """The first ``n_keep`` windows as a pack: a prefix of whole windows is the range of their tensors and
        nothing is re-encoded (RPK.md 8.3). The windows keep their certificates and the whole's is the bound
        over them (:func:`~dmipy_sim.replay.bank.combine_window_fidelity`)."""
        import copy
        from .bank import combine_window_fidelity
        seg = self.windows; n = int(seg["n"]); n_keep = int(n_keep)
        if n_keep < 1 or n_keep > n:
            raise ValueError(f"keep between 1 and {n} windows; got {n_keep}")
        if n_keep == n:
            return self
        keep = {k: v for k, v in self.arrays.items()
                if "/" not in k or int(k[1:k.index("/")]) < n_keep}
        meta = copy.deepcopy(self.meta)
        wp = meta["walk_params"]; T_seg = float(seg["T"])
        walks = [dict(w, last=min(int(w["last"]), n_keep - 1)) for w in seg.get("walks") or [] if int(w["first"]) < n_keep]
        wp.update(T_max=n_keep * T_seg, windows=dict(n=n_keep, T=T_seg, walks=walks))
        fid = dict(meta.get("fidelity") or {})
        per = fid.get("windows")
        if per:
            meta["fidelity"] = combine_window_fidelity(per[:n_keep])
        meta.setdefault("provenance", {})["truncated"] = dict(parent_id=self.meta.get("id"), windows_kept=n_keep, of=n)
        if id is not None:
            meta["id"] = id
        out = ReplayPack(keep, meta, source=None)
        if out_path is not None:
            out.save(out_path)
        return out

    def end_state(self):
        """``(r_end, pool)``: where every walker is at the walk's end and the pool it is in
        (:func:`~dmipy_sim.replay.continuation.end_state`)."""
        from .continuation import end_state
        return end_state(self)

    def extend(self, T_add, *, seed, out_path=None, require_gpu=None, envelope=None, device="auto"):
        """This pack lengthened by ``T_add`` seconds of new windows: the walk continued from its end on its own
        substrate with a fresh seed and appended (:func:`~dmipy_sim.replay.continuation.extend_pack`)."""
        from .continuation import extend_pack
        return extend_pack(self, T_add, seed=seed, out_path=out_path, require_gpu=require_gpu, envelope=envelope, device=device)

    def rewindow(self, T_seg, *, out_path, id=None, max_dropped_fraction=1e-3, envelope=None, tol=2.0):
        """This one-window walk stored in windows of ``T_seg`` seconds (RPK.md 4.3), so that a consumer holds only
        the windows its acquisitions reach (#525).

        Every window is the parent's stored functions cut to it and re-projected on the window's own basis
        (:func:`~dmipy_sim.replay.bank.window_of_pack`: the bridge and the field modes by their closed-form window
        maps, the events sliced), at the parent's bands per second (``K_seg = ceil(K T_seg / T)``, doubled until
        window 0's certificate passes), joined by :func:`~dmipy_sim.replay.bank.join_windows`. A whole number of
        windows cover the walk up to a trailing remainder of at most ``max_dropped_fraction`` of its duration,
        which is dropped (``time_dropped``); a walk that would leave more is refused. Each window's certificate
        (``fidelity.windows[i]``) carries the walk-free terms and the re-projection's measured truncation against
        the parent's exact reading of the window, and the whole's is the bound over them
        (:func:`~dmipy_sim.replay.bank.combine_window_fidelity`); a window outside ``tol`` times its floor is
        refused. ``provenance.rewindowed`` records the parent (id, sha256 of its file, certificate), ``T_seg``, the
        bands and ``time_dropped``. ``envelope`` is the battery every window is certified over: by default
        :func:`~dmipy_sim.replay.compression.default_envelope` with the parent's field strengths (``B0_list``) and
        the refocusing depth its field channel serves (``max_refocus_pulses``)."""
        import copy
        import hashlib
        from .bank import window_of_pack, combine_window_fidelity, join_windows
        if self.n_windows != 1:
            raise ValueError(f"this pack already stores its walk in {self.n_windows} windows of "
                             f"{float(self.windows['T']) * 1e3:.3f} ms; rewindow reads a one-window walk")
        T = float(self.T_window)
        T_seg = float(T_seg)
        n = int(np.floor(T / T_seg * (1.0 + 1e-9)))
        if n < 2:
            raise ValueError(f"a {T * 1e3:.3f} ms walk holds fewer than two windows of {T_seg * 1e3:.3f} ms")
        dropped = T - n * T_seg
        if dropped > float(max_dropped_fraction) * T + 1e-12:
            raise ValueError(f"windows of {T_seg * 1e3:.3f} ms leave {dropped * 1e3:.3f} ms of the {T * 1e3:.3f} ms walk uncovered, "
                             f"more than {max_dropped_fraction:g} of it; give a T_seg that divides it, or a larger max_dropped_fraction")
        env = dict(envelope) if envelope is not None else self.stated_envelope()
        K = max(2, int(np.ceil(self.K * T_seg / T)))
        stamp = id or f"{self.meta.get('id')}/seg{T_seg * 1e3:.0f}ms"
        packs, tried = [], []
        blt_K = path_K = None
        for i in range(n):
            while True:
                pk = window_of_pack(self, i * T_seg, T_seg, K=K, blt_K=blt_K, path_K=path_K, id=stamp, envelope=env)
                fid = pk.meta["fidelity"]
                if i > 0 or fid["err_reprojection"] <= tol * fid["floor_max"] or K >= int(self.K):
                    break
                tried.append(K)
                K = min(2 * K, int(self.K))
            if fid["err_reprojection"] > tol * fid["floor_max"]:                 # what the window ADDS to the parent's own terms
                raise ValueError(f"window {i} of {n} re-projected at K = {K} does not reproduce the parent's reading of it within "
                                 f"{tol:g} x its Monte-Carlo floor (re-projection error {fid['err_reprojection']:.3g}, floor {fid['floor_max']:.3g})")
            if i == 0:
                c2 = (pk.meta["compression"].get("channels") or {}).get("boundary_local_time")
                blt_K = None if c2 is None else int(c2["K"])
                pw = (pk.meta["compression"].get("channels") or {}).get("susc_field_basis")
                path_K = None if pw is None else int(pw["K"])
            packs.append(pk)
        fid = combine_window_fidelity([pk.meta["fidelity"] for pk in packs])
        wp = self.meta.get("walk_params", {}) or {}
        arrays, meta = join_windows(packs, fidelity=fid, walks=[dict(first=0, last=n - 1, seed=wp.get("seed"))],
                                     walkers_shuffled=bool(self.meta.get("compression", {}).get("precision_tiers", {}).get("walkers_shuffled", False)))
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
        meta["provenance"] = dict(copy.deepcopy(self.meta.get("provenance") or {}), rewindowed=dict(
            parent_id=self.meta.get("id"), parent_sha256=parent_sha, parent_digest=(self.digest if parent_sha is None else None),
            parent_fidelity=dict(err_max=pf.get("err_max"), floor_max=pf.get("floor_max"), certified=pf.get("certified")),
            T_seg=float(T_seg), n_seg=int(n), time_dropped=float(dropped),
            parent_K=int(self.K), K=int(K), K_tried=tried + [int(K)], blt_K=blt_K, path_K=path_K,
            note="every window the parent's stored functions cut to it and re-projected on its basis; its certificate carries the "
                 "re-projection's truncation measured against the parent's exact reading of the window, beside the walk-free terms"))
        meta["id"] = id if id is not None else stamp
        meta["license"], meta["citation"] = self.license, self.citation
        out = ReplayPack(arrays, meta, source=None)
        if out.has_surface:
            out.restate_surface_envelope(reference=self, tol=tol)
        out.save(out_path)
        return out

    def stated_envelope(self):
        """The envelope this pack states, as one a re-encoding of it (a window, a prefix) is certified under:
        :func:`~dmipy_sim.replay.compression.default_envelope`'s battery with the pack's field strengths
        (``acquisition.B0_list``), scanner class (``acquisition.scanner`` where its bound is the class's), the
        refocusing depth its field channel serves (``max_refocus_pulses``), its contact envelope as ``rho2_list`` (over
        the walk's ``D``) and its ``chi_max``."""
        from .compression import default_envelope
        env = default_envelope()
        renv = self.meta.get("replay_envelope") or {}
        acq = renv.get("acquisition") or {}
        if acq.get("B0_list"):
            env["B0_list"] = [float(b) for b in acq["B0_list"]]
        if acq.get("bound") == "scanner_class":
            env["scanner"] = acq["scanner"]
        pm = ((self.meta.get("compression") or {}).get("channels") or {}).get("susc_field_basis")
        if pm is not None and pm.get("max_refocus_pulses"):
            env["max_refocus_pulses"] = int(pm["max_refocus_pulses"])
        if self.rho2_over_D_max is not None and self.diffusivity:
            env["rho2_list"] = [float(self.rho2_over_D_max) * float(self.diffusivity)]
        if self._envelope_tissue().get("chi_max") is not None:
            env["chi_max"] = float(self._envelope_tissue()["chi_max"])
        return env

    def restate_surface_envelope(self, *, reference=None, tol=2.0, out_path=None):
        """State the contact tier at the envelope it serves (RPK.md 8.7): per window the largest ``rho2 / D`` at which
        the decoded contact gives a physical attenuation within ``tol`` floors
        (:func:`~dmipy_sim.replay.bank.surface_envelope`), the pack's ``replay_envelope.tissue.rho2_over_D_max`` the
        smallest over the windows, and every window's surface rows (``err_surface``, ``floor_surface``) restated up
        to that edge rather than at the stated envelope's top (the ``rho2_over_D_max`` the pack was built under), its
        maxima re-read and the whole's certificate rebuilt from them. ``reference`` is the one-window pack the windows were encoded from, whose decoded contact
        is then the error's reference; without it the error is the unphysical gain alone. Only metadata changes;
        ``out_path`` writes the pack with its tensors as they are. ``provenance.surface_envelope_restated`` records
        the edges, the battery's top and the reference."""
        import copy
        from .bank import surface_envelope, restate_maxima, combine_window_fidelity
        from dmipy_replay.replay.compression import local_time_at
        if not self.has_surface:
            raise ValueError("this pack carries no contact channel (C2): there is no surface envelope to state")
        D = self.diffusivity
        if D is None:
            raise ValueError("the pack records no diffusivity, so rho2 / D cannot be stated")
        if reference is not None and reference.n_windows != 1:
            raise ValueError("the reference is the one-window pack the windows were encoded from")
        hi = self.rho2_over_D_max
        if hi is None:
            raise ValueError("this pack states no contact envelope (replay_envelope.tissue.rho2_over_D_max), the range its "
                             "certificate was read over: build it under an envelope that states rho2_list")
        hi = float(hi)
        w = np.asarray(self.spin_weights, np.float64)
        reads = []                                                   # (contact per block, reference sums) per window
        if reference is not None:
            rm = dict((reference.meta["compression"].get("channels") or {})["boundary_local_time"])
            rm.setdefault("K", _cx_bands_K(reference.arrays, rm))
        for win, t0, T_w in self._windows():
            cm = dict((win.meta["compression"].get("channels") or {})["boundary_local_time"])
            cm.setdefault("K", _cx_bands_K(win.arrays, cm))
            ref = None
            if reference is not None:                                # the parent's contact over the same window, exact
                B = local_time_at(reference.arrays, rm, [t0, t0 + T_w])
                ref = B[:, 1] - B[:, 0]
            B = local_time_at(win.arrays, cm, self._block_ends(T_w))
            reads.append((np.diff(B, axis=1, prepend=B[:, :1]), ref))
        rows = [surface_envelope(ell, w, rho2_over_D_hi=hi, reference=ref, tol=tol) for ell, ref in reads]
        edge = float(min(r[0] for r in rows))
        restated = [surface_envelope(ell, w, rho2_over_D_hi=edge, reference=ref, tol=np.inf)[1:] if edge > 0 else (0.0, 0.0)
                    for ell, ref in reads]
        del reads
        fid = copy.deepcopy(self.meta.get("fidelity") or {})
        segs = fid.get("windows")
        if segs:
            for f_i, (e, f) in zip(segs, restated):
                f_i.update(err_surface=e, floor_surface=f, surface_rho2_over_D_max=edge)
                restate_maxima(f_i)
            if fid.get("combined") == "bound_over_windows":
                fid = combine_window_fidelity(segs)
            else:
                fid.update(err_surface=float(sum(e for e, _ in restated)), floor_surface=float(max(f for _, f in restated)), windows=segs)
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

    def walker_primitives(self, acquisition, *, tol=PLAN_TOL, K=None, modes=None):
        """What ``acquisition`` (a :class:`~dmipy_sim.replay.study.Acquisition`, or a sequence) leaves of every
        walker before any tissue or scanner is applied -- the bands and the path channel contracted once, the
        exposures and the contact read once: a :class:`~dmipy_sim.replay.study.Primitives`, whose ``signals(tissue,
        scanner)`` is :meth:`walker_signals` for every pair without touching the bands again (dmipy-sim#297). The bands and
        modes read are the plan's (``tol=``, ``K=``, ``modes=`` as in :meth:`replay`), recorded as ``Primitives.plan``. The
        plan weighs the acquisition's gradient and gate; the tissue and the scanner are not yet known, so a field is
        planned at the pack's certified envelope (its largest ``B0`` and ``chi``) -- the mode count :meth:`replay` reads as well, and a bound for every pair."""
        from .study import walker_primitives
        return walker_primitives(self, acquisition, tol=tol, K=K, modes=modes)

    def study(self, study):
        """The signal for every pair of a :class:`~dmipy_sim.replay.study.Study`, ``(pairs, n_meas)``: the
        protocol's acquisitions contracted once each, every tissue and scanner applied elementwise."""
        from .study import study_signals
        return study_signals(self, study)

    def pose_response(self, waveform, *, tissue=None, scanner=None, pose=None, compartment=None,
                      method="auto", keep=None, cache=None, backend="jax", device=None):
        """The pack's response over every pose of its substrate, for one acquisition: a :class:`PoseResponse` whose
        coefficients a voxel's orientation distribution contracts against (RPH.md 6).

        **Closed form, by default (#197).** For a single-direction encoding the response is a sum of plane waves in
        each walker's rotated vector, and its harmonics are the Rayleigh expansion, computed per walker with no
        rotation ever evaluated; with a field the response is that expansion times the field factor's, coupled
        with the real Clebsch-Gordan tables. The band follows the walkers' phase amplitudes by construction and
        nothing is chosen. ``keep = (lmax, nmax)`` restricts what is computed to what the composition retains: an
        ODF or peaks composition keeps ``n = 0``, a frame keeps everything; ``None`` in either slot means the
        response's own band.

        **A magnet's own gradient** (``ScannerSequence.with_background_gradient``) is a second plane wave in each
        walker's background vector, coupled to the first as the field factor is; the encoding beside it is what
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
            from dmipy_replay.acquisition.waveforms import rotate_waveform  # expansion reads, in P and from the waveform itself
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

    def pose_series(self, waveforms, *, tissue=None, scanner=None, pose=None, compartment=None, keep=None, tol=1e-8,
                    direction_tol=None, backend="jax", device=None):
        """The pose responses of a machine's encoding classes as one series (dmrai-lab/dmipy-sim#610): ONE pass over the
        walkers for the whole set, then every class an evaluation. Returns a :class:`~dmipy_sim.replay.pose_series.PoseSeries`
        whose :meth:`~dmipy_sim.replay.pose_series.PoseSeries.responses` are each class's :class:`PoseResponse`.

        The classes of one timing class differ only in each row's amplitude and direction, the magnet's own gradient's
        magnitude and direction, and each row's small departure from its commanded shape; the bodies of the closed form
        depend on the amplitude and ``|g0|`` alone, so they are a Chebyshev series in both whose coefficients come from
        the bodies at the series' nodes, and a class is that series at its own ``(amplitude, |g0|)`` with its exact lab
        side. Every truncation and each row's departure is bounded and added to its misfit
        (:mod:`~dmipy_sim.replay.pose_series` derives each bound).

        ``waveforms`` -- the classes, an iterable read once (a generator keeps one composed class in memory at a time):
        one timing class, refused by name otherwise. ``tissue``, ``scanner``, ``pose``, ``compartment``, ``keep``,
        ``backend`` and ``device`` are :meth:`pose_responses`'; ``tol`` is each series' and band's truncation target,
        ``direction_tol`` the largest departure a row may have from its shape (a tenth of the floor by default)."""
        from .pose_series import pose_series
        return pose_series(self, waveforms, tissue=tissue, scanner=scanner, pose=pose, compartment=compartment, keep=keep,
                           tol=tol, direction_tol=direction_tol, backend=backend, device=device)

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
        h.update(np.ascontiguousarray(P["G_eff_wf"], np.float64).tobytes())  # the acquisition on its own grid
        h.update(np.float64(P["dt_wf"]).tobytes())
        h.update(np.ascontiguousarray(P["ew"], np.float64).tobytes())        # the weights with every tissue knob applied
        h.update(np.ascontiguousarray(self.substrate_frame).tobytes())
        gate = waveform.effective_gate if waveform.gate is None else waveform.gate   # what the field accrues through
        h.update(np.ascontiguousarray(gate, np.float32).tobytes())
        h.update(repr((float(P["norm"]), P["B0"], tuple(np.round(np.asarray(P["b0_dir"], float), 12)), P["chi_iso"],
                       P["chi_aniso"], method, None if keep is None else tuple(keep),
                       tuple(np.round(np.asarray(P["voxel"], float), 12)))).encode())
        return root / (h.hexdigest() + ".npz")

    def prefix(self, TE, *, K=None, out_path=None, tol=2.0, id=None, provenance=None):
        """This walk cut at the shorter echo time ``TE`` (s) and re-projected on the prefix's own basis, at the same
        bands per second (#199).

        The TE-prefix property (RPK.md 3) makes the walk to ``TE`` a valid walk. Every channel's stored function is
        cut there in closed form (:func:`~dmipy_sim.replay.bank.window_of_pack`: the bridge and the field modes by
        their window maps, the events sliced), so the result is a pack like any other with its own certificate --
        and that certificate is what licenses the prefix: re-pinning the bridge at ``TE`` is a new truncation,
        measured against the parent's exact reading of the window on the fidelity battery
        (``fidelity.err_reprojection``), and a prefix whose re-projection error exceeds ``tol`` times its
        Monte-Carlo floor is refused rather than written. The band starts at the same bands per second,
        ``K' = ceil(K TE / T)``, and is doubled (up to the parent's ``K``) until the certificate passes: a short
        prefix needs more than its share, since the pinned residual's truncation error does not scale with the
        duration. ``K=`` fixes the band instead and is refused as it stands. On a pack of several windows a prefix
        on whole windows is a range of them (``truncate``: the stored bands of those windows, unchanged); a cut inside
        a later window is refused by name, since the window table holds windows of one duration.
        ``provenance.prefix`` records the parent (digest, id, T, K), the cut and every band tried.
        """
        h = float(self.dt)
        T = float(self.meta["walk_params"]["T_max"])
        TE = float(TE)
        if TE < 2.0 * h * (1.0 - 1e-9) or TE > T * (1.0 + 1e-9):
            raise ValueError(f"TE = {TE * 1e3:.3f} ms is not a prefix of this {T * 1e3:.3f} ms walk (block {h * 1e6:.1f} us): "
                             f"it needs at least two blocks and at most the walk")
        if self.n_windows > 1:
            T_w = self.T_window
            whole = TE / T_w
            if abs(whole - round(whole)) <= 1e-9 * max(whole, 1.0):
                return self.truncate(int(round(whole)), id=id, out_path=out_path)   # a prefix of whole windows is a range read
            raise ValueError(f"TE = {TE * 1e3:.3f} ms cuts inside window {int(np.floor(whole))} of this pack's {self.n_windows}: a "
                             "prefix of a window table is a whole number of its windows (truncate), or the prefix of one window")
        return self._prefix_within(TE, K=K, out_path=out_path, tol=tol, id=id, provenance=provenance)

    def _prefix_within(self, TE, *, K=None, out_path=None, tol=2.0, id=None, provenance=None):
        """:meth:`prefix` by re-projection: the stored functions cut at ``TE`` and re-projected on the prefix's basis
        (:func:`~dmipy_sim.replay.bank.window_of_pack` at ``t0 = 0``), as one window of its own duration."""
        from .bank import window_of_pack
        T = float(self.T_window)
        T_cut = float(TE)
        K_new = int(K) if K is not None else max(2, int(np.ceil(self.K * T_cut / T)))
        tried = []
        while True:
            prov = dict(provenance or {})
            prov["prefix"] = dict(parent_digest=self.digest, parent_id=self.meta.get("id"), parent_T_s=T, parent_K=int(self.K),
                                  TE_s=T_cut, K=K_new, K_tried=tried + [K_new],
                                  note="the parent's stored functions cut at TE and re-projected on the prefix's basis; the band "
                                       "starts at the parent's bands per second and doubles until the certificate passes")
            pk = window_of_pack(self, 0.0, T_cut, K=K_new, id=(id or f"{self.meta.get('id')}/prefix-{T_cut * 1e3:.0f}ms"), provenance=prov)
            fid = pk.meta.get("fidelity", {})
            # what the prefix ADDS is judged: its re-projection against the parent's exact reading of the window; the
            # walk-free terms the parent carries (its band truncation at the free variances) are inherited, scaled
            if K is not None or fid["err_reprojection"] <= tol * fid["floor_max"] or K_new >= int(self.K):
                break
            tried.append(K_new)
            K_new = min(2 * K_new, int(self.K))
        if fid["err_reprojection"] > tol * fid["floor_max"]:
            raise ValueError(f"the prefix at TE = {T_cut * 1e3:.1f} ms with K' = {K_new} does not reproduce the parent's reading of "
                             f"it within {tol:g} x its Monte-Carlo floor (re-projection error {fid.get('err_reprojection'):.3g}, floor "
                             f"{fid.get('floor_max'):.3g}); give a larger K= or keep the parent")
        if out_path is not None:
            write_rpk(out_path, pk.arrays, pk.meta)
            pk.source = out_path
        return pk

    def _pose_coeffs_closed_many(self, Ps, waveforms, keep=None, tol=1e-8, l_cap=64, direction_tol=None, device="auto"):
        """The pose expansion in closed form (#197) for a batch of acquisitions on this pack -- the encoding classes
        of a machine pass, or one class per voxel -- in ONE pass over the walkers: every acquisition's waveform
        groups lie along one group axis, so the vectors, the Bessel values, the vector harmonics and the products
        against the field factor run once for all of them, in chunks of groups, and the lab-side assembly is per
        acquisition. The field factor is one per pass: the acquisitions of a pass share the inputs of the field factor
        (:func:`_field_factor_inputs`: the gate the field accrues through, the field and the tissue's chi), and a
        batch whose acquisitions do not is expanded in one pass per distinct factor. Returns one
        :class:`PoseResponse` per acquisition, ``None`` where an acquisition is not single-direction
        (dmrai-lab/dmipy-sim#449).

        waves in the rotated vector of each walker, and a plane wave's harmonics are the Rayleigh expansion.
        
        Measurement ``i`` plays ``G_i(t) = g_i s_i(t)``; walker ``w``'s phase at pose ``R`` is ``kappa g^ . R m^``
        with ``m_w = gamma sum_t s_i(t) r_w(t) dt`` (the vector, in the canonical frame) and ``kappa = |g||m|``. Then
        
            exp(i kappa g^.R m^) = 4 pi sum_l i^l j_l(kappa) sum_m Y_lm(g^) sum_n M^l(R)[m, n] Y_ln(m^)
        
        so in the basis ``sqrt(2l+1) M^l(R)[m, n]`` the ``(l, m, n)`` coefficient of the ensemble is
        
            c^l_mn = 4 pi i^l / sqrt(2l+1) * Y_lm(g^_i) * sum_w ew_w j_l(kappa_w) Y_ln(m^_w) / norm
        
        -- one contraction over walkers per order, no rotation ever evaluated. ``j_l(kappa)`` dies above
        ``l ~ kappa``, so the band is the largest phase amplitude and nothing is chosen: orders are added until
        their weighted Bessel tail is below ``tol``. ``keep`` restrains the band as before: an ODF or peaks
        composition keeps ``n = 0`` only, which here is the Legendre polynomial of the vector's angle to the
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
        walker exactly as the vectors are and bounded over every pose, moves the ensemble's signal by at most
        ``direction_tol`` (default a tenth of the ensemble's floor ``1 / sqrt(n_w)``): since ``|exp(i a) - exp(i b)|
        <= |a - b|``, the walkers' bounds weighted as the signal weighs them bound it, and that is added to the
        measurement's misfit. Beyond it the acquisition takes the quadrature.

        **A magnet's own gradient.** An acquisition played in a magnet with its own gradient ``g0`` (through
        every pulse and dead time) plays two directions with two time courses; ``g0`` through the effective gate
        is taken out of the encoding before the direction is judged, and enters as a second plane-wave factor in
        the walkers' background vectors (:meth:`_closed_with_background`, dmrai-lab/dmipy-sim#565).

        **Where it runs.** ``device`` is the kernels' word (:func:`pose_device.route`): ``"auto"`` the JAX device
        when it is a GPU, else numpy; ``"torch"`` / ``"torch:<device>"`` torch (dmrai-lab/dmipy-sim#603), which keeps
        the walkers' vectors, their directions and the bodies on its device, takes every route the host takes (the
        shell series where the host takes it, so the misfit is the host's), and returns only what the lab side reads.
        """
        from . import so3
        from dmipy_replay.replay.compression import read_position_coeffs
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
        dt, ew, norm = P0["dt"], P0["pathway"] * P0["ew"], P0["norm"]
        n_w = ew.shape[0]
        field = self._field_quadratic(P0, waveforms[0]) if P0["B0"] is not None else None   # (a_w, A_w) or None
        run = current()
        _ph = (lambda name, **f: run.phase(name, **f)) if run is not None else (lambda name, **f: None)
        _ph("moments", n_acq=int(n_acq), n_w=int(n_w))
        # every acquisition's directions and grouped profiles; an acquisition with a multi-axis measurement is left out
        per = []
        gates = []                                                                 # the distinct gates a background accrues through
        for P, wf, bg in zip(Ps, waveforms, backgrounds):
            # every course on the waveform's own grid: the projection onto the bands reads it there exactly
            if bg is None:
                G_wf = np.asarray(P["G_eff_wf"], np.float64)
            else:
                # the magnet's own gradient leaves the encoding: g0 through the effective gate is the second plane wave
                g0, gate = bg
                G_wf = P["G_eff_wf"] - g0[:, None, :] * gate[None, :, None]
            G = G_wf
            if bg is not None:
                key = (gate.tobytes(), float(P["dt_wf"]))
                gate_id = next((j for j, gk in enumerate(gates) if gk[0] == key), None)
                if gate_id is None:
                    gate_id = len(gates); gates.append((key, gate, float(P["dt_wf"])))
            n_meas = G.shape[0]
            g_hat = np.zeros((n_meas, 3)); s_wave = np.zeros((n_meas, G.shape[1])); zero = np.zeros(n_meas, bool)
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
        for seg, t0, T_s in P0["windows"]:                                     # the windows' vectors sum (RPK.md 4.3)
            C = read_position_coeffs(seg.arrays, dtype=np.float64).reshape(n_w, -1)
            if kind == "torch":
                C = torch.as_tensor(np.require(C, requirements="W"), device=t_dev)
            # every group's profile and every background's gate projected on its own grid, as one axis: axis b of the
            # vectors reads its own coefficients against it, m_w[b] = gamma int s(t) r_w(t)_b dt
            W1 = np.zeros((seg.K + 2, n_grp + n_gates))
            for c in live:
                q = per[c]
                prof = np.einsum("mtc,mc->mt", q["G_wf"][q["first"]], q["g_hat"][q["first"]])
                W1[:, g_off[c]:g_off[c] + len(q["first"])] = _compile_continuous(prof[:, :, None], q["P"]["dt_wf"], seg.K, T_s, t0)
            for j, (_key, gate, dt_g) in enumerate(gates):
                W1[:, n_grp + j] = _compile_continuous(gate[None, :, None], dt_g, seg.K, T_s, t0)[:, 0]
            if kind == "torch":
                W1 = torch.as_tensor(W1, device=t_dev)
            for b_ in range(3):
                m[:, :, b_] += C[:, b_::3] @ W1
        if kind == "torch":
            del C
            m = m @ torch.as_tensor(self.substrate_frame, device=t_dev)       # stored -> canonical: F^T m, per walker
            n_bg = m[:, n_grp:, :].cpu().numpy()                               # (n_w, n_gates, 3): the background vectors
            m = m[:, :n_grp, :]
            kappa = torch.linalg.vector_norm(m, dim=2)                         # (n_w, n_grp), radians
            m_hat = m / torch.where(kappa > 0, kappa, torch.ones_like(kappa))[:, :, None]
            m_hat[kappa == 0] = torch.tensor([0.0, 0.0, 1.0], dtype=m.dtype, device=t_dev)
            del m
        else:
            m = m @ self.substrate_frame                                       # stored -> canonical: F^T m, per walker
            n_bg = m[:, n_grp:, :]                                             # (n_w, n_gates, 3): the background vectors
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
                # the body side: every walker's vector direction's harmonics, all orders in one recurrence pass,
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

        ``m`` the encoding's vector and ``n_w`` the walker's BACKGROUND vector (the column the time-segment layout
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
        channels, n_cols, K_fb = _factor_channels(None if F_sh is None else L_f, L_b)
        if F_sh is not None:
            b_lab = np.asarray(b0_dir, np.float64); b_lab = b_lab / np.linalg.norm(b_lab)
            Yb = so3.real_sh(L_f, b_lab[None, :], full=True)[0]
        Lam_max = L_f + L_b
        Y_n = [put(so3.real_sh(L_b, n_hat[:, j, :], full=True)) for j in range(n_hat.shape[1])]
        F_sh_k = None if F_sh is None else put(F_sh)
        K_fb_k = {} if F_sh is None else {pair: {Lam: put(K.conj()) for Lam, K in Kp.items()} for pair, Kp in K_fb.items()}

        def factor(k):
            """The per-walker factor of bucket ``k``, ``(n_w, n_cols)`` complex, where the route keeps its walkers."""
            return _factor_of(put(so3.spherical_jn_all(L_b, x[:, k])), Y_n[k_gate[k]], F_sh_k, K_fb_k, channels, L_b,
                              n_cols, new)

        L_tot = L + Lam_max
        want_l, want_n = (None, None) if keep is None else (keep[0], keep[1])
        keep_l = L_tot if want_l is None else min(int(want_l), L_tot)
        keep_n = L_tot if want_n is None else min(int(want_n), L_tot)
        n_feat = so3.n_so3_coeffs(keep_l, keep_n)
        _ph("harmonics", L=int(L), L_f=int(L_f), L_b=int(L_b), keep_l=int(keep_l), keep_n=int(keep_n), n_feat=int(n_feat))
        l_used = [l for l in range(L + 1) if l <= keep_l + Lam_max]
        offs, tables = _coupling_tables(l_used, Lam_max, keep_l, keep_n)   # (l, Lambda) -> (Ls, K_lab3, K_body, slices)
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
            lam = _factor_lab(channels, so3.real_sh(L_b, u_dirs, full=True), Yb, K_fb)   # per channel: (n_u, 2 Lambda + 1)
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
        vectors are ``a_g mu_w`` with one ``mu_w`` per walker, and ``j_l(a r) = sum_k c_lk (a r)^{l+2k}`` with
        ``c_lk = (-1)^k / (2^k k! (2l+2k+1)!!)`` makes every row's body a polynomial in its amplitude:

            B_g = sum_k (a_g / A)^{l+2k} M_k,     M_k = c_lk sum_w w_w (A |mu_w|)^{l+2k} Y_ln(mu^_w) F_w,

        ``A`` the shell's largest amplitude. The walkers are contracted once per power instead of once per row. A
        row's waveform is its shape times its amplitude only nearly -- a machine's Maxwell term along the encoding
        goes as the square of the amplitude, and its float32 rounding -- so its own vector ``m_gw`` differs from
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
            Y = so3.real_sh(L, m_hat[:, grp[g_ref], :], full=True)             # (n_w, (L+1)^2): the shell's vector directions
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
        ``mu_kw = sum_t u_k(t) r_w(t)`` contracted through the windows as the vectors are, and is bounded over the
        rotations by ``sum_k |c_ik| |mu_kw|`` (:meth:`_pose_coeffs_closed_many`). Windows add. The bound is the
        ENSEMBLE's, which is what the misfit bounds: ``|sum_w w_w (e^{i(phi_w + d_w)} - e^{i phi_w})| <= sum_w |w_w|
        |d_w|``, the walkers' bounds weighted as the signal weighs them. ``device`` as in
        :meth:`_pose_coeffs_closed_many`: on torch the walkers' contractions run on its device (the SVD of the
        residuals, a few thousand by the waveform's samples, on the host)."""
        from dmipy_replay.replay.compression import read_position_coeffs
        from .._blas import lapack_threads
        from .pose_device import route
        todo = [c for c, q in enumerate(per) if q is not None and q["residual"].size]
        if not todo:
            return
        kind, t_dev = route(device)
        live = np.asarray(ew) != 0
        w_live = np.abs(np.asarray(ew, np.float64)[live]) / float(norm)
        per_walker = {c: 0.0 for c in todo}                                    # (n_live, n_r) per acquisition, windows summed
        for seg, t0, T_s in P0["windows"]:
            C = read_position_coeffs(seg.arrays, dtype=np.float64).reshape(n_w, -1)[live]
            if kind == "torch":
                import torch
                C = torch.as_tensor(np.require(C, requirements="W"), device=t_dev)
            for c in todo:
                q = per[c]; rows = q["residual"]; ref = q["ref"][rows]
                G_s = np.asarray(q["G_wf"], np.float64)                                          # the waveform's own grid
                g = q["g_hat"][rows]
                s_ref = np.where(q["zero"][rows][:, None], 0.0, np.einsum("mtc,mc->mt", G_s[ref], q["g_hat"][ref]))
                res = G_s[rows] - s_ref[:, :, None] * g[:, None, :]                              # (n_r, n_wf, 3)
                if "g0" in q:                                       # the group's |g0| played for the row's own
                    d_beta = (q["beta"][rows] - q["beta"][ref]) / np.where(q["beta"][rows] > 0, q["beta"][rows], 1.0)
                    res = res + (d_beta[:, None] * q["g0"][rows])[:, None, :] * np.asarray(q["gate"], np.float64)[None, :, None]
                X = res.transpose(0, 2, 1).reshape(-1, res.shape[1])                              # (n_r * 3, n_wf)
                with lapack_threads():                                                            # #564: this shape hangs a known build
                    U, S, Vt = np.linalg.svd(X, full_matrices=False)
                k = int(np.sum(S > 1e-12 * S[0])) if S.size and S[0] > 0 else 0
                if k == 0:
                    continue
                coef = (U[:, :k] * S[:k]).reshape(len(rows), 3, k)                                # c_ik per axis
                if kind == "torch":
                    # every course in one projection, each axis reading its own rows of the coefficients
                    Wk = torch.as_tensor(_compile_continuous(Vt[:k, :, None], q["P"]["dt_wf"], seg.K, T_s, t0), device=t_dev)   # (K+2, k)
                    mu = torch.stack([C[:, b_::3] @ Wk for b_ in range(3)], dim=2)                 # (n_live, k, 3)
                    nrm = torch.linalg.vector_norm(mu, dim=2) @ torch.as_tensor(np.linalg.norm(coef, axis=1).T, device=t_dev)
                    per_walker[c] = per_walker[c] + nrm                                       # (n_live, n_r) on the device
                    continue
                # every course in ONE projection (as the torch branch above), each axis reading its own rows of the
                # coefficients: one batched projection and 3 BLAS level-3 matmuls over every mode at once (#698)
                Wk = _compile_continuous(Vt[:k, :, None], q["P"]["dt_wf"], seg.K, T_s, t0)          # (K+2, k)
                mu = np.stack([C[:, b_::3] @ Wk for b_ in range(3)], axis=2)                          # (n_live, k, 3)
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
        if P["ch"].get("susc_field_basis") is None:
            raise ValueError("the pose expansion with a field needs the pack's field channel (C3)")
        Psi, names = _path_field_channels(P, waveform)                           # the one read of C3, windows summed
        from dmipy_replay.replay.field import iso_P6
        i_a = names.index("aniso_G_xx") if "aniso_G_xx" in names else None
        a = float(chi_iso) * float(B0) * Psi[:, names.index("iso_local")]
        six = -float(chi_iso) * float(B0) * iso_P6(Psi)
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
        ``R`` is ``<R, M_w>`` with ``M_w[a, b] = gamma int G[t, a] r_w[t, b] dt``, so the walk is contracted once
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
        from dmipy_replay.replay.compression import read_position_coeffs
        G_wf, ew, norm, B0 = P["G_eff_wf"], P["pathway"] * P["ew"], P["norm"], P["B0"]
        T_w = self.T_window
        b0_dir, chi_iso, chi_aniso = P["b0_dir"], P["chi_iso"], P["chi_aniso"]
        n_meas, n_w = G_wf.shape[0], ew.shape[0]

        C = read_position_coeffs(self.arrays, dtype=np.float64).reshape(n_w, -1)
        # The gradient term at pose R is <R, M_w> with M_w[a, b] = gamma int G_i(t)_a r_w(t)_b dt: the walk is
        # contracted against the waveform once, and every pose after that is a 3x3 inner product. Exact for a
        # multi-axis waveform too, which an expansion in one gradient direction could not take at all.
        Q = np.empty((n_w, n_meas, 3, 3))
        for a in range(3):
            Wa = _compile_continuous(np.asarray(G_wf, np.float64)[:, :, a][:, :, None], P["dt_wf"], self.K, T_w, 0.0)
            for b_ in range(3):
                Q[:, :, a, b_] = C[:, b_::3] @ Wa

        Psi = names = i_a = None
        if B0 is not None:
            pm = self.meta.get("compression", {}).get("channels", {}).get("susc_field_basis")
            if pm is None:
                raise ValueError("the pose expansion with a field needs the pack's field channel (C3)")
            from dmipy_replay.replay.field import path_field_integral
            from dmipy_replay.replay.field import iso_P6
            Psi, names = path_field_integral(self.arrays, pm, waveform, T_w)   # (n_w, n_ch)
            i_a = names.index("aniso_G_xx") if "aniso_G_xx" in names else None
        b = np.asarray(b0_dir, float); b = b / np.linalg.norm(b)

        # a pose R is a rotation of the CANONICAL substrate frame: the stored walk is first turned into it
        # (F^T, RPK.md 4.2) and then by R -- the identity for a pack whose frame is the identity
        Qf_axis = self.substrate_frame.T

        from .pose_device import pose_samples
        field_terms = None
        if Psi is not None:
            field_terms = (float(chi_iso) * float(B0), (float(chi_aniso) * float(B0) if (chi_aniso and i_a is not None) else 0.0), b,
                           Psi[:, names.index("iso_local")], iso_P6(Psi), (Psi[:, i_a:i_a + 6] if i_a is not None else None))
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


def _jax_bloch():
    """The JAX vector-Bloch propagator (:func:`~dmipy_sim.replay.trajectories.replay_bloch_jax`)."""
    from .trajectories import replay_bloch_jax
    return replay_bloch_jax


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
        E = np.stack([_evaluate(form, waveform, _r_so3.rotation_of(u), evaluated) for u in dirs])   # (n_q, n_meas)
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


def _factor_channels(L_f, L_b):
    """The channels of the factor a gradient's plane wave is coupled to in the closed form with a magnet's own
    gradient (:meth:`ReplayPack._closed_with_background`): ``(channels, n_cols, K_fb)``, each channel ``(Lambda,
    its columns in the factor, its kind)``. Without a field (``L_f`` None) a channel is the background's order
    ``l2`` itself (kind ``("b", l2)``); with one it is an order ``Lambda`` the field's order ``l'`` and the
    background's ``l2`` couple into (kind ``("fb", l', l2)``, ``K_fb[(l', l2)]`` the real coupling tables)."""
    from . import so3
    if L_f is None:
        return [(l2, so3.sh_block(l2, True), ("b", l2)) for l2 in range(L_b + 1)], (L_b + 1) ** 2, {}
    channels, n_cols, K_fb = [], 0, {}
    for lp in range(L_f + 1):
        for l2 in range(L_b + 1):
            K_fb[(lp, l2)] = so3.coupling(lp, l2)
            for Lam in range(abs(lp - l2), lp + l2 + 1):
                channels.append((Lam, slice(n_cols, n_cols + 2 * Lam + 1), ("fb", lp, l2)))
                n_cols += 2 * Lam + 1
    return channels, n_cols, K_fb


def _factor_of(J, Y_n, F_sh, K_fb_conj, channels, L_b, n_cols, new):
    """The per-walker factor ``(n_w, n_cols)`` complex of a magnet's own gradient (with the field's coupled in): the
    background's columns ``4 pi i^l2 j_l2(|g0| |n_w|) Y_l2(n^_w)`` from ``J`` ``(L_b+1, n_w)`` and ``Y_n`` ``(n_w,
    (L_b+1)^2)``, then, with a field (``F_sh`` ``(n_w, (L_f+1)^2)``, ``K_fb_conj`` the conjugated coupling tables of
    :func:`_factor_channels`), every channel's coupling of the field's harmonics with them walker by walker. ``new``
    allocates on the route's device."""
    from . import so3
    n_w = J.shape[1]
    Fb = new((n_w, (L_b + 1) ** 2))
    for l2 in range(L_b + 1):
        b2 = so3.sh_block(l2, True)
        Fb[:, b2] = (4 * np.pi * (1j ** l2)) * J[l2][:, None] * Y_n[:, b2]
    if F_sh is None:
        return Fb
    F = new((n_w, n_cols))
    for (lp, l2), Kc in K_fb_conj.items():
        prod = (F_sh[:, so3.sh_block(lp, True)][:, :, None] * Fb[:, so3.sh_block(l2, True)][:, None, :]).reshape(n_w, -1)
        for Lam, cols, ch in channels:
            if ch[1:] == (lp, l2):
                F[:, cols] = prod @ Kc[Lam]
    return F


def _factor_lab(channels, Y_u, Yb, K_fb):
    """The lab side of every channel of :func:`_factor_channels`, ``[(n_u, 2 Lambda + 1)]`` in channel order: the
    background directions' harmonics ``Y_u`` ``(n_u, (L_b+1)^2)`` alone, or coupled with the field direction's ``Yb``
    ``((L_f+1)^2,)`` by the tables ``K_fb``."""
    from . import so3
    lam = []
    for Lam, _cols, ch in channels:
        if ch[0] == "b":
            lam.append(Y_u[:, so3.sh_block(ch[1], True)])
        else:
            _, lp, l2 = ch
            pair = (Yb[so3.sh_block(lp, True)][None, :, None] * Y_u[:, None, so3.sh_block(l2, True)]).reshape(Y_u.shape[0], -1)
            lam.append(pair @ K_fb[(lp, l2)][Lam])
    return lam


def _coupling_tables(l_used, Lam_max, keep_l, keep_n):
    """``(offs, tables)``: where each coupled order ``Lc`` starts in a :class:`PoseResponse`'s coefficients, and per
    gradient order ``l`` and factor order ``Lambda`` the coupled orders ``Ls`` they reach within ``keep_l`` with
    their real coupling tables on the lab index (``K_lab3`` ``(2l+1, 2Lambda+1, sum 2Lc+1)``) and on the body
    index restricted to ``|n| <= keep_n`` (``K_body`` ``((2l+1)(2Lambda+1), sum 2kk+1)``), and each ``Lc``'s slices
    of the two: ``tables[(l, Lambda)] = (Ls, K_lab3, K_body, lab_sl, body_sl)``."""
    from . import so3
    offs, off = {}, 0
    for Lc in range(keep_l + 1):
        offs[Lc] = off; off += (2 * Lc + 1) * (2 * (so3._n_cols(Lc, keep_n) // 2) + 1)
    tables = {}
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
    return offs, tables


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
    n_off = off.sum(axis=1)
    some = n_off > 0
    g0[some] = np.einsum("mt,mtc->mc", off[some].astype(np.float64), G[some]) / n_off[some, None]
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
            float(P["dt"]), tuple((float(t0), float(T_s)) for _seg, t0, T_s in P["windows"]))
















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
