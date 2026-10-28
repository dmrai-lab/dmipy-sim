"""A single ``.rpk`` pack on the Hub, read by plan: ``hf://owner/name/path.rpk`` opened by reference, never downloaded whole.

A ``.rpk`` is a safetensors file: a header that gives every tensor's dtype, shape and byte range, then the tensors. The
position bands of a pack stored in a band container lie in one tensor per axis and band range
(``pos_x_b<i>[u<bits>]``, :func:`~dmipy_sim.replay.compression.range_name`), the field modes in one tensor per mode range
(``susc_path_dct_b<i>``); each is a contiguous byte range, a walker's row of it contiguous within. :class:`RemotePack`
reads the header (two HTTP Range requests), plans an acquisition from it
(:meth:`RemotePack.plan`: the rule of :mod:`~dmipy_sim.replay.planning`, the same plan the pack gives once downloaded), fetches
the byte ranges of the tensors that plan names and of nothing else (adjacent ranges in one request), and replays on them::

    pack = ReplayPack.open("hf://SubstrateCommons/canonical-pores/packs/cylinder/d05.00um.rpk")
    plan = pack.plan(seq)                        # bands, modes, error and bytes, from the header alone (modes: a pilot of walkers)
    S = pack.replay(seq)                         # fetches the plan's tensors; S.plan is the plan, pack.bytes_fetched the cost
    whole = pack.load()                          # the whole file, when it is asked for

What a replay needs besides the bands is fetched whole: the endpoints and the band scales, the spin weights, the occupancy, the
contact tensors when the tissue has a relaxivity, the field modes when the tissue and the scanner give a field term. A tensor
fetched stays held, so a later acquisition fetches only the ranges it adds.
"""
import json
import re

import numpy as np

from .planning import PLAN_TOL, planned
from .publish import (_ST_DTYPES, _auth_headers, _coalesce, _fetch_header, _range_get, is_hub_uri, parse_uri)
from .replay import ReplayPack

_POS_RANGE = re.compile(r"pos_[xyz]_b(\d+)(?:u\d+)?$")
_MODE_RANGE = re.compile(r"susc_path_dct_b(\d+)(?:u\d+)?$")


class _Stub:
    """The shape and dtype of a tensor that has not been fetched: what a pack's shape readers ask of its arrays."""
    def __init__(self, shape, dtype):
        self.shape = tuple(int(x) for x in shape)
        self.dtype = np.dtype(dtype)
        self.ndim = len(self.shape)

    @property
    def nbytes(self):
        return int(np.prod(self.shape, dtype=np.int64)) * self.dtype.itemsize

    def __len__(self):
        return self.shape[0]


class _Skeleton(ReplayPack):
    """The pack of a remote file's header: shapes and metadata, no tensor. It plans; the field modes' pilot variance is
    the remote pack's to fetch."""
    remote = None

    def _mode_variance(self, seg, pm):
        from .planning import stored_mode_variance
        stored = stored_mode_variance(self.meta)
        return stored if stored is not None else self.remote._pilot_mode_variance(pm)


class RemotePack:
    """A ``.rpk`` file on the Hub, open by reference. ``uri`` is ``hf://owner/name/path.rpk``; ``url=`` / ``revision=`` replace
    the Hub's own URL and revision resolution (how a test drives it past a plain HTTP server, as
    :func:`~dmipy_sim.replay.publish.fetch_windows` is). ``meta`` is the pack's header; :attr:`bytes_fetched` the bytes read
    so far, the header's included."""

    def __init__(self, uri, *, url=None, revision=None, session=None):
        import requests
        self.uri = uri
        repo, path = parse_uri(uri)
        if not path.endswith(".rpk"):
            raise ValueError(f"{uri}: a pack on the Hub is a .rpk file; a columnar layout is opened by its prefix "
                             "(ReplayPack.open reads either)")
        if url is None:
            from huggingface_hub import HfApi, hf_hub_url
            revision = revision or HfApi().dataset_info(repo).sha
            url = hf_hub_url(repo, path, repo_type="dataset", revision=revision)
        self.url, self.revision = url, revision
        self._sess = session or requests.Session()
        self._headers = _auth_headers()
        n, header = _fetch_header(url, self._sess, self._headers)
        self._base = 8 + n
        self.bytes_fetched = 8 + n + 8
        self.meta = json.loads((header.pop("__metadata__", None) or {}).get("rpk") or "{}")
        self._table = header
        self._held = {}
        self._views = {}
        self._variance = {}
        from .replay import _refuse_predating
        _refuse_predating(self.meta, uri, tensors=list(header))
        stubs = {k: _Stub(d["shape"], _NP[d["dtype"]]) for k, d in header.items()}
        self._skeleton = _Skeleton(stubs, self.meta, source=uri)
        self._skeleton.remote = self
        self._skeleton.susceptibility_field                          # refuses a header whose declared field and channels disagree

    # ---- what the file holds
    @property
    def id(self):
        return self.meta.get("id")

    @property
    def K(self):
        return self._skeleton.K

    @property
    def n_walkers(self):
        return self._skeleton.n_walkers

    @property
    def nbytes(self):
        """The size of the file's tensors (bytes): what a whole download costs, beside the header."""
        return sum(int(d["data_offsets"][1] - d["data_offsets"][0]) for d in self._table.values())

    def load(self):
        """The whole file as a :class:`~dmipy_sim.replay.replay.ReplayPack`, downloaded and checked against the dataset's
        manifest (:func:`~dmipy_sim.replay.publish.fetch`): the one call that moves the whole pack."""
        return ReplayPack.load(self.uri)

    # ---- the plan
    def plan(self, waveform, *, tissue=None, scanner=None, orientation=None, tol=PLAN_TOL, K=None, modes=None):
        """The :class:`~dmipy_sim.replay.planning.ReadPlan` of ``waveform`` -- :meth:`ReplayPack.plan` on the header. The
        band variances are the header's record or the free bridge's, so the bands are planned before a tensor is read; a field
        term needs the mode variances, the header's record or a pilot of the first walkers' modes fetched once (their bytes are
        in :attr:`bytes_fetched`, not in the plan's ``bytes_read``)."""
        return self._skeleton.plan(waveform, tissue=tissue, scanner=scanner, orientation=orientation, tol=tol, K=K, modes=modes)

    def _pilot_mode_variance(self, pm):
        from .planning import pilot_size
        from .replay import _measured_mode_variance
        n_p, why = pilot_size(self.meta, self.n_walkers)
        key = (n_p, json.dumps(pm, sort_keys=True, default=str))
        if key not in self._variance:
            names = [k for k in self._table if "/" not in k and (k.startswith("susc_path") or k == "band_block")]
            rows = self._fetch_rows(names, n_p)
            self._variance[key] = _measured_mode_variance(rows, pm)
        return self._variance[key], why

    # ---- byte ranges
    def _bytes_of(self, name):
        a, b = self._table[name]["data_offsets"]
        return self._base + a, self._base + b - 1

    def _blob(self, ranges):
        """The bytes of ``ranges`` (inclusive ``(start, end)`` pairs), adjacent ranges coalesced into one request each."""
        merged = _coalesce(ranges)
        got = {r: _range_get(self.url, r[0], r[1], self._sess, self._headers) for r in merged}
        self.bytes_fetched += sum(e - s + 1 for s, e in merged)
        return merged, got

    def _fetch(self, names):
        """Fetch the tensors ``names`` not held yet, by byte range, and hold them."""
        want = [n for n in names if n not in self._held]
        if not want:
            return
        spans = {n: self._bytes_of(n) for n in want}
        merged, got = self._blob(list(spans.values()))
        for n, (s, e) in spans.items():
            ms, me = next(r for r in merged if r[0] <= s and e <= r[1])
            d = self._table[n]
            blob = got[(ms, me)][s - ms: e - ms + 1]
            self._held[n] = np.frombuffer(blob, dtype=_NP[d["dtype"]]).reshape(d["shape"])

    def _fetch_rows(self, names, n_rows):
        """The first ``n_rows`` rows of the walker-leading tensors ``names`` (contiguous bytes at the start of each) and the
        tables among them whole: what a pilot of the first walkers reads."""
        n_w = self.n_walkers
        spans, shapes = {}, {}
        for n in names:
            d = self._table[n]
            shape = list(d["shape"])
            s, e = self._bytes_of(n)
            if shape and shape[0] == n_w and not n.endswith("_scale"):
                row = (e - s + 1) // n_w
                e = s + n_rows * row - 1
                shape[0] = n_rows
            spans[n], shapes[n] = (s, e), shape
        merged, got = self._blob(list(spans.values()))
        out = {}
        for n, (s, e) in spans.items():
            ms, me = next(r for r in merged if r[0] <= s and e <= r[1])
            out[n] = np.frombuffer(got[(ms, me)][s - ms: e - ms + 1], dtype=_NP[self._table[n]["dtype"]]).reshape(shapes[n])
        return out

    # ---- the replay
    def view(self, plan, *, contact=False, field=True):
        """The :class:`~dmipy_sim.replay.replay.ReplayPack` of the tensors ``plan`` names: the band ranges up to its ``K``,
        the mode ranges up to its ``modes`` (with ``field``), the contact tensors with ``contact``, and the tensors every replay
        reads. The ranges are fetched here, once; the pack holds the bands of the ranges that cover the plan, which is its
        ``K`` rounded up to a range's end."""
        names, K_read, M_read = self._select(plan, contact, field)
        for (k, m, c, f), pack in self._views.items():          # a view that already holds what the plan needs is the view
            if k >= K_read and m >= M_read and c >= bool(contact) and f >= bool(field):
                return pack
        key = (K_read, M_read, bool(contact), bool(field))
        self._views.clear()                                      # a wider view supersedes the narrower ones (and their decoded bands)
        self._fetch(names)
        arrays = {n: self._held[n] for n in names}
        meta = json.loads(json.dumps(self.meta))
        cx = meta["compression"]
        cx["K"] = K_read
        if cx.get("container"):
            cx["container"] = [r for r in cx["container"] if int(r["bands"][0]) < K_read]
        for n in arrays:
            if n.split("/")[-1] == "pos_band_scale":
                arrays[n] = np.ascontiguousarray(arrays[n][..., :K_read])
        pmeta = (cx.get("channels") or {}).get("susc_field_basis")
        if M_read:
            pmeta["K"] = M_read
            if pmeta.get("container"):
                pmeta["container"] = [r for r in pmeta["container"] if int(r["modes"][0]) < M_read]
            for n in arrays:
                if n.split("/")[-1] == "susc_path_scale":
                    arrays[n] = np.ascontiguousarray(arrays[n][..., :M_read])
            if not (meta.get("columnar") or {}).get("path_mode_variance") and self._variance:
                var = next(iter(self._variance.values()))
                meta.setdefault("columnar", {})["path_mode_variance"] = [var.tolist()]          # the pilot's record, for the view's own plan
        elif pmeta is not None:
            cx["channels"].pop("susc_field_basis")
            cx["channels"].pop("susceptibility_grid", None)
            meta["view"] = dict(field="unread")
        if not contact:
            cx["channels"].pop("boundary_local_time", None)
        pack = ReplayPack(arrays, meta, source=self.uri)
        self._views[key] = pack
        return pack

    def _select(self, plan, contact, field):
        """``(names, K_read, M_read)``: the tensors a read of ``plan`` takes from the file, the bands the band ranges cover
        and the modes the mode ranges cover (0 without the field channel)."""
        cx = self.meta["compression"]
        b_starts = [int(r["bands"][0]) for r in cx.get("container") or []]
        b_ends = [int(r["bands"][1]) for r in cx.get("container") or []]
        pmeta = (cx.get("channels") or {}).get("susc_field_basis") or {}
        m_starts = [int(r["modes"][0]) for r in pmeta.get("container") or []]
        m_ends = [int(r["modes"][1]) for r in pmeta.get("container") or []]
        names, K_read, M_read = [], 0, 0
        for n in self._table:
            base = n.split("/")[-1]
            mp, mm = _POS_RANGE.match(base), _MODE_RANGE.match(base)
            if mp:
                i = int(mp.group(1))
                if b_starts[i] >= plan.K:
                    continue
                K_read = max(K_read, b_ends[i])
            elif mm:
                i = int(mm.group(1))
                if not field or m_starts[i] >= max(plan.modes, 1):
                    continue
                M_read = max(M_read, m_ends[i])
            elif base == "susc_path_dct":
                if not field:
                    continue
                M_read = int(self._table[n]["shape"][2])
            elif base in ("susc_path_scale",):
                if not field:
                    continue
            elif base.startswith("blt_") and not contact:
                continue
            elif base.startswith(("susc_grid_", "voxel_")):
                continue
            elif base in ("pos_x", "pos_y", "pos_z"):
                K_read = int(self._table[n]["shape"][1]) - 2
            names.append(n)
        if not b_starts and K_read == 0:
            K_read = self.K
        return names, K_read, M_read

    def replay(self, waveform, *, tissue=None, scanner=None, orientation=None, compartment=None, complex_signal=False,
               tol=PLAN_TOL, K=None, modes=None):
        """:meth:`ReplayPack.replay` on the tensors the plan reads, fetched by byte range: the same signal, bit for bit, as the
        pack downloaded whole and read at the same ``K`` and ``modes``. The array returned carries the plan
        (``S.plan``), of the whole pack; :attr:`bytes_fetched` is what the file cost so far."""
        waveform = waveform.waveform if hasattr(waveform, "waveform") else waveform
        plan = self.plan(waveform, tissue=tissue, scanner=scanner, orientation=orientation, tol=tol, K=K, modes=modes)
        contact = tissue is not None and (getattr(tissue, "rho2", None) is not None or getattr(tissue, "rho1", None) is not None)
        field = plan.modes_stored > 0 and self._field_active(tissue, scanner)
        view = self.view(plan, contact=contact, field=field)
        S = view.replay(waveform, tissue=tissue, scanner=scanner, orientation=orientation, compartment=compartment,
                        complex_signal=complex_signal, K=plan.K, modes=(plan.modes if field else None))
        return planned(S, plan)

    def _field_active(self, tissue, scanner):
        from .replay import field_term
        sk = self._skeleton
        return field_term(sk.susceptibility_field, tissue, scanner, sk.field_envelope).active


_NP = {k: np.dtype(v) for k, v in _ST_DTYPES.items()}


def open_remote(uri, **kw):
    """:class:`RemotePack` for ``uri``, a ``.rpk`` on the Hub (``hf://owner/name/path.rpk``)."""
    return RemotePack(uri, **kw)
