"""The decoded form of a replay pack's stored tensors, held once per pack.

A pack stores its position bands, contact bands, field modes and occupancy events quantised and bit-packed; every
reader contracts the dequantised floats. :class:`PackArrays` is the ``dict`` of a pack's tensors with a
:class:`DecodedCache` beside it, and the readers in :mod:`~dmipy_sim.replay.compression`
(:func:`~dmipy_sim.replay.compression.read_position_coeffs`, :func:`~dmipy_sim.replay.compression.susc_path_coeffs`,
:func:`~dmipy_sim.replay.compression.bridge_bands`, :func:`~dmipy_sim.replay.compression.occupancy_events`) decode a
tensor selection once and return the same read-only array to every later call that reads it.

A **prefix** read (:meth:`DecodedCache.get_prefix`) is the first ``n`` bands or modes of a tensor selection. The cache
holds, per selection, the longest prefix decoded so far: a later read of at most that length is a view of it, a longer one
decodes only the bands beyond it and replaces the entry by the join. A prefix of a selection whose full decode is held is a
view of that decode (:meth:`DecodedCache.peek`). Every prefix is bit for bit the leading entries of the full decode.

The cache holds performance copies of values the stored tensors determine; it holds nothing a certificate reads and a
reader that bypasses it computes the same numbers.
"""
import threading

import numpy as np

#: Bytes of decoded arrays one pack holds (:attr:`DecodedCache.budget`): 1 GiB, four decoded float64 copies of
#: 40,000 walkers at K = 256 and three axes.
DECODE_BUDGET_BYTES = 1 << 30


class DecodedCache:
    """Decoded arrays keyed on what was decoded: the reader, its selection (axes, dtype, channel) and the identity of
    every stored tensor the decode reads.

    An entry keeps references to those tensors, so their identities cannot be reused while it lives, and a pack whose
    tensor is replaced (a different object under the same name) reads a different key. An in-place write into a stored
    tensor is not detected: stored tensors are not written after a pack is built.

    **Budget.** An array is stored only when it fits in :attr:`budget` beside what the cache already holds; one that
    does not is returned and decoded again on the next call exactly as without a cache, and so is one that is a view of
    a stored tensor (nothing was decoded, and the stored tensor is never frozen). Nothing is evicted: the
    first selections to fit stay until :meth:`clear`. Stored arrays are read-only, so a caller cannot change what
    a later caller reads.
    """

    def __init__(self, budget=DECODE_BUDGET_BYTES):
        self.budget = int(budget)
        self._entries = {}
        self._prefixes = {}
        self._bytes = 0
        self._lock = threading.Lock()

    @property
    def nbytes(self):
        """Bytes of decoded arrays held."""
        return self._bytes

    def __len__(self):
        return len(self._entries) + len(self._prefixes)

    def clear(self):
        """Release every decoded array."""
        with self._lock:
            self._entries.clear()
            self._prefixes.clear()
            self._bytes = 0

    @staticmethod
    def _key(kind, selection, tensors):
        return (kind, selection, tuple((k, id(v)) for k, v in sorted(tensors.items())))

    def peek(self, kind, selection, tensors):
        """The held decode of ``tensors`` for ``selection`` under ``kind``, or ``None`` when it is not held. Decodes nothing."""
        with self._lock:
            hit = self._entries.get(self._key(kind, selection, tensors))
        return None if hit is None else hit[0]

    def get_prefix(self, kind, selection, tensors, n, decode, *, axis, offset=0):
        """The first ``offset + n`` entries along ``axis`` of the decode of ``tensors`` for ``selection`` under ``kind``:
        ``offset`` leading entries that are not bands (a position's two endpoints) and the ``n`` lowest bands or modes.

        ``decode(lo, hi)`` returns the array of bands ``lo .. hi - 1`` (with the ``offset`` leading entries when ``lo`` is 0).
        The entry held is the longest prefix decoded so far, so a read of at most that many is a read-only view of it and a
        longer read decodes bands ``held .. n - 1`` only and holds the join in its place, within the budget (a join that does
        not fit is returned and not held, and the shorter entry stays)."""
        key = self._key(kind, selection, tensors)
        with self._lock:
            hit = self._prefixes.get(key)
        if hit is not None and hit[1] >= n:
            return _leading(hit[0], axis, offset + n)
        lo = 0 if hit is None else hit[1]                     # a held prefix of no bands (only the endpoints) is replaced
        piece = decode(lo, n)
        stored = [t for t in tensors.values() if isinstance(t, np.ndarray)]
        if hit is None and any(np.may_share_memory(piece, t) for t in stored):
            return piece                       # a view of a stored tensor decodes nothing: neither held nor frozen
        value = piece if lo == 0 else np.concatenate([hit[0], piece], axis=axis)
        value.flags.writeable = False
        with self._lock:
            old = self._prefixes.get(key)
            freed = 0 if old is None else old[0].nbytes
            if (old is None or old[1] < n) and self._bytes - freed + value.nbytes <= self.budget:
                self._prefixes[key] = (value, int(n), tuple(tensors.values()))
                self._bytes += value.nbytes - freed
        return value

    def get(self, kind, selection, tensors, decode):
        """The decode of ``tensors`` (a ``{name: stored tensor}`` mapping) for ``selection`` under ``kind``: the held
        value, else ``decode()`` stored when it fits the budget. ``decode()`` returns an array or a tuple of arrays
        (and other objects, which count no bytes); the arrays it returns are made read-only whether held or not."""
        key = (kind, selection, tuple((k, id(v)) for k, v in sorted(tensors.items())))
        with self._lock:
            hit = self._entries.get(key)
        if hit is not None:
            return hit[0]
        value = decode()
        stored = [t for t in tensors.values() if isinstance(t, np.ndarray)]
        if any(np.may_share_memory(a, t) for a in _arrays_of(value) for t in stored):
            return value                       # a view of a stored tensor decodes nothing: it is neither held nor frozen
        n = _freeze(value)
        with self._lock:
            if key not in self._entries and self._bytes + n <= self.budget:
                self._entries[key] = (value, n, tuple(tensors.values()))
                self._bytes += n
        return value


def _leading(a, axis, count):
    """``a`` cut to its first ``count`` entries along ``axis``: a view."""
    return a[(slice(None),) * axis + (slice(0, int(count)),)]


def _arrays_of(value):
    """The arrays of ``value`` (an array, or a tuple / list / slotted record of them)."""
    if isinstance(value, np.ndarray):
        return [value]
    if isinstance(value, (tuple, list)):
        return [a for v in value for a in _arrays_of(v)]
    slots = getattr(type(value), "__slots__", None)
    return [a for s in slots for a in _arrays_of(getattr(value, s))] if slots else []


def _freeze(value):
    """Mark every array of ``value`` (an array, or a tuple / list / slotted record of them) read-only; their bytes."""
    if isinstance(value, np.ndarray):
        value.flags.writeable = False
        return value.nbytes
    if isinstance(value, (tuple, list)):
        return sum(_freeze(v) for v in value)
    slots = getattr(type(value), "__slots__", None)
    if slots:
        return sum(_freeze(getattr(value, s)) for s in slots)
    return 0


class PackArrays(dict):
    """A pack's tensors by name, with the pack's :class:`DecodedCache` as :attr:`decoded`. A ``dict`` in every other
    respect; a copy (``dict(arrays)``), a pickle and a deep copy carry the tensors and not the cache."""

    def __init__(self, arrays=(), decoded=None):
        super().__init__(arrays)
        self.decoded = DecodedCache() if decoded is None else decoded

    def __reduce__(self):
        return (PackArrays, (dict(self),))
