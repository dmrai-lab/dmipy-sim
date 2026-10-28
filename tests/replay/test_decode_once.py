"""A pack decodes each stored tensor selection once (dmipy-sim#812): the position bands, the contact bands, the field
modes and the occupancy events are dequantised on the first read and every later replay, on any route, reads the held
read-only array -- bit for bit what a decode per call gives. The decoders are counted, never timed."""
import collections

import numpy as np
import pytest

import dmipy_sim as d
from dmipy_sim import build_replay_pack
from dmipy_sim.replay import bank, compression as cx
from dmipy_sim.replay.decoded import DecodedCache, PackArrays
from dmipy_sim.replay.replay import ReplayPack
from dmipy_sim.spec.tissue import Tissue

TIS = Tissue(T2={"extra": 0.05, "intra": 0.02}, rho2=1e-5)


@pytest.fixture(scope="module")
def pack():
    """Two storage windows of a permeable cell: quantised bit-packed bands, the contact bridge, the occupancy events."""
    w = d.simulate_trajectories(300, 2e-9, d.PackedCylinders([1e-6], [[0.0, 0.0]], 10e-6, permeability=2e-5), 4e-3, 1e-4,
                                seed=1, require_gpu=False, K=8, window_T=2e-3)
    return build_replay_pack(w, id="t/decode-once", license="x", citation="x")


def _seqs(pk, n=3):
    n_t = int(round(4e-3 / pk.dt)) + 1
    return [d.pgse([[1, 0, 0]], 5e-4, 1.5e-3, bvalues=[b], TE=4e-3, n_t=n_t, slew_rate=float("inf")) for b in (5e7, 1e8, 2e8)[:n]]


@pytest.fixture
def counted(monkeypatch):
    """The decodes behind every reader, counted by the kind of tensor decoded (each reader's decode runs on a cache miss)."""
    n = collections.Counter()
    get = DecodedCache.get

    def counting(self, kind, selection, tensors, decode):
        def counted_decode():
            n[kind] += 1
            return decode()
        return get(self, kind, selection, tensors, counted_decode)
    monkeypatch.setattr(DecodedCache, "get", counting)
    get_prefix = DecodedCache.get_prefix

    def counting_prefix(self, kind, selection, tensors, count, decode, **kw):
        def counted_decode(lo, hi):
            n[kind] += 1
            return decode(lo, hi)
        return get_prefix(self, kind, selection, tensors, count, counted_decode, **kw)
    monkeypatch.setattr(DecodedCache, "get_prefix", counting_prefix)
    return n


def _fresh(pk):
    """A pack of the same tensors with an empty cache of its own."""
    return ReplayPack(pk.arrays, pk.meta)


def test_n_calls_decode_each_tensor_once(pack, counted):
    pk = _fresh(pack)
    seqs = _seqs(pk)
    for _ in range(4):
        for s in seqs:
            pk.walker_signals(s, tissue=TIS, tol=None)                  # every band: the full decode, held (a plan's prefix: test_plan_before_read)
    assert counted["position_coeffs"] == pk.n_windows           # one decode per window, not per call
    assert counted["bridge_bands"] == pk.n_windows
    assert counted["occupancy_runs"] == pk.n_windows
    assert pk.decoded_bytes > 0


def test_results_are_bit_identical_to_a_decode_per_call(pack):
    cached, bare = _fresh(pack), _fresh(pack)
    bare.decode_budget = 0                                # nothing held: every call decodes
    for s in _seqs(pack):
        for tissue in (None, TIS):
            for _ in range(2):
                a, b = cached.walker_signals(s, tissue=tissue), bare.walker_signals(s, tissue=tissue)
                for x, y in zip(a, b):
                    np.testing.assert_array_equal(x, y)
                assert complex(cached.replay(s, tissue=tissue)[0]) == complex(bare.replay(s, tissue=tissue)[0])
    assert bare.decoded_bytes == 0 and cached.decoded_bytes > 0


def test_the_reader_returns_what_the_plain_dict_decodes(pack):
    plain = dict(pack.arrays)                             # no cache: every call decodes
    for dtype in (np.float64, np.float32):
        for axes in (None, (0,), (0, 2)):
            got = cx.read_position_coeffs(pack.arrays, axes=axes, dtype=dtype)
            want = cx.read_position_coeffs(plain, axes=axes, dtype=dtype)
            assert got.dtype == want.dtype
            np.testing.assert_array_equal(got, want)


def test_the_key_is_the_selection_and_the_dtype(pack, counted):
    pk = _fresh(pack)
    a = cx.read_position_coeffs(pk.arrays)
    assert cx.read_position_coeffs(pk.arrays) is a
    assert cx.read_position_coeffs(pk.arrays, axes=[0, 1, 2]) is not a        # another selection: its own entry
    assert cx.read_position_coeffs(pk.arrays, dtype=np.float32).dtype == np.float32
    assert cx.read_position_coeffs(pk.arrays, axes=(0,)).shape[2] == 1
    assert counted["position_coeffs"] == 4
    cx.read_position_coeffs(pk.arrays, axes=(0,)); cx.read_position_coeffs(pk.arrays, dtype=np.float32)
    assert counted["position_coeffs"] == 4


def test_held_arrays_are_read_only_and_a_pack_cannot_corrupt_the_next_read(pack):
    pk = _fresh(pack)
    a = cx.read_position_coeffs(pk.arrays)
    with pytest.raises(ValueError, match="read-only"):
        a[0, 0, 0] = 1.0
    assert cx.read_position_coeffs(pk.arrays, cached=False).flags.writeable      # the uncached read is the caller's own


def test_a_replaced_tensor_is_read_afresh(pack):
    pk = _fresh(pack)
    before = cx.read_position_coeffs(pk.arrays).copy()
    ends = np.array(pk.arrays["pos_x_ends"]); ends[:, 1] += 1.0
    pk.arrays["pos_x_ends"] = ends                                                 # a different object under the same name
    after = cx.read_position_coeffs(pk.arrays)
    np.testing.assert_array_equal(after, cx.read_position_coeffs(dict(pk.arrays), cached=False))
    assert not np.array_equal(after[:, 1, 0], before[:, 1, 0])
    np.testing.assert_array_equal(after[:, 2:, :], before[:, 2:, :])


def test_a_budget_that_does_not_fit_decodes_per_call(pack, counted):
    pk = _fresh(pack)
    one = cx.read_position_coeffs(pk.arrays).nbytes
    pk.release_decoded(); counted["position_coeffs"] = 0
    pk.decode_budget = one - 1
    for _ in range(3):
        cx.read_position_coeffs(pk.arrays)
    assert counted["position_coeffs"] == 3 and pk.decoded_bytes == 0
    pk.decode_budget = one
    for _ in range(3):
        cx.read_position_coeffs(pk.arrays)
    assert counted["position_coeffs"] == 4 and pk.decoded_bytes == one
    cx.read_position_coeffs(pk.arrays, dtype=np.float32)                            # beside the first it does not fit
    cx.read_position_coeffs(pk.arrays, dtype=np.float32)
    assert counted["position_coeffs"] == 6 and pk.decoded_bytes == one


def test_release_drops_every_decoded_array(pack, counted):
    pk = _fresh(pack)
    s = _seqs(pk, 1)[0]
    pk.walker_signals(s, tissue=TIS)
    assert pk.decoded_bytes > 0
    n = counted["position_coeffs"]
    pk.release_decoded()
    assert pk.decoded_bytes == 0
    pk.walker_signals(s, tissue=TIS)
    assert counted["position_coeffs"] == 2 * n


def test_the_windows_of_a_pack_share_its_cache(pack, counted):
    pk = _fresh(pack)
    assert pk.n_windows == 2
    w0, w1 = pk.window(0), pk.window(1)
    assert w0.arrays.decoded is pk.arrays.decoded is w1.arrays.decoded
    a0, a1 = cx.read_position_coeffs(w0.arrays), cx.read_position_coeffs(w1.arrays)
    assert not np.array_equal(a0, a1) and counted["position_coeffs"] == 2                  # one entry per window, not one per pack
    assert cx.read_position_coeffs(pk.window(1).arrays) is a1
    assert pk.decoded_bytes == a0.nbytes + a1.nbytes


def test_a_derived_pack_holds_its_own_decode(pack):
    """A walker range, a prefix and a copy of the arrays are other tensors: none reads the parent's held array."""
    pk = _fresh(pack)
    whole = cx.read_position_coeffs(pk.window(0).arrays)
    sub = pk.walkers(10, 25)
    assert sub.arrays.decoded is not pk.arrays.decoded
    np.testing.assert_array_equal(cx.read_position_coeffs(sub.arrays), cx.read_position_coeffs(pk.arrays)[10:25] if pk.n_windows == 1
                                  else cx.read_position_coeffs(pk.window(0).arrays)[10:25])
    assert cx.read_position_coeffs(sub.arrays).shape[0] == 15 and whole.shape[0] == 300
    assert sub.decoded_bytes == cx.read_position_coeffs(sub.arrays).nbytes
    assert dict(pk.arrays).__class__ is dict                                        # a copy carries no cache


def test_arrays_pickle_and_copy_without_the_cache(pack):
    import copy
    import pickle
    pk = _fresh(pack)
    cx.read_position_coeffs(pk.window(0).arrays)
    for clone in (copy.deepcopy(pk.arrays), pickle.loads(pickle.dumps(pk.arrays))):
        assert isinstance(clone, PackArrays) and clone.decoded.nbytes == 0 and clone.decoded is not pk.arrays.decoded
        np.testing.assert_array_equal(clone["pos_x_ends"], pk.arrays["pos_x_ends"])


def test_the_assigned_arrays_of_a_pack_are_a_pack_arrays(pack):
    pk = _fresh(pack)
    pk.arrays = {**pk.arrays}
    assert isinstance(pk.arrays, PackArrays) and pk.decoded_bytes == 0


def test_the_field_modes_decode_once(counted):
    from dmipy_sim.fields.hollow_cylinder import CHANNEL_NAMES
    rng = np.random.default_rng(8)
    C = rng.normal(size=(200, 9, 7))
    arrays, pm = bank.susc_path_encode_modes(C, list(CHANNEL_NAMES[:7]), T=0.02, container=((None, 8),))
    held = PackArrays(arrays)
    first, names = bank.susc_path_coeffs(held, pm)
    for _ in range(3):
        again, names2 = bank.susc_path_coeffs(held, pm)
    assert counted["susc_path_coeffs"] == 1 and again is first and names2 == names and names2 is not names
    plain, _ = bank.susc_path_coeffs(dict(arrays), pm)
    np.testing.assert_array_equal(first, plain)
    assert not first.flags.writeable


def test_a_cache_with_no_room_holds_nothing_and_a_view_of_a_stored_tensor_is_never_frozen():
    stored = np.arange(12.0).reshape(3, 4)
    cache = DecodedCache(budget=0)
    out = cache.get("k", None, {"t": stored}, lambda: stored[:2].copy())
    assert cache.nbytes == 0 and not out.flags.writeable and stored.flags.writeable
    cache = DecodedCache()
    view = cache.get("k", None, {"t": stored}, lambda: stored[:2])
    assert cache.nbytes == 0 and stored.flags.writeable and np.shares_memory(view, stored)        # not decoded, not held, not frozen


def _fid(pk):
    return d.gre(4e-3, n_t=int(round(4e-3 / pk.dt)) + 1)


def test_a_sequence_without_gradient_reads_no_position(pack, counted):
    """A free-induction decay has no gradient phase: the occupancy and the contact carry its signal, and no band of the
    position is decoded for it -- the phase it has is the contraction of the bands with a zero weight, exactly."""
    pk = _fresh(pack)
    fid = _fid(pk)
    w, ew, E = pk.walker_signals(fid, tissue=TIS)
    assert counted["position_coeffs"] == 0 and counted["bridge_bands"] > 0 and counted["occupancy_runs"] > 0
    pk.replay(fid, tissue=TIS)
    assert counted["position_coeffs"] == 0
    phi = 0.0                                              # the contraction the skipped route stands for: bands against zero weights
    for win, _t0, _T in _fresh(pack)._windows():
        C = cx.read_position_coeffs(win.arrays).reshape(pk.n_walkers, -1)
        phi = phi + C @ np.zeros((C.shape[1], fid.G_eff.shape[0]))
    np.testing.assert_array_equal(np.exp(1j * phi), E)
    counted.clear()
    pk.walker_signals(_seqs(pk, 1)[0], tissue=TIS)                                       # a gradient reads its bands as before
    assert counted["position_coeffs"] + counted["position_prefix"] == pk.n_windows


def test_unpack_bits_inverts_pack_bits_at_every_depth_and_width():
    rng = np.random.default_rng(3)
    for bits in range(2, 17):
        for m in (1, 3, 7, 8, 9, 16, 21, 100):
            u = rng.integers(0, 1 << bits, size=(37, m))
            u[0], u[-1] = 0, (1 << bits) - 1
            np.testing.assert_array_equal(cx.unpack_bits(cx.pack_bits(u, bits), bits, m), u)


def test_the_position_decode_is_the_integer_values_times_their_scales(pack):
    """The dequantisation read back by hand from the stored tensors: ``q * scale`` in float64, rounded once to ``dtype``."""
    a = pack.window(1).arrays
    kind, suffix, present = cx._position_tensors(a)
    table = np.asarray(a["pos_band_scale"], np.float64)
    for dtype in (np.float64, np.float32):
        got = cx.read_position_coeffs(dict(a), dtype=dtype)
        for j, ax in enumerate(cx.POSITION_AXES):
            parts, k0 = [], 0
            for r, bits in cx.position_ranges(a, ax):
                t = a[cx.range_name(ax, r, bits)]
                m = int(t.shape[1]) if bits in cx.NATIVE_BITS else min(int(t.shape[1]) * 8 // bits, got.shape[1] - 2 - k0)
                parts.append(np.asarray(cx._load_range(t, bits, m), np.float64) * table[0, j, k0:k0 + m])
                k0 += m
            np.testing.assert_array_equal(got[:, 2:, j], np.concatenate(parts, axis=1).astype(dtype))
