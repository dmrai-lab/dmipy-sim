"""Memory bound for :func:`dmipy_sim.phantom.bore.encoding_classes` (dmipy-sim#563).

Before this, ``encoding_classes`` composed and kept every distinct class's ``ScannerSequence`` in one list --
``G`` and ``imposed_gradient``, both ``(n_meas, n_t, 3)`` float32, about 12 MB a class at a clinical protocol
(485 measurements, ``n_t`` 1000). A magnet's own gradient (``g0``) varies across a head, so the class count
follows the effect: a handful for a uniform scanner, tens of thousands for a permanent magnet's own gradient
over a whole head -- hundreds of GB held at once.

The fix: a class is identified by its member's exact tensor / background / position values (a 3x3 and two
3-vectors), which is what ``encoding_classes`` now stores (:class:`~dmipy_sim.phantom.bore.EncodingClasses`);
the full ``ScannerSequence`` is built from them only when a consumer indexes or iterates it, and nothing
caches it afterwards. Two things are checked here: that nothing a reader sees has changed (the pre-#563
algorithm is reproduced below, unchanged, as the oracle this must agree with to the bit), and that iterating a
synthetic machine's more-than-``10**4`` classes keeps the process' resident memory bounded -- read from
``/proc/self/status``'s ``VmRSS``, never ``statm`` pages (a 64 kB-page host under-reads those sixteenfold).
"""
import numpy as np
import pytest

from dmipy_sim.acquisition.scanner_sequence import ScannerSequence
from dmipy_sim.acquisition.scanners import ScannerLimits
from dmipy_sim.phantom.grid import Grid
from dmipy_sim.phantom.bore import (_b0_axis, _bore, _model, background_gradient_map, encoding_classes,
                                    gradient_tensor_map)
from dmipy_sim.replay.phantom import quantise


def _rss_bytes():
    """The process's resident bytes, from ``VmRSS`` (never ``statm`` pages, which this family of hosts pages
    at 64 kB and so under-reads sixteenfold)."""
    with open("/proc/self/status") as fh:
        for line in fh:
            if line.startswith("VmRSS:"):
                return int(line.split()[1]) * 1024
    pytest.skip("this host's /proc/self/status states no VmRSS")


def _reference_encoding_classes(scanner, grid, sequence, voxel_index, *, tolerance=1e-3, to_scanner=None,
                                nonlinearity=True, background=True, concomitant=True):
    """The algorithm ``bore.encoding_classes`` ran before dmipy-sim#563: every class composed and kept in one
    list, picking a bin's member the same way (the first, in ``voxel_index``'s own order, to fall in it). Kept
    here untouched as the oracle a lazy class must match to the bit; the package itself no longer builds this
    list."""
    scanner = _model(scanner)
    B0 = scanner.field_T
    has_L = bool(nonlinearity and scanner.has_gradient_nonlinearity)
    has_g0 = bool(background and scanner.has_field_law)
    has_c = bool(concomitant and B0 is not None)
    if not (has_L or has_g0 or has_c):
        return None
    idx = np.asarray(voxel_index)
    pos = grid.positions_m(idx)
    R, _into_bore = _bore(grid, to_scanner)
    d_grid = np.asarray(pos, dtype=np.float64).reshape(-1, 3) - np.asarray(grid.isocenter_m, dtype=np.float64)
    n = d_grid.shape[0]
    G_max = max(float(np.abs(np.asarray(sequence.G)).max()), 1e-30)
    tol = None if tolerance is None else float(tolerance)
    Ls = gradient_tensor_map(scanner, grid, to_scanner=R)(pos) if has_L else None
    g0 = background_gradient_map(scanner, grid, to_scanner=R)(pos) if has_g0 else None
    parts = []
    if has_L:
        parts.append(quantise(Ls.reshape(n, 9), None if tol is None else 0.5 * tol))
    if has_g0:
        parts.append(quantise(g0, None if tol is None else 0.5 * tol * G_max))
    if has_c:
        parts.append(quantise(d_grid, None if tol is None else tol * float(B0) / G_max))
    key = np.concatenate(parts, axis=1)
    _uniq, inverse = np.unique(key, axis=0, return_inverse=True)
    inverse = np.asarray(inverse).reshape(-1)
    b0_axis = _b0_axis(scanner, R)
    played = []
    for c in range(int(inverse.max()) + 1):
        k = int(np.flatnonzero(inverse == c)[0])                  # a member's exact values stand for the bin
        seq = sequence
        if has_L:
            seq = seq.with_gradient_nonlinearity(Ls[k])
        if has_g0:
            seq = seq.with_background_gradient(g0[k])
        if has_c:
            seq = seq.with_concomitant(d_grid[k], float(B0), b0_axis=b0_axis)
        played.append(seq)
    return inverse, played


def _small_fixture():
    """A fixture small enough for the CPU: the Swoop (all three terms live -- tensor, background, concomitant)
    over a coarse grid inside its 8 cm validity radius, with a random multi-axis waveform."""
    sw = ScannerLimits.of("swoop")
    shape = (5, 5, 3)
    vox = 0.02
    grid = Grid(shape=shape, voxel_size_m=(vox,) * 3,
               origin_m=tuple(-0.5 * (n - 1) * vox for n in shape), isocenter_m=(0.0, 0.0, 0.0))
    n_meas, n_t = 7, 40
    rng = np.random.default_rng(0)
    G = np.zeros((n_meas, n_t, 3), np.float32)
    G[:, 4:36, :] = (rng.normal(size=(n_meas, 32, 3)) * 0.02).astype(np.float32)
    seq = ScannerSequence(G=G, dt=2e-4)
    return sw, grid, seq


@pytest.mark.parametrize("tolerance", [None, 1e-2, 0.2])
def test_lazy_classes_match_the_pre_563_eager_list_bit_for_bit(tolerance):
    """What every reader of ``encoding_classes`` sees -- the class-of-voxel index, and each class's ``G``,
    ``imposed_gradient`` and ``b()`` -- is unchanged: :class:`EncodingClasses` against the pre-#563 eager list,
    binned (``tolerance`` a number) and exact (``None``)."""
    sw, grid, seq = _small_fixture()
    idx = grid.every_voxel
    inv_new, played_new = encoding_classes(sw, grid, seq, idx, tolerance=tolerance)
    inv_ref, played_ref = _reference_encoding_classes(sw, grid, seq, idx, tolerance=tolerance)
    np.testing.assert_array_equal(inv_new, inv_ref)
    assert len(played_new) == len(played_ref) >= 1
    for c in range(len(played_ref)):
        a, b = played_new[c], played_ref[c]
        np.testing.assert_array_equal(a.G, b.G)
        np.testing.assert_array_equal(np.asarray(a.imposed_gradient), np.asarray(b.imposed_gradient))
        np.testing.assert_array_equal(a.b(), b.b())
    # negative indexing, slicing and plain iteration all agree with the list too
    np.testing.assert_array_equal(played_new[-1].G, played_ref[-1].G)
    sl_new, sl_ref = played_new[0:2], played_ref[0:2]
    assert len(sl_new) == len(sl_ref)
    for s, r in zip(sl_new, sl_ref):
        np.testing.assert_array_equal(s.G, r.G)
    for s, r in zip(played_new, played_ref):
        np.testing.assert_array_equal(s.G, r.G)


def test_nothing_is_cached_between_accesses():
    """Reading the same class twice gives two independently-built sequences, not one held and handed back --
    what "composed on demand, nothing cached" has to mean operationally."""
    sw, grid, seq = _small_fixture()
    idx = grid.every_voxel
    _inv, played = encoding_classes(sw, grid, seq, idx, tolerance=None)
    a = played[0]
    b = played[0]
    assert a is not b
    np.testing.assert_array_equal(a.G, b.G)


def test_more_than_1e4_classes_streams_under_a_stated_rss_bound():
    """The issue's own scenario: a synthetic machine whose ``g0`` (the Swoop's real catalogued field law) bins
    into more than ``10**4`` classes over a grid. Iterating every one of them (as
    ``ReplayPack.pose_series`` does, one composed class in memory at a time) must not grow the process' RSS
    anywhere near what holding them all would cost -- about a GB here, by the issue's own formula
    (``n_classes * n_meas * n_t * 3 * 4 * 2``, ``G`` and ``imposed_gradient`` both float32) -- which the bound
    below is well under a third of."""
    sw = ScannerLimits.of("swoop")
    shape = (22, 22, 22)
    vox = 0.0032                                       # keeps the grid inside the Swoop's 8 cm validity radius
    grid = Grid(shape=shape, voxel_size_m=(vox,) * 3,
               origin_m=tuple(-0.5 * (n - 1) * vox for n in shape), isocenter_m=(0.0, 0.0, 0.0))
    idx = grid.every_voxel
    n_meas, n_t = 61, 64
    G = np.zeros((n_meas, n_t, 3), np.float32)
    G[:, 6:58, 0] = 0.02
    seq = ScannerSequence(G=G, dt=2e-4)

    # measured BEFORE encoding_classes is even called: the pre-#563 algorithm pays its whole cost composing
    # the list inside this call, so the baseline has to be taken ahead of it -- an eager list leaves this
    # baseline already a GB high, and the loop below would then see no further growth and miss the regression.
    rss0 = _rss_bytes()
    _inv, played = encoding_classes(sw, grid, seq, idx, tolerance=None)
    n_classes = len(played)
    assert n_classes > 10_000, n_classes                # the acceptance test's own threshold
    eager_bytes = n_classes * n_meas * n_t * 3 * 4 * 2  # the issue's own estimate, G + imposed_gradient
    assert eager_bytes > 5e8, eager_bytes               # half a GB were every class held at once

    peak = max(rss0, _rss_bytes())
    for c in range(n_classes):
        s = played[c]
        _ = float(s.G[0, 0, 0])                         # touch it, as a consumer reading the sequence would
        del s
        if c % 500 == 0:
            peak = max(peak, _rss_bytes())
    peak = max(peak, _rss_bytes())
    growth = peak - rss0
    bound = 300 * 1024 * 1024                            # well under the ~1 GB holding them all would cost
    assert growth < bound, (
        f"streaming {n_classes} classes grew RSS by {growth / 1e6:.1f} MB, past the {bound / 1e6:.0f} MB bound "
        f"this test set; holding them all at once would have cost about {eager_bytes / 1e9:.2f} GB")
