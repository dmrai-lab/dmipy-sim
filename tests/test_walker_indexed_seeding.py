"""Walker-indexed seeding: walker ``i`` of a seed is a function of the seed and ``i`` alone, so any partition of a
seed's walkers into walks (and of a walk into batches) walks the same walkers, bit for bit, and a pack built from the
pieces joined is the pack of the walk walked at once."""
import hashlib

import numpy as np
import pytest

import dmipy_sim as d
from dmipy_sim.coefficient_walk import join_walks
from dmipy_sim.engine.physics import seed_walkers
from dmipy_sim.geometry.base import PLACEMENT_BLOCK


def _sha(path):
    return hashlib.sha256(open(path, "rb").read()).hexdigest()


@pytest.mark.parametrize("geometry", [
    lambda: d.Cylinder(radius=3e-6, orientation=(0, 0, 1)),
    lambda: d.PackedSpheres([1.5e-6, 2e-6], [[0.0, 0.0, 0.0], [4e-6, 4e-6, 4e-6]], 10e-6),
    lambda: d.MyelinatedCylinder(inner_radius=2e-6, outer_radius=3e-6),
])
def test_a_walker_is_its_seed_and_index(geometry):
    """The starts, the step keys and the seeding labels of any range are the whole's rows, across placement blocks."""
    g = geometry()
    n = PLACEMENT_BLOCK + 700
    _, r0, keys, labels = seed_walkers(g, n, 11)
    for a, b in ((0, 1), (5, 300), (PLACEMENT_BLOCK - 3, PLACEMENT_BLOCK + 10), (1000, n)):
        _, r, k, lab = seed_walkers(g, b - a, 11, first_walker=a)
        np.testing.assert_array_equal(np.asarray(r), np.asarray(r0)[a:b])
        np.testing.assert_array_equal(np.asarray(k), np.asarray(keys)[a:b])
        if labels is not None:
            np.testing.assert_array_equal(np.asarray(lab), np.asarray(labels)[a:b])
    if labels is not None:                       # a stratified seeding's proportions hold in any range, not in order
        head = np.asarray(labels)[:300]
        assert len(np.unique(head)) == len(np.unique(np.asarray(labels)))


def test_a_walk_in_uneven_pieces_is_the_walk_bit_for_bit(tmp_path):
    """One walk of N walkers against the same N walked as uneven pieces, each in batches of another size: the
    coefficient arrays, the block record and the work record are identical, and so is the pack's sha256."""
    from dmipy_replay import envelope_of
    from dmipy_sim.replay.bank import build_replay_pack
    spec = d.Cylinder(radius=3e-6, orientation=(0, 0, 1)).spec
    kw = dict(T_max=0.01, envelope=envelope_of("connectome_2"), K=4, seed=7, diffusivity=2e-9)
    n = PLACEMENT_BLOCK + 404
    whole = d.walk(spec, n, walker_batch_size=2500, **kw)
    cuts = [0, 1000, 3100, PLACEMENT_BLOCK + 1, n]
    pieces = [d.walk(spec, b - a, first_walker=a, walker_batch_size=1700, **kw) for a, b in zip(cuts[:-1], cuts[1:])]
    joined = join_walks(pieces)
    for key in ("position_coeffs", "boundary_endpoint", "boundary_bands", "compartment", "qv_positions", "qv_boundary"):
        np.testing.assert_array_equal(getattr(whole, key), getattr(joined, key), err_msg=key)
    assert whole.block == joined.block and whole.work == joined.work
    shas = []
    for name, w in (("whole", whole), ("joined", joined)):
        out = str(tmp_path / f"{name}.rpk")
        build_replay_pack(w, id="partition", license="CC-BY-4.0", citation="test", out_path=out)
        shas.append(_sha(out))
    assert shas[0] == shas[1]


def test_a_batch_of_any_size_walks_the_same_walkers():
    """The JAX walk pads a batch to whole vector lanes and sums each walker's projections in its own order, so the
    batch size is not in the bytes: batches of 1100, 333 and 7 walk the 1100 walkers alike."""
    from dmipy_sim.engine.core import _walk_geometry
    g = d.PackedSpheres([1.5e-6, 2e-6], [[0.0, 0.0, 0.0], [4e-6, 4e-6, 4e-6]], 10e-6)
    ref = _walk_geometry(1100, 2e-9, g, 0.005, K=4, seed=3, walker_batch_size=1100)
    for b in (333, 7):
        w = _walk_geometry(1100, 2e-9, g, 0.005, K=4, seed=3, walker_batch_size=b)
        np.testing.assert_array_equal(ref.position_coeffs, w.position_coeffs)
        np.testing.assert_array_equal(ref.boundary_bands, w.boundary_bands)


def test_join_refuses_what_is_not_one_walk():
    from dmipy_sim.engine.core import _walk_geometry
    g = d.Cylinder(radius=3e-6, orientation=(0, 0, 1))
    a = _walk_geometry(64, 2e-9, g, 0.002, K=2, seed=1)
    b = _walk_geometry(64, 2e-9, g, 0.002, K=2, seed=1, first_walker=64)
    assert join_walks([a, b]).n_walkers == 128
    with pytest.raises(ValueError, match="contiguous"):
        join_walks([b, a])
    with pytest.raises(ValueError, match="seed"):
        join_walks([a, _walk_geometry(64, 2e-9, g, 0.002, K=2, seed=2, first_walker=64)])


def test_a_range_of_a_bundle_walk_is_refused_by_name():
    """The pool-by-pool walk deals its walkers to the pools from its own count: a range of it is not a range of one
    walk, and is refused rather than walked."""
    from dmipy_replay import envelope_of
    from dmipy_replay.spec.substrate import SpecError
    from dmipy_sim.spec.walk import _needs_bundle_walk
    import glob
    import os
    from dmipy_replay.spec.substrate import load_spec
    here = os.path.dirname(__file__)
    for path in sorted(glob.glob(os.path.join(here, "fixtures", "substrates", "*.sub.json"))):
        try:
            spec = load_spec(path)
        except Exception:
            continue
        if _needs_bundle_walk(spec):
            with pytest.raises(SpecError, match="first_walker"):
                d.walk(spec, 10, T_max=0.001, envelope=envelope_of("connectome_2"), first_walker=5)
            return
    pytest.skip("no multi-surface fixture spec loads here")
