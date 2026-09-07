"""The setup-time loops are vectorised and give the same answer as the loops they replace.

Packings, minimum gaps and grid tables are constructed once per geometry but used to cost
seconds to a minute in Python loops. Each vectorised routine is checked here against the plain
loop written out, on small inputs where the loop is cheap: same centres for a seed (the draw
order is unchanged, so old seeds still reproduce), the same gap, the same cell table.
"""
import numpy as np
import pytest

import dmipy_sim as d
from dmipy_sim.geometry._grid import bucket_by_bbox
from dmipy_sim.geometry.packing import periodic_min_gap


def _rsa_reference(radii, L, dim, seed, max_attempts=100_000):
    """The packer as it was: one candidate per trial, a Python loop over the placed objects."""
    rng = np.random.default_rng(int(seed))
    order = np.argsort(radii)[::-1]
    radii_s = radii[order]
    centers_s = np.zeros((len(radii), dim))
    for i, r_new in enumerate(radii_s):
        for _ in range(max_attempts):
            c_new = rng.uniform(-L / 2.0, L / 2.0, dim)
            ok = True
            for j in range(i):
                dq = c_new - centers_s[j]
                dq -= L * np.round(dq / L)
                if np.linalg.norm(dq) < r_new + radii_s[j]:
                    ok = False
                    break
            if ok:
                centers_s[i] = c_new
                break
        else:
            raise RuntimeError("reference RSA failed")
    out = np.empty_like(centers_s)
    out[order] = centers_s
    return out


def _gap_reference(centers, radii, L):
    N = len(radii)
    g = float("inf")
    for i in range(N):
        for j in range(i + 1, N):
            dq = centers[i] - centers[j]
            dq -= L * np.round(dq / L)
            g = min(g, np.linalg.norm(dq) - radii[i] - radii[j])
        g = min(g, L - 2.0 * radii[i])
    return float(g)


def _bucket_reference(lo, hi, dims, cap):
    from collections import defaultdict
    buckets = defaultdict(list)
    for t in range(len(lo)):
        for ix in range(lo[t, 0], hi[t, 0] + 1):
            for iy in range(lo[t, 1], hi[t, 1] + 1):
                for iz in range(lo[t, 2], hi[t, 2] + 1):
                    buckets[(ix * dims[1] + iy) * dims[2] + iz].append(t)
    occ = np.array([len(v) for v in buckets.values()]) if buckets else np.array([0])
    C = int(occ.max()) if cap is None else int(cap)
    cell = np.full((int(np.prod(dims)), C), -1, np.int32)
    overflow = 0
    for cid, lst in buckets.items():
        if len(lst) > C:
            overflow += len(lst) - C
            lst = lst[:C]
        cell[cid, :len(lst)] = lst
    return cell, C, int(occ.max()), overflow


@pytest.mark.parametrize("seed", [0, 3])
def test_packers_reproduce_the_loop_for_a_seed(seed):
    radii = np.array([1.0, 0.7, 1.3, 0.9, 1.1, 0.8, 1.2, 1.0]) * 1e-6
    c, L, _ = d.pack_cylinders(radii, target_vf=0.35, seed=seed)
    np.testing.assert_array_equal(c, _rsa_reference(radii, L, 2, seed))
    cs, Ls, _ = d.pack_spheres(radii, target_vf=0.15, seed=seed)
    np.testing.assert_array_equal(cs, _rsa_reference(radii, Ls, 3, seed))
    g = 0.7
    Lm = float(np.sqrt(np.pi * np.sum((radii / g) ** 2) / 0.5))
    _, _, cm = d.pack_myelinated_cylinders(radii, g, None, cell_size=Lm, seed=seed)
    np.testing.assert_array_equal(cm, _rsa_reference(radii / g, Lm, 2, seed))


def test_packers_still_refuse_an_impossible_packing():
    with pytest.raises(RuntimeError, match="RSA failed"):
        d.pack_cylinders([1e-6] * 40, target_vf=0.85, seed=0, max_attempts=200)


def test_min_gap_equals_the_pairwise_loop():
    rng = np.random.default_rng(1)
    radii = rng.uniform(0.5e-6, 1.5e-6, 12)
    c, L, _ = d.pack_cylinders(radii, target_vf=0.3, seed=1)
    assert periodic_min_gap(c, radii, L) == pytest.approx(_gap_reference(c, radii, L), rel=1e-12)
    assert d.PackedCylinders(radii, c, L).min_gap == pytest.approx(_gap_reference(c, radii, L), rel=1e-12)
    cs, Ls, _ = d.pack_spheres(radii, target_vf=0.12, seed=1)
    assert d.PackedSpheres(radii, cs, Ls).min_gap == pytest.approx(_gap_reference(cs, radii, Ls), rel=1e-12)
    assert periodic_min_gap(np.zeros((1, 2)), [1e-6], 5e-6) == pytest.approx(3e-6)   # one object: its image
    Lm = float(np.sqrt(np.pi * 12 * (1e-6 / 0.7) ** 2 / 0.5))
    _, _, cm = d.pack_myelinated_cylinders([1e-6] * 12, 0.7, None, cell_size=Lm, seed=2)
    pm = d.PackedMyelinatedCylinders([1e-6] * 12, 0.7, cm, Lm, N_max=16)
    assert pm.min_gap == pytest.approx(_gap_reference(cm, np.full(12, 1e-6 / 0.7), Lm), rel=1e-12)


@pytest.mark.parametrize("cap", [None, 3])
def test_grid_bucketing_equals_the_triple_loop(cap):
    rng = np.random.default_rng(4)
    dims = np.array([5, 4, 6])
    lo = rng.integers(0, 4, (60, 3))
    hi = np.minimum(lo + rng.integers(0, 3, (60, 3)), dims - 1)
    cell, C, occ, over = bucket_by_bbox(lo, hi, dims, cap)
    ref_cell, ref_C, ref_occ, ref_over = _bucket_reference(lo, hi, dims, cap)
    assert (C, occ, over) == (ref_C, ref_occ, ref_over)
    np.testing.assert_array_equal(cell, ref_cell)
    empty, C0, occ0, over0 = bucket_by_bbox(np.zeros((0, 3), int), np.zeros((0, 3), int), dims, None)
    assert empty.shape == (int(np.prod(dims)), 1) and (empty == -1).all() and (occ0, over0) == (0, 0)


def test_mesh_and_curved_pack_tables_are_unchanged():
    from dmipy_sim.geometry import mesh_shapes
    V, F = mesh_shapes.icosphere(2e-6, subdivisions=2)
    V = np.asarray(V, np.float32).astype(np.float64)     # exactly representable both ways, so the
    m = d.Mesh(V, F, feature_radius=1e-6)                  # float32 device copy IS the float64 input
    tri = np.asarray(m._TRIS, np.float64)
    cs = m.cell_size
    lo = np.clip(np.floor((tri.min(1) - m.grid_min) / cs).astype(int), 0, m.dims - 1)
    hi = np.clip(np.floor((tri.max(1) - m.grid_min) / cs).astype(int), 0, m.dims - 1)
    ref_cell, ref_C, ref_occ, ref_over = _bucket_reference(lo, hi, m.dims, None)
    np.testing.assert_array_equal(np.asarray(m._CELL), ref_cell)
    assert (m.C, m.max_occ, m.overflow) == (ref_C, ref_occ, ref_over)
    cl = np.stack([np.zeros(6), np.linspace(0, 8e-6, 6) ** 2 / 8e-6, np.linspace(0, 8e-6, 6)], axis=1)
    pk = d.PackedCurvedCylinders([cl, cl + np.array([4e-6, 0, 0])], [1e-6, 1.5e-6])
    assert np.asarray(pk._CELL).shape[1] == pk.C and (np.asarray(pk._CELL) >= -1).all()
