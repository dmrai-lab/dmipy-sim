"""An ASCII PLY may declare a face-index type too small for its own vertex count.

MATLAB's plywrite emits `property list uchar ushort vertex_indices` even past 65535 vertices. The decimal
text carries the true indices, but a reader that honours the declared type masks them to 16 bits and
silently rewires every face above the limit -- the bounding box still looks right while the volume and
surface area become meaningless (measured on a real axon: volume 3.5x too small, area 43x too large).
"""
import numpy as np
import pytest

from dmipy_sim.mesh import load_ply

N_V = 70000          # > 65536, so uint16 indices cannot address the whole mesh


def _write_ascii_ply(path, n_v=N_V, declared="ushort"):
    """A closed strip of triangles whose last faces reference vertices above 65535."""
    rng = np.random.default_rng(0)
    V = np.column_stack([np.arange(n_v) * 1e-3, rng.normal(scale=1e-3, size=n_v),
                         rng.normal(scale=1e-3, size=n_v)])
    F = np.column_stack([np.arange(0, n_v - 2), np.arange(1, n_v - 1), np.arange(2, n_v)])
    with open(path, "w") as f:
        f.write("ply\nformat ascii 1.0\ncomment created by MATLAB plywrite\n")
        f.write(f"element vertex {n_v}\nproperty float x\nproperty float y\nproperty float z\n")
        f.write(f"element face {len(F)}\nproperty list uchar {declared} vertex_indices\nend_header\n")
        for v in V:
            f.write(f"{v[0]:.6f} {v[1]:.6f} {v[2]:.6f}\n")
        for t in F:
            f.write(f"3 {t[0]} {t[1]} {t[2]}\n")
    return V, F


def test_undersized_declared_index_type_is_repaired(tmp_path):
    p = tmp_path / "matlab_style.ply"
    V, F = _write_ascii_ply(str(p))
    with pytest.warns(UserWarning, match="truncated to 16 bits"):
        Vl, Fl = load_ply(str(p))
    assert len(Vl) == len(V)
    assert Fl.max() == F.max() > 65535, "the upper vertices must be reachable after the repair"
    np.testing.assert_array_equal(Fl, F)


def test_correctly_declared_file_is_untouched(tmp_path):
    """A file whose declared type is adequate must not go down the re-parse path."""
    p = tmp_path / "uint_style.ply"
    V, F = _write_ascii_ply(str(p), declared="uint")
    import warnings
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        Vl, Fl = load_ply(str(p))
    assert not [x for x in w if "truncated" in str(x.message)]
    np.testing.assert_array_equal(Fl, F)


def test_scale_is_applied_after_repair(tmp_path):
    """Compare the two loads against each other: the file is written at 6 decimals, so comparing to the
    in-memory array would be testing the text precision rather than the scaling."""
    p = tmp_path / "scaled.ply"
    _write_ascii_ply(str(p))
    with pytest.warns(UserWarning):
        V1, _ = load_ply(str(p))
    with pytest.warns(UserWarning):
        V2, _ = load_ply(str(p), scale=1e-6)
    np.testing.assert_allclose(V2, V1 * 1e-6, rtol=1e-12, atol=0)
