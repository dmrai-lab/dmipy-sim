"""The g-ratio is a property of the meshes, not an argument.

Both surface volumes are already computed when a bundle is loaded, and for a tube V is proportional to r^2 L,
so g = r_in/r_out = sqrt(V_in/V_out) follows. Accepting it as a parameter instead let a caller's default be
written into pack provenance and substrate cards while the geometry said otherwise -- and nothing could
notice, because nothing compared them.

The shipped default (0.7) happens to be true for the Winther axons, whose outer surfaces are constructed
from the inner at a fixed g: measured across all 29, g = 0.695-0.704, std 0.002. It is meaningless for a
bundle, where g varies fibre to fibre -- 0.560-0.872, std 0.065, across 80 CACTUS strands.
"""
from __future__ import annotations

import numpy as np
import pytest

trimesh = pytest.importorskip("trimesh")


def _tube_pair(tmp_path, g=0.62, r_out=2.0, height=20.0):
    """Concentric closed tubes with a known g-ratio, written as PLY."""
    paths = []
    for radius in (g * r_out, r_out):
        m = trimesh.creation.cylinder(radius=radius, height=height, sections=64)
        p = tmp_path / f"r{radius:.3f}.ply"
        m.export(p)
        paths.append(str(p))
    return paths[0], paths[1]


def test_g_ratio_comes_from_the_meshes(tmp_path):
    """Loading with no g_ratio must recover the geometry's own value."""
    from dmipy_sim.io.winther import load_winther_bundle

    inner, outer = _tube_pair(tmp_path, g=0.62)
    b = load_winther_bundle(inner, outer, scale=1.0, pad=1.0)
    assert b.g_ratio == pytest.approx(0.62, abs=0.01), (
        f"g_ratio {b.g_ratio:.4f} does not match the 0.62 the meshes were built with")


def test_a_disagreeing_argument_warns_and_loses(tmp_path):
    """A supplied value is a cross-check, not the stored truth.

    Silently preferring the argument is the original defect: it is how 0.7 would end up in the provenance
    of a substrate whose meshes say 0.62.
    """
    from dmipy_sim.io.winther import load_winther_bundle

    inner, outer = _tube_pair(tmp_path, g=0.62)
    with pytest.warns(UserWarning, match="disagrees with the meshes"):
        b = load_winther_bundle(inner, outer, scale=1.0, pad=1.0, g_ratio=0.70)
    assert b.g_ratio == pytest.approx(0.62, abs=0.01), "the measured value must win"


def test_an_agreeing_argument_is_silent(tmp_path):
    """Passing the right number must not produce noise."""
    import warnings
    from dmipy_sim.io.winther import load_winther_bundle

    inner, outer = _tube_pair(tmp_path, g=0.62)
    with warnings.catch_warnings():
        warnings.simplefilter("error")       # any warning fails the test
        b = load_winther_bundle(inner, outer, scale=1.0, pad=1.0, g_ratio=0.62)
    assert b.g_ratio == pytest.approx(0.62, abs=0.01)


@pytest.mark.parametrize("g", [0.45, 0.62, 0.80])
def test_it_tracks_the_geometry_across_g(tmp_path, g):
    """Not a constant that happens to match one case."""
    from dmipy_sim.io.winther import load_winther_bundle

    d = tmp_path / f"g{g}"
    d.mkdir()
    inner, outer = _tube_pair(d, g=g)
    b = load_winther_bundle(inner, outer, scale=1.0, pad=1.0)
    assert b.g_ratio == pytest.approx(g, abs=0.015)
