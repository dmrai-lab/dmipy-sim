"""The public surface, and the geometry protocol every exported geometry satisfies.

Two things are pinned here. Every name the package exports resolves, and the submodules
downstream packages import exist. And every geometry declares the protocol the engine reads --
``length_scales``, the capability flags, the wall/bulk attributes, ``classify_position`` and its
two helpers, ``interact`` -- so a geometry the engine cannot size or label fails here, by name,
rather than inside a walk.
"""
import importlib
import re
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest

import dmipy_sim
from dmipy_sim.geometry import Geometry, LengthScales
from dmipy_sim.geometry._boundary import WallHit

_SRC = Path(dmipy_sim.__file__).parent

# Submodules imported by dmipy-fit / dmipy-design, so they must keep resolving.
_DOWNSTREAM_MODULES = [
    "dmipy_sim.replay", "dmipy_sim.constants", "dmipy_sim.waveforms", "dmipy_sim.sh_convolution",
    "dmipy_sim.gaunt", "dmipy_sim.geometries", "dmipy_sim.compression", "dmipy_sim.rf",
    "dmipy_sim.bank", "dmipy_sim.pulse_sequence", "dmipy_sim.sequences", "dmipy_sim.sequences.pulseq",
    "dmipy_sim.substrate", "dmipy_sim.substrate.biophysical_constants", "dmipy_sim.substrate.substrate",
    "dmipy_sim.geometry", "dmipy_sim.geometry.mesh", "dmipy_sim.geometry.curved_tube",
    "dmipy_sim.phantom", "dmipy_sim.trajectories", "dmipy_sim.physics", "dmipy_sim.bloch",
    "dmipy_sim.mt", "dmipy_sim.mt_walk", "dmipy_sim.susceptibility", "dmipy_sim.susceptibility_field",
]


def test_every_public_name_resolves():
    missing = [n for n in dmipy_sim.__all__ if not hasattr(dmipy_sim, n)]
    assert not missing, f"names in dmipy_sim.__all__ that do not resolve: {missing}"
    import dmipy_sim.geometry as g
    missing = [n for n in g.__all__ if not hasattr(g, n)]
    assert not missing, f"names in dmipy_sim.geometry.__all__ that do not resolve: {missing}"


@pytest.mark.parametrize("name", _DOWNSTREAM_MODULES)
def test_submodule_imports(name):
    importlib.import_module(name)


# ── every exported geometry, built small ─────────────────────────────────────────────────
# Built inside the test, never at collection: a geometry holds device buffers.
def _all_geometries():
    from dmipy_sim import mesh_shapes
    from dmipy_sim.geometry import (FreeDiffusion, Box1D, Sphere, Cylinder, Ellipsoid,
                                    PermeableSlab1D, PermeableShell, PackedCylinders, PackedSpheres,
                                    MyelinatedCylinder, PackedMyelinatedCylinders, CurvedTube,
                                    MultiShellCurvedTube, PackedCurvedTubes, Mesh,
                                    pack_cylinders, pack_spheres)
    R = 1e-6
    c2, L2, _ = pack_cylinders([R] * 4, target_vf=0.3, seed=0)
    c3, L3, _ = pack_spheres([R] * 4, target_vf=0.1, seed=0)
    V, F = mesh_shapes.icosphere(5e-6, subdivisions=2)
    cl = np.stack([np.zeros(8), np.zeros(8), np.linspace(0, 2e-5, 8)], axis=1)
    return {
        "FreeDiffusion": FreeDiffusion(),
        "Box1D": Box1D(length=4e-6),
        "Sphere": Sphere(radius=5e-6),
        "Cylinder": Cylinder(radius=5e-6, orientation=(0, 0, 1)),
        "Ellipsoid": Ellipsoid(semiaxes=(5e-6, 3e-6, 4e-6)),
        "PermeableSlab1D": PermeableSlab1D(length=4e-6, permeability=1e-5),
        "PermeableShell": PermeableShell(r_inner=3e-6, r_outer=5e-6, permeability=1e-5),
        "PackedCylinders": PackedCylinders([R] * 4, c2, L2),
        "PackedSpheres": PackedSpheres([R] * 4, c3, L3),
        "MyelinatedCylinder": MyelinatedCylinder(3e-6, 5e-6, (0, 0, 1), 2e-9, 2e-9),
        "PackedMyelinatedCylinders": PackedMyelinatedCylinders([R] * 4, 0.7, c2, L2, N_max=8),
        "CurvedTube": CurvedTube(cl, radius=2e-6),
        "MultiShellCurvedTube": MultiShellCurvedTube(cl, r_in=2e-6, r_out=3e-6),
        "PackedCurvedTubes": PackedCurvedTubes([cl], [2e-6]),
        "Mesh": Mesh(V, F, feature_radius=1e-6),
    }


_NAMES = ["FreeDiffusion", "Box1D", "Sphere", "Cylinder", "Ellipsoid", "PermeableSlab1D",
          "PermeableShell", "PackedCylinders", "PackedSpheres", "MyelinatedCylinder",
          "PackedMyelinatedCylinders", "CurvedTube", "MultiShellCurvedTube", "PackedCurvedTubes",
          "Mesh"]
# stepped by their own fused kernel; `reflect` raises by design
_NO_REFLECT = {"MyelinatedCylinder", "PackedMyelinatedCylinders"}


def _num_or_none(x):
    return x is None or isinstance(x, float)


@pytest.mark.parametrize("name", _NAMES)
def test_geometry_declares_the_protocol(name):
    g = _all_geometries()[name]
    assert isinstance(g, Geometry)

    ls = g.length_scales
    assert isinstance(ls, LengthScales)
    for f in ("min_feature", "surface_pore", "lookup_cell", "min_gap"):
        assert _num_or_none(getattr(ls, f)), f"{name}.length_scales.{f} = {getattr(ls, f)!r}"
    assert isinstance(ls.is_mesh_feature, bool)
    if name != "FreeDiffusion":
        assert ls.min_feature is not None and ls.min_feature > 0, f"{name} has walls but no scale"
    assert ls.is_mesh_feature == (name == "Mesh")
    assert (ls.lookup_cell is not None) == (name in ("Mesh", "PackedCurvedTubes"))

    for flag in ("supports_permeability", "carries_side", "_is_myelinated", "_is_packed_myelinated",
                 "classify_returns_object_id", "radius_is_mesh_feature"):
        assert isinstance(getattr(g, flag), bool), f"{name}.{flag}"
    for attr in ("permeability", "surface_relaxivity_t2", "surface_substep_frac", "_orient_R",
                 "_D_comp_jax", "_inv_T2_comp_jax", "_inv_T1_comp_jax", "_D_comp_max",
                 "_T2_comp", "_T1_comp"):
        getattr(g, attr)                      # declared on every geometry, None when unset

    key = jax.random.PRNGKey(0)
    r0 = g.init_positions(16, key)
    assert r0.shape == (16, 3) and r0.dtype == jnp.float32

    lab = g.classify_position(r0[0])
    assert jnp.shape(lab) == () and jnp.asarray(lab).dtype == jnp.int32
    assert g.classify_positions_exact(r0).shape == (16,)
    assert jnp.shape(g.classify_position_carry(r0[0], jnp.int32(0))) == ()

    step = jnp.asarray([1e-8, 0.0, 0.0], jnp.float32)
    if name in _NO_REFLECT:
        with pytest.raises(NotImplementedError):
            g.interact(r0[0], step)
    else:
        hit = g.interact(r0[0], step)
        assert isinstance(hit, WallHit) and hit.r.shape == (3,)


def test_length_scales_match_the_geometry_definition():
    """Each scale is the quantity its class defines it to be."""
    G = _all_geometries()
    ls = lambda n: G[n].length_scales
    assert ls("Box1D").min_feature == 4e-6
    assert ls("Sphere").min_feature == 5e-6
    assert ls("Cylinder").min_feature == 5e-6
    assert ls("Ellipsoid").min_feature == 3e-6                       # smallest semi-axis
    assert ls("PermeableSlab1D").min_feature == 2e-6                 # one compartment's width
    assert ls("PermeableShell").min_feature == 3e-6                  # the membrane radius
    assert ls("PackedCylinders").min_feature == 1e-6
    assert ls("PackedCylinders").min_gap == G["PackedCylinders"].min_gap
    assert ls("MyelinatedCylinder").min_feature == 3e-6              # the lumen
    pm = G["PackedMyelinatedCylinders"]
    assert ls("PackedMyelinatedCylinders").min_feature == 1e-6
    outer = pm._outer_radii_np[:pm.N_actual]
    pore = (pm._L_float ** 2 - np.sum(np.pi * outer ** 2)) / np.sum(2 * np.pi * outer)
    assert ls("PackedMyelinatedCylinders").surface_pore == pytest.approx(pore)
    assert ls("MultiShellCurvedTube").min_feature == 2e-6            # the inner wall
    m = G["Mesh"]
    assert ls("Mesh") == LengthScales(min_feature=1e-6, lookup_cell=m.cell_size, is_mesh_feature=True)
    pk = G["PackedCurvedTubes"]
    assert ls("PackedCurvedTubes") == LengthScales(min_feature=2e-6, lookup_cell=pk.cell_size)


def test_duck_typed_objects_still_read_through_the_legacy_attributes():
    """An object that is not a Geometry is sized from its legacy attributes, in one place."""
    from dmipy_sim.physics import length_scales_of

    class Slab:
        length = 3e-6

    class Indexed:
        radius = 2e-6
        cell_size = 5e-7
        radius_is_mesh_feature = True

    class Bare:
        pass

    assert length_scales_of(Slab()) == LengthScales(min_feature=3e-6)
    assert length_scales_of(Indexed()) == LengthScales(min_feature=2e-6, lookup_cell=5e-7,
                                                       is_mesh_feature=True)
    assert length_scales_of(Bare()) == LengthScales()


# ── the engine reads the protocol, not attribute probes ──────────────────────────────────
_ENGINE_MODULES = ["core.py", "physics.py", "bloch.py", "mt_walk.py", "mt.py", "pedagogy.py",
                   "mesh_bundle.py"]
# Probes that remain, and why. Every other property is a declared attribute.
_ALLOWED_PROBES = {
    # physics.length_scales_of: the ONE reader of legacy attributes for non-Geometry objects
    "length_scales", "radius", "sphere_radius", "length", "_radii_np", "_inner_radii_np",
    "cell_size", "radius_is_mesh_feature",
    # sub-step helpers accept duck-typed objects (tests pass bare classes)
    "permeability", "surface_substep_frac",
    # optional methods: not every geometry records a boundary local time / binding / membrane
    "reflect_with_log_weight", "reflect_with_binding", "permeate",
    # core.simulate_trajectories compartment-id derivation (issue #94, A7)
    "_R",
    # BoxedMesh wraps a Mesh and reads its box mode
    "box_reflect",
}


def test_engine_probes_no_undeclared_geometry_attribute():
    pat = re.compile(r"(?:getattr|hasattr)\((?:geometry|geom|self\.mesh|mesh),\s*['\"]([A-Za-z_]+)['\"]")
    found = {}
    for mod in _ENGINE_MODULES:
        for m in pat.finditer((_SRC / mod).read_text()):
            found.setdefault(m.group(1), set()).add(mod)
    undeclared = {k: sorted(v) for k, v in found.items() if k not in _ALLOWED_PROBES}
    assert not undeclared, (
        f"the engine probes geometry attributes by name: {undeclared}. Declare them on "
        f"dmipy_sim.geometry.base.Geometry (or in LengthScales) and read them directly.")
