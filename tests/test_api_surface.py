"""The public surface, and the geometry protocol every exported geometry satisfies.

Three things are pinned here. Every name the package exports resolves, and the submodules
downstream packages import exist. Every geometry declares the protocol the engine reads --
``length_scales``, the capability flags, the wall/bulk attributes, ``classify_position`` and its
two helpers, ``interact`` -- so a geometry the engine cannot size or label fails here, by name,
rather than inside a walk. And the acquisition surface is an inventory (#173): the containers that
carry what the scanner does, the RF dialects they speak, the consumers that take the RF apart from
the gradient, the scanner catalogues -- each a declared set, so the spread that #173 collapses into
one ``ScannerSequence`` can only shrink.
"""
import dmipy_replay.replay.field as _r_field
import dmipy_replay.replay.time_segments as _r_time_segments
import importlib
import re
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
from dmipy_replay.acquisition.timing import SequenceTiming
import pytest

import dmipy_sim
from dmipy_sim.geometry import Geometry, LengthScales
from dmipy_sim.geometry._boundary import WallHit
from dmipy_sim import Compartments, Pool

_SRC = Path(dmipy_sim.__file__).parent
import dmipy_replay  # noqa: E402
_REPLAY_SRC = Path(dmipy_replay.__file__).parent      # the reader dmipy-sim imports: the same rules hold over its code

# Every module path of the package; downstream (dmipy-fit / dmipy-design) imports a subset of these.
_DOWNSTREAM_MODULES = [
    "dmipy_sim.replay.replay", "dmipy_replay.constants", "dmipy_replay.acquisition.waveforms",
    "dmipy_sim.replay.so3", "dmipy_sim.replay.fod", "dmipy_sim.replay.compression", "dmipy_replay.acquisition.rf",
    "dmipy_sim.replay.bank", "dmipy_sim.sequences", "dmipy_sim.sequences.pulseq",
    "dmipy_sim.substrate", "dmipy_sim.substrate.biophysical_constants", "dmipy_sim.substrate.substrate",
    "dmipy_sim.geometry", "dmipy_sim.geometry.mesh", "dmipy_sim.geometry.curved_cylinder",
    "dmipy_sim.replay.phantom", "dmipy_sim.replay.trajectories", "dmipy_sim.phantom",
    "dmipy_sim.fields.susceptibility", "dmipy_sim.fields.susceptibility_field",
    # the engine package
    "dmipy_sim.engine", "dmipy_sim.engine.core", "dmipy_sim.engine.physics", "dmipy_sim.engine.bloch",
    "dmipy_sim.engine.pulse_sequence", "dmipy_sim.engine.mt", "dmipy_sim.engine.mt_walk",
    "dmipy_sim.engine.gpu", "dmipy_sim.engine._gpu_config",
    # the replay package; dmipy_sim.replay is the package and keeps replay.py's surface
    "dmipy_replay.replay._replay_kernel", "dmipy_sim.spec", "dmipy_sim.spec.build", "dmipy_sim.spec.walk",
    "dmipy_sim.spec.producers", "dmipy_sim.io.caterpillar", "dmipy_sim.io.strands", "dmipy_sim.geometry.sphere_union",
    "dmipy_sim.io.label_volume", "dmipy_sim.geometry.label_volume",
    # acquisition / fields / viz; dmipy_sim.viz is the package re-exporting viz.py
    "dmipy_sim.acquisition.noise", "dmipy_replay.acquisition.scanners", "dmipy_replay.acquisition.scanner_constants",
    "dmipy_replay.acquisition.timing",
    "dmipy_sim.viz", "dmipy_sim.viz.viz", "dmipy_sim.viz.pedagogy",
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
def _slab_labels():
    """A 10 um slab pore between two grain slabs, the smallest label volume with a wall."""
    lab = np.ones((40, 8, 8), np.uint8)
    lab[10:30] = 0
    return lab


def _all_geometries():
    from dmipy_sim.geometry import mesh_shapes
    from dmipy_sim.geometry import (FreeDiffusion, Box1D, Sphere, Cylinder, Ellipsoid,
                                    PermeableSlab1D, PermeableShell, PackedCylinders, PackedSpheres,
                                    MyelinatedCylinder, PackedMyelinatedCylinders, CurvedCylinder,
                                    CurvedMyelinatedCylinder, PackedCurvedCylinders, Mesh, LabelVolume,
                                    pack_cylinders, pack_spheres, pack_myelinated_cylinders)
    R = 1e-6
    c2, L2, _ = pack_cylinders([R] * 4, target_vf=0.3, seed=0)
    c3, L3, _ = pack_spheres([R] * 4, target_vf=0.1, seed=0)
    L4 = float(np.sqrt(np.pi * 4 * (R / 0.7) ** 2 / 0.5))
    _, _, c4 = pack_myelinated_cylinders([R] * 4, 0.7, None, cell_size=L4, seed=0)
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
        "MyelinatedCylinder": MyelinatedCylinder(3e-6, 5e-6, (0, 0, 1), compartments=Compartments(intra=Pool(D=2e-9), extra=Pool(D=2e-9))),
        "PackedMyelinatedCylinders": PackedMyelinatedCylinders([R] * 4, 0.7, c4, L4, N_max=8),
        "CurvedCylinder": CurvedCylinder(cl, radius=2e-6),
        "CurvedMyelinatedCylinder": CurvedMyelinatedCylinder(cl, r_in=2e-6, r_out=3e-6),
        "PackedCurvedCylinders": PackedCurvedCylinders([cl], [2e-6]),
        "Mesh": Mesh(V, F, feature_radius=1e-6),
        "LabelVolume": LabelVolume(_slab_labels(), 0.5e-6),
    }


_NAMES = ["FreeDiffusion", "Box1D", "Sphere", "Cylinder", "Ellipsoid", "PermeableSlab1D",
          "PermeableShell", "PackedCylinders", "PackedSpheres", "MyelinatedCylinder",
          "PackedMyelinatedCylinders", "CurvedCylinder", "CurvedMyelinatedCylinder", "PackedCurvedCylinders",
          "Mesh", "LabelVolume"]
# stepped by their own fused kernel; `reflect` raises by design
_NO_REFLECT = {"MyelinatedCylinder", "PackedMyelinatedCylinders"}


def _num_or_none(x):
    return x is None or isinstance(x, float)


def test_one_wall_per_geometry_and_the_wrappers_live_on_the_base():
    """A geometry implements `_wall` and nothing else of the wall contract (#633): `reflect`,
    `reflect_with_log_weight`, `permeate` and `interact` are the base class's wrappers at argument values, so no
    geometry can drift into a reduced copy of its own physics again (#88). `has_walls` says whether `_wall` is a
    wall or a free step, and the two myelinated substrates, stepped by their own kernels, refuse `_wall` by design."""
    import inspect
    for name, g in _all_geometries().items():
        cls = type(g)
        for wrapper in ("reflect", "reflect_with_log_weight", "permeate", "interact"):
            owner = next(k for k in cls.__mro__ if wrapper in k.__dict__)
            assert owner is Geometry, f"{name} defines {wrapper} itself; implement `_wall` and leave the wrapper to Geometry"
        assert "_wall" in {k for c in cls.__mro__ for k in c.__dict__ if c is not Geometry}, f"{name} has no `_wall`"
        assert isinstance(g.has_walls, bool)
        assert g.has_walls == (name not in _NO_REFLECT and name != "FreeDiffusion"), name
        sig = inspect.signature(cls._wall)
        assert list(sig.parameters)[:6] == ["self", "r", "step", "kappa_over_D", "rho_over_D"] + [list(sig.parameters)[5]], f"{name}._wall{sig}"
        assert g.carries_side == ("side" in sig.parameters), name


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
    assert (ls.lookup_cell is not None) == (name in ("Mesh", "PackedCurvedCylinders"))

    for flag in ("supports_permeability", "carries_side", "has_walls", "_is_myelinated", "_is_packed_myelinated",
                 "classify_returns_object_id", "radius_is_mesh_feature"):
        assert isinstance(getattr(g, flag), bool), f"{name}.{flag}"
    for attr in ("permeability", "surface_relaxivity_t2", "surface_substep_frac", "reflection_step_fraction", "_orient_R",
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
    assert ls("CurvedMyelinatedCylinder").min_feature == 2e-6            # the inner wall
    m = G["Mesh"]
    assert ls("Mesh") == LengthScales(min_feature=1e-6, lookup_cell=m.cell_size, is_mesh_feature=True)
    pk = G["PackedCurvedCylinders"]
    assert ls("PackedCurvedCylinders") == LengthScales(min_feature=2e-6, lookup_cell=pk.cell_size)
    # a label volume: the voxel, and the walking pool's measured V/S (a 10 um slab pore -> 5 um)
    # a label volume: the voxel, and no separate pore -- V/S is a mean, and the surface rule divides
    # the worst case a segmentation can hold, which is one voxel
    assert ls("LabelVolume") == LengthScales(min_feature=0.5e-6, pore_rate=0.0)   # the voxel is a resolution, not a pore


def test_an_object_without_length_scales_is_refused():
    """The engine sizes a geometry from what it declares as ``length_scales``; an object without them is refused,
    not guessed at from its attributes."""
    from dmipy_sim.engine.physics import length_scales_of

    class Declared:
        length_scales = LengthScales(min_feature=2e-6, lookup_cell=5e-7, is_mesh_feature=True)

    class Bare:
        radius = 2e-6

    assert length_scales_of(Declared()) == LengthScales(min_feature=2e-6, lookup_cell=5e-7, is_mesh_feature=True)
    with pytest.raises(TypeError, match="declares no length_scales"):
        length_scales_of(Bare())


# ── the engine reads the protocol, not attribute probes ──────────────────────────────────
_ENGINE_MODULES = ["engine/core.py", "engine/physics.py", "engine/bloch.py", "engine/mt_walk.py", "engine/mt.py", "viz/pedagogy.py",
                   "spec/walk.py"]
# Probes that remain, and why. Every other property is a declared attribute.
_ALLOWED_PROBES = {
    # physics.length_scales_of: the one reader of the declared length scales, which refuses their absence
    "length_scales",
    # sub-step helpers accept duck-typed objects (tests pass bare classes)
    "permeability", "surface_substep_frac", "reflection_step_fraction",
    # optional methods: not every geometry records a boundary local time / binding / membrane
    "reflect_with_log_weight", "reflect_with_binding", "permeate",
    # core._walk_geometry compartment-id derivation (issue #94, A7)
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


def test_every_loaded_dmipy_sim_module_comes_from_this_package_tree():
    """An editable install of another checkout registers a meta-path finder that resolves any
    ``dmipy_sim.<name>`` the imported package lacks from THAT checkout, so a module deleted here
    can silently import from elsewhere and a test of its absence passes vacuously. Every loaded
    dmipy_sim module must live under the package that was imported."""
    import sys
    root = Path(dmipy_sim.__file__).parent.resolve()
    for name in _DOWNSTREAM_MODULES:
        importlib.import_module(name)
    strays = {n: getattr(m, "__file__", None) for n, m in list(sys.modules.items())
              if (n == "dmipy_sim" or n.startswith("dmipy_sim.")) and getattr(m, "__file__", None)
              and not Path(m.__file__).resolve().is_relative_to(root)}
    assert not strays, f"modules loaded from outside {root}: {strays}"


# ── the door is locked: substrates enter as specs, nothing else constructs one (#130) ────────────
def test_io_readers_construct_nothing_and_the_builders_are_gone():
    """A file reader parses; the geometry a file describes is built from the SPEC the producer emits. So no
    module under ``dmipy_sim.io`` may import the geometry or engine packages, and the bespoke walk builders
    (``replay.builders``) no longer exist."""
    import ast
    pkg = Path(dmipy_sim.__file__).parent
    for py in sorted((pkg / "io").glob("*.py")):
        tree = ast.parse(py.read_text())
        for node in ast.walk(tree):
            names = []
            if isinstance(node, ast.ImportFrom):
                names = [(node.module or "") + ("." * node.level)]
            elif isinstance(node, ast.Import):
                names = [a.name for a in node.names]
            for n in names:
                assert not re.search(r"(^|\.)(geometry|engine|replay|spec)(\.|$)", n), \
                    f"{py.name} imports {n!r}: a reader must not build or walk a geometry"
    assert not list((pkg / "replay" / "builders").glob("*.py")) if (pkg / "replay" / "builders").exists() else True
    assert not any((pkg / "io" / f).exists() for f in ("cactus.py", "winther.py", "mesh_substrate.py"))
    with pytest.raises(ImportError):
        importlib.import_module("dmipy_sim.replay.builders")


def test_every_public_geometry_has_a_spec():
    """Every geometry class the package exports is written by ``spec_of`` (and so read by
    ``geometry_from_spec``): a substrate a walk can run on is a substrate a spec can describe."""
    import inspect
    import dmipy_sim.geometry as g
    from dmipy_sim.spec import build
    src = inspect.getsource(build)
    for name in g.__all__:
        obj = getattr(g, name)
        if inspect.isclass(obj) and issubclass(obj, g.Geometry) and obj is not g.Geometry:
            assert name in src, f"{name} is exported but spec_of / geometry_from_spec do not know it"


def test_a_substrate_without_a_spec_spelling_is_refused_by_every_driver():
    """The door: a driver walks what a spec can describe. An object that only quacks like a geometry is refused,
    with the reason; a Mesh built from arrays gets its surface written to the spec cache so it HAS a spelling."""
    from dmipy_sim.spec import SpecError, as_geometry

    class Quack:
        def init_positions(self, n, key): return jnp.zeros((n, 3), jnp.float32)
        def reflect(self, r, step): return r + step
        length_scales = LengthScales()

    with pytest.raises(TypeError, match="neither a SubstrateSpec nor a Geometry"):
        as_geometry(Quack())

    class Unspelled(Geometry):                       # a Geometry spec_of does not know
        def init_positions(self, n, key): return jnp.zeros((n, 3), jnp.float32)
        def reflect(self, r, step): return r + step
        length_scales = LengthScales()

    with pytest.raises(SpecError, match="spec spelling"):
        as_geometry(Unspelled())
    g = dmipy_sim.Cylinder(2e-6, (0, 0, 1))
    assert as_geometry(g) is g and g._spec_source is not None and g._spec_source == g.spec


# ── the acquisition surface is an inventory that can only shrink (#173) ─────────────────────────
# What the scanner does from t=0 to TE is carried by these containers, in these RF dialects, read by
# these consumers, against these scanner catalogues. Each set is declared; a NEW member fails here
# by name. #173 collapses them into one ScannerSequence piece by piece, and each piece deletes from
# these sets -- never adds.

# §1.1 -- a class whose instances carry G on a dt grid, or a B1 envelope on one
_ACQUISITION_CONTAINERS = {
    "dmipy_replay.acquisition.scanner_sequence.ScannerSequence",  # the one: physical G, the schedule, readout, timing, encoding
}

# §1.1 -- the RF dialect: one. Every builder's events are RFEvent; a dict is read only by from_dict
_RF_DICT_READ_BOUNDARY = {"dmipy_replay.acquisition.rf"}

# §1.1 / #171 -- Sequence constructors that declare no RF schedule at all
_SEQUENCE_CONSTRUCTORS_WITHOUT_RF = {
    "sequences.from_waveform",                                     # arbitrary: cannot know its schedule
}

# §1.2 -- callables that take an acquisition AND its RF / echo / refocus time as separate arguments
_RF_SIDE_CHANNELS = set()      # #173 piece 6: every consumer reads the object's rf / readout / refocus time

# §1.3 -- the scanner catalogues: ONE source (the cited JSON) and the views derived from it at import
_SCANNER_SOURCE = {"dmipy_replay.acquisition.scanner_constants.SCANNER_CONSTANTS"}
_SCANNER_VIEWS = {"dmipy_replay.acquisition.scanners.SCANNERS", "dmipy_sim.sequences.pulseq.PULSEQ_SYSTEMS"}


def _class_attribute_names(cls_node):
    """Names a class gives its instances: dataclass fields and every ``self.<name> =`` in its methods."""
    import ast
    names = set()
    for node in cls_node.body:
        if isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
            names.add(node.target.id)
    for fn in ast.walk(cls_node):
        if isinstance(fn, (ast.FunctionDef, ast.AsyncFunctionDef)):
            for node in ast.walk(fn):
                targets = []
                if isinstance(node, ast.Assign):
                    targets = node.targets
                elif isinstance(node, ast.AnnAssign):
                    targets = [node.target]
                for t in targets:
                    if isinstance(t, ast.Attribute) and isinstance(t.value, ast.Name) and t.value.id == "self":
                        names.add(t.attr)
    return names


def _package_modules():
    """Every module of dmipy-sim and of the dmipy-replay it imports, ``(path, dotted name)``."""
    for src in (_SRC, _REPLAY_SRC):
        for py in sorted(src.rglob("*.py")):
            rel = py.relative_to(src.parent).with_suffix("")
            yield py, ".".join(rel.parts)


def test_no_new_acquisition_container():
    """A class whose instances carry a gradient ``G`` on a ``dt`` grid is an acquisition container. There is
    exactly one (a ``B1Pulse`` is a pulse's envelope, held by an ``RFEvent``, not a container)."""
    import ast
    found = set()
    for py, modname in _package_modules():
        for node in ast.walk(ast.parse(py.read_text())):
            if isinstance(node, ast.ClassDef):
                names = _class_attribute_names(node)
                if {"G", "dt"} <= names:
                    found.add(f"{modname}.{node.name}")
    assert found == _ACQUISITION_CONTAINERS, (
        f"acquisition containers changed. new: {sorted(found - _ACQUISITION_CONTAINERS)}, "
        f"gone: {sorted(_ACQUISITION_CONTAINERS - found)}. #173 collapses these into ScannerSequence; "
        f"a new one is a step the other way.")


def _every_builder():
    """One instance of every constructor that declares an RF schedule (or is documented not to), built small."""
    from dmipy_sim import sequences as _seqmod
    import dmipy_sim.engine.pulse_sequence as P
    bv = np.array([[1.0, 0.0, 0.0]])
    out = {
        "amplitude.pgse": _seqmod.pgse(bv, 4e-3, 20e-3, gradient_strengths=0.05, n_t=200),
        "amplitude.pgste": _seqmod.pgste(bv, 4e-3, 20e-3, gradient_strengths=0.05, n_t=200),
        "amplitude.ogse.cosine": _seqmod.ogse(bv, 100.0, 20e-3, gradient_strengths=0.05, shape="cosine", slew_rate=np.inf, n_t=400),
        "amplitude.ogse.trapezoid": _seqmod.ogse(bv, 50.0, 20e-3, gradient_strengths=0.05, shape="trapezoid", TE=44e-3, n_t=400),
        "amplitude.cpmg": _seqmod.cpmg(3, 20e-3, gradient_strengths=0.05, gradient_directions=bv, n_t_per_echo=50),
        "amplitude.ste": _seqmod.ste(24e-3, gradient_strengths=0.05, n_t=240),
        "amplitude.pte": _seqmod.pte([0.0, 0.0, 1.0], 24e-3, gradient_strengths=0.05, n_t=240),
        "sequences.pgse": _seqmod.pgse(bv, 4e-3, 20e-3, bvalues=[1e9], n_t=200),
        "sequences.cpmg": _seqmod.cpmg(3, 20e-3, bvalues=[1e9] * 3, n_t_per_echo=50),
        "sequences.ogse": _seqmod.ogse(bv, 100.0, 20e-3, bvalues=[1e8], timing=SequenceTiming(t_excite=0.0, t_refocus=4e-3, t_readout_pre_echo=0.0), n_t=400),
        "sequences.ste": _seqmod.ste(24e-3, bvalues=[1e8], n_t=240),
        "sequences.pte": _seqmod.pte([0.0, 0.0, 1.0], 24e-3, bvalues=[1e8], n_t=240),
        "pulse_sequence.bare_gradient_echo": P.bare_gradient_echo(20e-3, 1e-4),
        "pulse_sequence.bare_spin_echo": P.bare_spin_echo(20e-3, 1e-4),
        "sequences.pgste": _seqmod.pgste(bv, 4e-3, 20e-3, bvalues=[1e9], n_t=240),
        "sequences.gre": _seqmod.gre(20e-3, bvalues=[1e9], gradient_directions=bv, delta=4e-3, Delta=10e-3, n_t=200),
        "pulse_sequence.prepend_mt_prep": P.prepend_mt_prep(
            P.bare_spin_echo(20e-3, 1e-4), P.saturation_pulse(2000.0, 2e-3, flip_deg=500.0)),
    }
    G = np.zeros((1, 200, 3), np.float32); G[0, :50, 0] = 0.05; G[0, 50:100, 0] = -0.05
    out["sequences.from_waveform"] = _seqmod.from_waveform(G, 1e-4, bv)
    out["sequences.from_btensor_waveform"] = _seqmod.from_btensor_waveform(G, 1e-4)
    return out


def test_every_builder_emits_rf_events_of_the_one_dialect():
    """Every builder's ``rf`` is an ``RFSchedule`` of ``RFEvent`` -- one dialect, whatever the family; the
    Sequence constructors that declare no schedule at all are the declared ones (#171) and no others."""
    from dmipy_replay.acquisition.rf import RFEvent, RFSchedule
    without = set()
    for name, obj in _every_builder().items():
        rf = getattr(obj, "rf", None)
        if not rf:
            without.add(name)
            continue
        assert isinstance(rf, RFSchedule) and all(isinstance(e, RFEvent) for e in rf), f"{name}: {rf!r}"
        assert all(e.label for e in rf), f"{name}: an event without a coherence label: {rf!r}"
    assert without == _SEQUENCE_CONSTRUCTORS_WITHOUT_RF, (
        f"constructors without an RF schedule: {sorted(without)} (declared {sorted(_SEQUENCE_CONSTRUCTORS_WITHOUT_RF)}). "
        f"#173 piece 5 gives every builder a schedule; this set only shrinks.")


def test_nothing_re_derives_from_a_list_of_events():
    """The coherence mask, the spin-echo sign, the refocus and mixing times are RFSchedule's; the free
    functions that each re-derived one of them from a bare list of events are gone, and so is the
    normaliser every reader once called."""
    import dmipy_replay.acquisition.waveforms as W
    import dmipy_sim.sequences.pulseq as Q
    import dmipy_replay.acquisition.rf as RF
    for mod, name in ((W, "rf_schedule_coherence"), (W, "effective_gradient_sign"), (Q, "_ste_from_rf_schedule"),
                      (Q, "_longitudinal_mask"), (RF, "as_rf_events")):
        assert not hasattr(mod, name), f"{mod.__name__}.{name} is back; it is RFSchedule's"


def test_rf_events_are_read_as_attributes_everywhere_but_the_boundary():
    """An RF event is its attributes. ``['t_s']`` / ``.get('t_s')`` -- the key every event dialect had -- and
    the retired ``b1_envelope`` are read off a dict in exactly one module: ``RFEvent.from_dict``, the inverse
    of ``to_dict``; anywhere else is a reader that kept its own dialect."""
    pat = re.compile(r"""(?:\.get\(|\[)\s*['"](t_s|b1_envelope)['"]""")
    found = {}
    for py, modname in _package_modules():
        for m in pat.finditer(py.read_text()):
            found.setdefault(modname, set()).add(m.group(1))
    strays = {k: sorted(v) for k, v in found.items() if k not in _RF_DICT_READ_BOUNDARY}
    assert not strays, f"RF events read as dicts outside the boundary {sorted(_RF_DICT_READ_BOUNDARY)}: {strays}"


def test_no_rf_event_is_spelled_as_a_dict_anywhere():
    """A pulse is an ``RFEvent`` and nothing else. Building one as ``{'t_s': ...}`` or ``dict(t_s=...)`` -- the
    spellings the readers once accepted -- is the backdoor a tolerant boundary would let back in; it exists
    only inside ``acquisition.rf`` (``to_dict`` / ``from_dict``, the object's own serialised record)."""
    pat = re.compile(r"""['"]t_s['"]\s*:|\bdict\(\s*t_s\s*=""")
    found = {}
    for py, modname in _package_modules():
        if modname in _RF_DICT_READ_BOUNDARY:
            continue
        n = len(pat.findall(py.read_text()))
        if n:
            found[modname] = n
    assert not found, f"RF events spelled as dicts (build an RFEvent): {found}"


def test_no_stored_effective_gradient_and_no_flag_standing_in_for_the_schedule():
    """``G`` is the physical gradient and ``G_eff`` is derived from the schedule (#173 piece 3). The stored copy of
    the physical one (``G_display``), the flag that said which convention ``G`` was in (``_effective_gradient``),
    the index that stood in for a declared 180 (``_refocus_idx``), the switch that allowed a misaligned one
    (``allow_offcenter_180``) and the write-only timing flags that restated the gradient-free span around the
    180 (``_refocus_gap``, ``_ogse_two_train``, ``_refocus_duration``; it is ``refocus_gap`` now, derived) are
    gone, and stay gone."""
    gone = ("G_display", "_effective_gradient", "_refocus_idx", "_display_G", "_derive_display_gradient",
            "allow_offcenter_180",
            "_refocus_gap", "_ogse_two_train", "_refocus_duration",   # piece 4: questions asked of G and the schedule
            "class Waveform", "class Sequence:", "class BlochSequence", "def apply_rf_schedule",   # piece 5a: one container
            "allow_unmatched_periods",
            "def trapezoidal_ogse", "def b_trapezoidal_ogse", "_fill_lobe", "_lobe_n_rise", "_trap_cosine_profile",
            "G_magnitude", "refocus_duration", "kind='sine'",                        # piece 5b: one set of builders
            "def run_bloch_sequence", "def _refocus_time_of", 'refocus_time="auto"', "rf_events=None")   # piece 6
    found = {}
    for py, modname in _package_modules():
        text = py.read_text()
        hits = [g for g in gone if g in text]
        if hits:
            found[modname] = hits
    assert not found, f"retired names back in the package: {found}"


def test_rf_is_taken_apart_from_the_gradient_only_where_declared():
    """A callable that takes an acquisition AND separately its RF schedule / echo / refocus time is a side
    channel: the two can then disagree by signature (#172 is one such bug). #173 piece 6 emptied the set: every
    consumer reads ``seq.rf``, ``seq.readout`` and ``seq.rf.refocus_time`` from the object."""
    import ast
    acq = {"waveform", "wf", "seq", "sequence", "acq"}
    side = {"rf_events", "refocus_time", "echo_steps"}
    found = set()
    for py, modname in _package_modules():
        tree = ast.parse(py.read_text())
        for node in ast.walk(tree):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                params = {a.arg for a in node.args.args + node.args.kwonlyargs}
                if params & acq and params & side:
                    found.add((f"{modname}.{node.name}", tuple(sorted(params & side))))
    assert found == _RF_SIDE_CHANNELS, (
        f"side channels changed. new: {sorted(found - _RF_SIDE_CHANNELS)}, gone: {sorted(_RF_SIDE_CHANNELS - found)}")


def test_scanner_numbers_live_in_one_place():
    """A module-scope name spelled like a scanner catalogue (``...SCANNER...`` / ``...SYSTEMS``) is either THE
    source -- the cited JSON, loaded -- or a view derived from it. A view's expression carries no numeric
    literal: every scanner number in Python is read, never written (#173 piece 1)."""
    import ast
    pat = re.compile(r"^[A-Z_]*(SCANNER|SYSTEMS)[A-Z_]*$")
    found, literal = set(), set()
    for py, modname in _package_modules():
        tree = ast.parse(py.read_text())
        nested = {id(n) for scope in ast.walk(tree) if isinstance(scope, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef))
                  for n in ast.walk(scope) if n is not scope}
        for node in ast.walk(tree):
            if id(node) in nested or not isinstance(node, (ast.Assign, ast.AnnAssign)):
                continue
            targets = node.targets if isinstance(node, ast.Assign) else [node.target]
            for t in targets:
                if isinstance(t, ast.Name) and pat.match(t.id):
                    q = f"{modname}.{t.id}"
                    found.add(q)
                    if node.value is not None and any(isinstance(n, ast.Constant) and isinstance(n.value, (int, float))
                                                      and not isinstance(n.value, bool) for n in ast.walk(node.value)):
                        literal.add(q)
    assert found == _SCANNER_SOURCE | _SCANNER_VIEWS, (
        f"scanner catalogues changed. new: {sorted(found - _SCANNER_SOURCE - _SCANNER_VIEWS)}, "
        f"gone: {sorted((_SCANNER_SOURCE | _SCANNER_VIEWS) - found)}")
    assert not literal, f"a scanner catalogue carries numbers in Python instead of reading the JSON: {sorted(literal)}"


def test_replay_knobs_are_tissue_orientation_and_scanner():
    """A replay setting is three things, each stated once (dmipy-sim#283): ``tissue`` (a ``spec.Tissue`` or None),
    ``orientation`` / the phantom's pose, ``scanner``. No replay entry point takes a flat physical value -- a T2, a
    B0, a chi, a field direction -- and none takes a string or a boolean for the tissue."""
    import inspect
    from dmipy_sim.replay.replay import ReplayPack
    from dmipy_sim.replay.phantom import ReplayPhantom
    from dmipy_sim.phantom.phantom import Phantom
    from dmipy_sim.phantom.partition import PartitionPhantom
    from dmipy_sim.phantom.substrates import PackSubstrate, FreeWater
    flat = {"T2", "T1", "rho2", "rho1", "D", "B0", "b0_dir", "chi_iso", "chi_aniso", "T2_s", "T1_s", "rho_m_s", "D_m2_s", "B0_T"}
    entry_points = [ReplayPack.replay, ReplayPack.walker_signals, ReplayPack.walker_phases, ReplayPack.replay_bloch, ReplayPack.pose_response,
                    ReplayPhantom.replay, ReplayPhantom.replay_bloch, Phantom.replay, PartitionPhantom.replay,
                    PackSubstrate.__init__, FreeWater.__init__]
    for fn in entry_points:
        params = inspect.signature(fn).parameters
        assert not (flat & set(params)), f"{fn.__qualname__} takes a flat physical value: {sorted(flat & set(params))}"
        if "tissue" in params:
            assert params["tissue"].default in (None, inspect.Parameter.empty), fn.__qualname__
    for fn in (ReplayPack.replay, ReplayPack.walker_signals, ReplayPack.walker_phases, ReplayPack.replay_bloch, ReplayPack.pose_response,
               ReplayPhantom.replay, ReplayPhantom.replay_bloch, Phantom.replay, PartitionPhantom.replay):
        assert "scanner" in inspect.signature(fn).parameters, fn.__qualname__


def test_the_walk_keeps_its_bands_and_has_no_second_spelling():
    """The producer's walk keeps the bands (dmrai-lab/dmipy-sim#741): ``K`` is its parameter, ``raw_subset`` the
    opt-in positions; there is no switch to a walk of saves (``accumulate=``, ``compress=``, ``storage_dtype=``) on
    the producer or its cost, and none is spelled anywhere in the package."""
    import inspect
    from dmipy_sim.engine.core import _walk_geometry, walk_cost
    params = inspect.signature(_walk_geometry).parameters
    assert {"K", "block_order", "window_T", "raw_subset"} <= set(params)
    assert not {"compress", "accumulate", "storage_dtype"} & set(params)
    for gone in ("compress", "accumulate"):
        with pytest.raises(TypeError, match=gone):
            walk_cost(10, 2e-9, dmipy_sim.FreeDiffusion(), 1e-3, 1e-4, **{gone: 4})
    hits = [str(f.relative_to(_SRC)) for f in _SRC.rglob("*.py") if re.search(r"\b(compress|accumulate)\s*=", f.read_text())]
    assert not hits, f"a switch to a walk of saves is spelled in {hits}"


def test_a_backends_request_names_the_walk_and_no_save_grid():
    """``WalkRequest`` (dmrai-lab/dmipy-sim#747, dmipy-sim-cuda#73 section 3) names the walk a backend accumulates --
    the block, the blocks, the steps per block, the band ``K``, the order ``m``, the field's modes and read interval,
    the event capacity, the raw subset -- and carries no save grid: no ``n_t``, no ``dt_save``, no
    ``field_sample_every``. A backend returns the coefficient form (``CoefficientBatch``); ``WalkBatch`` is the
    adaptive producer's walk of saves until dmipy-sim#742."""
    from dmipy_sim.engine.backends import WalkRequest, CoefficientBatch
    assert not {"n_t", "dt_save", "field_sample_every", "accumulate", "block_order"} & set(WalkRequest._fields)
    assert {"block_h", "n_blocks", "sub_steps", "K", "m", "K_c3", "field_read_interval", "events", "raw_subset"} <= set(WalkRequest._fields)
    assert {"position_coeffs", "boundary_bands", "boundary_endpoint", "compartment", "qv_positions", "qv_boundary",
            "events", "bound_events", "field_bands", "field_qv", "dt_f", "saves", "precision", "K", "m"} <= set(CoefficientBatch._fields)
    assert not {"positions", "n_t"} & set(CoefficientBatch._fields)
    from dmipy_sim.engine import backends
    assert "n_t" not in backends.REQUEST_TEMPLATE_KEYS and {"block_h", "dt_sim", "K", "m", "ring", "events"} <= set(backends.REQUEST_TEMPLATE_KEYS)
    assert not {"sample_every", "n_tf"} & set(backends.FIELD_STRUCT_KEYS) and {"K_c3", "read_every", "mean"} <= set(backends.FIELD_STRUCT_KEYS)


def test_one_basis_and_no_header_with_a_save_grid():
    """A pack's bands are on one basis, the continuous one: no basis dispatch is left in ``compression`` (``BASES``,
    ``basis_of``, ``require_basis``, the per-save ``bridge_projection``, the per-save contact codec), the sine
    matrices are read in ``compression`` alone, and no module names a second basis. A header carrying a save grid or
    a basis is refused by name on load."""
    import ast
    from dmipy_sim.replay import compression as cx
    from dmipy_replay.replay.replay import _refuse_predating
    for gone in ("BASES", "basis_of", "require_basis", "bridge_projection", "rank_of", "is_lossless_at",
                 "encode_boundary_local_time", "decode_boundary_local_time", "surface_logweight_continuous",
                 "saves_basis", "decode_saves", "decode_bridge_dst", "decode_boundary_bridge", "_dst1_to_continuous",
                 "relaxation_logweight", "surface_logweight_series"):
        assert not hasattr(cx, gone), gone
    private = {"_dst_matrix", "_idst"}
    outside = set()
    for f in _SRC.rglob("*.py"):
        if f.name == "compression.py":
            continue
        tree = ast.parse(f.read_text())
        for fn in ast.walk(tree):
            if isinstance(fn, (ast.FunctionDef, ast.AsyncFunctionDef)):
                for node in ast.walk(fn):
                    if isinstance(node, ast.Call) and getattr(node.func, "id", getattr(node.func, "attr", None)) in private:
                        outside.add((str(f.relative_to(_SRC)), fn.name))
    assert not outside, f"the sine matrices are read outside compression: {sorted(outside)}"
    hits = [str(f.relative_to(_SRC)) for f in _SRC.rglob("*.py") if re.search(r"[\"']dst1[\"']", f.read_text())]
    assert not hits, f"a second basis is named in {hits}"
    for old, key in ((dict(walk_params=dict(n_t=11, dt_traj=1e-3, block=dict(h=1e-3))), "walk_params.n_t"),
                     (dict(walk_params=dict(block=dict(h=1e-3)), compression=dict(basis="continuous")), "compression.basis"),
                     (dict(walk_params=dict(block=dict(h=1e-3), cell_size=1e-5)), "walk_params.cell_size"),
                     (dict(walk_params=dict(block=dict(h=1e-3)), compression=dict(method="identity")), "compression.method = identity"),
                     (dict(walk_params=dict(T_max=0.01)), "no walk_params.block")):
        with pytest.raises(ValueError, match=r"This pack carries `" + key + r".*which this revision does not read; "
                                             r"re-walk the pack with a conformant producer\. A pack is never converted"):
            _refuse_predating(old, "old.rpk")


# ── the save grid is gone: no spelling of it survives, and every producer returns the bands ───────────────────
#: The names of the save grid and of the routes that read or wrote a walk kept at its saves. None is spelled anywhere
#: in the package: a walk has a block (``block_h``), a pack a ``walk_params.block``, and the saves route's encoders,
#: its RLE occupancy and its dense fidelity oracle are deleted rather than kept beside the band route.
_SAVE_GRID_NAMES = (r"dt_save", r"field_sample_every", r"save_interval", r"fill_field", r"defer_field", r"FIELD_DT_CAP",
                    r"encode_bridge_dst", r"measure_fidelity", r"quadrature_reference", r"comp_rle\w*", r"saves_basis", r"decode_saves", r"decode_bridge_dst", r"decode_boundary_bridge",
                    r"_decoded_channels", r"_window_n_t", r"_dst1_to_continuous", r"pathway_sign_se", r"unwrap_periodic",
                    r"replay_jax", r"pre_pulse_gradient_phase")


#: the band readers: a window's duration is their ``T`` (the third positional argument, or ``T=``)
_BAND_READERS = {"band_projection": 2, "band_projection_matrix": 2, "segment_projection": 1, "cosine_projection": 2,
                 "path_field_integral": 3, "_compile_continuous": 3, "bridge_window_matrix": 0, "mode_window_matrix": 0, "bridge_basis": 2,
                 "bridge_at": 2, "band_phase_variances": 2, "bridge_phase_variance": 2}
_SAVE_COUNTS = re.compile(r"\b(?:n_t|n_s|n_seg|n_saves|_window_n_t|Nt|N_t)\b")


#: the modules that READ a pack (the consumer routes): their band readers take a stated duration
_READER_MODULES = ("replay/replay.py", "replay/columnar.py", "replay/study.py", "replay/pose_series.py",
                   "replay/pose_device.py", "replay/continuation.py", "replay/phantom.py", "replay/time_segments.py",
                   "replay/so3.py", "fill/consolidate.py")


def _band_reader_T_args():
    """``(file, line, T argument's source)`` of every call to a band reader in the reader modules."""
    import ast
    out = []
    for f in (_SRC / m for m in _READER_MODULES):
        src = f.read_text()
        for node in ast.walk(ast.parse(src)):
            if not isinstance(node, ast.Call):
                continue
            name = getattr(node.func, "id", getattr(node.func, "attr", None))
            if name not in _BAND_READERS:
                continue
            at = _BAND_READERS[name]
            arg = next((k.value for k in node.keywords if k.arg == "T"), None)
            if arg is None and len(node.args) > at:
                arg = node.args[at]
            if arg is not None:
                out.append((str(f.relative_to(_SRC)), node.lineno, ast.get_source_segment(src, arg)))
    return out


def test_no_reader_indexes_a_band_by_a_save_count():
    """Every band reader is handed a window's DURATION (``T_window``, a window table's ``T``, the walk's ``T``), never
    one rebuilt from a count of saves (``(n_t - 1) * dt``): a band is a projection over an interval, and the interval is
    what the pack states. The readers' windows are ``(pack, t0, T)``; ``ReplayPack`` keeps no per-window save count;
    no reader reads ``compression.basis`` (only the refusal names it)."""
    from dmipy_sim.replay.replay import ReplayPack
    calls = _band_reader_T_args()
    assert len(calls) >= 8, calls                                         # the scan sees the readers
    bad = [c for c in calls if _SAVE_COUNTS.search(c[2] or "")]
    assert not bad, f"a band reader is given a duration rebuilt from a save count: {bad}"
    assert not hasattr(ReplayPack, "_window_n_t") and not hasattr(ReplayPack, "_decoded_channels")
    reads = []
    for f in sorted(_SRC.rglob("*.py")):
        for i, line in enumerate(f.read_text().splitlines(), 1):
            if re.search(r"""(?:\[|get\()\s*["']basis["']""", line) and "REFUSED" not in line:
                reads.append(f"{f.relative_to(_SRC)}:{i}")
    assert not reads, f"compression.basis is read in {reads}"


def test_no_spelling_of_the_save_grid_survives_in_the_package():
    pat = re.compile(r"\b(?:" + "|".join(_SAVE_GRID_NAMES) + r")\b")
    hits = {}
    for f in sorted(_SRC.rglob("*.py")):
        found = sorted({m.group(0) for m in pat.finditer(f.read_text())})
        if found:
            hits[str(f.relative_to(_SRC))] = found
    assert not hits, f"the save grid is spelled in {hits}"


# ── the field read is charged from the walk's measured read series, never from an assumed gradient ──────────────
#: The names of the field read's assumed-gradient route: the basis's largest gradient over a sheath and the rule and
#: record keys built on it. The read's term is ``(gamma B0 chi)^2 dt_f^2 / 12 x (sum_c sqrt(QV_c))^2`` of the read series (RPK.md
#: 9.5.5), measured by the walk; nothing in the package spells the assumed maximum.
_GRADIENT_ROUTE_NAMES = (r"gradient_scale", r"grad_b_per_m", r"grad_b", r"thinnest_sheath_m", r"_field_rule_inputs")


def test_the_field_read_has_no_assumed_gradient():
    pat = re.compile(r"\b(?:" + "|".join(_GRADIENT_ROUTE_NAMES) + r")\b")
    hits = {}
    for f in sorted(_SRC.rglob("*.py")):
        found = sorted({m.group(0) for m in pat.finditer(f.read_text())})
        if found:
            hits[str(f.relative_to(_SRC))] = found
    assert not hits, f"the assumed-gradient field read is spelled in {hits}"
    import inspect
    from dmipy_sim.engine import accumulate
    from dmipy_sim.fields.strand_field import StrandFieldBasis, StrandFieldRecord
    from dmipy_sim.fields.susceptibility_field import FieldGrid, FieldGridRecord
    assert not hasattr(accumulate, "field_read_interval")             # the rule is field_read_rule on a measured QV
    assert list(inspect.signature(accumulate.field_read_term).parameters)[:4] == ["dt_f", "qv_increments", "B0", "chi"]
    assert list(inspect.signature(accumulate.field_read_rule).parameters)[:3] == ["qv_increments", "B0", "chi"]
    for cls in (StrandFieldBasis, StrandFieldRecord, FieldGrid, FieldGridRecord):
        assert not hasattr(cls, "gradient_scale"), cls.__name__
    from dmipy_sim.engine.backends import CoefficientBatch
    assert "field_qv_increments" in CoefficientBatch._fields
    from dmipy_sim import CoefficientWalk
    assert "qv_field_increments" in CoefficientWalk.__dataclass_fields__


def test_the_read_term_folds_the_channels_as_the_square_of_their_summed_roots():
    """The channels are linear functionals of one path: ``v_read = (gamma B0 chi dt_f)^2 (sum_c sqrt(kappa_c QV_c))^2``,
    ``kappa_c = 1/12`` unless a larger one is given; the rule and the envelope's ``B0 chi`` hold the same term at the
    floor (RPK.md 9.5.5: the term, ``v_read / 2``, never ``v_read`` against the floor's square)."""
    from dmipy_replay.constants import GAMMA
    from dmipy_sim.engine.accumulate import field_read_term, field_read_rule, field_read_limit_B0_chi
    qv, dt, B0, chi = np.array([4.0, 1.0, 0.0]), 2e-6, 7.0, 1e-6
    v = (GAMMA * B0 * chi * dt) ** 2 * (2.0 + 1.0) ** 2 / 12.0
    assert field_read_term(dt, qv, B0, chi) == pytest.approx(0.5 * v)
    k = np.array([0.25, 1.0 / 12.0, 1.0])
    assert field_read_term(dt, qv, B0, chi, k) == pytest.approx(0.5 * (GAMMA * B0 * chi * dt) ** 2 * (np.sqrt(1.0) + np.sqrt(1.0 / 12.0)) ** 2)
    dt_r = field_read_rule(qv, B0, chi, 1e-3)
    assert field_read_term(dt_r, qv, B0, chi) == pytest.approx(1e-3)                # the term = floor at the rule's read
    bc = field_read_limit_B0_chi(dt, qv, 1e-3)
    assert field_read_term(dt, qv, bc, 1.0) == pytest.approx(1e-3)                  # the term = floor at the envelope's B0 chi


def test_a_field_pack_without_the_read_series_qv_is_refused():
    """A pack whose field channel carries no ``walk_params.qv.field_increments`` was charged by another rule; it is
    refused by name, never re-charged."""
    from dmipy_replay.replay.replay import _refuse_predating
    meta = dict(walk_params=dict(block=dict(h=1e-4), qv=dict(positions=[1.0, 1.0, 1.0], field=[0.0])),
                compression=dict(channels=dict(susc_field_basis=dict(K=4))))
    with pytest.raises(ValueError, match=r"This pack carries `no walk_params.qv.field_increments`.*re-walk"):
        _refuse_predating(meta, "old.rpk")


def test_every_producer_takes_a_block_and_returns_a_coefficient_walk():
    """The fifth positional argument of every producer is ``block_h`` (the accumulation block, seconds); none takes a
    walk's ``n_t`` or a ``dt_save``; every one returns a ``CoefficientWalk``, whose raw subset is its own arrays (no
    view of saves)."""
    import inspect
    from dmipy_sim import CoefficientWalk
    from dmipy_sim.engine.adaptive import simulate_trajectories_adaptive
    from dmipy_sim.engine.core import _walk_geometry, walk_cost
    from dmipy_sim.engine.mt_walk import simulate_mt_trajectories
    from dmipy_sim import walk
    for fn, at in ((_walk_geometry, 4), (simulate_mt_trajectories, 4), (simulate_trajectories_adaptive, 4), (walk, 3)):
        names = list(inspect.signature(fn).parameters)
        assert names[at] == "block_h", (fn.__qualname__, names[:6])
        assert not {"n_t", "dt_save", "field_sample_every", "fill_field", "defer_field"} & set(names), fn.__qualname__
    assert not {"n_t", "dt_save"} & set(inspect.signature(walk_cost).parameters)
    fields = set(CoefficientWalk.__dataclass_fields__)
    assert not {"n_t", "dt", "dt_save", "positions", "boundary_local_time", "bound_frac", "field_samples", "field_sample_every"} & fields
    assert not hasattr(CoefficientWalk, "saves")
    cyl = dmipy_sim.Cylinder(2e-6, (0, 0, 1))
    for cw in (_walk_geometry(6, 2e-9, cyl, 2e-3, 5e-4, seed=0, require_gpu=False),
               simulate_mt_trajectories(6, 2e-9, cyl, 2e-3, 5e-4, kappa_MT=1e-5, dwell_time=1e-3, seed=0, require_gpu=False)):
        assert isinstance(cw, CoefficientWalk) and cw.h == pytest.approx(5e-4)


# ── the public surface is the replay surface; the fused engine is an internal oracle (#740) ──────────────────────
def test_walk_is_the_one_public_producer_and_the_engine_primitive_is_not_exported():
    """``walk(spec, ...)`` is the one public producer of a walk; the engine primitive (one resolved geometry, no spec
    conversion) lives in ``engine.core`` under a private name and is exported nowhere, and neither is the retired
    spelling of the producer."""
    from dmipy_sim.engine import core
    assert "walk" in dmipy_sim.__all__ and callable(dmipy_sim.walk)
    for name in ("_walk_geometry", "simulate_trajectories", "walk_spec"):
        assert name not in dmipy_sim.__all__ and not hasattr(dmipy_sim, name), name
    assert callable(core._walk_geometry) and not hasattr(core, "simulate_trajectories")
    import dmipy_sim.spec as spec_pkg
    assert not hasattr(spec_pkg, "walk_spec") and "walk_spec" not in spec_pkg.__all__
    assert importlib.import_module("dmipy_sim.spec.walk").walk is dmipy_sim.walk


def test_a_package_carries_no_private_name_of_its_module():
    """``dmipy_sim.replay`` and ``dmipy_sim.viz`` re-export their module's PUBLIC names; a private name is reached at
    its module (``dmipy_sim.replay.replay._x``), never through the package. A name a caller needs is public."""
    import dmipy_sim.replay as replay_pkg
    import dmipy_sim.viz as viz_pkg
    for pkg in (replay_pkg, viz_pkg):
        private = sorted(k for k in vars(pkg) if k.startswith("_") and not k.startswith("__"))
        assert not private, f"{pkg.__name__} carries private names {private}"
    assert callable(viz_pkg.split_cells) and not hasattr(viz_pkg, "_split_cells")


def test_one_spelling_of_the_columnar_open_and_the_bounce_budget():
    """A pack is opened by ``ReplayPack.open(uri)``; the bounce budget is ``geometry._boundary.bounce_budget``. The
    module-level opener and the packed alias are removed."""
    from dmipy_sim.geometry import _boundary, packed
    from dmipy_sim.replay import columnar
    from dmipy_sim.replay.replay import ReplayPack
    assert not hasattr(columnar, "open_columnar") and "open_columnar" not in columnar.__all__
    assert not hasattr(packed, "packed_bounce_budget") and callable(_boundary.bounce_budget)
    assert callable(ReplayPack.open)


def test_the_public_surface_is_the_replay_surface():
    """The top level exports the walk and its contraction: ``simulate`` (the coefficient walk and its contraction in
    one call), the producers (each returns a ``CoefficientWalk``), ``walk_cost`` / ``walk``, the pack side
    (``build_replay_pack``, ``read_rpk``, ``ReplayPack``) and the phantom and scanner objects. No walk of saves, no
    positions-replay helper and no forward integrator is exported: the fused scalar engine and the forward vector-Bloch
    engine are the oracles the replay is held to, underscored and imported by the admissions that hold it."""
    import inspect
    from dmipy_sim.engine import bloch, core
    for name in ("simulate", "simulate_mixture", "simulate_cpmg", "simulate_mt_trajectories",
                 "walk_cost", "walk", "CoefficientWalk", "build_replay_pack", "read_rpk", "ReplayPack",
                 "Phantom", "ScannerSequence", "ScannerLimits"):
        assert name in dmipy_sim.__all__ and hasattr(dmipy_sim, name), name
    gone = ("simulate_bloch", "simulate_fused", "unwrap_periodic", "replay_jax", "replay_bloch",
            "replay_bloch_jax", "finite_180_longitudinal_dwell", "pre_pulse_gradient_phase", "pathway_sign_se")
    for name in gone:
        assert name not in dmipy_sim.__all__ and not hasattr(dmipy_sim, name), name
    params = set(inspect.signature(dmipy_sim.simulate).parameters)
    assert not {"engine", "return_positions", "return_compartments", "return_walker_signals"} & params, params
    assert "simulate_fused" not in vars(core) and "simulate_bloch" not in vars(bloch)
    from dmipy_sim.engine._fused import simulate_fused
    assert callable(simulate_fused) and callable(bloch._simulate_bloch)
    exporting = [str(f.relative_to(_SRC)) for f in _SRC.rglob("*.py")
                 if re.search(r"__all__[^\n]*[\"'](?:simulate_fused|_simulate_bloch|simulate_bloch)[\"']", f.read_text())]
    assert not exporting, f"an oracle is exported from {exporting}"


# ── every band channel's container is derived; none is flat or a fixed curve by default (#769) ──────────────────
def test_no_band_channel_container_is_flat_or_fixed_by_default():
    """Positions, contact and field are held in containers derived per pack against its floor (``"derived"``); the
    registry's flat depth and the fixed curve are only ever stated explicitly, and the spelling of the old default is
    gone. The field's builder knob is a container, not a depth: no ``susc_path_bits``, no two-depth ladder."""
    import inspect
    from dmipy_sim.replay import bank
    p = inspect.signature(bank.build_replay_pack).parameters
    assert [p[k].default for k in ("position_container", "blt_container", "field_container")] == ["derived"] * 3
    assert "susc_path_bits" not in p
    w = inspect.signature(bank.window_of_pack).parameters
    assert "susc_path_bits" not in w and w["field_container"].default is None            # None: the parent's own ranges
    assert not hasattr(bank, "path_container")
    assert "containers" not in inspect.signature(bank.derive_field_band).parameters
    assert inspect.signature(bank.susc_path_encode_modes).parameters["container"].default is inspect.Parameter.empty
    assert inspect.signature(bank.susc_path_encode_series).parameters["container"].default is None
    assert isinstance(bank._field_container("derived"), bank.DerivedFieldContainer)
    assert isinstance(bank._container("derived"), dmipy_sim.replay.compression.DerivedContainer)
    for spelling in ("bands", "flat", "pilot"):
        with pytest.raises(ValueError, match="derived"):
            bank._field_container(spelling)
        with pytest.raises(ValueError, match="derived"):
            bank._container(spelling)


def test_the_time_segment_layout_is_partitions_in_words(tmp_path):
    """#770: the layout stores partitions of the walk's time, not shapes -- no writer of a vector per shape (the shape
    profile, the per-shape contraction, the background stamp over a columnar source and the shape-vector width helpers
    are gone; the writers take a block's rows and partitions); every column the layout writes is uint32 words but the
    weights (float32) and the pool (uint8), so no tier column is float32; and no reader of ``time_segments/1``."""
    import inspect
    import json
    from dmipy_sim.replay import time_segments as ts
    for gone in ("stamp_background", "vector_width", "vector_code", "vector_fields", "pack_vectors", "unpack_vectors",
                 "_contractions", "_profile", "_axis_sequence", "_pack_column", "_vector_declaration", "_background_rows"):
        assert not hasattr(ts, gone), gone
    assert list(inspect.signature(ts.write_time_segment_layout).parameters)[:2] == ["blocks", "out_dir"]
    assert list(inspect.signature(ts.segment_rows).parameters)[:3] == ["source", "partitions", "out_dir"]
    src = inspect.getsource(ts.write_time_segment_layout)
    maps = re.findall(r"\bmm\(([^,]+),\s*np\.(\w+)", src)
    assert maps and all(dtype == "uint32" or (name, dtype) in (('"w.npy"', "float32"), ('"pool.npy"', "uint8")) for name, dtype in maps), maps
    assert _r_time_segments.FORMAT == "time_segments/2"
    (tmp_path / "manifest.json").write_text(json.dumps({"format": "time_segments/1"}))
    with pytest.raises(ValueError, match="time_segments/1"):
        _r_time_segments.read_manifest(str(tmp_path))


# ── the field carries twelve components: iso_P_zz is the reader's (RPK.md 5.3) ───────────────────────────────────
def test_the_field_carries_twelve_components_and_iso_P_zz_is_never_one():
    """Every field basis emits, the walk accumulates and the pack stores ``iso_local``, five ``iso_P`` and six
    ``aniso_G``; ``iso_P_zz = 3 iso_local - iso_P_xx - iso_P_yy`` is formed by the contraction. A producer that offers
    the thirteenth is refused, and so is a pack that lists it."""
    import inspect
    from dmipy_sim.fields import hollow_cylinder as hc
    from dmipy_sim.replay import bank
    from dmipy_replay.replay.replay import _refuse_predating
    assert len(_r_field.CHANNEL_NAMES) == 12 and "iso_P_zz" not in _r_field.CHANNEL_NAMES and _r_field.ISO_CHANNEL_NAMES == _r_field.CHANNEL_NAMES[:6]
    cyl = hc.HollowCylinderField(np.zeros((1, 2)), [1e-6], [2e-6])
    assert cyl.channels(np.array([[3e-6, 0.0, 0.0], [0.0, 1.5e-6, 0.0]])).shape == (2, 12)
    # the contraction forms the thirteenth: the field of a tensor with a trace is the 13-channel contraction
    rng = np.random.default_rng(0); c = rng.normal(size=(5, 12)); H = [0.3, -0.5, 0.8]
    P = _r_field.iso_P6(c); q = _r_field.q_of_H(H)
    np.testing.assert_allclose(P[:, 2], 3 * c[:, 0] - c[:, 1] - c[:, 2])
    np.testing.assert_allclose(_r_field.field_terms(c, H)[0], c[:, 0] - P @ q, atol=1e-12)
    np.testing.assert_allclose(_r_field.field_terms(c, H)[1], c[:, 6:12] @ q) and _r_field.field_terms(c[:, :6], H)[1] is None
    # refusals by name
    with pytest.raises(ValueError, match="iso_P_zz"):
        bank.susc_path_encode_modes(np.zeros((4, 3, 13)), ["iso_local", "iso_P_xx", "iso_P_yy", "iso_P_zz", "iso_P_xy", "iso_P_xz", "iso_P_yz",
                                                         "aniso_G_xx", "aniso_G_yy", "aniso_G_zz", "aniso_G_xy", "aniso_G_xz", "aniso_G_yz"],
                                    T=1e-3, container=None)
    for sfb in ({"channels": ["iso_local", "iso_P_xx", "iso_P_yy", "iso_P_zz"]}, {"iso_P_zz": "stored", "channels": ["iso_local"]}):
        meta = {"walk_params": {"block": {}, "qv": {"field_increments": [0.0]}}, "compression": {"channels": {"susc_field_basis": sfb}}}
        with pytest.raises(ValueError, match="iso_P_zz"):
            _refuse_predating(meta, "p.rpk")
    assert "implied" not in inspect.signature(bank.FieldCodec.__init__).parameters
    assert "atol_trace" not in inspect.signature(bank.susc_path_encode_modes).parameters
