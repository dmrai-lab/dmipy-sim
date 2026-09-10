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
import importlib
import re
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
from dmipy_sim.acquisition.timing import SequenceTiming
import pytest

import dmipy_sim
from dmipy_sim.geometry import Geometry, LengthScales
from dmipy_sim.geometry._boundary import WallHit

_SRC = Path(dmipy_sim.__file__).parent

# Every module path of the package; downstream (dmipy-fit / dmipy-design) imports a subset of these.
_DOWNSTREAM_MODULES = [
    "dmipy_sim.replay.replay", "dmipy_sim.constants", "dmipy_sim.acquisition.waveforms",
    "dmipy_sim.replay.so3", "dmipy_sim.replay.fod", "dmipy_sim.replay.compression", "dmipy_sim.acquisition.rf",
    "dmipy_sim.replay.bank", "dmipy_sim.sequences", "dmipy_sim.sequences.pulseq",
    "dmipy_sim.substrate", "dmipy_sim.substrate.biophysical_constants", "dmipy_sim.substrate.substrate",
    "dmipy_sim.geometry", "dmipy_sim.geometry.mesh", "dmipy_sim.geometry.curved_cylinder",
    "dmipy_sim.replay.phantom", "dmipy_sim.replay.trajectories",
    "dmipy_sim.fields.susceptibility", "dmipy_sim.fields.susceptibility_field",
    # the engine package
    "dmipy_sim.engine", "dmipy_sim.engine.core", "dmipy_sim.engine.physics", "dmipy_sim.engine.bloch",
    "dmipy_sim.engine.pulse_sequence", "dmipy_sim.engine.mt", "dmipy_sim.engine.mt_walk",
    "dmipy_sim.engine.gpu", "dmipy_sim.engine._gpu_config",
    # the replay package; dmipy_sim.replay is the package and keeps replay.py's surface
    "dmipy_sim.replay._replay_kernel", "dmipy_sim.spec", "dmipy_sim.spec.build", "dmipy_sim.spec.walk",
    "dmipy_sim.spec.producers", "dmipy_sim.io.caterpillar", "dmipy_sim.io.strands", "dmipy_sim.geometry.sphere_union",
    # acquisition / fields / viz; dmipy_sim.viz is the package re-exporting viz.py
    "dmipy_sim.acquisition.noise", "dmipy_sim.acquisition.scanners", "dmipy_sim.acquisition.scanner_constants",
    "dmipy_sim.acquisition.timing",
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
def _all_geometries():
    from dmipy_sim.geometry import mesh_shapes
    from dmipy_sim.geometry import (FreeDiffusion, Box1D, Sphere, Cylinder, Ellipsoid,
                                    PermeableSlab1D, PermeableShell, PackedCylinders, PackedSpheres,
                                    MyelinatedCylinder, PackedMyelinatedCylinders, CurvedCylinder,
                                    CurvedMyelinatedCylinder, PackedCurvedCylinders, Mesh,
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
        "MyelinatedCylinder": MyelinatedCylinder(3e-6, 5e-6, (0, 0, 1), 2e-9, 2e-9),
        "PackedMyelinatedCylinders": PackedMyelinatedCylinders([R] * 4, 0.7, c4, L4, N_max=8),
        "CurvedCylinder": CurvedCylinder(cl, radius=2e-6),
        "CurvedMyelinatedCylinder": CurvedMyelinatedCylinder(cl, r_in=2e-6, r_out=3e-6),
        "PackedCurvedCylinders": PackedCurvedCylinders([cl], [2e-6]),
        "Mesh": Mesh(V, F, feature_radius=1e-6),
    }


_NAMES = ["FreeDiffusion", "Box1D", "Sphere", "Cylinder", "Ellipsoid", "PermeableSlab1D",
          "PermeableShell", "PackedCylinders", "PackedSpheres", "MyelinatedCylinder",
          "PackedMyelinatedCylinders", "CurvedCylinder", "CurvedMyelinatedCylinder", "PackedCurvedCylinders",
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
    assert (ls.lookup_cell is not None) == (name in ("Mesh", "PackedCurvedCylinders"))

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
    assert ls("CurvedMyelinatedCylinder").min_feature == 2e-6            # the inner wall
    m = G["Mesh"]
    assert ls("Mesh") == LengthScales(min_feature=1e-6, lookup_cell=m.cell_size, is_mesh_feature=True)
    pk = G["PackedCurvedCylinders"]
    assert ls("PackedCurvedCylinders") == LengthScales(min_feature=2e-6, lookup_cell=pk.cell_size)


def test_duck_typed_objects_still_read_through_the_legacy_attributes():
    """An object that is not a Geometry is sized from its legacy attributes, in one place."""
    from dmipy_sim.engine.physics import length_scales_of

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
_ENGINE_MODULES = ["engine/core.py", "engine/physics.py", "engine/bloch.py", "engine/mt_walk.py", "engine/mt.py", "viz/pedagogy.py",
                   "spec/walk.py"]
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
              if n.startswith("dmipy_sim") and getattr(m, "__file__", None)
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

    with pytest.raises(SpecError, match="spec spelling"):
        as_geometry(Quack())
    g = dmipy_sim.Cylinder(2e-6, (0, 0, 1))
    assert as_geometry(g) is g and g._spec_source is not None and g._spec_source == g.spec


# ── the acquisition surface is an inventory that can only shrink (#173) ─────────────────────────
# What the scanner does from t=0 to TE is carried by these containers, in these RF dialects, read by
# these consumers, against these scanner catalogues. Each set is declared; a NEW member fails here
# by name. #173 collapses them into one ScannerSequence piece by piece, and each piece deletes from
# these sets -- never adds.

# §1.1 -- a class whose instances carry G on a dt grid, or a B1 envelope on one
_ACQUISITION_CONTAINERS = {
    "dmipy_sim.acquisition.scanner_sequence.ScannerSequence",     # the one: physical G, the schedule, readout, timing, encoding
}

# §1.1 -- the RF dialect: one. Every builder's events are RFEvent; a dict is read only by from_dict
_RF_DICT_READ_BOUNDARY = {"dmipy_sim.acquisition.rf"}

# §1.1 / #171 -- Sequence constructors that declare no RF schedule at all
_SEQUENCE_CONSTRUCTORS_WITHOUT_RF = {
    "sequences.from_waveform",                                     # arbitrary: cannot know its schedule
}

# §1.2 -- callables that take an acquisition AND its RF / echo / refocus time as separate arguments
_RF_SIDE_CHANNELS = {
    ("dmipy_sim.engine.bloch.simulate_bloch", ("echo_steps", "rf_events")),
    ("dmipy_sim.engine.bloch._simulate_bloch_mt", ("echo_steps", "rf_events")),
    ("dmipy_sim.replay.replay.replay", ("refocus_time",)),
    ("dmipy_sim.replay.replay.replay_bloch", ("echo_steps", "rf_events")),
    ("dmipy_sim.replay.replay.pose_response", ("refocus_time",)),
    ("dmipy_sim.replay.replay._pose_coeffs", ("refocus_time",)),
    ("dmipy_sim.replay.bank.replay_susc", ("refocus_time",)),
    ("dmipy_sim.replay.phantom.replay", ("refocus_time",)),
    ("dmipy_sim.replay.phantom.replay_bloch", ("rf_events",)),
    ("dmipy_sim.replay.phantom._apply_layers", ("refocus_time",)),
    ("dmipy_sim.replay.phantom._static_spin_rf", ("rf_events",)),
}

# §1.3 -- the scanner catalogues: ONE source (the cited JSON) and the views derived from it at import
_SCANNER_SOURCE = {"dmipy_sim.acquisition.scanner_constants.SCANNER_CONSTANTS"}
_SCANNER_VIEWS = {"dmipy_sim.acquisition.scanners.SCANNERS", "dmipy_sim.sequences.pulseq.PULSEQ_SYSTEMS"}


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
    for py in sorted(_SRC.rglob("*.py")):
        rel = py.relative_to(_SRC.parent).with_suffix("")
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
    from dmipy_sim.acquisition.rf import RFEvent, RFSchedule
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
    import dmipy_sim.acquisition.waveforms as W
    import dmipy_sim.sequences.pulseq as Q
    import dmipy_sim.acquisition.rf as RF
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
            "G_magnitude", "refocus_duration", "kind='sine'")                        # piece 5b: one set of builders
    found = {}
    for py, modname in _package_modules():
        text = py.read_text()
        hits = [g for g in gone if g in text]
        if hits:
            found[modname] = hits
    assert not found, f"retired names back in the package: {found}"


def test_rf_is_taken_apart_from_the_gradient_only_where_declared():
    """A callable that takes an acquisition AND separately its RF schedule / echo / refocus time is a side
    channel: the two can then disagree by signature (#172 is one such bug). The declared set is where that
    still happens; #173 piece 6 empties it."""
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
