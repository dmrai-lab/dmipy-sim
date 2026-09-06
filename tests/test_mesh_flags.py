"""The Mesh collision-response flags are constructor arguments with the validated defaults, so a
geometry's behaviour is fixed at construction and readable from the call that built it."""
import inspect

import numpy as np

import dmipy_sim as d
from dmipy_sim.geometry import mesh_shapes


def test_flags_are_kwargs_with_the_validated_defaults():
    sig = inspect.signature(d.Mesh.__init__)
    assert (sig.parameters["reject_escape"].default, sig.parameters["box_reflect"].default,
            sig.parameters["adaptive_nudge"].default) == (True, True, False)
    V, F = mesh_shapes.icosphere(2e-6, subdivisions=1)
    m = d.Mesh(V, F, feature_radius=1e-6)
    assert (m.reject_escape, m.box_reflect, m.adaptive_nudge) == (True, True, False)
    m2 = d.Mesh(V, F, feature_radius=1e-6, reject_escape=False, adaptive_nudge=True)
    assert (m2.reject_escape, m2.box_reflect, m2.adaptive_nudge) == (False, True, True)


def test_a_flag_is_part_of_the_compiled_program_key():
    """Two meshes that differ only in a flag must not share a batch program (core.cached_batch keys
    on the geometry's scalar state, which the flags are part of)."""
    from dmipy_sim import core
    V, F = mesh_shapes.icosphere(2e-6, subdivisions=1)
    a = d.Mesh(V, F, feature_radius=1e-6)
    b = d.Mesh(V, F, feature_radius=1e-6, reject_escape=False)
    assert core._geometry_state(a) != core._geometry_state(b)
