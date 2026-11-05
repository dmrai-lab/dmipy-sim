"""``walk(sub_steps=)`` pins the physics steps per block: the walk steps at block / sub_steps, and the walk's record says so."""
import pytest

import dmipy_sim as d
from dmipy_sim import walk
from dmipy_replay.spec.substrate import SpecError
from tests._envelope import walk_envelope

D = 2e-9


def _sphere():
    return d.Sphere(5e-6).spec


def test_walk_steps_at_the_pinned_count_and_records_it():
    spec = _sphere()
    free = walk(spec, 16, T_max=2e-3, envelope=walk_envelope(of=spec), require_gpu=False, diffusivity=D)
    n = free.sub_steps + 3
    pinned = walk(spec, 16, T_max=2e-3, envelope=walk_envelope(of=spec), require_gpu=False, diffusivity=D, sub_steps=n)
    assert pinned.sub_steps == n
    assert pinned.dt_sim == pytest.approx(free.dt_sim * free.sub_steps / n, rel=1e-12)
    assert pinned.work["step_rule"]["override"] == n


@pytest.mark.parametrize("bad", [0, -2, 1.5, True])
def test_walk_refuses_a_sub_step_count_that_is_not_a_positive_whole_number(bad):
    spec = _sphere()
    with pytest.raises(SpecError, match="sub_steps"):
        walk(spec, 8, T_max=1e-3, envelope=walk_envelope(of=spec), require_gpu=False, diffusivity=D, sub_steps=bad)


def test_walk_refuses_a_pinned_count_coarser_than_the_step_law():
    spec = _sphere()
    free = walk(spec, 8, T_max=1e-3, envelope=walk_envelope(of=spec), require_gpu=False, diffusivity=D)
    if free.sub_steps < 2:
        pytest.skip("the step law resolves to one step per block here; nothing is coarser")
    with pytest.raises(SpecError, match="coarser than the step law"):
        walk(spec, 8, T_max=1e-3, envelope=walk_envelope(of=spec), require_gpu=False, diffusivity=D,
             sub_steps=free.sub_steps - 1)
