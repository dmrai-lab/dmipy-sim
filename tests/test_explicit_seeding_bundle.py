"""Explicit seeding on a multi-surface spec: the cited start list is read for each seeded pool inside the per-pool
walk loop, so a spec with several walls and pools walks from it (dmipy-sim#712)."""
import hashlib

import pytest

from dmipy_sim.spec import Seeding, SpecError
from dmipy_sim import walk
from tests.conftest import spec_without_source


def _explicit(spec, tmp, rows, pools):
    path = tmp / "starts.xyz"
    path.write_text("".join(f"{x} {y} {z}\n" for x, y, z in rows))
    positions = dict(file=str(path), format="xyz", scale=1.0, sha256=hashlib.sha256(path.read_bytes()).hexdigest(), count=len(rows), read="cyclic")
    return spec.replace(seeding=Seeding(list(pools), "explicit", "water_fraction", positions=positions))


def test_explicit_seeds_walk_a_multi_surface_spec(spec_grid):
    spec, _, tmp = spec_grid
    spec = spec_without_source(spec)
    assert len(spec.walls) == 2 and len(spec.pools) == 3
    intra = next(p.id for p in spec.pools if p.name == "intra")
    rows = [(10e-6, 10.2e-6, z) for z in (8e-6, 10e-6, 12e-6)]
    w = walk(_explicit(spec, tmp, rows, [intra]), 7, 4e-4, 2e-4, seed=0, require_gpu=False)
    assert w.n_walkers == 7                                   # the list is read cyclically: walker i starts at row i mod 3


def test_explicit_seeds_outside_their_pool_are_refused(spec_grid):
    spec, _, tmp = spec_grid
    spec = spec_without_source(spec)
    extra = next(p.id for p in spec.pools if p.name == "extra")
    rows = [(10e-6, 10.2e-6, 10e-6), (10e-6, 10.2e-6, 9e-6)]                        # inside the intra pool, listed for the extra pool
    with pytest.raises(SpecError, match="not inside pool"):
        walk(_explicit(spec, tmp, rows, [extra]), 4, 4e-4, 2e-4, seed=0, require_gpu=False)
