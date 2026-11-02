"""The substrate specification: the situation a walker is in, as data (SUBSTRATE.md in replay-pack-spec).

`SubstrateSpec` is the only input the engine will accept for a walk (issue #130); a pack embeds it.
"""
from .build import spec_of, geometry_from_spec, as_geometry
from .walk import draw_seeds, WalkContext, field_grid_of_spec, field_raster_plan, check_field_raster
from .seeding import StratifiedByVoxel, DrawnSeeds, plan_seeding
from .producers import (cactus_spec, winther_spec, caterpillar_spec, label_volume_spec, strands_spec,
                        disco_spec, mcdc_axon_spec, wm_pools, packed_request, RSA_LIMIT)
from .preview import preview

__all__ = ["StratifiedByVoxel", "DrawnSeeds", "draw_seeds", "WalkContext", "field_grid_of_spec", "field_raster_plan", "check_field_raster", "plan_seeding", "preview", "spec_of", "geometry_from_spec", "as_geometry", "cactus_spec", "winther_spec", "caterpillar_spec", "label_volume_spec", "strands_spec", "disco_spec", "mcdc_axon_spec", "wm_pools", "packed_request", "RSA_LIMIT"]


def __getattr__(name):
    from .._moved import refuse
    raise refuse(__name__, name)
