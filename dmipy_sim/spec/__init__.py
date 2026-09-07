"""The substrate specification: the situation a walker is in, as data (SUBSTRATE.md in replay-pack-spec).

`SubstrateSpec` is the only input the engine will accept for a walk (issue #130); a pack embeds it.
"""
from .substrate import (SubstrateSpec, Domain, Frame, Pool, Susceptibility, Surface, Wall, Directional, Sided,
                        Seeding, Validity, SpecError, validate, load_spec, SCHEMA_PATH, SPEC_VERSION)

__all__ = ["SubstrateSpec", "Domain", "Frame", "Pool", "Susceptibility", "Surface", "Wall", "Directional", "Sided",
           "Seeding", "Validity", "SpecError", "validate", "load_spec", "SCHEMA_PATH", "SPEC_VERSION"]
