"""The physical forward-truth substrate (sim-owned).

* :class:`Substrate` -- the white-matter physical ground truth (geometry,
  diffusivity, relaxation, surface, water content, susceptibility amplitude).
* :class:`GreyMatterSubstrate` -- the cortical grey-matter physical ground truth
  (packed somata + membrane physics + glial-iron static susceptibility field).
* biophysical_constants -- the cited constant catalogue + ``canonical_white_matter``
  (the single source for every physical constant; dmipy-fit re-exports from here).
"""
from .substrate import Substrate, mean_inv_diameter_4
from .grey_matter import GreyMatterSubstrate
from .biophysical_constants import (
    BIOPHYSICAL_CONSTANTS, canonical_white_matter, canonical_grey_matter,
    get_constant, get_default_value, get_value,
)

__all__ = [
    'Substrate', 'mean_inv_diameter_4',
    'GreyMatterSubstrate',
    'BIOPHYSICAL_CONSTANTS', 'canonical_white_matter', 'canonical_grey_matter',
    'get_constant', 'get_default_value', 'get_value',
]
