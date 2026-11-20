"""The membrane water-exchange entries of the constants table: sourced, in their measured unit, and
``kappa_membrane``'s value unchanged for its consumers.

Covers only ``kappa_membrane`` and the five exchange entries; the rest of the table is not checked here.
"""
import math

import pytest

from dmipy_sim.substrate import biophysical_constants as bc

EXCHANGE_ENTRIES = {
    'exchange_time_grey_matter': 's',
    'exchange_rate_grey_matter': 's^-1',
    'exchange_rate_white_matter': 's^-1',
    'intracellular_residence_time': 's',
    'kappa_membrane_direct': 'm/s',
}
ENTRIES = {'kappa_membrane': 'm/s', **EXCHANGE_ENTRIES}
UNITS = {'s', 's^-1', 'm/s'}
NEEDS = 'NEEDS VERIFICATION'
# The one exemption (its source's full text was not reached): the default's location quotes the
# abstract, labelled as such, and the value is flagged as not printed by the cited paper.
ABSTRACT_ONLY_DEFAULTS = {'kappa_membrane'}


def _sources(name):
    entry = bc.get_constant(name)
    return [('default', entry['default'])] + [(f'alternatives[{i}]', a)
                                              for i, a in enumerate(entry['alternatives'])]


@pytest.mark.parametrize('name', sorted(ENTRIES))
def test_every_source_has_a_location_and_a_doi(name):
    for where, src in _sources(name):
        loc = src.get('location', '')
        assert isinstance(loc, str) and loc.strip(), f'{name} {where}: empty location'
        if loc != NEEDS:
            assert src.get('doi', '').startswith('10.'), f'{name} {where}: no DOI'
        else:
            assert src.get('value') is None, f'{name} {where}: an unverified source carries a value'


@pytest.mark.parametrize('name', sorted(ENTRIES))
def test_no_default_needs_verification(name):
    default = bc.get_constant(name)['default']
    assert NEEDS not in default['location'], name
    assert isinstance(default['value'], float) and math.isfinite(default['value']), name
    if name in ABSTRACT_ONLY_DEFAULTS:
        assert default['location'].startswith('ABSTRACT ONLY'), name
        assert default['value_in_cited_paper'] is False, name
    else:
        assert not default['location'].startswith('ABSTRACT ONLY'), name


@pytest.mark.parametrize('name', sorted(ENTRIES))
def test_units_are_the_measured_quantity(name):
    for where, src in _sources(name):
        assert src['unit'] in UNITS, f'{name} {where}: unit {src["unit"]!r}'
        assert src['unit'] == ENTRIES[name], f'{name} {where}: {src["unit"]!r} is not this entry\'s quantity'
        if src.get('value') is None and src['location'] != NEEDS:
            lo, hi = src['range']
            assert 0 < lo < hi, f'{name} {where}: bad range'


@pytest.mark.parametrize('name', sorted(ENTRIES))
def test_derived_permeabilities_state_their_geometry(name):
    for where, src in _sources(name):
        derived = src.get('derived_permeability')
        if derived is None:
            continue
        assert derived['unit'] == 'm/s', f'{name} {where}'
        for key in ('by', 'formula', 'geometry', 'radius_m'):
            assert derived.get(key), f'{name} {where}: derived permeability without {key}'


def test_kappa_membrane_default_is_unchanged_for_its_consumers():
    assert bc.get_value('kappa_membrane') == 1e-5
    assert bc.canonical_white_matter(3.0)['kappa'] == 1e-5
