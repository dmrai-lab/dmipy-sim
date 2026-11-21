"""The membrane water-exchange entries of the constants table: sourced and in their measured unit.
``kappa_membrane``'s value and its consumers are covered by ``test_default_permeability.py``.

Covers only ``kappa_membrane`` and the five exchange entries; the rest of the table is not checked here.
"""
import math
import re

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


# SI factor of each unit a paper prints, to the entry's unit (s, s^-1 or m/s).
PRINTED_UNIT_TO_SI = {
    'ms': 1e-3, 's': 1.0, 's^-1': 1.0, 's -1': 1.0,
    'cm/sec': 1e-2, 'mm/s': 1e-3, 'μm/ms': 1e-3, 'μm/s': 1e-6, 'm/s': 1.0,
}
_NUM = r'(\d+(?:\.\d+)?)'


def _parse_printed(text):
    """(low, high, unit) of a printed value: the central number (a range's ends), times any
    ``x 10^k``, in the printed unit; uncertainties (``± u``, ``(u)``, ``[lo, hi]`` after the number)
    and trailing context are ignored."""
    t = (text.replace('−', '-').replace('–', '-').replace('×', 'x').replace('∼', '')
         .replace('µ', 'μ'))
    t = re.sub(r'^\s*P\s+', '', t).strip()
    m = re.match(rf'\[?\s*{_NUM}(?:\s*-\s*{_NUM})?\s*\]?', t)
    assert m, f'no number in printed {text!r}'
    lo = float(m.group(1))
    hi = float(m.group(2)) if m.group(2) else lo
    rest = re.sub(r'^\s*(±\s*[\d.]+|\([\d.]+\)|\[[^\]]*\])', '', t[m.end():])
    e = re.match(r'\s*x\s*10\^\s*(-?\d+)', rest)
    if e:
        lo, hi = lo * 10.0 ** int(e.group(1)), hi * 10.0 ** int(e.group(1))
        rest = rest[e.end():]
    units = sorted(PRINTED_UNIT_TO_SI, key=len, reverse=True)
    u = re.match(r'\s*(' + '|'.join(re.escape(x) for x in units) + r')(?![A-Za-z])', rest)
    assert u, f'no known unit in printed {text!r}'
    return lo, hi, u.group(1)


def _printed_sources(name):
    for where, src in _sources(name):
        yield where, src
        if src.get('derived_permeability'):
            yield where + '.derived_permeability', src['derived_permeability']


@pytest.mark.parametrize('name', sorted(ENTRIES))
def test_value_is_the_printed_number_in_si(name):
    for where, src in _printed_sources(name):
        if 'printed' not in src:
            # only a value no paper prints: an unverified source, or a permeability derived by this
            # table (kappa_membrane's default among them)
            assert (src.get('value_in_cited_paper') is False or src.get('location') == NEEDS
                    or src.get('by', '').startswith('this table')), f'{name} {where}: no printed value'
            continue
        lo, hi, unit = _parse_printed(src['printed'])
        f = PRINTED_UNIT_TO_SI[unit]
        if src.get('value') is not None:
            assert lo == hi, f'{name} {where}: a printed range stored as one value'
            assert math.isclose(src['value'], lo * f, rel_tol=1e-9), \
                f'{name} {where}: value {src["value"]} != printed {src["printed"]!r} in SI ({lo * f})'
        else:
            r_lo, r_hi = src['range']
            assert math.isclose(r_lo, lo * f, rel_tol=1e-9) and math.isclose(r_hi, hi * f, rel_tol=1e-9), \
                f'{name} {where}: range {src["range"]} != printed {src["printed"]!r} in SI'


def test_the_printed_parser_reads_every_form_used():
    assert _parse_printed('42.3 [40.0, 44.7] ms (mean)') == (42.3, 42.3, 'ms')
    assert _parse_printed('110 ± 20 s −1 at 25° C') == (110.0, 110.0, 's -1')
    assert _parse_printed('0.57(0.05) s') == (0.57, 0.57, 's')
    assert _parse_printed('3.8–4.5 ms') == (3.8, 4.5, 'ms')
    lo, hi, unit = _parse_printed('P ∼ [2.1 − 33] × 10^−3 μm/ms')
    assert unit == 'μm/ms' and math.isclose(lo, 2.1e-3) and math.isclose(hi, 33e-3)
    lo, _, unit = _parse_printed('2.5 ± 0.15 x 10^-4 cm/sec (4 axons)')
    assert unit == 'cm/sec' and math.isclose(lo, 2.5e-4)
