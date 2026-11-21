"""The default membrane permeability is ONE number in ONE place: the permeability the constants table
derives from Nedjati-Gilani et al. 2017's intra-axonal residence time (``intracellular_residence_time``
-> ``default`` -> ``derived_permeability`` -> ``value``).

* every consumer reads that entry when it is called (a patched entry reaches all of them);
* no permeability literal equal to the old default (1e-5 m/s) or to the table's value stands in
  the package source, and ``kappa_membrane`` stores no value of its own;
* the ``kappa_membrane`` row cites the paper the value comes from, and states the derivation.
"""
import io
import math
import pathlib
import re
import tokenize

import pytest

from dmipy_sim.substrate import Substrate
from dmipy_sim.substrate import biophysical_constants as bc

SOURCE_PATH = ('intracellular_residence_time', 'default', 'derived_permeability')
PACKAGE = pathlib.Path(bc.__file__).resolve().parents[1]

# The value the default was before this table referred to its source; no cited paper gave it.
OLD_DEFAULT = 1e-5
# A physical source line names a permeability when it holds one of these words (kappa_MT, the MT
# surface reactivity, is a different quantity and does not match: the underscore is a word character).
PERMEABILITY_WORDS = re.compile(r'\b(kappa|kappa_inner|kappa_outer|permeability|in_to_out|out_to_in)\b')

# The lines allowed to hold such a number, each with its reason (file relative to the package, and
# a substring the physical line must contain).
ALLOWED = {
    ('substrate/biophysical_constants.py', "'by': 'this table (not printed by the paper)', 'value': "):
        'the source entry itself: the one place the derived permeability is stored',
    ('engine/backends.py', '"sphere_perm"'): 'a parity case of the backends contract: a chosen input',
    ('engine/backends.py', '"cylinder_perm"'): 'a parity case of the backends contract: a chosen input',
    ('engine/backends.py', '"label_slab_perm"'): 'a parity case of the backends contract: a chosen input',
    ('engine/backends.py', '"mesh_perm"'): 'a parity case of the backends contract: a chosen input',
}


def _source_entry():
    node = bc.BIOPHYSICAL_CONSTANTS
    for key in SOURCE_PATH:
        node = node[key]
    return node


def _consumers():
    return {
        "get_value('kappa_membrane')": bc.get_value('kappa_membrane'),
        "get_default_value('kappa_membrane')": bc.get_default_value('kappa_membrane'),
        "canonical_white_matter(3.0)['kappa']": bc.canonical_white_matter(3.0)['kappa'],
        'Substrate.canonical().kappa': Substrate.canonical().kappa,
        'Substrate().kappa': Substrate().kappa,
    }


def test_every_consumer_reads_the_source_entry_when_called(monkeypatch):
    stored = _source_entry()['value']
    assert all(v == stored for v in _consumers().values()), _consumers()

    sentinel = 3.21e-6
    assert sentinel != stored
    monkeypatch.setitem(_source_entry(), 'value', sentinel)
    patched = _consumers()
    assert all(v == sentinel for v in patched.values()), patched

    monkeypatch.undo()
    assert all(v == stored for v in _consumers().values()), _consumers()


def test_reading_the_entry_never_writes_into_the_table():
    row = bc.BIOPHYSICAL_CONSTANTS['kappa_membrane']['default']
    resolved = bc.get_constant('kappa_membrane')
    assert resolved['default']['value'] == _source_entry()['value']
    assert 'value' not in row, 'get_constant wrote the resolved value into the table'
    resolved['default']['value'] = -1.0
    assert bc.get_value('kappa_membrane') == _source_entry()['value']


def test_a_row_with_a_value_and_a_reference_is_refused(monkeypatch):
    monkeypatch.setitem(bc.BIOPHYSICAL_CONSTANTS['kappa_membrane']['default'], 'value', OLD_DEFAULT)
    with pytest.raises(ValueError, match='value_from'):
        bc.get_value('kappa_membrane')


def test_kappa_membrane_stores_no_value_of_its_own():
    row = bc.BIOPHYSICAL_CONSTANTS['kappa_membrane']
    for src in [row['default'], *row['alternatives']]:
        assert not isinstance(src.get('value'), (int, float)), \
            f"kappa_membrane holds a numeric value {src.get('value')!r}; it refers to {SOURCE_PATH}"
    assert tuple(row['default']['value_from']) == SOURCE_PATH


def _offending_lines():
    """(path, row, line, number) of every NUMBER token equal to the old default or to the table's
    value (rel 1e-3) on a physical line that names a permeability, outside ``ALLOWED``."""
    targets = (OLD_DEFAULT, _source_entry()['value'])
    hits = []
    for path in sorted(PACKAGE.rglob('*.py')):
        rel = path.relative_to(PACKAGE).as_posix()
        text = path.read_text(encoding='utf-8')
        lines = text.splitlines()
        for tok in tokenize.generate_tokens(io.StringIO(text).readline):
            if tok.type != tokenize.NUMBER:
                continue
            try:
                number = float(tok.string.replace('_', ''))
            except ValueError:       # complex literals
                continue
            if not any(math.isclose(number, t, rel_tol=1e-3) for t in targets):
                continue
            line = lines[tok.start[0] - 1]
            if not PERMEABILITY_WORDS.search(line):
                continue
            if any(rel == f and s in line for f, s in ALLOWED):
                continue
            hits.append((rel, tok.start[0], line.strip(), tok.string))
    return hits


def test_no_permeability_literal_stands_for_the_default():
    hits = _offending_lines()
    assert not hits, 'a permeability literal stands in the package source:\n' + '\n'.join(
        f'  {p}:{r}: {n} in {line!r}' for p, r, line, n in hits)


def test_every_allowance_is_still_needed():
    for (rel, substring), reason in ALLOWED.items():
        text = (PACKAGE / rel).read_text(encoding='utf-8')
        assert substring in text, f'allowance for {rel} ({reason}) matches no line'


def test_the_row_cites_nedjati_gilani_2017_and_states_the_derivation():
    row = bc.BIOPHYSICAL_CONSTANTS['kappa_membrane']
    source = bc.BIOPHYSICAL_CONSTANTS['intracellular_residence_time']
    default = row['default']
    assert row['citation'] == source['citation']
    assert row['citation']['key'] == default['source_key'] == 'nedjatigilani2017'
    assert row['citation']['doi'] == default['doi'] == '10.1016/j.neuroimage.2017.02.013'

    fields = {
        'citation': repr(row['citation']), 'source_key': default['source_key'], 'doi': default['doi'],
        'location': default['location'], 'method': default['method'], 'note': row['note'],
    }
    for name, text in fields.items():
        assert 'nilsson' not in text.lower(), f'{name} still credits Nilsson'
        assert '10.1002/mrm.24395' not in text, f'{name} still carries the Nilsson DOI'

    note = row['note']
    radius_um = source['default']['derived_permeability']['radius_m'] * 1e6
    tau_s = source['default']['value']
    assert 'R / (2 tau_i)' in note
    assert f'{radius_um:g} um' in note, f'the note does not state alpha = {radius_um:g} um'
    assert f'{tau_s:g} s' in note, f'the note does not state tau_i = {tau_s:g} s'
    assert 'volume-weighted' in note and 'does not print the permeability' in note
