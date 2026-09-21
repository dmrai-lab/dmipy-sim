"""MR scanner DELIVERABILITY constants with full citation provenance.

The hardware + safety limits that bound a *deliverable* acquisition — gradient
(max amplitude, slew, raster), RF (peak B1, raster), and safety (IEC SAR / B1+rms /
PNS) — per vendor and model.  This is the acquisition-side analogue of
:mod:`dmipy_sim.substrate.biophysical_constants`: **the one home of every scanner
number in the ecosystem**. :class:`dmipy_sim.acquisition.scanners.ScannerLimits` is the
typed view a consumer takes; the certificate's class table ``SCANNERS`` and the Pulseq
``PULSEQ_SYSTEMS`` presets are derived from here at import, never written by hand; the
NOW gradient designer and the RF co-optimizer read their limits through them.

The data lives in ``scanner_constants.json`` beside this module (human-readable,
diffable); this module loads it and adds typed accessors.

Schema
------
``SCANNER_CONSTANTS`` has these sections:
  * ``citations``  — shared citation dicts (key, authors, title, publisher, year, doi_or_url).
  * ``scanners``   — real machines, keyed by model; each has ``gradient`` / ``rf`` sub-dicts of entries.
  * ``envelopes``  — declared limit points that are not a machine: the Pulseq presets and the
    Pulseq example system's dead times. Same leaf schema; ``NEEDS VERIFICATION`` where nothing
    cites them, which is the honest state.
  * ``classes``    — the replay band-limit certificate's short names (``prisma``, ``connectom``,
    ``connectome_2``, ``magnus``, ``bruker_bga_s``, ``micro_insert``, ``extreme_insert``) -> a
    ``scanners`` key: every class is a cited machine (#220), the three preclinical ones Bruker's
    BGA-9S, Micro2.5 and Micro5.
  * ``aliases``    — every other short name (the Pulseq preset names, ``signa_magnus`` ...) -> a key.
  * ``safety``     — IEC 60601-2-33 SAR / B1+rms / dB-dt-PNS / SAFE-model (field-independent).
Each leaf entry carries ``value``, ``unit``, ``field_T`` (or null), ``context``,
``source_key``, ``location`` (the specific clause/table/figure), and ``confidence``
(``cited`` / ``derived`` / ``widely-quoted`` / ``NEEDS VERIFICATION``).  ``derived`` is a
number computed from a cited one (a microscopy probe's slew rate from its stated rise time
and amplitude: a lower bound, said so in ``context``).  ``value`` may be ``null`` for a
NEEDS-VERIFICATION entry (vendor-confidential / coil-dependent).

Caveats that travel with these numbers
---------------------------------------
* peak-B1 is coil- and patient-load-dependent (the GE 19 µT is "@ 75 kg");
* SAR is patient-mass-dependent and temperature-derated; B1+rms has no fixed IEC
  ceiling (it comes from the implant's MR-Conditional label);
* slew is usually PNS-limited well below the hardware max — Connectom is 200 T/m/s
  hardware but **62.5 T/m/s during diffusion encoding** (use ``regime='diffusion'``);
* Connectome-2.0 figures were published as design targets and are reached per axis on the
  built scanner (Ramos-Llorden 2026); the Bruker microscopy probes' slew rates are derived
  from rise times, not published.

Provenance compiled 2026-06-26 by an automated literature/standards sweep; every
``source_key`` resolves in ``citations`` to a DOI / IEC clause / vendor document.
"""

import json
from pathlib import Path

with open(Path(__file__).with_name("scanner_constants.json")) as _f:
    SCANNER_CONSTANTS = json.load(_f)

# catalogue stores convenient units; convert to SI for the solvers.
_TO_SI = {"mT/m": 1e-3, "T/m": 1.0, "T/m/s": 1.0, "us": 1e-6, "ms": 1e-3,
          "s": 1.0, "uT": 1e-6, "T": 1.0, "W/kg": 1.0,
          # a field SHAPE is a fraction of B0, so ppm carries the 1e-6 and nothing else
          "ppm": 1e-6, "ppm/m": 1e-6, "ppm/m^2": 1e-6, "m": 1.0,
          # descriptive leaves nothing reads in SI yet, listed so the conformance check passes
          # rather than so they are used: an unlisted unit is the silent 1.0 conversion
          "cm": 1e-2, "kW": 1e3, "MW": 1e6,
          # a transmit scale is dimensionless, and its fall-off is per square metre
          "": 1.0, "1/m^2": 1.0,
          # a temperature coefficient is a fraction per kelvin; a temperature SPAN is kelvin either way,
          # which is why the catalogue stores a span and not a degC endpoint -- degC to K is an offset,
          # and this table can only scale
          "1/K": 1.0, "K": 1.0, "Hz/K": 1.0,
          # a frequency offset is already SI; it is catalogued in Hz rather than converted to ppm
          # because the measurement is a frequency and the ppm depends on which B0 you divide by
          "Hz": 1.0,
          # a solid-harmonic coefficient of dB/B0 has the reciprocal length of its order
          "1/m^3": 1.0,
          # a gradient-nonlinearity coefficient is a fraction per metre
          "1/m": 1.0}


def resolve(name):
    """``(key, entry, kind)`` of a scanner given ANY of its names: a certificate class, an alias, a
    ``scanners`` model key or an ``envelopes`` key, case-insensitively. ``kind`` is ``"scanner"`` or
    ``"envelope"``. Unknown names raise ``ValueError`` listing every accepted name."""
    n = str(name).lower()
    short = {k.lower(): v for k, v in {**SCANNER_CONSTANTS["classes"], **SCANNER_CONSTANTS["aliases"]}.items()}
    key = short.get(n, n).lower()
    for kind in ("scanner", "envelope"):
        table = SCANNER_CONSTANTS[kind + "s"]
        by_lower = {k.lower(): k for k in table}
        if key in by_lower:
            return by_lower[key], table[by_lower[key]], kind
    raise ValueError(f"unknown scanner {name!r}; known: classes {sorted(SCANNER_CONSTANTS['classes'])}, "
                     f"aliases {sorted(SCANNER_CONSTANTS['aliases'])}, models {sorted(SCANNER_CONSTANTS['scanners'])}, "
                     f"envelopes {sorted(SCANNER_CONSTANTS['envelopes'])}")


def leaf_raw(entry, group, name):
    """The leaf's value EXACTLY as catalogued, with no unit conversion -- for leaves whose value is not a
    number. An axis letter is the case this exists for: ``leaf_si`` would try to float it."""
    lf = entry.get(group, {}).get(name)
    return None if lf is None else lf.get("value")


def leaf_si(entry, group, name):
    """The SI value of ``entry[group][name]``, or ``None`` when the leaf is absent or unverified (its
    ``value`` is null). The typed :class:`~dmipy_sim.acquisition.scanners.ScannerLimits` carries ``None``
    for what is not known rather than a number for it."""
    lf = entry.get(group, {}).get(name)
    if not isinstance(lf, dict) or lf.get("value") is None:
        return None
    return float(lf["value"]) * _TO_SI.get(lf.get("unit"), 1.0)


def list_scanners():
    """Available scanner model keys."""
    return list(SCANNER_CONSTANTS["scanners"].keys())


def get_scanner(model):
    """Return the full entry for a scanner model."""
    sc = SCANNER_CONSTANTS["scanners"]
    if model not in sc:
        raise KeyError(f"Unknown scanner '{model}'. Available: {list(sc.keys())}")
    return sc[model]


def get_limit(model, group, name, *, si=False):
    """Return a leaf entry (``si=False``) or its SI-converted scalar (``si=True``).

    ``group`` in {'gradient','rf'}; ``name`` e.g. 'max_amplitude','max_slew_rate',
    'rf_raster_time','peak_B1_body_coil'.  With ``si=True`` the value is converted to
    SI (mT/m→T/m, us→s, uT→T) and a ``None`` value (NEEDS VERIFICATION) RAISES, so a
    missing hardware number can never silently enter a design as 0/None.
    """
    entry = get_scanner(model).get(group, {})
    if name not in entry:
        raise KeyError(f"'{group}/{name}' not in '{model}'. Have: {list(entry.keys())}")
    leaf = entry[name]
    if not si:
        return leaf
    v = leaf.get("value")
    if v is None:
        raise ValueError(
            f"get_limit('{model}','{group}','{name}', si=True): value is None "
            f"(confidence={leaf.get('confidence')!r}). Supply a cited number or read "
            f"the leaf's 'confidence' and handle it explicitly.")
    return v * _TO_SI.get(leaf.get("unit"), 1.0)


def gradient_limits(model, *, regime="default"):
    """Convenience ``(G_max [T/m], slew_max [T/m/s])`` for the NOW designer.

    ``regime='diffusion'`` prefers a PNS-derated diffusion slew (``max_slew_rate_diffusion``)
    when the model defines one (e.g. Connectom 200→62.5 T/m/s) — the limit that actually
    binds a diffusion sequence — falling back to the hardware ``max_slew_rate`` otherwise.
    """
    g_max = get_limit(model, "gradient", "max_amplitude", si=True)
    grad = get_scanner(model).get("gradient", {})
    slew_name = ("max_slew_rate_diffusion"
                 if regime == "diffusion" and "max_slew_rate_diffusion" in grad
                 else "max_slew_rate")
    return g_max, get_limit(model, "gradient", slew_name, si=True)


def sar_limit(region="whole_body", mode="normal"):
    """IEC 60601-2-33 SAR limit (W/kg). ``region`` in {whole_body,head,local_head,...};
    ``mode`` in {normal, first_level}."""
    key = f"{region}_{mode}"
    sar = SCANNER_CONSTANTS["safety"]["sar"]
    if key not in sar:
        raise KeyError(f"No SAR limit '{key}'. Have: "
                       f"{[k for k in sar if sar[k].get('unit') == 'W/kg']}")
    return sar[key]["value"]


def get_citation(source_key):
    """Return the citation dict for a ``source_key``."""
    return SCANNER_CONSTANTS["citations"][source_key]


def needs_verification():
    """List ``(model, group, name)`` of every leaf, in every group, whose confidence is ``NEEDS VERIFICATION``.

    A null value is not by itself a gap: a null under a stated confidence is a CLAIM that there is no such
    number (a quadrature birdcage has no single ``b1_axis``), and it is the confidence that says which.
    """
    out = []
    for m, sc in {**SCANNER_CONSTANTS["scanners"], **SCANNER_CONSTANTS["envelopes"]}.items():
        for grp, leaves in sc.items():
            if not isinstance(leaves, dict):
                continue
            for n, leaf in leaves.items():
                if isinstance(leaf, dict) and leaf.get("confidence") == "NEEDS VERIFICATION":
                    out.append((m, grp, n))
    return out


SCHEMA_PATH = Path(__file__).with_name("scanner_catalogue.schema.json")


def conformance_problems(catalogue=None):
    """Every way the catalogue departs from its own schema (ACQUISITION.md 8), as a list of sentences;
    empty when it conforms. Checked here rather than with ``jsonschema`` so the package stays a test-time
    convenience, exactly as the substrate spec's validator does.

    The three referential rules are the ones a type schema cannot state: every leaf's ``source_key``
    resolves in ``citations``; a ``classes`` or ``aliases`` value names a key of ``scanners`` OR of
    ``envelopes``; and a citation MAY be referenced from an entry's prose ``notes`` alone, so an uncited
    citation is not an error while an unresolved key is.
    """
    cat = SCANNER_CONSTANTS if catalogue is None else catalogue
    schema = json.loads(SCHEMA_PATH.read_text())
    fields = tuple(cat["_schema"]["entry_fields"])
    levels = set(cat["_schema"]["confidence_levels"])
    cites = set(cat.get("citations", {}))
    bad = []
    for table in ("scanners", "envelopes"):
        for name, entry in cat.get(table, {}).items():
            for group, leaves in entry.items():
                if not isinstance(leaves, dict):
                    continue
                for leaf_name, leaf in leaves.items():
                    if not isinstance(leaf, dict) or "value" not in leaf:
                        continue
                    where = f"{table}.{name}.{group}.{leaf_name}"
                    missing = [f for f in fields if f not in leaf]
                    if missing:
                        bad.append(f"{where} is missing {missing}")
                    if leaf.get("source_key") not in cites:
                        bad.append(f"{where} cites {leaf.get('source_key')!r}, which is not in citations")
                    if leaf.get("confidence") not in levels:
                        bad.append(f"{where} has confidence {leaf.get('confidence')!r}, "
                                   f"which is not one of {sorted(levels)}")
                    if not leaf.get("context"):
                        bad.append(f"{where} has no context: a bare number loses what it means")
                    unit = leaf.get("unit")
                    if leaf.get("value") is not None and unit is not None and unit not in _TO_SI:
                        bad.append(f"{where} is in {unit!r}, which _TO_SI does not know: leaf_si would "
                                   f"convert it by 1.0 and say nothing")
    known = set(cat.get("scanners", {})) | set(cat.get("envelopes", {}))
    for table in ("classes", "aliases"):
        for short, target in cat.get(table, {}).items():
            if target not in known:
                bad.append(f"{table}.{short} points at {target!r}, which is neither a scanner nor an envelope")
    if schema.get("title", "").split()[0] != "Scanner":
        bad.append("the shipped schema is not the scanner catalogue's")
    return bad
