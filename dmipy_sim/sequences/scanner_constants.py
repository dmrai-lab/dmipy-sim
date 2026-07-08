"""MR scanner DELIVERABILITY constants with full citation provenance.

The hardware + safety limits that bound a *deliverable* acquisition — gradient
(max amplitude, slew, raster), RF (peak B1, raster), and safety (IEC SAR / B1+rms /
PNS) — per vendor and model.  This is the acquisition-side analogue of
:mod:`dmipy_sim.substrate.biophysical_constants`: one cited source for every
hardware number, so the NOW gradient designer and the RF co-optimizer pull their
limits from here rather than hard-coding them.

The data lives in ``scanner_constants.json`` beside this module (human-readable,
diffable); this module loads it and adds typed accessors.

Schema
------
``SCANNER_CONSTANTS`` has these sections:
  * ``citations``  — shared citation dicts (key, authors, title, publisher, year, doi_or_url).
  * ``scanners``   — keyed by model; each has ``gradient`` / ``rf`` sub-dicts of entries.
  * ``safety``     — IEC 60601-2-33 SAR / B1+rms / dB-dt-PNS / SAFE-model (field-independent).
Each leaf entry carries ``value``, ``unit``, ``field_T`` (or null), ``context``,
``source_key``, ``location`` (the specific clause/table/figure), and ``confidence``
(``cited`` / ``widely-quoted`` / ``NEEDS VERIFICATION``).  ``value`` may be ``null``
for a NEEDS-VERIFICATION entry (vendor-confidential / coil-dependent).

Caveats that travel with these numbers
---------------------------------------
* peak-B1 is coil- and patient-load-dependent (the GE 19 µT is "@ 75 kg");
* SAR is patient-mass-dependent and temperature-derated; B1+rms has no fixed IEC
  ceiling (it comes from the implant's MR-Conditional label);
* slew is usually PNS-limited well below the hardware max — Connectom is 200 T/m/s
  hardware but **62.5 T/m/s during diffusion encoding** (use ``regime='diffusion'``);
* Connectome-2.0 figures are published *design targets*, not production specs.

Provenance compiled 2026-06-26 by an automated literature/standards sweep; every
``source_key`` resolves in ``citations`` to a DOI / IEC clause / vendor document.
"""

import json
import warnings
from pathlib import Path

with open(Path(__file__).with_name("scanner_constants.json")) as _f:
    SCANNER_CONSTANTS = json.load(_f)

# catalogue stores convenient units; convert to SI for the solvers.
_TO_SI = {"mT/m": 1e-3, "T/m": 1.0, "T/m/s": 1.0, "us": 1e-6, "ms": 1e-3,
          "s": 1.0, "uT": 1e-6, "T": 1.0, "W/kg": 1.0}


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
    """List ``(model, group, name)`` of every entry whose value is unverified/None."""
    out = []
    for m, sc in SCANNER_CONSTANTS["scanners"].items():
        for grp in ("gradient", "rf"):
            for n, leaf in sc.get(grp, {}).items():
                if isinstance(leaf, dict) and (leaf.get("confidence") == "NEEDS VERIFICATION"
                                               or leaf.get("value") is None):
                    out.append((m, grp, n))
    return out
