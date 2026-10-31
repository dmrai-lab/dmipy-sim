"""Adding a scanner, and the rule that keeps every number in the engine traceable to a source.

No scanner number lives in Python. Field strengths, gradient ceilings, slew limits, rasters and dead times are
all leaves of one catalogue file, and every leaf carries its own provenance: the value, its unit, the field it
was measured at, the context that says what it actually means, and a citation key pointing at a paper in the
same file.

The reason is that a scanner constant is a claim about hardware, and a claim without a source is a number
somebody will later have to re-derive or distrust. The `context` field carries the part a bare number always
loses. The Swoop's gradient ceiling below is the WEAKEST of its three axes, because a deliverability check
that used the strongest would pass sequences the scanner cannot play.

Adding a machine is therefore an edit to the catalogue and nothing else: an entry under `scanners`, a citation
it points at, and a short name under `aliases` so users can say what they mean.

Assumes rung 17.
"""
import json
from pathlib import Path

from dmipy_sim.acquisition.scanners import ScannerLimits
import dmipy_sim.acquisition as acq

cat = json.loads((Path(acq.__file__).parent / "scanner_constants.json").read_text())
print(f"the catalogue holds {len(cat['scanners'])} machines, {len(cat['envelopes'])} declared envelopes, "
      f"{len(cat['aliases'])} short names and {len(cat['citations'])} citations")
print(f"every entry's leaves carry: {', '.join(cat['_schema']['entry_fields'])}")

entry = cat["scanners"]["hyperfine_swoop_64mT"]
leaf = entry["gradient"]["max_amplitude"]
print(f"\nthe most recently added machine, {entry['vendor']} {entry['model']}:")
print(f"  gradient ceiling {leaf['value']} {leaf['unit']} at {leaf['field_T']} T, confidence '{leaf['confidence']}'")
print(f"  context: {leaf['context']}")
src = cat["citations"][leaf["source_key"]]
first = src["authors"].split(",")[0].strip()
print(f"  source:  {first} et al., {src['journal_or_publisher']} {src['year']}, {src['doi_or_url']}")
print(f"  located: {leaf['location']}")

print("\nWhat the rest of the engine sees is an object, resolved by any of its names:")
for name in ("swoop", "hyperfine_swoop", "hyperfine_swoop_64mT"):
    s = ScannerLimits.of(name)
    print(f"  {name:20s} -> {s.name}, {s.field_T} T, {s.gradient_limits[0]*1e3:.1f} mT/m, "
          f"{s.gradient_limits[1]:.0f} T/m/s")

print("\nand an unknown name is refused with the list, rather than falling back to a default:")
try:
    ScannerLimits.of("a scanner that does not exist")
except ValueError as e:
    print(f"  {str(e).splitlines()[0][:110]}")

print("\nThe rule to keep when adding one: put the number in the catalogue with its context and its citation,")
print("and let the code read it. A constant inlined in Python is a number nobody can check and nobody dares")
print("change.")
