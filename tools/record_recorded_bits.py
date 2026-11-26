"""Print the canary and hashes of the reference walks as an entry of ``FAMILIES`` in ``tests/recorded_bits.py``.
With ``--fixed-step-default CANARY``, print only a JSON entry of ``FIXED_STEP_DEFAULT`` keyed by the head's canary.
Run that mode on the pre-calibration simulator revision (d2867a9) on the same CPU to record the baseline.

The values are those of jax / jaxlib ``RECORDED_WITH`` on the CPU (``JAX_PLATFORMS=cpu``); the script refuses any
other build or platform (exit 2). The entry is copied from its output, never edited by hand. The canary only selects
the family: the recorded tests, run afterwards on a host of the family, are what show its bits match.

Run from the repository root, with the recorded build installed::

    JAX_PLATFORMS=cpu python tools/record_recorded_bits.py
"""
import argparse
import json
import os
import pprint
import sys
from pathlib import Path
from tempfile import TemporaryDirectory

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--fixed-step-default", metavar="CANARY",
                        help="emit the fixed-step baseline for the head's canary from the base checkout")
    args = parser.parse_args()

    import jax
    import jaxlib
    from tests.recorded_bits import RECORDED_WITH, canary, fixed_step_default_hashes
    build = (jax.__version__, jaxlib.__version__)
    if build != RECORDED_WITH:
        print(f"not the recorded build: jax / jaxlib {build}, the recorded build is {RECORDED_WITH}")
        return 2
    if os.environ.get("JAX_PLATFORMS") != "cpu":
        print(f"not the recorded platform: JAX_PLATFORMS={os.environ.get('JAX_PLATFORMS')!r}, the recorded is 'cpu'")
        return 2
    if args.fixed_step_default is not None:
        with TemporaryDirectory() as directory:
            baseline = fixed_step_default_hashes(Path(directory) / "default.walk")
        print("# the entry of this host's family in tests/recorded_bits.py FIXED_STEP_DEFAULT", file=sys.stderr)
        print(json.dumps({args.fixed_step_default: baseline}, indent=4))
    else:
        from tests.geometry.test_mixed_myelinated_plain_cell import RECORDED_WALKS, recorded_hashes
        values = {name: recorded_hashes(name) for name in RECORDED_WALKS}
        print("# the entry of this host's family in tests/recorded_bits.py FAMILIES (canary first)")
        print(pprint.pformat({canary(): values}, width=120, sort_dicts=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
