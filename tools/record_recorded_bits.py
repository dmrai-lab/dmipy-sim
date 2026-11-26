"""Print the canary and hashes of every recorded walk as an entry of ``FAMILIES`` in ``tests/recorded_bits.py``.
Print the default fixed-step field walk separately as an entry of ``FIXED_STEP_DEFAULT``. Run on the pre-calibration
simulator revision (d2867a9) to record that baseline.

The values are those of jax / jaxlib ``RECORDED_WITH`` on the CPU (``JAX_PLATFORMS=cpu``); the script refuses any
other build or platform (exit 2). The entry is copied from its output, never edited by hand. The canary only selects
the family: the recorded tests, run afterwards on a host of the family, are what show its bits match.

Run from the repository root, with the recorded build installed::

    JAX_PLATFORMS=cpu python tools/record_recorded_bits.py
"""
import os
import pprint
import sys
from pathlib import Path
from tempfile import TemporaryDirectory

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def main():
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
    from tests.geometry.test_mixed_myelinated_plain_cell import RECORDED_WALKS, recorded_hashes
    canary_hash = canary()
    values = {name: recorded_hashes(name) for name in RECORDED_WALKS}
    with TemporaryDirectory() as directory:
        baseline = fixed_step_default_hashes(Path(directory) / "default.walk")
    print("# the entry of this host's family in tests/recorded_bits.py FAMILIES (canary first)")
    print(pprint.pformat({canary_hash: values}, width=120, sort_dicts=False))
    print("# the entry of this host's family in tests/recorded_bits.py FIXED_STEP_DEFAULT")
    print(pprint.pformat({canary_hash: baseline}, width=120, sort_dicts=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
