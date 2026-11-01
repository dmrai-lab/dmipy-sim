"""Every test directory runs in CI: a directory of tests that no group of ``.github/workflows/tests.yml`` names is a
directory whose failures nothing reports (dmipy-sim#848)."""
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def test_every_test_directory_is_in_a_ci_group():
    workflow = ROOT / ".github" / "workflows" / "tests.yml"
    if not workflow.exists():
        return                                              # an installed copy of the tests has no workflow
    text = workflow.read_text()
    named = set(re.findall(r"tests/([a-z_]+)", text))
    dirs = {p.parent.name for p in (ROOT / "tests").glob("*/test_*.py")}
    assert dirs, "no test directory found"
    missing = sorted(dirs - named)
    assert not missing, f"test directories in no CI group of tests.yml: {missing}"
