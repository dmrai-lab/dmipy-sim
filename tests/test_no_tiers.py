"""There are two tiers, by hardware: the regular suite on a CPU runner, and the tests marked ``gpu`` on a GPU
(``.github/workflows/gpu-tier.yml``). ``gpu`` is the only marker. A test that is merely slow is slimmed (a closed form, a
walker count derived from the standard error it must resolve, a shared fixture, a smaller substrate) or deleted; it is
not parked behind a marker. This file fails when a ``slow`` marker comes back, when the CPU job selects on anything
but ``gpu``, or when the GPU tier has no workflow."""
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SLOW = re.compile(r"mark\.slow\b|-m\s+[\"']?[^\"'\n]*\bslow\b|^\s*\"slow:|`slow`")


def _sources():
    for p in sorted((ROOT / "tests").rglob("*.py")):
        if p.name != "test_no_tiers.py":
            yield p
    yield from sorted((ROOT / ".github" / "workflows").glob("*.yml"))
    yield ROOT / "pyproject.toml"


def test_there_is_no_slow_marker_anywhere():
    found = [f"{p.relative_to(ROOT)}:{i}: {line.strip()}" for p in _sources()
             for i, line in enumerate(p.read_text().splitlines(), 1) if SLOW.search(line)]
    assert not found, "\n".join(found)


def test_gpu_is_the_only_registered_marker_and_the_cpu_job_deselects_it():
    toml = (ROOT / "pyproject.toml").read_text()
    registered = re.findall(r'^\s*"(\w+):', toml.split("markers = [", 1)[1].split("]", 1)[0], re.M)
    assert registered == ["gpu"], registered
    tests = (ROOT / ".github" / "workflows" / "tests.yml").read_text()
    assert tests.count('-m "not gpu"') >= 2 and "slow" not in tests


def test_the_gpu_tier_has_its_workflow_and_a_budget_guard():
    wf = (ROOT / ".github" / "workflows" / "gpu-tier.yml").read_text()
    assert "MODAL_TOKEN_ID" in wf and "tools/gpu_tier.py" in wf and "workflow_dispatch" in wf
    assert "dmipy_sim/engine/**" in wf and "dmipy_sim/replay/**" in wf
