"""CI hang diagnostics: a stuck test must name itself and end its job.

The ``Tests`` workflow runs every test with a per-test time limit given on its command line
(``-o faulthandler_timeout=... -o faulthandler_exit_on_timeout=true``). pytest's faulthandler
dumps the stack of every thread when a test exceeds the limit and then exits the whole process
with a non-zero status, so a test that hangs is named in the log and the job ends in minutes
instead of hours. The workflow prints each test's name as it finishes (the ``-v`` of
``addopts``, with no ``-q`` on the command line), and every job has a ``timeout-minutes`` as a
last resort. The per-test limit is a CI command-line option only, never in the project
config, so a local run is never limited by it.
"""
import os
import re
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
WORKFLOW = ROOT / ".github" / "workflows" / "tests.yml"


def test_a_test_over_the_limit_is_dumped_and_ends_the_process(tmp_path):
    """A test that sleeps past a 2 s limit has its stack dumped and the process ends at once.

    Runs one sleeping test in a subprocess with a 2 s per-test limit and the exit-on-timeout
    option, and checks the dump text, the non-zero exit status and the wall time (the process
    must end at the limit, not wait out the 60 s sleep nor the 45 s subprocess timeout).
    """
    test_file = tmp_path / "test_sleeps.py"
    test_file.write_text("import time\n\n\ndef test_sleeps_forever():\n    time.sleep(60)\n")
    env = {key: value for key, value in os.environ.items() if key != "JAX_ENABLE_X64"}
    env["JAX_PLATFORMS"] = "cpu"
    started = time.monotonic()
    proc = subprocess.run(
        [sys.executable, "-m", "pytest", "-p", "no:cacheprovider", "--rootdir", str(tmp_path),
         "-o", "faulthandler_timeout=2", "-o", "faulthandler_exit_on_timeout=true", str(test_file)],
        capture_output=True, text=True, timeout=45, cwd=tmp_path, env=env,
    )
    elapsed = time.monotonic() - started
    assert proc.returncode != 0, proc.stderr
    assert "Timeout (0:00:02)" in proc.stderr, proc.stderr
    assert "most recent call first" in proc.stderr, proc.stderr
    assert "test_sleeps_forever" in proc.stderr, proc.stderr
    assert elapsed < 30, f"the subprocess ran {elapsed:.1f} s; it must end at the 2 s limit"


def test_the_workflow_sets_the_limit_the_log_names_and_the_job_timeouts():
    """Every test-running invocation in the workflow carries the per-test limit, no ``-q``, and
    both jobs declare a ``timeout-minutes``.

    Reads ``.github/workflows/tests.yml`` the same way ``tests/test_no_tiers.py`` does and
    checks: each line that invokes the test runner (the ``--collect-only`` line excluded)
    passes the dump-and-exit options and no `` -q`` token; the ``-m "not gpu"`` selection is
    still there at least twice; and each of the jobs ``test`` and ``recorded-bits`` has a
    ``timeout-minutes:`` with an integer value, the one of ``test`` at least 50 (the measured
    healthy maximum of that job is 45 minutes).
    """
    text = WORKFLOW.read_text()
    lines = text.splitlines()
    invocation_lines = []
    for line in lines:
        without_env = re.sub(r"^(\s*)(?:[A-Za-z_][A-Za-z0-9_]*=\S+\s+)+", r"\1", line)
        if without_env.lstrip().startswith("pytest ") and "--collect-only" not in line:
            invocation_lines.append(line)
    assert invocation_lines, "no test-running invocation found in the workflow"
    for line in invocation_lines:
        assert "faulthandler_timeout=" in line, line
        assert "faulthandler_exit_on_timeout=true" in line, line
        assert " -q" not in line, line
    assert text.count('-m "not gpu"') >= 2
    blocks = {}
    current = None
    for line in lines:
        match = re.match(r"^  ([A-Za-z0-9_-]+):\s*$", line)
        if match:
            current = match.group(1)
            blocks[current] = []
        elif current is not None:
            blocks[current].append(line)
    for job in ("test", "recorded-bits"):
        assert job in blocks, f"job {job!r} not found in the workflow"
        match = re.search(r"^    timeout-minutes:\s*(\d+)\s*$", "\n".join(blocks[job]), re.M)
        assert match, f"job {job!r} has no timeout-minutes line"
        if job == "test":
            assert int(match.group(1)) >= 50, match.group(0)


def test_the_per_test_limit_is_not_set_in_the_project_config():
    """The per-test limit is a CI command-line option only; ``pyproject.toml`` sets none, so a
    local run is never limited by the limit written for CI."""
    toml = (ROOT / "pyproject.toml").read_text()
    assert "faulthandler_timeout" not in toml
