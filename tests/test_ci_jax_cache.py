"""The ``test`` job of ``.github/workflows/tests.yml`` keeps JAX's persistent compilation cache in
``.jax_cache`` as one entry per (Python version, test group), written on ``main`` also after a
failed or timed-out test step. These tests read the workflow as text and report every way it could
silently fall back to the old scheme: the combined ``actions/cache@v4`` (whose post-job save is
skipped when the job fails), one key shared by all eight groups of a version, restore-keys without
the group, a save from a pull request or of an existing key, or a save that runs before the tests.
The checks run against the real file, against copies with one planted fault each, and against the
base workflow's own cache step embedded below, so a check that only reads the real file cannot pass."""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
WORKFLOW = ROOT / ".github" / "workflows" / "tests.yml"

# assembled from parts so that this file itself never spells the marker the guard scans for
SLOW_WORD = "sl" "ow"

RESTORE = "actions/cache/restore@v4"
SAVE = "actions/cache/save@v4"
COMBINED = "actions/cache@v4"
KEY_LINE = ("key: jax-${{ matrix.python-version }}-g${{ matrix.group }}"
            "-e${{ env.JAX_CACHE_EPOCH }}-${{ hashFiles('dmipy_sim/**/*.py') }}")
RESTORE_KEYS_LINE = "jax-${{ matrix.python-version }}-g${{ matrix.group }}-e${{ env.JAX_CACHE_EPOCH }}-"

# the cache step of the base workflow, before this change: the combined action, one key per Python
# version and source hash, no group anywhere, no save step of its own
BASE_CACHE_STEP = """\
      - name: Restore JAX compilation cache
        uses: actions/cache@v4
        with:
          path: .jax_cache
          key: jax-${{ matrix.python-version }}-${{ hashFiles('dmipy_sim/**/*.py') }}
          restore-keys: |
            jax-${{ matrix.python-version }}-
"""


def _job_lines(text):
    """The lines of the ``test`` job: everything between its header and the next job's header."""
    lines = text.splitlines()
    try:
        start = lines.index("  test:")
        end = lines.index("  recorded-bits:")
    except ValueError:
        return None
    return lines[start:end]


def _steps(job_lines):
    """Split the job's lines into steps; each step is ``(first line index, its lines)``."""
    steps, current, start = [], [], None
    for i, line in enumerate(job_lines):
        if line.startswith("      - "):
            if current is not None:
                steps.append((start, current))
            current, start = [line], i
        elif current is not None:
            current.append(line)
    if current is not None:
        steps.append((start, current))
    return steps


def _field(step, prefix):
    """The first line of the step that carries ``prefix``, with or without the list dash."""
    for line in step:
        s = line.strip()
        if s.startswith("- "):
            s = s[2:]
        if s.startswith(prefix):
            return s
    return None


def _problems(text):
    """Every fault the workflow text has against the per-group cache scheme, one sentence each."""
    problems = []
    job = _job_lines(text)
    if job is None:
        return ["the boundaries of the test job were not found in the workflow text"]
    steps = _steps(job)

    uses = []
    for line in job:
        s = line.strip()
        if s.startswith("- "):
            s = s[2:]
        if s.startswith("uses:"):
            uses.append(s)
    if "uses: " + COMBINED in uses:
        problems.append("combined cache action: a step still uses actions/cache@v4, whose "
                        "post-job save is skipped when the job fails or is cancelled")
    restore = [st for _, st in steps if _field(st, "uses:") == "uses: " + RESTORE]
    save = [st for _, st in steps if _field(st, "uses:") == "uses: " + SAVE]
    if not restore:
        problems.append("no cache restore step: no step uses actions/cache/restore@v4")
    if not save:
        problems.append("no cache save step: no step uses actions/cache/save@v4")

    keys = [k for st in restore + save if (k := _field(st, "key:")) is not None]
    if len(keys) != 2:
        problems.append("expected exactly two key lines, one per cache step, found %d" % len(keys))
    for k in keys:
        if "matrix.python-version" not in k:
            problems.append("primary key without the Python version: " + k)
        if "matrix.group" not in k:
            problems.append("primary key without the group: " + k)
        if "JAX_CACHE_EPOCH" not in k:
            problems.append("primary key without the epoch: " + k)
        if "hashFiles(" not in k:
            problems.append("primary key without the source hash: " + k)
    if len(keys) == 2 and keys[0] != keys[1]:
        problems.append("keys differ: the restore and the save do not use the same key")

    for st in restore:
        for i, line in enumerate(st):
            if line.strip().startswith("restore-keys:"):
                indent = len(line) - len(line.lstrip())
                for follower in st[i + 1:]:
                    if not follower.strip():
                        continue
                    if len(follower) - len(follower.lstrip()) <= indent:
                        break
                    if "matrix.group" not in follower:
                        problems.append("restore-keys without the group: " + follower.strip())

    for st in save:
        cond = _field(st, "if:")
        if cond is None:
            problems.append("save without a condition: the save step has no if: line")
            continue
        if "always()" not in cond:
            problems.append("save without always: " + cond)
        if "steps.tests.outcome" not in cond:
            problems.append("save without the skipped guard: " + cond)
        if "github.ref == 'refs/heads/main'" not in cond:
            problems.append("save not restricted to main: " + cond)
        if "cache-hit" not in cond:
            problems.append("save without the cache-hit guard: " + cond)
        if "success()" in cond or "cancelled()" in cond:
            problems.append("save condition uses success() or cancelled(): " + cond)

    tests_steps = [st for _, st in steps
                   if _field(st, "id:") == "id: tests"
                   or (_field(st, "name:") or "").endswith("Run tests (CPU)")]
    if not tests_steps:
        problems.append("no test step found: no step has id: tests or the name Run tests (CPU)")
    for st in tests_steps:
        if _field(st, "id:") != "id: tests":
            problems.append("missing step id: the Run tests (CPU) step has no id: tests")
    for st in restore:
        if _field(st, "id:") != "id: jaxcache":
            problems.append("missing step id: the restore step has no id: jaxcache")

    if save and tests_steps:
        save_pos = next(p for p, st in steps if st is save[0])
        tests_pos = next(p for p, st in steps if st is tests_steps[0])
        if save_pos < tests_pos:
            problems.append("save step before the test step: the cache is saved before the tests ran")

    for label, group in (("restore", restore), ("save", save)):
        for st in group:
            if _field(st, "path:") != "path: .jax_cache":
                problems.append("missing .jax_cache path in the " + label + " step")

    if SLOW_WORD in text:
        problems.append("the word " + SLOW_WORD + " appears in the workflow")
    if text.count('-m "not gpu"') < 2:
        problems.append('fewer than two -m "not gpu" selections in the workflow')
    if not any(l.strip().startswith("JAX_CACHE_EPOCH:") for l in text.splitlines()):
        problems.append("missing JAX_CACHE_EPOCH in the workflow-level env")
    return problems


def test_the_workflow_restores_and_saves_one_cache_per_python_and_group():
    assert _problems(WORKFLOW.read_text()) == []


def test_each_planted_fault_is_found():
    real = WORKFLOW.read_text()
    save_key_without_group = KEY_LINE.replace("-g${{ matrix.group }}-", "-")
    restore_keys_without_group = RESTORE_KEYS_LINE.replace("-g${{ matrix.group }}-", "-")
    plants = [
        ("save without always", "if: always() && steps.tests.outcome", "if: steps.tests.outcome"),
        ("primary key without the group",
         KEY_LINE + "\n\n      - name: Collect the whole suite",
         save_key_without_group + "\n\n      - name: Collect the whole suite"),
        ("restore-keys without the group",
         "            " + RESTORE_KEYS_LINE,
         "            " + restore_keys_without_group),
        ("combined cache action", "uses: " + SAVE, "uses: " + COMBINED),
        ("combined cache action",
         "      - name: Save JAX compilation cache",
         "      - uses: actions/cache@v4\n        with: {path: .jax_cache, key: x}\n\n"
         "      - name: Save JAX compilation cache"),
        ("missing step id", "        id: tests\n", ""),
        ("save not restricted to main", " && github.ref == 'refs/heads/main'", ""),
        ("no test step found",
         "      - name: Run tests (CPU)\n        id: tests\n",
         "      - name: Run the group\n"),
    ]
    for phrase, old, new in plants:
        assert real.count(old) == 1, "planted string occurs %d times: %r" % (real.count(old), old)
        problems = _problems(real.replace(old, new))
        assert problems, "the planted fault was not reported: " + phrase
        assert any(phrase in p for p in problems), (phrase, problems)


def test_the_base_workflows_cache_step_is_reported():
    base_job = "\n".join([
        "  test:",
        "    runs-on: ubuntu-latest",
        "    steps:",
        "      - uses: actions/checkout@v4",
        BASE_CACHE_STEP,
        "      - name: Run tests (CPU)",
        "        run: |",
        "          JAX_PLATFORMS=cpu pytest tests -q --tb=short -m \"not gpu\"",
        "  recorded-bits:",
        "",
    ])
    problems = _problems(base_job)
    assert problems, "the base workflow's cache step was not reported"
    assert any("combined cache action" in p for p in problems), problems
    assert any("missing step id" in p for p in problems), problems
