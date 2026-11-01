"""CI coverage: every directory holding ``test_*.py`` is named by a CI workflow, so a failing test there is
reported. A directory is covered when a workflow file mentions its path (``tests/geometry``, ``tests/geometry/x.py``)
or, unless ``strict``, when it lies under a directory the workflow names whole (``pytest tests/``).
``strict`` asks for the directory itself: for a repository whose CI splits one test root into groups."""
import re
from pathlib import Path

from .allow import Finding, line_hash, tracked

CHECK = "ci-coverage"
DEFAULT_WORKFLOWS = (".github/workflows/tests.yml",)


def _mentions(text, path, whole):
    tail = r"/?(?=[\s\"'\]),;]|$)" if whole else r"(?![\w.-])"
    return re.search(r"(?<![\w./-])" + re.escape(path) + tail, text, re.M) is not None


def check(root, workflows=DEFAULT_WORKFLOWS, strict=False, **_):
    texts = [(Path(root) / w).read_text() for w in workflows if (Path(root) / w).exists()]
    if not texts:
        return []                                             # an installed copy of the tests has no workflow
    text = "\n".join(texts)
    dirs = sorted({p.rsplit("/", 1)[0] for p in tracked(root)
                   if p.endswith(".py") and p.rsplit("/", 1)[-1].startswith("test_") and "/" in p})
    found = []
    for d in dirs:
        parents = [] if strict else ["/".join(d.split("/")[:k]) for k in range(1, len(d.split("/")))]
        if _mentions(text, d, False) or any(_mentions(text, a, True) for a in parents):
            continue
        found.append(Finding(CHECK, d, 1, "holds test_*.py but no CI workflow names it", line_hash(CHECK, d)))
    return found
