"""This repository passes the shared static guards that the other public repositories also call (the AST guards
run in the ``guards`` workflow against ``.guards-allow``)."""
import re

import pytest
from guards import allow, ci_coverage, private_ids

from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]

pytestmark = pytest.mark.skipif(not (ROOT / ".git").exists(), reason="an installed copy of the tests is no checkout")


def test_no_tracked_file_names_anything_outside_the_public_project():
    found = private_ids.check(ROOT)
    assert not found, "forbidden private identifier(s):\n" + "\n".join(map(str, found))


def test_every_test_directory_is_in_a_ci_group():
    found = ci_coverage.check(ROOT, strict=True)
    assert not found, "\n".join(map(str, found))


def test_allow_list_entries_are_well_formed():
    assert all(re.fullmatch(r"[0-9a-f]{10}", k) for _, k in allow.load(ROOT))
