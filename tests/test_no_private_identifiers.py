"""This repository is public: no tracked file names a project, service, machine or local path outside it.

The forbidden words are never spelled here, in a comment, in a commit message or in test output. Two encodings
are used, both resolved at runtime:

* single words (a repository, a rented-compute vendor, a lab machine) are stored as sha256 hashes of the
  lowercase token; a file is tokenised on ``[A-Za-z0-9_-]+`` (and on ``-`` / ``_`` inside a token) and each
  token is hashed the same way;
* multi-token patterns (a package with an issue number, a workspace path, ownership phrasing) are stored as
  split fragments that are joined into a regular expression when the test runs.

A failure names the FILE (and the pattern's index), never the matched text. Hardware classes (GPU model names)
may be named; which machine ran a measurement may not."""
import hashlib
import os
import re
import subprocess

#: sha256 of the lowercase token (single words)
FORBIDDEN_TOKEN_DIGESTS = frozenset((
    "2f1e83d30fff12f10f4a956d08bd6b200ae89e24621c2066c1a902aab2da7acb",
    "01db71ab8048f74a4b92c26ba77285ade0687ac192758e8185ad52701f649ef2",
    "e839e162d8106fe35160d44676559cc5b7cd62284058792011d52976e8a2a0e1",
    "b95ee4efc49878ebc6e8f93c29d7f692742dad7330574634428c8d6355794d88",
    "294aa8d75483b8331e3ba6a7f24aea15202747f36de65197e7bc6194880b2558",
    "be2abdeb021a79176e2dbf116831429aee5d4bd795b9d659f5ce5bf666492889",
    "eb782b881d4729e55686ccf5debe37f3708bdab49152e1b4b1599a906a563f70",
    "75481d9af31632532bc70496d9416e28a2a6e3948ed7944a100de667c214885c",
    "c040c5a65fdb91143944def084d1a1d1dd16973552fecf60ad94b70e2e11de91",
    "16f1496bf9d0e3d13476dff58afe84023ab9db78d596a440bb2bd27d1dcc086e",
))

#: split fragments, joined into case-insensitive regular expressions at run time
FORBIDDEN_FRAGMENTS = (
    ("dmipy", "[-_]sim", "[-_]cu", "da"),                   # the non-public backend package
    ("cu", "da#", r"\d"),                                    # its issue references
    ("dmrai", "-w", "s"),                                    # a lab workspace path
    ("/ho", "me/"),
    ("/m", "nt/"),
    (r"\bmo", "dal (run|t4|container)", "|import mo", "dal|mo", r"dal\.(app|image)"),   # a rented-compute vendor
    ("shared ", "box"),                                      # ownership phrasing
    ("vast", r"\.ai"),
    ("dmrai", "-pa", "pers"),                                # a private repository, by prefix
)

TOKEN_RE = re.compile(r"[A-Za-z0-9_-]+")
_PATTERNS = [re.compile("".join(f), re.IGNORECASE) for f in FORBIDDEN_FRAGMENTS]


def _digest(token):
    return hashlib.sha256(token.lower().encode()).hexdigest()


def _tokens(text):
    for tok in TOKEN_RE.findall(text):
        yield tok
        parts = re.split(r"[-_]", tok)
        if len(parts) > 1:
            yield from parts


def tracked_files():
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    out = subprocess.run(["git", "-C", root, "ls-files"], capture_output=True, text=True, check=True)
    return [os.path.join(root, p) for p in out.stdout.splitlines() if p]


def test_no_tracked_file_names_anything_outside_the_public_project():
    hits = []
    for path in tracked_files():
        if os.path.abspath(path) == os.path.abspath(__file__):
            continue                                          # this file's own encoded tables
        try:
            with open(path, encoding="utf-8") as f:
                text = f.read()
        except (UnicodeDecodeError, IsADirectoryError, FileNotFoundError):
            continue                                          # a binary asset
        if any(_digest(t) in FORBIDDEN_TOKEN_DIGESTS for t in _tokens(text)):
            hits.append((os.path.relpath(path), "token"))
        for i, pat in enumerate(_PATTERNS):
            if pat.search(text):
                hits.append((os.path.relpath(path), f"pattern {i}"))
    assert not hits, f"{len(hits)} forbidden private identifier(s) in tracked files: {hits}"
