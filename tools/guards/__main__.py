"""``python -m guards <check> <repo-root> [options]``; ``all`` runs every guard. Exit status 1 on any finding."""
import argparse
import sys
import time

from . import allow, ci_coverage, hidden_defaults, history_words, private_ids

CHECKS = {"private-ids": private_ids, "ci-coverage": ci_coverage,
          "hidden-defaults": hidden_defaults, "history-wording": history_words}


def run(name, root, src=(), workflows=ci_coverage.DEFAULT_WORKFLOWS, strict=False):
    found = CHECKS[name].check(root, src=src, workflows=workflows, strict=strict)
    return found if name == "private-ids" else allow.unallowed(found, root)


def main(argv=None):
    ap = argparse.ArgumentParser(prog="guards", description=__doc__)
    ap.add_argument("check", choices=["all", *CHECKS])
    ap.add_argument("root")
    ap.add_argument("--src", default="", help="space-separated package paths for the AST guards (default: detected)")
    ap.add_argument("--workflows", default=" ".join(ci_coverage.DEFAULT_WORKFLOWS),
                    help="space-separated workflow files that collect the tests")
    ap.add_argument("--strict-test-dirs", action="store_true", help="each test directory must itself be named")
    ap.add_argument("--record", metavar="REASON", help="append every finding to .guards-allow with this reason")
    a = ap.parse_args(argv)
    failed = 0
    for name in (CHECKS if a.check == "all" else [a.check]):
        t0 = time.perf_counter()
        found = run(name, a.root, a.src.split(), a.workflows.split(), a.strict_test_dirs)
        found = sorted(found)
        if a.record and name != "private-ids":
            print(f"guards: {name}: recorded {allow.record(found, a.root, a.record)} entries", file=sys.stderr)
            found = []
        for f in found:
            print(f)
        print(f"guards: {name}: {len(found)} finding(s) in {time.perf_counter() - t0:.2f} s", file=sys.stderr)
        failed += bool(found)
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
