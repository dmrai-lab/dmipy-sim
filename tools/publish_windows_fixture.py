"""Publish the window-read test fixture to the Hub, by hand: ``tests/replay/test_hub_windows.py``'s ``_fixture_pack``
(a 3-window, 300-walker slab pack) written to ``SubstrateCommons/test-fixtures/windows-fixture.rpk``, which the
suite's tier-2 test ``test_a_windowed_read_from_the_real_hub`` reads (and never writes).

    python tools/publish_windows_fixture.py              # builds the pack and prints its sha256; uploads nothing
    python tools/publish_windows_fixture.py --upload     # creates the dataset if needed and uploads when the sha256 differs

Run from the repository root with a Hub login that can write to the dataset. The dataset's README says what it is
(:data:`README`)."""
import argparse
import os
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
REPO = "SubstrateCommons/test-fixtures"
PATH = "windows-fixture.rpk"

README = """---
license: cc-by-4.0
pretty_name: dmipy-sim test fixtures
---
# dmipy-sim test fixtures

Small artefacts dmipy-sim's test suite reads to exercise the Hub read paths -- not a curated reference dataset. A
file here is rebuilt by the script of dmipy-sim that owns it (`tools/publish_windows_fixture.py`) when its fixture
changes; nothing here is a citeable measurement. See `SubstrateCommons/parity-fixtures` for the licensed,
cross-engine parity packs this is not.
"""


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--upload", action="store_true", help="upload to the Hub (without it nothing leaves this machine)")
    a = ap.parse_args(argv)
    sys.path.insert(0, str(ROOT))
    from dmipy_replay.replay.hub import sha256_of
    from dmipy_sim.replay.container import save_pack
    from tests.replay.test_hub_windows import _fixture_pack
    with tempfile.TemporaryDirectory(prefix="dmipy-windows-fixture-") as tmp:
        local = os.path.join(tmp, PATH)
        save_pack(_fixture_pack(), local)
        sha = sha256_of(local)
        print(f"{PATH}: sha256 {sha}, {os.path.getsize(local)} bytes")
        if not a.upload:
            return 0
        from huggingface_hub import HfApi
        api = HfApi()
        api.create_repo(REPO, repo_type="dataset", exist_ok=True, private=False)
        if not api.file_exists(REPO, "README.md", repo_type="dataset"):
            api.upload_file(path_or_fileobj=README.encode(), path_in_repo="README.md", repo_id=REPO, repo_type="dataset",
                            commit_message="describe the dataset")
        info = api.get_paths_info(REPO, [PATH], repo_type="dataset")
        held = info[0].lfs.sha256 if info and getattr(info[0], "lfs", None) is not None else None
        if held == sha:
            print("the Hub holds this file already")
            return 0
        api.upload_file(path_or_fileobj=local, path_in_repo=PATH, repo_id=REPO, repo_type="dataset",
                        commit_message="the windows=range(k) hub-read test fixture (dmipy-sim#525)")
        print(f"uploaded hf://{REPO}/{PATH}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
