"""The dataset repository a fill runs against, behind one small interface: files down, ONE write primitive, and
a fake of it for tests.

A fill's workers coordinate through files in the repository and nothing else. The repository allows 128 commits
an hour (huggingface.co, per repository), so everything a block leaves -- its shard, its summary, the run
records of its walk, the release of its claim -- goes up in one :meth:`Hub.commit`, and claims are written
several per commit. A 429 from the hub waits :data:`RATE_LIMIT_WAIT_S` and tries again; any other error backs
off; a heartbeat that must not block passes ``tries=1``. Every added file is verified after the commit: the
hub's LFS sha256 (or the size of a small file) must match the local one.

Recipe files are read at the revision the worker started on (:meth:`Hub.get`): the repository gains a commit
every few minutes while a fill runs, and files fetched at different revisions land in different snapshot
directories (a far grid's ``.json`` was not beside its ``.npy``). Claims are read as they are now
(:meth:`Hub.get_live`).

:class:`FakeHub` is the same interface over a directory, with a commit log and an injectable 429, for the tests
of everything above it."""
import hashlib
import json
import logging
import os
import shutil
import time

RATE_LIMIT_WAIT_S = 600      # a 429 (128 commits an hour per repository) waits this long before the retry
UPLOAD_TRIES = 6             # a commit retried with backoff: a transient hub error must not lose a shard
log = logging.getLogger("dmipy_sim.fill")


def sha256_of(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 24), b""):
            h.update(chunk)
    return h.hexdigest()


def _expected(adds):
    return {r: ((sha256_of(l), os.path.getsize(l)) if isinstance(l, str) else (hashlib.sha256(l).hexdigest(), len(l)))
            for r, l in adds.items()}


def is_rate_limited(e):
    """Whether an exception is the hub's 429."""
    try:
        from huggingface_hub.errors import HfHubHTTPError
    except ImportError:                                    # pragma: no cover
        return False
    return isinstance(e, HfHubHTTPError) and getattr(getattr(e, "response", None), "status_code", None) == 429


class HubBase:
    """The write primitive's retry: :meth:`commit` calls the subclass's ``_commit`` up to ``tries`` times, a 429
    after :data:`RATE_LIMIT_WAIT_S`, any other error after a backoff of 30, 60, 120... s."""

    def commit(self, adds, deletes, message, *, tries=UPLOAD_TRIES):
        """ONE commit: ``adds`` is ``{remote path: local path or bytes}``, ``deletes`` a list of remote paths (a
        path that is not there -- a claim released before -- is skipped). Returns ``{remote path: sha256}``."""
        expect = _expected(adds)
        for k in range(tries):
            try:
                self._commit(dict(adds), list(deletes), message, expect)
                return {r: expect[r][0] for r in adds}
            except Exception as e:
                if k == tries - 1:
                    raise
                wait = RATE_LIMIT_WAIT_S if is_rate_limited(e) else 30 * 2 ** k
                log.warning("commit %r failed (%s); retry %d/%d in %d s", message, str(e).splitlines()[0][:160], k + 2, tries, wait)
                time.sleep(wait)

    def put_json(self, obj, path, message, *, tries=UPLOAD_TRIES):
        return self.commit({path: json.dumps(obj, indent=1).encode()}, [], message, tries=tries)

    def delete(self, path, message):
        return self.commit({}, [path], message)


class Hub(HubBase):
    """A dataset repository on the hub. ``repo`` is ``owner/name``; the token is the cached login's."""

    def __init__(self, repo):
        from huggingface_hub import HfApi
        self.repo = repo
        self.api = HfApi()
        self.revision = self.api.dataset_info(repo).sha    # the recipe as of the start: every file from ONE revision
        self.queue = []                                    # blocks claimed ahead in one commit (claims.claim_next)

    def get(self, f):
        """A recipe file, at the revision the worker started on (a local path)."""
        from huggingface_hub import hf_hub_download
        return hf_hub_download(self.repo, f, repo_type="dataset", revision=self.revision)

    def get_live(self, f):
        """A file as it is now (a claim, another worker's heartbeat)."""
        from huggingface_hub import hf_hub_download
        return hf_hub_download(self.repo, f, repo_type="dataset")

    def files(self):
        return set(self.api.list_repo_files(self.repo, repo_type="dataset"))

    def exists(self, path):
        return self.api.file_exists(self.repo, path, repo_type="dataset")

    def _commit(self, adds, deletes, message, expect):
        from huggingface_hub import CommitOperationAdd, CommitOperationDelete
        if deletes:
            deletes = [i.path for i in self.api.get_paths_info(self.repo, deletes, repo_type="dataset")]
        ops = ([CommitOperationAdd(path_in_repo=r, path_or_fileobj=l) for r, l in adds.items()]
               + [CommitOperationDelete(path_in_repo=d) for d in deletes])
        if not ops:
            return
        self.api.create_commit(repo_id=self.repo, repo_type="dataset", operations=ops, commit_message=message)
        if adds:                                           # verified: the hub holds what was sent
            for info in self.api.get_paths_info(self.repo, list(adds), repo_type="dataset"):
                sha, size = expect[info.path]
                if not ((info.lfs is not None and info.lfs.sha256 == sha) or (info.lfs is None and info.size == size)):
                    raise RuntimeError(f"{info.path}: the hub holds {getattr(info.lfs, 'sha256', None) or info.size}, local {sha} ({size} B)")

    def commits(self):
        """The commit log, newest first: ``[(time, title)]``."""
        return [(c.created_at.timestamp(), c.title) for c in self.api.list_repo_commits(self.repo, repo_type="dataset")]


class FakeHub(HubBase):
    """The same interface over a directory ``root``: what the tests of a fill run against. ``log`` holds every
    commit as ``dict(time, message, adds, deletes)``; ``fail_429`` is a list of message prefixes whose first
    commit raises the hub's 429 (a retry then succeeds)."""

    def __init__(self, root, *, fail_429=()):
        self.root = os.path.abspath(root); self.repo = f"fake:{self.root}"
        os.makedirs(self.root, exist_ok=True)
        self.revision = "fake"; self.queue = []; self.log = []; self.fail_429 = list(fail_429)

    def _p(self, f):
        return os.path.join(self.root, f)

    def get(self, f):
        p = self._p(f)
        if not os.path.isfile(p):
            raise FileNotFoundError(f)
        return p

    get_live = get

    def files(self):
        out = set()
        for d, _, fs in os.walk(self.root):
            for f in fs:
                out.add(os.path.relpath(os.path.join(d, f), self.root))
        return out

    def exists(self, path):
        return os.path.isfile(self._p(path))

    def _commit(self, adds, deletes, message, expect):
        for i, prefix in enumerate(self.fail_429):
            if message.startswith(prefix):
                del self.fail_429[i]
                raise _fake_429(message)
        for r, l in adds.items():
            p = self._p(r); os.makedirs(os.path.dirname(p), exist_ok=True)
            if isinstance(l, str):
                shutil.copyfile(l, p)
            else:
                open(p, "wb").write(l)
        gone = []
        for d in deletes:
            if os.path.isfile(self._p(d)):
                os.remove(self._p(d)); gone.append(d)
        self.log.append(dict(time=time.time(), message=message, adds=sorted(adds), deletes=gone))

    def commits(self):
        return [(c["time"], c["message"]) for c in reversed(self.log)]


def _fake_429(message):
    import requests
    from huggingface_hub.errors import HfHubHTTPError
    r = requests.Response(); r.status_code = 429; r.reason = "Too Many Requests"; r._content = b"rate limited"
    r.request = requests.Request("POST", "https://fake/api/commit").prepare()
    return HfHubHTTPError(f"429 Client Error: Too Many Requests ({message})", response=r)
