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
OUTAGE_WAIT_S = 60           # a 5xx (the hub refusing commits) waits this long before the retry ...
OUTAGE_TRIES = 60            # ... up to this many times (an hour): a hub outage must not crash every worker
READ_TRIES = 8               # a read (a listing, an exists, a download) retried with backoff 30, 60, ... s: a network blip must not end a loop
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


def _status(e):
    """The HTTP status an exception carries (``e.response.status_code``: the hub's errors, and the fake hub's), else None."""
    return getattr(getattr(e, "response", None), "status_code", None)


def is_rate_limited(e):
    """Whether an exception is the hub's 429."""
    return _status(e) == 429


def is_outage(e):
    """Whether an exception is the hub failing on its side (a 5xx)."""
    st = _status(e)
    return st is not None and 500 <= st < 600


def is_stale_parent(e):
    """The hub refused a commit whose declared parent is no longer the head (HTTP 412): another writer got there."""
    return _status(e) == 412


class StaleParent(RuntimeError):
    """A commit declared ``parent=`` and the repository's head had moved: the caller re-reads and commits again."""


def is_not_found(e):
    """Whether an exception says a file is not there (which a read reports, never retries)."""
    if isinstance(e, FileNotFoundError):
        return True
    try:
        from huggingface_hub.errors import EntryNotFoundError, RepositoryNotFoundError, RevisionNotFoundError
    except ImportError:                                    # pragma: no cover
        return _status(e) == 404
    return isinstance(e, (EntryNotFoundError, RepositoryNotFoundError, RevisionNotFoundError)) or _status(e) == 404


class HubBase:
    """The two primitives' retries: :meth:`commit` calls the subclass's ``_commit`` up to ``tries`` times, a 429
    after :data:`RATE_LIMIT_WAIT_S`, any other error after a backoff of 30, 60, 120... s; :meth:`read` calls a
    read (a listing, an exists, a download) up to :data:`READ_TRIES` times with the same backoff, since a
    network blip or a 5xx on a read ended a worker's loop where a commit would have waited (dmipy-sim#286)."""

    def read(self, what, fn, *, tries=READ_TRIES):
        """``fn()`` with the retry of a read: any error but a not-found is retried after 30, 60, 120... s up to
        ``tries`` times; a not-found is raised at once (it is an answer)."""
        k = 0
        while True:
            try:
                return fn()
            except Exception as e:
                if is_not_found(e):
                    raise
                k += 1
                if k >= tries:
                    raise
                wait = RATE_LIMIT_WAIT_S if is_rate_limited(e) else 30 * 2 ** (k - 1)
                log.warning("%s failed (%s); retry %d/%d in %d s", what, str(e).splitlines()[0][:160], k + 1, tries, wait)
                time.sleep(wait)

    def commit(self, adds, deletes, message, *, tries=UPLOAD_TRIES, parent=None):
        """ONE commit: ``adds`` is ``{remote path: local path or bytes}``, ``deletes`` a list of remote paths (a
        path that is not there -- a claim released before -- is skipped). Returns ``{remote path: sha256}``.
        ``tries`` bounds the retries of an error of ours (backoff 30, 60, 120... s) and of a 429 (a wait of
        :data:`RATE_LIMIT_WAIT_S`); a 5xx is the hub's outage and is retried every :data:`OUTAGE_WAIT_S` up to
        :data:`OUTAGE_TRIES` times on its own count (``tries=1`` skips every retry: a heartbeat). ``parent`` is
        the head (:meth:`head`) the commit was prepared against: a commit on a file two writers read, modify and
        write (a manifest) declares it, and the hub refuses it when the head has moved -- raised at once as
        :class:`StaleParent`, never retried, since the caller must read again."""
        expect = _expected(adds)
        k = outages = 0
        while True:
            try:
                self._commit(dict(adds), list(deletes), message, expect, parent)
                return {r: expect[r][0] for r in adds}
            except StaleParent:
                raise
            except Exception as e:
                if is_stale_parent(e):
                    raise StaleParent(f"commit {message!r}: the head moved past {parent}") from e
                if tries <= 1:
                    raise
                if is_outage(e):
                    outages += 1
                    if outages >= OUTAGE_TRIES:
                        raise
                    wait = OUTAGE_WAIT_S; n, of = outages + 1, OUTAGE_TRIES
                else:
                    k += 1
                    if k >= tries:
                        raise
                    wait = RATE_LIMIT_WAIT_S if is_rate_limited(e) else 30 * 2 ** (k - 1); n, of = k + 1, tries
                log.warning("commit %r failed (%s); retry %d/%d in %d s", message, str(e).splitlines()[0][:160], n, of, wait)
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
        return self.read(f"download of {f}", lambda: hf_hub_download(self.repo, f, repo_type="dataset", revision=self.revision))

    def get_live(self, f):
        """A file as it is now (a claim, another worker's heartbeat)."""
        from huggingface_hub import hf_hub_download
        return self.read(f"download of {f}", lambda: hf_hub_download(self.repo, f, repo_type="dataset"))

    def head(self):
        """The repository's current commit."""
        return self.read("head", lambda: self.api.dataset_info(self.repo).sha)

    def get_at(self, f, revision):
        """A file at ``revision`` (a local path): what a read-modify-write reads, so its commit can declare that parent."""
        from huggingface_hub import hf_hub_download
        return self.read(f"download of {f}@{revision[:8]}", lambda: hf_hub_download(self.repo, f, repo_type="dataset", revision=revision))

    def files(self):
        return self.read("listing", lambda: set(self.api.list_repo_files(self.repo, repo_type="dataset")))

    def exists(self, path):
        return self.read(f"exists of {path}", lambda: self.api.file_exists(self.repo, path, repo_type="dataset"))

    def _commit(self, adds, deletes, message, expect, parent=None):
        from huggingface_hub import CommitOperationAdd, CommitOperationDelete
        if deletes:
            deletes = [i.path for i in self.api.get_paths_info(self.repo, deletes, repo_type="dataset")]
        ops = ([CommitOperationAdd(path_in_repo=r, path_or_fileobj=l) for r, l in adds.items()]
               + [CommitOperationDelete(path_in_repo=d) for d in deletes])
        if not ops:
            return
        self.api.create_commit(repo_id=self.repo, repo_type="dataset", operations=ops, commit_message=message,
                               parent_commit=parent)
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
    commit raises the hub's 429 (a retry then succeeds); ``fail_read`` a list of read names (``"exists"``,
    ``"files"``, ``"get"``) each of which makes the next such read raise a 5xx once."""

    def __init__(self, root, *, fail_429=(), fail_500=(), fail_read=()):
        self.root = os.path.abspath(root); self.repo = f"fake:{self.root}"
        os.makedirs(self.root, exist_ok=True)
        self.revision = "fake"; self.queue = []; self.log = []; self.fail_429 = list(fail_429); self.fail_500 = list(fail_500)
        self.fail_read = list(fail_read)

    def _maybe_fail(self, name):
        if name in self.fail_read:
            self.fail_read.remove(name)
            raise _fake_http(f"{name}: the hub failed", 502, "Bad Gateway")

    def _p(self, f):
        return os.path.join(self.root, f)

    def get(self, f):
        def go():
            self._maybe_fail("get")
            p = self._p(f)
            if not os.path.isfile(p):
                raise FileNotFoundError(f)
            return p
        return self.read(f"download of {f}", go)

    get_live = get

    def head(self):
        return f"fake-{len(self.log)}"

    def get_at(self, f, revision):
        return self.get(f)                                     # the directory holds the latest only

    def files(self):
        def go():
            self._maybe_fail("files")
            out = set()
            for d, _, fs in os.walk(self.root):
                for f in fs:
                    out.add(os.path.relpath(os.path.join(d, f), self.root))
            return out
        return self.read("listing", go)

    def exists(self, path):
        return self.read(f"exists of {path}", lambda: (self._maybe_fail("exists"), os.path.isfile(self._p(path)))[1])

    def _commit(self, adds, deletes, message, expect, parent=None):
        if parent is not None and parent != self.head():
            raise _fake_http(f"{message}: parent {parent} is not the head {self.head()}", 412, "Precondition Failed")
        for i, prefix in enumerate(self.fail_429):
            if message.startswith(prefix):
                del self.fail_429[i]
                raise _fake_http(message, 429, "Too Many Requests")
        for i, prefix in enumerate(self.fail_500):
            if message.startswith(prefix):
                del self.fail_500[i]
                raise _fake_http(message, 500, "Internal Server Error")
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


class _HubHTTPError(Exception):
    """The fake hub's HTTP error where ``huggingface_hub`` is not installed: the status on ``response`` like the real one's."""

    def __init__(self, message, response):
        super().__init__(message); self.response = response


def _fake_http(message, code, reason):
    import types
    try:
        import requests
        from huggingface_hub.errors import HfHubHTTPError
    except ImportError:
        return _HubHTTPError(f"{code} Error: {reason} ({message})", types.SimpleNamespace(status_code=code, reason=reason))
    r = requests.Response(); r.status_code = code; r.reason = reason; r._content = reason.encode()
    r.request = requests.Request("POST", "https://fake/api/commit").prepare()
    return HfHubHTTPError(f"{code} Error: {reason} ({message})", response=r)
