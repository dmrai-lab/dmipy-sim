"""The record of a run: what every long producer leaves behind while it runs and when it stops.

A producer (``simulate``, the trajectory walks, ``walk_spec``, ``build_replay_pack``, ``merge_packs``) opens a
:class:`Run`; a caller never does. The run keeps its events in memory from the first moment and PERSISTS itself
-- a directory under ``$DMIPY_SIM_RUN_DIR`` (default ``~/.cache/dmipy-sim/runs``), or ``run_dir=`` at once -- the
first time it outlives the sampling interval, so a half-second call in a test suite touches no disk and a walk of
minutes has a record that covers it from its start. A producer opened inside another joins the outer run (a
``join`` row; its events go to the outer's current phase): ``walk_spec`` -> the walk -> the pack is one record.

The directory holds ``manifest.json`` (written first: producer, parameters, code, host, devices, the memory
ceiling, the command line -- a dead run is identifiable from it alone), ``events.jsonl`` (append-only, one JSON
row per line, flushed per row, so a kill loses at most the row being written: ``phase``, ``progress`` with rate
and ETA, ``resource`` from the sampler thread -- host RSS from ``/proc/self/statm``, host available, the cgroup's
current and ceiling, every JAX device's bytes in use and peak, and ``where``, the main thread's stack at that moment -- ``artifact``,
``warning``, ``end`` with the exception's traceback when there was one) and ``summary.json`` (the wall time per phase, the peaks, the status).
The last resource row's age is the run's heartbeat: :func:`list_runs` reports a run whose heartbeat is older
than twice the interval as ``stale``. ``python -m dmipy_sim.run`` lists the runs on this machine or reports one.

A ``run_dir`` that already holds a record of the same producer with the same parameters is RESUMED: the events
are appended to, and what the earlier run spooled (:meth:`Run.spool`: a walk's finished batches, one safetensors
file each under ``spool/``) is there for the producer to read back (:meth:`Run.spooled`) instead of redoing; a
record of a different run in that directory is refused.

Progress is throttled here, not by the caller: a producer reports every batch or save it finishes, and a row
is written when ``DMIPY_SIM_PROGRESS_S`` has passed since the last (or the phase completes). The sampler reads
``/proc`` (microseconds) and ``device.memory_stats()``; nothing touches the device's compute.

The budget guard: the host kills a process that reaches its ceiling with ``SIGKILL`` and no last word, so the run
raises first. At every sample the RSS is held against ``DMIPY_SIM_BUDGET_FRACTION`` (0.9) of the ceiling
(:func:`memory_ceiling`: the cgroup's, else ``MemTotal``, or ``DMIPY_SIM_MEMORY_CEILING_BYTES`` when set), and at
every batch boundary the RSS the run will reach at its pace -- the median growth over the last ``BUDGET_WINDOW``
batches times the batches left, after ``BUDGET_WARMUP`` batches (both at least ``BUDGET_WARMUP_SHARE`` of the run),
at ``BUDGET_STRIKES`` consecutive boundaries -- is held against it too; past either, :class:`ResourceBudgetError` is raised from the producer at the next
batch boundary (the spool intact, the record ended with the projection), and at the next progress report when the
margin itself is crossed. ``DMIPY_SIM_BUDGET_FRACTION=0`` disables the guard.
"""
from __future__ import annotations

import datetime as _dt
import json
import logging
import os
import platform
import socket
import subprocess
import sys
import threading
import time
import traceback

import numpy as np
from dataclasses import dataclass, field
from typing import Optional

log = logging.getLogger("dmipy_sim.run")

#: seconds between resource rows; a run that outlives this is persisted
SAMPLE_S = float(os.environ.get("DMIPY_SIM_SAMPLE_S", "10"))
#: seconds between progress rows of one phase (the phase's completion is always written)
PROGRESS_S = float(os.environ.get("DMIPY_SIM_PROGRESS_S", "10"))
#: a heartbeat older than this many intervals is stale
STALE_INTERVALS = 2.0
#: the share of the memory ceiling a run may reach before it stops itself; 0 disables the guard
BUDGET_FRACTION = float(os.environ.get("DMIPY_SIM_BUDGET_FRACTION", "0.9"))


#: batches before the guard projects a run's pace from its RSS growth, and the window of batches the pace is the
#: median growth over: the first batches carry the compile, the runtime's buffers and the first pages of the output
#: arrays, which a projection over ten thousand batches multiplies into nonsense (the DiSCo far-grid build's first
#: batches grew 44 MB, then 250 MB, and were projected to 1.8 TB and 10 TB against a 3 GB run); a median over a
#: window ignores a one-off jump and keeps a sustained leak; and the projection must exceed the budget at
#: BUDGET_STRIKES consecutive boundaries before the run stops
BUDGET_WARMUP = 8
BUDGET_WINDOW = 32
BUDGET_STRIKES = 2
#: and for a long run both scale with it: the guard judges a pace after this share of the batches (a 42,000-batch
#: build's first 420: what its warm-up amounts to; on a 42,000-batch run any growth above 12 MB per batch projects
#: past a 500 GB ceiling, so the pace must be the run's own, not its start's)
BUDGET_WARMUP_SHARE = 0.01


class ResourceBudgetError(RuntimeError):
    """The run would exceed the host's memory ceiling: stopped before the kernel does it, its record complete."""

_local = threading.local()


def run_root():
    """The directory runs persist under: ``$DMIPY_SIM_RUN_DIR``, ``~/.cache/dmipy-sim/runs`` by default; ``None``
    when the variable is set and empty (persistence off)."""
    env = os.environ.get("DMIPY_SIM_RUN_DIR")
    if env is None:
        return os.path.join(os.path.expanduser("~"), ".cache", "dmipy-sim", "runs")
    return env or None


def current():
    """The innermost open :class:`Run` of this thread, or ``None``."""
    stack = getattr(_local, "stack", None)
    return stack[-1] if stack else None


_CODE = None


def package_version():
    """The installed dmipy-sim version, ``"dev"`` for a checkout that is not installed."""
    try:
        from importlib.metadata import version
        return version("dmipy-sim")
    except Exception:
        return "dev"


def _code():
    """The package's version and, in a checkout, its commit (read once, on the first record: not at import)."""
    global _CODE
    if _CODE is not None:
        return _CODE
    out = {"version": package_version()}
    here = os.path.dirname(os.path.abspath(__file__))
    try:
        r = subprocess.run(["git", "-C", here, "rev-parse", "HEAD"], capture_output=True, text=True, timeout=5)
        if r.returncode == 0:
            out["commit"] = r.stdout.strip()
    except Exception:
        pass
    _CODE = out
    return out


def _read_int(path):
    try:
        v = open(path).read().strip()
        return None if v == "max" else int(v)
    except Exception:
        return None


def _meminfo():
    """``(MemTotal, MemAvailable)`` in bytes, or ``(None, None)``."""
    try:
        d = {}
        for line in open("/proc/meminfo"):
            k, v = line.split(":", 1); d[k] = int(v.split()[0]) * 1024
        return d.get("MemTotal"), d.get("MemAvailable")
    except Exception:
        return None, None


def _rss():
    try:
        return int(open("/proc/self/statm").read().split()[1]) * os.sysconf("SC_PAGE_SIZE")
    except Exception:
        return None


def _cgroup():
    """The cgroup's ``(current, max)`` in bytes (v2, then v1), or ``(None, None)``."""
    cur = _read_int("/sys/fs/cgroup/memory.current"); mx = _read_int("/sys/fs/cgroup/memory.max")
    if cur is None:
        cur = _read_int("/sys/fs/cgroup/memory/memory.usage_in_bytes"); mx = _read_int("/sys/fs/cgroup/memory/memory.limit_in_bytes")
        if mx is not None and mx > 1 << 60:
            mx = None
    return cur, mx


def _devices():
    """The JAX devices' memory, without initialising JAX when the process has not."""
    if "jax" not in sys.modules:
        return []
    try:
        import jax
        out = []
        for d in jax.local_devices():
            s = d.memory_stats() if hasattr(d, "memory_stats") else None
            out.append(dict(device=str(d), bytes_in_use=(s or {}).get("bytes_in_use"), peak_bytes_in_use=(s or {}).get("peak_bytes_in_use"),
                            bytes_limit=(s or {}).get("bytes_limit")))
        return out
    except Exception:
        return []


def memory_ceiling():
    """The bytes this process may use before the host kills it: ``DMIPY_SIM_MEMORY_CEILING_BYTES`` when set (a
    shared machine's share, a test), else the cgroup's ceiling when there is one, else ``MemTotal``; ``None`` when
    none is readable."""
    env = os.environ.get("DMIPY_SIM_MEMORY_CEILING_BYTES")
    if env:
        return int(float(env))
    _, mx = _cgroup()
    if mx is not None:
        return mx
    return _meminfo()[0]


def _main_thread_frames(depth=6):
    """Where the main thread is, as ``file:line function`` from the innermost frame out: the run's own stack trace
    in every resource row, so a stalled run says what it was doing (a sampler that cannot run at all -- the main
    thread inside a C call holding the GIL -- leaves the last row's ``where`` as the answer)."""
    try:
        frame = sys._current_frames().get(threading.main_thread().ident)
        out = []
        while frame is not None and len(out) < depth:
            out.append(f"{os.path.basename(frame.f_code.co_filename)}:{frame.f_lineno} {frame.f_code.co_name}")
            frame = frame.f_back
        return out
    except Exception:
        return None


def _jsonable(x):
    try:
        json.dumps(x); return x
    except Exception:
        return repr(x)


@dataclass
class _Phase:
    name: str
    started: float
    ended: Optional[float] = None
    first_progress: Optional[tuple] = None          # (t, done)
    last_written: float = -1e9


class Run:
    """The record of one producer's run. ``params`` are recorded in the manifest (JSON-able values; anything
    else by its ``repr``); ``run_dir`` persists at once, at that path."""

    def __init__(self, producer, *, params=None, run_dir=None):
        self.producer = str(producer)
        self.params = {k: _jsonable(v) for k, v in (params or {}).items()}
        self.run_dir = run_dir
        self.resumed = False
        if run_dir is not None and os.path.isfile(os.path.join(run_dir, "manifest.json")):
            man = json.load(open(os.path.join(run_dir, "manifest.json")))
            if man.get("producer") != self.producer or man.get("params") != self.params:
                raise ValueError(f"{run_dir} holds the record of another run ({man.get('producer')} {man.get('params')}); "
                                 f"this is {self.producer} {self.params}")
            self.resumed = True
        self.started = time.time()
        self.id = f"{_dt.datetime.utcfromtimestamp(self.started).strftime('%Y%m%dT%H%M%S')}-{self.producer}-{os.getpid()}"
        self._events = []                            # buffered until persisted
        self._fh = None
        self._dir = None
        self._phases = []
        self._lock = threading.Lock()
        self._sampler = None
        self._stop = threading.Event()
        self._peak_rss = 0; self._peak_dev = 0
        self._outer = None
        self.status = None
        self._joined = False
        self._ceiling = memory_ceiling()
        self._budget = None                          # the ResourceBudgetError to raise at the next boundary
        self._rss_batch = []                         # RSS at the start of every batch of the current batches()
        self._strikes = 0

    # ------------------------------------------------------------------ the context
    def __enter__(self):
        outer = current()
        if outer is not None:                         # a producer inside a producer: one record, the outer's phases
            self._joined = True; self._outer = outer
            outer._event("join", producer=self.producer, **self.params)
            return outer
        stack = getattr(_local, "stack", None)
        if stack is None:
            stack = _local.stack = []
        stack.append(self)
        self._event("start", producer=self.producer, resumed=self.resumed)
        if self.run_dir is not None:
            self._persist()
        self._sampler = threading.Thread(target=self._sample_loop, name=f"dmipy-sim run {self.id}", daemon=True)
        self._sampler.start()
        return self

    def __exit__(self, exc_type, exc, tb):
        if self._joined:
            return False
        self._stop.set()
        try:
            if self._dir is not None:
                self._sample()                        # the last resource row, before the end
            if exc is None:
                self.status = "ok"; self._event("end", status="ok")
            else:
                self.status = "error"
                self._event("end", status="error", error=repr(exc), traceback="".join(traceback.format_exception(exc_type, exc, tb)))
            for ph in self._phases:
                if ph.ended is None:
                    ph.ended = time.time()
            if self._dir is not None:
                json.dump(self.summary, open(os.path.join(self._dir, "summary.json"), "w"), indent=1)
            log.info("run %s: %s in %.0f s%s", self.id, self.status, time.time() - self.started,
                     f" (record {self._dir})" if self._dir else "")
        finally:
            stack = getattr(_local, "stack", None)
            if stack and stack[-1] is self:
                stack.pop()
            if self._fh is not None:
                self._fh.close(); self._fh = None
        return False

    # ------------------------------------------------------------------ what a producer reports
    def phase(self, name, **fields):
        """A phase begins (the previous open one ends)."""
        now = time.time()
        with self._lock:
            for ph in self._phases:
                if ph.ended is None:
                    ph.ended = now
            self._phases.append(_Phase(str(name), now))
        self._event("phase", name=str(name), **{k: _jsonable(v) for k, v in fields.items()})

    def _end_phase(self, name):
        now = time.time()
        with self._lock:
            for ph in reversed(self._phases):
                if ph.name == name and ph.ended is None:
                    ph.ended = now; break

    def progress(self, done, total, *, unit="walkers"):
        """Where the current phase is: rate and ETA from the phase's first report; written at most every
        ``PROGRESS_S`` (always when ``done == total``)."""
        now = time.time()
        with self._lock:
            ph = self._phases[-1] if self._phases else None
            if ph is None:
                ph = _Phase("run", self.started); self._phases.append(ph)
            if ph.first_progress is None:
                ph.first_progress = (now, float(done))
            t0, d0 = ph.first_progress
            rate = (float(done) - d0) / (now - t0) if now > t0 and done > d0 else None
            eta = (float(total) - float(done)) / rate if rate else None
            if done < total and now - ph.last_written < PROGRESS_S:
                return
            ph.last_written = now
        self._event("progress", phase=ph.name, done=float(done), total=float(total), unit=unit,
                    rate_per_s=rate, eta_s=eta, elapsed_s=now - self.started)
        if self._budget is not None:                 # the margin was crossed since: stop at this report
            raise self._budget

    def batches(self, n, size, *, unit="walkers", what=None):
        """The walker batches of a producer: yields ``(start, end)`` over ``n`` in batches of ``size`` (one batch
        when ``size`` is None or covers ``n``), logging each and reporting the progress -- the ONE batch loop of
        every producer (``what`` names it in the log: the producer by default)."""
        n = int(n); size = n if (size is None or int(size) >= n) else int(size)
        what = what or self.producer
        n_batches = -(-n // size); self._rss_batch = []; self._strikes = 0
        for b, start in enumerate(range(0, n, size)):
            end = min(start + size, n)
            self._check_budget(batches_left=n_batches - b)
            log.info("  %s: %s %d-%d (%d%%)...", what, unit, start, end - 1, int(100 * end / n))
            self.progress(start, n, unit=unit)
            yield start, end
        self.progress(n, n, unit=unit)

    def _check_budget(self, *, batches_left=None):
        """Raise the pending budget error, and at a batch boundary project the run's RSS to its end."""
        if self._budget is not None:
            raise self._budget
        if batches_left is None or not self._ceiling or BUDGET_FRACTION <= 0:
            return
        rss = _rss() or 0; self._rss_batch.append(rss)
        n_batches = len(self._rss_batch) + batches_left - 1
        warmup = max(BUDGET_WARMUP, int(np.ceil(BUDGET_WARMUP_SHARE * n_batches)))
        window = max(BUDGET_WINDOW, warmup)
        if len(self._rss_batch) > warmup:
            inc = np.diff(np.asarray(self._rss_batch[-(window + 1):], np.float64))
            growth = float(np.median(inc))                                  # the pace: a median, not the last jump
            projected = rss + max(growth, 0.0) * (batches_left - 1)
            if projected > BUDGET_FRACTION * self._ceiling:
                self._strikes += 1
                if self._strikes >= BUDGET_STRIKES:
                    self._fail_budget(rss, projected=projected, batches_left=batches_left, growth_per_batch=growth)
            else:
                self._strikes = 0

    def _fail_budget(self, rss, **fields):
        limit = BUDGET_FRACTION * self._ceiling
        msg = (f"the run would exceed its memory budget: host RSS {rss / 2 ** 30:.1f} GB"
               + (f", projected {fields['projected'] / 2 ** 30:.1f} GB over the {fields['batches_left']} batches left" if "projected" in fields else "")
               + f", against {limit / 2 ** 30:.1f} GB ({BUDGET_FRACTION:.0%} of the {self._ceiling / 2 ** 30:.1f} GB ceiling); stopped before the host "
               f"does it, the record and the spool intact")
        self._event("warning", message=msg, rss_bytes=rss, ceiling_bytes=self._ceiling, **fields)
        self._budget = ResourceBudgetError(msg)
        raise self._budget

    def spool(self, name, arrays, header=None):
        """A finished piece of the run's result -- a walk's batch -- written now, so a kill loses at most the piece
        in progress and a resumed run reads it back: ``spool/<name>.safetensors`` under the record (persisting the
        record at once), the arrays as tensors, ``header`` (JSON-able) in the metadata. Returns the path."""
        from safetensors.numpy import save_file
        self._persist()
        if self._dir is None:
            raise ValueError("the run is not persisted (DMIPY_SIM_RUN_DIR is empty): nothing to spool to")
        d = os.path.join(self._dir, "spool"); os.makedirs(d, exist_ok=True)
        path = os.path.join(d, f"{name}.safetensors"); tmp = path + ".part"
        save_file({k: np.ascontiguousarray(v) for k, v in arrays.items()}, tmp, metadata={"spool": json.dumps(header or {})})
        os.replace(tmp, path)                                            # whole or absent, never half
        self._event("spool", name=str(name), path=path, bytes=os.path.getsize(path))
        return path

    def spooled(self, name):
        """The arrays and header spooled under ``name`` by this run or the one it resumes, or ``None``."""
        root = self._dir if self._dir is not None else (self.run_dir if self.run_dir is not None else None)
        if root is None:
            return None
        path = os.path.join(root, "spool", f"{name}.safetensors")
        if not os.path.isfile(path):
            return None
        from safetensors import safe_open
        arrays = {}
        with safe_open(path, framework="numpy") as f:
            header = json.loads((f.metadata() or {}).get("spool") or "{}")
            for k in f.keys():
                arrays[k] = f.get_tensor(k)
        return arrays, header

    def artifact(self, path, **fields):
        """Something written to disk, as it is written."""
        self._event("artifact", path=str(path), **{k: _jsonable(v) for k, v in fields.items()})

    def warning(self, message, **fields):
        self._event("warning", message=str(message), **{k: _jsonable(v) for k, v in fields.items()})

    # ------------------------------------------------------------------ the record
    @property
    def dir(self):
        """Where the record is, once persisted (``None`` before)."""
        return self._dir

    @property
    def summary(self):
        """The run in a few hundred bytes: what a pack records about the run that made it."""
        now = time.time()
        phases = {}
        for ph in self._phases:
            phases[ph.name] = phases.get(ph.name, 0.0) + ((ph.ended or now) - ph.started)
        return dict(id=self.id, producer=self.producer, status=self.status or "running", started=_iso(self.started),
                    wall_s=now - self.started, phases_s=phases, peak_rss_bytes=self._peak_rss or None,
                    peak_device_bytes=self._peak_dev or None, host=socket.gethostname(), record=self._dir, code=_code())

    def _manifest(self):
        total, avail = _meminfo(); _, cg_max = _cgroup()
        return dict(id=self.id, producer=self.producer, params=self.params, started=_iso(self.started), code=_code(),
                    host=socket.gethostname(), platform=platform.platform(), python=sys.version.split()[0], pid=os.getpid(),
                    argv=sys.argv, devices=[d["device"] for d in _devices()], memory_total_bytes=total,
                    memory_ceiling_bytes=(cg_max if cg_max is not None else total),
                    xla_mem_fraction=os.environ.get("XLA_PYTHON_CLIENT_MEM_FRACTION"), sample_s=SAMPLE_S)

    def _persist(self):
        root = self.run_dir if self.run_dir is not None else run_root()
        if root is None or self._dir is not None:
            return
        d = root if self.run_dir is not None else os.path.join(root, self.id)
        os.makedirs(d, exist_ok=True)
        if not self.resumed:
            json.dump(self._manifest(), open(os.path.join(d, "manifest.json"), "w"), indent=1)
        fh = open(os.path.join(d, "events.jsonl"), "a")
        with self._lock:
            for row in self._events:
                fh.write(json.dumps(row) + "\n")
            self._events = []
            self._fh = fh; self._dir = d

    def _event(self, kind, **fields):
        row = dict(t=_iso(time.time()), kind=kind, **fields)
        with self._lock:
            if self._fh is not None:
                self._fh.write(json.dumps(row) + "\n"); self._fh.flush()
            else:
                self._events.append(row)
        if kind in ("phase", "end", "warning"):
            log.info("run %s: %s %s", self.producer, kind, {k: v for k, v in fields.items() if k != "traceback"})

    def _sample(self):
        rss = _rss(); total, avail = _meminfo(); cg_cur, cg_max = _cgroup(); dev = _devices(); where = _main_thread_frames()
        if rss:
            self._peak_rss = max(self._peak_rss, rss)
            if self._ceiling and BUDGET_FRACTION > 0 and self._budget is None and rss > BUDGET_FRACTION * self._ceiling:
                try:
                    self._fail_budget(rss)           # from the sampler: recorded now, raised at the next report
                except ResourceBudgetError:
                    pass
        for d in dev:
            if d.get("peak_bytes_in_use"):
                self._peak_dev = max(self._peak_dev, int(d["peak_bytes_in_use"]))
        self._event("resource", elapsed_s=time.time() - self.started, rss_bytes=rss, host_available_bytes=avail,
                    cgroup_bytes=cg_cur, cgroup_max_bytes=cg_max, devices=dev, where=where)

    def _sample_loop(self):
        while not self._stop.wait(SAMPLE_S):
            if self._dir is None:
                self._persist()                       # the run outlived the interval: it is worth a record
            self._sample()


def _iso(t):
    return _dt.datetime.utcfromtimestamp(t).strftime("%Y-%m-%dT%H:%M:%S.%f")[:-3] + "Z"



# ---------------------------------------------------------------------- reading records
def read_events(run_dir):
    """The rows of ``events.jsonl`` (a row cut short by a kill is skipped)."""
    rows = []
    try:
        for line in open(os.path.join(run_dir, "events.jsonl")):
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                pass
    except FileNotFoundError:
        pass
    return rows


def _parse(t):
    return _dt.datetime.strptime(t, "%Y-%m-%dT%H:%M:%S.%fZ").replace(tzinfo=_dt.timezone.utc).timestamp()


def report(run_dir):
    """What a record says: the manifest, the status (``ok`` / ``error`` / ``running`` / ``stale`` / ``killed``),
    the last progress (rate, ETA), the peak host and device memory per phase, the wall time per phase, the
    last event."""
    man = json.load(open(os.path.join(run_dir, "manifest.json")))
    rows = read_events(run_dir)
    sample_s = float(man.get("sample_s") or SAMPLE_S)
    ends = [r for r in rows if r["kind"] == "end"]
    hb = [r for r in rows if r["kind"] == "resource"]
    last_hb = _parse(hb[-1]["t"]) if hb else _parse(man["started"])
    if ends:
        status = ends[-1]["status"]
    elif time.time() - last_hb > STALE_INTERVALS * sample_s:
        status = "stale" if time.time() - last_hb < 100 * sample_s else "killed"
    else:
        status = "running"
    phase = None; peaks = {}; walls = {}; last_progress = None; phase_t0 = _parse(man["started"])
    for r in rows:
        if r["kind"] == "phase":
            if phase is not None:
                walls[phase] = walls.get(phase, 0.0) + _parse(r["t"]) - phase_t0
            phase = r["name"]; phase_t0 = _parse(r["t"])
        elif r["kind"] == "resource":
            p = peaks.setdefault(phase or "run", dict(rss_bytes=0, device_bytes=0))
            p["rss_bytes"] = max(p["rss_bytes"], r.get("rss_bytes") or 0)
            p["device_bytes"] = max([p["device_bytes"]] + [int(d.get("bytes_in_use") or 0) for d in r.get("devices") or []])
        elif r["kind"] == "progress":
            last_progress = r
    if phase is not None:
        t_end = _parse(ends[-1]["t"]) if ends else last_hb
        walls[phase] = walls.get(phase, 0.0) + t_end - phase_t0
    return dict(id=man["id"], producer=man["producer"], host=man.get("host"), started=man["started"], status=status,
                heartbeat_age_s=time.time() - last_hb, last_progress=last_progress, peaks=peaks, phases_s=walls,
                error=(ends[-1].get("error") if ends and ends[-1]["status"] == "error" else None),
                last_event=(rows[-1] if rows else None), manifest=man)


def list_runs(root=None):
    """Every record under ``root`` (the default root), newest first: ``report`` of each, without the manifest."""
    root = root or run_root()
    if not root or not os.path.isdir(root):
        return []
    out = []
    for name in sorted(os.listdir(root), reverse=True):
        d = os.path.join(root, name)
        if os.path.isfile(os.path.join(d, "manifest.json")):
            try:
                r = report(d); r.pop("manifest", None); r["dir"] = d; out.append(r)
            except Exception as e:
                out.append(dict(id=name, dir=d, status=f"unreadable ({e})"))
    return out


def _fmt_bytes(b):
    return "-" if not b else f"{b / 2 ** 30:.1f} GB"


def main(argv=None):
    """``python -m dmipy_sim.run`` (or ``list``): the runs on this machine; ``python -m dmipy_sim.run <dir>``: one."""
    argv = sys.argv[1:] if argv is None else argv
    if not argv or argv == ["list"]:
        for r in list_runs():
            lp = r.get("last_progress") or {}
            prog = f"{lp['done']:.0f}/{lp['total']:.0f} {lp.get('unit', '')} ETA {lp['eta_s'] / 60:.0f} min" if lp and lp.get("eta_s") is not None else \
                (f"{lp['done']:.0f}/{lp['total']:.0f}" if lp else "-")
            print(f"{r['id']:48s} {r['status']:8s} heartbeat {r.get('heartbeat_age_s', 0):6.0f} s ago  {prog}")
        return 0
    r = report(argv[0])
    print(f"{r['id']}  {r['producer']} on {r['host']}  started {r['started']}  status {r['status']}")
    if r["error"]:
        print(f"  error: {r['error']}")
    for ph, s in r["phases_s"].items():
        p = r["peaks"].get(ph, {})
        print(f"  {ph:24s} {s:8.0f} s   peak host {_fmt_bytes(p.get('rss_bytes')):>8s}   peak device {_fmt_bytes(p.get('device_bytes')):>8s}")
    lp = r["last_progress"]
    if lp:
        eta = f"ETA {lp['eta_s'] / 60:.1f} min" if lp.get("eta_s") is not None else ""
        print(f"  last progress: {lp['done']:.0f}/{lp['total']:.0f} {lp.get('unit', '')} in {lp['phase']} ({lp.get('rate_per_s') or 0:.1f}/s) {eta}")
    print(f"  last event: {r['last_event']['kind'] if r['last_event'] else '-'} at {r['last_event']['t'] if r['last_event'] else '-'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
