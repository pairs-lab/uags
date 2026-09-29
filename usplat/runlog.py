"""
runlog.py — run records (K14) and timing / memory instrumentation (K15).

Every number in a table must come from a JSON record that says what produced it:

  RunRecord(path, config=..., seed=...)   context manager; writes `path` (JSON)
    run      : name, id, argv, command (the interpreter command line, sys.orig_argv, so
               flags such as -O and -m are kept), cwd, executable, host, start/end (ISO
               8601, local zone), wall_s, cpu_s, status ("running" -> "ok" | "failed"),
               error + traceback
    code     : git commit, branch, dirty flag, sha256 of `git diff HEAD --binary
               --full-index` (tracked changes, staged or not; diff options pinned so user
               git config cannot change the bytes), untracked source and config files (.py,
               .sh, .json, .yaml, .txt, ...) with their sha256, and a content-addressed patch
               <dir>/patches/<sha256>.patch (tracked diff + untracked files) that `git apply`
               turns the commit back into the code that ran
    env      : python / torch / numpy / gsplat / open_clip versions, CUDA build (cuDNN only
               if the run initialised CUDA: querying it would create a context),
               GPU name, total and used memory (nvidia-smi: no CUDA context is created),
               CPU model, thread count, and a few env vars (never the whole environment)
    seed     : set with set_seed() on entry when given
    config   : canonical JSON (sorted keys, no spaces, non-finite floats as "inf"/"nan")
               and its sha256, so equal configs have equal hashes whatever the key order
    inputs   : data files the run read, record.add_input(path) (size and sha256)
    memory   : peak torch.cuda.max_memory_allocated / _reserved over the run (only when
               the run used CUDA), peak process RSS
    metrics  : user values, record.add(key, value); tensors and numpy values are converted
    timing   : record.timer.summary(): per named section, count, total and ms statistics

The record is written on entry (status "running": a killed job leaves a trace), on every
flush(), and on exit. On an exception the record is still written, with status "failed",
the exception type, message and traceback; the exception is re-raised. If the record
itself cannot be completed (e.g. a CUDA query fails after a device error), a smaller
fallback record is written with `record_error` set: status "failed" when the run raised
(the run's exception still reaches the caller), "incomplete" when it did not (the record
error is then raised). A failure inside __enter__ (bad config or seed) writes a "failed"
record too. Every value is passed through to_jsonable() before it is written, so numpy /
torch scalars, paths and NaN never break the write.

Timer: named sections with cumulative wall time. When CUDA is initialised, each section
boundary calls torch.cuda.synchronize(), so asynchronous kernels are charged to the section
that launched them. A section run once per keyframe gives ms per keyframe directly
(mean_ms); summary(per=n) also divides every total by n (e.g. n keyframes). The first
`warmup` calls of a section (JIT builds, cuDNN autotune) are left out of the ms statistics
but kept in total_s (and so in ms_per_<unit>). With memory=True a section also reports its
own peak allocated memory.

Peak memory: torch keeps one peak counter per device. RunRecord (on entry) and sections
reset it to measure their own peak; every reset first folds the counter into all open
watchers (the RunRecord and the enclosing sections), so nested measurements stay correct.
Code that calls torch.cuda.reset_peak_memory_stats() directly bypasses this and loses the
earlier peak. Only the current CUDA device is measured.

CUDA is never initialised by this module: synchronisation and memory queries run only if
torch.cuda.is_initialized(), so CPU runs on the shared machine create no GPU context.
"""
from __future__ import annotations
import argparse
import dataclasses
import datetime as _dt
import enum
import hashlib
import json
import math
import operator
import os
import pathlib
import platform
import random
import resource
import shlex
import socket
import subprocess
import sys
import time
import traceback
import uuid
import warnings
from contextlib import contextmanager

import numpy as np
import torch

SCHEMA = "uags.runrecord/1"
# untracked files that count as code (hashed and put in the patch): sources and configs
# (e.g. configs/vocab_replica.json sets the Tab I class list). Run records are skipped.
SOURCE_SUFFIXES = (".py", ".sh", ".cu", ".cuh", ".cpp", ".cc", ".c", ".h", ".hpp",
                   ".yaml", ".yml", ".toml", ".cfg", ".ini", ".json", ".txt")
# never part of the code state: bytecode (tracked in this repo), reports, paper, outputs.
# git glob pathspecs, anchored at the repo top ("**/" matches in every directory)
DEFAULT_EXCLUDE = ("**/*.pyc", "docs/**", "papers/**", "outputs/**")
ENV_KEYS = ("PYTHONPATH", "CUDA_HOME", "CUDA_VISIBLE_DEVICES", "UAGS_GPU_MEM_FRACTION",
            "OMP_NUM_THREADS", "MKL_NUM_THREADS", "PYTHONHASHSEED", "CUBLAS_WORKSPACE_CONFIG")
_MAX_HASH_BYTES = 256 * 2 ** 20            # larger untracked files: size only
_MAX_PATCH_BYTES = 16 * 2 ** 20            # larger untracked files: hashed, not in the patch


# ---------------------------------------------------------------- seeds -----------------
def set_seed(seed: int, deterministic: bool = False) -> int:
    """Seed python `random`, numpy and torch (CPU and, lazily, every CUDA device).

    deterministic=True also asks torch for deterministic kernels (warn_only) and turns off
    the cuDNN autotuner; set CUBLAS_WORKSPACE_CONFIG before CUDA starts for cuBLAS.
    PYTHONHASHSEED cannot be changed for a running interpreter (set it in the shell).
    """
    seed = int(seed)
    random.seed(seed)
    np.random.seed(seed % 2 ** 32)
    torch.manual_seed(seed)                 # also queues torch.cuda.manual_seed_all
    if deterministic:
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
        torch.use_deterministic_algorithms(True, warn_only=True)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    return seed


# ---------------------------------------------------------------- JSON ------------------
# strict mode: a dict with any non-str key is written as {_ITEMS: [[key, value], ...]}, so
# {1: x} and {"1": x} hash differently; a str-keyed dict may not use this key itself
_ITEMS = "__dict_items__"


def _strict_key(k):
    if isinstance(k, np.generic):
        k = k.item()
    if k is None or isinstance(k, (str, int)):          # bool is an int, JSON keeps true
        return k
    raise TypeError(f"no canonical JSON form for dict key of type {type(k).__name__}: {k!r}")


def to_jsonable(obj, strict: bool = False):
    """Plain JSON value for obj. Non-finite floats become "inf" / "-inf" / "nan" strings
    (float() parses them back), so the file is strict JSON. strict=True raises on types
    without a stable text form instead of falling back to repr() (used for config hashes);
    dict keys must then be str, int, bool or None (see _ITEMS).
    """
    if obj is None or isinstance(obj, (bool, str)):
        return obj
    if isinstance(obj, int):
        return int(obj)
    if isinstance(obj, float):
        if math.isfinite(obj):
            return float(obj)
        return "nan" if math.isnan(obj) else ("inf" if obj > 0 else "-inf")
    if isinstance(obj, np.generic):
        return to_jsonable(obj.item(), strict)
    if isinstance(obj, (np.ndarray, torch.Tensor)):
        if isinstance(obj, torch.Tensor):
            obj = obj.detach().cpu()
        return to_jsonable(obj.tolist(), strict)
    if isinstance(obj, dict):
        if not strict:
            return {str(k): to_jsonable(v) for k, v in obj.items()}
        keys = [_strict_key(k) for k in obj]
        vals = [to_jsonable(v, True) for v in obj.values()]
        if all(isinstance(k, str) for k in keys):
            if _ITEMS in obj:
                raise TypeError(f"config key {_ITEMS!r} is reserved")
            return dict(zip(keys, vals))
        items = [[k, v] for k, v in zip(keys, vals)]
        return {_ITEMS: sorted(items, key=lambda kv: json.dumps(kv[0]))}
    if isinstance(obj, (list, tuple)):
        return [to_jsonable(v, strict) for v in obj]
    if isinstance(obj, (set, frozenset)):
        items = [to_jsonable(v, strict) for v in obj]
        return sorted(items, key=lambda v: json.dumps(v, sort_keys=True))
    if isinstance(obj, argparse.Namespace):
        return to_jsonable(vars(obj), strict)
    if hasattr(obj, "to_dict") and callable(obj.to_dict):
        return to_jsonable(obj.to_dict(), strict)
    if dataclasses.is_dataclass(obj) and not isinstance(obj, type):
        return to_jsonable({f.name: getattr(obj, f.name) for f in dataclasses.fields(obj)},
                           strict)
    if isinstance(obj, enum.Enum):
        return to_jsonable(obj.value, strict)
    if isinstance(obj, (pathlib.PurePath, torch.dtype, torch.device)):
        return str(obj)
    if strict:
        raise TypeError(f"no canonical JSON form for {type(obj).__name__}: {obj!r}")
    return repr(obj)


def canonical_json(obj) -> str:
    """Sorted keys, no whitespace, ASCII: equal values give equal text. int 1 and float
    1.0 stay different ("1" vs "1.0")."""
    return json.dumps(to_jsonable(obj, strict=True), sort_keys=True,
                      separators=(",", ":"), ensure_ascii=True, allow_nan=False)


def config_hash(config) -> str:
    """sha256 (hex) of canonical_json(config)."""
    return hashlib.sha256(canonical_json(config).encode("ascii")).hexdigest()


def _sha256_file(path, chunk=2 ** 20):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(chunk), b""):
            h.update(b)
    return h.hexdigest()


def _write_json_atomic(path, data):
    """Write to_jsonable(data) as strict JSON. Serialised before the temp file is opened,
    and the temp file is removed on any failure, so a failed write leaves no debris."""
    path = pathlib.Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    text = json.dumps(to_jsonable(data), indent=2, sort_keys=False, allow_nan=False) + "\n"
    tmp = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        with open(tmp, "w") as f:
            f.write(text)
        os.replace(tmp, path)
    except BaseException:
        try:
            tmp.unlink()
        except OSError:
            pass
        raise


def load_record(path) -> dict:
    with open(path) as f:
        return json.load(f)


# ---------------------------------------------------------------- git -------------------
def _git(repo, *args, ok=(0,)):
    # user config must not change the bytes that get hashed: colour, path quoting, blank
    # context lines, prefixes (the diff options themselves are pinned in _DIFF_FLAGS)
    r = subprocess.run(["git", "-C", str(repo), "-c", "color.ui=never",
                        "-c", "core.quotePath=true", "-c", "diff.suppressBlankEmpty=false",
                        "-c", "diff.noprefix=false", "-c", "diff.mnemonicPrefix=false", *args],
                       capture_output=True, timeout=120)
    if r.returncode not in ok:
        raise RuntimeError(f"git {' '.join(args)}: {r.stderr.decode(errors='replace').strip()}")
    return r.stdout


# --full-index: text hunks otherwise abbreviate blob ids by core.abbrev, whose default
# ("auto") grows with the object count; the rest overrides the diff.context / algorithm /
# indentHeuristic / interHunkContext / orderFile / submodule config
_DIFF_FLAGS = ("--binary", "--full-index", "--no-color", "--no-ext-diff", "--no-textconv",
               "--src-prefix=a/", "--dst-prefix=b/", "--no-renames", "--unified=3",
               "--inter-hunk-context=0", "--diff-algorithm=myers", "--indent-heuristic",
               "-O/dev/null", "--submodule=short")
_RECORD_HEAD = b'"schema": "' + SCHEMA.split("/")[0].encode()


def _is_run_record(path):
    """A JSON file written by RunRecord (its first key is the schema): output, not code."""
    try:
        with open(path, "rb") as f:
            return _RECORD_HEAD in f.read(256)
    except OSError:
        return False


def git_state(repo=None, exclude=DEFAULT_EXCLUDE, patch_dir=None):
    """Code state of the git repo containing `repo` (default: this package's repo).

    diff_sha256 hashes `git diff HEAD --binary --full-index` minus the `exclude` pathspecs
    (git globs, anchored at the repo top), so it covers staged and unstaged edits to
    tracked files.
    Untracked files (not git-ignored, not excluded by the same pathspecs) whose suffix is in
    SOURCE_SUFFIXES are listed with their sha256; JSON run records are skipped.
    code_sha256 hashes the tracked diff hash plus every untracked (path, sha256, bytes).
    With patch_dir, the tracked diff plus every untracked source file up to
    _MAX_PATCH_BYTES (as a new-file diff) is saved once under patch_dir/<sha256>.patch. File names that are not
    UTF-8 are kept with os.fsdecode (surrogate escapes). Returns {"error": ...} outside a
    git repo.
    """
    repo = pathlib.Path(repo) if repo else pathlib.Path(__file__).resolve().parent
    try:
        top = pathlib.Path(os.fsdecode(_git(repo, "rev-parse", "--show-toplevel").strip()))
        commit = _git(top, "rev-parse", "HEAD").decode().strip()
        branch = _git(top, "rev-parse", "--abbrev-ref", "HEAD").decode(errors="replace").strip()
        spec = [f":(exclude,glob){p}" for p in exclude]
        diff = _git(top, "diff", *_DIFF_FLAGS, "HEAD", "--", ".", *spec)
        names = _git(top, "ls-files", "--others", "--exclude-standard", "-z", "--", ".", *spec)
        names = sorted(os.fsdecode(n) for n in names.split(b"\0") if n)
    except (RuntimeError, OSError, subprocess.SubprocessError, UnicodeError) as e:
        return {"error": str(e)}
    untracked, parts, complete = [], [diff], True
    for rel in names:
        if not rel.endswith(SOURCE_SUFFIXES):
            continue
        f = top / rel
        if rel.endswith(".json") and _is_run_record(f):
            continue
        try:
            size = f.stat().st_size
        except OSError:                                 # e.g. a broken symlink
            untracked.append({"path": rel, "bytes": None, "sha256": None})
            complete = False
            continue
        entry = {"path": rel, "bytes": size,
                 "sha256": _sha256_file(f) if size <= _MAX_HASH_BYTES else None}
        untracked.append(entry)
        if patch_dir is None:
            continue
        try:
            if entry["sha256"] is None or size > _MAX_PATCH_BYTES:
                raise RuntimeError("too large")
            parts.append(_git(top, "diff", *_DIFF_FLAGS, "--no-index", "--", "/dev/null", rel,
                              ok=(0, 1)))
        except (RuntimeError, OSError, subprocess.SubprocessError):
            complete = False                            # listed and hashed, not in the patch
    diff_sha = hashlib.sha256(diff).hexdigest()
    # one id for the whole code state: tracked diff + every untracked file's name, hash and
    # size (files too large to hash count by size only)
    state = diff_sha + "".join(f"\n{u['path']}\0{u['sha256']}\0{u['bytes']}" for u in untracked)
    out = {"repo": str(top), "commit": commit, "branch": branch,
           "dirty": bool(diff) or bool(untracked), "dirty_tracked": bool(diff),
           "diff_sha256": diff_sha, "diff_bytes": len(diff),
           "code_sha256": hashlib.sha256(state.encode("utf-8", "surrogateescape")).hexdigest(),
           "exclude": list(exclude), "untracked": untracked}
    if patch_dir is not None and out["dirty"]:
        patch = b"".join(parts)
        digest = hashlib.sha256(patch).hexdigest()
        pdir = pathlib.Path(patch_dir)
        pdir.mkdir(parents=True, exist_ok=True)
        pfile = pdir / f"{digest}.patch"
        if not pfile.exists():
            tmp = pdir / f".{digest}.{os.getpid()}.tmp"
            tmp.write_bytes(patch)
            os.replace(tmp, pfile)
        # patch_complete False: some untracked file is only listed (unreadable or too large)
        out.update(patch_sha256=digest, patch_file=str(pfile), patch_complete=complete)
    return out


# ---------------------------------------------------------------- environment -----------
def _version(dist):
    try:
        from importlib.metadata import version
        return version(dist)
    except Exception:
        return None


def _nvidia_smi():
    """GPU facts without creating a CUDA context. None if nvidia-smi is missing."""
    q = "name,memory.total,memory.used,driver_version"
    try:
        r = subprocess.run(["nvidia-smi", f"--query-gpu={q}", "--format=csv,noheader,nounits"],
                           capture_output=True, timeout=20, text=True)
    except (OSError, subprocess.SubprocessError):
        return None
    if r.returncode != 0:
        return None
    gpus = []
    for line in r.stdout.strip().splitlines():
        f = [x.strip() for x in line.split(",")]
        if len(f) == 4:
            num = lambda s: int(s) if s.isdigit() else None
            gpus.append({"name": f[0], "memory_total_mib": num(f[1]),
                         "memory_used_mib": num(f[2]), "driver": f[3]})
    return gpus


def _cpu_model():
    try:
        with open("/proc/cpuinfo") as f:
            for line in f:
                if line.startswith("model name"):
                    return line.split(":", 1)[1].strip()
    except OSError:
        pass
    return platform.processor() or None


def environment():
    return {
        "python": platform.python_version(),
        "torch": torch.__version__, "numpy": np.__version__,
        "gsplat": _version("gsplat"), "open_clip_torch": _version("open_clip_torch"),
        "torch_cuda_build": torch.version.cuda,
        "platform": platform.platform(),
        "host": socket.gethostname(),
        "cpu": _cpu_model(), "cpu_count": os.cpu_count(),
        "torch_threads": torch.get_num_threads(),
        "gpus": _nvidia_smi(),
        "env": {k: os.environ[k] for k in ENV_KEYS if k in os.environ},
    }


# ---------------------------------------------------------------- GPU peak memory -------
def _cuda_on():
    return torch.cuda.is_initialized()


class _PeakWatch:
    """Running max of the device peak counters over the watcher's lifetime."""
    __slots__ = ("alloc", "reserved")

    def __init__(self):
        self.alloc = 0
        self.reserved = 0


_WATCHES: list = []


def _fold():
    if not _cuda_on():
        return
    a, r = torch.cuda.max_memory_allocated(), torch.cuda.max_memory_reserved()
    for w in _WATCHES:
        w.alloc = max(w.alloc, a)
        w.reserved = max(w.reserved, r)


def _watch_start():
    """New watcher starting now: fold the counter into the open ones, then reset it (the
    reset sets the peak to the memory allocated right now)."""
    _fold()
    if _cuda_on():
        torch.cuda.reset_peak_memory_stats()
    w = _PeakWatch()
    _WATCHES.append(w)
    return w


def _watch_stop(w):
    try:
        _fold()
    finally:                                            # never leak a watcher
        if w in _WATCHES:
            _WATCHES.remove(w)
    return w


# ---------------------------------------------------------------- Timer -----------------
class Timer:
    """Named wall-clock sections with cumulative totals.

        timer = Timer()
        for frame in frames:
            with timer.section("keyframe"):
                with timer.section("clip"):
                    ...
        timer.summary(per=len(frames))   # also ms per keyframe for every section

    sync: None = synchronise CUDA at section boundaries when CUDA is initialised;
    False = never. clock: seconds counter (injectable for tests). A section that raises
    is still timed, and an error while closing it (e.g. the CUDA sync after a device
    error) never replaces the section's own exception. Sections may nest; a name cannot be
    re-entered while it is open.
    """

    def __init__(self, sync=None, warmup: int = 0, memory: bool = False,
                 clock=time.perf_counter):
        self.sync, self.warmup, self.memory, self.clock = sync, int(warmup), memory, clock
        self._times: dict[str, list] = {}
        self._peak: dict[str, int] = {}
        self._open: dict = {}

    def _sync(self):
        if self.sync is not False and _cuda_on():
            torch.cuda.synchronize()

    def start(self, name: str):
        if name in self._open:
            raise RuntimeError(f"timer section {name!r} is already open")
        w = _watch_start() if self.memory else None
        self._sync()
        self._open[name] = (self.clock(), w)

    def stop(self, name: str) -> float:
        if name not in self._open:
            raise RuntimeError(f"timer section {name!r} is not open")
        self._sync()
        t0, w = self._open.pop(name)
        dt = self.clock() - t0
        self._times.setdefault(name, []).append(dt)
        if w is not None:
            _watch_stop(w)
            if _cuda_on():
                self._peak[name] = max(self._peak.get(name, 0), w.alloc)
        return dt

    @contextmanager
    def section(self, name: str):
        self.start(name)
        try:
            yield
        except BaseException:
            try:
                self.stop(name)
            except Exception:                           # keep the body's exception
                _, w = self._open.pop(name, (None, None))
                if w in _WATCHES:
                    _WATCHES.remove(w)
            raise
        self.stop(name)

    def add(self, name: str, seconds: float):
        """Add an externally measured duration as one call of `name`."""
        self._times.setdefault(name, []).append(float(seconds))

    def total(self, name: str) -> float:
        return float(sum(self._times.get(name, ())))

    def count(self, name: str) -> int:
        return len(self._times.get(name, ()))

    def summary(self, per=None, unit: str = "keyframe") -> dict:
        """{name: {count, total_s, n_warmup, mean_ms, std_ms, median_ms, p90_ms, min_ms,
        max_ms[, ms_per_<unit>][, peak_alloc_bytes]}}. The ms statistics use the calls
        after the first `warmup` ones (None if no call is left; std is the population std,
        ddof=0; p90 is numpy's linear percentile); total_s uses all calls.
        per: divide each total by this count (e.g. keyframes) -> ms_per_<unit>, warm-up
        calls included."""
        per = float(per) if per else None               # numpy / torch scalars too
        out = {}
        for name, ts in self._times.items():
            post = np.asarray(ts[self.warmup:], dtype=np.float64) * 1e3
            s = {"count": len(ts), "total_s": float(sum(ts)),
                 "n_warmup": min(self.warmup, len(ts))}
            if post.size:
                s.update(mean_ms=float(post.mean()), std_ms=float(post.std()),
                         median_ms=float(np.median(post)),
                         p90_ms=float(np.percentile(post, 90)),
                         min_ms=float(post.min()), max_ms=float(post.max()))
            else:
                s.update(mean_ms=None, std_ms=None, median_ms=None, p90_ms=None,
                         min_ms=None, max_ms=None)
            if per:
                s[f"ms_per_{unit}"] = float(sum(ts)) * 1e3 / per
            if name in self._peak:
                s["peak_alloc_bytes"] = int(self._peak[name])
            out[name] = s
        return out

    def report(self, per=None, unit: str = "keyframe") -> str:
        """Plain-text table of summary()."""
        rows = [f"{'section':<24}{'n':>6}{'total s':>10}{'mean ms':>10}{'p90 ms':>10}"
                + (f"{'ms/' + unit:>14}" if per else "")]
        f = lambda v: "-" if v is None else f"{v:.2f}"
        for name, s in self.summary(per, unit).items():
            rows.append(f"{name:<24}{s['count']:>6}{s['total_s']:>10.3f}{f(s['mean_ms']):>10}"
                        f"{f(s['p90_ms']):>10}"
                        + (f"{f(s['ms_per_' + unit]):>14}" if per else ""))
        return "\n".join(rows)

    def reset(self):
        if self._open:
            raise RuntimeError(f"cannot reset with open sections {list(self._open)}")
        self._times.clear()
        self._peak.clear()


# ---------------------------------------------------------------- RunRecord -------------
def _now():
    return _dt.datetime.now().astimezone().isoformat(timespec="milliseconds")


def _error(exc_type, exc, tb, **extra):
    return {"type": exc_type.__name__, "message": str(exc),
            "traceback": "".join(traceback.format_exception(exc_type, exc, tb)), **extra}


class RunRecord:
    """Context manager that writes a JSON record of one experiment run to `path`.

        with RunRecord("outputs/runs/tab2_scene01.json", config=vars(args), seed=0,
                       name="tab2_scene01") as rec:
            rec.add_input(args.vocab)           # data / config files the run read
            with rec.timer.section("keyframe"):
                ...
            rec.add("ap_stale", ap)
            rec.timing_per = n_keyframes        # optional: ms per keyframe in the summary

    path: made absolute at construction, so a chdir during the run changes nothing.
    config: dict, argparse.Namespace, dataclass or object with to_dict(); hashed with
    config_hash(). seed: an integer (numpy / torch integers too) passed to set_seed() on
    entry (None = not seeded, recorded as null). save_patch: store the uncommitted code as
    <path dir>/patches/<sha256>.patch. argv: the script's argument list (default
    sys.argv); `command` is the full interpreter command line (sys.orig_argv) unless argv
    differs from sys.argv (tests), then it is rebuilt from the executable name and argv.
    """

    def __init__(self, path, config=None, seed=None, name=None, repo=None,
                 save_patch=True, exclude=DEFAULT_EXCLUDE, deterministic=False,
                 timer: Timer | None = None, argv=None, tags=None):
        self.path = pathlib.Path(path).absolute()
        self.config = {} if config is None else config
        self.seed = None if seed is None else int(operator.index(seed))
        self.deterministic = bool(deterministic)
        self.name = str(name) if name else self.path.stem
        self.repo, self.save_patch, self.exclude = repo, save_patch, tuple(exclude)
        self.timer = timer if timer is not None else Timer()
        self.argv = [str(a) for a in (sys.argv if argv is None else argv)]
        orig = [str(a) for a in getattr(sys, "orig_argv", None) or ()]
        if orig and self.argv == [str(a) for a in sys.argv]:
            self.command = shlex.join(orig)
        else:
            self.command = shlex.join([os.path.basename(sys.executable)] + self.argv)
        self.orig_argv = orig or None
        self.tags = list(tags or [])
        self.metrics: dict = {}
        self.inputs: list = []
        self.outputs: list = []
        self.timing_per, self.timing_unit = None, "keyframe"
        self.data: dict = {}
        self._watch = None
        self._t0 = self._c0 = None

    # ---- user API ----
    def add(self, key: str, value):
        """Store a metric (overwrites an earlier value of the same key)."""
        self.metrics[str(key)] = to_jsonable(value)
        return value

    def update(self, values: dict):
        for k, v in values.items():
            self.add(k, v)

    @staticmethod
    def _file_entry(path, hash):
        p = pathlib.Path(path)
        if p.is_dir():                                  # a directory: listed, not hashed
            return {"path": str(p.resolve()), "bytes": None, "sha256": None, "dir": True}
        return {"path": str(p.resolve()), "bytes": p.stat().st_size,
                "sha256": _sha256_file(p) if hash else None}

    def add_input(self, path, hash: bool = True):
        """List an input file the run read (vocabulary, scene list, .ply, checkpoint) with
        its size and, if hash, sha256: data files are not part of the git code state."""
        self.inputs.append(self._file_entry(path, hash))

    def add_output(self, path, hash: bool = True):
        """List an output file (size and, if hash, sha256) in the record."""
        self.outputs.append(self._file_entry(path, hash))

    def flush(self):
        """Write the record now (status stays "running")."""
        self._snapshot()
        _write_json_atomic(self.path, self.data)

    # ---- internals ----
    def _snapshot(self):
        d = self.data
        d["metrics"] = self.metrics
        d["inputs"] = self.inputs
        d["outputs"] = self.outputs
        d["timing"] = self.timer.summary(self.timing_per, self.timing_unit)
        d["timing_per"] = ({"unit": self.timing_unit, "count": self.timing_per}
                           if self.timing_per else None)
        d["wall_s"] = time.perf_counter() - self._t0
        d["cpu_s"] = time.process_time() - self._c0
        d["deterministic_algorithms"] = torch.are_deterministic_algorithms_enabled()
        _fold()
        cuda = _cuda_on()          # CUDA cannot be de-initialised: True if it ever ran
        d["memory"] = {
            "cuda_initialized": cuda,
            "cuda_device": (torch.cuda.get_device_name(torch.cuda.current_device())
                            if cuda else None),
            "peak_gpu_alloc_bytes": int(self._watch.alloc) if cuda else None,
            "peak_gpu_reserved_bytes": int(self._watch.reserved) if cuda else None,
            "peak_gpu_alloc_gib": self._watch.alloc / 2 ** 30 if cuda else None,
            # Linux ru_maxrss is in KiB; peak over the whole process, not only this run
            "peak_rss_bytes": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024,
        }

    def _write_fallback(self, status, error, record_error):
        """Minimal record when the full one cannot be built; never raises."""
        d = {"schema": SCHEMA, "name": self.name, "run_id": self.data.get("run_id"),
             "status": status, "argv": self.argv, "command": self.command,
             "cwd": self.data.get("cwd", os.getcwd()), "start": self.data.get("start"),
             "end": _now(), "seed": self.seed, "config_sha256": self.data.get("config_sha256"),
             "wall_s": None if self._t0 is None else time.perf_counter() - self._t0,
             "metrics": self.metrics, "inputs": self.inputs, "outputs": self.outputs,
             "error": error, "record_error": record_error}
        try:
            _write_json_atomic(self.path, d)
        except Exception as e:
            warnings.warn(f"run record {self.path} could not be written: {e!r}", RuntimeWarning)

    def __enter__(self):
        self._t0, self._c0 = time.perf_counter(), time.process_time()
        self.data = {"run_id": uuid.uuid4().hex, "cwd": os.getcwd(), "start": _now()}
        try:
            cfg = to_jsonable(self.config, strict=True)
            chash = config_hash(cfg)
            if self.seed is not None:
                set_seed(self.seed, self.deterministic)
            self._watch = _watch_start()
            patch_dir = self.path.parent / "patches" if self.save_patch else None
            self.data = {
                "schema": SCHEMA,
                "name": self.name,
                "run_id": self.data["run_id"],
                "status": "running",
                "tags": self.tags,
                "argv": self.argv,
                "command": self.command,
                "orig_argv": self.orig_argv,
                "cwd": self.data["cwd"],
                "executable": sys.executable,
                "pid": os.getpid(),
                "start": self.data["start"],
                "end": None,
                "seed": self.seed,
                "deterministic": self.deterministic,
                "config": cfg,
                "config_sha256": chash,
                "git": git_state(self.repo, self.exclude, patch_dir),
                "env": environment(),
                "error": None,
                "record_error": None,
            }
            self.flush()
        except BaseException as e:                      # no __exit__ follows: clean up here
            if self._watch is not None:
                try:
                    _watch_stop(self._watch)                # removes it even if _fold fails
                except Exception:
                    pass
                self._watch = None
            self._write_fallback("failed", _error(type(e), e, e.__traceback__, stage="enter"),
                                 None)
            raise
        return self

    def __exit__(self, exc_type, exc, tb):
        ok = exc_type is None or (issubclass(exc_type, SystemExit)
                                  and exc.code in (None, 0))
        err = None if ok else _error(exc_type, exc, tb)
        try:
            try:
                self._snapshot()
            finally:
                _watch_stop(self._watch)
            self.data["end"] = _now()
            self.data["status"] = "ok" if ok else "failed"
            # cudnn.version() initialises CUDA: ask only if the run already did
            self.data["env"]["cudnn"] = torch.backends.cudnn.version() if _cuda_on() else None
            gpus = _nvidia_smi()
            self.data["env"]["gpu_used_mib_at_end"] = (
                [g["memory_used_mib"] for g in gpus] if gpus else None)
            self.data["error"] = err
            _write_json_atomic(self.path, self.data)
        except Exception as e:
            rec_err = _error(type(e), e, e.__traceback__)
            self._write_fallback("incomplete" if ok else "failed", err, rec_err)
            if exc_type is None:
                raise                                   # the run itself did not fail
            warnings.warn(f"run record {self.path} is incomplete: {e!r}", RuntimeWarning)
        return False                                    # never swallow the run's exception
