"""Low-overhead CUDA component timing for the runtime-audit training runs.

Enable with ``STCL_PROFILE_JSON=/path/to/profile.json``.  Training scripts mark
components with :func:`start`/:func:`end` and call :func:`epoch_start` and
:func:`epoch_end`.  CUDA events are synchronized only once per epoch, avoiding
the severe perturbation caused by synchronizing after every component.
"""
from __future__ import annotations

import atexit
import json
import os
import time
from collections import defaultdict
from pathlib import Path

import torch


_OUTPUT = os.environ.get("STCL_PROFILE_JSON")
_ACTIVE = bool(_OUTPUT)
_WARMUP_EPOCHS = int(os.environ.get("STCL_PROFILE_WARMUP_EPOCHS", "5"))
_PENDING = []
_TOTAL_MS = defaultdict(float)
_CALLS = defaultdict(int)
_EPOCHS = []
_MEASURED_STEPS = 0
_EPOCH_T0 = None
_WRITTEN = False


def active() -> bool:
    return _ACTIVE


def start(name: str):
    """Start a component interval and return an opaque token."""
    if not _ACTIVE:
        return None
    if torch.cuda.is_available():
        event = torch.cuda.Event(enable_timing=True)
        event.record()
        return ("cuda", name, event)
    return ("cpu", name, time.perf_counter())


def end(token) -> None:
    """End an interval returned by :func:`start`."""
    if token is None:
        return
    kind, name, begin = token
    if kind == "cuda":
        finish = torch.cuda.Event(enable_timing=True)
        finish.record()
        _PENDING.append((name, begin, finish))
    else:
        _PENDING.append((name, begin, time.perf_counter()))


def epoch_start(epoch: int) -> None:
    del epoch
    global _EPOCH_T0
    if _ACTIVE:
        _EPOCH_T0 = time.perf_counter()


def epoch_end(epoch: int, train_steps: int) -> None:
    """Flush one epoch of events.

    ``train_steps`` is the number of minibatches, used to report milliseconds
    per optimizer step.  Epochs at or below the configured warm-up count are
    retained in the raw epoch list but excluded from component means.
    """
    global _EPOCH_T0, _MEASURED_STEPS
    if not _ACTIVE:
        return
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    wall_s = None if _EPOCH_T0 is None else time.perf_counter() - _EPOCH_T0
    measured = epoch > _WARMUP_EPOCHS
    epoch_ms = defaultdict(float)
    epoch_calls = defaultdict(int)
    for name, begin, finish in _PENDING:
        if isinstance(begin, torch.cuda.Event):
            elapsed_ms = begin.elapsed_time(finish)
        else:
            elapsed_ms = (finish - begin) * 1000.0
        epoch_ms[name] += float(elapsed_ms)
        epoch_calls[name] += 1
        if measured:
            _TOTAL_MS[name] += float(elapsed_ms)
            _CALLS[name] += 1
    _PENDING.clear()
    if measured:
        _MEASURED_STEPS += int(train_steps)
    _EPOCHS.append(
        {
            "epoch": int(epoch),
            "measured": measured,
            "train_steps": int(train_steps),
            "wall_s": wall_s,
            "component_ms": dict(epoch_ms),
            "component_calls": dict(epoch_calls),
        }
    )
    _EPOCH_T0 = None
    _write()


def _metadata() -> dict:
    return {
        "pde": os.environ.get("STCL_PROFILE_PDE"),
        "method": os.environ.get("STCL_PROFILE_METHOD"),
        "n": int(os.environ["STCL_PROFILE_N"]) if os.environ.get("STCL_PROFILE_N") else None,
        "epochs_requested": (
            int(os.environ["STCL_PROFILE_EPOCHS"])
            if os.environ.get("STCL_PROFILE_EPOCHS")
            else None
        ),
        "warmup_epochs_excluded": _WARMUP_EPOCHS,
        "device": torch.cuda.get_device_name() if torch.cuda.is_available() else "cpu",
    }


def _write() -> None:
    global _WRITTEN
    if not _ACTIVE:
        return
    components = {}
    for name in sorted(_TOTAL_MS):
        total = _TOTAL_MS[name]
        calls = _CALLS[name]
        components[name] = {
            "total_ms": total,
            "calls": calls,
            "mean_call_ms": total / max(calls, 1),
            "mean_per_train_step_ms": total / max(_MEASURED_STEPS, 1),
        }
    measured_epoch_wall = [
        row["wall_s"] for row in _EPOCHS if row["measured"] and row["wall_s"] is not None
    ]
    payload = {
        "metadata": _metadata(),
        "measured_epochs": sum(row["measured"] for row in _EPOCHS),
        "measured_train_steps": _MEASURED_STEPS,
        "mean_epoch_wall_s": (
            sum(measured_epoch_wall) / len(measured_epoch_wall)
            if measured_epoch_wall
            else None
        ),
        "components": components,
        "epochs": _EPOCHS,
        "timing_note": (
            "CUDA-event component times; one synchronization per epoch. "
            "Nested derivative_total includes online_jvp, pde_residual, and "
            "preconditioner_krylov where applicable."
        ),
    }
    path = Path(_OUTPUT)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w") as f:
        json.dump(payload, f, indent=2)
    tmp.replace(path)
    _WRITTEN = True


@atexit.register
def _finalize() -> None:
    if _ACTIVE and not _WRITTEN:
        _write()
