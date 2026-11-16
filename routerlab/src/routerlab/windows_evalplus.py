"""Windows compatibility layer for EvalPlus' untrusted checker.

EvalPlus 0.3.1 relies on Unix-only ``resource`` limits and ``SIGALRM``.
This module keeps EvalPlus' execution and oracle logic, but supplies Windows
process, timeout, and memory-limit handling.
"""

from __future__ import annotations

import contextlib
import multiprocessing
import os
import _thread
import threading
import time
from multiprocessing import Array, Value
from typing import Any

import numpy as np
import psutil

from evalplus.config import DEFAULT_GT_TIME_LIMIT_FACTOR, DEFAULT_MIN_TIME_LIMIT
from evalplus.eval import FAIL, PASS, TIMEOUT, query_maximum_memory_bytes
from evalplus.eval.utils import TimeoutException


_SUCCESS = 0
_FAILED = 1
_UNKNOWN = 3
_STATUS = {_SUCCESS: PASS, _FAILED: FAIL, _UNKNOWN: None}


@contextlib.contextmanager
def _windows_time_limit(seconds: float):
    """Interrupt Python code after ``seconds`` without using SIGALRM."""

    expired = threading.Event()

    def interrupt() -> None:
        expired.set()
        _thread.interrupt_main()

    timer = threading.Timer(max(0.001, float(seconds)), interrupt)
    timer.daemon = True
    timer.start()
    try:
        yield
    except KeyboardInterrupt as exc:
        if expired.is_set():
            raise TimeoutException("Timed out!") from exc
        raise
    finally:
        timer.cancel()


def _windows_unsafe_execute(
    dataset: str,
    entry_point: str,
    code: str,
    inputs,
    expected,
    time_limits,
    atol,
    fast_check: bool,
    stat,
    details,
    progress,
) -> None:
    """Run EvalPlus' official worker after installing Windows-safe guards."""

    import evalplus.eval as eval_module

    unix_guard = eval_module.reliability_guard

    def guard_without_unix_rlimit(maximum_memory_bytes=None) -> None:
        # The parent process enforces the memory ceiling with psutil. Passing
        # None here avoids EvalPlus importing the Unix-only resource module.
        unix_guard(maximum_memory_bytes=None)

    eval_module.reliability_guard = guard_without_unix_rlimit
    eval_module.time_limit = _windows_time_limit
    eval_module.unsafe_execute(
        dataset,
        entry_point,
        code,
        inputs,
        expected,
        time_limits,
        atol,
        fast_check,
        stat,
        details,
        progress,
    )


def _rss_bytes(pid: int) -> int:
    try:
        process = psutil.Process(pid)
        return process.memory_info().rss + sum(
            child.memory_info().rss for child in process.children(recursive=True)
        )
    except (psutil.Error, OSError):
        return 0


def windows_untrusted_check(
    dataset: str,
    code: str,
    inputs: list[Any],
    entry_point: str,
    expected,
    atol,
    ref_time: list[float],
    fast_check: bool = False,
    min_time_limit: float = DEFAULT_MIN_TIME_LIMIT,
    gt_time_limit_factor: float = DEFAULT_GT_TIME_LIMIT_FACTOR,
) -> tuple[str, np.ndarray]:
    """Windows equivalent of ``evalplus.eval.untrusted_check``."""

    time_limits = [max(min_time_limit, gt_time_limit_factor * t) for t in ref_time]
    configured_timeout = float(os.getenv("EVALPLUS_TIMEOUT_PER_TASK", "60"))
    timeout = min(configured_timeout, sum(time_limits)) + 1
    if not fast_check:
        timeout += 1

    progress = Value("i", 0)
    stat = Value("i", _UNKNOWN)
    details = Array("b", [False for _ in range(len(inputs))])

    process = multiprocessing.Process(
        target=_windows_unsafe_execute,
        args=(
            dataset,
            entry_point,
            code,
            inputs,
            expected,
            time_limits,
            atol,
            fast_check,
            stat,
            details,
            progress,
        ),
    )
    process.start()

    deadline = time.monotonic() + timeout + 1
    memory_limit = query_maximum_memory_bytes()
    memory_exceeded = False
    while process.is_alive() and time.monotonic() < deadline:
        process.join(timeout=min(0.05, max(0.0, deadline - time.monotonic())))
        if (
            process.is_alive()
            and memory_limit is not None
            and _rss_bytes(process.pid) > memory_limit
        ):
            memory_exceeded = True
            break

    if process.is_alive():
        process.terminate()
        process.join(timeout=0.2)
    if process.is_alive():
        process.kill()
        process.join(timeout=0.2)

    status = FAIL if memory_exceeded else _STATUS.get(stat.value)
    completed_details = np.asarray(details[: progress.value], dtype=bool)

    if not status:
        status = TIMEOUT
    if status == PASS and (
        len(completed_details) != len(inputs) or not all(completed_details)
    ):
        status = FAIL

    return status, completed_details
