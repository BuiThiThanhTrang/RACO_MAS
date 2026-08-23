from __future__ import annotations

import json
from pathlib import Path


def resolve_data_window(runner, data_start, data_limit):
    resolver = getattr(runner, "resolve_data_window", None)
    if resolver is None:
        return data_start, data_limit
    return resolver(data_start, data_limit)


def register_result_path(runner, result_path):
    register = getattr(runner, "register_result_path", None)
    if register is None:
        return "w"
    return register(result_path)


def complete_item(runner, task_id, next_split_offset, result_path):
    callback = getattr(runner, "complete_item", None)
    if callback is not None:
        callback(task_id, next_split_offset, result_path)


def finalize_run(runner):
    callback = getattr(runner, "finalize_run", None)
    if callback is not None:
        return callback()
    return None


def read_jsonl(path):
    target = Path(path)
    if not target.is_file():
        return []
    return [
        json.loads(line)
        for line in target.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
