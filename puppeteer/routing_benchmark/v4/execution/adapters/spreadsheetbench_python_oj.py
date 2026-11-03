"""Source-faithful SpreadsheetBench Python-OJ adapter.

Only public case 1 is materialized during an episode.  Private evaluator cases
are consumed later by the trusted final evaluator and never enter router or
executor context.
"""
from __future__ import annotations

import ast
from collections import Counter, defaultdict
from datetime import date, datetime, time as datetime_time
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import time
from typing import Any, Mapping

from openpyxl import load_workbook
from openpyxl.utils import column_index_from_string, get_column_letter
from openpyxl.utils.cell import range_boundaries

from ..base import EnvironmentHandle, PreconditionResult, ToolDefinition, ToolResult
from ..spreadsheet_worker import CandidatePolicyError, normalize_and_validate_source
from .common import read_json, write_json
from ...artifacts.manifest import load_manifest
from ...artifacts.schema import sha256_file


ROLE_TOOLS = {
    "Workbook Surveyor": ("inspect_workbook", "read_artifact"),
    "Data Quality Profiler": ("profile_workbook_data", "read_artifact"),
    "Formula Dependency Analyst": ("inspect_formulas", "read_artifact"),
    "Table Reshaping Specialist": ("write_solution", "read_artifact"),
    "Formula Computation Specialist": ("write_solution", "read_artifact"),
    "Cross-sheet Reconciler": ("write_solution", "read_artifact"),
    "Formatting & Presentation Specialist": ("write_solution", "read_artifact"),
    "Workbook Repair Specialist": ("revise_solution", "read_artifact"),
    "Workbook Test Executor": ("run_solution_public",),
    "Workbook Integrity Validator": ("validate_public_output",),
}


def _json_value(value: Any) -> Any:
    if hasattr(value, "text"):
        return value.text
    if isinstance(value, (datetime, date, datetime_time)):
        return value.isoformat()
    if isinstance(value, bytes):
        return {"type": "bytes", "hex": value.hex()}
    return value


def _meaningful_last_row(ws, max_col: int) -> int:
    rows = [row for row in range(1, ws.max_row + 1) if any(ws.cell(row, col).value is not None for col in range(1, max_col + 1))]
    return max(rows) if rows else 0


def workbook_snapshot(
    path: Path,
    *,
    max_rows: int = 200,
    max_cols: int = 50,
    sheet_name: str | None = None,
    cell_range: str | None = None,
) -> dict[str, Any]:
    workbook = load_workbook(path, data_only=False, read_only=False)
    result: dict[str, Any] = {"file": path.name, "sheets": []}
    try:
        worksheets = [workbook[sheet_name]] if sheet_name else workbook.worksheets
        for ws in worksheets:
            if cell_range:
                range_min_col, range_min_row, range_max_col, range_max_row = range_boundaries(cell_range)
            else:
                range_min_col, range_min_row = 1, 1
                range_max_col, range_max_row = ws.max_column, ws.max_row
            last = _meaningful_last_row(ws, ws.max_column)
            shown_max_row = min(range_max_row, range_min_row + max_rows - 1)
            shown_max_col = min(range_max_col, range_min_col + max_cols - 1)
            rows = [
                [_json_value(ws.cell(row, col).value) for col in range(range_min_col, shown_max_col + 1)]
                for row in range(range_min_row, shown_max_row + 1)
            ]
            formulas = [
                {"cell": cell.coordinate, "formula": _json_value(cell.value)}
                for row in ws.iter_rows(min_row=range_min_row, max_row=shown_max_row, min_col=range_min_col, max_col=shown_max_col)
                for cell in row if cell.data_type == "f"
            ]
            result["sheets"].append({
                "name": ws.title,
                "state": ws.sheet_state,
                "calculated_dimension": ws.calculate_dimension(),
                "meaningful_last_row": last,
                "max_column": ws.max_column,
                "requested_range": cell_range,
                "start_row": range_min_row,
                "start_column": range_min_col,
                "rows_shown": max(0, shown_max_row - range_min_row + 1),
                "columns_shown": max(0, shown_max_col - range_min_col + 1),
                "truncated": shown_max_row < range_max_row or shown_max_col < range_max_col,
                "rows": rows,
                "formulas": formulas,
            })
    finally:
        workbook.close()
    return result


def compare_answer_region(output: Path, answer: Path, sheet: str, cell_range: str) -> dict[str, Any]:
    from openpyxl.utils.cell import range_boundaries
    try:
        generated = load_workbook(output, data_only=True, read_only=False)
        expected = load_workbook(answer, data_only=True, read_only=False)
    except Exception as error:
        return {"passed": False, "reason": "workbook_open_failed", "error": f"{type(error).__name__}: {error}"}
    try:
        if sheet not in generated.sheetnames:
            return {"passed": False, "reason": "worksheet_not_found", "sheet": sheet}
        min_col, min_row, max_col, max_row = range_boundaries(cell_range)
        mismatches = []
        for row in range(min_row, max_row + 1):
            for col in range(min_col, max_col + 1):
                actual = generated[sheet].cell(row, col).value
                target = expected[sheet].cell(row, col).value
                if actual == "" and target is None or actual is None and target == "":
                    continue
                if isinstance(actual, (int, float)) and isinstance(target, (int, float)):
                    equal = round(float(actual), 2) == round(float(target), 2)
                elif isinstance(actual, datetime) and isinstance(target, datetime):
                    equal = actual.date() == target.date()
                else:
                    equal = type(actual) is type(target) and actual == target
                if not equal:
                    mismatches.append({
                        "cell": generated[sheet].cell(row, col).coordinate,
                        "actual": _json_value(actual),
                        "expected": _json_value(target),
                    })
                    if len(mismatches) >= 20:
                        break
            if len(mismatches) >= 20:
                break
        return {"passed": not mismatches, "sheet": sheet, "cell_range": cell_range, "mismatches": mismatches}
    finally:
        generated.close()
        expected.close()


def _number(value: Any) -> float | int:
    return value if isinstance(value, (int, float)) else 0


def _values_equal(left: Any, right: Any) -> bool:
    if hasattr(left, "text"):
        left = left.text
    if hasattr(right, "text"):
        right = right.text
    if left == "" and right is None or left is None and right == "":
        return True
    if isinstance(left, (int, float)) and isinstance(right, (int, float)):
        return round(float(left), 2) == round(float(right), 2)
    if isinstance(left, datetime) and isinstance(right, datetime):
        return left.date() == right.date()
    return type(left) is type(right) and left == right


def _formula_equal(left: Any, right: Any) -> bool:
    if hasattr(left, "text"):
        left = left.text
    if hasattr(right, "text"):
        right = right.text
    if not isinstance(left, str) or not isinstance(right, str):
        return False
    return re.sub(r"\s+", "", left).upper() == re.sub(r"\s+", "", right).upper()


def _color_signature(color) -> tuple[Any, ...]:
    return (color.type, color.rgb, color.indexed, color.theme, color.tint)


def _style_signature(cell) -> tuple[Any, ...]:
    return (
        cell.font.name, cell.font.sz, bool(cell.font.bold), bool(cell.font.italic),
        cell.font.underline, _color_signature(cell.font.color) if cell.font.color else None,
        cell.fill.fill_type, _color_signature(cell.fill.fgColor), _color_signature(cell.fill.bgColor),
        cell.number_format,
        cell.alignment.horizontal, cell.alignment.vertical, bool(cell.alignment.wrap_text),
        cell.alignment.text_rotation, bool(cell.alignment.shrink_to_fit), cell.alignment.indent,
        _border_signature(cell.border),
        bool(cell.protection.locked), bool(cell.protection.hidden),
    )


def _border_signature(border) -> tuple[Any, ...]:
    def side(value):
        if value is None or value.style is None:
            return (None, None)
        return (value.style, _color_signature(value.color) if value.color else None)
    return tuple(side(getattr(border, name)) for name in ("left", "right", "top", "bottom", "diagonal"))


def _has_visible_border(cell) -> bool:
    return any(getattr(cell.border, name).style for name in ("left", "right", "top", "bottom"))


def _date_key(value: Any) -> Any:
    if isinstance(value, datetime):
        return value.date().isoformat()
    if isinstance(value, date):
        return value.isoformat()
    if isinstance(value, str):
        for pattern in ("%m/%d/%Y", "%Y-%m-%d"):
            try:
                return datetime.strptime(value.strip(), pattern).date().isoformat()
            except ValueError:
                pass
    return value


def _preservation_issues(source, candidate, allowed: Mapping[str, set[tuple[int, int]]], *, styles: bool) -> list[str]:
    issues: list[str] = []
    for sheet_name in source.sheetnames:
        left, right = source[sheet_name], candidate[sheet_name]
        allowed_cells = allowed.get(sheet_name, set())
        max_row, max_column = max(left.max_row, right.max_row), max(left.max_column, right.max_column)
        for row in range(1, max_row + 1):
            for column in range(1, max_column + 1):
                if (row, column) in allowed_cells:
                    continue
                left_cell, right_cell = left.cell(row, column), right.cell(row, column)
                if not _values_equal(left_cell.value, right_cell.value):
                    return [f"unrelated_value_changed:{sheet_name}!{left_cell.coordinate}"]
                if styles and _style_signature(left_cell) != _style_signature(right_cell):
                    return [f"unrelated_style_changed:{sheet_name}!{left_cell.coordinate}"]
        if styles and left.sheet_view.showGridLines != right.sheet_view.showGridLines:
            return [f"sheet_view_changed:{sheet_name}:showGridLines"]
    return issues


def _fill_rgb(cell) -> str | None:
    color = cell.fill.fgColor
    if color.type == "rgb" and isinstance(color.rgb, str):
        return color.rgb[-6:].upper()
    return None


def _unchanged_sheet_values(source, candidate, excluded_sheet: str) -> list[str]:
    issues: list[str] = []
    if source.sheetnames != candidate.sheetnames:
        return ["sheet_names_or_order_changed"]
    for name in source.sheetnames:
        if name == excluded_sheet:
            continue
        left, right = source[name], candidate[name]
        max_row, max_col = max(left.max_row, right.max_row), max(left.max_column, right.max_column)
        for row in range(1, max_row + 1):
            for column in range(1, max_col + 1):
                if not _values_equal(left.cell(row, column).value, right.cell(row, column).value):
                    issues.append(f"unrelated_sheet_changed:{name}!{left.cell(row, column).coordinate}")
                    return issues
    return issues


class SpreadsheetBenchPythonOJAdapter:
    adapter_id = "spreadsheetbench.python_oj.v1"

    def __init__(self) -> None:
        self.runtime: dict[str, Any] = {}

    def configure_runtime(self, profile: Mapping[str, Any]) -> None:
        self.runtime = dict(profile)

    def materialize(self, manifest_path, workspace) -> EnvironmentHandle:
        path, workspace = Path(manifest_path), Path(workspace)
        pack = path if path.is_dir() else path.parent
        manifest = load_manifest(pack)
        if manifest.adapter_id != self.adapter_id:
            raise ValueError("Adapter mismatch")
        workspace.mkdir(parents=True, exist_ok=False)
        for item in manifest.public_files:
            source = (pack / item.path).resolve()
            if not source.is_relative_to(pack.resolve()) or not item.path.startswith("public/"):
                raise ValueError("Invalid public artifact path")
            if sha256_file(source) != item.sha256:
                raise ValueError(f"Public source checksum mismatch: {item.path}")
            destination = workspace / item.path
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, destination)
        state = {
            "artifacts": {}, "artifact_history": [], "rejected_attempts": [],
            "events": [], "terminal": False, "latest_execution": None,
        }
        write_json(workspace / "_runtime/state.json", state)
        return EnvironmentHandle(manifest, pack, workspace)

    def _state(self, env) -> dict[str, Any]:
        return dict(read_json(env.workspace_dir / "_runtime/state.json"))

    def _save(self, env, state) -> None:
        write_json(env.workspace_dir / "_runtime/state.json", state)

    def _input(self, env) -> Path:
        return env.workspace_dir / "public/cases/case_1/input.xlsx"

    def versions(self, env) -> dict[str, Any]:
        state = self._state(env)
        candidate = state["artifacts"].get("candidate_solution")
        return {
            "public_input_hash": sha256_file(self._input(env)),
            "candidate_hash": candidate.get("sha256") if candidate else None,
            "runtime_profile": json.dumps(self.runtime, sort_keys=True, default=str),
        }

    @staticmethod
    def _fresh(artifact, state, versions) -> bool:
        return all(versions.get(key) == value for key, value in artifact.get("input_versions", {}).items()) and all(
            state["artifacts"].get(ref["key"], {}).get("sha256") == ref["sha256"]
            for ref in artifact.get("input_refs", [])
        )

    def _artifact_view(self, env, state, artifact) -> dict[str, Any]:
        view = dict(artifact)
        view["fresh"] = self._fresh(artifact, state, self.versions(env))
        path = env.workspace_dir / artifact["path"]
        if path.suffix in {".py", ".json", ".txt"} and path.is_file():
            text = path.read_text(encoding="utf-8", errors="replace")
            view["content_preview"] = text[:8000]
            view["content_length"] = len(text)
        return view

    def public_observation(self, env) -> dict[str, Any]:
        state, versions = self._state(env), self.versions(env)
        artifacts = {key: self._artifact_view(env, state, value) for key, value in state["artifacts"].items()}
        validation = artifacts.get("public_validation")
        terminal = bool(validation and validation["fresh"] and validation.get("passed"))
        return {
            "public_task": read_json(env.workspace_dir / "public/task.json"),
            "candidate_contract": read_json(env.workspace_dir / "public/execution_contract.json")["candidate"],
            "public_case_summary": {
                "input_hash": versions["public_input_hash"],
                "sheets": [item["name"] for item in workbook_snapshot(self._input(env), max_rows=0)["sheets"]],
            },
            "versions": versions,
            "artifacts": artifacts,
            "latest_execution": state.get("latest_execution"),
            "rejected_attempt_count": len(state.get("rejected_attempts", [])),
            "terminal": terminal,
        }

    def executor_context(self, env, action, role_profile=None):
        state = self._state(env)
        return {
            "task": read_json(env.workspace_dir / "public/task.json"),
            "execution_contract": read_json(env.workspace_dir / "public/execution_contract.json"),
            "validation_contract": read_json(env.workspace_dir / "public/validation_contract.json"),
            "public_case_1": workbook_snapshot(self._input(env)),
            "runtime_state": self.public_observation(env),
            "recent_tool_events": state["events"][-4:],
            "candidate_authoring_constraints": {
                "entrypoint": "solve(input_path, output_path)",
                "allowed_library": "OpenPyXL load_workbook and style helpers Alignment, Border, Font, PatternFill and Side are prebound. Documented imports from openpyxl/openpyxl.styles plus collections.defaultdict and copy.copy are accepted and normalized.",
                "forbidden": "No other imports, dunder/private attributes, arbitrary file access, network, top-level execution or class definitions.",
                "output": "The function must save a readable XLSX workbook exactly to output_path.",
            },
        }

    def public_preconditions(self, env, action_id, role_profile=None):
        state = self._state(env)
        if action_id not in ROLE_TOOLS:
            return PreconditionResult(False, ("Unknown role",))
        if action_id == "Workbook Test Executor" and "candidate_solution" not in state["artifacts"]:
            return PreconditionResult(False, ("No accepted candidate solution exists",))
        if action_id == "Workbook Integrity Validator":
            execution = state.get("latest_execution")
            if not execution or execution.get("candidate_hash") != self.versions(env)["candidate_hash"] or execution.get("status") != "success":
                return PreconditionResult(False, ("No successful fresh public execution exists",))
        if action_id == "Workbook Repair Specialist" and not state.get("latest_execution"):
            return PreconditionResult(False, ("No concrete public execution evidence exists",))
        return PreconditionResult(True)

    def tool_spec(self, env, action_id, role_profile=None):
        tools = ROLE_TOOLS.get(action_id, ())
        definitions = {
            "inspect_workbook": ToolDefinition("inspect_workbook", "Inspect the complete bounded public workbook structure and values without modifying it.", {
                "type": "object", "properties": {"sheet": {"type": ["string", "null"]}, "cell_range": {"type": ["string", "null"]}, "include_formulas": {"type": "boolean"}, "include_styles": {"type": "boolean"}}, "additionalProperties": False}),
            "profile_workbook_data": ToolDefinition("profile_workbook_data", "Profile public case-1 rows, nulls and duplicate keys without modifying the workbook.", {
                "type": "object", "properties": {"sheet": {"type": "string"}, "cell_range": {"type": ["string", "null"]}, "key_columns": {"type": "array", "items": {"type": "string"}}}, "required": ["sheet"], "additionalProperties": False}),
            "inspect_formulas": ToolDefinition("inspect_formulas", "Inspect formulas in the public workbook without modifying it.", {
                "type": "object", "properties": {"sheet": {"type": ["string", "null"]}, "cell_range": {"type": ["string", "null"]}}, "additionalProperties": False}),
            "read_artifact": ToolDefinition("read_artifact", "Read an allowlisted immutable runtime artifact by ID with bounded offsets.", {
                "type": "object", "properties": {"artifact_id": {"type": "string"}, "version": {"type": ["integer", "null"]}, "offset": {"type": "integer", "minimum": 0}, "limit": {"type": "integer", "minimum": 1, "maximum": 8000}}, "required": ["artifact_id"], "additionalProperties": False}),
            "write_solution": ToolDefinition("write_solution", "Create a reusable Python solve(input_path, output_path) candidate. Do not use private cases or hard-coded case-1 values.", {
                "type": "object", "properties": {"python_source": {"type": "string", "minLength": 1}, "rationale": {"type": "string"}, "input_artifact_refs": {"type": "array", "items": {"type": "string"}}}, "required": ["python_source", "input_artifact_refs"], "additionalProperties": False}),
            "revise_solution": ToolDefinition("revise_solution", "Revise the current candidate using concrete public failure evidence.", {
                "type": "object", "properties": {"base_candidate_artifact_id": {"type": "string"}, "python_source": {"type": "string", "minLength": 1}, "evidence_refs": {"type": "array", "items": {"type": "string"}}}, "required": ["base_candidate_artifact_id", "python_source", "evidence_refs"], "additionalProperties": False}),
            "run_solution_public": ToolDefinition("run_solution_public", "Execute the current candidate on public case 1 in the restricted runner.", {
                "type": "object", "properties": {"candidate_artifact_id": {"type": "string"}}, "required": ["candidate_artifact_id"], "additionalProperties": False}),
            "validate_public_output": ToolDefinition("validate_public_output", "Validate fresh public output against the public deterministic contract; never read private cases.", {
                "type": "object", "properties": {"candidate_artifact_id": {"type": "string"}, "execution_artifact_id": {"type": "string"}}, "required": ["candidate_artifact_id", "execution_artifact_id"], "additionalProperties": False}),
        }
        return tuple(definitions[name] for name in tools)

    def _store_bytes(self, env, state, key, raw: bytes, suffix: str, inputs, tool, refs=(), extra=None):
        version = len(state["artifact_history"])
        relative = f"_runtime/artifacts/{version:04}_{key}{suffix}"
        path = env.workspace_dir / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(raw)
        artifact = {
            "artifact_id": f"{key}:{version}", "version": version, "logical_key": key,
            "sha256": sha256_file(path), "path": relative, "input_versions": dict(inputs),
            "input_refs": [
                {"key": ref, "artifact_id": state["artifacts"][ref]["artifact_id"], "sha256": state["artifacts"][ref]["sha256"]}
                for ref in refs if ref in state["artifacts"]
            ],
            "producer_tool": tool, "visibility": "public",
        }
        if extra:
            artifact.update(extra)
        state["artifacts"][key] = artifact
        state["artifact_history"].append(artifact)
        return artifact

    def _store_json(self, env, state, key, value, inputs, tool, refs=(), extra=None):
        raw = (json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2, default=_json_value) + "\n").encode("utf-8")
        return self._store_bytes(env, state, key, raw, ".json", inputs, tool, refs, extra)

    @staticmethod
    def _strip_code_fence(source: str) -> str:
        block = re.search(r"```(?:python)?\s*\n(.*?)```", source, re.S | re.I)
        return (block.group(1) if block else source).strip() + "\n"

    def _candidate_source(self, env, state) -> tuple[dict[str, Any], Path]:
        artifact = state["artifacts"]["candidate_solution"]
        return artifact, env.workspace_dir / artifact["path"]

    def _execute_candidate(self, env, candidate_path: Path, input_path: Path, output_path: Path, timeout: int):
        worker = Path(__file__).resolve().parents[1] / "spreadsheet_worker.py"
        python = self.runtime.get("python_executable") or sys.executable
        input_before = sha256_file(input_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        argv = [str(python), str(worker), "--candidate", str(candidate_path.resolve()), "--input", str(input_path.resolve()), "--output", str(output_path.resolve())]
        started = time.perf_counter()
        try:
            completed = subprocess.run(
                argv, cwd=output_path.parent, capture_output=True, text=True,
                encoding="utf-8", errors="replace", timeout=timeout,
                creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
            )
            status = "success" if completed.returncode == 0 else "failed"
            result = {
                "status": status, "returncode": completed.returncode,
                "stdout": completed.stdout[-16000:], "stderr": completed.stderr[-16000:],
                "latency_ms": int((time.perf_counter() - started) * 1000),
                "input_unchanged": input_before == sha256_file(input_path),
                "failure_origin": None if status == "success" else "candidate_execution",
            }
        except subprocess.TimeoutExpired as error:
            result = {
                "status": "failed", "returncode": -1, "stdout": (error.stdout or "")[-16000:] if isinstance(error.stdout, str) else "",
                "stderr": "Candidate execution timeout", "latency_ms": int((time.perf_counter() - started) * 1000),
                "input_unchanged": input_before == sha256_file(input_path), "failure_origin": "candidate_execution", "timeout": True,
            }
        maximum = int(read_json(env.workspace_dir / "public/execution_contract.json")["maximum_output_bytes"])
        if output_path.exists() and output_path.stat().st_size > maximum:
            result.update(status="failed", failure_origin="candidate_execution", output_too_large=True)
        if not result["input_unchanged"]:
            result.update(status="tool_failure", failure_origin="tool_environment", environment_error="Public input changed")
        if result["status"] == "success":
            try:
                workbook = load_workbook(output_path, read_only=True, data_only=False)
                result["output_sheets"] = list(workbook.sheetnames)
                workbook.close()
                result["output_sha256"] = sha256_file(output_path)
                result["output_bytes"] = output_path.stat().st_size
            except Exception as error:
                result.update(status="failed", failure_origin="candidate_execution", workbook_error=f"{type(error).__name__}: {error}")
        return result

    def _profile(self, path: Path, arguments: Mapping[str, Any]):
        workbook = load_workbook(path, data_only=False)
        try:
            sheet = str(arguments["sheet"])
            ws = workbook[sheet]
            requested_range = arguments.get("cell_range")
            if requested_range:
                min_col, header_row, max_col, max_row = range_boundaries(str(requested_range))
            else:
                min_col, header_row, max_col, max_row = 1, 1, ws.max_column, ws.max_row
            last = min(max_row, _meaningful_last_row(ws, max_col))
            headers = {
                str(ws.cell(header_row, col).value): col
                for col in range(min_col, max_col + 1)
                if ws.cell(header_row, col).value is not None
            }
            columns = {}
            for name, col in headers.items():
                values = [ws.cell(row, col).value for row in range(header_row + 1, last + 1)]
                columns[name] = {"non_null": sum(value is not None for value in values), "null": sum(value is None for value in values), "unique": len(set(map(str, values)))}
            keys = list(arguments.get("key_columns") or [])
            duplicates = []
            if keys:
                indices = [column_index_from_string(key) if key.isalpha() else headers[key] for key in keys]
                grouped: dict[tuple[Any, ...], list[int]] = defaultdict(list)
                for row in range(header_row + 1, last + 1):
                    grouped[tuple(ws.cell(row, col).value for col in indices)].append(row)
                duplicates = [{"key": [_json_value(value) for value in key], "rows": rows} for key, rows in grouped.items() if len(rows) > 1]
            return {"sheet": sheet, "cell_range": requested_range, "header_row": header_row, "last_row": last, "columns": columns, "duplicate_groups": duplicates}
        finally:
            workbook.close()

    def validate_output(self, input_path: Path, output: Path, contract: Mapping[str, Any]):
        source = load_workbook(input_path, data_only=False)
        candidate = load_workbook(output, data_only=False)
        kind = contract["kind"]
        issues: list[str] = []
        details: dict[str, Any] = {"kind": kind}
        try:
            if source.sheetnames != candidate.sheetnames:
                return {"passed": False, "issues": ["sheet_names_or_order_changed"], **details}

            if kind == "deduplicate_sum":
                sheet = contract["sheet"]
                source_ws, output_ws = source[sheet], candidate[sheet]
                key_cols = [column_index_from_string(item) for item in contract["key_columns"]]
                aggregate_col = column_index_from_string(next(iter(contract["aggregate_columns"])))
                last = _meaningful_last_row(source_ws, source_ws.max_column)
                expected, positions = [], {}
                for row in range(2, last + 1):
                    values = [source_ws.cell(row, column).value for column in range(1, source_ws.max_column + 1)]
                    key = tuple(values[column - 1] for column in key_cols)
                    if key not in positions:
                        positions[key] = len(expected)
                        expected.append(values)
                    else:
                        index = positions[key]
                        expected[index][aggregate_col - 1] = _number(expected[index][aggregate_col - 1]) + _number(values[aggregate_col - 1])
                actual_last = _meaningful_last_row(output_ws, output_ws.max_column)
                actual = [[output_ws.cell(row, column).value for column in range(1, source_ws.max_column + 1)] for row in range(2, actual_last + 1)]
                if actual != expected:
                    issues.append("transformation_contract_mismatch")
                details.update(expected_data_rows=len(expected), actual_data_rows=len(actual))
                issues.extend(_unchanged_sheet_values(source, candidate, sheet))

            elif kind == "collection_aggregate":
                sheet = contract["output_sheet"]
                source_output, actual_output = source[sheet], candidate[sheet]
                order: list[tuple[Any, Any]] = []
                records: dict[tuple[Any, Any], dict[str, Any]] = {}
                for row in range(2, source_output.max_row + 1):
                    key = (source_output.cell(row, 3).value, source_output.cell(row, 4).value)
                    if all(value is not None for value in key) and key not in records:
                        order.append(key)
                        records[key] = {"br": source_output.cell(row, 2).value, "sale": 0, "ret": 0}
                for workbook_sheet in source.worksheets:
                    if workbook_sheet.title == sheet:
                        continue
                    headers = {str(workbook_sheet.cell(1, column).value).strip().upper(): column for column in range(1, workbook_sheet.max_column + 1) if workbook_sheet.cell(1, column).value is not None}
                    for row in range(2, workbook_sheet.max_row + 1):
                        ty, order_code = workbook_sheet.cell(row, headers["TY"]).value, workbook_sheet.cell(row, headers["OR"]).value
                        if ty is None or order_code is None:
                            continue
                        key = (ty, order_code)
                        if key not in records:
                            order.append(key)
                            records[key] = {"br": workbook_sheet.cell(row, headers["BR"]).value, "sale": 0, "ret": 0}
                        if "SALE" in headers:
                            records[key]["sale"] += _number(workbook_sheet.cell(row, headers["SALE"]).value)
                        if "RET" in headers:
                            records[key]["ret"] += _number(workbook_sheet.cell(row, headers["RET"]).value)
                expected = [[index, records[key]["br"], key[0], key[1], records[key]["sale"], records[key]["ret"], records[key]["sale"] - records[key]["ret"]] for index, key in enumerate(order, 1)]
                actual_last = _meaningful_last_row(actual_output, 7)
                actual = [[actual_output.cell(row, column).value for column in range(1, 8)] for row in range(2, actual_last + 1)]
                if actual != expected:
                    issues.append("collection_values_mismatch")
                required_format = contract["zero_display_number_format"]
                for row in range(1, len(expected) + 2):
                    for column in range(1, 8):
                        cell = actual_output.cell(row, column)
                        if cell.font.name != contract["font"]["name"] or cell.font.sz != contract["font"]["size"] or bool(cell.font.bold) != contract["font"]["bold"]:
                            issues.append(f"font_contract_mismatch:{cell.coordinate}")
                            break
                        expected_alignment = "right" if column in (1, 5, 6, 7) and row > 1 else "left"
                        if cell.alignment.horizontal != expected_alignment:
                            issues.append(f"alignment_contract_mismatch:{cell.coordinate}")
                            break
                        if row > 1 and column in (5, 6, 7) and cell.number_format != required_format:
                            issues.append(f"number_format_contract_mismatch:{cell.coordinate}")
                            break
                    if issues and issues[-1].startswith(("font_", "alignment_", "number_format_")):
                        break
                details.update(expected_data_rows=len(expected), actual_data_rows=len(actual))
                issues.extend(_unchanged_sheet_values(source, candidate, sheet))

            elif kind == "account_bucket_aggregate":
                sheet = contract["output_sheet"]
                output_ws = candidate[sheet]
                groups = [(item["prefix"], column_index_from_string(item["output_columns"][0])) for item in contract["buckets"]]
                grouped: dict[str, dict[str, float | int]] = {name: {} for name, _ in groups}
                for workbook_sheet in source.worksheets:
                    if workbook_sheet.title == sheet:
                        continue
                    for row in range(2, workbook_sheet.max_row + 1):
                        reference = workbook_sheet.cell(row, 2).value
                        if not isinstance(reference, str):
                            continue
                        for name, _ in groups:
                            if reference.startswith(name):
                                grouped[name][reference] = grouped[name].get(reference, 0) + _number(workbook_sheet.cell(row, 3).value) + _number(workbook_sheet.cell(row, 4).value)
                                break

                def sort_key(reference: str):
                    match = re.search(r"(\d+)$", reference)
                    return (int(match.group(1)) if match else 10**30, reference)

                for name, start_column in groups:
                    items = sorted(grouped[name].items(), key=lambda item: sort_key(item[0]))
                    expected = [[index, reference, amount] for index, (reference, amount) in enumerate(items, 1)] + [["TOTAL", None, sum(amount for _, amount in items)]]
                    last_value_row = max((row for row in range(2, output_ws.max_row + 1) if any(output_ws.cell(row, column).value is not None for column in range(start_column, start_column + 3))), default=1)
                    actual = [[output_ws.cell(row, column).value for column in range(start_column, start_column + 3)] for row in range(2, last_value_row + 1)]
                    if actual != expected:
                        issues.append(f"bucket_mismatch:{name}")
                    details[f"{name}_rows"] = len(items)
                issues.extend(_unchanged_sheet_values(source, candidate, sheet))

            elif kind == "sheet_column_summary":
                sheet = contract["output_sheet"]
                output_ws = candidate[sheet]
                input_sheets = [workbook_sheet for workbook_sheet in source.worksheets if workbook_sheet.title != sheet]
                expected = []
                for index, workbook_sheet in enumerate(input_sheets, 1):
                    imported = sum(_number(workbook_sheet.cell(row, 3).value) for row in range(2, workbook_sheet.max_row + 1))
                    exported = sum(_number(workbook_sheet.cell(row, 4).value) for row in range(2, workbook_sheet.max_row + 1))
                    expected.append([index, workbook_sheet.title, imported, exported, imported - exported])
                expected.append([None, "TOTAL", sum(row[2] for row in expected), sum(row[3] for row in expected), sum(row[4] for row in expected)])
                actual = [[output_ws.cell(row, column).value for column in range(1, 6)] for row in range(2, 2 + len(expected))]
                mismatched = False
                for row in range(len(expected)):
                    worksheet_row = row + 2
                    for column in range(5):
                        value = actual[row][column]
                        accepted_formula = None
                        if column == 4 and row < len(expected) - 1:
                            accepted_formula = f"=C{worksheet_row}-D{worksheet_row}"
                        elif column >= 2 and row == len(expected) - 1:
                            letter = chr(ord("A") + column)
                            accepted_formula = f"=SUM({letter}2:{letter}{worksheet_row - 1})"
                        if not _values_equal(value, expected[row][column]) and not (accepted_formula and isinstance(value, str) and value.upper() == accepted_formula.upper()):
                            mismatched = True
                            break
                    if mismatched:
                        break
                if mismatched:
                    issues.append("sheet_summary_mismatch")
                details["summarized_sheet_count"] = len(input_sheets)
                issues.extend(_unchanged_sheet_values(source, candidate, sheet))

            elif kind == "conditional_replace_and_fill":
                sheet = contract["sheet"]
                source_ws, output_ws = source[sheet], candidate[sheet]
                first, second = contract["condition"]["BG_in"]
                start_row, end_row = contract["row_range"]
                qualified = []
                for row in range(start_row, end_row + 1):
                    condition = source_ws.cell(row, 64).value == contract["condition"]["BL"] and source_ws.cell(row, 59).value in (first, second)
                    expected_bn = source_ws.cell(row, 66).value
                    if condition:
                        qualified.append(row)
                        expected_bn = contract["branch"]["then_BN"] if source_ws.cell(row, 51).value in contract["branch"]["AY_in"] else contract["branch"]["else_BN"]
                        if _fill_rgb(output_ws.cell(row, 64)) != contract["qualified_BL_fill_rgb"]:
                            issues.append(f"fill_contract_mismatch:BL{row}")
                    if not _values_equal(output_ws.cell(row, 66).value, expected_bn):
                        issues.append(f"conditional_value_mismatch:BN{row}")
                for row in range(1, max(source_ws.max_row, output_ws.max_row) + 1):
                    for column in range(1, max(source_ws.max_column, output_ws.max_column) + 1):
                        if column == 66 and start_row <= row <= end_row:
                            continue
                        if not _values_equal(source_ws.cell(row, column).value, output_ws.cell(row, column).value):
                            issues.append(f"unrelated_value_changed:{output_ws.cell(row, column).coordinate}")
                            break
                    if issues and issues[-1].startswith("unrelated_value_changed"):
                        break
                details["qualified_rows"] = qualified

            elif kind == "row_text_dedup_output":
                sheet = contract["sheet"]
                source_ws, output_ws = source[sheet], candidate[sheet]
                source_min_col, source_min_row, source_max_col, source_max_row = range_boundaries(contract["source_range"])
                output_min_col, output_min_row, output_max_col, output_max_row = range_boundaries(contract["output_range"])
                expected = []
                for row in range(source_min_row, source_max_row + 1):
                    values = [source_ws.cell(row, column).value for column in range(source_min_col, source_max_col + 1)]
                    if row > source_min_row:
                        seen = set()
                        for index in range(1, len(values)):
                            value = values[index]
                            if value is not None and value in seen:
                                values[index] = None
                            elif value is not None:
                                seen.add(value)
                    expected.append(values)
                actual = [
                    [output_ws.cell(row, column).value for column in range(output_min_col, output_max_col + 1)]
                    for row in range(output_min_row, output_max_row + 1)
                ]
                if actual != expected:
                    issues.append("row_dedup_output_mismatch")
                allowed = {(row, column) for row in range(output_min_row, output_max_row + 1) for column in range(output_min_col, output_max_col + 1)}
                issues.extend(_preservation_issues(source, candidate, {sheet: allowed}, styles=False))
                details["output_rows"] = len(expected)

            elif kind == "delete_reversing_row_pairs":
                sheet = contract["sheet"]
                source_ws, output_ws = source[sheet], candidate[sheet]
                reference_col = column_index_from_string(contract["reference_column"])
                debit_col = column_index_from_string(contract["debit_column"])
                credit_col = column_index_from_string(contract["credit_column"])
                net_col = column_index_from_string(contract["net_formula_column"])
                narrative_col = column_index_from_string(contract["narrative_column"])
                last = _meaningful_last_row(source_ws, narrative_col)
                removed: set[int] = set()
                for left in range(2, last + 1):
                    if left in removed:
                        continue
                    for right in range(left + 1, last + 1):
                        if right in removed:
                            continue
                        if (
                            source_ws.cell(left, reference_col).value == source_ws.cell(right, reference_col).value
                            and source_ws.cell(left, narrative_col).value == source_ws.cell(right, narrative_col).value
                            and source_ws.cell(left, debit_col).value == source_ws.cell(right, credit_col).value
                            and source_ws.cell(left, credit_col).value == source_ws.cell(right, debit_col).value
                        ):
                            removed.update((left, right))
                            break
                retained = [row for row in range(1, last + 1) if row not in removed]
                expected = []
                for output_row, source_row in enumerate(retained, 1):
                    values = [source_ws.cell(source_row, column).value for column in range(1, narrative_col + 1)]
                    if output_row > 1:
                        values[net_col - 1] = f"=+C{output_row}-D{output_row}"
                    expected.append(values)
                actual_last = _meaningful_last_row(output_ws, narrative_col)
                actual = [[output_ws.cell(row, column).value for column in range(1, narrative_col + 1)] for row in range(1, actual_last + 1)]
                if len(actual) != len(expected) or any(
                    not _values_equal(actual[row][column], expected[row][column])
                    for row in range(min(len(actual), len(expected)))
                    for column in range(narrative_col)
                ):
                    issues.append("reversing_pair_deletion_mismatch")
                details.update(removed_source_rows=sorted(removed), remaining_rows=max(0, len(expected) - 1))

            elif kind == "strip_non_numeric_text":
                sheet = contract["sheet"]
                source_ws, output_ws = source[sheet], candidate[sheet]
                min_col, min_row, max_col, max_row = range_boundaries(contract["target_range"])
                allowed = set()
                for row in range(min_row, max_row + 1):
                    for column in range(min_col, max_col + 1):
                        allowed.add((row, column))
                        original = source_ws.cell(row, column).value
                        expected = original
                        if isinstance(original, str):
                            cleaned = re.sub(r"[^0-9.,]", "", original).replace(",", "")
                            expected = float(cleaned) if "." in cleaned else int(cleaned) if cleaned else original
                        if not _values_equal(output_ws.cell(row, column).value, expected):
                            issues.append(f"numeric_cleaning_mismatch:{output_ws.cell(row, column).coordinate}")
                        if _style_signature(output_ws.cell(row, column)) != _style_signature(source_ws.cell(row, column)):
                            issues.append(f"target_style_changed:{output_ws.cell(row, column).coordinate}")
                issues.extend(_preservation_issues(source, candidate, {sheet: allowed}, styles=True))

            elif kind == "sort_digits_and_format":
                sheet = contract["sheet"]
                source_ws, output_ws = source[sheet], candidate[sheet]
                source_col = column_index_from_string(contract["source_column"])
                output_col = column_index_from_string(contract["output_column"])
                start_row, end_row = contract["row_range"]
                allowed = {(row, output_col) for row in range(start_row, end_row + 1)}
                for row in range(start_row, end_row + 1):
                    digits = f"{int(source_ws.cell(row, source_col).value):0{contract['digit_width']}d}"
                    expected = "".join(sorted(digits))
                    cell = output_ws.cell(row, output_col)
                    if cell.value != expected:
                        issues.append(f"sorted_digits_mismatch:{cell.coordinate}")
                    if bool(cell.font.bold) is not contract["font_bold"]:
                        issues.append(f"font_contract_mismatch:{cell.coordinate}")
                    if _fill_rgb(cell) != contract["fill_rgb"]:
                        issues.append(f"fill_contract_mismatch:{cell.coordinate}")
                    if cell.number_format != contract["number_format"]:
                        issues.append(f"number_format_contract_mismatch:{cell.coordinate}")
                issues.extend(_preservation_issues(source, candidate, {sheet: allowed}, styles=True))

            elif kind == "workday_formula_series":
                sheet = contract["sheet"]
                source_ws, output_ws = source[sheet], candidate[sheet]
                output_col = column_index_from_string(contract["output_column"])
                start_row, end_row = contract["row_range"]
                allowed = {(row, output_col) for row in range(start_row, end_row + 1)}
                for row in range(start_row, end_row + 1):
                    workday = f"WORKDAY($B$3-1,ROWS($B$7:$B{row}),$J$5:$J$7)"
                    expected = f'=IF({workday}<=$F$3,{workday},"")'
                    cell = output_ws.cell(row, output_col)
                    if not _formula_equal(cell.value, expected):
                        issues.append(f"workday_formula_mismatch:{cell.coordinate}")
                    if _style_signature(cell) != _style_signature(source_ws.cell(row, output_col)):
                        issues.append(f"target_style_changed:{cell.coordinate}")
                issues.extend(_preservation_issues(source, candidate, {sheet: allowed}, styles=True))

            elif kind == "weighted_average_formulas":
                sheet = contract["sheet"]
                source_ws, output_ws = source[sheet], candidate[sheet]
                start_row, end_row = contract["row_range"]
                output_columns = [column_index_from_string(item) for item in contract["output_columns"]]
                value_columns = [column_index_from_string(item) for item in contract["value_columns"]]
                group_column = column_index_from_string(contract["group_column"])
                weight_column = column_index_from_string(contract["weight_column"])
                output_group_column = column_index_from_string(contract["output_group_column"])
                source_last = max(
                    row for row in range(2, source_ws.max_row + 1)
                    if any(source_ws.cell(row, column).value is not None for column in (group_column, weight_column, *value_columns))
                )
                allowed = {(row, column) for row in range(start_row, end_row + 1) for column in output_columns}
                for row in range(start_row, end_row + 1):
                    for output_column, value_column in zip(output_columns, value_columns):
                        output_letter = get_column_letter(output_column)
                        value_letter = get_column_letter(value_column)
                        group_letter = get_column_letter(group_column)
                        weight_letter = get_column_letter(weight_column)
                        key_letter = get_column_letter(output_group_column)
                        expected = (
                            f'=SUMPRODUCT(($${group_letter}$2:$${group_letter}${source_last}=$${key_letter}{row})*'
                            f'$${weight_letter}$2:$${weight_letter}${source_last}*$${value_letter}$2:$${value_letter}${source_last})/'
                            f'SUMIF($${group_letter}$2:$${group_letter}${source_last},$${key_letter}{row},'
                            f'$${weight_letter}$2:$${weight_letter}${source_last})'
                        ).replace("$$", "$")
                        cell = output_ws.cell(row, output_column)
                        if not _formula_equal(cell.value, expected):
                            issues.append(f"weighted_average_formula_mismatch:{output_letter}{row}")
                issues.extend(_preservation_issues(source, candidate, {sheet: allowed}, styles=True))

            elif kind == "progressive_percentile_median_formulas":
                sheet = contract["sheet"]
                source_ws, output_ws = source[sheet], candidate[sheet]
                start_row, end_row = contract["row_range"]
                value_column = column_index_from_string(contract["value_column"])
                percentile_column = column_index_from_string(contract["percentile_column"])
                output_column = column_index_from_string(contract["output_column"])
                last_row = max(
                    row for row in range(start_row, source_ws.max_row + 1)
                    if source_ws.cell(row, value_column).value is not None or source_ws.cell(row, percentile_column).value is not None
                )
                value_letter = get_column_letter(value_column)
                percentile_letter = get_column_letter(percentile_column)
                allowed = {(row, output_column) for row in range(start_row, end_row + 1)}
                for row in range(start_row, end_row + 1):
                    expected = (
                        f'=MEDIAN(_xlfn._xlws.FILTER({value_letter}{row}:${value_letter}${last_row},'
                        f'({percentile_letter}{row}:${percentile_letter}${last_row}>0%)*'
                        f'({percentile_letter}{row}:${percentile_letter}${last_row}<1%)))'
                    )
                    cell = output_ws.cell(row, output_column)
                    if not _formula_equal(cell.value, expected):
                        issues.append(f"percentile_median_formula_mismatch:{cell.coordinate}")
                    if not _has_visible_border(cell):
                        issues.append(f"border_contract_mismatch:{cell.coordinate}")
                    if cell.alignment.horizontal != contract["style"]["horizontal_alignment"]:
                        issues.append(f"alignment_contract_mismatch:{cell.coordinate}")
                    if cell.number_format != contract["style"]["number_format"]:
                        issues.append(f"number_format_contract_mismatch:{cell.coordinate}")
                issues.extend(_preservation_issues(source, candidate, {sheet: allowed}, styles=True))

            elif kind == "custom_day_sales_formulas":
                sheet = contract["sheet"]
                source_ws, output_ws = source[sheet], candidate[sheet]
                min_col, min_row, max_col, max_row = range_boundaries(contract["output_range"])
                source_last = max(
                    row for row in range(4, source_ws.max_row + 1)
                    if any(source_ws.cell(row, column).value is not None for column in range(1, 5))
                )
                allowed = {(row, column) for row in range(min_row, max_row + 1) for column in range(min_col, max_col + 1)}
                for row in range(min_row, max_row + 1):
                    for column in range(min_col, max_col + 1):
                        letter = get_column_letter(column)
                        expected = (
                            f"=SUMPRODUCT((($C$4:$C${source_last}={letter}$11)*('Sample Data'!$A$4:$A${source_last}+"
                            f"'Sample Data'!$B$4:$B${source_last})>='Sample Data'!$F{row}+$F$1)*'Sample Data'!$D$4:$D${source_last})-"
                            f"SUMPRODUCT((($C$4:$C${source_last}={letter}$11)*('Sample Data'!$A$4:$A${source_last}+"
                            f"'Sample Data'!$B$4:$B${source_last})>'Sample Data'!$F{row + 1}+$F$2)*'Sample Data'!$D$4:$D${source_last})"
                        )
                        cell = output_ws.cell(row, column)
                        if not _formula_equal(cell.value, expected):
                            issues.append(f"custom_day_formula_mismatch:{cell.coordinate}")
                        if "$" not in cell.number_format or "[$" in cell.number_format:
                            issues.append(f"currency_format_mismatch:{cell.coordinate}")
                issues.extend(_preservation_issues(source, candidate, {sheet: allowed}, styles=True))

            elif kind == "bond_cashflow_formulas":
                sheet = contract["sheet"]
                source_ws, output_ws = source[sheet], candidate[sheet]
                min_col, min_row, max_col, max_row = range_boundaries(contract["output_range"])
                allowed = {(row, column) for row in range(min_row, max_row + 1) for column in range(min_col, max_col + 1)}
                for row in range(min_row, max_row + 1):
                    for column in range(min_col, max_col + 1):
                        letter = get_column_letter(column)
                        expected = (
                            f'=(MOD(YEARFRAC({letter}$4,EOMONTH($D{row},0))*12,12/$C{row})=0)*'
                            f'(EDATE($D{row},1)>{letter}$4)*$B{row}+'
                            f'(EOMONTH($D{row},0)={letter}$4)*$A{row}'
                        )
                        cell = output_ws.cell(row, column)
                        if not _formula_equal(cell.value, expected):
                            issues.append(f"bond_cashflow_formula_mismatch:{cell.coordinate}")
                        if cell.alignment.horizontal != contract["horizontal_alignment"]:
                            issues.append(f"alignment_contract_mismatch:{cell.coordinate}")
                        if cell.number_format != source_ws.cell(row, column).number_format:
                            issues.append(f"number_format_contract_mismatch:{cell.coordinate}")
                issues.extend(_preservation_issues(source, candidate, {sheet: allowed}, styles=True))

            elif kind == "section_aggregate_lists":
                source_sheet, output_sheet = contract["source_sheet"], contract["output_sheet"]
                source_ws, output_ws = source[source_sheet], candidate[output_sheet]
                sections: dict[str, list[list[Any]]] = {}
                for header_row in range(1, source_ws.max_row + 1):
                    if [source_ws.cell(header_row, column).value for column in range(1, 6)] != contract["headers"]:
                        continue
                    title = source_ws.cell(header_row - 1, 3).value
                    order: list[tuple[Any, Any]] = []
                    totals: dict[tuple[Any, Any], float | int] = {}
                    row = header_row + 1
                    while row <= source_ws.max_row and source_ws.cell(row, 1).value is not None:
                        date_value, reference = source_ws.cell(row, 2).value, source_ws.cell(row, 4).value
                        key = (_date_key(date_value), reference)
                        if key not in totals:
                            order.append((date_value, reference))
                            totals[key] = 0
                        totals[key] += _number(source_ws.cell(row, 5).value)
                        row += 1
                    ordered = sorted(order, key=lambda item: (_date_key(item[0]), str(item[1])))
                    sections[str(title)] = [
                        [index, date_value, reference, totals[(_date_key(date_value), reference)]]
                        for index, (date_value, reference) in enumerate(ordered, 1)
                    ]
                allowed: set[tuple[int, int]] = set()
                for header_row in range(1, output_ws.max_row + 1):
                    if [output_ws.cell(header_row, column).value for column in range(1, 5)] != ["SN", "DATE", "REF", "AMOUNTS"]:
                        continue
                    title = str(output_ws.cell(header_row - 1, 3).value)
                    expected = sections.get(title, [])
                    total_row = header_row + 1 + len(expected)
                    for offset, values in enumerate(expected, 1):
                        for column, value in enumerate(values, 1):
                            actual = output_ws.cell(header_row + offset, column).value
                            equal = _date_key(actual) == _date_key(value) if column == 2 else _values_equal(actual, value)
                            if not equal:
                                issues.append(f"section_value_mismatch:{title}:{output_ws.cell(header_row + offset, column).coordinate}")
                    if output_ws.cell(total_row, 1).value != "TOTAL":
                        issues.append(f"section_total_label_mismatch:{title}")
                    total_value = output_ws.cell(total_row, 4).value
                    expected_total = sum(_number(item[3]) for item in expected)
                    expected_formula = f"=SUM(D{header_row + 1}:D{total_row - 1})"
                    if not _values_equal(total_value, expected_total) and not _formula_equal(total_value, expected_formula):
                        issues.append(f"section_total_mismatch:{title}")
                    for row in range(header_row + 1, total_row + 1):
                        for column in range(1, 5):
                            allowed.add((row, column))
                issues.extend(_preservation_issues(source, candidate, {output_sheet: allowed}, styles=True))

            elif kind == "monthly_sheet_lookup_merge":
                output_sheet = contract["output_sheet"]
                output_ws = candidate[output_sheet]
                start_row, end_row = contract["row_range"]
                allowed: set[tuple[int, int]] = set()
                for output_letter, source_sheet in contract["output_columns"].items():
                    source_ws = source[source_sheet]
                    headers = {
                        str(source_ws.cell(1, column).value).strip().upper(): column
                        for column in range(1, source_ws.max_column + 1)
                        if source_ws.cell(1, column).value is not None
                    }
                    lookup = {
                        source_ws.cell(row, headers[contract["source_id_header"]]).value:
                        source_ws.cell(row, headers[contract["source_value_header"]]).value
                        for row in range(2, source_ws.max_row + 1)
                    }
                    output_column = column_index_from_string(output_letter)
                    for row in range(start_row, end_row + 1):
                        allowed.add((row, output_column))
                        key = source[output_sheet].cell(row, column_index_from_string(contract["id_column"])).value
                        expected = lookup.get(key)
                        if expected in (None, ""):
                            expected = contract["missing_value"]
                        if not _values_equal(output_ws.cell(row, output_column).value, expected):
                            issues.append(f"monthly_lookup_mismatch:{output_letter}{row}")
                issues.extend(_preservation_issues(source, candidate, {output_sheet: allowed}, styles=True))

            elif kind == "wide_to_long_unpivot":
                source_sheet, output_sheet = contract["source_sheet"], contract["output_sheet"]
                source_ws, output_ws = source[source_sheet], candidate[output_sheet]
                header_row = contract["source_header_row"]
                item_column = column_index_from_string(contract["item_column"])
                first_date_column = column_index_from_string(contract["first_date_column"])
                last_date_column = max(
                    column for column in range(first_date_column, source_ws.max_column + 1)
                    if source_ws.cell(header_row, column).value is not None
                )
                last_item_row = max(
                    row for row in range(header_row + 1, source_ws.max_row + 1)
                    if source_ws.cell(row, item_column).value is not None
                )
                expected = [
                    [source_ws.cell(row, item_column).value, source_ws.cell(header_row, column).value, source_ws.cell(row, column).value]
                    for row in range(header_row + 1, last_item_row + 1)
                    for column in range(first_date_column, last_date_column + 1)
                ]
                start_row = contract["output_start_row"]
                for index, values in enumerate(expected):
                    for column, value in enumerate(values, 1):
                        if not _values_equal(output_ws.cell(start_row + index, column).value, value):
                            issues.append(f"unpivot_mismatch:{output_ws.cell(start_row + index, column).coordinate}")
                            break
                    if issues and issues[-1].startswith("unpivot_mismatch"):
                        break
                actual_last = _meaningful_last_row(output_ws, 3)
                if actual_last != start_row + len(expected) - 1:
                    issues.append("unpivot_row_count_mismatch")
                allowed = {(row, column) for row in range(start_row, max(source[output_sheet].max_row, actual_last) + 1) for column in range(1, 4)}
                issues.extend(_preservation_issues(source, candidate, {output_sheet: allowed}, styles=False))
                details.update(expected_rows=len(expected), actual_rows=max(0, actual_last - start_row + 1))

            elif kind == "split_multiline_rows":
                source_sheet, output_sheet = contract["source_sheet"], contract["output_sheet"]
                source_ws, output_ws = source[source_sheet], candidate[output_sheet]
                expected: list[list[Any]] = []
                for row in range(1, source_ws.max_row + 1):
                    key, raw = source_ws.cell(row, 1).value, source_ws.cell(row, 2).value
                    if key is None and raw is None:
                        continue
                    tokens = str(raw).splitlines() if raw not in (None, "") else [None]
                    for token in tokens:
                        value: Any = token
                        if isinstance(token, str):
                            stripped = token.strip()
                            value = int(stripped) if re.fullmatch(r"[-+]?\d+", stripped) else float(stripped) if re.fullmatch(r"[-+]?(?:\d+\.\d*|\d*\.\d+)", stripped) else stripped
                        expected.append([key, value])
                start_row = contract["output_start_row"]
                for index, values in enumerate(expected):
                    for column, value in enumerate(values, 1):
                        if not _values_equal(output_ws.cell(start_row + index, column).value, value):
                            issues.append(f"split_row_mismatch:{output_ws.cell(start_row + index, column).coordinate}")
                actual_last = _meaningful_last_row(output_ws, 2)
                if actual_last != start_row + len(expected) - 1:
                    issues.append("split_row_count_mismatch")
                allowed = {(row, column) for row in range(start_row, max(source[output_sheet].max_row, actual_last) + 1) for column in range(1, 3)}
                issues.extend(_preservation_issues(source, candidate, {output_sheet: allowed}, styles=False))
                details.update(expected_rows=len(expected), actual_rows=max(0, actual_last - start_row + 1))

            elif kind == "last_transfer_count_formulas":
                sheet = contract["sheet"]
                source_ws, output_ws = source[sheet], candidate[sheet]
                start_row, end_row = contract["row_range"]
                output_column = column_index_from_string(contract["output_column"])
                allowed = {(row, output_column) for row in range(start_row, end_row + 1)}
                for row in range(start_row, end_row + 1):
                    expected = (
                        f'=IF(OR(D{row}<>{contract["target_operator"]},D{row + 1}={contract["target_operator"]}),"",'
                        f'COUNTIF($D$5:D{row},{contract["target_operator"]})-SUM($H$4:H{row - 1}))'
                    )
                    if not _formula_equal(output_ws.cell(row, output_column).value, expected):
                        issues.append(f"transfer_formula_mismatch:H{row}")
                blank_column = column_index_from_string(contract["blank_unbordered_column"])
                for row in range(1, output_ws.max_row + 1):
                    cell = output_ws.cell(row, blank_column)
                    if cell.value is not None or _has_visible_border(cell):
                        issues.append(f"separator_column_contract_mismatch:{cell.coordinate}")
                        break
                issues.extend(_preservation_issues(source, candidate, {sheet: allowed}, styles=True))

            elif kind == "two_key_lookup_formulas":
                sheet = contract["sheet"]
                source_ws, output_ws = source[sheet], candidate[sheet]
                start_row, end_row = contract["row_range"]
                output_column = column_index_from_string(contract["output_column"])
                lookup_last = max(
                    row for row in range(3, source_ws.max_row + 1)
                    if any(source_ws.cell(row, column).value is not None for column in (8, 9, 10))
                )
                allowed = {(row, output_column) for row in range(start_row, end_row + 1)}
                for row in range(start_row, end_row + 1):
                    expected = (
                        f'=IFERROR(INDEX($H$3:$H${lookup_last},MATCH(1,INDEX(($J$3:$J${lookup_last}=C{row})*'
                        f'(I$3:$I${lookup_last}=B{row}),0),0)),"")'
                    )
                    cell = output_ws.cell(row, output_column)
                    if not _formula_equal(cell.value, expected):
                        issues.append(f"lookup_formula_mismatch:{cell.coordinate}")
                    if _fill_rgb(cell) != contract["output_fill_rgb"]:
                        issues.append(f"fill_contract_mismatch:{cell.coordinate}")
                if output_ws.sheet_view.showGridLines is not False:
                    issues.append("sheet_view_contract_mismatch:showGridLines")
                # The sheet-view mutation is explicitly requested; cell preservation remains strict.
                original_gridlines = source_ws.sheet_view.showGridLines
                source_ws.sheet_view.showGridLines = output_ws.sheet_view.showGridLines
                issues.extend(_preservation_issues(source, candidate, {sheet: allowed}, styles=True))
                source_ws.sheet_view.showGridLines = original_gridlines

            else:
                issues.append(f"unknown_validation_kind:{kind}")
            return {"passed": not issues, "issues": issues[:50], **details}
        finally:
            source.close()
            candidate.close()

    def _public_contract_validation(self, env, output: Path):
        contract = read_json(env.workspace_dir / "public/validation_contract.json")
        return self.validate_output(self._input(env), output, contract)

    def execute_tool(self, env, tool_name, arguments=None):
        arguments, state = dict(arguments or {}), self._state(env)
        versions = self.versions(env)
        status, output = "success", {}
        try:
            if tool_name == "inspect_workbook":
                output = workbook_snapshot(
                    self._input(env),
                    sheet_name=arguments.get("sheet"),
                    cell_range=arguments.get("cell_range"),
                )
                self._store_json(env, state, "workbook_inventory", output, {"public_input_hash": versions["public_input_hash"]}, tool_name)
            elif tool_name == "profile_workbook_data":
                output = self._profile(self._input(env), arguments)
                self._store_json(env, state, "data_profile", output, {"public_input_hash": versions["public_input_hash"]}, tool_name)
            elif tool_name == "inspect_formulas":
                snapshot = workbook_snapshot(
                    self._input(env),
                    sheet_name=arguments.get("sheet"),
                    cell_range=arguments.get("cell_range"),
                )
                output = {"sheets": [{"name": sheet["name"], "formulas": sheet["formulas"]} for sheet in snapshot["sheets"]]}
                self._store_json(env, state, "formula_analysis", output, {"public_input_hash": versions["public_input_hash"]}, tool_name)
            elif tool_name == "read_artifact":
                artifact_id = str(arguments["artifact_id"])
                artifacts = [item for item in state["artifact_history"] if item["artifact_id"] == artifact_id or item["logical_key"] == artifact_id]
                if arguments.get("version") is not None:
                    artifacts = [item for item in artifacts if item["version"] == int(arguments["version"])]
                if not artifacts:
                    raise ValueError("Unknown allowlisted artifact")
                artifact = artifacts[-1]
                path = (env.workspace_dir / artifact["path"]).resolve()
                if not path.is_relative_to((env.workspace_dir / "_runtime/artifacts").resolve()):
                    raise ValueError("Artifact path is outside the allowlist")
                text = path.read_text(encoding="utf-8", errors="replace")
                offset, limit = int(arguments.get("offset", 0)), int(arguments.get("limit", 8000))
                output = {"artifact_id": artifact["artifact_id"], "offset": offset, "next_offset": min(offset + limit, len(text)), "total_length": len(text), "content": text[offset:offset + limit]}
            elif tool_name in {"write_solution", "revise_solution"}:
                source = self._strip_code_fence(str(arguments["python_source"]))
                if tool_name == "revise_solution":
                    current = state["artifacts"].get("candidate_solution")
                    if not current or arguments["base_candidate_artifact_id"] != current["artifact_id"]:
                        raise ValueError("Repair must reference the current candidate artifact")
                try:
                    normalize_and_validate_source(source)
                except CandidatePolicyError as error:
                    rejected = env.workspace_dir / f"_runtime/rejected/{len(state['rejected_attempts']):04}.py"
                    rejected.parent.mkdir(parents=True, exist_ok=True)
                    rejected.write_text(source, encoding="utf-8", newline="\n")
                    item = {"path": str(rejected.relative_to(env.workspace_dir)).replace("\\", "/"), "sha256": sha256_file(rejected), "error": str(error), "producer_tool": tool_name}
                    state["rejected_attempts"].append(item)
                    status, output = "invalid_output", {"error_type": type(error).__name__, "error": str(error), "rejected_attempt": item}
                else:
                    refs = ["candidate_solution"] if tool_name == "revise_solution" else []
                    artifact = self._store_bytes(env, state, "candidate_solution", source.encode("utf-8"), ".py", {"public_input_hash": versions["public_input_hash"], "runtime_profile": versions["runtime_profile"]}, tool_name, refs)
                    output = {"artifact": artifact, "entrypoint": "solve(input_path, output_path)"}
            elif tool_name == "run_solution_public":
                candidate, candidate_path = self._candidate_source(env, state)
                if arguments["candidate_artifact_id"] != candidate["artifact_id"]:
                    raise ValueError("Execution must reference the current candidate")
                run_dir = env.workspace_dir / f"_runtime/executions/{len(state['events']):04}"
                output_path = run_dir / "output.xlsx"
                result = self._execute_candidate(env, candidate_path, self._input(env), output_path, int(read_json(env.workspace_dir / "public/execution_contract.json")["case_timeout_seconds"]))
                if output_path.exists():
                    self._store_bytes(env, state, "public_output", output_path.read_bytes(), ".xlsx", {"candidate_hash": candidate["sha256"], "public_input_hash": versions["public_input_hash"], "runtime_profile": versions["runtime_profile"]}, tool_name, ["candidate_solution"])
                execution = self._store_json(env, state, "public_execution", result, {"candidate_hash": candidate["sha256"], "public_input_hash": versions["public_input_hash"], "runtime_profile": versions["runtime_profile"]}, tool_name, ["candidate_solution", "public_output"])
                state["latest_execution"] = {"status": result["status"], "candidate_hash": candidate["sha256"], "artifact_id": execution["artifact_id"], "summary": {key: result.get(key) for key in ("returncode", "latency_ms", "failure_origin", "timeout", "workbook_error")}}
                output = {"result": result, "execution_artifact": execution, "output_artifact": state["artifacts"].get("public_output")}
                status = "success" if result["status"] == "success" else "failed"
            elif tool_name == "validate_public_output":
                candidate = state["artifacts"]["candidate_solution"]
                execution = state["artifacts"]["public_execution"]
                if arguments["candidate_artifact_id"] != candidate["artifact_id"] or arguments["execution_artifact_id"] != execution["artifact_id"]:
                    raise ValueError("Validation must reference current candidate and execution")
                if not self._fresh(execution, state, self.versions(env)):
                    raise ValueError("Execution evidence is stale")
                output_artifact = state["artifacts"]["public_output"]
                validation = self._public_contract_validation(env, env.workspace_dir / output_artifact["path"])
                artifact = self._store_json(env, state, "public_validation", validation, {"candidate_hash": candidate["sha256"], "public_input_hash": versions["public_input_hash"], "runtime_profile": versions["runtime_profile"]}, tool_name, ["candidate_solution", "public_execution", "public_output"], {"passed": validation["passed"]})
                output = {"validation": validation, "artifact": artifact}
                status = "success" if validation["passed"] else "failed"
            else:
                raise ValueError("Unknown tool")
        except (ValueError, KeyError, OSError) as error:
            status, output = "invalid_output", {"error_type": type(error).__name__, "error": str(error)}
        event = {"tool_name": tool_name, "status": status, "output": output, "versions_after": self.versions(env), "origin": "selected_agent"}
        state["events"].append(event)
        self._save(env, state)
        return ToolResult(tool_name, status, output, json.dumps(output, ensure_ascii=False, default=_json_value))

    def terminal_status(self, env):
        return {"terminal": bool(self.public_observation(env)["terminal"])}

    def validate_postcondition(self, env):
        return self.terminal_status(env)

    def artifact_diff(self, before, after):
        from ..base import ArtifactDiff
        before, after = dict(before), dict(after)
        return ArtifactDiff(
            added=tuple({"path": key, "sha256": value} for key, value in after.items() if key not in before),
            modified=tuple({"path": key, "before": before[key], "after": value} for key, value in after.items() if key in before and before[key] != value),
            removed=tuple({"path": key, "sha256": value} for key, value in before.items() if key not in after),
        )
