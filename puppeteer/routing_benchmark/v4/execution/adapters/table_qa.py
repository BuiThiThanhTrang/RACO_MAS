"""Shared live adapter for question answering over normalized tables."""
from __future__ import annotations

from collections import Counter
from datetime import date, datetime
from decimal import Decimal, InvalidOperation, ROUND_HALF_UP, localcontext
import json
from pathlib import Path
import re
import shutil
import unicodedata

import jsonschema

from ...artifacts.manifest import load_manifest
from ...artifacts.schema import sha256_file
from ..base import ArtifactDiff, EnvironmentHandle, PreconditionResult, ToolDefinition, ToolResult
from .common import read_json, write_json


ADAPTER_ID = "structured_data.table_qa.v1"
TEXT = {"type": "string", "minLength": 1, "maxLength": 250}
IDS = {"type": "array", "items": TEXT, "minItems": 1, "maxItems": 25, "uniqueItems": True}
ANSWER_SCHEMA = {
    "type": "array", "minItems": 1, "maxItems": 25,
    "items": {
        "oneOf": [
            {
                "type": "object", "additionalProperties": False, "required": ["type", "value"],
                "properties": {
                    "type": {"const": kind}, "value": value,
                    **({"unit": TEXT} if kind == "number" else {}),
                },
            }
            for kind, value in (
                ("text", TEXT), ("number", {"type": ["number", "string"]}),
                ("date", {"type": "string", "pattern": r"^\d{4}(?:-\d{2}(?:-\d{2})?)?$"}),
                ("boolean", {"type": "boolean"}),
            )
        ]
    },
}


def decimal_value(value):
    if isinstance(value, bool):
        raise ValueError("Expected a numeric cell")
    text = str(value).strip().replace("−", "-")
    if re.fullmatch(r"[-+]?\d{1,3}(?:,\d{3})+(?:\.\d+)?", text):
        text = text.replace(",", "")
    if not re.fullmatch(r"[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d{1,3})?", text) or len(text) > 100:
        raise ValueError("Expected a numeric value")
    number = Decimal(text)
    if number and abs(number.adjusted()) > 100:
        raise ValueError("Number outside supported range")
    return number


def normalized_text(value):
    text = unicodedata.normalize("NFKC", value)
    return " ".join(re.sub(r"[‐‑‒–—−]", "-", text).split()).casefold()


def canonical_value(item):
    kind, value = item["type"], item["value"]
    if kind == "number":
        return kind, decimal_value(value), normalized_text(item.get("unit", ""))
    if kind == "date":
        date.fromisoformat({4: value + "-01-01", 7: value + "-01"}.get(len(value), value))
        return kind, value
    return kind, normalized_text(value) if kind == "text" else value


def score_answer(answer, gold):
    """Compare typed answers using the same rules for every source."""
    jsonschema.validate(gold["answer"], ANSWER_SCHEMA)
    expected = [canonical_value(item) for item in gold["answer"]]
    try:
        jsonschema.validate(answer, ANSWER_SCHEMA)
        actual = [canonical_value(item) for item in answer]
    except (ValueError, InvalidOperation, jsonschema.ValidationError):
        return False
    aliases = {normalized_text(key): normalized_text(value) for key, value in gold.get("aliases", {}).items()}
    actual = [("text", aliases.get(item[1], item[1])) if item[0] == "text" else item for item in actual]
    tolerance = decimal_value(gold.get("absolute_tolerance", 0))

    def match(left, right):
        if left[0] != right[0]:
            return False
        if left[0] == "number":
            return left[2] == right[2] and abs(left[1] - right[1]) <= tolerance
        return left == right

    if gold.get("order", "sequence") == "sequence":
        return len(actual) == len(expected) and all(match(a, b) for a, b in zip(actual, expected))
    if gold["order"] == "set":
        actual, expected = list(dict.fromkeys(actual)), list(dict.fromkeys(expected))
    if len(actual) != len(expected):
        return False
    # A bipartite match avoids order-dependent scoring when tolerance windows overlap.
    matched = {}

    def assign(index, seen):
        for target, value in enumerate(expected):
            if target in seen or not match(actual[index], value):
                continue
            seen.add(target)
            if target not in matched or assign(matched[target], seen):
                matched[target] = index
                return True
        return False

    return all(assign(index, set()) for index in range(len(actual)))


def typed_cell(value):
    if isinstance(value, bool):
        return {"type": "boolean", "value": value}
    if isinstance(value, (int, float)):
        return {"type": "number", "value": value}
    if isinstance(value, str):
        text = value.strip()
        if re.fullmatch(r"\d{4}-\d{2}(?:-\d{2})?", text):
            try:
                item = {"type": "date", "value": text}
                canonical_value(item)
                return item
            except ValueError:
                pass
        if re.fullmatch(r"[A-Za-z]+ \d{1,2}, \d{4}", text):
            for pattern in ("%B %d, %Y", "%b %d, %Y"):
                try:
                    return {"type": "date", "value": datetime.strptime(text, pattern).date().isoformat()}
                except ValueError:
                    continue
        if not re.match(r"^0\d", text):
            try:
                return {"type": "number", "value": str(decimal_value(text.rstrip("%"))),
                        **({"unit": "%"} if text.endswith("%") else {})}
            except ValueError:
                pass
        return {"type": "text", "value": text}
    raise ValueError("The selected cell is empty or unsupported")


def evidence_cell(value, headers):
    cell = typed_cell(value)
    if cell["type"] == "number" and "unit" not in cell:
        for label in headers:
            label = normalized_text(label)
            if label in ("%", "percent", "percentage") or re.search(r"\((?:%|percent|percentage)\)$", label):
                cell["unit"] = "%"
                break
    return cell


def row_preview(rows, limit):
    shown = rows[:limit]
    return {"rows": shown, "row_count": len(rows), "shown_row_count": len(shown), "rows_truncated": len(shown) < len(rows)}


def tool(name, description, properties, required=()):
    return ToolDefinition(name, description, {
        "type": "object", "properties": properties,
        "required": list(required), "additionalProperties": False,
    })


TABLE = {"table_id": TEXT}
DEPENDENCIES = {"evidence_ids": IDS}
FILTERS = {
    "type": "array", "maxItems": 5,
    "items": {
        "type": "object", "additionalProperties": False, "required": ["column", "op", "value"],
        "properties": {
            "column": TEXT, "op": {"enum": ["eq", "ne", "contains", "lt", "le", "gt", "ge"]},
            "value": {"type": ["string", "number", "boolean", "null"]},
        },
    },
}
TOOLS = {
    "inspect_table": tool("inspect_table", "Read up to 12 rows from a public table. Offset is zero-based.", {
        **TABLE, "offset": {"type": "integer", "minimum": 0}, "columns": IDS,
    }, ("table_id",)),
    "resolve_headers": tool("resolve_headers", "Record which public row and column IDs resolve the question's header hierarchy.", {
        **TABLE, "row_ids": IDS, "columns": IDS,
    }, ("table_id", "columns")),
    "select_evidence": tool("select_evidence", "Select rows by ID or by an AND of filters, then keep the specified columns.", {
        **TABLE, "columns": IDS, "row_ids": IDS, "filters": FILTERS, **DEPENDENCIES,
    }, ("table_id", "columns")),
    "analyze_table": tool("analyze_table", "Compute from a table or selection. Supply two ordered row IDs for difference or ratio. Use columns for sum across columns, or argmin/argmax across columns of one row. header_level is zero-based for returning the winning column's header label.", {
        **TABLE, "selection_id": TEXT, "column": TEXT, "columns": IDS, "return_column": TEXT, "row_ids": IDS,
        "header_level": {"type": "integer", "minimum": 0, "maximum": 8},
        "operation": {"enum": ["lookup", "count", "distinct_count", "mode", "sum", "mean", "min", "max", "argmin", "argmax", "difference", "ratio"]},
        "submit": {"type": "boolean"}, "unit": TEXT, "round_digits": {"type": "integer", "minimum": 0, "maximum": 10},
        **DEPENDENCIES,
    }, ("table_id", "operation")),
    "submit_answer": tool("submit_answer", "Submit typed answer values supported by fresh evidence IDs or explicit cells in the visible table preview.", {
        "answer": ANSWER_SCHEMA, **DEPENDENCIES,
        "preview_cells": {
            "type": "array", "minItems": 1, "maxItems": 25, "uniqueItems": True,
            "items": {
                "type": "object", "additionalProperties": False,
                "required": ["table_id", "row_id", "column"],
                "properties": {**TABLE, "row_id": TEXT, "column": TEXT},
            },
        },
    }, ("answer",)),
    "verify_answer": tool("verify_answer", 'Verify the current answer using public evidence. For no issue, return {"tool":"verify_answer","arguments":{}}. Otherwise supply one concrete issue of 1–250 characters.', {
        "issue": TEXT,
    }),
    "diagnose_failure": tool("diagnose_failure", "Record the reason for the last failed verification and suggest which operation needs revision.", {
        "problem": TEXT, "step": {"enum": ["headers", "selection", "calculation", "answer"]},
    }, ("problem", "step")),
}
ROLE_TO_TOOL = {
    "Table Inspector": "inspect_table",
    "Header Resolver": "resolve_headers",
    "Evidence Selector": "select_evidence",
    "Numerical Analyst": "analyze_table",
    "Answer Composer": "submit_answer",
    "Evidence Verifier": "verify_answer",
    "Failure Diagnoser": "diagnose_failure",
}
SINGLE_AGENT_ROLE = "Table QA Agent"


class TableQAAdapter:
    adapter_id = ADAPTER_ID

    def materialize(self, manifest_path, workspace):
        manifest = load_manifest(manifest_path)
        pack = Path(manifest_path).parent if Path(manifest_path).is_file() else Path(manifest_path)
        workspace = Path(workspace)
        if manifest.adapter_id != self.adapter_id:
            raise ValueError("Wrong adapter for the table QA packet")
        if {item.path for item in manifest.public_files} != {"public/task.json", "public/tables.json"}:
            raise ValueError("Unexpected public files")
        workspace.mkdir(parents=True, exist_ok=False)
        (workspace / "public").mkdir()
        for item in manifest.public_files:
            source = (pack / item.path).resolve()
            if not source.is_relative_to(pack.resolve()) or sha256_file(source) != item.sha256:
                raise ValueError("Public table artifact failed integrity check")
            shutil.copyfile(source, workspace / item.path)
        write_json(workspace / "state.json", {"current": {}, "history": {}, "last_error": None})
        return EnvironmentHandle(manifest, pack, workspace)

    def _tables(self, env):
        return read_json(env.workspace_dir / "public/tables.json")["tables"]

    def _state(self, env):
        return read_json(env.workspace_dir / "state.json")

    def record_contract_error(self, env, error):
        state = self._state(env)
        state["last_error"] = error
        write_json(env.workspace_dir / "state.json", state)

    @staticmethod
    def progress_observation(observation):
        return {key: value for key, value in observation.items() if key != "last_error"}

    def _fresh(self, state, ref):
        item = state["history"][ref]
        return (item["kind"] == "inspection" or state["current"].get(item["kind"]) == ref) and all(
            self._fresh(state, parent) for parent in item["depends_on"]
        )

    def _evidence(self, state, refs, kinds=None):
        if any(ref not in state["history"] or not self._fresh(state, ref) for ref in refs):
            raise ValueError("Evidence ID is missing or stale")
        items = [state["history"][ref] for ref in refs]
        if kinds is not None and any(item["kind"] not in kinds for item in items):
            raise ValueError("Evidence kind is incompatible with this operation")
        return items

    def _record(self, state, kind, data, refs=()):
        payload = {"kind": kind, "data": data, "depends_on": list(refs)}
        current = state["current"].get(kind)
        if kind == "inspection":
            matches = [ref for ref, item in state["history"].items() if item == payload]
            if matches:
                return matches[0], False
        if current and state["history"][current] == payload:
            return current, False
        ref = f"{kind}:{len(state['history']) + 1}"
        state["history"][ref] = payload
        state["current"][kind] = ref
        return ref, True

    def public_observation(self, env):
        state = self._state(env)
        visible = dict(state["current"])
        recent_inspections = [ref for ref, item in state["history"].items() if item["kind"] == "inspection"][-3:]
        refs = dict.fromkeys([*visible.values(), *recent_inspections])
        answer = visible.get("answer")
        if answer:
            dependencies = list(state["history"][answer]["depends_on"])
            while dependencies:
                ref = dependencies.pop()
                refs.setdefault(ref)
                dependencies.extend(state["history"][ref]["depends_on"])
        artifacts = {}
        for ref in refs:
            item = state["history"][ref]
            data = dict(item["data"])
            if "rows" in data:
                data.update(row_preview(data["rows"], 12 if item["kind"] == "inspection" else 6))
            artifacts[ref] = {**item, "data": data, "fresh": self._fresh(state, ref)}
        check = visible.get("verification")
        verification = {"answer_id": answer, "status": "not_checked", "artifact_id": None}
        if check and artifacts[check]["fresh"]:
            verification.update(status="passed" if artifacts[check]["data"]["passed"] else "failed", artifact_id=check)
        return {
            "state_version": self.adapter_id,
            "task": read_json(env.workspace_dir / "public/task.json"),
            "tables": [{"table_id": t["table_id"], "title": t.get("title"), "row_count": len(t["rows"]),
                        "columns": t["columns"], "hierarchical": t.get("hierarchical", False)}
                       for t in self._tables(env)],
            "current": visible, "artifacts": artifacts, "last_error": state["last_error"],
            "verification_status": verification, "terminal": verification["status"] == "passed",
        }

    def executor_context(self, env, action, role_profile=None):
        context = self.public_observation(env)
        context["table_preview"] = [
            {"table_id": table["table_id"], **row_preview(table["rows"], 5)} for table in self._tables(env)
        ]
        return context

    def public_preconditions(self, env, action_id, role_profile=None):
        if action_id == SINGLE_AGENT_ROLE:
            return PreconditionResult(True)
        if action_id not in ROLE_TO_TOOL:
            return PreconditionResult(False, ("Unknown role",))
        state = self._state(env)
        current = state["current"]
        if action_id == "Evidence Verifier" and (
            "answer" not in current or not self._fresh(state, current["answer"])
        ):
            return PreconditionResult(False, ("A current answer with fresh evidence is required",))
        if action_id == "Failure Diagnoser" and not state["last_error"]:
            return PreconditionResult(False, ("No public failure to diagnose",))
        return PreconditionResult(True)

    def tool_spec(self, env, action_id, role_profile=None):
        if action_id == SINGLE_AGENT_ROLE:
            return tuple(TOOLS.values())
        name = ROLE_TO_TOOL.get(action_id)
        return (TOOLS[name],) if name else ()

    @staticmethod
    def _matches(actual, op, expected):
        if op == "contains":
            return isinstance(actual, str) and normalized_text(str(expected)) in normalized_text(actual)
        if op in ("eq", "ne"):
            try:
                equal = decimal_value(actual) == decimal_value(expected)
            except ValueError:
                equal = (normalized_text(actual) == normalized_text(expected)
                         if isinstance(actual, str) and isinstance(expected, str)
                         else type(actual) is type(expected) and actual == expected)
            return equal if op == "eq" else not equal
        try:
            a, b = decimal_value(actual), decimal_value(expected)
        except ValueError:
            return False
        return {"lt": a < b, "le": a <= b, "gt": a > b, "ge": a >= b}[op]

    def _table_operation(self, env, state, name, arguments):
        table_id = arguments["table_id"]
        table = next((t for t in self._tables(env) if t["table_id"] == table_id), None)
        if table is None:
            raise ValueError("Unknown public table ID")
        columns = {col["id"]: col for col in table["columns"]}
        rows = {row["id"]: row for row in table["rows"]}
        chosen_columns = arguments.get("columns", list(columns))
        if not set(chosen_columns) <= columns.keys():
            raise ValueError("Unknown column ID")
        refs = list(arguments.get("evidence_ids", []))
        self._evidence(state, refs, ("inspection", "headers", "selection") if name == "analyze_table"
                       else ("inspection", "headers"))
        selected_ids = arguments.get("row_ids")
        if selected_ids and not set(selected_ids) <= rows.keys():
            raise ValueError("Unknown row ID")
        chosen_rows = [rows[ref] for ref in selected_ids] if selected_ids else list(rows.values())
        if name == "inspect_table":
            offset = arguments.get("offset", 0)
            if offset >= len(chosen_rows):
                raise ValueError("Offset beyond the table")
            chosen_rows = chosen_rows[offset:offset + 12]
            kind = "inspection"
        elif name == "resolve_headers":
            return self._record(state, "headers", {
                "table_id": table_id, "columns": [columns[col] for col in chosen_columns],
                "rows": [{"id": row["id"], "header_path": row.get("header_path", [])}
                         for row in chosen_rows] if selected_ids else [],
            })
        elif name == "select_evidence":
            filters = arguments.get("filters", [])
            if any(f["column"] not in columns for f in filters):
                raise ValueError("Unknown filter column")
            chosen_rows = [
                row for row in chosen_rows if all(
                    self._matches(row["values"].get(f["column"]), f["op"], f["value"]) for f in filters
                )
            ]
            kind = "selection"
        else:
            selection_id = arguments.get("selection_id")
            if selection_id:
                selection, = self._evidence(state, [selection_id])
                if selection["kind"] != "selection" or selection["data"]["table_id"] != table_id:
                    raise ValueError("Expected a selection from this table")
                selected = selection["data"]
                allowed_rows = {row["id"] for row in selected["rows"]}
                if selected_ids and not set(selected_ids) <= allowed_rows:
                    raise ValueError("Requested rows outside the selection")
                chosen_rows = [row for row in chosen_rows if row["id"] in allowed_rows]
                if arguments.get("column") not in (None, *selected["columns"]) or arguments.get("return_column") not in (None, *selected["columns"]):
                    raise ValueError("Requested column outside the selection")
                if not set(arguments.get("columns", [])) <= set(selected["columns"]):
                    raise ValueError("Requested columns outside the selection")
                refs = list(dict.fromkeys([*refs, selection_id]))
            answer = self._calculate(columns, chosen_rows, arguments)
            ref, changed = self._record(state, "calculation", {
                "table_id": table_id, "row_ids": [row["id"] for row in chosen_rows],
                "operation": arguments["operation"],
                "parameters": {key: value for key, value in arguments.items() if key not in (
                    "table_id", "operation", "row_ids", "evidence_ids", "submit",
                )},
                "answer": answer,
            }, refs)
            if arguments.get("submit"):
                return self._record(state, "answer", {"answer": answer}, [ref])
            return ref, changed
        return self._record(state, kind, {
            "table_id": table_id, "columns": chosen_columns,
            "rows": [{**row, "values": {col: row["values"].get(col) for col in chosen_columns}}
                     for row in chosen_rows],
            **({"filters": arguments.get("filters", [])} if kind == "selection" else {"offset": offset}),
        }, refs)

    @staticmethod
    def _calculate(columns, rows, arguments):
        op, column = arguments["operation"], arguments.get("column")
        selected_columns = arguments.get("columns")
        if selected_columns:
            if column is not None or op not in ("sum", "argmin", "argmax"):
                raise ValueError("Specify either one column or a list of columns for sum, argmin or argmax")
            if not rows:
                raise ValueError("No rows in the selected scope")
            if op == "sum":
                values = [
                    decimal_value(row["values"][col]) for row in rows for col in selected_columns
                    if row["values"].get(col) not in (None, "")
                ]
                if not values:
                    raise ValueError("No numeric values in the selected scope")
                answer = [{"type": "number", "value": str(sum(values))}]
                if "unit" in arguments:
                    answer[0]["unit"] = arguments["unit"]
                return answer
            if len(rows) != 1:
                raise ValueError("Select exactly one row for a column comparison")
            values = {col: decimal_value(rows[0]["values"][col]) for col in selected_columns}
            extreme = (min if op == "argmin" else max)(values.values())
            level = arguments.get("header_level", 0)
            return [
                typed_cell(columns[col]["header_path"][level])
                for col in selected_columns if values[col] == extreme
            ]
        if column is not None and column not in columns:
            raise ValueError("Unknown computation column")
        if op == "count":
            value = sum(row["values"].get(column) is not None for row in rows) if column else len(rows)
            return [typed_cell(value)]
        if column is None or not rows:
            raise ValueError("This operation requires a column and at least one row")
        if op == "lookup":
            answer = [typed_cell(row["values"].get(column)) for row in rows]
        elif op in ("distinct_count", "mode"):
            items = [typed_cell(row["values"][column]) for row in rows if row["values"].get(column) not in (None, "")]
            by_key = {canonical_value(item): item for item in items}
            if op == "distinct_count":
                answer = [typed_cell(len(by_key))]
            elif items:
                counts = Counter(canonical_value(item) for item in items)
                maximum = max(counts.values())
                answer = [by_key[key] for key, count in counts.items() if count == maximum]
            else:
                raise ValueError("No populated values in the selected scope")
        else:
            pairs = [(row, decimal_value(row["values"][column])) for row in rows
                     if row["values"].get(column) not in (None, "")]
            values = [value for _, value in pairs]
            if not values:
                raise ValueError("No numeric values in the selected scope")
            if op in ("argmin", "argmax"):
                output = arguments.get("return_column")
                if output not in columns:
                    raise ValueError("argmin and argmax require a return_column")
                extreme = (min if op == "argmin" else max)(values)
                answer = [typed_cell(row["values"].get(output)) for row, value in pairs if value == extreme]
            else:
                with localcontext() as context:
                    context.prec = 28
                    if op in ("difference", "ratio"):
                        if len(rows) != 2 or len(values) != 2 or not arguments.get("row_ids"):
                            raise ValueError("Specify exactly two non-empty operands using row_ids")
                        if op == "ratio" and values[1] == 0:
                            raise ValueError("Cannot divide by zero")
                        result = values[0] - values[1] if op == "difference" else values[0] / values[1]
                    elif op == "mean":
                        result = sum(values) / len(values)
                    else:
                        result = {"min": min, "max": max, "sum": sum}[op](values)
                    if "round_digits" in arguments:
                        result = result.quantize(Decimal(1).scaleb(-arguments["round_digits"]), rounding=ROUND_HALF_UP)
                    answer = [{"type": "number", "value": str(result)}]
        if arguments.get("unit"):
            answer = [{**item, "unit": arguments["unit"]} if item["type"] == "number" else item for item in answer]
        jsonschema.validate(answer, ANSWER_SCHEMA)
        return answer

    def _preview_values(self, env, refs):
        tables = {table["table_id"]: table for table in self._tables(env)}
        values = []
        for ref in refs:
            table = tables.get(ref["table_id"])
            row = next((row for row in table["rows"][:5] if row["id"] == ref["row_id"]), None) if table else None
            if row is None or ref["column"] not in row["values"]:
                raise ValueError("Cell reference must identify a cell in the public table preview")
            column = next(column for column in table["columns"] if column["id"] == ref["column"])
            values.append(evidence_cell(row["values"][ref["column"]], [*row.get("header_path", []), *column["header_path"]]))
        return values

    def _verify(self, env, state, arguments):
        ref = state["current"].get("answer")
        if not ref or not self._fresh(state, ref):
            raise ValueError("No fresh candidate answer")
        answer = state["history"][ref]
        supported = {canonical_value(item) for item in self._preview_values(env, answer["data"].get("preview_cells", []))}
        column_paths = {table["table_id"]: {column["id"]: column["header_path"] for column in table["columns"]}
                        for table in self._tables(env)}
        for item in self._evidence(state, answer["depends_on"]):
            data = item["data"]
            values = list(data.get("answer", []))
            paths = column_paths.get(data.get("table_id"), {})
            for row in data.get("rows", []):
                headers = row.get("header_path", [])
                values.extend(evidence_cell(value, [*headers, *paths.get(column, [])])
                              for column, value in row.get("values", {}).items() if value not in (None, ""))
                values.extend(typed_cell(label) for label in headers)
            if item["kind"] == "headers":
                for column in data["columns"]:
                    values.extend(typed_cell(label) for label in column["header_path"])
            supported.update(canonical_value(value) for value in values)
        issue = arguments.get("issue")
        if any(canonical_value(item) not in supported for item in answer["data"]["answer"]):
            issue = issue or "Candidate value not supported by the referenced public evidence."
        return self._record(state, "verification", {"passed": issue is None, "issue": issue}, [ref])

    def execute_tool(self, env, tool_name, arguments=None):
        arguments = dict(arguments or {})
        state = self._state(env)
        try:
            jsonschema.validate(arguments, TOOLS[tool_name].input_schema)
            if tool_name == "submit_answer":
                for item in arguments["answer"]:
                    canonical_value(item)
                refs = arguments.get("evidence_ids", [])
                preview_cells = arguments.get("preview_cells", [])
                if not refs and not preview_cells:
                    raise ValueError("Supply evidence_ids or preview_cells for the answer")
                self._evidence(state, refs, ("inspection", "headers", "selection", "calculation"))
                if preview_cells:
                    self._preview_values(env, preview_cells)
                ref, changed = self._record(state, "answer", {
                    "answer": arguments["answer"], **({"preview_cells": preview_cells} if preview_cells else {}),
                }, refs)
            elif tool_name == "verify_answer":
                ref, changed = self._verify(env, state, arguments)
            elif tool_name == "diagnose_failure":
                if not state["last_error"]:
                    raise ValueError("No public failure to diagnose")
                ref, changed = self._record(state, "diagnosis", arguments)
            else:
                ref, changed = self._table_operation(env, state, tool_name, arguments)
            item = state["history"][ref]
            issue = item["data"].get("issue") if item["kind"] == "verification" else None
            if issue:
                state["last_error"] = {"tool": tool_name, "message": issue}
            elif (changed and item["kind"] in ("answer", "verification")) or (state["last_error"] or {}).get("tool") == tool_name:
                state["last_error"] = None
            status = "failed" if issue else "success" if changed else "no_progress"
            data = dict(item["data"])
            if "rows" in data:
                data.update(row_preview(data["rows"], 12))
            output = {"artifact_id": ref, "kind": item["kind"], "data": data}
        except (KeyError, IndexError, ValueError, InvalidOperation, jsonschema.ValidationError) as error:
            message = error.message if isinstance(error, jsonschema.ValidationError) else str(error)
            output = {"error": message}
            state["last_error"] = {"tool": tool_name, "message": message}
            status = "invalid_output"
        write_json(env.workspace_dir / "state.json", state)
        return ToolResult(tool_name, status, output, json.dumps(output, ensure_ascii=False))

    def terminal_status(self, env):
        return {"terminal": self.public_observation(env)["terminal"]}

    def validate_postcondition(self, env):
        return self.terminal_status(env)

    def artifact_diff(self, before, after):
        return ArtifactDiff(
            added=tuple(after["artifacts"][ref] for ref in after["artifacts"].keys() - before["artifacts"].keys()),
        )

    def final_audit(self, manifest_path, episode_folder):
        pack = Path(manifest_path).parent
        manifest = load_manifest(manifest_path)
        path = pack / "private/answer.json"
        artifact = next(item for item in manifest.private_files if item.path == "private/answer.json")
        if not path.resolve().is_relative_to(pack.resolve()) or sha256_file(path) != artifact.sha256:
            raise ValueError("Trusted answer failed integrity check")
        gold = read_json(path)
        state = read_json(episode_folder / "workspace/state.json")
        ref = state["current"].get("answer")
        fresh = bool(ref and self._fresh(state, ref))
        passed = fresh and score_answer(state["history"][ref]["data"]["answer"], gold)
        return {"passed": passed, "hard_pass": passed, "candidate_artifact_id": ref, "candidate_fresh": fresh}
