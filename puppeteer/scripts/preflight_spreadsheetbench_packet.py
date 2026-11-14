"""Deterministic readiness checks for the first SpreadsheetBench Python-OJ packet."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import tempfile

from routing_benchmark.v4.artifacts.manifest import load_manifest, validate_artifact_pack
from routing_benchmark.v4.execution.adapters.spreadsheetbench_python_oj import compare_answer_region
from routing_benchmark.v4.execution.final_evaluation import _spreadsheetbench_audit
from routing_benchmark.v4.execution.registry import adapter_for_manifest


REFERENCE_SOURCE = '''from openpyxl import load_workbook

def solve(input_path, output_path):
    workbook = load_workbook(input_path)
    sheet = workbook["RNM"]
    first_rows = {}
    totals = {}
    duplicates = []
    last_row = max(row for row in range(2, sheet.max_row + 1) if any(sheet.cell(row, column).value is not None for column in range(1, 12)))
    for row in range(2, last_row + 1):
        key = (sheet.cell(row, 2).value, sheet.cell(row, 3).value)
        value = sheet.cell(row, 10).value or 0
        if key not in first_rows:
            first_rows[key] = row
            totals[key] = value
        else:
            totals[key] += value
            duplicates.append(row)
    for key, row in first_rows.items():
        sheet.cell(row, 10).value = totals[key]
    for row in reversed(duplicates):
        sheet.delete_rows(row, 1)
    workbook.save(output_path)
'''


COPY_EXAMPLE_SOURCE = '''from openpyxl import load_workbook

def solve(input_path, output_path):
    workbook = load_workbook(input_path)
    source = workbook["Desired outcome"]
    target = workbook["RNM"]
    for row in range(1, 21):
        for column in range(1, 11):
            target.cell(row, column).value = source.cell(row, column).value
    workbook.save(output_path)
'''


FORBIDDEN_SOURCE = '''import os

def solve(input_path, output_path):
    os.listdir(".")
'''


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("packet", type=Path)
    parser.add_argument("--python-executable")
    args = parser.parse_args()
    packet = args.packet.resolve()
    manifest_path = packet / "manifest.json"
    checks: dict[str, object] = {"manifest": validate_artifact_pack(packet)}
    manifest = load_manifest(manifest_path)
    adapter = adapter_for_manifest(manifest)
    adapter.configure_runtime({"python_executable": args.python_executable} if args.python_executable else {})
    scratch_parent = Path(__file__).resolve().parents[1] / ".cache" / "preflight"
    scratch_parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="spreadsheetbench-", dir=scratch_parent) as temporary:
        root = Path(temporary)
        episode_folder = root / "episode"
        env = adapter.materialize(manifest_path, episode_folder / "workspace")
        materialized = [str(path.relative_to(env.workspace_dir)).replace("\\", "/") for path in env.workspace_dir.rglob("*") if path.is_file()]
        checks["public_only_materialization"] = not any(path.startswith(("private/", "fixtures/")) for path in materialized)

        write = adapter.execute_tool(env, "write_solution", {"python_source": REFERENCE_SOURCE, "input_artifact_refs": []})
        candidate_id = write.output["artifact"]["artifact_id"]
        run = adapter.execute_tool(env, "run_solution_public", {"candidate_artifact_id": candidate_id})
        execution_id = run.output["execution_artifact"]["artifact_id"]
        validation = adapter.execute_tool(env, "validate_public_output", {"candidate_artifact_id": candidate_id, "execution_artifact_id": execution_id})
        checks["reference_public_pass"] = validation.status == "success" and validation.output["validation"]["passed"]
        checks["reference_terminal"] = adapter.terminal_status(env)["terminal"]

        case_reports = []
        candidate_path = env.workspace_dir / write.output["artifact"]["path"]
        for case_id in (1, 3):
            input_path = packet / ("public/cases/case_1/input.xlsx" if case_id == 1 else "private/cases/case_3/input.xlsx")
            output_path = root / f"reference_case_{case_id}.xlsx"
            execution = adapter._execute_candidate(env, candidate_path, input_path, output_path, 60)
            comparison = compare_answer_region(output_path, packet / f"private/cases/case_{case_id}/answer.xlsx", "RNM", "A1:J20") if execution["status"] == "success" else {"passed": False}
            case_reports.append({"case_id": case_id, "execution": execution["status"], "passed": comparison["passed"], "comparison": comparison})
        checks["reference_private_cases"] = case_reports
        checker_env = adapter.materialize(manifest_path, root / "checker_workspace")
        independent = _spreadsheetbench_audit(adapter, checker_env, manifest_path, episode_folder)
        checks["independent_final_evaluator"] = {
            "status": independent["status"],
            "passed": independent["passed"],
            "soft_score": independent["soft_score"],
            "same_candidate_hash_all_cases": independent["same_candidate_hash_all_cases"],
        }

        shortcut_env = adapter.materialize(manifest_path, root / "shortcut_episode")
        shortcut = adapter.execute_tool(shortcut_env, "write_solution", {"python_source": COPY_EXAMPLE_SOURCE, "input_artifact_refs": []})
        shortcut_path = shortcut_env.workspace_dir / shortcut.output["artifact"]["path"]
        shortcut_reports = []
        for case_id in (1, 3):
            input_path = packet / ("public/cases/case_1/input.xlsx" if case_id == 1 else "private/cases/case_3/input.xlsx")
            output_path = root / f"shortcut_case_{case_id}.xlsx"
            execution = adapter._execute_candidate(shortcut_env, shortcut_path, input_path, output_path, 60)
            comparison = compare_answer_region(output_path, packet / f"private/cases/case_{case_id}/answer.xlsx", "RNM", "A1:J20") if execution["status"] == "success" else {"passed": False}
            shortcut_reports.append({"case_id": case_id, "execution": execution["status"], "passed": comparison["passed"], "comparison": comparison})
        checks["copy_example_negative_control"] = shortcut_reports

        rejected = adapter.execute_tool(shortcut_env, "write_solution", {"python_source": FORBIDDEN_SOURCE, "input_artifact_refs": []})
        checks["forbidden_import_rejected"] = rejected.status == "invalid_output"

    passed = (
        bool(checks["manifest"]["valid"])
        and checks["public_only_materialization"] is True
        and checks["reference_public_pass"] is True
        and checks["reference_terminal"] is True
        and all(item["passed"] for item in checks["reference_private_cases"])
        and checks["independent_final_evaluator"]["passed"] is True
        and checks["independent_final_evaluator"]["same_candidate_hash_all_cases"] is True
        and checks["copy_example_negative_control"][0]["passed"] is True
        and checks["copy_example_negative_control"][1]["passed"] is False
        and checks["forbidden_import_rejected"] is True
    )
    report = {"passed": passed, "checks": checks}
    print(json.dumps(report, indent=2, ensure_ascii=False, default=str))
    raise SystemExit(0 if passed else 1)


if __name__ == "__main__":
    main()
