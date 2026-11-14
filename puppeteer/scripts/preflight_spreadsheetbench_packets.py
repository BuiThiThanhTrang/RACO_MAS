"""Run deterministic public/private readiness witnesses for migrated packets."""
from __future__ import annotations

import argparse
import inspect
import json
from pathlib import Path
import tempfile

from routing_benchmark.v4.artifacts.manifest import load_manifest, validate_artifact_pack
from routing_benchmark.v4.execution.final_evaluation import _spreadsheetbench_audit
from routing_benchmark.v4.execution.registry import adapter_for_manifest
from scripts.verify_spreadsheetbench_reference_cases import PREFLIGHT_SOLVERS, SOLVERS, number


IDENTITY_SOURCE = """from openpyxl import load_workbook

def solve(input_path, output_path):
    workbook = load_workbook(input_path)
    workbook.save(output_path)
"""


FORBIDDEN_SOURCE = """import os

def solve(input_path, output_path):
    os.listdir('.')
"""


PUBLIC_SHORTCUTS = {
    "227-40": """from openpyxl import load_workbook

def solve(input_path, output_path):
    workbook = load_workbook(input_path)
    sheet = workbook["Sheet1"]
    for source_row in range(9, 13):
        for column in range(1, 6):
            sheet.cell(source_row + 5, column).value = sheet.cell(source_row, column).value
    workbook.save(output_path)
""",
    "493-5": """from openpyxl import load_workbook

def solve(input_path, output_path):
    workbook = load_workbook(input_path)
    sheet = workbook["Imported Data"]
    sheet.delete_rows(8, 1)
    sheet.delete_rows(7, 1)
    for row in range(2, 9):
        sheet.cell(row, 5).value = f"=+C{row}-D{row}"
    workbook.save(output_path)
""",
    "585-41": """from openpyxl import load_workbook

def solve(input_path, output_path):
    workbook = load_workbook(input_path)
    values = [11604.67, 11604.67, 390.68, 390.68, 0.28135, 0.28135, 16.1921, 16.19]
    for column, value in enumerate(values, 1):
        workbook["Sheet1"].cell(1, column).value = value
    workbook.save(output_path)
""",
    "47827": """from copy import copy
from openpyxl import load_workbook
from openpyxl.styles import PatternFill

def solve(input_path, output_path):
    workbook = load_workbook(input_path)
    sheet = workbook["3 straights to 3 boxed"]
    values = ["014", "000", "015", "024", "033", "024", "015", "006", "015", "015", "033", "123", "033", "015", "015"]
    for row, value in enumerate(values, 2):
        cell = sheet.cell(row, 7)
        cell.value = value
        font = copy(cell.font)
        font.bold = True
        cell.font = font
        cell.fill = PatternFill(fill_type="solid", fgColor="EBF1DE")
        cell.number_format = "000"
    workbook.save(output_path)
""",
}


def reference_source(source_id: str) -> str:
    transform = PREFLIGHT_SOLVERS[source_id]
    helper_source = inspect.getsource(SOLVERS[source_id]) + "\n" if transform is not SOLVERS[source_id] and source_id == "13-1" else ""
    return (
        "import openpyxl\n"
        "import re\n"
        "from openpyxl import load_workbook\n"
        "from openpyxl.styles import Alignment, Border, Font, PatternFill, Side\n\n"
        + inspect.getsource(number)
        + "\n"
        + helper_source
        + inspect.getsource(transform)
        + "\n\ndef solve(input_path, output_path):\n"
        "    workbook = load_workbook(input_path)\n"
        f"    {transform.__name__}(workbook)\n"
        "    workbook.save(output_path)\n"
    )


def run_packet(packet: Path, python_executable: str | None, scratch: Path) -> dict:
    manifest_path = packet / "manifest.json"
    manifest = load_manifest(manifest_path)
    source_id = str(manifest.metadata["source_id"])
    adapter = adapter_for_manifest(manifest)
    adapter.configure_runtime({"python_executable": python_executable} if python_executable else {})
    episode_folder = scratch / packet.name / "episode"
    env = adapter.materialize(manifest_path, episode_folder / "workspace")
    materialized = [str(path.relative_to(env.workspace_dir)).replace("\\", "/") for path in env.workspace_dir.rglob("*") if path.is_file()]

    written = adapter.execute_tool(env, "write_solution", {"python_source": reference_source(source_id), "input_artifact_refs": []})
    if written.status != "success":
        return {"packet_id": packet.name, "passed": False, "stage": "reference_candidate_acceptance", "evidence": written.output}
    candidate_id = written.output["artifact"]["artifact_id"]
    executed = adapter.execute_tool(env, "run_solution_public", {"candidate_artifact_id": candidate_id})
    execution_id = executed.output["execution_artifact"]["artifact_id"]
    validated = adapter.execute_tool(env, "validate_public_output", {"candidate_artifact_id": candidate_id, "execution_artifact_id": execution_id}) if executed.status == "success" else None
    checker_env = adapter.materialize(manifest_path, scratch / packet.name / "checker")
    final = _spreadsheetbench_audit(adapter, checker_env, manifest_path, episode_folder)

    negative_env = adapter.materialize(manifest_path, scratch / packet.name / "negative")
    identity = adapter.execute_tool(negative_env, "write_solution", {"python_source": IDENTITY_SOURCE, "input_artifact_refs": []})
    negative_run = adapter.execute_tool(negative_env, "run_solution_public", {"candidate_artifact_id": identity.output["artifact"]["artifact_id"]})
    negative_validation = adapter.validate_output(
        negative_env.workspace_dir / "public/cases/case_1/input.xlsx",
        negative_env.workspace_dir / negative_run.output["output_artifact"]["path"],
        json.loads((negative_env.workspace_dir / "public/validation_contract.json").read_text(encoding="utf-8")),
    )
    rejected = adapter.execute_tool(negative_env, "write_solution", {"python_source": FORBIDDEN_SOURCE, "input_artifact_refs": []})
    negative_after_rejection = adapter.public_observation(negative_env)

    stale_env = adapter.materialize(manifest_path, scratch / packet.name / "stale")
    stale_first = adapter.execute_tool(stale_env, "write_solution", {"python_source": IDENTITY_SOURCE, "input_artifact_refs": []})
    stale_first_id = stale_first.output["artifact"]["artifact_id"]
    stale_run = adapter.execute_tool(stale_env, "run_solution_public", {"candidate_artifact_id": stale_first_id})
    stale_second = adapter.execute_tool(stale_env, "write_solution", {"python_source": IDENTITY_SOURCE + "\n# accepted revision\n", "input_artifact_refs": []})
    stale_observation = adapter.public_observation(stale_env)

    shortcut_evidence = None
    if source_id in PUBLIC_SHORTCUTS:
        shortcut_folder = scratch / packet.name / "shortcut_episode"
        shortcut_env = adapter.materialize(manifest_path, shortcut_folder / "workspace")
        shortcut_written = adapter.execute_tool(
            shortcut_env,
            "write_solution",
            {"python_source": PUBLIC_SHORTCUTS[source_id], "input_artifact_refs": []},
        )
        shortcut_candidate_id = shortcut_written.output["artifact"]["artifact_id"]
        shortcut_run = adapter.execute_tool(shortcut_env, "run_solution_public", {"candidate_artifact_id": shortcut_candidate_id})
        shortcut_validation = adapter.execute_tool(
            shortcut_env,
            "validate_public_output",
            {
                "candidate_artifact_id": shortcut_candidate_id,
                "execution_artifact_id": shortcut_run.output["execution_artifact"]["artifact_id"],
            },
        )
        shortcut_checker_env = adapter.materialize(manifest_path, scratch / packet.name / "shortcut_checker")
        shortcut_final = _spreadsheetbench_audit(adapter, shortcut_checker_env, manifest_path, shortcut_folder)
        shortcut_evidence = {
            "public_passed": shortcut_validation.status == "success" and shortcut_validation.output["validation"]["passed"],
            "private_rejected": shortcut_final["status"] == "FAIL",
        }

    checks = {
        "manifest_valid": validate_artifact_pack(packet)["valid"],
        "public_only_materialization": not any(path.startswith(("private/", "fixtures/")) for path in materialized),
        "reference_public_execution": executed.status == "success",
        "reference_public_validation": bool(validated and validated.status == "success" and validated.output["validation"]["passed"]),
        "reference_terminal": adapter.terminal_status(env)["terminal"],
        "independent_final_status": final["status"],
        "independent_final_score": final.get("soft_score"),
        "same_candidate_hash_all_cases": final.get("same_candidate_hash_all_cases"),
        "identity_negative_control_rejected": not negative_validation["passed"],
        "forbidden_import_rejected": rejected.status == "invalid_output",
        "rejected_attempt_preserves_candidate": (
            negative_after_rejection["artifacts"]["candidate_solution"]["artifact_id"] == identity.output["artifact"]["artifact_id"]
            and negative_after_rejection["rejected_attempt_count"] >= 1
        ),
        "superseded_execution_becomes_stale": (
            stale_run.status == "success"
            and stale_second.status == "success"
            and stale_observation["artifacts"]["public_execution"]["fresh"] is False
        ),
    }
    if shortcut_evidence is not None:
        checks["public_shortcut_passed"] = shortcut_evidence["public_passed"]
        checks["private_shortcut_rejected"] = shortcut_evidence["private_rejected"]
    passed = all(value is True for key, value in checks.items() if key not in {"independent_final_status", "independent_final_score"}) and final["status"] == "PASS" and final.get("soft_score") == 1.0
    report = {"packet_id": packet.name, "source_id": source_id, "passed": passed, "checks": checks, "final_case_results": [{"case_id": item["case_id"], "status": item["status"]} for item in final.get("case_results", [])]}
    if not passed:
        report["diagnostics"] = {
            "public_execution": executed.output,
            "public_validation": validated.output if validated is not None else None,
            "final_failures": [
                {"case_id": item["case_id"], "execution": item.get("execution"), "comparison": item.get("comparison")}
                for item in final.get("case_results", []) if item.get("status") != "PASS"
            ],
        }
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("packet_root", type=Path)
    parser.add_argument("--packet", action="append", required=True)
    parser.add_argument("--python-executable")
    args = parser.parse_args()
    packets_root = args.packet_root.resolve() / "packs" / "structured_data_processing"
    scratch_parent = Path(__file__).resolve().parents[1] / ".cache" / "preflight"
    scratch_parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="spreadsheetbench-set-", dir=scratch_parent) as temporary:
        reports = [run_packet(packets_root / packet_id, args.python_executable, Path(temporary)) for packet_id in args.packet]
    result = {"passed": all(item["passed"] for item in reports), "packets": reports}
    print(json.dumps(result, indent=2, ensure_ascii=False, default=str))
    raise SystemExit(0 if result["passed"] else 1)


if __name__ == "__main__":
    main()
