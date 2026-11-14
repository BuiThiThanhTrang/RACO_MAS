"""Run the three-source expansion with a fixed policy and the shared workbook runner."""
from __future__ import annotations

import argparse
from pathlib import Path
import os

from dotenv import load_dotenv
from openai import OpenAI
from openpyxl import load_workbook
from openpyxl.utils.cell import range_boundaries
import yaml

from build_structured_data_expansion import DOMAIN, OUTPUT, PROJECT
from run_shared_environment import resolve, source_hashes
from routing_benchmark.routers import RoutingDecision
from routing_benchmark.v4.artifacts.manifest import load_manifest
from routing_benchmark.v4.artifacts.schema import sha256_file
from routing_benchmark.v4.execution.adapters.common import read_json, write_json
from routing_benchmark.v4.execution.adapters.spreadsheetbench_python_oj import (
    SpreadsheetBenchPythonOJAdapter,
    _values_equal,
)
from routing_benchmark.v4.execution.final_evaluation import _spreadsheetbench_audit
from routing_benchmark.v4.execution.runtime import resolve_runtime
from routing_benchmark.v4.execution.shared_actor import SharedRoleActor
from routing_benchmark.v4.execution.shared_runner import SharedTrajectoryRunner


class TargetCellAdapter(SpreadsheetBenchPythonOJAdapter):
    """Use one sealed workbook comparison for all three upstream sources."""

    def __init__(self, pack):
        super().__init__()
        self.pack = pack

    def validate_output(self, input_path, output, contract):
        if contract.get("kind") != "workbook_target_cells":
            raise ValueError("Expected the shared target-cell contract")
        case = input_path.parent.name.removeprefix("case_")
        target = read_json(self.pack / "private/evaluation_target.json")["cases"][case]
        answer_path = self.pack / "private/cases" / f"case_{case}" / "answer.xlsx"
        sheet = target["sheet"]
        issues = []
        actual = load_workbook(output, read_only=False, data_only=False)
        expected = cached = None
        try:
            if sheet not in actual.sheetnames:
                issues.append(f"worksheet_not_found:{sheet}")
            else:
                expected = load_workbook(answer_path, read_only=False, data_only=False)
                if not target["require_formula"]:
                    cached = load_workbook(answer_path, read_only=False, data_only=True)
                c1, r1, c2, r2 = range_boundaries(target["range"])
                for row in actual[sheet].iter_rows(min_row=r1, max_row=r2, min_col=c1, max_col=c2):
                    for cell in row:
                        reference = expected[sheet][cell.coordinate]
                        gold = reference.value
                        if reference.data_type == "f" and type(cell.value) in (int, float) and cached:
                            value = cached[sheet][cell.coordinate].value
                            if type(value) in (int, float):
                                gold = value
                        equal = (
                            type(cell.value) is type(gold) and cell.value == gold
                            if isinstance(cell.value, bool) or isinstance(gold, bool)
                            else _values_equal(cell.value, gold)
                        )
                        if not equal:
                            issues.append(f"target_cell_mismatch:{sheet}!{cell.coordinate}")
        finally:
            actual.close()
            if expected:
                expected.close()
            if cached:
                cached.close()
        return {"passed": not issues, "issues": issues[:50], "kind": contract["kind"]}


class FixedWorkflow:
    name = "structured_data_fixed_workflow"

    def reset(self, seed=None):
        self.previous = None

    def select(self, observation, candidate_actions, oracle_valid_actions=None):
        state = observation["current_state"]
        artifacts = state["artifacts"]
        candidate = artifacts.get("candidate_solution")
        execution = artifacts.get("public_execution")
        validation = artifacts.get("public_validation")
        if state["terminal"]:
            action = "STOP"
        elif self.previous is None:
            action = "Workbook Surveyor"
        elif candidate is None:
            action = "Table Reshaping Specialist"
        elif (
            not execution or not execution["fresh"]
            or not any(ref["artifact_id"] == candidate["artifact_id"] for ref in execution["input_refs"])
        ):
            action = "Workbook Test Executor"
        elif state["latest_execution"]["status"] != "success":
            action = "Workbook Repair Specialist"
        elif (
            validation and validation["fresh"]
            and validation["input_versions"]["candidate_hash"] == candidate["sha256"]
            and any(ref["artifact_id"] == execution["artifact_id"] for ref in validation["input_refs"])
        ):
            action = "Workbook Repair Specialist"
        else:
            action = "Workbook Integrity Validator"
        self.previous = action
        return RoutingDecision(action)


def pilot_hashes(packet_root):
    digest = source_hashes(packet_root)
    for path in (Path(__file__).resolve(), PROJECT / "scripts/build_structured_data_expansion.py"):
        digest[str(path.relative_to(PROJECT))] = sha256_file(path)
    return digest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--packet-root", type=Path, default=OUTPUT)
    parser.add_argument("--config", type=Path, default=PROJECT / "config/shared_environment/structured_data_v1_qwen9b_direct_jev.yaml")
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--packet", action="append", default=[])
    args = parser.parse_args()
    config = yaml.safe_load(args.config.read_text(encoding="utf-8-sig"))
    if config["domain"] != DOMAIN:
        parser.error("Config must use structured_data_processing")
    packet_root = args.packet_root.resolve()
    config.update(packet_root=str(packet_root), router={"type": FixedWorkflow.name}, selected_packets=args.packet)
    role_path = packet_root / "agent_pools" / f"{DOMAIN}.json"
    config["role_cards"] = str(role_path)
    roles = read_json(role_path)
    manifests = sorted((packet_root / "packs" / DOMAIN).glob("*/manifest.json"))
    if args.packet:
        manifests = [path for path in manifests if path.parent.name in args.packet]
        if {path.parent.name for path in manifests} != set(args.packet):
            parser.error("Unknown packet selection")
    if not manifests:
        parser.error("No packet manifests found")
    load_dotenv(PROJECT.parent / ".env")
    load_dotenv(PROJECT / ".env")
    executor = config["executor"]
    if executor["provider"] != "openrouter":
        parser.error("Use the OpenRouter Qwen executor config")
    key = os.environ[executor["api_key_env"]]
    run = resolve(config["output_root"]) / args.run_id
    run.mkdir(parents=True, exist_ok=False)
    digest = pilot_hashes(packet_root)
    write_json(run / "resolved_config.json", config)
    write_json(run / "source_hashes.json", digest)
    write_json(run / "role_cards.json", roles)
    episodes = []
    with OpenAI(api_key=key, base_url="https://openrouter.ai/api/v1",
                timeout=executor["timeout_seconds"], max_retries=0) as client:
        actor = SharedRoleActor(client, executor["model"], max_tokens=executor["max_tokens"],
                               transport_retries=executor["transport_retries"], max_tool_calls=executor["max_tool_calls"])
        for index, manifest_path in enumerate(manifests):
            manifest = load_manifest(manifest_path)
            folder = run / "episodes" / manifest.base_workflow_id
            runtime = resolve_runtime(config, manifest_path, PROJECT)
            adapter = TargetCellAdapter(manifest_path.parent)
            adapter.configure_runtime(runtime)
            write_json(folder / "runtime.json", runtime)
            runner = SharedTrajectoryRunner(router=FixedWorkflow(), actor=actor, max_calls=config["max_calls"])
            result = runner.run(manifest_path=manifest_path, adapter=adapter, workspace=folder / "workspace",
                                output_dir=folder, roles=roles, seed=config["seed"] + index)
            audit_env = adapter.materialize(manifest_path, folder / "evaluation/checker_workspace")
            audit = _spreadsheetbench_audit(adapter, audit_env, manifest_path, folder)
            write_json(folder / "evaluation/final_task_audit.json", audit)
            episodes.append({**result, "trusted_hard_pass": audit.get("hard_pass", audit["passed"])})
            write_json(run / "summary.json", {
                "requested": len(manifests), "completed": len(episodes),
                "passed": sum(item["passed"] for item in episodes),
                "trusted_hard_passed": sum(item["trusted_hard_pass"] is True for item in episodes),
                "episodes": episodes,
            })
    unchanged = pilot_hashes(packet_root) == digest
    write_json(run / "freeze_verification.json", {"unchanged": unchanged, "tracked_files": len(digest)})
    if not unchanged:
        raise RuntimeError("Source freeze mismatch")


if __name__ == "__main__":
    main()
