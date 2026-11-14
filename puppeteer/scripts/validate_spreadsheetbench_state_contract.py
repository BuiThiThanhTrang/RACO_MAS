"""Validate state sufficiency and role/tool conformance for the SpreadsheetBench release."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import tempfile

from openpyxl import load_workbook

from routing_benchmark.v4.artifacts.manifest import load_manifest, validate_artifact_pack
from routing_benchmark.v4.execution.registry import adapter_for_manifest


REQUIRED_OBSERVATION_KEYS = {
    "public_task",
    "candidate_contract",
    "public_case_summary",
    "versions",
    "artifacts",
    "latest_execution",
    "rejected_attempt_count",
    "terminal",
}

REQUIRED_EXECUTOR_KEYS = {
    "task",
    "execution_contract",
    "validation_contract",
    "public_case_1",
    "runtime_state",
    "recent_tool_events",
    "candidate_authoring_constraints",
}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("release_root", type=Path)
    args = parser.parse_args()
    release = args.release_root.resolve()
    packets_root = release / "packs" / "structured_data_processing"
    pool = json.loads((release / "agent_pools" / "structured_data_processing.json").read_text(encoding="utf-8"))
    routing = json.loads((release / "routing_contract.json").read_text(encoding="utf-8"))
    role_cards = pool["role_cards"]
    checks: list[dict[str, object]] = []
    range_probe_passed = False

    scratch_parent = Path(__file__).resolve().parents[1] / ".cache" / "state_validation"
    scratch_parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="spreadsheetbench-state-", dir=scratch_parent) as temporary:
        scratch = Path(temporary)
        for packet in sorted(path for path in packets_root.iterdir() if path.is_dir()):
            manifest = load_manifest(packet)
            adapter = adapter_for_manifest(manifest)
            env = adapter.materialize(packet / "manifest.json", scratch / packet.name)
            materialized = [
                str(path.relative_to(env.workspace_dir)).replace("\\", "/")
                for path in env.workspace_dir.rglob("*")
                if path.is_file()
            ]
            observation = adapter.public_observation(env)
            context = adapter.executor_context(env, role_cards[0]["agent_id"], role_cards[0])
            context_json = json.dumps(context, ensure_ascii=False)
            spec = json.loads((packet / "public" / "packet_spec.json").read_text(encoding="utf-8"))
            packet_roles = {item["agent_id"]: item for item in spec["role_contracts"]}
            role_tool_match = all(
                [tool.name for tool in adapter.tool_spec(env, role["agent_id"], role)]
                == [tool["name"] for tool in packet_roles[role["agent_id"]]["typed_tools"]]
                for role in role_cards
            )

            if not range_probe_passed:
                public_input = packet / "public" / "cases" / "case_1" / "input.xlsx"
                workbook = load_workbook(public_input, read_only=False, data_only=False)
                try:
                    probe = next((sheet for sheet in workbook.worksheets if sheet.max_row > 200), None)
                    if probe is not None:
                        result = adapter.execute_tool(
                            env,
                            "inspect_workbook",
                            {"sheet": probe.title, "cell_range": "A201:A205", "include_formulas": True, "include_styles": False},
                        )
                        snapshot = result.output["sheets"][0]
                        range_probe_passed = result.status == "success" and snapshot["start_row"] == 201 and snapshot["rows_shown"] == 5
                finally:
                    workbook.close()

            packet_checks = {
                "packet_id": packet.name,
                "manifest_valid": validate_artifact_pack(packet)["valid"],
                "exact_adapter": manifest.adapter_id == "spreadsheetbench.python_oj.v1",
                "public_only_materialization": not any(path.startswith(("private/", "fixtures/")) for path in materialized),
                "observation_complete": REQUIRED_OBSERVATION_KEYS.issubset(observation),
                "executor_context_complete": REQUIRED_EXECUTOR_KEYS.issubset(context),
                "executor_context_json_serializable": bool(context_json),
                "role_tool_contract_match": role_tool_match,
                "private_assets_absent_from_observation": "private/cases/" not in json.dumps(observation, ensure_ascii=False),
                "selection_archetype_absent_from_observation": "selection_archetype" not in json.dumps(observation, ensure_ascii=False),
            }
            packet_checks["passed"] = all(value is True for key, value in packet_checks.items() if key not in {"packet_id", "passed"})
            checks.append(packet_checks)

    release_checks = {
        "packet_count": len(checks),
        "full_pool_plus_stop": len(role_cards) == 10 and pool["router_actions"] == [item["agent_id"] for item in role_cards] + ["STOP"],
        "guard_disabled": pool["guard_enabled"] is False and routing["guard_enabled"] is False,
        "workflow_graph_absent": routing["workflow_graph_provided"] is False and routing["valid_action_mask_provided"] is False,
        "range_reader_beyond_default_preview": range_probe_passed,
    }
    passed = len(checks) == 20 and all(item["passed"] for item in checks) and all(
        value is True for key, value in release_checks.items() if key != "packet_count"
    )
    print(json.dumps({"passed": passed, "release_checks": release_checks, "packets": checks}, indent=2, ensure_ascii=False))
    raise SystemExit(0 if passed else 1)


if __name__ == "__main__":
    main()
