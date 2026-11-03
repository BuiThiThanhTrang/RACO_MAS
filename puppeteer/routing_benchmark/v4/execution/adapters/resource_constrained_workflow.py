from __future__ import annotations

from typing import Any, Mapping

from .common import ConcreteWorkflowAdapter, read_json, write_json


class ResourceConstrainedWorkflowAdapter(ConcreteWorkflowAdapter):
    adapter_id = "resource_constrained_workflow.v1"

    def compute_candidate(self, env, arguments: Mapping[str, Any]) -> Mapping[str, Any]:
        graph = read_json(env.workspace_dir / "public/tool_graph.json")
        contract = read_json(env.workspace_dir / "public/task_contract.json")
        requested = arguments.get("route_id")
        if requested:
            route = next((item for item in graph["routes"] if item["route_id"] == requested), None)
        else:
            feasible = [item for item in graph["routes"] if item["cost"] <= contract["budget"] and item["meets_postcondition"]]
            route = min(feasible, key=lambda item: item["cost"]) if feasible else None
        return {"selected_route": route, "budget": contract["budget"]}

    def validate_candidate(self, env, candidate: Mapping[str, Any]) -> Mapping[str, Any]:
        route = candidate.get("selected_route")
        budget = int(candidate.get("budget", 0))
        valid = bool(route and route["cost"] <= budget and route["meets_postcondition"])
        return {"valid": valid, "issues": [] if valid else ["route_exceeds_budget_or_misses_postcondition"]}

    def apply_candidate(self, env, candidate: Mapping[str, Any]) -> Mapping[str, Any]:
        route = candidate["selected_route"]
        ledger_path = env.workspace_dir / "fixtures/budget_ledger.json"
        ledger = read_json(ledger_path)
        if route["cost"] > ledger["remaining"]:
            return {"applied": False, "reason": "budget_exhausted", "remaining": ledger["remaining"]}
        ledger["remaining"] -= route["cost"]
        ledger["events"].append({"route_id": route["route_id"], "cost": route["cost"]})
        write_json(ledger_path, ledger)
        return {"applied": True, "route_id": route["route_id"], "cost": route["cost"], "remaining_budget": ledger["remaining"], "postcondition_observed": True}
