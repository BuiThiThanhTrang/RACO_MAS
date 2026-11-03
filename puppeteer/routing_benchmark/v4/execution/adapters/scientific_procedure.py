from __future__ import annotations

from typing import Any, Mapping

from routing_benchmark.v4.artifacts.logic import scientific_result

from .common import ConcreteWorkflowAdapter, read_csv, read_json


class ScientificProcedureAdapter(ConcreteWorkflowAdapter):
    adapter_id = "scientific_procedure.v1"

    def compute_candidate(self, env, arguments: Mapping[str, Any]) -> Mapping[str, Any]:
        rows = read_csv(env.workspace_dir / "public/measurements.csv")
        protocol = read_json(env.workspace_dir / "public/protocol.json")
        if arguments.get("operation"):
            protocol = {**protocol, "operation": str(arguments["operation"])}
        return scientific_result(rows, protocol)

    def validate_candidate(self, env, candidate: Mapping[str, Any]) -> Mapping[str, Any]:
        expected = self.compute_candidate(env, {})
        valid = dict(candidate) == dict(expected)
        return {"valid": valid, "issues": [] if valid else ["method_unit_or_constraint_mismatch"]}

    def apply_candidate(self, env, candidate: Mapping[str, Any]) -> Mapping[str, Any]:
        return {"applied": True, "supported_conclusion": f"{candidate['operation']} result is {candidate['result']} {candidate['unit']}.", "constraint_checked": True, "postcondition_observed": True}
