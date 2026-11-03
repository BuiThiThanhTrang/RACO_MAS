from __future__ import annotations

from typing import Any, Mapping

from routing_benchmark.v4.artifacts.logic import structured_transform

from .common import ConcreteWorkflowAdapter, read_json, write_json


class StructuredDataProcessingAdapter(ConcreteWorkflowAdapter):
    adapter_id = "structured_data_processing.v1"

    def compute_candidate(self, env, arguments: Mapping[str, Any]) -> Mapping[str, Any]:
        rows = read_json(env.workspace_dir / "public/source.json")
        contract = read_json(env.workspace_dir / "public/transformation_contract.json")
        operation = str(arguments.get("operation") or contract["operation"])
        transformed = structured_transform(rows, operation)
        write_json(env.workspace_dir / "_runtime/transformed.json", transformed)
        return {"operation": operation, "rows": transformed, "row_count": len(transformed)}

    def validate_candidate(self, env, candidate: Mapping[str, Any]) -> Mapping[str, Any]:
        reference = read_json(env.pack_dir / "private/reference.json")["expected_rows"]
        normalized = list(candidate.get("rows", ()))
        valid = normalized == reference
        return {"valid": valid, "issues": [] if valid else ["transformation_or_schema_mismatch"], "row_count": len(normalized)}

    def apply_candidate(self, env, candidate: Mapping[str, Any]) -> Mapping[str, Any]:
        rows = list(candidate.get("rows") or ())
        if not rows:
            return {"applied": False, "reason": "empty_transformation"}
        output = env.workspace_dir / "_runtime/published.json"
        write_json(output, rows)
        return {"applied": True, "published_path": "_runtime/published.json", "row_count": len(rows), "postcondition_observed": True}
