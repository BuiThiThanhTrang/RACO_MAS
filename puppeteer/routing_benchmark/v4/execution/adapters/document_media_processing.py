from __future__ import annotations

from typing import Any, Mapping

from .common import ConcreteWorkflowAdapter, read_json


class DocumentMediaProcessingAdapter(ConcreteWorkflowAdapter):
    adapter_id = "document_media_processing.v1"

    def compute_candidate(self, env, arguments: Mapping[str, Any]) -> Mapping[str, Any]:
        contract = read_json(env.workspace_dir / "public/extraction_contract.json")
        # The fixture is the deterministic extraction backend's output, not router input.
        media_index = read_json(env.workspace_dir / "fixtures/media_index.json")
        fields = dict(media_index["machine_readable_data"])
        requested = set(arguments.get("fields") or contract["required_fields"])
        return {"fields": {key: fields[key] for key in contract["required_fields"] if key in requested and key in fields}, "source": contract["source"], "extraction_backend": "deterministic_media_index_v1"}

    def validate_candidate(self, env, candidate: Mapping[str, Any]) -> Mapping[str, Any]:
        contract = read_json(env.workspace_dir / "public/extraction_contract.json")
        fields = dict(candidate.get("fields") or {})
        missing = sorted(set(contract["required_fields"]) - set(fields))
        reference = read_json(env.pack_dir / "private/reference.json")["expected_fields"]
        content_ok = fields == reference
        total_ok = True
        if contract.get("total_rule") == "subtotal + tax" and not missing:
            total_ok = abs(float(fields["subtotal"]) + float(fields["tax"]) - float(fields["total"])) < 1e-8
        valid = not missing and total_ok and content_ok
        issues = (
            ([f"missing_fields:{','.join(missing)}"] if missing else [])
            + ([] if total_ok else ["total_rule_failed"])
            + ([] if content_ok else ["extracted_content_mismatch"])
        )
        return {"valid": valid, "issues": issues, "total_rule_passed": total_ok}

    def apply_candidate(self, env, candidate: Mapping[str, Any]) -> Mapping[str, Any]:
        return {"applied": True, "structured_record": dict(candidate.get("fields") or {}), "source": candidate.get("source"), "postcondition_observed": True}
