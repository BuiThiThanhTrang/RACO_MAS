from __future__ import annotations

from typing import Any, Mapping

from .common import ConcreteWorkflowAdapter, read_json, write_json


class CrossAppWorkflowAdapter(ConcreteWorkflowAdapter):
    adapter_id = "cross_app_workflow.v1"

    def compute_candidate(self, env, arguments: Mapping[str, Any]) -> Mapping[str, Any]:
        source = read_json(env.workspace_dir / "public/source_record.json")
        contract = read_json(env.workspace_dir / "public/destination_contract.json")
        return {contract["mapping"][key]: value for key, value in source.items() if key in contract["mapping"]}

    def validate_candidate(self, env, candidate: Mapping[str, Any]) -> Mapping[str, Any]:
        contract = read_json(env.workspace_dir / "public/destination_contract.json")
        required = set(contract["destination_fields"])
        missing = sorted(required - set(candidate))
        expected = self.compute_candidate(env, {})
        valid = not missing and dict(candidate) == dict(expected)
        return {"valid": valid, "issues": ([f"missing_fields:{','.join(missing)}"] if missing else []) + ([] if dict(candidate) == dict(expected) else ["field_mapping_mismatch"])}

    def apply_candidate(self, env, candidate: Mapping[str, Any]) -> Mapping[str, Any]:
        store_path = env.workspace_dir / "fixtures/destination_store.json"
        store = read_json(store_path)
        key = candidate["external_id"]
        records = [row for row in store["records"] if row.get("external_id") != key]
        records.append(dict(candidate))
        store["records"] = records
        store["audit_log"].append({"external_id": key, "operation": "upsert"})
        write_json(store_path, store)
        return {"applied": True, "external_id": key, "record_count": len(records), "postcondition_observed": True}
