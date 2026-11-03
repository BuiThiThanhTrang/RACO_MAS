from __future__ import annotations

from typing import Any, Mapping

from .common import ConcreteWorkflowAdapter, read_json, write_json


class TransactionalServiceAdapter(ConcreteWorkflowAdapter):
    adapter_id = "transactional_service.v1"

    def compute_candidate(self, env, arguments: Mapping[str, Any]) -> Mapping[str, Any]:
        record = read_json(env.workspace_dir / "public/account_record.json")
        request = read_json(env.workspace_dir / "public/request.json")
        policy = read_json(env.workspace_dir / "public/policy.json")
        eligible = (
            (not policy.get("active_account_required") or record.get("status") == "active")
            and float(record.get("balance_due", 0)) <= float(policy["maximum_balance_due"])
            and request.get("operation") in policy["allowed_operations"]
        )
        return {
            "account_id": record["account_id"],
            "request_id": request["request_id"],
            "operation": request["operation"],
            "changes": request["changes"],
            "effective_date": request["effective_date"],
            "eligible": eligible,
        }

    def validate_candidate(self, env, candidate: Mapping[str, Any]) -> Mapping[str, Any]:
        expected = self.compute_candidate(env, {})
        valid = dict(candidate) == dict(expected)
        return {"valid": valid, "issues": [] if valid else ["eligibility_or_mutation_mismatch"]}

    def apply_candidate(self, env, candidate: Mapping[str, Any]) -> Mapping[str, Any]:
        if not candidate.get("eligible"):
            return {"applied": False, "reason": "request_not_eligible", "candidate": dict(candidate)}
        store_path = env.workspace_dir / "fixtures/service_store.json"
        store = read_json(store_path)
        store["account"].setdefault("applied_changes", {}).update(candidate["changes"])
        store["ledger"].append({"request_id": candidate["request_id"], "operation": candidate["operation"], "changes": candidate["changes"]})
        write_json(store_path, store)
        return {"applied": True, "recorded_operation": candidate["operation"], "request_id": candidate["request_id"], "postcondition_observed": True}
