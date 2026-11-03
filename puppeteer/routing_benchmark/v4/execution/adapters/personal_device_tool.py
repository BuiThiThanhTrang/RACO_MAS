from __future__ import annotations

from typing import Any, Mapping

from .common import ConcreteWorkflowAdapter, read_json, write_json


class PersonalDeviceToolAdapter(ConcreteWorkflowAdapter):
    adapter_id = "personal_device_tool.v1"

    def compute_candidate(self, env, arguments: Mapping[str, Any]) -> Mapping[str, Any]:
        request = read_json(env.workspace_dir / "public/request.json")
        return {"operation": "set_setting", "setting": request["setting"], "value": request["value"]}

    def validate_candidate(self, env, candidate: Mapping[str, Any]) -> Mapping[str, Any]:
        request = read_json(env.workspace_dir / "public/request.json")
        valid = candidate.get("operation") == "set_setting" and candidate.get("setting") == request["setting"] and candidate.get("value") == request["value"]
        return {"valid": valid, "issues": [] if valid else ["operation_or_postcondition_mismatch"]}

    def apply_candidate(self, env, candidate: Mapping[str, Any]) -> Mapping[str, Any]:
        runtime_path = env.workspace_dir / "fixtures/device_runtime.json"
        runtime = read_json(runtime_path)
        runtime["state"]["settings"][candidate["setting"]] = candidate["value"]
        runtime["state"]["revision"] += 1
        runtime["operation_log"].append(dict(candidate))
        write_json(runtime_path, runtime)
        return {"applied": True, "setting": candidate["setting"], "effective_value": runtime["state"]["settings"][candidate["setting"]], "revision": runtime["state"]["revision"], "postcondition_observed": True}
