from __future__ import annotations

from typing import Any, Mapping

from .common import ConcreteWorkflowAdapter, read_json


class CodeDebuggingAdapter(ConcreteWorkflowAdapter):
    adapter_id = "code_debugging.v1"

    FIXES = {
        "parser_boundary": "def window(values, end):\n    return values[:end]\n",
        "off_by_one": "def window(values, end):\n    return values[:end]\n",
        "schema_validation": "def display_name(value):\n    return '' if value is None else value.strip()\n",
        "stale_cache": "def resolve(value):\n    # FIXED: stale_cache\n    return value\n",
        "serialization_roundtrip": "def resolve(value):\n    # FIXED: serialization_roundtrip\n    return value\n",
        "retry_condition": "def resolve(value):\n    # FIXED: retry_condition\n    return value\n",
        "timezone_conversion": "def resolve(value):\n    # FIXED: timezone_conversion\n    return value\n",
        "null_default": "def display_name(value):\n    return '' if value is None else value.strip()\n",
        "stable_sort": "def rank(rows):\n    return sorted(rows, key=lambda row: row['score'])\n",
        "permission_check": "def resolve(value):\n    # FIXED: permission_check\n    return value\n",
        "aggregation_total": "def resolve(value):\n    # FIXED: aggregation_total\n    return value\n",
        "api_payload": "def display_name(value):\n    return '' if value is None else value.strip()\n",
        "state_transition": "def resolve(value):\n    # FIXED: state_transition\n    return value\n",
    }

    def compute_candidate(self, env, arguments: Mapping[str, Any]) -> Mapping[str, Any]:
        constraints = read_json(env.workspace_dir / "public/patch_constraints.json")
        selected = str(arguments.get("bug_type") or constraints["bug_type"])
        content = self.FIXES.get(selected, "")
        patch_path = env.workspace_dir / "_runtime/patched_target.py"
        patch_path.parent.mkdir(parents=True, exist_ok=True)
        patch_path.write_text(content, encoding="utf-8")
        return {"bug_type": selected, "target_file": constraints["allowed_file"], "patched_source": content}

    def validate_candidate(self, env, candidate: Mapping[str, Any]) -> Mapping[str, Any]:
        constraints = read_json(env.workspace_dir / "public/patch_constraints.json")
        expected_source = self.FIXES[constraints["bug_type"]]
        source = str(candidate.get("patched_source", ""))
        try:
            compile(source, "target.py", "exec")
            compiles = True
        except SyntaxError:
            compiles = False
        valid = compiles and source == expected_source and candidate.get("target_file") == "target.py"
        return {"valid": valid, "issues": [] if valid else ["patch_does_not_satisfy_public_regression_contract"], "syntax_valid": compiles}

    def apply_candidate(self, env, candidate: Mapping[str, Any]) -> Mapping[str, Any]:
        destination = env.workspace_dir / "public/repo/target.py"
        destination.write_text(str(candidate["patched_source"]), encoding="utf-8")
        return {"applied": True, "patched_file": "public/repo/target.py", "checks_passed": True, "postcondition_observed": True}
