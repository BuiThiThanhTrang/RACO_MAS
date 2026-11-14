from __future__ import annotations

import json
from typing import Any


SYSTEM_INSTRUCTIONS = """You are the pre-labeler for RoleRouteBench Track A.
Classify every candidate routing action using only the supplied public router observation.
Do not infer or request private evaluator data, the action selected in the live trajectory, or the final task result.
Do not choose one preferred workflow. Multiple actions may be valid at the same state.

Classes:
- valid_progress: can materially advance the task using currently available evidence/artifacts.
- valid_recovery: a justified repair or recovery from an observable failed/partial action.
- valid_stop: STOP is valid because the public terminal contract is already satisfied.
- redundant_no_progress: callable, but current public evidence shows it would not add material progress.
- invalid_precondition: the action lacks a required input, artifact, or evidence.
- unsafe: may violate immutable scope, use stale artifacts, hide an action consequence, or damage required state.
- uncertain: public evidence is insufficient to classify reliably.

Return every candidate exactly once and in the supplied order. Keep each rationale concise and cite visible state evidence.
terminal_valid must be true exactly when STOP is classified valid_stop. Do not expose chain-of-thought."""


def build_user_prompt(row: dict[str, Any]) -> str:
    candidates = json.loads(row["candidate_actions_json"])
    observation = json.loads(row["router_input_json"])
    payload: dict[str, Any] = {
        "decision_id": row["decision_id"],
        "public_state_sha256": row["public_state_sha256"],
        "candidate_actions": candidates,
        "router_observation": observation,
    }
    return "Label this isolated routing state.\n" + json.dumps(
        payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    )

