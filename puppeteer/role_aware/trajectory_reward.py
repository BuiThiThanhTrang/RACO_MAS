from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable


def _clean_text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value.strip()
    if isinstance(value, (int, float, bool)):
        return str(value)
    try:
        return json.dumps(value, ensure_ascii=False, sort_keys=True, default=str).strip()
    except (TypeError, ValueError):
        return str(value).strip()


def _clip(text: str, limit: int) -> str:
    if limit <= 0 or len(text) <= limit:
        return text
    marker = "\n...[truncated]...\n"
    if limit <= len(marker):
        return text[:limit]
    head = max(1, (limit - len(marker)) // 3)
    tail = limit - len(marker) - head
    return text[:head] + marker + text[-tail:]


def _safe_artifact_text(candidate_output: Any, artifact_root: str | Path | None) -> str:
    """Read generated artifacts only when they are inside the path workspace."""
    candidate = _clean_text(candidate_output)
    if not candidate or artifact_root is None:
        return candidate
    try:
        root = Path(artifact_root).resolve()
        candidate_path = Path(candidate)
        target = (
            candidate_path.resolve()
            if candidate_path.is_absolute()
            else (root / candidate_path).resolve()
        )
        target.relative_to(root)
        if target.is_file():
            return target.read_text(encoding="utf-8", errors="replace")
    except (OSError, ValueError):
        pass
    return candidate


@dataclass(frozen=True)
class TrajectoryRewardInput:
    """Model-identity-free, gold-free input for a trajectory reward model."""

    task_text: str
    trajectory_text: str
    candidate_output: str

    @property
    def messages(self) -> list[dict[str, str]]:
        return [
            {"role": "user", "content": f"Task:\n{self.task_text}"},
            {
                "role": "assistant",
                "content": (
                    f"Trajectory:\n{self.trajectory_text}\n\n"
                    f"Final candidate output:\n{self.candidate_output}"
                ),
            },
        ]

    @property
    def digest(self) -> str:
        payload = json.dumps(
            self.messages, ensure_ascii=False, sort_keys=True, separators=(",", ":")
        )
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _step_text(action: Any, index: int, max_step_chars: int) -> str:
    action_data = getattr(action, "action", {}) or {}
    result_data = getattr(action, "result", {}) or {}

    # Deliberately whitelist public fields. In particular, never serialize
    # agent_model, teammate_id, provider_profile, tokens, or cost.
    action_name = _clean_text(action_data.get("action"))
    parameter = _clean_text(
        action_data.get("parameter", action_data.get("parameters", ""))
    )
    role_name = _clean_text(getattr(action, "agent_role", ""))
    success = _clean_text(getattr(action, "success", ""))

    result_parts = []
    for key in ("step_data", "answer", "code", "text"):
        value = _clean_text(result_data.get(key))
        if value and value not in result_parts:
            result_parts.append(value)
    result = _clip("\n".join(result_parts), max_step_chars)

    fields = [
        f"Step {index}",
        f"Role: {role_name or 'Unknown'}",
        f"Action: {action_name or 'Unknown'}",
    ]
    if parameter:
        fields.append(f"Parameter: {_clip(parameter, max_step_chars // 4)}")
    fields.extend((f"Success: {success or 'Unknown'}", f"Result:\n{result}"))
    return "\n".join(fields)


def build_trajectory_reward_input(
    task: dict[str, Any],
    workflow: Any,
    candidate_output: Any,
    *,
    artifact_root: str | Path | None = None,
    max_chars: int = 12000,
    max_step_chars: int = 3000,
) -> TrajectoryRewardInput:
    """Build a deterministic path transcript without labels or model identity."""
    if max_chars < 512:
        raise ValueError("trajectory reward max_chars must be at least 512")
    if max_step_chars < 128:
        raise ValueError("trajectory reward max_step_chars must be at least 128")

    # Whitelist Question only so Answer and evaluator metadata can never leak.
    task_text = _clip(_clean_text((task or {}).get("Question", "")), max_chars // 3)
    candidate_text = _safe_artifact_text(candidate_output, artifact_root)
    candidate_text = _clip(candidate_text, max_chars // 3)

    actions: Iterable[Any] = getattr(workflow, "workflow", workflow or ())
    step_blocks = [
        _step_text(action, index, max_step_chars)
        for index, action in enumerate(actions, start=1)
    ]

    fixed_size = len(task_text) + len(candidate_text) + 64
    remaining = max(0, max_chars - fixed_size)
    selected_reversed: list[str] = []
    used = 0
    # Preserve the latest actions when the path is too large; they are closest
    # to the final candidate and usually contain repair/integration evidence.
    for block in reversed(step_blocks):
        separator = 2 if selected_reversed else 0
        if used + separator + len(block) <= remaining:
            selected_reversed.append(block)
            used += separator + len(block)
            continue
        if not selected_reversed and remaining > 0:
            selected_reversed.append(_clip(block, remaining))
        break
    trajectory_text = "\n\n".join(reversed(selected_reversed)) or "No recorded action."

    return TrajectoryRewardInput(
        task_text=task_text,
        trajectory_text=trajectory_text,
        candidate_output=candidate_text,
    )
