from __future__ import annotations

import json
import os
import subprocess
import tempfile
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Protocol

from .canonicalize import append_jsonl, canonical_json, read_jsonl, sha256_text
from .prompt import SYSTEM_INSTRUCTIONS, build_user_prompt
from .schema import (
    LABEL_CONTRACT_VERSION,
    PROMPT_VERSION,
    derived_action_sets,
    prelabel_json_schema,
    validate_prelabel,
)


MODEL_ID = "gpt-6.1-terra"
REASONING_EFFORT = "high"
PROJECT = Path(__file__).resolve().parents[3]


class LabelClient(Protocol):
    provider_name: str
    model_id: str
    reasoning_effort: str

    def label(self, row: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any], str]: ...


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _as_dict(value: Any) -> dict[str, Any]:
    if value is None:
        return {}
    if isinstance(value, dict):
        return value
    if hasattr(value, "model_dump"):
        return value.model_dump()
    return json.loads(json.dumps(value, default=lambda item: getattr(item, "__dict__", str(item))))


@dataclass
class OpenAIResponsesClient:
    api_key_env: str = "OPENAI_API_KEY"
    base_url: str | None = None
    max_output_tokens: int = 25_000
    timeout_seconds: float = 900.0
    provider_name: str = "openai_responses"
    model_id: str = MODEL_ID
    reasoning_effort: str = REASONING_EFFORT

    def __post_init__(self) -> None:
        from dotenv import load_dotenv

        load_dotenv(PROJECT.parent / ".env", override=False)
        api_key = os.getenv(self.api_key_env)
        if not api_key:
            raise RuntimeError(f"missing API credential: set {self.api_key_env}")
        from openai import OpenAI

        self.client = OpenAI(api_key=api_key, base_url=self.base_url, timeout=self.timeout_seconds, max_retries=0)

    def label(self, row: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any], str]:
        response = self.client.responses.create(
            model=self.model_id,
            reasoning={"effort": self.reasoning_effort},
            max_output_tokens=self.max_output_tokens,
            instructions=SYSTEM_INSTRUCTIONS,
            input=build_user_prompt(row),
            text={
                "format": {
                    "type": "json_schema",
                    "name": "roleroutebench_track_a_prelabel",
                    "strict": True,
                    "schema": prelabel_json_schema(),
                }
            },
            store=False,
            metadata={"decision_id": row["decision_id"], "prompt_version": PROMPT_VERSION},
        )
        raw_text = response.output_text
        return json.loads(raw_text), _as_dict(response.usage), raw_text


@dataclass
class CodexExecClient:
    """Credential fallback using the authenticated Codex CLI.

    This mode is explicit in provenance because the Codex agent wrapper adds its
    own system context. The canonical API mode remains OpenAIResponsesClient.
    """

    max_output_tokens: int = 25_000
    timeout_seconds: float = 900.0
    provider_name: str = "codex_exec_openai"
    model_id: str = MODEL_ID
    reasoning_effort: str = REASONING_EFFORT

    def label(self, row: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any], str]:
        prompt = SYSTEM_INSTRUCTIONS + "\n\n" + build_user_prompt(row)
        temp_root = PROJECT / ".codex_tmp" / "roleroutebench_labels"
        temp_root.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(prefix="call-", dir=temp_root) as temp:
            root = Path(temp)
            schema_path = root / "schema.json"
            output_path = root / "output.json"
            schema_path.write_text(json.dumps(prelabel_json_schema(), ensure_ascii=False), encoding="utf-8")
            command = [
                "codex",
                "exec",
                "--ephemeral",
                "--ignore-user-config",
                "--ignore-rules",
                "--skip-git-repo-check",
                "--sandbox",
                "read-only",
                "--model",
                self.model_id,
                "-c",
                f'model_reasoning_effort="{self.reasoning_effort}"',
                "--output-schema",
                str(schema_path),
                "--output-last-message",
                str(output_path),
                "--json",
                "-",
            ]
            completed = subprocess.run(
                command,
                cwd=root,
                input=prompt,
                text=True,
                encoding="utf-8",
                capture_output=True,
                timeout=self.timeout_seconds,
                check=False,
            )
            if completed.returncode != 0:
                detail = completed.stderr[-4000:] or completed.stdout[-4000:]
                raise RuntimeError(f"codex exec failed ({completed.returncode}): {detail}")
            raw_text = output_path.read_text(encoding="utf-8")
            usage: dict[str, Any] = {}
            for line in completed.stdout.splitlines():
                try:
                    event = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if event.get("type") in {"turn.completed", "task_complete"}:
                    usage = event.get("usage") or usage
                if isinstance(event.get("usage"), dict):
                    usage = event["usage"]
            return json.loads(raw_text), usage, raw_text


@dataclass
class OpenRouterChatClient:
    api_key_env: str = "OPENROUTER_API_KEY"
    base_url: str = "https://openrouter.ai/api/v1"
    max_output_tokens: int = 25_000
    timeout_seconds: float = 900.0
    provider_name: str = "openrouter_chat_completions"
    model_id: str = "openai/gpt-6.1-terra"
    reasoning_effort: str = "high"

    def __post_init__(self) -> None:
        from dotenv import load_dotenv

        load_dotenv(PROJECT.parent / ".env", override=False)
        api_key = os.getenv(self.api_key_env)
        if not api_key:
            raise RuntimeError(
                f"missing OpenRouter credential: set {self.api_key_env}; "
                "no request has been sent"
            )
        from openai import OpenAI

        self.client = OpenAI(
            api_key=api_key,
            base_url=self.base_url,
            timeout=self.timeout_seconds,
            max_retries=0,
            default_headers={
                "HTTP-Referer": "https://huggingface.co/datasets/TrangBui/RoleRouteBench",
                "X-Title": "RoleRouteBench Track A labeling",
            },
        )

    def label(self, row: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any], str]:
        response = self.client.chat.completions.create(
            model=self.model_id,
            messages=[
                {"role": "system", "content": SYSTEM_INSTRUCTIONS},
                {"role": "user", "content": build_user_prompt(row)},
            ],
            response_format={
                "type": "json_schema",
                "json_schema": {
                    "name": "roleroutebench_track_a_prelabel",
                    "strict": True,
                    "schema": prelabel_json_schema(),
                },
            },
            max_tokens=self.max_output_tokens,
            extra_body={
                "reasoning": {"effort": self.reasoning_effort, "exclude": True},
                "provider": {"allow_fallbacks": True},
            },
        )
        choice = response.choices[0]
        raw_text = choice.message.content or ""
        usage = _as_dict(response.usage)
        usage["response_id"] = response.id
        usage["resolved_model"] = response.model
        raw_response = _as_dict(response)
        if raw_response.get("provider"):
            usage["resolved_provider"] = raw_response["provider"]
        return json.loads(raw_text), usage, raw_text


def completed_decisions(path: Path) -> set[str]:
    return {
        row["decision_id"]
        for row in read_jsonl(path)
        if row.get("status") == "proposed" and not row.get("validation_errors")
    }


def run_prelabeling(
    sample_states_path: Path,
    output_dir: Path,
    client: LabelClient,
    *,
    max_attempts: int = 2,
) -> dict[str, Any]:
    rows = read_jsonl(sample_states_path)
    attempts_path = output_dir / "llm_attempts.jsonl"
    prelabels_path = output_dir / "llm_prelabels.jsonl"
    output_dir.mkdir(parents=True, exist_ok=True)
    done = completed_decisions(prelabels_path)
    successes = 0
    failures = 0
    skipped = 0
    for position, row in enumerate(rows, start=1):
        decision_id = row["decision_id"]
        if decision_id in done:
            skipped += 1
            continue
        candidates = json.loads(row["candidate_actions_json"])
        final_record: dict[str, Any] | None = None
        for attempt_number in range(1, max_attempts + 1):
            started = time.perf_counter()
            started_at = _utc_now()
            try:
                payload, usage, raw_text = client.label(row)
                errors = validate_prelabel(payload, candidates)
                status = "proposed" if not errors else "schema_failed"
                error_text = None
            except Exception as exc:  # transport and provider failures are audit records
                payload, usage, raw_text = {}, {}, ""
                errors = [str(exc)]
                status = "transport_failed"
                error_text = str(exc)
            latency_ms = round((time.perf_counter() - started) * 1000, 3)
            attempt = {
                "decision_id": decision_id,
                "public_state_sha256": row["public_state_sha256"],
                "sample_position": position,
                "attempt_number": attempt_number,
                "started_at": started_at,
                "model_provider": client.provider_name,
                "model_id": client.model_id,
                "reasoning_effort": client.reasoning_effort,
                "prompt_version": PROMPT_VERSION,
                "label_contract_version": LABEL_CONTRACT_VERSION,
                "status": status,
                "validation_errors": errors,
                "error": error_text,
                "raw_response_sha256": sha256_text(raw_text),
                "raw_response": raw_text,
                "usage": usage,
                "latency_ms": latency_ms,
            }
            append_jsonl(attempts_path, attempt)
            if status == "proposed":
                sets = derived_action_sets(payload)
                final_record = {
                    **{key: attempt[key] for key in (
                        "decision_id",
                        "public_state_sha256",
                        "model_provider",
                        "model_id",
                        "reasoning_effort",
                        "prompt_version",
                        "label_contract_version",
                        "status",
                        "usage",
                        "latency_ms",
                    )},
                    "classification": payload,
                    **sets,
                    "terminal_valid": payload["terminal_valid"],
                    "state_phase": payload["state_phase"],
                    "human_review_status": "pending",
                }
                append_jsonl(prelabels_path, final_record)
                successes += 1
                break
        if final_record is None:
            failures += 1
        print(
            f"[{position}/{len(rows)}] {decision_id}: "
            f"{'ok' if final_record is not None else 'failed'}",
            flush=True,
        )
    summary = {
        "requested": len(rows),
        "successes_this_run": successes,
        "failures_this_run": failures,
        "skipped_completed": skipped,
        "completed_total": len(completed_decisions(prelabels_path)),
        "model_id": client.model_id,
        "reasoning_effort": client.reasoning_effort,
        "provider": client.provider_name,
    }
    (output_dir / "labeling_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return summary

