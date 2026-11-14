from __future__ import annotations

import json
import os
import time
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from routing_benchmark.routers import DecisionModelRouter

from .canonicalize import append_jsonl, canonical_json, read_jsonl, sha256_text, write_json


MODEL_ID = "openai/gpt-6-sol"
REASONING_EFFORT = "low"
PROMPT_VERSION = "roleroutebench.track_a.sol_router.v1"
PROJECT = Path(__file__).resolve().parents[3]


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


def _routing_schema(candidates: list[str]) -> dict[str, Any]:
    return {
        "type": "object",
        "required": ["action", "rationale"],
        "properties": {
            "action": {"type": "string", "enum": candidates},
            "rationale": {"type": "string"},
        },
        "additionalProperties": False,
    }


def _messages(observation: dict[str, Any], candidates: list[str]) -> tuple[list[dict[str, str]], dict[str, Any]]:
    benchmark_version = str(observation.get("benchmark_version", ""))
    criteria = DecisionModelRouter._criteria(observation, candidates, "structured")
    instructions = DecisionModelRouter._instructions(
        "structured",
        benchmark_version.startswith(("2.0-beta", "3.", "4.")),
        benchmark_version,
    )
    schema = _routing_schema(candidates)
    payload = {
        "instructions": instructions,
        "observation": observation,
        "candidate_contracts": criteria,
        "output_schema": schema,
    }
    return [
        {
            "role": "system",
            "content": (
                "You are the Sol router for RoleRouteBench Track A. Select exactly one "
                "listed action using only the public observation and role contracts. "
                "Do not classify the pool, rank multiple actions, or request hidden "
                "ground truth. Return one JSON object matching the schema."
            ),
        },
        {"role": "user", "content": json.dumps(payload, ensure_ascii=False)},
    ], schema


@dataclass
class OpenRouterSolClient:
    api_key_env: str = "OPENROUTER_API_KEY"
    base_url: str = "https://openrouter.ai/api/v1"
    model_id: str = MODEL_ID
    reasoning_effort: str = REASONING_EFFORT
    max_output_tokens: int = 4096
    timeout_seconds: float = 900.0

    def __post_init__(self) -> None:
        from dotenv import load_dotenv
        from openai import OpenAI

        load_dotenv(PROJECT.parent / ".env", override=False)
        api_key = os.getenv(self.api_key_env)
        if not api_key:
            raise RuntimeError(f"missing OpenRouter credential: set {self.api_key_env}")
        self.client = OpenAI(
            api_key=api_key,
            base_url=self.base_url,
            timeout=self.timeout_seconds,
            max_retries=0,
            default_headers={
                "HTTP-Referer": "https://huggingface.co/datasets/TrangBui/RoleRouteBench",
                "X-Title": "RoleRouteBench Track A Sol router",
            },
        )

    def select(self, observation: dict[str, Any], candidates: list[str]) -> tuple[dict[str, Any], dict[str, Any], str]:
        messages, schema = _messages(observation, candidates)
        response = self.client.chat.completions.create(
            model=self.model_id,
            messages=messages,
            response_format={
                "type": "json_schema",
                "json_schema": {
                    "name": "roleroutebench_track_a_sol_route",
                    "strict": True,
                    "schema": schema,
                },
            },
            max_tokens=self.max_output_tokens,
            extra_body={
                "reasoning": {"effort": self.reasoning_effort, "exclude": True},
                "provider": {"require_parameters": True, "allow_fallbacks": True},
            },
        )
        raw_text = response.choices[0].message.content or ""
        payload = json.loads(raw_text)
        usage = _as_dict(response.usage)
        usage["response_id"] = response.id
        usage["resolved_model"] = response.model
        raw_response = _as_dict(response)
        if raw_response.get("provider"):
            usage["resolved_provider"] = raw_response["provider"]
        return payload, usage, raw_text


def _completed(path: Path) -> set[str]:
    return {row["decision_id"] for row in read_jsonl(path) if row.get("status") == "proposed"}


def run_sol_router(
    states_path: Path,
    output_dir: Path,
    client: OpenRouterSolClient,
    *,
    max_attempts: int = 2,
    max_consecutive_state_failures: int = 3,
) -> dict[str, Any]:
    rows = read_jsonl(states_path)
    output_dir.mkdir(parents=True, exist_ok=True)
    attempts_path = output_dir / "attempts.jsonl"
    predictions_path = output_dir / "predictions.jsonl"
    done = _completed(predictions_path)
    successes = failures = skipped = 0
    consecutive_failures = 0
    aborted = False

    for position, row in enumerate(rows, start=1):
        decision_id = row["decision_id"]
        if decision_id in done:
            skipped += 1
            continue
        observation = json.loads(row["router_input_json"])
        candidates = json.loads(row["candidate_actions_json"])
        if observation.get("constraints", {}).get("allowed_actions") != candidates:
            raise RuntimeError(f"candidate pool mismatch: {decision_id}")
        prediction: dict[str, Any] | None = None
        for attempt_number in range(1, max_attempts + 1):
            started = time.perf_counter()
            started_at = _utc_now()
            try:
                payload, usage, raw_text = client.select(observation, candidates)
                action = str(payload.get("action", "")).strip()
                rationale = str(payload.get("rationale", "")).strip()
                errors = []
                if action not in candidates:
                    errors.append("action is outside the full candidate pool")
                if not rationale:
                    errors.append("rationale is empty")
                status = "proposed" if not errors else "schema_failed"
                error_text = None
            except Exception as exc:
                payload, usage, raw_text = {}, {}, ""
                action, rationale = "", ""
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
                "model_provider": "openrouter_chat_completions",
                "model_id": client.model_id,
                "reasoning_effort": client.reasoning_effort,
                "prompt_version": PROMPT_VERSION,
                "orchestration_profile": "structured",
                "full_pool_unmasked": True,
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
                prediction = {
                    "decision_id": decision_id,
                    "domain": row["domain"],
                    "packet_id": row["packet_id"],
                    "public_state_sha256": row["public_state_sha256"],
                    "selected_action": action,
                    "rationale": rationale,
                    "candidate_actions_sha256": sha256_text(canonical_json(candidates)),
                    "model_provider": attempt["model_provider"],
                    "model_id": attempt["model_id"],
                    "reasoning_effort": attempt["reasoning_effort"],
                    "prompt_version": attempt["prompt_version"],
                    "orchestration_profile": attempt["orchestration_profile"],
                    "full_pool_unmasked": True,
                    "status": status,
                    "usage": usage,
                    "latency_ms": latency_ms,
                }
                append_jsonl(predictions_path, prediction)
                successes += 1
                consecutive_failures = 0
                break
        if prediction is None:
            failures += 1
            consecutive_failures += 1
        print(f"[{position}/{len(rows)}] {decision_id}: {'ok' if prediction else 'failed'}", flush=True)
        if consecutive_failures >= max_consecutive_state_failures:
            aborted = True
            print(
                f"aborting after {consecutive_failures} consecutive state failures; rerun to resume",
                flush=True,
            )
            break

    summary = {
        "requested": len(rows),
        "successes_this_run": successes,
        "failures_this_run": failures,
        "skipped_completed": skipped,
        "completed_total": len(_completed(predictions_path)),
        "aborted_after_consecutive_failures": aborted,
        "model_id": client.model_id,
        "reasoning_effort": client.reasoning_effort,
        "prompt_version": PROMPT_VERSION,
        "orchestration_profile": "structured",
        "full_pool_unmasked": True,
    }
    write_json(output_dir / "summary.json", summary)
    return summary


def score_sol_against_terra(root: Path, run_dir: Path, output_path: Path) -> dict[str, Any]:
    predictions = {row["decision_id"]: row for row in read_jsonl(run_dir / "predictions.jsonl")}
    labels: dict[str, dict[str, Any]] = {}
    for cohort in sorted(root.glob("labeling*")):
        if (cohort / "sample_manifest.json").exists():
            labels.update({row["decision_id"]: row for row in read_jsonl(cohort / "llm_prelabels.jsonl")})
    states = {row["decision_id"]: row for row in read_jsonl(root / "track_a_states.jsonl")}
    matched = sorted(set(states) & set(labels) & set(predictions))
    totals = Counter()
    by_domain: dict[str, Counter[str]] = defaultdict(Counter)
    by_role: dict[str, Counter[str]] = defaultdict(Counter)
    by_phase: dict[str, Counter[str]] = defaultdict(Counter)
    classification_counts: Counter[str] = Counter()
    records: list[dict[str, Any]] = []

    for decision_id in matched:
        state = states[decision_id]
        label = labels[decision_id]
        prediction = predictions[decision_id]
        selected = prediction["selected_action"]
        valid_actions = set(label.get("valid_actions", []))
        action_classes = {
            item["action"]: item["classification"]
            for item in label["classification"]["actions"]
        }
        selected_class = action_classes.get(selected, "missing_from_label")
        valid = selected in valid_actions
        terminal_valid = bool(label.get("terminal_valid"))
        is_stop = selected == "STOP"
        singleton = len(valid_actions) == 1
        domain = state["domain"]
        phase = label.get("state_phase", "unknown")
        totals.update(
            states=1,
            correct=int(valid),
            singleton=int(singleton),
            singleton_correct=int(singleton and valid),
            executor=int(not is_stop),
            executor_correct=int(not is_stop and valid),
            stop_selected=int(is_stop),
            stop_correct=int(is_stop and terminal_valid),
            terminal=int(terminal_valid),
        )
        for bucket, key in ((by_domain, domain), (by_role, selected), (by_phase, phase)):
            bucket[key].update(states=1, correct=int(valid))
        classification_counts[selected_class] += 1
        records.append(
            {
                "decision_id": decision_id,
                "domain": domain,
                "selected_action": selected,
                "valid_actions": sorted(valid_actions),
                "valid": valid,
                "terra_selected_classification": selected_class,
                "state_phase": phase,
                "terminal_valid": terminal_valid,
            }
        )

    def rate(a: int, b: int) -> float | None:
        return a / b if b else None

    def groups(source: dict[str, Counter[str]]) -> dict[str, Any]:
        return {
            key: {
                **dict(value),
                "accuracy": rate(value["correct"], value["states"]),
            }
            for key, value in sorted(source.items())
        }

    attempts = read_jsonl(run_dir / "attempts.jsonl")
    usages = [row.get("usage") or {} for row in predictions.values()]
    input_tokens = sum(int(item.get("prompt_tokens") or item.get("input_tokens") or 0) for item in usages)
    output_tokens = sum(int(item.get("completion_tokens") or item.get("output_tokens") or 0) for item in usages)
    reasoning_tokens = sum(
        int((item.get("completion_tokens_details") or {}).get("reasoning_tokens") or 0)
        for item in usages
    )
    report = {
        "metric_definition": "Sol selected_action is a member of Terra valid_actions",
        "coverage": {
            "primary_states": len(states),
            "predictions": len(predictions),
            "matched": len(matched),
            "missing_predictions": len(set(states) - set(predictions)),
        },
        "overall": {
            **dict(totals),
            "accuracy": rate(totals["correct"], totals["states"]),
            "singleton_exact_agreement": rate(totals["singleton_correct"], totals["singleton"]),
            "executor_accuracy": rate(totals["executor_correct"], totals["executor"]),
            "stop_precision": rate(totals["stop_correct"], totals["stop_selected"]),
            "stop_recall": rate(totals["stop_correct"], totals["terminal"]),
        },
        "terra_classification_of_sol_action": dict(classification_counts),
        "by_domain": groups(by_domain),
        "by_selected_role": groups(by_role),
        "by_state_phase": groups(by_phase),
        "usage": {
            "attempts": len(attempts),
            "attempt_status_counts": dict(Counter(row.get("status") for row in attempts)),
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "reasoning_tokens": reasoning_tokens,
            "total_tokens": input_tokens + output_tokens,
            "provider_reported_usd": sum(float(item.get("cost") or 0) for item in usages),
            "resolved_models": dict(Counter(item.get("resolved_model") for item in usages)),
            "resolved_providers": dict(Counter(item.get("resolved_provider") for item in usages)),
        },
        "records": records,
    }
    write_json(output_path, report)
    return report
