from __future__ import annotations

import json
import os
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from routing_benchmark.routers import DecisionModelRouter

from .canonicalize import append_jsonl, canonical_json, read_jsonl, sha256_text, write_json


PROMPT_VERSION = DecisionModelRouter.STRUCTURED_PROMPT_VERSION
PROJECT = Path(__file__).resolve().parents[3]


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


@dataclass
class OpenRouterDecisionTrackAClient:
    model_id: str
    timeout_seconds: float = 300.0
    max_retries: int = 5
    retry_backoff_seconds: float = 1.0
    max_retry_delay_seconds: float = 30.0

    def __post_init__(self) -> None:
        from dotenv import load_dotenv

        load_dotenv(PROJECT.parent / ".env", override=False)
        if not os.getenv("OPENROUTER_API_KEY", "").strip():
            raise RuntimeError("missing OpenRouter credential: set OPENROUTER_API_KEY")
        self.router = DecisionModelRouter(
            {
                "provider": "openrouter",
                "model": self.model_id,
                "api_key_env": "OPENROUTER_API_KEY",
                "timeout_seconds": self.timeout_seconds,
                "max_retries": self.max_retries,
                "retry_backoff_seconds": self.retry_backoff_seconds,
                "max_retry_delay_seconds": self.max_retry_delay_seconds,
            },
            orchestration_profile="structured",
            observable_rejection_guard_enabled=False,
        )

    def select(self, observation: dict[str, Any], candidates: list[str]):
        return self.router.select(observation, candidates)


def _completed(path: Path) -> set[str]:
    return {
        row["decision_id"]
        for row in read_jsonl(path)
        if row.get("status") == "proposed"
    }


def run_decision_router(
    states_path: Path,
    output_dir: Path,
    client: OpenRouterDecisionTrackAClient,
    *,
    max_attempts: int = 2,
    max_consecutive_state_failures: int = 3,
    limit: int | None = None,
) -> dict[str, Any]:
    all_rows = read_jsonl(states_path)
    rows = all_rows[:limit] if limit is not None else all_rows
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
            started_at = _utc_now()
            try:
                decision = client.select(observation, candidates)
                action = str(decision.action).strip()
                metadata = dict(decision.metadata)
                probabilities = dict(metadata.get("probabilities") or {})
                errors: list[str] = []
                if action not in candidates:
                    errors.append("action is outside the full candidate pool")
                missing_probabilities = [
                    candidate for candidate in candidates if candidate not in probabilities
                ]
                extra_probabilities = [
                    candidate for candidate in probabilities if candidate not in candidates
                ]
                if missing_probabilities:
                    errors.append(
                        "probability vector is missing candidates: "
                        + ", ".join(missing_probabilities)
                    )
                if extra_probabilities:
                    errors.append(
                        "probability vector has extra candidates: "
                        + ", ".join(extra_probabilities)
                    )
                status = "proposed" if not errors else "schema_failed"
                error_text = None
            except Exception as exc:
                action = ""
                metadata = {}
                probabilities = {}
                errors = [str(exc)]
                status = "transport_failed"
                error_text = str(exc)

            attempt = {
                "decision_id": decision_id,
                "public_state_sha256": row["public_state_sha256"],
                "sample_position": position,
                "attempt_number": attempt_number,
                "started_at": started_at,
                "model_provider": "openrouter_decisions",
                "requested_model_id": client.model_id,
                "resolved_model": metadata.get("model"),
                "model_version": metadata.get("model_version"),
                "prompt_version": metadata.get("prompt_version", PROMPT_VERSION),
                "orchestration_profile": "structured",
                "full_pool_unmasked": True,
                "observable_rejection_guard_enabled": False,
                "status": status,
                "validation_errors": errors,
                "error": error_text,
                "input_tokens": int(metadata.get("input_tokens") or 0),
                "latency_ms": float(metadata.get("latency_ms") or 0),
            }
            append_jsonl(attempts_path, attempt)
            if status == "proposed":
                prediction = {
                    "decision_id": decision_id,
                    "domain": row["domain"],
                    "packet_id": row["packet_id"],
                    "public_state_sha256": row["public_state_sha256"],
                    "selected_action": action,
                    "confidence": metadata.get("confidence"),
                    "probabilities": probabilities,
                    "candidate_actions_sha256": sha256_text(canonical_json(candidates)),
                    "model_provider": attempt["model_provider"],
                    "requested_model_id": attempt["requested_model_id"],
                    "resolved_model": attempt["resolved_model"],
                    "model_version": attempt["model_version"],
                    "prompt_version": attempt["prompt_version"],
                    "orchestration_profile": attempt["orchestration_profile"],
                    "full_pool_unmasked": True,
                    "observable_rejection_guard_enabled": False,
                    "status": status,
                    "input_tokens": attempt["input_tokens"],
                    "latency_ms": attempt["latency_ms"],
                }
                append_jsonl(predictions_path, prediction)
                successes += 1
                consecutive_failures = 0
                break

        if prediction is None:
            failures += 1
            consecutive_failures += 1
        print(
            f"[{position}/{len(rows)}] {decision_id}: "
            f"{'ok' if prediction else 'failed'}",
            flush=True,
        )
        if consecutive_failures >= max_consecutive_state_failures:
            aborted = True
            print(
                f"aborting after {consecutive_failures} consecutive state failures; "
                "rerun to resume",
                flush=True,
            )
            break

    summary = {
        "available_primary_states": len(all_rows),
        "requested_this_invocation": len(rows),
        "successes_this_run": successes,
        "failures_this_run": failures,
        "skipped_completed": skipped,
        "completed_total": len(_completed(predictions_path)),
        "aborted_after_consecutive_failures": aborted,
        "model_id": client.model_id,
        "prompt_version": PROMPT_VERSION,
        "orchestration_profile": "structured",
        "full_pool_unmasked": True,
        "observable_rejection_guard_enabled": False,
        "labels_loaded_during_inference": False,
    }
    write_json(output_dir / "summary.json", summary)
    return summary


def score_decision_router_against_terra(
    root: Path,
    run_dir: Path,
    output_path: Path,
) -> dict[str, Any]:
    predictions = {
        row["decision_id"]: row for row in read_jsonl(run_dir / "predictions.jsonl")
    }
    labels: dict[str, dict[str, Any]] = {}
    for cohort in sorted(root.glob("labeling*")):
        if (cohort / "sample_manifest.json").exists():
            labels.update(
                {
                    row["decision_id"]: row
                    for row in read_jsonl(cohort / "llm_prelabels.jsonl")
                }
            )
    states = {
        row["decision_id"]: row for row in read_jsonl(root / "track_a_states.jsonl")
    }
    matched = sorted(set(states) & set(labels) & set(predictions))
    totals: Counter[str] = Counter()
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
        for bucket, key in (
            (by_domain, domain),
            (by_role, selected),
            (by_phase, phase),
        ):
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

    def rate(numerator: int, denominator: int) -> float | None:
        return numerator / denominator if denominator else None

    def groups(source: dict[str, Counter[str]]) -> dict[str, Any]:
        return {
            key: {
                **dict(value),
                "accuracy": rate(value["correct"], value["states"]),
            }
            for key, value in sorted(source.items())
        }

    attempts = read_jsonl(run_dir / "attempts.jsonl")
    report = {
        "metric_definition": (
            "selected_action is a member of Terra Track A valid_actions"
        ),
        "model_id": next(
            (row.get("requested_model_id") for row in predictions.values()), None
        ),
        "coverage": {
            "primary_states": len(states),
            "predictions": len(predictions),
            "matched": len(matched),
            "missing_predictions": len(set(states) - set(predictions)),
        },
        "overall": {
            **dict(totals),
            "accuracy": rate(totals["correct"], totals["states"]),
            "singleton_exact_agreement": rate(
                totals["singleton_correct"], totals["singleton"]
            ),
            "executor_accuracy": rate(
                totals["executor_correct"], totals["executor"]
            ),
            "stop_precision": rate(totals["stop_correct"], totals["stop_selected"]),
            "stop_recall": rate(totals["stop_correct"], totals["terminal"]),
        },
        "terra_classification_of_selected_action": dict(classification_counts),
        "by_domain": groups(by_domain),
        "by_selected_role": groups(by_role),
        "by_state_phase": groups(by_phase),
        "usage": {
            "attempts": len(attempts),
            "attempt_status_counts": dict(
                Counter(row.get("status") for row in attempts)
            ),
            "input_tokens": sum(
                int(row.get("input_tokens") or 0) for row in predictions.values()
            ),
            "mean_latency_ms": (
                sum(float(row.get("latency_ms") or 0) for row in predictions.values())
                / len(predictions)
                if predictions
                else None
            ),
            "resolved_models": dict(
                Counter(row.get("resolved_model") for row in predictions.values())
            ),
            "model_versions": dict(
                Counter(row.get("model_version") for row in predictions.values())
            ),
        },
        "records": records,
    }
    write_json(output_path, report)
    return report
