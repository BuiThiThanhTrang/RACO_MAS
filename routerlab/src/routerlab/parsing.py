from __future__ import annotations

import json
import re
from typing import Any


def extract_code(text: str) -> str:
    blocks = re.findall(r"```(?:python)?\s*([\s\S]*?)```", text, flags=re.IGNORECASE)
    candidate = blocks[-1] if blocks else text
    candidate = candidate.strip()
    if candidate.lower().startswith("python\n"):
        candidate = candidate[7:].lstrip()
    return candidate


def parse_findings(text: str) -> tuple[str, ...]:
    cleaned = text.strip()
    if cleaned.startswith("```"):
        cleaned = re.sub(r"^```(?:json)?\s*|\s*```$", "", cleaned, flags=re.IGNORECASE)
    try:
        payload: Any = json.loads(cleaned)
        values = payload.get("findings", payload) if isinstance(payload, dict) else payload
        if isinstance(values, list):
            return tuple(str(item).strip() for item in values if str(item).strip())
    except json.JSONDecodeError:
        pass
    lines = [re.sub(r"^[-*\d.\s]+", "", line).strip() for line in cleaned.splitlines()]
    return tuple(line for line in lines if line)


def parse_structured_findings(
    text: str,
) -> tuple[tuple[str, ...], tuple[dict[str, Any], ...]]:
    cleaned = text.strip()
    if cleaned.startswith("```"):
        cleaned = re.sub(r"^```(?:json)?\s*|\s*```$", "", cleaned, flags=re.IGNORECASE)
    try:
        payload: Any = json.loads(cleaned)
    except json.JSONDecodeError:
        return parse_findings(text), ()
    values = payload.get("findings", payload) if isinstance(payload, dict) else payload
    if not isinstance(values, list):
        return parse_findings(text), ()
    claims: list[str] = []
    records: list[dict[str, Any]] = []
    for index, item in enumerate(values, start=1):
        if isinstance(item, dict):
            claim = str(item.get("claim", "")).strip()
            if not claim:
                continue
            span = item.get("code_span")
            if span is not None and not isinstance(span, dict):
                span = None
            claims.append(claim)
            records.append(
                {
                    "finding_id": str(item.get("finding_id") or f"finding-{index}"),
                    "category": str(item.get("category") or "unknown"),
                    "claim": claim,
                    "code_span": span,
                    "confidence": str(item.get("confidence") or "unknown"),
                }
            )
        else:
            claim = str(item).strip()
            if claim:
                claims.append(claim)
    return tuple(claims), tuple(records)


def parse_router_action(text: str, action_space: tuple[str, ...]) -> tuple[str, bool]:
    cleaned = text.strip()
    try:
        payload = json.loads(cleaned)
        action = str(payload.get("action", "")).strip()
    except json.JSONDecodeError:
        action = cleaned.splitlines()[-1].strip() if cleaned else ""
    if action in action_space:
        return action, True
    fallback = "STOP" if "STOP" in action_space else action_space[0]
    return fallback, False
