from __future__ import annotations

import json
import re
import time
from typing import Any, Mapping

from routing_benchmark.v4.artifacts.schema import canonical_hash


TOOL_ACTOR_PROMPT_VERSION = "routingbench_v4_tool_actor_v4_repository_grounding"


READ_ONLY_TOOLS = frozenset(
    {"inspect_inputs", "inspect_repository_file", "search_repository"}
)


def _parse_object(text: str) -> dict[str, Any] | None:
    candidate = text.strip()
    fenced = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", candidate, re.S | re.I)
    if fenced:
        candidate = fenced.group(1)
    try:
        value = json.loads(candidate)
        return value if isinstance(value, dict) else None
    except json.JSONDecodeError:
        start = candidate.find("{")
        end = candidate.rfind("}")
        if start >= 0 and end > start:
            try:
                value = json.loads(candidate[start : end + 1])
                return value if isinstance(value, dict) else None
            except json.JSONDecodeError:
                return None
    return None


def _choose_work_tool(tools, stage: str = "") -> str | None:
    del stage
    names = [item.name for item in tools]
    # The adapter already locks tools by role. The actor must not infer a
    # hidden workflow stage to decide which capability the role owns.
    return next((name for name in names if name not in READ_ONLY_TOOLS), None)


def _extract_unified_diff(text: str) -> str | None:
    """Extract an executor-authored diff without synthesizing or repairing it."""

    fenced = re.search(r"```(?:diff|patch)?\s*(---\s+.*?\n\+\+\+\s+.*?)(?:```|\Z)", text, re.S | re.I)
    if fenced:
        # Preserve whitespace-only hunk context lines. ``strip()`` corrupts a
        # valid unified diff when the hunk ends with blank source lines because
        # those lines are represented by a single leading space.
        return fenced.group(1).rstrip("\r\n") + "\n"
    start = re.search(r"(?m)^---\s+\S+", text)
    if start and re.search(r"(?m)^\+\+\+\s+\S+", text[start.start() :]):
        return text[start.start() :].rstrip("\r\n") + "\n"
    return None


def _decode_json_string_fragment(fragment: str) -> str:
    """Decode JSON string escapes without requiring a valid outer JSON object.

    Patch-producing models occasionally leave quotes from Python docstrings
    unescaped while still escaping newlines.  The outer object is then invalid
    JSON even though the patch payload is recoverable.  This decoder only
    normalizes transport escapes; it never edits patch lines or synthesizes a
    diff.
    """

    decoded: list[str] = []
    index = 0
    escapes = {
        '"': '"',
        "\\": "\\",
        "/": "/",
        "b": "\b",
        "f": "\f",
        "n": "\n",
        "r": "\r",
        "t": "\t",
    }
    while index < len(fragment):
        current = fragment[index]
        if current != "\\" or index + 1 >= len(fragment):
            decoded.append(current)
            index += 1
            continue
        marker = fragment[index + 1]
        if marker == "u" and index + 5 < len(fragment):
            codepoint = fragment[index + 2 : index + 6]
            if re.fullmatch(r"[0-9A-Fa-f]{4}", codepoint):
                decoded.append(chr(int(codepoint, 16)))
                index += 6
                continue
        if marker in escapes:
            decoded.append(escapes[marker])
            index += 2
            continue
        # Preserve unknown escapes exactly. A transport normalizer must not
        # silently rewrite executor-authored source code.
        decoded.extend(("\\", marker))
        index += 2
    return "".join(decoded)


def _extract_executor_patch(text: str) -> tuple[str, str] | None:
    """Extract an exact executor-authored patch from supported transports.

    Returns ``(patch, source_format)`` so audits can distinguish native JSON,
    an explicit patch envelope, a direct diff, and recovery from a malformed
    JSON wrapper.  This function extracts only; ``git apply --check`` remains
    authoritative.
    """

    envelope = re.search(r"<patch>(.*?)</patch>", text, re.S | re.I)
    if envelope:
        patch = _extract_unified_diff(envelope.group(1))
        if patch:
            return patch, "patch_envelope"

    patch = _extract_unified_diff(text)
    if patch:
        return patch, "direct_unified_diff"

    # Recover only the explicitly named patch argument and require it to
    # contain a complete unified diff. The greedy terminal match deliberately
    # tolerates unescaped quotes inside the patch body (for example Python
    # triple-quoted docstrings) without guessing where source strings end.
    start = re.search(r'"patch"\s*:\s*"', text, re.I)
    if not start:
        return None
    tail = text[start.end() :]
    terminal = re.search(r'"\s*}\s*}\s*(?:```)?\s*$', tail, re.S)
    if not terminal:
        return None
    decoded = _decode_json_string_fragment(tail[: terminal.start()])
    patch = _extract_unified_diff(decoded)
    if patch:
        return patch, "malformed_json_patch_argument"
    return None


class DeterministicToolActor:
    """Local smoke-test actor. Adapter tools, not this actor, produce artifacts."""

    model = "deterministic-tool-actor"

    def execute(self, *, env, adapter, action: str, role_profile: Mapping[str, Any], task: str) -> dict[str, Any]:
        del task
        before = adapter.filesystem_hashes(env)
        observation = adapter.public_observation(env)
        tools = adapter.tool_spec(env, action, role_profile)
        selected = _choose_work_tool(tools)
        events = []
        if selected is None:
            result = adapter.execute_tool(env, "inspect_inputs", {})
            events.append({"tool_call": {"name": "inspect_inputs", "arguments": {}}, "tool_result": result.to_dict()})
            final = "Inspected the concrete inputs but produced no state-changing artifact."
        else:
            arguments = {}
            if selected == "diagnose_failure":
                arguments = {"analysis": "Deterministic smoke diagnosis based on the latest concrete test result."}
            result = adapter.execute_tool(env, selected, arguments)
            events.append({"tool_call": {"name": selected, "arguments": {}}, "tool_result": result.to_dict()})
            final = result.observable_message
        after = adapter.filesystem_hashes(env)
        return {
            "raw_output": final,
            "work_product": final,
            "raw_model_responses": [],
            "tool_events": events,
            "artifact_diff": adapter.artifact_diff(before, after).to_dict(),
            "transport_status": "success",
            "transport_error": None,
            "token_usage": {"total_tokens": 0},
            "generation": {
                "model": self.model,
                "prompt_version": "deterministic_smoke_v1",
                "prompt_hash": canonical_hash([action, role_profile, observation], 64),
            },
        }


class ToolAwareActorClient:
    """Bounded LLM tool loop backed by authoritative domain adapters."""

    def __init__(self, model: str, *, max_tool_calls: int = 2) -> None:
        from model.query_manager import query_manager

        self.query_manager = query_manager
        self.model = model
        self.max_tool_calls = max(1, int(max_tool_calls))

    def execute(self, *, env, adapter, action: str, role_profile: Mapping[str, Any], task: str) -> dict[str, Any]:
        tools = list(adapter.tool_spec(env, action, role_profile))
        allowed = {item.name: item for item in tools}
        # Code repair commonly needs search -> exact file inspection -> patch.
        # This is an executor-local tool budget, not a routing subgraph: no
        # agent order is imposed and other domains retain their configured
        # bound.
        effective_max_tool_calls = self.max_tool_calls
        if "repair_candidate" in allowed and any(
            name in allowed for name in ("inspect_repository_file", "search_repository")
        ):
            effective_max_tool_calls = max(effective_max_tool_calls, 3)
        if "diagnose_failure" in allowed and any(
            name in allowed for name in ("inspect_repository_file", "search_repository")
        ):
            effective_max_tool_calls = max(effective_max_tool_calls, 3)
        observation = adapter.public_observation(env)
        executor_context = {}
        context_builder = getattr(adapter, "executor_context", None)
        if callable(context_builder):
            executor_context = dict(context_builder(env, action, role_profile) or {})
        payload = {
            "task": task,
            "current_state": observation,
            "executor_context": executor_context,
            "your_role": dict(role_profile),
            "available_tools": [item.to_dict() for item in tools],
            "protocol": {
                "tool_call": {"type": "tool_call", "tool": "tool name", "arguments": {}},
                "final": {"type": "final", "content": "concise work product"},
                "repair_patch_envelope": (
                    "<tool>repair_candidate</tool><patch>complete unified diff</patch>"
                ),
            },
            "instructions": (
                "Perform only this role. Use an available tool when concrete work is needed. "
                "Return exactly one JSON object per response. Tool results are authoritative; "
                "do not invent file contents or claim a tool ran when it did not. Do not give "
                "routing advice or name the next agent. Use the exact flat form "
                "{\"type\":\"tool_call\",\"tool\":\"name\",\"arguments\":{...}}. "
                "Do not emit tool_code or executable snippets. If repair_candidate is available, "
                "ground every patch in the pinned checkout. Use search_repository and "
                "inspect_repository_file when exact source is not already present; never invent file contents. "
                "After inspecting the exact source, call repair_candidate with a complete unified diff "
                "or return the exact non-JSON patch envelope "
                "<tool>repair_candidate</tool><patch>...</patch>; do not place a multiline patch inside a JSON string. "
                "a diagnosis or suggested edit alone does not complete a repair role. If diagnose_failure "
                "is available, analyze the supplied empirical test output and pass that analysis verbatim "
                "in the tool's analysis argument."
            ),
        }
        prompt_hash = canonical_hash({"version": TOOL_ACTOR_PROMPT_VERSION, "payload": payload}, 64)
        messages = [
            {"role": "system", "content": "You are a bounded executor in a multi-agent workflow."},
            {"role": "user", "content": json.dumps(payload, ensure_ascii=False)},
        ]
        before = adapter.filesystem_hashes(env)
        events: list[dict[str, Any]] = []
        raw_responses: list[str] = []
        final = ""
        tokens = 0
        error = None
        status = "success"
        failure_type = None
        protocol_retries = 0
        started = time.perf_counter()
        try:
            for _ in range(effective_max_tool_calls + 1):
                raw, used = self.query_manager.query(self.model, messages)
                raw = str(raw or "")
                raw_responses.append(raw)
                tokens += int(used or 0)
                parsed = _parse_object(raw)
                if not parsed:
                    extracted = (
                        _extract_executor_patch(raw)
                        if "repair_candidate" in allowed
                        else None
                    )
                    if extracted:
                        patch, source_format = extracted
                        parsed = {
                            "type": "tool_call",
                            "tool": "repair_candidate",
                            "arguments": {"patch": patch},
                            "derived_from_raw_output": source_format,
                        }
                    elif protocol_retries < 1:
                        protocol_retries += 1
                        messages.extend(
                            [
                                {"role": "assistant", "content": raw},
                                {
                                    "role": "user",
                                    "content": (
                                        "PROTOCOL_ERROR\nThe response could not be parsed. Return exactly one flat "
                                        "JSON object. If you need to submit a multiline patch, avoid JSON escaping "
                                        "and return exactly <tool>repair_candidate</tool><patch>the complete unified "
                                        "diff</patch>. Do not add prose."
                                    ),
                                },
                            ]
                        )
                        continue
                    else:
                        final = raw
                        status = "invalid_protocol"
                        failure_type = (
                            "patch_extraction_failed"
                            if "repair_candidate" in allowed
                            else "executor_protocol_failure"
                        )
                        break
                nested_tool_call = parsed.get("tool_call")
                if isinstance(nested_tool_call, Mapping):
                    parsed = {
                        "type": "tool_call",
                        "tool": nested_tool_call.get("tool") or nested_tool_call.get("name"),
                        "arguments": nested_tool_call.get("arguments") or {},
                    }
                nested_final = parsed.get("final")
                if isinstance(nested_final, Mapping):
                    parsed = {
                        "type": "final",
                        "content": nested_final.get("content") or "",
                    }
                if "tool_code" in parsed and isinstance(parsed.get("tool_code"), str):
                    code = str(parsed["tool_code"])
                    inferred = next(
                        (name for name in allowed if re.search(rf"\b{re.escape(name)}\s*\(", code)),
                        None,
                    )
                    if inferred is None and "inspect_inputs" in allowed and (
                        "open(" in code or "get_artifact(" in code
                    ):
                        inferred = "inspect_inputs"
                    if inferred is not None:
                        parsed = {"type": "tool_call", "tool": inferred, "arguments": {}}
                if parsed.get("type") == "final":
                    final = str(parsed.get("content", ""))
                    # Some tool-capable models return a complete unified diff in
                    # their final field instead of issuing the explicitly offered
                    # repair call. Deterministically commit that exact diff; do
                    # not rewrite it or consult gold data.
                    if "repair_candidate" in allowed:
                        extracted_patch = _extract_unified_diff(final)
                        if extracted_patch:
                            result = adapter.execute_tool(
                                env,
                                "repair_candidate",
                                {"patch": extracted_patch},
                            ).to_dict()
                            events.append(
                                {
                                    "raw_model_response": raw,
                                    "tool_call": {
                                        "name": "repair_candidate",
                                        "arguments": {"patch": extracted_patch},
                                        "derived_from_final_output": True,
                                    },
                                    "tool_result": result,
                                }
                            )
                    break
                if parsed.get("type") != "tool_call":
                    final = raw
                    status = "invalid_protocol"
                    failure_type = "executor_protocol_failure"
                    break
                name = str(parsed.get("tool", ""))
                arguments = parsed.get("arguments") if isinstance(parsed.get("arguments"), Mapping) else {}
                if name not in allowed:
                    result = {"tool_name": name, "status": "rejected", "output": {}, "observable_message": "Tool is not authorized for this role action.", "cost": 0.0}
                else:
                    result = adapter.execute_tool(env, name, arguments).to_dict()
                tool_call = {"name": name, "arguments": dict(arguments)}
                if parsed.get("derived_from_raw_output"):
                    tool_call["derived_from_raw_output"] = str(
                        parsed["derived_from_raw_output"]
                    )
                events.append({"raw_model_response": raw, "tool_call": tool_call, "tool_result": result})
                if (
                    name in allowed
                    and name not in READ_ONLY_TOOLS
                    and result.get("status") == "success"
                ):
                    # A selected agent performs one bounded workflow operation.
                    # Once its authoritative state-changing tool succeeds, the
                    # action is complete; another model turn can only duplicate
                    # work or attempt a tool outside this role's contract.
                    final = str(result.get("observable_message", ""))
                    failure_type = None
                    break
                if name == "repair_candidate" and result.get("status") == "invalid_output":
                    failure_type = "patch_apply_rejected"
                elif name == "validate_candidate" and result.get("status") == "invalid_output":
                    failure_type = "patch_test_failed"
                elif result.get("status") in {"invalid_output", "rejected", "no_progress"}:
                    failure_type = "executor_invalid_output"
                messages.extend(
                    [
                        {"role": "assistant", "content": raw},
                        {
                            "role": "user",
                            "content": (
                                "TOOL_RESULT\n"
                                + json.dumps(result, ensure_ascii=False)
                                + "\nReturn one flat JSON response. If an authorized state-changing tool "
                                "remains necessary for your role (especially diagnose_failure or "
                                "repair_candidate after repository inspection), call it now instead "
                                "of returning advice."
                            ),
                        },
                    ]
                )
                if len(events) >= effective_max_tool_calls:
                    final = str(result.get("observable_message", ""))
                    break
        except Exception as exc:
            status = "transport_failure"
            error = f"{type(exc).__name__}: {exc}"
            failure_type = "executor_transport_failure"
        after = adapter.filesystem_hashes(env)
        raw_output = "\n\n".join(item for item in raw_responses if item)
        return {
            # Exact model responses are the empirical output. The normalized
            # work product/tool message is stored separately and never replaces
            # what the executor actually emitted.
            "raw_output": raw_output,
            "work_product": final,
            "raw_model_responses": raw_responses,
            "tool_events": events,
            "artifact_diff": adapter.artifact_diff(before, after).to_dict(),
            "transport_status": status,
            "transport_error": error,
            "failure_type": failure_type,
            "protocol_retry_count": protocol_retries,
            "token_usage": {"total_tokens": tokens},
            "latency_ms": (time.perf_counter() - started) * 1000.0,
            "generation": {
                "model": self.model,
                "prompt_version": TOOL_ACTOR_PROMPT_VERSION,
                "prompt_hash": prompt_hash,
            },
        }
