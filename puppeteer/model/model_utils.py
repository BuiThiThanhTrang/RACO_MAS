from __future__ import annotations

import json
import logging
import math
import time
from typing import Any, Dict, Iterable, Mapping

import yaml
import httpx
from openai.types.chat import ChatCompletion

logger = logging.getLogger("model")

try:
    with open("./config/global.yaml", "r", encoding="utf-8") as source:
        GLOBAL_CONFIG = yaml.safe_load(source) or {}
except FileNotFoundError:
    GLOBAL_CONFIG = {}

CHAT_MAX_RETRY_TIMES = int(GLOBAL_CONFIG.get("max_retry_times", 3))
CHAT_RETRY_WAIT_MIN = float(GLOBAL_CONFIG.get("retry_wait_min", 1))
CHAT_RETRY_WAIT_MAX = float(GLOBAL_CONFIG.get("retry_wait_max", 3))
CHAT_CONTEXT_CONFIG = dict(GLOBAL_CONFIG.get("chat_context") or {})
CHAT_CONTEXT_MAX_INPUT_TOKENS = int(
    CHAT_CONTEXT_CONFIG.get("max_input_tokens", 24000)
)
CHAT_CONTEXT_RESERVED_OUTPUT_TOKENS = int(
    CHAT_CONTEXT_CONFIG.get("reserved_output_tokens", 4096)
)
CHAT_CONTEXT_RETRY_SHRINK_FACTOR = float(
    CHAT_CONTEXT_CONFIG.get("retry_shrink_factor", 0.75)
)
CHAT_CONTEXT_MIN_INPUT_TOKENS = int(
    CHAT_CONTEXT_CONFIG.get("min_input_tokens", 8000)
)
CHAT_CONTEXT_PRESERVE_SYSTEM = bool(
    CHAT_CONTEXT_CONFIG.get("preserve_system", True)
)
CHAT_CONTEXT_PRESERVE_LATEST = int(
    CHAT_CONTEXT_CONFIG.get("preserve_latest_messages", 4)
)
CHAT_CONTEXT_CHARS_PER_TOKEN = float(
    CHAT_CONTEXT_CONFIG.get("chars_per_token", 3.0)
)
MESSAGE_OVERHEAD_TOKENS = 8
TRUNCATION_MARKER = "\n[... earlier content truncated ...]\n"

if CHAT_CONTEXT_MAX_INPUT_TOKENS < CHAT_CONTEXT_MIN_INPUT_TOKENS:
    raise ValueError("chat_context.max_input_tokens must be >= min_input_tokens")
if CHAT_CONTEXT_MIN_INPUT_TOKENS <= 0:
    raise ValueError("chat_context.min_input_tokens must be positive")
if CHAT_CONTEXT_RESERVED_OUTPUT_TOKENS <= 0:
    raise ValueError("chat_context.reserved_output_tokens must be positive")
if not 0.0 < CHAT_CONTEXT_RETRY_SHRINK_FACTOR < 1.0:
    raise ValueError("chat_context.retry_shrink_factor must be between 0 and 1")
if CHAT_CONTEXT_PRESERVE_LATEST <= 0:
    raise ValueError("chat_context.preserve_latest_messages must be positive")
if CHAT_CONTEXT_CHARS_PER_TOKEN <= 0:
    raise ValueError("chat_context.chars_per_token must be positive")


def model_log_and_print(content):
    if content is not None:
        logger.info(content)
        print(content)


def estimate_text_tokens(content: Any, chars_per_token: float | None = None) -> int:
    chars_per_token = float(chars_per_token or CHAT_CONTEXT_CHARS_PER_TOKEN)
    if chars_per_token <= 0:
        raise ValueError("chars_per_token must be positive")
    text = "" if content is None else str(content)
    return int(math.ceil(len(text) / chars_per_token)) if text else 0


def estimate_message_tokens(
    message: Mapping[str, Any], chars_per_token: float | None = None
) -> int:
    return MESSAGE_OVERHEAD_TOKENS + estimate_text_tokens(
        message.get("content", ""), chars_per_token
    )


def estimate_messages_tokens(
    messages: Iterable[Mapping[str, Any]], chars_per_token: float | None = None
) -> int:
    return sum(
        estimate_message_tokens(message, chars_per_token) for message in messages
    )


def _truncate_content(
    content: Any, token_budget: int, chars_per_token: float
) -> tuple[str, bool]:
    text = "" if content is None else str(content)
    max_chars = max(0, int(token_budget * chars_per_token))
    if len(text) <= max_chars:
        return text, False
    if max_chars <= 0:
        return "", True
    if max_chars <= len(TRUNCATION_MARKER) + 16:
        return text[-max_chars:], True
    available = max_chars - len(TRUNCATION_MARKER)
    head_chars = max(1, int(available * 0.4))
    tail_chars = max(1, available - head_chars)
    return (
        text[:head_chars] + TRUNCATION_MARKER + text[-tail_chars:],
        True,
    )


def _copy_message_with_budget(
    message: Mapping[str, Any], token_budget: int, chars_per_token: float
) -> tuple[dict[str, Any] | None, bool]:
    content_budget = int(token_budget) - MESSAGE_OVERHEAD_TOKENS
    if content_budget <= 0:
        return None, False
    content, truncated = _truncate_content(
        message.get("content", ""), content_budget, chars_per_token
    )
    copied = dict(message)
    copied["content"] = content
    return copied, truncated


def build_context_window(
    messages: Iterable[Mapping[str, Any]],
    max_input_tokens: int = CHAT_CONTEXT_MAX_INPUT_TOKENS,
    preserve_system: bool = CHAT_CONTEXT_PRESERVE_SYSTEM,
    preserve_latest_messages: int = CHAT_CONTEXT_PRESERVE_LATEST,
    chars_per_token: float = CHAT_CONTEXT_CHARS_PER_TOKEN,
) -> tuple[list[dict[str, Any]], dict[str, int]]:
    """Build a bounded request copy without mutating the full dialog."""
    full_messages = [dict(message) for message in messages]
    budget = int(max_input_tokens)
    if budget <= 0:
        raise ValueError("max_input_tokens must be positive")
    if preserve_latest_messages <= 0:
        raise ValueError("preserve_latest_messages must be positive")

    full_tokens = estimate_messages_tokens(full_messages, chars_per_token)
    if full_tokens <= budget:
        request = [dict(message) for message in full_messages]
        return request, {
            "full_messages": len(full_messages),
            "request_messages": len(request),
            "full_estimated_tokens": full_tokens,
            "request_estimated_tokens": full_tokens,
            "dropped_messages": 0,
            "truncated_messages": 0,
            "budget": budget,
        }

    selected: dict[int, dict[str, Any]] = {}
    truncated_count = 0
    remaining = budget
    system_index = next(
        (
            index
            for index, message in enumerate(full_messages)
            if message.get("role") == "system"
        ),
        None,
    )

    if preserve_system and system_index is not None:
        system_allowance = min(remaining, max(512, budget // 3))
        copied, truncated = _copy_message_with_budget(
            full_messages[system_index], system_allowance, chars_per_token
        )
        if copied is not None:
            selected[system_index] = copied
            used = estimate_message_tokens(copied, chars_per_token)
            remaining -= used
            truncated_count += int(truncated)

    non_system_indices = [
        index for index in range(len(full_messages)) if index != system_index
    ]
    latest_user_index = next(
        (
            index
            for index in reversed(non_system_indices)
            if full_messages[index].get("role") == "user"
        ),
        None,
    )
    if latest_user_index is not None and remaining > MESSAGE_OVERHEAD_TOKENS:
        latest_user = full_messages[latest_user_index]
        required = estimate_message_tokens(latest_user, chars_per_token)
        copied, truncated = _copy_message_with_budget(
            latest_user, min(required, remaining), chars_per_token
        )
        if copied is not None:
            selected[latest_user_index] = copied
            remaining -= estimate_message_tokens(copied, chars_per_token)
            truncated_count += int(truncated)

        # Add only complete user -> assistant turns. Keeping an arbitrary recent
        # message suffix can make a compacted request begin with ``assistant``;
        # strict chat templates (including some Hugging Face providers) reject it.
        cursor = latest_user_index - 1
        while cursor > 0:
            assistant_index = cursor
            user_index = cursor - 1
            if system_index == user_index:
                break
            if (
                full_messages[assistant_index].get("role") != "assistant"
                or full_messages[user_index].get("role") != "user"
            ):
                break
            turn_tokens = estimate_messages_tokens(
                [full_messages[user_index], full_messages[assistant_index]],
                chars_per_token,
            )
            if turn_tokens > remaining:
                break
            selected[user_index] = dict(full_messages[user_index])
            selected[assistant_index] = dict(full_messages[assistant_index])
            remaining -= turn_tokens
            cursor -= 2

    request = [selected[index] for index in sorted(selected)]
    request_tokens = estimate_messages_tokens(request, chars_per_token)
    return request, {
        "full_messages": len(full_messages),
        "request_messages": len(request),
        "full_estimated_tokens": full_tokens,
        "request_estimated_tokens": request_tokens,
        "dropped_messages": len(full_messages) - len(request),
        "truncated_messages": truncated_count,
        "budget": budget,
    }


def calc_max_token(messages, max_tokens):
    return max(0, int(max_tokens) - estimate_messages_tokens(messages))


def _request_payload(
    messages, model, model_config_dict, new_client, reserved_output_tokens
):
    base_url = str(getattr(new_client, "base_url", ""))
    is_gemini = model.startswith("gemini") or "generativelanguage.googleapis.com" in base_url
    is_huggingface_router = "router.huggingface.co" in base_url
    max_tokens = min(
        int(model_config_dict.get("max_tokens", reserved_output_tokens)),
        int(reserved_output_tokens),
    )
    common = {
        "model": model,
        "messages": messages,
        "max_tokens": max_tokens,
        "temperature": model_config_dict.get("temperature", 0.1),
        "stream": False,
    }
    if is_gemini or is_huggingface_router:
        return common
    return {
        **common,
        "top_p": model_config_dict.get("top_p", 1.0),
        "n": model_config_dict.get("n", 1),
        "stream": model_config_dict.get("stream", False),
        "frequency_penalty": model_config_dict.get("frequency_penalty", 0.0),
        "presence_penalty": model_config_dict.get("presence_penalty", 0.0),
        "logit_bias": model_config_dict.get("logit_bias", {}),
    }


def _recover_chat_completion(raw_response, parse_error):
    """Recover a complete ChatCompletion from a response with trailing JSON.

    Some OpenAI-compatible providers occasionally concatenate a valid response
    with another JSON value. We accept only a fully validated ChatCompletion;
    truncated or otherwise invalid payloads still raise and enter normal retry.
    """
    text = raw_response.http_response.text
    decoder = json.JSONDecoder()
    offset = 0
    decoded_values = 0
    while offset < len(text):
        while offset < len(text) and text[offset].isspace():
            offset += 1
        if offset >= len(text):
            break
        try:
            value, end = decoder.raw_decode(text, offset)
        except json.JSONDecodeError:
            break
        decoded_values += 1
        try:
            completion = ChatCompletion.model_validate(value)
        except Exception:
            offset = end
            continue
        if completion.choices:
            return completion, {
                "kind": "validated_chat_completion_from_malformed_json",
                "decoded_values": decoded_values,
                "trailing_chars": len(text[end:].strip()),
            }
        offset = end
    raise parse_error


def _create_chat_completion(completions, json_data):
    raw_api = getattr(completions, "with_raw_response", None)
    if raw_api is None:
        return completions.create(**json_data), None
    raw_response = raw_api.create(**json_data)
    if not isinstance(getattr(raw_response, "http_response", None), httpx.Response):
        return completions.create(**json_data), None
    try:
        return raw_response.parse(), None
    except json.JSONDecodeError as error:
        return _recover_chat_completion(raw_response, error)


from role_aware.audit_trace import observe_provider_call, CALL_CONTEXT


def chat_completion_request(messages, model, new_client, model_config_dict: Dict = None):
    if model_config_dict is None:
        model_config_dict = {
            "temperature": 0.1,
            "top_p": 1.0,
            "n": 1,
            "stream": False,
            "frequency_penalty": 0.0,
            "presence_penalty": 0.0,
            "logit_bias": {},
            "max_tokens": CHAT_CONTEXT_RESERVED_OUTPUT_TOKENS,
        }

    last_error = None
    for attempt in range(CHAT_MAX_RETRY_TIMES):
        budget = max(
            CHAT_CONTEXT_MIN_INPUT_TOKENS,
            int(
                CHAT_CONTEXT_MAX_INPUT_TOKENS
                * (CHAT_CONTEXT_RETRY_SHRINK_FACTOR ** attempt)
            ),
        )
        request_messages, context_stats = build_context_window(
            messages,
            max_input_tokens=budget,
        )
        json_data = _request_payload(
            request_messages,
            model,
            model_config_dict,
            new_client,
            CHAT_CONTEXT_RESERVED_OUTPUT_TOKENS,
        )
        logger.info(
            "[Model Query Context] model=%s attempt=%s/%s stats=%s",
            model,
            attempt + 1,
            CHAT_MAX_RETRY_TIMES,
            context_stats,
        )
        logger.info("[Model Query] model=%s messages=%s", model, request_messages)
        print(
            "[Model Query] model={} input_tokens~{} output_tokens={} "
            "total_tokens~{} budget={} messages={}/{}".format(
                model,
                context_stats["request_estimated_tokens"],
                json_data["max_tokens"],
                context_stats["request_estimated_tokens"] + json_data["max_tokens"],
                budget,
                context_stats["request_messages"],
                context_stats["full_messages"],
            )
        )
        try:
            context = CALL_CONTEXT.get()
            size = context[1].get("model_size", 0) if context else 0
            with observe_provider_call(json_data["messages"], model, size) as receipt:
                response, recovery = _create_chat_completion(
                    new_client.chat.completions, json_data
                )
                if recovery is not None:
                    receipt["response_recovery"] = recovery
                    logger.warning(
                        "Recovered a validated ChatCompletion from malformed "
                        "provider JSON: %s",
                        recovery,
                    )
                completion_tokens = response.usage.completion_tokens if response.usage else 0
                prompt_tokens = response.usage.prompt_tokens if response.usage else 0
                total_tokens = response.usage.total_tokens if response.usage else 0
                if total_tokens == 0:
                    total_tokens = prompt_tokens + completion_tokens
                if total_tokens == 0:
                    total_tokens = estimate_text_tokens(
                        response.choices[0].message.content
                    )
                model_log_and_print(
                    "[Model Query] Token Usage: \nCompletion Tokens: {} "
                    "\nPrompt Tokens: {} \nTotal Tokens: {}".format(
                        completion_tokens, prompt_tokens, total_tokens
                    )
                )
                provider_total = ((response.usage.total_tokens or 0) or
                                  (response.usage.prompt_tokens or 0) + (response.usage.completion_tokens or 0)) if response.usage else 0
                receipt["tokens"] = provider_total if provider_total else None
                receipt["estimated_output_tokens"] = total_tokens if not provider_total else None
                receipt["usage_source"] = "provider" if provider_total else "unavailable_estimated_output_only"
                return response, total_tokens
        except Exception as error:
            last_error = error
            model_log_and_print(
                "[Model Query: ChatCompletion] model={} attempt={}/{} "
                "budget={} query failed: {}".format(
                    model,
                    attempt + 1,
                    CHAT_MAX_RETRY_TIMES,
                    budget,
                    error,
                )
            )
            if attempt + 1 < CHAT_MAX_RETRY_TIMES:
                wait_seconds = min(
                    CHAT_RETRY_WAIT_MAX,
                    CHAT_RETRY_WAIT_MIN * (2 ** attempt),
                )
                time.sleep(wait_seconds)

    raise last_error
