"""HTTP transport for the reward/state API; no model dependencies."""
import math
import os
import time
from urllib.parse import urlsplit

import requests


class RewardAPIError(RuntimeError):
    pass


_TRUNCATION_MARKER = "\n...[truncated for reward model context]...\n"


def _truncate_middle(content, max_chars):
    if len(content) <= max_chars:
        return content
    if max_chars <= len(_TRUNCATION_MARKER):
        return content[:max_chars]

    retained = max_chars - len(_TRUNCATION_MARKER)
    head = (retained + 1) // 2
    tail = retained - head
    suffix = content[-tail:] if tail else ""
    return content[:head] + _TRUNCATION_MARKER + suffix


def compact_messages(messages, max_chars):
    """Copy and compact a conversation without mutating agent history."""
    compacted = [dict(message) for message in messages]
    total = sum(
        len(message.get("content", ""))
        for message in compacted
        if isinstance(message, dict)
        and isinstance(message.get("content", ""), str)
    )

    while total > max_chars:
        candidates = [
            (len(message.get("content", "")), index)
            for index, message in enumerate(compacted)
            if isinstance(message, dict)
            and isinstance(message.get("content", ""), str)
            and message.get("content")
        ]
        if not candidates:
            break

        _, index = max(candidates)
        content = compacted[index]["content"]
        excess = total - max_chars
        target = max(0, len(content) - excess)
        shortened = _truncate_middle(content, target)
        if len(shortened) >= len(content):
            break
        compacted[index]["content"] = shortened
        total -= len(content) - len(shortened)

    return compacted


def validate_response(data):
    if (
        not isinstance(data, dict)
        or type(data.get("schema_version")) is not int
        or data["schema_version"] != 1
    ):
        raise RewardAPIError("Unsupported reward API response schema")
    state, reward = data.get("last_hidden_state"), data.get("reward")
    if not isinstance(state, list) or len(state) != 8192:
        raise RewardAPIError("Reward API must return 8192 hidden-state values")

    def finite(value):
        return type(value) in (int, float) and math.isfinite(value)

    if not finite(reward) or not all(finite(value) for value in state):
        raise RewardAPIError("Reward API returned non-finite or non-numeric values")
    if (
        type(data.get("input_tokens")) is not int
        or not 1 <= data["input_tokens"] <= 4096
    ):
        raise RewardAPIError("Invalid input token count")
    for field in ("model_revision", "quantization"):
        if not isinstance(data.get(field), str) or not data[field]:
            raise RewardAPIError("Missing model provenance in reward response")
    return state, float(reward)


def _response_detail(response):
    try:
        data = response.json()
    except ValueError:
        return ""
    if not isinstance(data, dict) or not isinstance(data.get("detail"), str):
        return ""
    return data["detail"][:300]


class RewardClient:
    def __init__(self, config, *, session=None, sleep=time.sleep):
        self.url = os.environ.get("REWARD_MODEL_URL") or config.get(
            "endpoint_url", ""
        )
        parsed = urlsplit(self.url)
        if (
            parsed.scheme != "https"
            or not parsed.hostname
            or parsed.username
            or parsed.password
            or parsed.fragment
        ):
            raise ValueError(
                "Set reward_model.endpoint_url or REWARD_MODEL_URL "
                "to an HTTPS endpoint"
            )

        key = os.environ.get("MODAL_KEY")
        secret = os.environ.get("MODAL_SECRET")
        if not key or not secret:
            raise ValueError(
                "Set MODAL_KEY and MODAL_SECRET to Modal Proxy Token credentials"
            )
        self.headers = {"Modal-Key": key, "Modal-Secret": secret}
        self.timeout = (
            float(config.get("connect_timeout_seconds", 10)),
            float(config.get("read_timeout_seconds", 180)),
        )
        self.attempts = config.get("max_attempts", 3)
        self.max_input_chars = config.get("max_input_chars", 8000)
        if type(self.attempts) is not int or not 1 <= self.attempts <= 10:
            raise ValueError("max_attempts must be an integer from 1 to 10")
        if (
            type(self.max_input_chars) is not int
            or not 512 <= self.max_input_chars <= 65536
        ):
            raise ValueError(
                "max_input_chars must be an integer from 512 to 65536"
            )
        if not all(math.isfinite(value) and value > 0 for value in self.timeout):
            raise ValueError("Reward API timeouts must be finite and positive")
        self.session = session or requests.Session()
        self.sleep = sleep

    def score(self, messages):
        outbound = compact_messages(messages, self.max_input_chars)
        sent_chars = sum(len(message.get("content", "")) for message in outbound)

        for attempt in range(self.attempts):
            try:
                response = self.session.post(
                    self.url,
                    headers=self.headers,
                    json={"schema_version": 1, "messages": outbound},
                    timeout=self.timeout,
                    allow_redirects=False,
                )
            except (requests.ConnectionError, requests.Timeout):
                failure = "Reward API connection failed or timed out"
            else:
                try:
                    status = response.status_code
                    if status == 200:
                        try:
                            data = response.json()
                        except ValueError:
                            raise RewardAPIError(
                                "Reward API returned invalid JSON"
                            ) from None
                        return validate_response(data)

                    failure = f"Reward API returned HTTP {status}"
                    detail = _response_detail(response)
                    if detail:
                        failure += f": {detail}"
                    if status == 422:
                        failure += (
                            f" (sent {sent_chars} content characters after "
                            f"the {self.max_input_chars}-character client limit)"
                        )
                    if status not in (408, 429, 500, 502, 503, 504):
                        raise RewardAPIError(failure)
                finally:
                    response.close()

            if attempt + 1 < self.attempts:
                self.sleep(min(2 ** attempt, 10))

        raise RewardAPIError(
            f"{failure}; exhausted {self.attempts} attempts"
        ) from None

    def close(self):
        self.session.close()
