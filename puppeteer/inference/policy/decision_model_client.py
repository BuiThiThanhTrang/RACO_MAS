from __future__ import annotations

import os
import time
import logging
from dataclasses import dataclass
from typing import Any, Mapping

import httpx


LOGGER = logging.getLogger(__name__)


class DecisionModelError(RuntimeError):
    """Base error raised by a typed decision-model provider."""


class DecisionModelTransportError(DecisionModelError):
    """The provider could not be reached or rejected the request."""


class DecisionModelResponseError(DecisionModelError):
    """The provider returned a response that violates the decision contract."""


@dataclass(frozen=True)
class DecisionResult:
    choice: str
    confidence: float | None
    probabilities: Mapping[str, float]
    input_tokens: int
    model: str
    model_version: str | None
    latency_ms: float


@dataclass(frozen=True)
class DecisionJudgment:
    """One typed Decisions API response before policy-specific interpretation."""

    answers: Mapping[str, Any]
    input_tokens: int
    model: str
    model_version: str | None
    latency_ms: float


def _required_env(name: str) -> str:
    value = os.getenv(name, "").strip()
    if not value:
        raise ValueError(f"Required environment variable {name} is not set")
    return value


class SystemOneDecisionClient:
    """HTTP client for OpenRouter Decisions and compatible System One APIs."""

    RETRYABLE_STATUS_CODES = {
        408,
        409,
        425,
        429,
        500,
        502,
        503,
        504,
        524,
        529,
    }

    def __init__(self, config: Mapping[str, Any], http_client=None) -> None:
        self.config = dict(config or {})
        self.provider = str(self.config.get("provider", "jev")).strip().lower()
        if self.provider not in {"openrouter", "jev", "cloudflare", "systemone"}:
            raise ValueError(
                "decision.provider must be openrouter, jev, cloudflare, or systemone"
            )
        self.model = str(self.config.get("model", "")).strip()
        if not self.model:
            raise ValueError("decision.model is required")
        self.timeout_seconds = float(self.config.get("timeout_seconds", 30.0))
        self.max_retries = int(self.config.get("max_retries", 2))
        self.retry_backoff_seconds = float(
            self.config.get("retry_backoff_seconds", 1.0)
        )
        self.max_retry_delay_seconds = float(
            self.config.get("max_retry_delay_seconds", 30.0)
        )
        if (
            self.timeout_seconds <= 0
            or self.max_retries < 0
            or self.retry_backoff_seconds < 0
            or self.max_retry_delay_seconds < 0
        ):
            raise ValueError("Decision timeout must be positive and retries nonnegative")
        self.endpoint, self.api_key = self._resolve_endpoint_and_key()
        self._client = http_client or httpx.Client(timeout=self.timeout_seconds)

    def _resolve_endpoint_and_key(self) -> tuple[str, str | None]:
        endpoint = str(self.config.get("endpoint", "")).strip()
        endpoint_env = str(self.config.get("endpoint_env", "")).strip()
        if endpoint_env:
            endpoint = _required_env(endpoint_env)

        api_key_env = str(self.config.get("api_key_env", "")).strip()
        api_key = _required_env(api_key_env) if api_key_env else None

        if self.provider == "openrouter":
            endpoint = endpoint or "https://openrouter.ai/api/alpha/decisions"
            api_key = api_key or _required_env("OPENROUTER_API_KEY")
        elif self.provider == "jev":
            endpoint = endpoint or "https://api.typesafe.ai/v1/systemone"
            api_key = api_key or _required_env("JEV_API_KEY")
        elif self.provider == "cloudflare":
            account_env = str(
                self.config.get("account_id_env", "CLOUDFLARE_ACCOUNT_ID")
            )
            account_id = _required_env(account_env)
            model_path = str(
                self.config.get("model_path", f"@cf/cloudflare/{self.model}")
            ).strip()
            endpoint = endpoint or (
                "https://api.cloudflare.com/client/v4/accounts/"
                f"{account_id}/ai/run/{model_path}"
            )
            api_key = api_key or _required_env("CLOUDFLARE_AUTH_TOKEN")
        else:
            if not endpoint:
                base_url = str(self.config.get("base_url", "")).rstrip("/")
                base_url_env = str(self.config.get("base_url_env", "")).strip()
                if base_url_env:
                    base_url = _required_env(base_url_env).rstrip("/")
                if not base_url:
                    raise ValueError(
                        "systemone provider requires decision.endpoint or decision.base_url"
                    )
                endpoint = f"{base_url}/v1/systemone"
        return endpoint, api_key

    def _headers(self) -> dict[str, str]:
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        if self.provider == "openrouter":
            site_url = os.getenv("OPENROUTER_SITE_URL", "").strip()
            app_name = os.getenv("OPENROUTER_APP_NAME", "").strip()
            if site_url:
                headers["HTTP-Referer"] = site_url
            if app_name:
                headers["X-OpenRouter-Title"] = app_name
        for name, value in (self.config.get("headers") or {}).items():
            rendered = str(value)
            if rendered.startswith("$"):
                rendered = os.getenv(rendered[1:], "")
            if rendered:
                headers[str(name)] = rendered
        return headers

    def _retry_delay(self, response: httpx.Response | None, attempt: int) -> float:
        if response is not None:
            retry_after = response.headers.get("Retry-After", "").strip()
            if retry_after:
                try:
                    return min(
                        max(float(retry_after), 0.0), self.max_retry_delay_seconds
                    )
                except ValueError:
                    pass
        return min(
            self.retry_backoff_seconds * (2**attempt),
            self.max_retry_delay_seconds,
        )

    @staticmethod
    def _error_detail(response: httpx.Response) -> str:
        try:
            body = response.json()
        except ValueError:
            body = None
        if isinstance(body, Mapping):
            error = body.get("error")
            if isinstance(error, Mapping):
                message = error.get("message")
                if isinstance(message, str) and message.strip():
                    return message.strip()[:500]
            if isinstance(error, str) and error.strip():
                return error.strip()[:500]
        return ""

    def _wait_before_retry(
        self,
        response: httpx.Response | None,
        attempt: int,
        reason: str,
    ) -> None:
        delay = self._retry_delay(response, attempt)
        LOGGER.warning(
            "Decision request retry %s/%s after %s (waiting %.1fs)",
            attempt + 1,
            self.max_retries,
            reason,
            delay,
        )
        if delay > 0:
            time.sleep(delay)

    def _post(self, payload: Mapping[str, Any]) -> tuple[dict, float]:
        start = time.perf_counter()
        for attempt in range(self.max_retries + 1):
            try:
                response = self._client.post(
                    self.endpoint,
                    headers=self._headers(),
                    json=dict(payload),
                    timeout=self.timeout_seconds,
                )
            except httpx.HTTPError as error:
                if attempt >= self.max_retries:
                    raise DecisionModelTransportError(
                        f"{self.provider} request failed: {type(error).__name__}"
                    ) from error
                self._wait_before_retry(
                    None, attempt, f"transport error {type(error).__name__}"
                )
                continue
            if response.status_code < 400:
                try:
                    body = response.json()
                except ValueError as error:
                    raise DecisionModelResponseError(
                        f"{self.provider} returned non-JSON content"
                    ) from error
                if not isinstance(body, dict):
                    raise DecisionModelResponseError(
                        f"{self.provider} returned a non-object response"
                    )
                return body, (time.perf_counter() - start) * 1000.0
            if (
                response.status_code in self.RETRYABLE_STATUS_CODES
                and attempt < self.max_retries
            ):
                self._wait_before_retry(
                    response, attempt, f"HTTP {response.status_code}"
                )
                continue
            detail = self._error_detail(response)
            suffix = f": {detail}" if detail else ""
            raise DecisionModelTransportError(
                f"{self.provider} request failed with HTTP "
                f"{response.status_code}{suffix} (model={self.model})"
            )
        raise AssertionError("unreachable decision-model retry state")

    @staticmethod
    def _unwrap(body: Mapping[str, Any]) -> Mapping[str, Any]:
        if "success" in body and body.get("success") is False:
            raise DecisionModelTransportError("Cloudflare decision request was unsuccessful")
        result = body.get("result")
        return result if isinstance(result, Mapping) else body

    @staticmethod
    def _selected_value(answer: Any) -> tuple[str, Mapping[str, Any]]:
        if isinstance(answer, str):
            return answer, {}
        if not isinstance(answer, Mapping):
            raise DecisionModelResponseError("Decision answer must be a string or object")
        for key in ("choice", "value", "answer", "label"):
            value = answer.get(key)
            if isinstance(value, str) and value.strip():
                return value.strip(), answer
        raise DecisionModelResponseError("Decision answer contains no selected choice")

    def decide(
        self,
        state: Mapping[str, Any],
        questions: Mapping[str, Any],
        question_name: str,
    ) -> DecisionResult:
        judgment = self.judge(state, questions)
        answers = judgment.answers
        if question_name not in answers:
            raise DecisionModelResponseError(
                f"Decision response has no answer for {question_name!r}"
            )
        choice, answer = self._selected_value(answers[question_name])
        raw_probabilities = (
            answer.get("probabilities")
            or answer.get("distribution")
            or answer.get("scores")
            or {}
        )
        probabilities = {
            str(key): float(value)
            for key, value in raw_probabilities.items()
            if isinstance(value, (int, float))
        } if isinstance(raw_probabilities, Mapping) else {}
        raw_confidence = answer.get("confidence")
        confidence = (
            float(raw_confidence)
            if isinstance(raw_confidence, (int, float))
            else probabilities.get(choice)
        )
        return DecisionResult(
            choice=choice,
            confidence=confidence,
            probabilities=probabilities,
            input_tokens=judgment.input_tokens,
            model=judgment.model,
            model_version=judgment.model_version,
            latency_ms=judgment.latency_ms,
        )

    def judge(
        self,
        state: Mapping[str, Any],
        questions: Mapping[str, Any],
    ) -> DecisionJudgment:
        """Return all typed judgments so routing can use atomic semantic probes."""
        request = {
            "model": self.model,
            "state": dict(state),
            "questions": dict(questions),
        }
        body, latency_ms = self._post(request)
        payload = self._unwrap(body)
        answers = payload.get("answers")
        if not isinstance(answers, Mapping):
            raise DecisionModelResponseError(
                "Decision response has no typed answers"
            )
        usage = payload.get("usage") if isinstance(payload.get("usage"), Mapping) else {}
        input_tokens = int(
            usage.get("input_tokens", usage.get("prompt_tokens", 0)) or 0
        )
        return DecisionJudgment(
            answers={str(key): value for key, value in answers.items()},
            input_tokens=input_tokens,
            model=str(payload.get("model", self.model)),
            model_version=(
                str(payload["model_version"])
                if payload.get("model_version") is not None
                else None
            ),
            latency_ms=latency_ms,
        )
