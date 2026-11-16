from __future__ import annotations

import json
import os
import time
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from typing import Any, Callable, Mapping, Protocol

from .config import DecisionModelConfig
from .capabilities import ROUTER_CAPABILITY_CARDS
from .contracts import ModelUsage


# Backward-compatible name for code that imported the old constant. The source
# of truth is now the shared router-facing capability-card module.
ACTION_DESCRIPTIONS = ROUTER_CAPABILITY_CARDS

QUESTION_NAME = "next_action"
QUESTION_INSTRUCTIONS = (
    "Which action should the multi-agent code workflow execute next to maximize "
    "the chance of solving the task within the remaining budget? Base Test outcomes, "
    "whether pass or fail, are incomplete evidence rather than a routing rule; a "
    "Base-Test pass alone is not proof of full correctness. Review is optional rather "
    "than a required procedure. Choose each action, including review or STOP, only "
    "when it has the highest expected value for the current state and remaining budget."
)


@dataclass(frozen=True)
class DecisionModelSpec:
    adapter: str
    checkpoint: str
    base_url: str | None = None
    base_url_env: str | None = None
    api_key_env: str | None = None


@dataclass(frozen=True)
class ResolvedDecisionModel:
    name: str
    provider: str
    adapter: str
    checkpoint: str
    revision: str | None
    base_url: str | None
    api_key_env: str | None
    device: str
    context_limit: int
    timeout_seconds: float


# One registry entry is enough for every model/provider pair that follows an
# existing wire protocol. Config values may override any deployment detail.
DECISION_MODEL_REGISTRY: dict[tuple[str, str], DecisionModelSpec] = {
    ("laya", "local"): DecisionModelSpec(
        adapter="laya",
        checkpoint="convaiinnovations/laya-typed-decisions",
    ),
    ("jev", "typesafe"): DecisionModelSpec(
        adapter="systemone_http",
        checkpoint="jev-1.13.0",
        base_url="https://api.typesafe.ai/v1/systemone",
        api_key_env="TYPESAFE_API_KEY",
    ),
    ("jev", "openrouter"): DecisionModelSpec(
        adapter="systemone_http",
        checkpoint="typesafe/jev-1.13",
        base_url="https://openrouter.ai/api/alpha/decisions",
        api_key_env="OPENROUTER_API_KEY",
    ),
    ("jev", "huggingface"): DecisionModelSpec(
        adapter="hf_decision",
        checkpoint="autotrust/JEV-9B",
    ),
    ("jev", "http"): DecisionModelSpec(
        adapter="systemone_http",
        checkpoint="jev-1.13.0",
        base_url_env="JEV_BASE_URL",
        api_key_env="JEV_API_KEY",
    ),
    ("openai_decisions", "openai"): DecisionModelSpec(
        adapter="openai_decisions",
        checkpoint="gpt-6-luna",
        base_url="https://api.openai.com/v1/decisions",
        api_key_env="OPENAI_API_KEY",
    ),
    ("openai_decisions", "openrouter"): DecisionModelSpec(
        # OpenRouter normalizes Luna to its state/questions Decisions schema,
        # the same wire contract used by the System One HTTP adapter.
        adapter="systemone_http",
        checkpoint="openai/gpt-6-luna-decisions",
        base_url="https://openrouter.ai/api/alpha/decisions",
        api_key_env="OPENROUTER_API_KEY",
    ),
    ("drex", "local"): DecisionModelSpec(
        adapter="systemone_http",
        checkpoint="drex-v1.5",
        base_url="http://127.0.0.1:8000/v1/systemone",
    ),
    ("drex", "http"): DecisionModelSpec(
        adapter="systemone_http",
        checkpoint="drex-v1.5",
        base_url_env="DREX_BASE_URL",
        api_key_env="DREX_API_KEY",
    ),
    ("drex", "modal"): DecisionModelSpec(
        adapter="systemone_http",
        checkpoint="drex-v1.5",
        base_url_env="DREX_BASE_URL",
        api_key_env="DREX_API_KEY",
    ),
}


def registered_decision_models() -> tuple[str, ...]:
    return tuple(
        f"{name}@{provider}" for name, provider in DECISION_MODEL_REGISTRY
    )


def resolve_decision_model(config: DecisionModelConfig) -> ResolvedDecisionModel:
    key = (config.name.strip().lower(), config.provider.strip().lower())
    spec = DECISION_MODEL_REGISTRY.get(key)
    if spec is None:
        known = ", ".join(registered_decision_models())
        raise ValueError(
            f"Unknown decision model/provider {config.name}@{config.provider}. "
            f"Registered pairs: {known}"
        )
    base_url_env = config.base_url_env or spec.base_url_env
    base_url = config.base_url
    if base_url is None and base_url_env:
        base_url = os.getenv(base_url_env) or None
    if base_url is None:
        base_url = spec.base_url
    return ResolvedDecisionModel(
        name=key[0],
        provider=key[1],
        adapter=config.adapter or spec.adapter,
        checkpoint=config.checkpoint or spec.checkpoint,
        revision=config.revision,
        base_url=base_url,
        api_key_env=config.api_key_env or spec.api_key_env,
        device=config.device,
        context_limit=config.context_limit,
        timeout_seconds=config.timeout_seconds,
    )


@dataclass(frozen=True)
class AdapterDecision:
    choice: str | None
    probabilities: Mapping[str, float] | None
    confidence: float | None
    usage: ModelUsage
    raw: Any
    metadata: Mapping[str, Any] = field(default_factory=dict)


class DecisionAdapter(Protocol):
    def decide(self, state: Mapping[str, Any]) -> AdapterDecision: ...


def _usage(payload: Mapping[str, Any] | None) -> ModelUsage:
    data = payload or {}
    return ModelUsage(
        input_tokens=int(data.get("input_tokens", data.get("prompt_tokens", 0)) or 0),
        output_tokens=int(
            data.get("output_tokens", data.get("completion_tokens", 0)) or 0
        ),
        cost_usd=float(data.get("cost_usd", data.get("cost", 0.0)) or 0.0),
    )


def _probabilities(value: Any) -> dict[str, float] | None:
    if isinstance(value, Mapping):
        return {str(key): float(probability) for key, probability in value.items()}
    if isinstance(value, list):
        normalized: dict[str, float] = {}
        for item in value:
            if isinstance(item, Mapping):
                key = item.get("value", item.get("label"))
                probability = item.get("probability")
                if key is not None and probability is not None:
                    normalized[str(key)] = float(probability)
        return normalized or None
    return None


def _systemone_question() -> dict[str, Any]:
    return {
        QUESTION_NAME: {
            "type": "choice",
            "instructions": QUESTION_INSTRUCTIONS,
            "criteria": dict(ACTION_DESCRIPTIONS),
        }
    }


def _answer_from_systemone(result: Mapping[str, Any]) -> AdapterDecision:
    answers = result.get("answers", {})
    answer = answers.get(QUESTION_NAME, {}) if isinstance(answers, Mapping) else {}
    if not isinstance(answer, Mapping):
        answer = {}
    usage_payload = result.get("usage", {})
    if not isinstance(usage_payload, Mapping):
        usage_payload = {}
    return AdapterDecision(
        choice=str(answer["choice"]).strip() if answer.get("choice") is not None else None,
        probabilities=_probabilities(answer.get("probabilities")),
        confidence=(
            float(answer.get("answer_confidence", answer.get("confidence")))
            if answer.get("answer_confidence", answer.get("confidence")) is not None
            else None
        ),
        usage=_usage(usage_payload),
        raw=result,
        metadata={
            "resolved_model": result.get("model"),
            "refusal": answer.get("type") == "refusal",
            "input_truncated": usage_payload.get("truncated"),
            "state_tokens": usage_payload.get("state_tokens"),
            "state_tokens_dropped": usage_payload.get("state_tokens_dropped"),
        },
    )


class LayaAdapter:
    def __init__(self, model: ResolvedDecisionModel, agent=None) -> None:
        self.model = model
        if agent is None:
            device = None if model.device == "auto" else model.device
            agent = _load_laya(
                model.checkpoint,
                device=device,
                revision=model.revision,
            )
        self.agent = agent

    def decide(self, state: Mapping[str, Any]) -> AdapterDecision:
        result = self.agent.predict(
            dict(state),
            _systemone_question(),
            max_len=self.model.context_limit,
        )
        return _answer_from_systemone(result)


def _load_laya(
    checkpoint: str,
    *,
    device: str | None,
    revision: str | None = None,
):
    """Load Laya with a scoped ModernBERT workaround on Windows.

    Some Windows CPU builds of PyTorch crash natively while Transformers performs
    ModernBERT's throwaway weight-initialization pass. Laya immediately replaces
    every parameter from a strict checkpoint, so initialization is unnecessary.
    Keep the monkeypatch scoped to ``laya.load`` and restore the original method
    before any other model can be constructed in this process.
    """

    if os.name != "nt":
        import laya

        return laya.load(checkpoint, device=device, revision=revision)

    import transformers.modeling_utils as modeling_utils

    original = modeling_utils.PreTrainedModel.initialize_weights

    def skip_throwaway_initialization(self) -> None:
        return None

    modeling_utils.PreTrainedModel.initialize_weights = skip_throwaway_initialization
    try:
        import laya

        return laya.load(checkpoint, device=device, revision=revision)
    finally:
        modeling_utils.PreTrainedModel.initialize_weights = original


class HuggingFaceDecisionAdapter:
    def __init__(self, model: ResolvedDecisionModel, decision_pipeline=None) -> None:
        self.model = model
        if decision_pipeline is None:
            try:
                from transformers import pipeline

                kwargs: dict[str, Any] = {}
                if model.device == "auto":
                    kwargs["device_map"] = "auto"
                elif model.device == "cpu":
                    kwargs["device"] = -1
                else:
                    kwargs["device"] = model.device
                decision_pipeline = pipeline(
                    "decision", model=model.checkpoint, **kwargs
                )
            except Exception as exc:
                raise RuntimeError(
                    "The Hugging Face decision adapter requires a Transformers "
                    "version that provides pipeline('decision'). Install the "
                    "project's decision-local optional dependency."
                ) from exc
        self.pipeline = decision_pipeline

    def decide(self, state: Mapping[str, Any]) -> AdapterDecision:
        result = self.pipeline(
            {"state": dict(state), "questions": _systemone_question()}
        )
        return _answer_from_systemone(result)


class DecisionHTTPError(RuntimeError):
    def __init__(self, status: int, url: str, body: str) -> None:
        super().__init__(f"Decision endpoint returned HTTP {status} for {url}: {body}")
        self.status = status
        self.url = url
        self.body = body


Transport = Callable[[str, Mapping[str, Any], Mapping[str, str], float], Mapping[str, Any]]


def post_json(
    url: str,
    payload: Mapping[str, Any],
    headers: Mapping[str, str],
    timeout_seconds: float,
) -> Mapping[str, Any]:
    request = urllib.request.Request(
        url,
        data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
        headers={"Content-Type": "application/json", **dict(headers)},
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout_seconds) as response:
            return json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        body = exc.read().decode("utf-8", errors="replace")
        raise DecisionHTTPError(exc.code, url, body) from exc


class SystemOneHTTPAdapter:
    """Shared wire adapter for TypeSafe JEV and DREX-compatible servers."""

    def __init__(
        self,
        model: ResolvedDecisionModel,
        transport: Transport = post_json,
    ) -> None:
        if not model.base_url:
            raise ValueError(
                f"Missing endpoint URL for {model.name}@{model.provider}"
            )
        self.api_key = ""
        if model.api_key_env:
            self.api_key = os.getenv(model.api_key_env, "")
            if not self.api_key:
                raise ValueError(f"Missing API key in {model.api_key_env}")
        self.model = model
        self.transport = transport

    def decide(self, state: Mapping[str, Any]) -> AdapterDecision:
        headers: dict[str, str] = {}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        result = self.transport(
            self.model.base_url or "",
            {
                "model": self.model.checkpoint,
                "state": dict(state),
                "questions": _systemone_question(),
            },
            headers,
            self.model.timeout_seconds,
        )
        return _answer_from_systemone(result)


class OpenAIDecisionsAdapter:
    def __init__(
        self,
        model: ResolvedDecisionModel,
        transport: Transport = post_json,
    ) -> None:
        if not model.base_url:
            raise ValueError("Missing OpenAI Decisions endpoint URL")
        self.model = model
        self.transport = transport

    def decide(self, state: Mapping[str, Any]) -> AdapterDecision:
        api_key_env = self.model.api_key_env or "OPENAI_API_KEY"
        api_key = os.getenv(api_key_env, "")
        if not api_key:
            raise ValueError(f"Missing API key in {api_key_env}")
        result = self.transport(
            self.model.base_url or "",
            {
                "model": self.model.checkpoint,
                "input": json.dumps(state, ensure_ascii=False),
                "questions": [
                    {
                        "type": "choice",
                        "name": QUESTION_NAME,
                        "instructions": QUESTION_INSTRUCTIONS,
                        "choices": [
                            {"value": action, "description": description}
                            for action, description in ACTION_DESCRIPTIONS.items()
                        ],
                    }
                ],
            },
            {"Authorization": f"Bearer {api_key}"},
            self.model.timeout_seconds,
        )
        answers = result.get("answers", [])
        answer = answers[0] if isinstance(answers, list) and answers else {}
        if not isinstance(answer, Mapping):
            answer = {}
        return AdapterDecision(
            choice=(
                str(answer["choice"]).strip()
                if answer.get("choice") is not None
                else None
            ),
            probabilities=_probabilities(answer.get("probabilities")),
            confidence=(
                float(answer["confidence"])
                if answer.get("confidence") is not None
                else None
            ),
            usage=_usage(result.get("usage") if isinstance(result, Mapping) else None),
            raw=result,
            metadata={
                "resolved_model": result.get("model"),
                "refusal": answer.get("type") == "refusal",
            },
        )


def build_decision_adapter(
    config: DecisionModelConfig,
    *,
    agent=None,
    decision_pipeline=None,
    transport: Transport = post_json,
) -> tuple[ResolvedDecisionModel, DecisionAdapter]:
    model = resolve_decision_model(config)
    if model.adapter == "laya":
        adapter: DecisionAdapter = LayaAdapter(model, agent=agent)
    elif model.adapter == "hf_decision":
        adapter = HuggingFaceDecisionAdapter(
            model, decision_pipeline=decision_pipeline
        )
    elif model.adapter == "systemone_http":
        adapter = SystemOneHTTPAdapter(model, transport=transport)
    elif model.adapter == "openai_decisions":
        adapter = OpenAIDecisionsAdapter(model, transport=transport)
    else:
        raise ValueError(f"Unsupported decision adapter: {model.adapter}")
    return model, adapter
