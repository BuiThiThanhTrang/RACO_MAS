from __future__ import annotations

import os
import time
from abc import ABC, abstractmethod
from collections.abc import Callable, Iterable
from typing import Any

from .config import ModelConfig
from .contracts import ModelResponse, ModelUsage


class ModelClient(ABC):
    @abstractmethod
    def generate(self, messages: list[dict[str, str]]) -> ModelResponse:
        raise NotImplementedError


class ReplayModelClient(ModelClient):
    """Deterministic client used by unit and integration tests."""

    def __init__(self, responses: Iterable[str] | Callable[[list[dict[str, str]]], str]):
        self._callable = responses if callable(responses) else None
        self._responses = iter(()) if callable(responses) else iter(responses)

    def generate(self, messages: list[dict[str, str]]) -> ModelResponse:
        text = self._callable(messages) if self._callable else next(self._responses)
        return ModelResponse(
            text=text,
            usage=ModelUsage(input_tokens=1, output_tokens=1),
            model_id="replay",
        )


class OpenAICompatibleClient(ModelClient):
    def __init__(self, config: ModelConfig) -> None:
        from openai import OpenAI

        api_key = os.getenv(config.api_key_env or "OPENAI_API_KEY", "")
        if not api_key:
            raise ValueError(f"Missing API key in {config.api_key_env or 'OPENAI_API_KEY'}")
        self.config = config
        self.client = OpenAI(api_key=api_key, base_url=config.base_url)

    def generate(self, messages: list[dict[str, str]]) -> ModelResponse:
        started = time.perf_counter()
        response = self.client.chat.completions.create(
            model=self.config.checkpoint,
            messages=messages,
            temperature=self.config.temperature,
            top_p=self.config.top_p,
            max_tokens=self.config.max_output_tokens,
        )
        latency_ms = (time.perf_counter() - started) * 1000.0
        usage = response.usage
        return ModelResponse(
            text=response.choices[0].message.content or "",
            usage=ModelUsage(
                input_tokens=int(getattr(usage, "prompt_tokens", 0) or 0),
                output_tokens=int(getattr(usage, "completion_tokens", 0) or 0),
            ),
            latency_ms=latency_ms,
            model_id=self.config.checkpoint,
        )


class TransformersClient(ModelClient):
    def __init__(self, config: ModelConfig) -> None:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.config = config
        self.torch = torch
        self.tokenizer = AutoTokenizer.from_pretrained(config.checkpoint)
        dtype = torch.float16 if torch.cuda.is_available() else torch.float32
        self.model = AutoModelForCausalLM.from_pretrained(
            config.checkpoint,
            torch_dtype=dtype,
            device_map=config.device,
        )

    def generate(self, messages: list[dict[str, str]]) -> ModelResponse:
        torch = self.torch
        text = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
        inputs = self.tokenizer([text], return_tensors="pt").to(self.model.device)
        started = time.perf_counter()
        with torch.inference_mode():
            output_ids = self.model.generate(
                **inputs,
                max_new_tokens=self.config.max_output_tokens,
                do_sample=self.config.temperature > 0.0,
                temperature=max(self.config.temperature, 1e-6),
                top_p=self.config.top_p,
            )
        latency_ms = (time.perf_counter() - started) * 1000.0
        generated = output_ids[:, inputs.input_ids.shape[1] :]
        return ModelResponse(
            text=self.tokenizer.batch_decode(generated, skip_special_tokens=True)[0],
            usage=ModelUsage(
                input_tokens=int(inputs.input_ids.numel()),
                output_tokens=int(generated.numel()),
            ),
            latency_ms=latency_ms,
            model_id=self.config.checkpoint,
        )


def build_model_client(config: ModelConfig) -> ModelClient:
    backend = config.backend.strip().lower()
    if backend == "openai_compatible":
        return OpenAICompatibleClient(config)
    if backend == "transformers":
        return TransformersClient(config)
    raise ValueError(f"Unsupported model backend: {config.backend!r}")
