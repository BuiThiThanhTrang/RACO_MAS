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

    def count_tokens(self, messages: list[dict[str, str]]) -> tuple[int, str]:
        # Safe fallback for remote clients whose exact tokenizer is unavailable.
        # The method is surfaced in request audit logs and must not be represented
        # as an exact provider token count.
        text = "\n".join(item.get("content", "") for item in messages)
        return max(1, (len(text.encode("utf-8")) + 3) // 4), "utf8_bytes_estimate_v1"


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
        base_url = (
            os.getenv(config.base_url_env, "")
            if config.base_url_env
            else config.base_url
        )
        if not base_url:
            source = config.base_url_env or "expert_model.base_url"
            raise ValueError(f"Missing API base URL in {source}")
        self.config = config
        self.client = OpenAI(api_key=api_key, base_url=base_url)
        self._local_tokenizer = None
        self._tokenizer_checked = False

    def count_tokens(self, messages: list[dict[str, str]]) -> tuple[int, str]:
        if not self._tokenizer_checked:
            self._tokenizer_checked = True
            try:
                from transformers import AutoTokenizer

                self._local_tokenizer = AutoTokenizer.from_pretrained(
                    self.config.checkpoint,
                    local_files_only=True,
                )
            except Exception:
                self._local_tokenizer = None
        if self._local_tokenizer is None:
            return super().count_tokens(messages)
        text = self._local_tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
        return (
            len(self._local_tokenizer.encode(text, add_special_tokens=False)),
            "checkpoint_tokenizer_exact_v1",
        )

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
        dtype_name = config.dtype.strip().lower()
        if dtype_name == "auto":
            dtype = torch.float16 if torch.cuda.is_available() else torch.float32
        else:
            dtypes = {
                "float32": torch.float32,
                "float16": torch.float16,
                "bfloat16": torch.bfloat16,
            }
            if dtype_name not in dtypes:
                raise ValueError(
                    "Unsupported dtype. Use auto, float32, float16 or bfloat16."
                )
            dtype = dtypes[dtype_name]
        automatic_maps = {"auto", "balanced", "balanced_low_0", "sequential"}
        device_map = (
            config.device
            if config.device in automatic_maps
            else {"": config.device}
        )
        self.model = AutoModelForCausalLM.from_pretrained(
            config.checkpoint,
            torch_dtype=dtype,
            device_map=device_map,
        )

    def count_tokens(self, messages: list[dict[str, str]]) -> tuple[int, str]:
        text = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
        return (
            len(self.tokenizer.encode(text, add_special_tokens=False)),
            "checkpoint_tokenizer_exact_v1",
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
