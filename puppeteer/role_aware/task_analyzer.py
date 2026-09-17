from __future__ import annotations

import os
from typing import Iterable

import torch
from role_aware.audit_trace import observe_provider_call


class TaskAnalyzerRepresentation:
    """Frozen task encoder served through an OpenAI-compatible embedding API."""

    REMOTE_BACKENDS = {
        "openai_compatible_embedding",
        "openai_compatible",
        "remote",
        "http",
    }

    def __init__(self, config):
        self.config = dict(config or {})
        self.primary = dict(self.config.get("primary") or {})
        self.fallback = dict(self.config.get("fallback") or {})
        self.dim = int(self.primary.get("dim", 1024))
        self._client = None
        self._backend = None
        self._active_config = None

    @staticmethod
    def _text(messages) -> str:
        if isinstance(messages, str):
            return messages
        if isinstance(messages, Iterable):
            return "\n".join(
                str(item.get("content", item)) if isinstance(item, dict) else str(item)
                for item in messages
            )
        return str(messages)

    def _load(self) -> None:
        if self._backend is not None:
            return
        configured = [("primary", self.primary), ("fallback", self.fallback)]
        for name, backend_config in configured:
            if not backend_config:
                continue
            backend = str(backend_config.get("backend", "")).lower()
            if backend not in self.REMOTE_BACKENDS:
                if name == "primary":
                    raise ValueError(
                        "Task Analyzer is remote-only; configure primary.backend="
                        "openai_compatible_embedding"
                    )
                continue

            base_url = str(backend_config.get("base_url", "")).strip()
            if not base_url:
                base_url = os.getenv(
                    backend_config.get("base_url_env", "TASK_ANALYZER_BASE_URL"),
                    "",
                ).strip()
            api_key = str(backend_config.get("api_key", "")).strip()
            if not api_key:
                api_key = os.getenv(
                    backend_config.get("api_key_env", "TASK_ANALYZER_API_KEY"),
                    "",
                ).strip()
            if not base_url or not api_key:
                continue

            from openai import OpenAI

            self._client = OpenAI(base_url=base_url, api_key=api_key, max_retries=0)
            self._active_config = backend_config
            self.dim = int(backend_config.get("dim", 1024))
            self._backend = name
            return
        raise RuntimeError(
            "Task Analyzer remote endpoint is unavailable. Configure "
            "TASK_ANALYZER_BASE_URL and TASK_ANALYZER_API_KEY."
        )

    def __call__(self, messages):
        self._load()
        text = self._text(messages)

        model = self._active_config.get("model", "BAAI/bge-large-en-v1.5")
        with observe_provider_call(text, model) as receipt:
            response = self._client.embeddings.create(
                model=model, input=[text], dimensions=self.dim, encoding_format="float")
            usage = getattr(response, "usage", None)
            receipt["tokens"] = getattr(usage, "total_tokens", None)
            receipt["usage_source"] = "provider" if receipt["tokens"] is not None else "unavailable"
        return torch.tensor([response.data[0].embedding], dtype=torch.float32), 0.0
