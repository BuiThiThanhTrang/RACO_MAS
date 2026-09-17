from __future__ import annotations

import hashlib
import json
import math
import os
import time

import torch


class AuxiliaryRewardScorer:
    """Lazy Skywork scorer supporting local Transformers and a remote API."""

    def __init__(self, config):
        self.config = dict(config or {})
        self.model_name = self.config.get(
            "model", "Skywork/Skywork-Reward-V2-Llama-3.1-8B"
        )
        self.max_length = int(self.config.get("max_length", 4096))
        self.backend = str(self.config.get("backend", "transformers_local")).lower()
        self.device_map = self.config.get("device_map", "auto")
        self.quantization = str(self.config.get("quantization", "none")).lower()
        self.timeout_seconds = float(self.config.get("timeout_seconds", 60))
        self.max_retries = max(1, int(self.config.get("max_retries", 3)))
        self.cache_enabled = bool(
            (self.config.get("cache") or {}).get("enabled", True)
        )
        self._cache = {}
        self.model = None
        self.tokenizer = None
        self.input_device = None

    def _load(self):
        if self.model is not None:
            return
        from transformers import AutoModelForSequenceClassification, AutoTokenizer

        kwargs = {"device_map": self.device_map, "trust_remote_code": True}
        dtype = str(self.config.get("torch_dtype", "auto")).lower()
        if dtype in {"bf16", "bfloat16"}:
            kwargs["torch_dtype"] = torch.bfloat16
        elif dtype in {"fp16", "float16"}:
            kwargs["torch_dtype"] = torch.float16
        elif dtype in {"fp32", "float32"}:
            kwargs["torch_dtype"] = torch.float32
        elif dtype == "auto":
            kwargs["torch_dtype"] = (
                torch.bfloat16 if torch.cuda.is_available() else torch.float32
            )
        if self.quantization in {"8bit", "int8"}:
            kwargs["load_in_8bit"] = True
        elif self.quantization in {"4bit", "nf4"}:
            kwargs["load_in_4bit"] = True

        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_name, trust_remote_code=True
        )
        self.model = AutoModelForSequenceClassification.from_pretrained(
            self.model_name, num_labels=1, **kwargs
        )
        self.model.eval()
        self.input_device = next(self.model.parameters()).device

    @staticmethod
    def _cache_key(model_name, messages):
        payload = json.dumps(
            {"model": model_name, "messages": messages},
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            default=str,
        )
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()

    @staticmethod
    def _validated_score(value) -> float:
        score = float(value)
        if not math.isfinite(score) or not 0.0 <= score <= 1.0:
            raise ValueError(
                f"Reward score must be finite and in [0, 1], got {value!r}"
            )
        return score

    def _score_local(self, messages) -> float:
        self._load()
        if isinstance(messages, str):
            text = messages
        else:
            text = self.tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=False
            )
        inputs = self.tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            max_length=self.max_length,
        )
        inputs = {key: value.to(self.input_device) for key, value in inputs.items()}
        with torch.no_grad():
            logits = self.model(**inputs).logits.reshape(-1)
        return self._validated_score(
            torch.sigmoid(logits[0].float()).cpu().item()
        )

    def _score_http(self, messages) -> float:
        import requests

        base_url = str(self.config.get("base_url", "")).strip()
        if not base_url:
            base_url = os.getenv(
                self.config.get("base_url_env", "SKYWORK_REWARD_BASE_URL"), ""
            ).strip()
        if not base_url:
            raise RuntimeError(
                "Skywork reward URL is missing; configure base_url or "
                "SKYWORK_REWARD_BASE_URL"
            )
        endpoint = str(self.config.get("endpoint", "/v1/reward"))
        base_url = base_url.rstrip("/")
        if base_url.endswith("/v1") and endpoint.startswith("/v1/"):
            url = base_url + endpoint[len("/v1"):]
        else:
            url = f"{base_url}/{endpoint.lstrip('/')}"

        api_key = str(self.config.get("api_key", "")).strip()
        if not api_key:
            api_key = os.getenv(
                self.config.get("api_key_env", "SKYWORK_REWARD_API_KEY"), ""
            ).strip()
        headers = {"Content-Type": "application/json"}
        if api_key:
            headers["Authorization"] = f"Bearer {api_key}"
        payload = {"model": self.model_name, "messages": messages}

        last_error = None
        for attempt in range(self.max_retries):
            try:
                from role_aware.audit_trace import observe_provider_call
                with observe_provider_call(messages, self.model_name) as receipt:
                    response = requests.post(
                        url,
                        headers=headers,
                        json=payload,
                        timeout=self.timeout_seconds,
                    )
                    response.raise_for_status()
                    data = response.json()
                    usage = data.get("usage") or {}
                    receipt["tokens"] = usage.get("total_tokens")
                    receipt["usage_source"] = "provider" if receipt["tokens"] is not None else "unavailable"
                    return self._validated_score(data["score"])
            except (
                requests.RequestException,
                KeyError,
                TypeError,
                ValueError,
            ) as error:
                last_error = error
                if attempt + 1 < self.max_retries:
                    time.sleep(min(2 ** attempt, 4))
        raise RuntimeError(
            f"Skywork reward request failed: {last_error}"
        ) from last_error

    def __call__(self, messages) -> float:
        key = self._cache_key(self.model_name, messages)
        if self.cache_enabled and key in self._cache:
            return self._cache[key]

        if self.backend in {"http", "remote", "http_reward"}:
            score = self._score_http(messages)
        elif self.backend in {"transformers_local", "local", "transformers"}:
            score = self._score_local(messages)
        else:
            raise ValueError(
                f"Unsupported trajectory reward backend: {self.backend!r}"
            )

        if self.cache_enabled:
            self._cache[key] = score
        return score
