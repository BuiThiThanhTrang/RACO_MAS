import os
from pathlib import Path
import yaml


class LazyProviderClient:
    """Optional clients must not require credentials during package import."""
    def __init__(self, factory):
        self._factory = factory
        self._client = None

    def __getattr__(self, name):
        if self._client is None:
            self._client = self._factory()
        return getattr(self._client, name)


class APIConfig:
    def __init__(self, config_path="config/global.yaml"):
        self.global_config = yaml.safe_load(Path(config_path).read_text(encoding="utf-8")) or {}
        keys = self.global_config.get("api_keys") or {}
        self._config = {
            "openai": {"openai_api_key": keys.get("openai_api_key"),
                       "openai_base_url": keys.get("openai_base_url")},
            "retry_times": self.global_config.get("max_retry_times") or 10,
            "weight_path": self.global_config.get("model_weight_path"),
        }

    def get(self, provider):
        return self._config.get(provider, {})

    def provider_settings(self, provider, url=None):
        defaults = {
            "openai": ("https://api.openai.com/v1", "OPENAI_API_KEY"),
            "openrouter": ("https://openrouter.ai/api/v1", "OPENROUTER_API_KEY"),
            "huggingface": ("https://router.huggingface.co/v1", "HF_TOKEN"),
            "local": (None, "LOCAL_API_KEY"),
        }
        if provider not in defaults:
            raise ValueError(f"Unsupported model provider: {provider}")
        base_url, key_env = defaults[provider]
        custom = (self.global_config.get("providers") or {}).get(provider) or {}
        key_env = custom.get("api_key_env") or key_env
        base_url = url or custom.get("base_url") or base_url
        key = os.environ.get(key_env)
        if provider == "openai":
            legacy = self.get("openai")
            base_url = url or custom.get("base_url") or legacy.get("openai_base_url") or os.environ.get("OPENAI_BASE_URL") or base_url
            key = key or legacy.get("openai_api_key")
        if provider == "local":
            key = key or "none"
        if not base_url or base_url in ("http://", "https://"):
            raise ValueError(f"Configure a valid base_url for provider {provider}")
        return {"base_url": base_url, "api_key": key, "api_key_env": key_env}

    def create_client(self, provider, url=None):
        from openai import OpenAI
        settings = self.provider_settings(provider, url)
        if not settings["api_key"]:
            raise ValueError(f"Missing environment variable {settings['api_key_env']} for provider {provider}")
        return OpenAI(api_key=settings["api_key"], base_url=settings["base_url"])

    def global_openai_client(self):
        # Kept for legacy image/audio conversion. Text-only agents do not use it.
        return LazyProviderClient(lambda: self.create_client("openai"))


api_config = APIConfig()
