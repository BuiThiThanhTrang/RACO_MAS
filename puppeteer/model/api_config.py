import os
import yaml

class APIConfig:
    def __init__(self):
        self._config = self._init_config()
    
    def _resolve_key_value(self, api_key):
        if isinstance(api_key, str) and api_key.startswith("$"):
            return os.getenv(api_key[1:], "")
        if isinstance(api_key, str) and os.getenv(api_key):
            return os.getenv(api_key)
        return api_key

    def _resolve_provider(self, provider_config):
        provider_config = provider_config or {}
        api_key = self._resolve_key_value(provider_config.get("api_key"))
        api_key_env = provider_config.get("api_key_env")
        if api_key_env:
            api_key = os.getenv(api_key_env, api_key)

        headers = provider_config.get("headers", {}) or {}
        resolved_headers = {}
        for key, value in headers.items():
            if isinstance(value, str) and value.startswith("$"):
                resolved_headers[key] = os.getenv(value[1:], "")
            else:
                resolved_headers[key] = value

        return {
            "api_key": api_key,
            "api_key_env": api_key_env,
            "base_url": provider_config.get("base_url"),
            "headers": {k: v for k, v in resolved_headers.items() if v},
        }

    def _init_config(self):
        global_config = yaml.safe_load(open("config/global.yaml", "r"))
        legacy_api_keys = global_config.get("api_keys", {}) or {}
        provider_configs = global_config.get("api_providers", {}) or {}
        providers = {
            name: self._resolve_provider(config)
            for name, config in provider_configs.items()
        }

        if "gemini_openai" in providers and not providers["gemini_openai"].get("api_key"):
            providers["gemini_openai"]["api_key"] = self._resolve_key_value(legacy_api_keys.get("openai_api_key"))
            providers["gemini_openai"]["base_url"] = (
                providers["gemini_openai"].get("base_url")
                or legacy_api_keys.get("openai_base_url")
            )

        if "openai" not in providers:
            providers["openai"] = {
                "api_key": self._resolve_key_value(legacy_api_keys.get("openai_api_key")),
                "api_key_env": None,
                "base_url": legacy_api_keys.get("openai_base_url", None),
                "headers": {},
            }

        key_config = {
            "openai":{
            "openai_api_key": self._resolve_key_value(legacy_api_keys.get("openai_api_key")),
            "openai_base_url": legacy_api_keys.get("openai_base_url", None),
            },
            "providers": providers,
            "retry_times": global_config.get("max_retry_times", 10),
            "weight_path": global_config.get("model_weight_path")
        }
        return key_config

    def get(self, provider: str) -> dict:
        return self._config.get(provider, {})

    def get_provider(self, provider: str) -> dict:
        return self._config.get("providers", {}).get(provider, {})
    
    def global_openai_client(self):
        from openai import OpenAI
        api_key = self._config.get("openai").get("openai_api_key", None)
        base_url = self._config.get("openai").get("openai_base_url", None)
        return OpenAI(api_key=api_key, base_url=base_url)

api_config = APIConfig()
