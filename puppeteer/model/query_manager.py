from typing import List, Dict, Any, Optional, Tuple
import yaml
from model.model_config import  model_registry, ModelConfig
from model.api_config import api_config
from model.model_utils import chat_completion_request, model_log_and_print


class ModelQueryManager:
    def __init__(self):
        self.registry = model_registry
        self.config_manager = api_config
        self.clients = {}
        self.client_errors = {}
        self._setup_clients()
    
    def _setup_clients(self):
        from openai import OpenAI
        for key, config in self.registry.get_all_models().items():
            if config.url:
                self.clients[config.name] = OpenAI(api_key="none", base_url=config.url)
                continue

            if config.provider in {"openai", "openai_compatible"}:
                profile_name = config.api_profile or "openai"
                provider_config = self.config_manager.get_provider(profile_name)
                if not provider_config and profile_name == "openai":
                    legacy_openai = self.config_manager.get("openai")
                    provider_config = {
                        "api_key": legacy_openai.get("openai_api_key"),
                        "api_key_env": None,
                        "base_url": legacy_openai.get("openai_base_url"),
                        "headers": {},
                    }

                if not provider_config.get("api_key"):
                    api_key_env = provider_config.get("api_key_env") or "api key"
                    message = (
                        f"Missing API key for model '{config.name}' "
                        f"(api_model_name='{config.api_model_name}', profile='{profile_name}'). "
                        f"Set environment variable {api_key_env} before running."
                    )
                    self.client_errors[config.name] = message
                    self.client_errors[key] = message
                    continue

                client_kwargs = {
                    "api_key": provider_config.get("api_key"),
                    "base_url": provider_config.get("base_url"),
                }
                headers = provider_config.get("headers") or {}
                if headers:
                    client_kwargs["default_headers"] = headers

                self.clients[config.name] = OpenAI(**client_kwargs)
                if key != config.name:
                    self.clients[key] = self.clients[config.name]
    
    def query(self, model_key: str, messages: List[Dict[str, str]], 
              system_prompt: Optional[str] = None) -> Tuple[str, int]:
        config = self.registry.get_model_config(model_key)
        if not config:
            available_models = ", ".join(self.registry.list_available_models())
            raise ValueError(f"Unknown model: {model_key}. Available models: {available_models}")

        if model_key in self.client_errors or config.name in self.client_errors:
            raise ValueError(self.client_errors.get(model_key) or self.client_errors[config.name])
        
        return self._query_with_config(messages, config, system_prompt)
    
    def _query_with_config(self, messages, config: ModelConfig,  system_prompt=None):
        model_config_dict = {
            "temperature": config.temperature,
            "top_p": 1.0,
            "n": 1,
            "stream": False,
            "frequency_penalty": 0.0,
            "presence_penalty": 0.0,
            "logit_bias": {},
            "max_tokens": config.max_tokens
        }

        if not isinstance(messages, list):
            system_prompt = "You are an assistant" if system_prompt is None else system_prompt
            messages = [
                {'role': 'system', 'content': system_prompt},
                {'role': 'user', 'content': messages}
            ]
        response, total_tokens = chat_completion_request(
            messages=messages,
            model=config.api_model_name,  
            new_client=self.clients.get(config.name),
            model_config_dict=model_config_dict
        )
        
        if isinstance(response, str):
            return response, 1
        
        response_message = response.choices[0].message.content or ""
        return response_message, total_tokens
    
    
    def get_available_models(self) -> List[str]:
        return self.registry.list_available_models()
    
    def get_model_info(self, model_key: str) -> Optional[Dict[str, Any]]:
        config = self.registry.get_model_config(model_key)
        if config:
            return {
                "function_name": config.function_name,
                "api_model_name": config.api_model_name,
                "provider": config.provider,
                "max_tokens": config.max_tokens,
                "description": config.description
            }
        return None

query_manager = ModelQueryManager()
