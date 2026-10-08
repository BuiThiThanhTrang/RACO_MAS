from typing import List, Dict, Any, Optional, Tuple
import json
import yaml
from model.model_config import  model_registry, ModelConfig
from model.api_config import api_config
from model.model_utils import chat_completion_request, model_log_and_print


class StructuredResponseError(ValueError):
    pass


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
                self.clients[config.name] = OpenAI(api_key="none", base_url=config.url, max_retries=0)
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
                    "max_retries": 0,
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

    @staticmethod
    def _apply_model_request_options(
        config: ModelConfig,
        model_config_dict: Dict[str, Any],
        reasoning_effort: Optional[str] = None,
    ) -> None:
        """Translate model capabilities into provider-compatible request fields."""
        extra_body: Dict[str, Any] = dict(config.request_extra_body or {})
        effort = reasoning_effort or config.reasoning_effort
        if effort:
            if config.reasoning_parameter == "reasoning":
                extra_body["reasoning"] = {"effort": effort}
            else:
                model_config_dict["reasoning_effort"] = effort
        if config.provider_require_parameters:
            extra_body["provider"] = {"require_parameters": True}
        if extra_body:
            model_config_dict["extra_body"] = extra_body
        if config.request_parameter_allowlist is not None:
            model_config_dict["request_parameter_allowlist"] = list(
                config.request_parameter_allowlist
            )

    def query_structured(
        self,
        model_key: str,
        messages: List[Dict[str, str]],
        schema: Dict[str, Any],
        schema_name: str = "structured_response",
        reasoning_effort: Optional[str] = None,
    ) -> Tuple[Dict[str, Any], int]:
        """Query a model for a JSON object validated by the caller's schema.

        Providers advertising structured-output support receive a strict JSON
        schema.  Other OpenAI-compatible providers are prompted for JSON and the
        same parser/validator path is used by the planner.
        """
        config = self.registry.get_model_config(model_key)
        if not config:
            raise ValueError(f"Unknown model: {model_key}")
        if model_key in self.client_errors or config.name in self.client_errors:
            raise ValueError(self.client_errors.get(model_key) or self.client_errors[config.name])
        request_messages = list(messages)
        model_config_dict = {
            "temperature": config.temperature,
            "top_p": 1.0,
            "n": 1,
            "stream": False,
            "frequency_penalty": 0.0,
            "presence_penalty": 0.0,
            "logit_bias": {},
            "max_tokens": config.max_tokens,
        }
        if config.structured_outputs:
            model_config_dict["response_format"] = {
                "type": "json_schema",
                "json_schema": {
                    "name": schema_name,
                    "strict": True,
                    "schema": schema,
                },
            }
        else:
            request_messages = request_messages + [{
                "role": "user",
                "content": "Return only one JSON object matching the supplied schema.",
            }]
        self._apply_model_request_options(
            config,
            model_config_dict,
            reasoning_effort=reasoning_effort,
        )
        response, total_tokens = chat_completion_request(
            messages=request_messages,
            model=config.api_model_name,
            new_client=self.clients.get(config.name),
            model_config_dict=model_config_dict,
        )
        text = response if isinstance(response, str) else response.choices[0].message.content
        cleaned = (text or "").strip()
        if cleaned.startswith("```"):
            lines = cleaned.splitlines()
            cleaned = "\n".join(lines[1:-1])
        try:
            payload = json.loads(cleaned)
        except json.JSONDecodeError as error:
            raise StructuredResponseError("Structured model response is not valid JSON") from error
        if not isinstance(payload, dict):
            raise StructuredResponseError("Structured model response must be a JSON object")
        return payload, total_tokens
    
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
        self._apply_model_request_options(config, model_config_dict)

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
            return response, total_tokens
        
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
