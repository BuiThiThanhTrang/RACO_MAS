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
        # Resolve credentials only for a model that is actually called.

    def _client_for(self, config):
        if config.name not in self.clients:
            self.clients[config.name] = self.config_manager.create_client(config.provider, config.url)
        return self.clients[config.name]

    def query(self, model_key: str, messages: List[Dict[str, str]], 
              system_prompt: Optional[str] = None) -> Tuple[str, int]:
        config = self.registry.get_model_config(model_key)
        if not config:
            available_models = ", ".join(self.registry.list_available_models())
            raise ValueError(f"Unknown model: {model_key}. Available models: {available_models}")
        
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
            new_client=self._client_for(config),
            model_config_dict=model_config_dict
        )
        
        if isinstance(response, str):
            return response, 1
        
        response_message = response.choices[0].message.content
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
