from model.api_config import api_config
global_openai_client = api_config.global_openai_client()

from model.model_config import model_registry
from model.query_manager import query_manager
def _create_query_function(model_key: str):
    def query_func(messages, system_prompt=None):
        # Internal text helpers historically used query_gpt, outside the persona pool.
        selected = api_config.global_config.get("auxiliary_model") if model_key == "gpt-3.5" else None
        return query_manager.query(selected or model_key, messages, system_prompt)
    return query_func

_generated_functions = {}
for model_key, config in model_registry.get_all_models().items():
    func = _create_query_function(model_key)
    func.__name__ = config.function_name
    _generated_functions[config.function_name] = func
    globals()[config.function_name] = func

__all__ = ['ModelQueryManager', 'query_manager'] + list(_generated_functions.keys()) 