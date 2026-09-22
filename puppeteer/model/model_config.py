from typing import Dict, Any, Optional, List
from dataclasses import dataclass

@dataclass
class ModelConfig:
    name: str
    function_name: str     
    api_model_name: str    
    provider: str          
    max_tokens: int      
    model_size: int   # for open-source models, this is the number of parameters in millions; but for API models, this is just an estimate
    url: Optional[str] = None      
    api_profile: Optional[str] = None
    temperature: float = 0.0       
    description: str = ""          


MODEL_REGISTRY: Dict[str, ModelConfig] = {
    "gpt-3.5": ModelConfig(
        name = "gpt-3.5",
        function_name="query_gpt",
        api_model_name="gpt-3.5-turbo",
        provider="openai",
        model_size=175,# which is estimated
        max_tokens=4096,
        description="OpenAI GPT-3.5 Turbo model"
    ),
    
    "gpt-4o": ModelConfig(
        name = "gpt-4o",
        function_name="query_gpt4o",
        api_model_name="gpt-4o",
        provider="openai", 
        model_size=200,# which is estimated
        max_tokens=128000,
        description="OpenAI GPT-4o model"
    ),

    "qwen-3.5-9b": ModelConfig(
        name="qwen-3.5-9b",
        function_name="query_qwen3_5_9b",
        api_model_name="Qwen/Qwen3.5-9B:featherless-ai",
        provider="openai_compatible",
        api_profile="huggingface_router",
        model_size=9,
        max_tokens=8192,
        description="Qwen 3.5 9B via Hugging Face Inference Providers on Featherless AI",
    ),

    "qwen-3.5-4b": ModelConfig(
        name="qwen-3.5-4b",
        function_name="query_qwen3_5_4b",
        api_model_name="Qwen/Qwen3.5-4B:featherless-ai",
        provider="openai_compatible",
        api_profile="huggingface_router",
        model_size=4,
        max_tokens=8192,
        description="Qwen 3.5 4B via Hugging Face Inference Providers on Featherless AI",
    ),

    "gemma-3-12b-it": ModelConfig(
        name="gemma-3-12b-it",
        function_name="query_gemma_3_12b_it",
        api_model_name="google/gemma-3-12b-it:featherless-ai",
        provider="openai_compatible",
        api_profile="huggingface_router",
        model_size=12,
        max_tokens=8192,
        description="Gemma 3 12B IT via Hugging Face Inference Providers on Featherless AI",
    ),

    "phi-4-mini-instruct": ModelConfig(
        name="phi-4-mini-instruct",
        function_name="query_phi_4_mini_instruct",
        api_model_name="microsoft/Phi-4-mini-instruct:featherless-ai",
        provider="openai_compatible",
        api_profile="huggingface_router",
        model_size=4,
        # Featherless accepts this routed model through 3,072 output tokens,
        # but rejects a 4,096-token request with HTTP 400.
        max_tokens=3072,
        description="Phi-4 Mini Instruct via Hugging Face Inference Providers on Featherless AI (3,072 output-token provider limit)",
    ),

    "qwen-2.5-14b": ModelConfig(
        name = "qwen-2.5-14b",
        function_name="query_qwen2_5_14b",
        api_model_name="Qwen/Qwen2.5-14B-Instruct:featherless-ai",
        provider="openai_compatible",
        api_profile="huggingface_router",
        model_size=14,
        max_tokens=8192,    
        description="Qwen 2.5 14B Instruct model via Hugging Face router on featherless-ai"
    ),

    "qwen-2.5-7b": ModelConfig(
        name="qwen-2.5-7b",
        function_name="query_qwen2_5_7b",
        api_model_name="Qwen/Qwen2.5-7B-Instruct:featherless-ai",
        provider="openai_compatible",
        api_profile="huggingface_router",
        model_size=7,
        max_tokens=8192,
        description="Qwen 2.5 7B Instruct model via Hugging Face router on Featherless AI"
    ),

    "llama-3.1-8b": ModelConfig(
        name="llama-3.1-8b",
        function_name="query_llama_3_1_8b",
        api_model_name="meta-llama/Llama-3.1-8B-Instruct:novita",
        provider="openai_compatible",
        api_profile="huggingface_router",
        model_size=8,
        max_tokens=8192,
        description="Llama 3.1 8B Instruct model via Hugging Face router on Novita"
    ),

    "llama-3.2-3b": ModelConfig(
        name="llama-3.2-3b",
        function_name="query_llama_3_2_3b",
        api_model_name="meta-llama/Llama-3.2-3B-Instruct:featherless-ai",
        provider="openai_compatible",
        api_profile="huggingface_router",
        model_size=3,
        max_tokens=8192,
        description="Llama 3.2 3B Instruct model via Hugging Face router on Featherless AI"
    ),

    "llama-3.2-3b-instruct": ModelConfig(
        name="llama-3.2-3b-instruct",
        function_name="query_llama_3_2_3b_instruct",
        api_model_name="meta-llama/Llama-3.2-3B-Instruct:featherless-ai",
        provider="openai_compatible",
        api_profile="huggingface_router",
        model_size=3,
        max_tokens=8192,
        description="Llama 3.2 3B Instruct model via Hugging Face router on Featherless AI"
    ),

    "ministral-3b-2512": ModelConfig(
        name="ministral-3b-2512",
        function_name="query_ministral_3b_2512",
        api_model_name="mistralai/ministral-3b-2512",
        provider="openai_compatible",
        api_profile="openrouter",
        model_size=3,
        max_tokens=4096,
        description="Mistral Ministral 3B 2512 model via OpenRouter"
    ),

    "mistralai/ministral-3b-2512": ModelConfig(
        name="mistralai/ministral-3b-2512",
        function_name="query_mistralai_ministral_3b_2512",
        api_model_name="mistralai/ministral-3b-2512",
        provider="openai_compatible",
        api_profile="openrouter",
        model_size=3,
        max_tokens=4096,
        description="Mistral Ministral 3B 2512 model via OpenRouter"
    ),

    "mistral-nemo-12b": ModelConfig(
        name="mistral-nemo-12b",
        function_name="query_mistral_nemo_12b",
        api_model_name="mistralai/mistral-nemo",
        provider="openai_compatible",
        api_profile="openrouter",
        model_size=12,
        max_tokens=16384,
        description="Mistral Nemo 12B model via OpenRouter"
    ),

    "Mistral-Nemo-12B": ModelConfig(
        name="Mistral-Nemo-12B",
        function_name="query_mistral_nemo_12b_cased",
        api_model_name="mistralai/mistral-nemo",
        provider="openai_compatible",
        api_profile="openrouter",
        model_size=12,
        max_tokens=16384,
        description="Mistral Nemo 12B model via OpenRouter"
    ),

    "gemini-3.1-flash-lite": ModelConfig(
        name="gemini-3.1-flash-lite",
        function_name="query_gemini_3_1_flash_lite",
        api_model_name="gemini-3.1-flash-lite",
        provider="openai",
        api_profile="gemini_openai",
        model_size=100,  # estimated
        max_tokens=65536,
        description="Google Gemini 3.1 Flash Lite model via OpenAI-compatible API"
    ),

    "gemini-2.5-flash-lite": ModelConfig(
        name="gemini-2.5-flash-lite",
        function_name="query_gemini_2_5_flash_lite",
        api_model_name="gemini-2.5-flash-lite",
        provider="openai",
        api_profile="gemini_openai",
        model_size=100,  # estimated
        max_tokens=65536,
        description="Google Gemini 2.5 Flash Lite model via OpenAI-compatible API"
    ),

    "gemini-2.5-flash": ModelConfig(
        name="gemini-2.5-flash",
        function_name="query_gemini_2_5_flash",
        api_model_name="gemini-2.5-flash",
        provider="openai",
        api_profile="gemini_openai",
        model_size=300,  # estimated
        max_tokens=65536,
        description="Google Gemini 2.5 Flash model via OpenAI-compatible API"
    ),

    "gemini-3.5-flash": ModelConfig(
        name="gemini-3.5-flash",
        function_name="query_gemini_3_5_flash",
        api_model_name="gemini-3.5-flash",
        provider="openai",
        api_profile="gemini_openai",
        model_size=300,  # estimated
        max_tokens=65536,
        description="Google Gemini 3.5 Flash model via OpenAI-compatible API"
    ),

    "gemma-4-31b-it": ModelConfig(
        name="gemma-4-31b-it",
        function_name="query_gemma_4_31b_it",
        api_model_name="gemma-4-31b-it",
        provider="openai",
        api_profile="gemini_openai",
        model_size=31,
        max_tokens=32768,
        description="Gemma 4 31B instruction-tuned model via OpenAI-compatible API"
    ),

    "gemma-4-26b-a4b-it": ModelConfig(
        name="gemma-4-26b-a4b-it",
        function_name="query_gemma_4_26b_a4b_it",
        api_model_name="gemma-4-26b-a4b-it",
        provider="openai",
        api_profile="gemini_openai",
        model_size=26,
        max_tokens=32768,
        description="Gemma 4 26B A4B instruction-tuned model via OpenAI-compatible API"
    ),
}

class ModelRegistry:
    def __init__(self):
        self.registry = MODEL_REGISTRY.copy()
    
    def register_model(self, key: str, config: ModelConfig) -> None:
        self.registry[key] = config
    
    def get_model_config(self, key: str) -> Optional[ModelConfig]:
        return self.registry.get(key)

    def get_model_size(self, key: str) -> Optional[int]:
        config = self.get_model_config(key)
        return config.model_size if config else None    
    
    def get_all_models(self) -> Dict[str, ModelConfig]:
        return self.registry.copy()
    
    def get_models_by_provider(self, provider: str) -> Dict[str, ModelConfig]:
        return {k: v for k, v in self.registry.items() if v.provider == provider}
    
    def get_function_name(self, key: str) -> Optional[str]:
        config = self.get_model_config(key)
        return config.function_name if config else None
    
    def get_api_model_name(self, key: str) -> Optional[str]:
        config = self.get_model_config(key)
        return config.api_model_name if config else None
    
    def list_available_models(self) -> List[str]:
        return list(self.registry.keys())
    
    def search_models(self, keyword: str) -> Dict[str, ModelConfig]:
        keyword = keyword.lower()
        return {
            k: v for k, v in self.registry.items() 
            if keyword in k.lower() or keyword in v.display_name.lower()
        }

model_registry = ModelRegistry()
