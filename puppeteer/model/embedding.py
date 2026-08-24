import os
import openai
import yaml
import json
import hashlib
from model.model_utils import model_log_and_print
from model.api_config import api_config
from tenacity import retry, stop_after_attempt, wait_exponential
from typing import List, Any
import torch
import numpy as np

try:
    from chromadb import EmbeddingFunction, Embeddings
except ImportError:
    class EmbeddingFunction:
        pass

    Embeddings = list


try:
    with open("./config/global.yaml", "r", encoding="utf-8") as f:
        GLOBAL_CONFIG = yaml.safe_load(f)
except FileNotFoundError:
    raise FileNotFoundError("Global config file './config/global.yaml' not found!")

OPENAI_API_KEY = GLOBAL_CONFIG.get("api_keys", {}).get("openai_api_key")
BASE_URL = GLOBAL_CONFIG.get("api_keys", {}).get("openai_base_url", None)
MAX_RETRY_TIMES = GLOBAL_CONFIG.get("max_retry_times", 10)
MODEL_WEIGHT_PATH = GLOBAL_CONFIG.get("model_weight_path")
STATE_REPRESENTATION_CONFIG = GLOBAL_CONFIG.get("state_representation", {}) or {}
REWARD_MODEL_CONFIG = GLOBAL_CONFIG.get("reward_model", {}) or {}
EMBEDDING_MAX_RETRY_TIMES = int(STATE_REPRESENTATION_CONFIG.get("max_retry_times", min(MAX_RETRY_TIMES, 3)))

if BASE_URL:
    client = openai.OpenAI(api_key=OPENAI_API_KEY, base_url=BASE_URL)
else:
    client = openai.OpenAI(api_key=OPENAI_API_KEY)


class OpenAIEmbedding(EmbeddingFunction):
    @staticmethod
    @retry(wait=wait_exponential(min=5, max=10), stop=stop_after_attempt(MAX_RETRY_TIMES))
    def get_embedding(text) -> Embeddings:
        embedding_model = "text-embedding-ada-002"
        model_log_and_print(f"[Embedding] embedding from {embedding_model}")

        if isinstance(text, str):
            text = [text.replace("\n", " ")]
        else:
            text = [t.replace("\n", " ") for t in text]

        text = [t[:8191] if len(t) > 8191 else t for t in text]
        text = ["none"] if all(len(t) == 0 for t in text) else text

        model_log_and_print(f"[Embedding] {text}")

        response = client.embeddings.create(input=text, model=embedding_model)
        embeddings = [data.embedding for data in response.data]

        prompt_tokens = response.usage.prompt_tokens
        total_tokens = response.usage.total_tokens
        model_log_and_print(f"[Embedding] Token Usage\nPrompt Tokens: {prompt_tokens}\nTotal Tokens: {total_tokens}")

        return embeddings
    
    @property
    def dim(self):
        return 1536


class APIEmbeddingStateRepresentation:
    """Use an embedding API as Puppeteer's policy state representation.

    This replaces the local 70B reward model path for local/CPU-friendly runs.
    It returns a real task/history-dependent vector, while keeping reward as
    zero because task-level rewards are still computed by the benchmark
    evaluator during finalization.
    """

    DEFAULT_DIMS = {
        "text-embedding-004": 768,
        "embedding-001": 768,
        "models/gemini-embedding-001": 3072,
        "models/gemini-embedding-2-preview": 3072,
        "models/gemini-embedding-2": 3072,
        "text-embedding-3-small": 1536,
        "text-embedding-3-large": 3072,
        "text-embedding-ada-002": 1536,
    }

    def __init__(self, state_config=None):
        state_config = dict(state_config or GLOBAL_CONFIG.get("state_representation", {}) or {})
        cache_config = state_config.get("cache", {}) or {}

        self.provider = state_config.get("provider", "openai")
        self.provider_profile = state_config.get("provider_profile")
        self.model = state_config.get("model", "text-embedding-3-small")
        self.max_chars = int(state_config.get("max_chars", 12000))
        self.reward_value = float(state_config.get("reward", 0.0))
        self.cache_enabled = bool(cache_config.get("enabled", True))
        self.cache_path = cache_config.get("path", "cache/embeddings")
        self._dim = state_config.get("dim", self.DEFAULT_DIMS.get(self.model))
        self.client = self._build_client()

        if self._dim is None:
            raise ValueError(
                "state_representation.dim is required for unknown embedding model "
                f"'{self.model}'. Add it to config/global.yaml."
            )
        self._dim = int(self._dim)

        if self.cache_enabled:
            os.makedirs(self.cache_path, exist_ok=True)

        model_log_and_print(
            f"[State Representation] Using API embedding provider={self.provider}, "
            f"profile={self.provider_profile or 'legacy'}, "
            f"model={self.model}, dim={self._dim}"
        )

    def _build_client(self):
        if self.provider_profile:
            provider_config = api_config.get_provider(self.provider_profile)
            if provider_config:
                client_kwargs = {
                    "api_key": provider_config.get("api_key") or OPENAI_API_KEY,
                    "base_url": provider_config.get("base_url") or BASE_URL,
                }
                headers = provider_config.get("headers") or {}
                if headers:
                    client_kwargs["default_headers"] = headers
                return openai.OpenAI(**client_kwargs)

        if BASE_URL:
            return openai.OpenAI(api_key=OPENAI_API_KEY, base_url=BASE_URL)
        return openai.OpenAI(api_key=OPENAI_API_KEY)

    def _message_to_text(self, message: Any) -> str:
        if isinstance(message, dict):
            role = message.get("role", "unknown")
            content = message.get("content", "")
            return f"{role}: {content}"
        return str(message)

    def _format_messages(self, messages: Any) -> str:
        if isinstance(messages, list):
            text = "\n".join(self._message_to_text(message) for message in messages)
        else:
            text = str(messages)
        text = text.replace("\x00", " ").strip()
        if not text:
            text = "empty state"
        return text[-self.max_chars:]

    def _cache_file(self, text: str) -> str:
        key_data = json.dumps(
            {"provider": self.provider, "model": self.model, "text": text},
            ensure_ascii=False,
            sort_keys=True,
        )
        key = hashlib.sha256(key_data.encode("utf-8")).hexdigest()
        return os.path.join(self.cache_path, f"{key}.json")

    def _load_cached_embedding(self, text: str):
        if not self.cache_enabled:
            return None
        cache_file = self._cache_file(text)
        if not os.path.exists(cache_file):
            return None
        with open(cache_file, "r", encoding="utf-8") as f:
            cached = json.load(f)
        embedding = cached.get("embedding")
        if isinstance(embedding, list):
            return embedding
        return None

    def _save_cached_embedding(self, text: str, embedding: List[float]) -> None:
        if not self.cache_enabled:
            return
        cache_file = self._cache_file(text)
        with open(cache_file, "w", encoding="utf-8") as f:
            json.dump(
                {
                    "provider": self.provider,
                    "model": self.model,
                    "dim": len(embedding),
                    "embedding": embedding,
                },
                f,
                ensure_ascii=False,
            )

    @retry(wait=wait_exponential(min=2, max=5), stop=stop_after_attempt(EMBEDDING_MAX_RETRY_TIMES))
    def _request_embedding(self, text: str) -> List[float]:
        model_log_and_print(f"[State Representation] Request embedding from {self.model}")
        response = self.client.embeddings.create(input=text, model=self.model)
        embedding = response.data[0].embedding
        return list(embedding)

    def __call__(self, messages: List):
        text = self._format_messages(messages)
        embedding = self._load_cached_embedding(text)
        if embedding is None:
            embedding = self._request_embedding(text)
            self._save_cached_embedding(text, embedding)

        if len(embedding) != self._dim:
            raise ValueError(
                f"Embedding dimension mismatch for {self.model}: config dim={self._dim}, "
                f"API returned dim={len(embedding)}. Update state_representation.dim "
                "or choose a compatible embedding model."
            )

        state = torch.tensor([embedding], dtype=torch.float32)
        return state, self.reward_value

    @property
    def dim(self):
        return self._dim

class RewardModelTokenRepresentation():
    def __init__(self, model_weight_path=None, reward_model_config=None):
        model_weight_path = model_weight_path or MODEL_WEIGHT_PATH
        reward_model_config = dict(reward_model_config or REWARD_MODEL_CONFIG)

        try:
            from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer
        except ImportError as e:
            raise ImportError(
                "transformers is required for state_representation.type=local_reward_model. "
                "Run: pip install -r requirements.txt"
            ) from e

        if not model_weight_path:
            raise ValueError("model_weight_path is required for RewardModelTokenRepresentation.")

        self.model_name = model_weight_path
        self.device_map = reward_model_config.get("device_map", "auto")
        self.max_length = int(reward_model_config.get("max_length", 4096))
        self.max_chars = int(reward_model_config.get("max_chars", 12000))
        self.torch_dtype = self._resolve_torch_dtype(reward_model_config.get("torch_dtype", "auto"))
        self.model_config = AutoConfig.from_pretrained(model_weight_path, trust_remote_code=True)
        self._dim = int(getattr(self.model_config, "hidden_size", 8192))

        load_kwargs = {
            "torch_dtype": self.torch_dtype,
            "device_map": self.device_map,
            "trust_remote_code": True,
        }
        quantization = str(reward_model_config.get("quantization", "none")).lower()
        if quantization in {"8bit", "int8"}:
            load_kwargs["load_in_8bit"] = True
        elif quantization in {"4bit", "nf4"}:
            load_kwargs["load_in_4bit"] = True

        self.model = AutoModelForCausalLM.from_pretrained(model_weight_path, **load_kwargs)
        self.model.eval()
        self.tokenizer = AutoTokenizer.from_pretrained(model_weight_path, trust_remote_code=True)
        self.input_device = self._resolve_input_device()
        model_log_and_print(
            f"[Reward Model] Loaded {model_weight_path} with dtype={self.torch_dtype}, "
            f"device_map={self.device_map}, input_device={self.input_device}, hidden_size={self._dim}"
        )

    def _resolve_torch_dtype(self, dtype_name):
        if dtype_name in (None, "auto"):
            if torch.cuda.is_available():
                major, _ = torch.cuda.get_device_capability(0)
                return torch.bfloat16 if major >= 8 else torch.float16
            return torch.float32

        dtype_map = {
            "bf16": torch.bfloat16,
            "bfloat16": torch.bfloat16,
            "fp16": torch.float16,
            "float16": torch.float16,
            "fp32": torch.float32,
            "float32": torch.float32,
        }
        if dtype_name not in dtype_map:
            raise ValueError(
                f"Unsupported reward_model.torch_dtype='{dtype_name}'. "
                "Use auto, float16, bfloat16, or float32."
            )
        return dtype_map[dtype_name]

    def _resolve_input_device(self):
        if hasattr(self.model, "hf_device_map") and self.model.hf_device_map:
            for device in self.model.hf_device_map.values():
                if isinstance(device, int):
                    return torch.device(f"cuda:{device}")
                if isinstance(device, str) and device not in {"cpu", "disk"}:
                    return torch.device(device)
        try:
            return next(self.model.parameters()).device
        except StopIteration:
            return torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def truncate(self, messages):
        length = sum(len(message["content"]) for message in messages)
        
        while length > self.max_chars:
            for message in messages:
                message["content"] = message["content"][-int(len(message["content"]) * 0.75):]  
            length = sum(len(message["content"]) for message in messages)  
        
        return messages

    def __call__(self, messages:List):
        with torch.no_grad():
            messages = self.truncate(messages)
            model_log_and_print("tokenizing")
            model_log_and_print(messages)
            tokenized_message = self.tokenizer.apply_chat_template(
                messages,
                tokenize=True,
                add_generation_prompt=False,
                return_tensors="pt",
                return_dict=True,
                max_length=self.max_length,
                truncation=True,
            )
            model_log_and_print("tokenized done")
            input_ids = tokenized_message['input_ids'].to(self.input_device)
            attention_mask = tokenized_message['attention_mask'].to(self.input_device)
            response_token_ids = self.model.generate(input_ids=input_ids,
                                                attention_mask=attention_mask,  
                                                max_new_tokens=1, 
                                                return_dict_in_generate=True, 
                                                output_scores=True,
                                                output_logits=True,
                                                output_hidden_states=True)
            reward = response_token_ids['scores'][0][0][0].item()
            hidden_states = response_token_ids.hidden_states  
            state = hidden_states[0][-1]
            last_state = state[:,-1,:]
            print(reward)
            return last_state, reward

    @property
    def dim(self):
        return self._dim
