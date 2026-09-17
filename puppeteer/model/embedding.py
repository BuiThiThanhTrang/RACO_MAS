import os
import openai
import yaml
from chromadb import EmbeddingFunction, Embeddings
from model.model_utils import model_log_and_print
from tenacity import retry, stop_after_attempt, wait_exponential
from typing import List
import torch
import numpy as np


try:
    with open("./config/global.yaml", "r", encoding="utf-8") as f:
        GLOBAL_CONFIG = yaml.safe_load(f)
except FileNotFoundError:
    raise FileNotFoundError("Global config file './config/global.yaml' not found!")

OPENAI_API_KEY = GLOBAL_CONFIG.get("api_keys", {}).get("openai_api_key")
BASE_URL = GLOBAL_CONFIG.get("api_keys", {}).get("openai_base_url", None)
MAX_RETRY_TIMES = GLOBAL_CONFIG.get("max_retry_times", 10)
MODEL_WEIGHT_PATH = GLOBAL_CONFIG.get("model_weight_path")

from model.api_config import api_config
client = api_config.global_openai_client()

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

class RewardModelTokenRepresentation():
    def __init__(self, device="cuda"):
        config = GLOBAL_CONFIG.get("reward_model") or {"backend": "local"}
        self.backend = config.get("backend", "local")
        self.device = device
        if self.backend == "remote":
            from reward_api.representation import RemoteRewardRepresentation
            self.remote = RemoteRewardRepresentation(config, device)
            return
        if self.backend != "local":
            raise ValueError("reward_model.backend must be local or remote")
        from transformers import AutoModelForCausalLM, AutoTokenizer
        self.model_name = "nvidia/Llama-3.1-Nemotron-70B-Reward-HF"
        self.model = AutoModelForCausalLM.from_pretrained(MODEL_WEIGHT_PATH, torch_dtype=torch.bfloat16, device_map="auto")
        self.tokenizer = AutoTokenizer.from_pretrained(MODEL_WEIGHT_PATH)
        print("device: {}".format(self.model.device))
        
    
    def truncate(self, messages):
        length = sum(len(message["content"]) for message in messages)
        
        while length > 12000:
            for message in messages:
                message["content"] = message["content"][-int(len(message["content"]) * 0.75):]  
            length = sum(len(message["content"]) for message in messages)  
        
        return messages

    def __call__(self, messages:List):
        if self.backend == "remote":
            return self.remote(messages)
        with torch.no_grad():
            messages = self.truncate(messages)
            model_log_and_print("tokenizing")
            model_log_and_print(messages)
            tokenized_message = self.tokenizer.apply_chat_template(messages, tokenize=True, add_generation_prompt=False, return_tensors="pt", return_dict=True, max_length=4096)
            model_log_and_print("tokenized done")
            input_ids = tokenized_message['input_ids'].to('cuda')
            attention_mask = tokenized_message['attention_mask'].to('cuda')
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
            return last_state.to(device=self.device, dtype=torch.float32), reward

    @property
    def dim(self):
        return 8192