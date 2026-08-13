from typing import Dict
import logging
import yaml
from tenacity import retry
from tenacity.stop import stop_after_attempt
from tenacity.wait import wait_exponential

logger = logging.getLogger("model")

try:
    with open("./config/global.yaml", "r", encoding="utf-8") as f:
        GLOBAL_CONFIG = yaml.safe_load(f) or {}
except FileNotFoundError:
    GLOBAL_CONFIG = {}

CHAT_MAX_RETRY_TIMES = int(GLOBAL_CONFIG.get("max_retry_times", 3))
CHAT_RETRY_WAIT_MIN = float(GLOBAL_CONFIG.get("retry_wait_min", 1))
CHAT_RETRY_WAIT_MAX = float(GLOBAL_CONFIG.get("retry_wait_max", 3))

class APIConfig:
    SLOW_FLAG = False 
    TRUNCATE_FACTOR = 0

def model_log_and_print(content):
    if content is not None:
        logger.info(content)
        print(content)

def truncate_messages(messages):
    max_length = 0
    max_index = 0
    for i, msg in enumerate(messages):
        if len(msg.get('content', '')) > max_length:
            max_length = len(msg['content'])
            max_index = i

    content = messages[max_index]['content']
    factor = 1/(2**APIConfig.TRUNCATE_FACTOR)
    messages[max_index]['content'] = content[:int(len(content)*factor)]  
    return messages


def calc_max_token(messages, max_tokens):
    string = "\n".join([str(message["content"]) for message in messages])
    num_prompt_tokens = int(len(string)//1.8) # approximation of tokens number 
    gap_between_send_receive = 15 * len(messages)
    num_prompt_tokens += gap_between_send_receive

    num_max_completion_tokens = max_tokens - num_prompt_tokens
    logger.info(f"num_prompt_tokens: {num_prompt_tokens}, num_max_completion_tokens: {num_max_completion_tokens}")
    if num_max_completion_tokens < 0:
        logger.warning(f"num_max_completion_tokens is negative: {num_max_completion_tokens}")
        return 0
    return num_max_completion_tokens


@retry(
    wait=wait_exponential(min=CHAT_RETRY_WAIT_MIN, max=CHAT_RETRY_WAIT_MAX),
    stop=stop_after_attempt(CHAT_MAX_RETRY_TIMES),
)
def chat_completion_request(messages, model, new_client, model_config_dict: Dict = None):
    if model_config_dict is None:
        model_config_dict = {
            "temperature": 0.1,
            "top_p": 1.0,
            "n": 1,
            "stream": False,
            "frequency_penalty": 0.0,
            "presence_penalty": 0.0,
            "logit_bias": {},
        }

    # Check if using Gemini through its OpenAI-compatible API. Some utility
    # paths call the generic gpt alias while the configured base URL is Gemini.
    base_url = str(getattr(new_client, "base_url", ""))
    is_gemini = model.startswith("gemini") or "generativelanguage.googleapis.com" in base_url
    is_huggingface_router = "router.huggingface.co" in base_url

    if is_gemini or is_huggingface_router:
        # Some OpenAI-compatible routers only support a subset of parameters.
        json_data = {
            "model": model,
            "messages": messages,
            "max_tokens": min(model_config_dict.get("max_tokens", 4096), 4096),
            "temperature": model_config_dict.get("temperature", 0.1),
            "stream": False,
        }
    else:
        json_data = {
            "model": model,
            "messages": messages,
            "max_tokens": min(model_config_dict.get("max_tokens", 4096), 4096),
            "temperature": model_config_dict["temperature"],
            "top_p": model_config_dict["top_p"],
            "n": model_config_dict["n"],
            "stream": model_config_dict["stream"],
            "frequency_penalty": model_config_dict["frequency_penalty"],
            "presence_penalty": model_config_dict["presence_penalty"],
            "logit_bias": model_config_dict["logit_bias"],
        }

    try:
        model_log_and_print("[Model Query] model={} messages={}".format(model, messages))
        if APIConfig.SLOW_FLAG:
            messages = truncate_messages(messages=messages)

        response = new_client.chat.completions.create(**json_data)

        completion_tokens = response.usage.completion_tokens if response.usage else 0
        prompt_tokens = response.usage.prompt_tokens if response.usage else 0
        total_tokens = response.usage.total_tokens if response.usage else 0
        if total_tokens == 0:
            total_tokens = prompt_tokens + completion_tokens
        if total_tokens == 0:
            total_tokens = int(len(response.choices[0].message.content)//1.8)
        model_log_and_print(f"[Model Query] Token Usage: \nCompletion Tokens: {completion_tokens} \nPrompt Tokens: {prompt_tokens} \nTotal Tokens: {total_tokens}")
        APIConfig.SLOW_FLAG = False
        APIConfig.TRUNCATE_FACTOR = 0
        return response, total_tokens   

    except Exception as e:
        print("Unable to generate ChatCompletion response. " + f"API calling Exception: {e}")
        APIConfig.SLOW_FLAG = True
        APIConfig.TRUNCATE_FACTOR += 1
        model_log_and_print(f"[Model Query: ChatCompletion] model={model} query failed: {str(e)}")
        raise Exception()
