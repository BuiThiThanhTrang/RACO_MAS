"""Low-cost Modal deployment for Qwen2.5-Coder-3B-Instruct.

The endpoint implements the subset of OpenAI Chat Completions used by RouterLab.
It intentionally uses one T4, one container and one request at a time so benchmark
runs cannot silently scale GPU cost horizontally.
"""

from __future__ import annotations

import modal

APP_NAME = "routerlab-qwen25-coder-3b"
MODEL_ID = "Qwen/Qwen2.5-Coder-3B-Instruct"
MODEL_REVISION = "d3a7bedd480bc27342f4336c71e86858b25df935"
MODEL_DIR = "/models/qwen25-coder-3b"
MAX_CONTEXT_TOKENS = 8192
MAX_OUTPUT_TOKENS = 1024


def download_model() -> None:
    from huggingface_hub import snapshot_download

    snapshot_download(
        repo_id=MODEL_ID,
        revision=MODEL_REVISION,
        local_dir=MODEL_DIR,
        ignore_patterns=["*.bin", "*.h5", "*.msgpack"],
    )


image = (
    modal.Image.debian_slim(python_version="3.11")
    .uv_pip_install(
        "accelerate==1.4.0",
        "fastapi[standard]==0.115.8",
        "huggingface-hub==0.29.1",
        "torch==2.6.0",
        "transformers==4.49.0",
    )
    .env(
        {
            "HF_HUB_DISABLE_TELEMETRY": "1",
            "TOKENIZERS_PARALLELISM": "false",
        }
    )
    .run_function(download_model, timeout=1800)
)

app = modal.App(APP_NAME)


@app.function(
    image=image,
    gpu="T4",
    min_containers=0,
    max_containers=1,
    scaledown_window=30,
    timeout=900,
)
@modal.concurrent(max_inputs=1)
@modal.asgi_app(requires_proxy_auth=True)
def serve():
    import time
    import uuid

    import torch
    from fastapi import FastAPI, HTTPException
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR, local_files_only=True)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_DIR,
        local_files_only=True,
        torch_dtype=torch.float16,
        device_map={"": "cuda"},
        low_cpu_mem_usage=True,
        attn_implementation="sdpa",
    ).eval()
    web_app = FastAPI(title="RouterLab Qwen2.5 Coder")

    @web_app.get("/health")
    async def health() -> dict[str, str]:
        return {"status": "ok", "model": MODEL_ID}

    @web_app.post("/v1/chat/completions")
    async def chat_completions(payload: dict) -> dict:
        messages = payload.get("messages")
        if not isinstance(messages, list) or not messages:
            raise HTTPException(status_code=400, detail="messages must be non-empty")
        if payload.get("stream", False):
            raise HTTPException(status_code=400, detail="streaming is not supported")

        max_tokens = min(
            max(int(payload.get("max_tokens", MAX_OUTPUT_TOKENS)), 1),
            MAX_OUTPUT_TOKENS,
        )
        temperature = float(payload.get("temperature", 0.0))
        top_p = float(payload.get("top_p", 1.0))
        encoded = tokenizer.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            return_tensors="pt",
            return_dict=True,
        )
        input_tokens = int(encoded["input_ids"].shape[-1])
        if input_tokens + max_tokens > MAX_CONTEXT_TOKENS:
            raise HTTPException(
                status_code=400,
                detail=(
                    f"input ({input_tokens}) + output ({max_tokens}) exceeds "
                    f"the {MAX_CONTEXT_TOKENS}-token experiment limit"
                ),
            )
        encoded = {key: value.to("cuda") for key, value in encoded.items()}
        generation_kwargs = {
            "max_new_tokens": max_tokens,
            "do_sample": temperature > 0.0,
            "pad_token_id": tokenizer.eos_token_id,
        }
        if temperature > 0.0:
            generation_kwargs.update(temperature=temperature, top_p=top_p)

        with torch.inference_mode():
            output = model.generate(**encoded, **generation_kwargs)
        generated = output[0, input_tokens:]
        output_tokens = int(generated.numel())
        content = tokenizer.decode(generated, skip_special_tokens=True)
        finish_reason = "length" if output_tokens >= max_tokens else "stop"
        return {
            "id": f"chatcmpl-{uuid.uuid4().hex}",
            "object": "chat.completion",
            "created": int(time.time()),
            "model": MODEL_ID,
            "choices": [
                {
                    "index": 0,
                    "message": {"role": "assistant", "content": content},
                    "finish_reason": finish_reason,
                }
            ],
            "usage": {
                "prompt_tokens": input_tokens,
                "completion_tokens": output_tokens,
                "total_tokens": input_tokens + output_tokens,
            },
        }

    return web_app
