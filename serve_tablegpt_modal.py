"""Serve TableGPT2 through an authenticated OpenAI-compatible Modal endpoint."""

import subprocess

import modal

MODEL_ID = "tablegpt/TableGPT2-7B"
MODEL_REVISION = "c0ef9cb136c28f3617d51e1900e474ffb780037c"
PORT = 8000

app = modal.App("tablegpt2-7b")

image = (
    modal.Image.from_registry(
        "nvidia/cuda:12.9.0-devel-ubuntu22.04",
        add_python="3.12",
    )
    .entrypoint([])
    .uv_pip_install("vllm==0.21.0")
    .env({"HF_HOME": "/hf-cache"})
)

weights_cache = modal.Volume.from_name(
    "tablegpt2-hf-cache", create_if_missing=True
)


@app.function(
    image=image,
    gpu="L40S",
    volumes={"/hf-cache": weights_cache},
    max_containers=1,
    scaledown_window=300,
)
@modal.web_server(PORT, startup_timeout=900, requires_proxy_auth=True)
def serve():
    subprocess.Popen([
        "vllm", "serve", MODEL_ID,
        "--revision", MODEL_REVISION,
        "--served-model-name", MODEL_ID,
        "--host", "0.0.0.0",
        "--port", str(PORT),
        "--dtype", "bfloat16",
        "--max-model-len", "32768",
        "--max-num-seqs", "1",
    ])
