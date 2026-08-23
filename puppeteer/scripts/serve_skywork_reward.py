from __future__ import annotations

import os
import secrets
import sys
from pathlib import Path
from threading import Lock

from fastapi import FastAPI, Header, HTTPException
from pydantic import BaseModel, Field
from starlette.concurrency import run_in_threadpool


PROJECT_DIR = Path(__file__).resolve().parents[1]
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

from role_aware.reward_scorer import AuxiliaryRewardScorer


MODEL_NAME = os.getenv(
    "SKYWORK_REWARD_MODEL",
    "Skywork/Skywork-Reward-V2-Llama-3.1-8B",
)
API_KEY = os.getenv("SKYWORK_REWARD_API_KEY", "")
ALLOW_UNAUTHENTICATED = (
    os.getenv("SKYWORK_REWARD_ALLOW_UNAUTHENTICATED", "false").lower()
    in {"1", "true", "yes"}
)
MAX_CHARS = int(os.getenv("SKYWORK_REWARD_MAX_CHARS", "12000"))
if not API_KEY and not ALLOW_UNAUTHENTICATED:
    raise RuntimeError(
        "Set SKYWORK_REWARD_API_KEY before starting the server, or explicitly "
        "set SKYWORK_REWARD_ALLOW_UNAUTHENTICATED=true for a trusted local tunnel"
    )

scorer = AuxiliaryRewardScorer(
    {
        "backend": "transformers_local",
        "model": MODEL_NAME,
        "max_length": int(os.getenv("SKYWORK_REWARD_MAX_LENGTH", "4096")),
        "device_map": os.getenv("SKYWORK_REWARD_DEVICE_MAP", "auto"),
        "torch_dtype": os.getenv("SKYWORK_REWARD_TORCH_DTYPE", "auto"),
        "quantization": os.getenv("SKYWORK_REWARD_QUANTIZATION", "none"),
        "cache": {"enabled": True},
    }
)
model_lock = Lock()
app = FastAPI(title="Skywork Trajectory Reward API", version="1.0")


class RewardRequest(BaseModel):
    model: str = Field(default=MODEL_NAME)
    messages: list[dict[str, str]]


def _authorize(authorization: str | None) -> None:
    if not API_KEY:
        return
    expected = f"Bearer {API_KEY}"
    if authorization is None or not secrets.compare_digest(authorization, expected):
        raise HTTPException(status_code=401, detail="Invalid API key")


def _validate_request(request: RewardRequest) -> None:
    if request.model != MODEL_NAME:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported model: {request.model}",
        )
    if not request.messages:
        raise HTTPException(status_code=400, detail="messages must not be empty")
    total_chars = 0
    for message in request.messages:
        if message.get("role") not in {"system", "user", "assistant"}:
            raise HTTPException(status_code=400, detail="Invalid message role")
        content = message.get("content")
        if not isinstance(content, str):
            raise HTTPException(
                status_code=400,
                detail="Each message content must be a string",
            )
        total_chars += len(content)
    if total_chars > MAX_CHARS:
        raise HTTPException(
            status_code=413,
            detail=f"Trajectory exceeds {MAX_CHARS} characters",
        )


def _score(messages: list[dict[str, str]]) -> float:
    # Serialize access to lazy model loading and the single GPU.
    with model_lock:
        return scorer(messages)


@app.get("/health")
def health():
    return {
        "status": "ok",
        "model": MODEL_NAME,
        "loaded": scorer.model is not None,
        "backend": "transformers_local",
    }


@app.post("/v1/reward")
async def reward(
    request: RewardRequest,
    authorization: str | None = Header(default=None),
):
    _authorize(authorization)
    _validate_request(request)
    score = await run_in_threadpool(_score, request.messages)
    return {"model": MODEL_NAME, "score": score}


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        app,
        host=os.getenv("SKYWORK_REWARD_HOST", "127.0.0.1"),
        port=int(os.getenv("SKYWORK_REWARD_PORT", "8081")),
        workers=1,
    )
