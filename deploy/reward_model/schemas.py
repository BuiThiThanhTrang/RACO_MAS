"""Version-one contract, independent of the GPU runtime."""
import math

DIM = 8192
MAX_TOKENS = 4096


class InputTooLong(ValueError):
    pass


def validate_request(payload):
    if not isinstance(payload, dict) or set(payload) != {"schema_version", "messages"}:
        raise ValueError("Expected schema_version and messages only")
    if type(payload["schema_version"]) is not int or payload["schema_version"] != 1:
        raise ValueError("Unsupported schema_version")
    messages = payload["messages"]
    if not isinstance(messages, list) or not 1 <= len(messages) <= 256:
        raise ValueError("messages must contain 1 to 256 entries")
    for message in messages:
        if not isinstance(message, dict) or set(message) != {"role", "content"}:
            raise ValueError("Each message must contain role and content only")
        if message["role"] not in ("system", "user", "assistant") or not isinstance(message["content"], str):
            raise ValueError("Invalid message role or content")
    if sum(len(m["content"]) for m in messages) > 65536:
        raise InputTooLong("Input exceeds the character limit; shorten the conversation")
    return messages


def make_response(reward, state, token_count, lock):
    if len(state) != DIM or not all(math.isfinite(x) for x in state) or not math.isfinite(reward):
        raise RuntimeError("Invalid model output")
    return {"schema_version": 1, "reward": float(reward),
            "last_hidden_state": [float(x) for x in state], "input_tokens": token_count,
            "model_revision": lock["gguf_revision"], "quantization": lock["quantization"]}
