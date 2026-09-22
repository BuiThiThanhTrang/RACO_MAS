"""Replay one audited Hugging Face agent request with controlled output limits.

Example:
  python scripts/replay_audit_model_call.py \
    runs/<run_id>/audit/<task>/attempt-0001/events.jsonl \
    --model microsoft/Phi-4-mini-instruct:featherless-ai \
    --max-tokens 64 512 4096
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from openai import OpenAI

# Support direct execution: Python otherwise includes only scripts/ on sys.path.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from model.api_config import api_config


def _load_messages(events_path: Path, model: str) -> list[dict[str, str]]:
    events = [
        json.loads(line)
        for line in events_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    for event in reversed(events):
        if (
            event.get("event_type") == "model_call_started"
            and event.get("model") == model
        ):
            messages = event.get("messages")
            if not isinstance(messages, list) or not messages:
                raise ValueError(f"Audit event for {model} has no messages")
            return messages
    raise ValueError(f"No model_call_started event for {model!r} in {events_path}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("events", type=Path, help="Audit events.jsonl file")
    parser.add_argument("--model", required=True)
    parser.add_argument("--max-tokens", type=int, nargs="+", default=[64, 512, 4096])
    parser.add_argument("--temperature", type=float, default=0.0)
    args = parser.parse_args()

    provider = api_config.get_provider("huggingface_router")
    if not provider or not provider.get("api_key"):
        raise RuntimeError("HF_TOKEN is required for the Hugging Face Router")

    messages = _load_messages(args.events, args.model)
    client = OpenAI(
        api_key=provider["api_key"],
        base_url=provider["base_url"],
        max_retries=0,
    )
    print(
        json.dumps(
            {
                "model": args.model,
                "messages": len(messages),
                "message_characters": sum(len(str(item.get("content", ""))) for item in messages),
            },
            ensure_ascii=False,
        )
    )
    failed = False
    for max_tokens in args.max_tokens:
        try:
            response = client.chat.completions.create(
                model=args.model,
                messages=messages,
                max_tokens=max_tokens,
                temperature=args.temperature,
                stream=False,
            )
            print(
                json.dumps(
                    {
                        "max_tokens": max_tokens,
                        "status": "ok",
                        "finish_reason": response.choices[0].finish_reason,
                        "response_chars": len(response.choices[0].message.content or ""),
                        "usage": response.usage.model_dump() if response.usage else None,
                    },
                    ensure_ascii=False,
                )
            )
        except Exception as error:
            failed = True
            print(
                json.dumps(
                    {
                        "max_tokens": max_tokens,
                        "status": "error",
                        "error_type": type(error).__name__,
                        "error": str(error),
                    },
                    ensure_ascii=False,
                )
            )
    return int(failed)


if __name__ == "__main__":
    raise SystemExit(main())
