from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT = Path(__file__).resolve().parents[1]
if str(PROJECT) not in sys.path:
    sys.path.insert(0, str(PROJECT))

from routing_benchmark.roleroutebench.track_a.labeling import (
    CodexExecClient,
    OpenAIResponsesClient,
    OpenRouterChatClient,
    run_prelabeling,
)


def main() -> int:
    parser = argparse.ArgumentParser(description="Pre-label the frozen 100-state Track A sample")
    parser.add_argument("--root", type=Path, default=PROJECT / "data" / "RoleRouteBench" / "track_a_4d_live_v1")
    parser.add_argument("--provider", choices=("openai", "openrouter", "codex"), default="openai")
    parser.add_argument("--api-key-env", default="OPENAI_API_KEY")
    parser.add_argument("--base-url")
    parser.add_argument("--sample-dir", default="labeling")
    parser.add_argument("--max-attempts", type=int, default=2)
    args = parser.parse_args()
    if args.provider == "openai":
        client = OpenAIResponsesClient(api_key_env=args.api_key_env, base_url=args.base_url)
    elif args.provider == "openrouter":
        key_env = args.api_key_env if args.api_key_env != "OPENAI_API_KEY" else "OPENROUTER_API_KEY"
        client = OpenRouterChatClient(
            api_key_env=key_env,
            base_url=args.base_url or "https://openrouter.ai/api/v1",
        )
    else:
        client = CodexExecClient()
    sample_dir = args.root / args.sample_dir
    summary = run_prelabeling(
        sample_dir / "sample_states.jsonl",
        sample_dir,
        client,
        max_attempts=args.max_attempts,
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0 if summary["failures_this_run"] == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())

