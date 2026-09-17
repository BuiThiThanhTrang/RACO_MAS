"""Validate agent-provider routing and optionally make small live requests."""
import argparse
import os

from model.api_config import api_config
from model.model_config import model_registry


POOL_MODELS = (
    "qwen-2.5-7b",
    "qwen-2.5-14b",
    "llama-3.1-8b",
    "llama-3.2-3b",
    "mistral-nemo-12b",
    "mistralai/ministral-3b-2512",
)


def main():
    parser = argparse.ArgumentParser(description="Check OpenRouter/Hugging Face agent models")
    parser.add_argument("--live", action="store_true", help="send one short request to each model")
    parser.add_argument("--model", choices=POOL_MODELS, action="append",
                        help="check only this model; repeat for multiple models")
    args = parser.parse_args()
    aliases = args.model or POOL_MODELS

    missing = []
    for alias in aliases:
        config = model_registry.get_model_config(alias)
        settings = api_config.provider_settings(config.provider, config.url)
        has_key = bool(settings["api_key"])
        print({"alias": alias, "provider": config.provider,
               "model": config.api_model_name, "base_url": settings["base_url"],
               "credential": settings["api_key_env"], "credential_present": has_key})
        if not has_key:
            missing.append(settings["api_key_env"])

    if missing:
        raise SystemExit("Missing environment variables: " + ", ".join(sorted(set(missing))))
    if not args.live:
        print("Configuration is valid. Add --live to send provider requests.")
        return

    failures = []
    for alias in aliases:
        config = model_registry.get_model_config(alias)
        try:
            client = api_config.create_client(config.provider, config.url)
            response = client.chat.completions.create(
                model=config.api_model_name,
                messages=[{"role": "user", "content": "Reply with exactly: OK"}],
                temperature=0,
                max_tokens=8,
            )
            answer = response.choices[0].message.content
            print({"alias": alias, "provider": config.provider, "response": answer})
        except Exception as exc:
            status = getattr(exc, "status_code", None)
            print({"alias": alias, "provider": config.provider,
                   "status": status, "error": str(exc)})
            failures.append(alias)
    if failures:
        raise SystemExit("Provider smoke test failed for: " + ", ".join(failures))


if __name__ == "__main__":
    main()
