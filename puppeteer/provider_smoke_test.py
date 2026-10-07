"""Validate agent-provider routing and optionally make small live requests."""
import argparse
import os

from model.api_config import api_config
from model.model_config import model_registry
from utils.persona_adapter import load_runtime_personas


POOL_MODELS = (
    "qwen-3.5-9b",
    "gemma-3-12b-it",
    "llama-3.1-8b",
    "mistral-nemo-12b",
)


def main():
    parser = argparse.ArgumentParser(description="Check OpenRouter/Hugging Face agent models")
    parser.add_argument("--live", action="store_true", help="send one short request to each model")
    parser.add_argument("--model", choices=model_registry.list_available_models(), action="append",
                        help="check only this model; repeat for multiple models")
    parser.add_argument(
        "--personas",
        help="derive the model list from a legacy or schema-1.0 persona pool",
    )
    args = parser.parse_args()
    if args.model and args.personas:
        parser.error("--model and --personas cannot be used together")
    if args.personas:
        aliases = tuple(
            dict.fromkeys(
                persona["model_type"]
                for persona in load_runtime_personas(args.personas)
                if "terminate" not in persona["actions"]
            )
        )
    else:
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
            request = {
                "model": config.api_model_name,
                "messages": [{
                    "role": "user",
                    "content": "Reply with exactly OK and do not include reasoning.",
                }],
                "temperature": 0,
                "max_tokens": 256,
            }
            if config.extra_body:
                request["extra_body"] = config.extra_body
            response = client.chat.completions.create(
                **request
            )
            answer = response.choices[0].message.content
            print({"alias": alias, "provider": config.provider, "response": answer})
            if not isinstance(answer, str) or not answer.strip():
                raise RuntimeError("provider returned an empty response")
        except Exception as exc:
            status = getattr(exc, "status_code", None)
            print({"alias": alias, "provider": config.provider,
                   "status": status, "error": str(exc)})
            failures.append(alias)
    if failures:
        raise SystemExit("Provider smoke test failed for: " + ", ".join(failures))


if __name__ == "__main__":
    main()
