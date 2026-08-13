import argparse
import os
import sys


def main():
    parser = argparse.ArgumentParser(description="Check configured API provider profiles without printing secrets.")
    parser.add_argument("--profiles", nargs="*", default=None, help="Optional provider profile names to check.")
    args = parser.parse_args()

    project_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    os.chdir(project_dir)
    if project_dir not in sys.path:
        sys.path.insert(0, project_dir)

    from model.api_config import api_config

    providers = api_config.get("providers")
    if args.profiles:
        providers = {name: providers.get(name, {}) for name in args.profiles}

    missing = []
    for name, config in providers.items():
        api_key = config.get("api_key")
        api_key_env = config.get("api_key_env")
        base_url = config.get("base_url")
        headers = config.get("headers") or {}
        status = "OK" if api_key else "MISSING"
        print(f"{name}: {status}")
        print(f"  base_url: {base_url}")
        print(f"  api_key_env: {api_key_env}")
        print(f"  env_present: {bool(os.getenv(api_key_env)) if api_key_env else 'n/a'}")
        print(f"  headers_present: {sorted(headers.keys())}")
        if not api_key:
            missing.append((name, api_key_env))

    if missing:
        print("\nMissing API keys:")
        for name, api_key_env in missing:
            print(f"  {name}: set {api_key_env}")
        raise SystemExit(1)


if __name__ == "__main__":
    main()
