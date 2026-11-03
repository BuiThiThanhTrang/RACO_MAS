from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional

import yaml


class SearchProviderError(RuntimeError):
    """Raised when the configured web-search provider cannot serve a query."""


def load_search_config() -> Dict[str, Any]:
    config_path = Path(__file__).resolve().parents[2] / "config" / "global.yaml"
    with config_path.open("r", encoding="utf-8") as config_file:
        global_config = yaml.safe_load(config_file) or {}
    return dict(global_config.get("web_search") or {})


def build_web_search_client(
    *,
    config: Optional[Dict[str, Any]] = None,
    api_key: Optional[str] = None,
    request_kwargs: Optional[Dict[str, Any]] = None,
):
    """Build the configured provider while keeping imports dependency-light."""

    resolved = dict(config) if config is not None else load_search_config()
    provider = str(resolved.get("provider", "searxng")).strip().lower()
    if provider == "searxng":
        from tools.utils.searxng_client import SearxngSearchClient

        return SearxngSearchClient(
            config=resolved,
            request_kwargs=request_kwargs,
        )
    if provider == "searchapi":
        from tools.utils.searchapi_client import SearchApiBingClient

        return SearchApiBingClient(
            api_key=api_key,
            config=resolved,
            request_kwargs=request_kwargs,
        )
    raise SearchProviderError(
        f"Unsupported web search provider {provider!r}; expected 'searxng' or "
        "'searchapi'."
    )
