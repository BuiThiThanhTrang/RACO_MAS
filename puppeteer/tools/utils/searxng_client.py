from __future__ import annotations

import hashlib
import json
import os
import tempfile
import time
from pathlib import Path
from typing import Any, Callable, Dict, Optional

import requests

from tools.utils.web_search_client import SearchProviderError, load_search_config


DEFAULT_ENDPOINT = "http://127.0.0.1:8080/search"


class SearxngSearchClient:
    """Client for a private SearXNG JSON endpoint; no search API key required."""

    def __init__(
        self,
        config: Optional[Dict[str, Any]] = None,
        request_kwargs: Optional[Dict[str, Any]] = None,
    ) -> None:
        self.config = dict(config) if config is not None else load_search_config()
        self.request_kwargs = dict(request_kwargs or {})

    def _endpoint(self) -> str:
        endpoint_env = str(self.config.get("endpoint_env", "SEARXNG_URL"))
        endpoint = os.getenv(endpoint_env) or self.config.get(
            "endpoint", DEFAULT_ENDPOINT
        )
        endpoint = str(endpoint).rstrip("/")
        if not endpoint.endswith("/search"):
            endpoint += "/search"
        return endpoint

    def _params(self, query: str) -> Dict[str, Any]:
        params: Dict[str, Any] = {
            "q": query,
            "format": "json",
            "categories": self.config.get("categories", "general"),
            "language": self.config.get("language", "en"),
            "safesearch": int(self.config.get("safesearch", 0)),
            "pageno": 1,
        }
        engines = self.config.get("engines")
        if engines:
            params["engines"] = (
                ",".join(map(str, engines))
                if isinstance(engines, (list, tuple))
                else str(engines)
            )
        return params

    def _cache_path(self, endpoint: str, params: Dict[str, Any]) -> Optional[Path]:
        cache = dict(self.config.get("cache") or {})
        if not bool(cache.get("enabled", False)):
            return None
        root = Path(str(cache.get("path", "cache/web_search/searxng")))
        if not root.is_absolute():
            root = Path(__file__).resolve().parents[2] / root
        payload = json.dumps(
            {"endpoint": endpoint, "params": params},
            ensure_ascii=False,
            sort_keys=True,
        )
        return root / f"{hashlib.sha256(payload.encode('utf-8')).hexdigest()}.json"

    def _read_cache(self, cache_path: Optional[Path]) -> Optional[Dict[str, Any]]:
        if cache_path is None or not cache_path.is_file():
            return None
        ttl = float((self.config.get("cache") or {}).get("ttl_seconds", 86400))
        if ttl > 0 and time.time() - cache_path.stat().st_mtime > ttl:
            return None
        try:
            value = json.loads(cache_path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return None
        return value if isinstance(value, dict) else None

    @staticmethod
    def _write_cache(cache_path: Optional[Path], results: Dict[str, Any]) -> None:
        if cache_path is None:
            return
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=cache_path.parent,
            prefix=f".{cache_path.stem}.",
            suffix=".tmp",
            delete=False,
        ) as temporary_file:
            json.dump(results, temporary_file, ensure_ascii=False)
            temporary_path = Path(temporary_file.name)
        temporary_path.replace(cache_path)

    def search_json(self, query: str) -> Dict[str, Any]:
        query = query.strip()
        if not query:
            raise SearchProviderError("Search query must not be empty.")
        provider = str(self.config.get("provider", "searxng")).lower()
        if provider != "searxng":
            raise SearchProviderError(
                f"Unsupported web search provider {provider!r}; expected 'searxng'."
            )

        endpoint = self._endpoint()
        params = self._params(query)
        cache_path = self._cache_path(endpoint, params)
        cached = self._read_cache(cache_path)
        if cached is not None:
            return cached

        request_kwargs = dict(self.request_kwargs)
        request_params = dict(request_kwargs.get("params") or {})
        request_params.update(params)
        request_kwargs["params"] = request_params
        headers = dict(request_kwargs.get("headers") or {})
        headers.setdefault("Accept", "application/json")
        headers.setdefault("User-Agent", "RACO-MAS/1.0 local-searxng-client")
        request_kwargs["headers"] = headers
        request_kwargs["stream"] = False
        timeout_seconds = float(self.config.get("timeout_seconds", 20))
        request_kwargs.setdefault("timeout", (5, timeout_seconds))

        max_retries = max(1, int(self.config.get("max_retries", 2)))
        retryable_statuses = {408, 425, 429, 500, 502, 503, 504}
        last_error: Optional[Exception] = None
        for attempt in range(max_retries):
            try:
                response = requests.get(endpoint, **request_kwargs)
            except requests.exceptions.RequestException as exc:
                last_error = exc
                if attempt + 1 < max_retries:
                    time.sleep(1)
                    continue
                break

            if response.status_code in retryable_statuses and attempt + 1 < max_retries:
                last_error = requests.exceptions.HTTPError(
                    f"SearXNG returned retryable HTTP {response.status_code}"
                )
                time.sleep(1)
                continue
            try:
                response.raise_for_status()
            except requests.exceptions.RequestException as exc:
                raise SearchProviderError(
                    f"SearXNG request failed with HTTP {response.status_code}: {exc}"
                ) from exc
            try:
                results = response.json()
            except ValueError as exc:
                raise SearchProviderError("SearXNG returned invalid JSON.") from exc
            if not isinstance(results, dict):
                raise SearchProviderError("SearXNG returned non-object JSON.")
            self._write_cache(cache_path, results)
            return results

        detail = str(last_error) if last_error is not None else "unknown transport error"
        raise SearchProviderError(
            f"Cannot reach local SearXNG at {endpoint}: {detail}. "
            "Start it with docker compose -f infra/searxng/compose.yaml up -d."
        ) from last_error

    def format_results(
        self,
        query: str,
        results: Dict[str, Any],
        previous_visit: Optional[Callable[[str], str]] = None,
    ) -> str:
        previous_visit = previous_visit or (lambda _url: "")
        snippets = []
        max_results = max(1, int(self.config.get("max_results", 5)))
        for page in results.get("results", []):
            title = page.get("title")
            url = page.get("url")
            if not title or not url:
                continue
            engines = page.get("engines") or []
            engine_note = f"\nSources: {', '.join(map(str, engines))}" if engines else ""
            published = page.get("publishedDate") or page.get("published_date")
            date_note = f"\nDate published: {published}" if published else ""
            snippets.append(
                f"{len(snippets) + 1}. [{title}]({url})\n"
                f"{previous_visit(url)}{page.get('content') or ''}{date_note}{engine_note}"
            )
            if len(snippets) >= max_results:
                break
        if not snippets:
            return ""
        return (
            f"A local SearXNG search for '{query}' found {len(snippets)} results:\n\n"
            + "\n\n".join(snippets)
        )

    def search(self, query: str) -> str:
        return self.format_results(query, self.search_json(query))
