import os
import time
from typing import Any, Callable, Dict, Optional

import requests

from tools.utils.web_search_client import SearchProviderError, load_search_config


DEFAULT_ENDPOINT = "https://www.searchapi.io/api/v1/search"


def _load_search_config() -> Dict[str, Any]:
    """Backward-compatible alias used by existing callers and tests."""

    return load_search_config()


class SearchApiBingClient:
    """Small SearchApi.io client used by the public `search_bing` action."""

    def __init__(
        self,
        api_key: Optional[str] = None,
        config: Optional[Dict[str, Any]] = None,
        request_kwargs: Optional[Dict[str, Any]] = None,
    ) -> None:
        self.config = dict(config) if config is not None else _load_search_config()
        api_key_env = self.config.get("api_key_env", "SEARCHAPI_API_KEY")
        self.api_key = api_key or os.getenv(api_key_env)
        self.request_kwargs = dict(request_kwargs or {})

    def search_json(self, query: str) -> Dict[str, Any]:
        query = query.strip()
        if not query:
            raise SearchProviderError("Search query must not be empty.")
        if not self.api_key:
            api_key_env = self.config.get("api_key_env", "SEARCHAPI_API_KEY")
            raise SearchProviderError(
                f"Missing SearchApi.io API key. Set the {api_key_env} environment variable."
            )

        provider = str(self.config.get("provider", "searchapi")).lower()
        if provider != "searchapi":
            raise SearchProviderError(
                f"Unsupported web search provider '{provider}'. Expected 'searchapi'."
            )

        request_kwargs = dict(self.request_kwargs)
        headers = dict(request_kwargs.get("headers") or {})
        headers["Authorization"] = f"Bearer {self.api_key}"
        headers.setdefault("Accept", "application/json")
        request_kwargs["headers"] = headers

        params = dict(request_kwargs.get("params") or {})
        params.update(
            {
                "engine": self.config.get("engine", "bing"),
                "q": query,
                "num": int(self.config.get("max_results", 5)),
            }
        )
        request_kwargs["params"] = params
        request_kwargs["stream"] = False
        timeout_seconds = float(self.config.get("timeout_seconds", 20))
        request_kwargs.setdefault("timeout", (5, timeout_seconds))

        endpoint = self.config.get("endpoint", DEFAULT_ENDPOINT)
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
                    f"SearchApi.io returned retryable HTTP {response.status_code}"
                )
                time.sleep(1)
                continue

            try:
                response.raise_for_status()
            except requests.exceptions.RequestException as exc:
                raise SearchProviderError(
                    f"SearchApi.io request failed with HTTP {response.status_code}: {exc}"
                ) from exc

            try:
                results = response.json()
            except ValueError as exc:
                raise SearchProviderError("SearchApi.io returned invalid JSON.") from exc
            if not isinstance(results, dict):
                raise SearchProviderError("SearchApi.io returned a non-object JSON response.")
            return results

        detail = str(last_error) if last_error is not None else "unknown transport error"
        raise SearchProviderError(f"SearchApi.io request failed: {detail}") from last_error

    @staticmethod
    def format_results(
        query: str,
        results: Dict[str, Any],
        previous_visit: Optional[Callable[[str], str]] = None,
    ) -> str:
        previous_visit = previous_visit or (lambda _url: "")
        web_snippets = []
        news_snippets = []
        index = 0

        for page in results.get("organic_results", []):
            title = page.get("title")
            url = page.get("link")
            if not title or not url:
                continue
            index += 1
            web_snippets.append(
                f"{index}. [{title}]({url})\n{previous_visit(url)}{page.get('snippet', '')}"
            )

        for page in results.get("top_stories", []):
            title = page.get("title")
            url = page.get("link")
            if not title or not url:
                continue
            index += 1
            date_published = f"\nDate published: {page['date']}" if page.get("date") else ""
            news_snippets.append(
                f"{index}. [{title}]({url})\n{previous_visit(url)}"
                f"{page.get('snippet', '')}{date_published}"
            )

        if not web_snippets and not news_snippets:
            return ""

        content = (
            f"A Bing search via SearchApi.io for '{query}' found "
            f"{len(web_snippets) + len(news_snippets)} results:\n\n"
            "## Web Results\n"
            + "\n\n".join(web_snippets)
        )
        if news_snippets:
            content += "\n\n## News Results:\n" + "\n\n".join(news_snippets)
        return content

    def search(self, query: str) -> str:
        return self.format_results(query, self.search_json(query))
