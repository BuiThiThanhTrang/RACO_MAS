import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

import requests

from agent.reasoning_agent import Reasoning_Agent
from tools.utils.searxng_client import SearxngSearchClient
from tools.utils.web_search_client import (
    SearchProviderError,
    build_web_search_client,
    load_search_config,
)
from tools.web_search import Bing_SearchEngine, GeneralWebSearchEngine


class SearxngWebSearchTests(unittest.TestCase):
    def _config(self, **overrides):
        config = {
            "provider": "searxng",
            "endpoint": "http://127.0.0.1:8080/search",
            "endpoint_env": "SEARXNG_TEST_URL",
            "categories": "general",
            "language": "en",
            "safesearch": 0,
            "max_results": 5,
            "timeout_seconds": 20,
            "max_retries": 2,
            "cache": {"enabled": False},
        }
        config.update(overrides)
        return config

    @patch("tools.utils.searxng_client.requests.get")
    def test_search_calls_local_json_endpoint_without_api_key(self, request_get):
        response = Mock(status_code=200)
        response.raise_for_status.return_value = None
        response.json.return_value = {
            "results": [
                {
                    "title": "Example result",
                    "url": "https://example.com/result",
                    "content": "Example snippet",
                    "engines": ["duckduckgo", "brave"],
                }
            ]
        }
        request_get.return_value = response

        content = SearxngSearchClient(config=self._config()).search(
            "agent orchestration"
        )

        endpoint = request_get.call_args.args[0]
        kwargs = request_get.call_args.kwargs
        self.assertEqual(endpoint, "http://127.0.0.1:8080/search")
        self.assertEqual(kwargs["params"]["q"], "agent orchestration")
        self.assertEqual(kwargs["params"]["format"], "json")
        self.assertNotIn("Authorization", kwargs["headers"])
        self.assertIn("Example result", content)
        self.assertIn("duckduckgo, brave", content)

    @patch("tools.utils.searxng_client.requests.get")
    def test_retryable_failure_is_retried(self, request_get):
        failed = Mock(status_code=503)
        success = Mock(status_code=200)
        success.raise_for_status.return_value = None
        success.json.return_value = {"results": []}
        request_get.side_effect = [failed, success]

        with patch("tools.utils.searxng_client.time.sleep"):
            SearxngSearchClient(config=self._config()).search_json("test")

        self.assertEqual(request_get.call_count, 2)

    def test_transport_error_names_local_start_command(self):
        client = SearxngSearchClient(config=self._config(max_retries=1))
        with patch(
            "tools.utils.searxng_client.requests.get",
            side_effect=requests.exceptions.ConnectionError("offline"),
        ):
            with self.assertRaisesRegex(
                SearchProviderError, "infra/searxng/compose.yaml"
            ):
                client.search_json("test")

    def test_endpoint_can_be_overridden_by_environment(self):
        client = SearxngSearchClient(config=self._config())
        with patch.dict(
            os.environ,
            {"SEARXNG_TEST_URL": "http://127.0.0.1:8888/custom"},
            clear=False,
        ):
            self.assertEqual(
                client._endpoint(), "http://127.0.0.1:8888/custom/search"
            )

    @patch("tools.utils.searxng_client.requests.get")
    def test_disk_cache_reuses_response(self, request_get):
        response = Mock(status_code=200)
        response.raise_for_status.return_value = None
        response.json.return_value = {"results": []}
        request_get.return_value = response

        with tempfile.TemporaryDirectory() as temp_dir:
            config = self._config(
                cache={
                    "enabled": True,
                    "path": str(Path(temp_dir) / "search-cache"),
                    "ttl_seconds": 60,
                }
            )
            first = SearxngSearchClient(config=config).search_json("cached query")
            second = SearxngSearchClient(config=config).search_json("cached query")

        self.assertEqual(first, second)
        self.assertEqual(request_get.call_count, 1)

    def test_factory_and_provider_neutral_action(self):
        client = build_web_search_client(config=self._config())
        self.assertIsInstance(client, SearxngSearchClient)

        engine = GeneralWebSearchEngine("search_web_test")
        engine.client = Mock()
        engine.client.search.return_value = "local search result"
        success, content = engine.search("test query")

        self.assertTrue(success)
        self.assertEqual(content, "local search result")

    @patch("tools.utils.searxng_client.requests.get")
    def test_legacy_search_bing_alias_uses_global_searxng_provider(self, request_get):
        response = Mock(status_code=200)
        response.raise_for_status.return_value = None
        response.json.return_value = {
            "results": [
                {
                    "title": "Alias result",
                    "url": "https://example.com/alias",
                    "content": "SearXNG result",
                }
            ]
        }
        request_get.return_value = response

        self.assertEqual(load_search_config()["provider"], "searxng")
        with patch(
            "tools.utils.web_search_client.load_search_config",
            return_value=self._config(),
        ):
            engine = Bing_SearchEngine("search_bing_test")
            success, content = engine.search("legacy query")

        self.assertTrue(success)
        self.assertIn("Alias result", content)
        self.assertEqual(request_get.call_args.args[0], "http://127.0.0.1:8080/search")

    def test_search_actions_can_be_retrieval_only(self):
        agent = object.__new__(Reasoning_Agent)
        agent.runtime_config = {"web_search": {"retrieval_only": True}}

        self.assertTrue(agent._is_retrieval_only("search_web"))
        self.assertTrue(agent._is_retrieval_only("search_bing"))
        self.assertTrue(agent._is_retrieval_only("search_arxiv"))
        self.assertFalse(agent._is_retrieval_only("access_website"))

    def test_image_media_action_can_be_retrieval_only(self):
        agent = object.__new__(Reasoning_Agent)
        agent.runtime_config = {
            "media_tools": {"vision": {"retrieval_only": True}}
        }
        image_info = type("Info", (), {"file_extension": ".png"})()
        audio_info = type("Info", (), {"file_extension": ".mp3"})()

        self.assertTrue(agent._is_retrieval_only("inspect_media", image_info))
        self.assertFalse(agent._is_retrieval_only("inspect_media", audio_info))


if __name__ == "__main__":
    unittest.main()
