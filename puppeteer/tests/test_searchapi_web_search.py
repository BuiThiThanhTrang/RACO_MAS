import os
import unittest
from unittest.mock import Mock, patch

import requests

from tools.utils.searchapi_client import SearchApiBingClient, SearchProviderError
from tools.web_search import Bing_SearchEngine


class SearchApiWebSearchTests(unittest.TestCase):
    def _client(self, api_key="test-searchapi-key"):
        return SearchApiBingClient(
            api_key=api_key,
            config={
                "provider": "searchapi",
                "endpoint": "https://www.searchapi.io/api/v1/search",
                "engine": "bing",
                "max_results": 5,
                "timeout_seconds": 20,
                "max_retries": 2,
            },
        )

    def test_search_bing_action_uses_lightweight_client(self):
        engine = Bing_SearchEngine("search_bing_test")
        engine.client = Mock()
        engine.client.search.return_value = "formatted SearchApi result"

        success, content = engine.search("test query")

        self.assertTrue(success)
        self.assertEqual(content, "formatted SearchApi result")
        engine.client.search.assert_called_once_with("test query")

    @patch("tools.utils.searchapi_client.requests.get")
    def test_bing_action_calls_searchapi_and_formats_results(self, request_get):
        response = Mock(status_code=200)
        response.raise_for_status.return_value = None
        response.json.return_value = {
            "organic_results": [
                {
                    "position": 1,
                    "title": "Example result",
                    "link": "https://example.com/result",
                    "snippet": "Example snippet",
                }
            ],
            "top_stories": [
                {
                    "position": 1,
                    "title": "Example news",
                    "link": "https://example.com/news",
                    "snippet": "News snippet",
                    "date": "2h",
                }
            ],
        }
        request_get.return_value = response

        client = self._client()
        content = client.search("agent orchestration")

        request_get.assert_called_once()
        endpoint = request_get.call_args.args[0]
        kwargs = request_get.call_args.kwargs
        self.assertEqual(endpoint, "https://www.searchapi.io/api/v1/search")
        self.assertEqual(kwargs["headers"]["Authorization"], "Bearer test-searchapi-key")
        self.assertEqual(
            kwargs["params"],
            {"engine": "bing", "q": "agent orchestration", "num": 5},
        )
        self.assertIn("Example result", content)
        self.assertIn("https://example.com/result", content)
        self.assertIn("Example news", content)

    def test_missing_key_names_required_environment_variable(self):
        with patch.dict(os.environ, {}, clear=True):
            client = self._client(api_key=None)
            with self.assertRaisesRegex(SearchProviderError, "SEARCHAPI_API_KEY"):
                client.search("test")

    @patch("tools.utils.searchapi_client.requests.get")
    def test_retryable_failure_is_retried(self, request_get):
        failed = Mock(status_code=429)
        failed.raise_for_status.side_effect = requests.exceptions.HTTPError("rate limited")
        success = Mock(status_code=200)
        success.raise_for_status.return_value = None
        success.json.return_value = {"organic_results": []}
        request_get.side_effect = [failed, success]

        client = self._client()
        with patch("tools.utils.searchapi_client.time.sleep"):
            client.search("test")

        self.assertEqual(request_get.call_count, 2)


if __name__ == "__main__":
    unittest.main()
