import os
import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

from role_aware.task_analyzer import TaskAnalyzerRepresentation


class TaskAnalyzerRemoteTests(unittest.TestCase):
    def test_remote_primary_uses_openai_embedding_contract(self):
        response = SimpleNamespace(
            data=[SimpleNamespace(embedding=[0.25] * 1024)]
        )
        client = Mock()
        client.embeddings.create.return_value = response
        config = {
            "primary": {
                "backend": "openai_compatible_embedding",
                "model": "BAAI/bge-large-en-v1.5",
                "dim": 1024,
                "base_url_env": "TASK_ANALYZER_BASE_URL",
                "api_key_env": "TASK_ANALYZER_API_KEY",
            }
        }

        with (
            patch.dict(
                os.environ,
                {
                    "TASK_ANALYZER_BASE_URL": "http://embedding.local/v1",
                    "TASK_ANALYZER_API_KEY": "test-key",
                },
                clear=False,
            ),
            patch("openai.OpenAI", return_value=client) as client_class,
        ):
            analyzer = TaskAnalyzerRepresentation(config)
            state, reward = analyzer("solve the task")

        client_class.assert_called_once_with(
            base_url="http://embedding.local/v1", api_key="test-key", max_retries=0
        )
        client.embeddings.create.assert_called_once_with(
            model="BAAI/bge-large-en-v1.5",
            input=["solve the task"],
            dimensions=1024,
            encoding_format="float",
        )
        self.assertEqual(analyzer._backend, "primary")
        self.assertEqual(tuple(state.shape), (1, 1024))
        self.assertEqual(reward, 0.0)

    def test_local_sentence_transformers_backend_is_rejected(self):
        analyzer = TaskAnalyzerRepresentation(
            {
                "primary": {
                    "backend": "sentence_transformers",
                    "model": "BAAI/bge-large-en-v1.5",
                }
            }
        )
        with self.assertRaisesRegex(ValueError, "remote-only"):
            analyzer("task")

    def test_missing_remote_credentials_fails_fast(self):
        analyzer = TaskAnalyzerRepresentation(
            {
                "primary": {
                    "backend": "openai_compatible_embedding",
                    "model": "BAAI/bge-large-en-v1.5",
                }
            }
        )
        with (
            patch.dict(os.environ, {}, clear=True),
            self.assertRaisesRegex(RuntimeError, "remote endpoint"),
        ):
            analyzer("task")


if __name__ == "__main__":
    unittest.main()
