"""Exercise tool and persona behavior without importing API clients or calling providers."""
import ast
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import tempfile
import time
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import arxiv
import requests

ROOT = Path(__file__).resolve().parents[1]


def isolated_class(relative_path, name, namespace):
    # The package imports initialize provider clients. Load the production class
    # independently so these regressions do not require credentials or network.
    tree = ast.parse((ROOT / relative_path).read_text(encoding="utf-8"))
    node = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == name)
    node.decorator_list = []
    node.bases = []
    exec(compile(ast.Module(body=[node], type_ignores=[]), str(ROOT / relative_path), "exec"), namespace)
    return namespace[name]


class PersonaToolTests(unittest.TestCase):
    def test_actual_pool_recognized_by_actions(self):
        cls = isolated_class("puppeteer/inference/graph/agent_graph.py", "AgentGraph", {})
        graph = cls.__new__(cls)
        personas = [json.loads(line) for line in (ROOT / "puppeteer/personas/personas.jsonl").read_text().splitlines() if line.strip()]
        graph._nodes = [SimpleNamespace(index=i, role=p["name"], actions=p["actions"], model=p["model_type"], hash=str(i)) for i,p in enumerate(personas)]
        terminators = [n for n in graph._nodes if "terminate" in n.actions]
        self.assertEqual(len(terminators), 1)
        self.assertEqual(graph.terminator_agent_index, terminators[0].index)
        self.assertNotIn(terminators[0].role, graph.agent_prompt)
        search = [n.index for n in graph._nodes if any(a in ["search_bing", "search_arxiv", "access_website"] for a in n.actions)]
        self.assertEqual(len(search), 3)
        self.assertEqual(graph.search_agent_indices, search)

    def test_arxiv_uses_installed_library_api(self):
        cls = isolated_class("puppeteer/tools/web_search.py", "arXiv_SearchEngine", {"arxiv":arxiv, "requests":requests})
        tool = cls.__new__(cls)
        result = SimpleNamespace(title="Example paper", authors=[SimpleNamespace(name="Author")], summary="Summary", pdf_url="https://arxiv.org/pdf/example")
        with patch.object(arxiv.Client, "results", autospec=True, return_value=iter([result])) as query:
            ok, answer = tool.search("multi-agent learning")
        self.assertTrue(ok, answer)
        self.assertIn("Example paper", answer)
        self.assertEqual(query.call_args.args[1].query, "multi-agent learning")

    def python_tool(self):
        cls = isolated_class("puppeteer/tools/code_interpreter.py", "PythonInterpreter", {"os":os,"sys":sys,"subprocess":subprocess,"signal":signal,"time":time})
        return cls.__new__(cls)

    def test_python_tool_uses_active_interpreter_and_path_with_spaces(self):
        with tempfile.TemporaryDirectory(prefix="puppeteer tool ") as folder:
            path = Path(folder) / "agent-main.py"
            path.write_text("import sys; print(sys.executable)", encoding="utf-8")
            ok, output = self.python_tool().run(folder, str(path), "")
        self.assertTrue(ok, output)
        self.assertIn(sys.executable, output)

    def test_python_tool_timeout_is_failure(self):
        with patch("subprocess.Popen") as launch:
            launch.return_value.communicate.side_effect = subprocess.TimeoutExpired("python",10)
            launch.return_value.poll.return_value = 0
            ok, output = self.python_tool().run("unused", "agent-main.py", "")
        self.assertFalse(ok)
        self.assertIn("timed out", output)


if __name__ == "__main__":
    unittest.main()
