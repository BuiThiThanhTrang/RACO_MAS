from __future__ import annotations

import hashlib
import json
from pathlib import Path
import shutil
import tempfile
import unittest

from routing_benchmark.v4.artifacts.manifest import load_manifest
from routing_benchmark.v4.execution.registry import live_adapter_for_manifest
from routing_benchmark.v4.execution.shared_actor import RoleTransportError, SharedRoleActor


PROJECT = Path(__file__).resolve().parents[1]
RELEASE = PROJECT / "data/RoutingBench/v4_beta_4d_82_20261102"


CASES = {
    "database_migration_integrity": (
        "db-alembic-batchapplytest-test-add-ck",
        "SQL Migration Author",
    ),
    "code_debugging": ("code-bugsinpy-black-3", "Failure Diagnostician"),
    "formal_proof": ("proof-april-lme-0008eb23463d", "Tactic Synthesizer"),
    "structured_data_processing": (
        "sdp-spreadsheetbench-120-24",
        "Table Reshaping Specialist",
    ),
}


class _RejectedRequest(RuntimeError):
    status_code = 400


class _CaptureCompletions:
    def __init__(self) -> None:
        self.calls: list[dict] = []

    def create(self, **kwargs):
        self.calls.append(kwargs)
        raise _RejectedRequest("intentional sentinel capture")


class _CaptureClient:
    def __init__(self) -> None:
        self.chat = type("Chat", (), {})()
        self.chat.completions = _CaptureCompletions()


def _serialized(value) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, default=str)


class PrivateIsolationSentinelTests(unittest.TestCase):
    def test_private_sentinel_never_reaches_workspace_router_or_executor(self):
        temp_parent = PROJECT / ".test-tmp"
        temp_parent.mkdir(exist_ok=True)
        with tempfile.TemporaryDirectory(prefix="routingbench-sentinel-", dir=temp_parent) as temp:
            temp_root = Path(temp)
            for domain, (packet_id, action) in CASES.items():
                with self.subTest(domain=domain):
                    source = RELEASE / "packs" / domain / packet_id
                    packet = temp_root / "packets" / domain / packet_id
                    shutil.copytree(source, packet)

                    sentinel = f"ROUTINGBENCH_PRIVATE_SENTINEL::{domain}::{packet_id}"
                    sentinel_path = packet / "private/__sentinel__.txt"
                    sentinel_path.parent.mkdir(parents=True, exist_ok=True)
                    sentinel_path.write_text(sentinel, encoding="utf-8")
                    digest = hashlib.sha256(sentinel_path.read_bytes()).hexdigest()
                    manifest_value = json.loads((packet / "manifest.json").read_text(encoding="utf-8"))
                    manifest_value["private_files"].append(
                        {
                            "artifact_id": "private-sentinel",
                            "path": "private/__sentinel__.txt",
                            "media_type": "text/plain",
                            "sha256": digest,
                            "visibility": "private",
                        }
                    )
                    (packet / "manifest.json").write_text(
                        json.dumps(manifest_value, ensure_ascii=False, indent=2) + "\n",
                        encoding="utf-8",
                    )

                    manifest = load_manifest(packet)
                    adapter = live_adapter_for_manifest(manifest)
                    workspace = temp_root / "workspaces" / domain
                    env = adapter.materialize(packet, workspace)

                    workspace_text = "\n".join(
                        path.read_text(encoding="utf-8", errors="replace")
                        for path in workspace.rglob("*")
                        if path.is_file()
                    )
                    self.assertNotIn(sentinel, workspace_text)
                    self.assertFalse(any("private" in [part.casefold() for part in path.relative_to(workspace).parts]
                                         for path in workspace.rglob("*") if path.is_file()))

                    observation = adapter.public_observation(env)
                    self.assertNotIn(sentinel, _serialized(observation))

                    roles = json.loads((RELEASE / "agent_pools" / f"{domain}.json").read_text(encoding="utf-8"))
                    role = next(card for card in roles if card["agent_id"] == action)
                    context = adapter.executor_context(env, action, role)
                    self.assertNotIn(sentinel, _serialized(context))

                    client = _CaptureClient()
                    actor = SharedRoleActor(client, "sentinel-capture-model", transport_retries=0)
                    with self.assertRaises(RoleTransportError):
                        actor.execute(
                            env=env,
                            adapter=adapter,
                            action=action,
                            role_profile=role,
                            task=manifest.task,
                        )
                    self.assertEqual(len(client.chat.completions.calls), 1)
                    request = client.chat.completions.calls[0]
                    request_text = _serialized(request["messages"])
                    self.assertNotIn(sentinel, request_text)
                    self.assertNotIn("private/__sentinel__.txt", request_text.casefold())


if __name__ == "__main__":
    unittest.main()
