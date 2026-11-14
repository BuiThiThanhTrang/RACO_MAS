from __future__ import annotations

import json
import hashlib
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable


PUBLIC_FORBIDDEN_KEYS = frozenset(
    {
        "pair_id",
        "condition",
        "fault_class",
        "fault_variant",
        "fault_origin",
        "valid_actions",
        "unsafe_actions",
        "preferred_actions",
        "action_utilities",
        "oracle_min_remaining_calls",
        "internal_scenario_id",
    }
)


def _nested_keys(value: Any) -> set[str]:
    if isinstance(value, dict):
        keys = {str(key) for key in value}
        for item in value.values():
            keys.update(_nested_keys(item))
        return keys
    if isinstance(value, list):
        keys: set[str] = set()
        for item in value:
            keys.update(_nested_keys(item))
        return keys
    return set()


def public_leakage_keys(value: Any) -> list[str]:
    """Return forbidden evaluator-side keys found in a public router payload."""

    return sorted(PUBLIC_FORBIDDEN_KEYS.intersection(_nested_keys(value)))


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                value = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON at {path}:{line_number}") from exc
            if not isinstance(value, dict):
                raise ValueError(f"Expected object at {path}:{line_number}")
            rows.append(value)
    return rows


def _index_unique(
    rows: Iterable[dict[str, Any]], key: str
) -> dict[str, dict[str, Any]]:
    index: dict[str, dict[str, Any]] = {}
    for row in rows:
        value = str(row[key])
        if value in index:
            raise ValueError(f"Duplicate {key}: {value}")
        index[value] = row
    return index


class JsonlOffsetIndex:
    """Small in-memory index for large JSONL artifacts."""

    def __init__(self, path: Path, key: str):
        self.path = path
        self.key = key
        self.offsets: dict[str, int] = {}
        fingerprints: dict[str, bytes] = {}
        with path.open("rb") as handle:
            while True:
                offset = handle.tell()
                line = handle.readline()
                if not line:
                    break
                if not line.strip():
                    continue
                try:
                    row = json.loads(line)
                except json.JSONDecodeError as exc:
                    raise ValueError(f"Invalid JSON near byte {offset} in {path}") from exc
                value = str(row[key])
                if value in self.offsets:
                    fingerprint = hashlib.sha256(line.rstrip()).digest()
                    if fingerprints[value] != fingerprint:
                        raise ValueError(
                            f"Conflicting duplicate {key}: {value} in {path}"
                        )
                    continue
                self.offsets[value] = offset
                fingerprints[value] = hashlib.sha256(line.rstrip()).digest()

    def get(self, value: str) -> dict[str, Any]:
        try:
            offset = self.offsets[value]
        except KeyError as exc:
            raise KeyError(f"Unknown {self.key}: {value}") from exc
        with self.path.open("rb") as handle:
            handle.seek(offset)
            row = json.loads(handle.readline())
        if not isinstance(row, dict):
            raise ValueError(f"Expected object for {self.key}={value}")
        return row

    def __len__(self) -> int:
        return len(self.offsets)


def _base_id_from_scenario(row: dict[str, Any]) -> str:
    instance_id = str(row.get("instance_id", ""))
    if ":" in instance_id:
        return instance_id.rsplit(":", 1)[0]
    scenario_id = str(row.get("scenario_id", ""))
    for suffix in (
        "-clean",
        "-explicit_invalid",
        "-silent_corruption",
        "-retryable_failure",
        "-conflicting_outputs",
    ):
        if scenario_id.endswith(suffix):
            return scenario_id[: -len(suffix)].removeprefix("v4-")
    raise ValueError(f"Cannot derive base workflow from {scenario_id!r}")


def _base_id_from_pair(pair_id: str) -> str:
    parts = pair_id.split(":", 2)
    if len(parts) != 3 or parts[0] != "track-c":
        raise ValueError(f"Invalid Track C pair id: {pair_id}")
    return parts[1]


class RoutingBenchReviewDataset:
    """Read-only view of the RoutingBench V4-beta human-review package."""

    REQUIRED_FILES = (
        "metadata.json",
        "base_workflows.jsonl",
        "scenarios.jsonl",
        "agents.jsonl",
        "public_inputs.jsonl",
        "private_labels.jsonl",
        "provenance.jsonl",
        "human_review_queue.jsonl",
        "post_fault_inputs.jsonl",
        "post_fault_runtime.jsonl",
        "post_fault_labels.jsonl",
        "post_fault_pair_manifest.jsonl",
        "post_fault_human_review_queue.jsonl",
    )

    def __init__(self, root: str | Path):
        self.root = Path(root).resolve()
        missing = [name for name in self.REQUIRED_FILES if not (self.root / name).is_file()]
        if missing:
            raise FileNotFoundError(
                f"Dataset directory is missing required V4 files: {', '.join(missing)}"
            )

        self.metadata = json.loads(
            (self.root / "metadata.json").read_text(encoding="utf-8")
        )
        if not str(self.metadata.get("version", "")).startswith("4."):
            raise ValueError("The review dashboard accepts RoutingBench V4 datasets only")

        self.base_workflows = _index_unique(
            read_jsonl(self.root / "base_workflows.jsonl"), "base_workflow_id"
        )
        self.scenarios = _index_unique(
            read_jsonl(self.root / "scenarios.jsonl"), "scenario_id"
        )
        self.agents = _index_unique(read_jsonl(self.root / "agents.jsonl"), "agent_id")
        self.provenance = _index_unique(
            read_jsonl(self.root / "provenance.jsonl"), "scenario_id"
        )
        self.review_queue = _index_unique(
            read_jsonl(self.root / "human_review_queue.jsonl"), "base_workflow_id"
        )

        private_rows = read_jsonl(self.root / "private_labels.jsonl")
        # Multiple internal counterfactual states may intentionally share one
        # leakage-safe public decision payload. Internal decision_id stays unique.
        self.private_labels = _index_unique(private_rows, "decision_id")
        decisions_by_scenario: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for row in private_rows:
            decisions_by_scenario[str(row["internal_scenario_id"])].append(row)
        self._decisions_by_scenario = {
            scenario_id: sorted(rows, key=lambda item: str(item["decision_id"]))
            for scenario_id, rows in decisions_by_scenario.items()
        }
        self.public_inputs = JsonlOffsetIndex(
            self.root / "public_inputs.jsonl", "decision_id"
        )

        scenarios_by_base: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for scenario in self.scenarios.values():
            scenarios_by_base[_base_id_from_scenario(scenario)].append(scenario)
        self._scenarios_by_base = {
            base_id: sorted(rows, key=lambda item: str(item.get("instance_id", "")))
            for base_id, rows in scenarios_by_base.items()
        }

        self.post_fault_inputs = JsonlOffsetIndex(
            self.root / "post_fault_inputs.jsonl", "checkpoint_id"
        )
        self.post_fault_runtime = _index_unique(
            read_jsonl(self.root / "post_fault_runtime.jsonl"), "checkpoint_id"
        )
        self.post_fault_labels = _index_unique(
            read_jsonl(self.root / "post_fault_labels.jsonl"), "checkpoint_id"
        )
        self.post_fault_pairs = _index_unique(
            read_jsonl(self.root / "post_fault_pair_manifest.jsonl"), "pair_id"
        )
        self.post_fault_review_queue = _index_unique(
            read_jsonl(self.root / "post_fault_human_review_queue.jsonl"), "pair_id"
        )

    @property
    def version(self) -> str:
        return str(self.metadata.get("version", "unknown"))

    def base_workflow_ids(
        self,
        *,
        split: str | None = None,
        domain: str | None = None,
        pattern: str | None = None,
        source_kind: str | None = None,
    ) -> list[str]:
        result: list[str] = []
        for base_id, row in self.base_workflows.items():
            if split and row.get("split") != split:
                continue
            if domain and row.get("task_domain") != domain:
                continue
            if pattern and row.get("workflow_pattern") != pattern:
                continue
            if source_kind and row.get("source_kind") != source_kind:
                continue
            result.append(base_id)
        return sorted(result)

    def track_c_pair_ids(
        self,
        *,
        split: str | None = None,
        domain: str | None = None,
        fault_class: str | None = None,
        pattern: str | None = None,
    ) -> list[str]:
        result: list[str] = []
        for pair_id, row in self.post_fault_pairs.items():
            if split and row.get("split") != split:
                continue
            if domain and row.get("task_domain") != domain:
                continue
            if fault_class and row.get("fault_class") != fault_class:
                continue
            if pattern and row.get("workflow_pattern") != pattern:
                continue
            result.append(pair_id)
        return sorted(result)

    def scenarios_for_base(self, base_workflow_id: str) -> list[dict[str, Any]]:
        if base_workflow_id not in self.base_workflows:
            raise KeyError(f"Unknown base workflow: {base_workflow_id}")
        return [dict(row) for row in self._scenarios_by_base.get(base_workflow_id, [])]

    def decisions_for_scenario(self, scenario_id: str) -> list[dict[str, Any]]:
        rows = self._decisions_by_scenario.get(scenario_id, [])
        result = []
        for label in rows:
            public_id = str(label["public_decision_id"])
            result.append(
                {
                    "public": self.public_inputs.get(public_id),
                    "private": dict(label),
                }
            )
        return result

    def base_bundle(self, base_workflow_id: str) -> dict[str, Any]:
        scenarios = self.scenarios_for_base(base_workflow_id)
        return {
            "base": dict(self.base_workflows[base_workflow_id]),
            "queue": dict(self.review_queue[base_workflow_id]),
            "scenarios": scenarios,
            "provenance": [
                dict(self.provenance.get(str(row["scenario_id"]), {}))
                for row in scenarios
            ],
        }

    def track_c_bundle(self, pair_id: str) -> dict[str, Any]:
        manifest = dict(self.post_fault_pairs[pair_id])
        fault_id = str(manifest["fault_checkpoint_id"])
        control_id = str(manifest["control_checkpoint_id"])
        return {
            "manifest": manifest,
            "queue": dict(self.post_fault_review_queue[pair_id]),
            "base_workflow": dict(self.base_workflows[_base_id_from_pair(pair_id)]),
            "fault": {
                "public": self.post_fault_inputs.get(fault_id),
                "runtime": dict(self.post_fault_runtime[fault_id]),
                "label": dict(self.post_fault_labels[fault_id]),
            },
            "control": {
                "public": self.post_fault_inputs.get(control_id),
                "runtime": dict(self.post_fault_runtime[control_id]),
                "label": dict(self.post_fault_labels[control_id]),
            },
        }

    def pair_match_report(self, pair_id: str) -> dict[str, bool]:
        bundle = self.track_c_bundle(pair_id)
        fault_public = bundle["fault"]["public"]["observations"]["structured"]
        control_public = bundle["control"]["public"]["observations"]["structured"]
        fault_runtime = bundle["fault"]["runtime"]
        control_runtime = bundle["control"]["runtime"]

        def pool_ids(payload: dict[str, Any]) -> list[str]:
            return sorted(
                str(item.get("agent_id"))
                for item in payload.get("agent_profiles", [])
            )

        return {
            "same_task": fault_public.get("task") == control_public.get("task"),
            "same_agent_pool": pool_ids(fault_public) == pool_ids(control_public),
            "same_prefix_depth": fault_runtime.get("state", {}).get("steps_used")
            == control_runtime.get("state", {}).get("steps_used"),
            "same_prefix_cost": fault_runtime.get("state", {}).get("cost_used")
            == control_runtime.get("state", {}).get("cost_used"),
            "same_remaining_budget": fault_public.get("budget", {}).get(
                "calls_remaining"
            )
            == control_public.get("budget", {}).get("calls_remaining"),
        }

    def filter_values(self) -> dict[str, list[str]]:
        bases = list(self.base_workflows.values())
        pairs = list(self.post_fault_pairs.values())
        return {
            "splits": sorted({str(row.get("split")) for row in bases}),
            "domains": sorted({str(row.get("task_domain")) for row in bases}),
            "patterns": sorted({str(row.get("workflow_pattern")) for row in bases}),
            "sources": sorted({str(row.get("source_kind")) for row in bases}),
            "fault_classes": sorted({str(row.get("fault_class")) for row in pairs}),
        }
