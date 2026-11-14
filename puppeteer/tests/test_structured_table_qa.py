import json
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

from routing_benchmark.v4.artifacts.manifest import load_manifest
from routing_benchmark.routers import DecisionModelRouter, RoutingDecision, StructuredLLMRouter
from routing_benchmark.v4.execution.adapters.common import read_json
from routing_benchmark.v4.execution.adapters.table_qa import ROLE_TO_TOOL, TOOLS, score_answer, typed_cell
from routing_benchmark.v4.execution.final_evaluation import audit_domain
from routing_benchmark.v4.execution.registry import live_adapter_for_manifest
from routing_benchmark.v4.execution.shared_actor import SharedRoleActor
from routing_benchmark.v4.execution.shared_evaluation import table_qa_metrics
from routing_benchmark.v4.execution.shared_runner import SharedTrajectoryRunner
from routing_benchmark.v4.execution.table_qa_policy import (
    TableQAFixedPolicy,
    TableQAPredefinedPolicy,
    TableQASingleAgentPolicy,
)


PROJECT = Path(__file__).resolve().parents[1]
ROOT = PROJECT / "data/RoutingBench/structured_table_qa_v1_40"
DOMAIN = "structured_data_processing"
PACKS = ROOT / "packs" / DOMAIN
TABLE = {"table_id": "table"}


def lookup(column, *rows, **options):
    return {"operation": "lookup", "column": column, "row_ids": list(rows), **options}


def filtered(operation, column, conditions, **options):
    return {"operation": operation, "column": column, **options}, conditions


WTQ = {
    1: lookup("c3", "r2"),
    3: lookup("c5", "r12"),
    4: filtered("count", "c6", [("c6", "eq", 1)]),
    5: filtered("lookup", "c2", [("c4", "eq", "1st")]),
    6: filtered("count", "c4", [("c4", "eq", "Kannada")]),
    7: filtered("lookup", "c9", [("c5", "eq", "Monterrey Flash")]),
    8: filtered("lookup", "c1", [("c3", "eq", "")]),
    9: filtered("min", "c1", [("c4", "eq", "1st")]),
    10: lookup("c1", "r35", "r36", "r37"),
    12: {"operation": "sum", "columns": ["c2", "c3", "c4", "c5", "c6"], "row_ids": ["r18"]},
    16: lookup("c2", "r13"),
    17: filtered("count", "c2", [("c1", "contains", "Oct")]),
    19: lookup("c13", "r9"),
    20: filtered("count", "c1", [("c1", "eq", "Winner"), ("c5", "eq", "Grass")]),
    21: filtered("argmax", "c3", [("c1", "ne", "Total")], return_column="c2"),
    22: filtered("sum", "c6", [("c3", "eq", "Belgium")]),
    24: lookup("c1", "r6"),
    31: lookup("c2", "r14"),
    32: filtered("count", "c4", [("c4", "eq", "1st")]),
    37: filtered("count", "c4", [("c4", "contains", "United")]),
    40: filtered("count", "c2", [("c2", "gt", 20)]),
    47: filtered("count", "c7", [("c7", "eq", "Jaguar XJS")]),
}
HITAB = {
    "100": lookup("c3", "r4", unit="%"),
    "1004": lookup("c4", "r3", unit="%"),
    "103_totto12249-2": lookup("c4", "r9"),
    "104_totto12294-0": lookup("c1", "r13"),
    "1113": {"operation": "argmax", "columns": ["c5", "c8", "c11"], "row_ids": ["r1"]},
    "1114": lookup("c5", "r1", unit="%"),
    "1123": {"operation": "argmin", "columns": ["c5", "c8", "c11"], "row_ids": ["r1"]},
    "115": lookup("c5", "r1", unit="%"),
    "1153": lookup("c4", "r1", unit="%"),
    "115_totto13338-2": (["c2", "c3", "c4"], 1),
    "1180": lookup("c8", "r15", unit="%"),
    "1276": (["c3", "c4"], 1),
    "1277": filtered("argmax", "c5", [("c2", "ne", "any")], return_column="c2"),
    "1340": lookup("c5", "r1"),
}
TABLEBENCH = {
    "2438086f1b3d0cb3f63d46f4c9eb8dbf": {"operation": "sum", "column": "c5"},
    "8e11fccdc147a157e7d7e9471759e877": filtered("mean", "c7", [("c4", "ge", 6)], round_digits=2),
    "3642133c0b09a25ffa48bd6356c3a58d": filtered("sum", "c5", [("c3", "gt", 20000)]),
    "9294abdf58d2fa73160b9131f16ec61d": {"operation": "difference", "column": "c5", "row_ids": ["r5", "r2"]},
}


class TableQATests(unittest.TestCase):
    def test_single_agent_and_predefined_baseline_contracts(self):
        path = PACKS / "tableqa-wtq-nu-1" / "manifest.json"
        single_roles = read_json(PROJECT / "config/shared_environment/table_qa_single_agent_role.json")
        adapter = live_adapter_for_manifest(load_manifest(path))
        with TemporaryDirectory(dir=PROJECT / "tests") as directory:
            env = adapter.materialize(path, Path(directory) / "tool-workspace")
            self.assertEqual(
                {item.name for item in adapter.tool_spec(env, "Table QA Agent")},
                set(TOOLS),
            )
        adapter = live_adapter_for_manifest(load_manifest(path))
        actor = SharedRoleActor(None, "scripted", max_tool_calls=1)
        actor._query = Mock(side_effect=[
            (json.dumps({"tool": "analyze_table", "arguments": {
                **TABLE, **lookup("c3", "r2"), "submit": True,
            }}), {}),
            (json.dumps({"tool": "verify_answer", "arguments": {}}), {}),
        ])
        with TemporaryDirectory(dir=PROJECT / "tests") as directory:
            folder = Path(directory)
            runner = SharedTrajectoryRunner(
                router=TableQASingleAgentPolicy(), actor=actor, max_calls=12,
            )
            report = runner.run(
                manifest_path=path,
                adapter=adapter,
                workspace=folder / "workspace",
                output_dir=folder,
                roles=single_roles,
            )
            self.assertTrue(report["passed"], report)
            self.assertEqual(report["executor_calls"], 2)
            decisions = [row["agent_id"] for row in (
                json.loads(line) for line in (folder / "decisions.jsonl").read_text(encoding="utf-8").splitlines()
            )]
            self.assertEqual(decisions, ["Table QA Agent", "Table QA Agent", "STOP"])

        predefined = TableQAPredefinedPolicy()
        predefined.reset(42)
        observation = {"current_state": {
            "terminal": False,
            "current": {},
            "artifacts": {},
            "tables": [{"hierarchical": False}],
        }}
        self.assertEqual(
            predefined.select(observation, [*ROLE_TO_TOOL, "STOP"]).action,
            "Table Inspector",
        )

    def test_router_adapters_receive_same_contract_error_and_executor_can_retry(self):
        path = PACKS / "tableqa-wtq-nu-22" / "manifest.json"
        roles = read_json(PROJECT / "config/shared_environment/table_qa_roles.json")
        reference = None
        for model in ("typesafe/jev-1.13", "nace-ai/drex-v1.5", "gpt-6-sol-openrouter"):
            with self.subTest(router=model), TemporaryDirectory(dir=PROJECT / "tests") as directory:
                observed = []
                actions = iter(["Evidence Selector", "Numerical Analyst", "Evidence Verifier",
                                "Evidence Verifier", "Table Inspector", "STOP"])
                if model == "gpt-6-sol-openrouter":
                    def query(_model, messages, *_args, **_kwargs):
                        observed.append(json.loads(messages[-1]["content"])["observation"])
                        return {"action": next(actions), "rationale": "scripted"}, 0

                    with patch.dict("sys.modules", {"model.query_manager": SimpleNamespace(
                        query_manager=SimpleNamespace(query_structured=query),
                    )}):
                        router = StructuredLLMRouter(model, observable_rejection_guard_enabled=False)
                else:
                    def post(payload):
                        self.assertEqual(payload["model"], model)
                        observed.append(payload["state"])
                        return {"answers": {"next_agent": next(actions)}}, 0

                    with patch.dict("os.environ", {"TABLE_QA_TEST_ROUTER_KEY": "test-only"}):
                        router = DecisionModelRouter({"model": model, "provider": "openrouter",
                                                      "api_key_env": "TABLE_QA_TEST_ROUTER_KEY"},
                                                     observable_rejection_guard_enabled=False)
                    self.addCleanup(router.client._client.close)
                    router.client._post = Mock(side_effect=post)

                responses = iter([
                    ("select_evidence", {**TABLE, "columns": ["c3", "c6"],
                                         "filters": [{"column": "c3", "op": "eq", "value": "Belgium"}]}),
                    ("analyze_table", {**TABLE, "column": "c6", "operation": "sum", "submit": True}),
                    ("verify_answer", {"issue": ""}),
                    ("verify_answer", {}),
                    ("inspect_table", TABLE),
                ])
                executor_contexts = []

                def reply(messages):
                    context = json.loads(messages[-1]["content"].split("\n", 1)[1])["public_context"]
                    executor_contexts.append(context)
                    tool, arguments = next(responses)
                    if tool == "analyze_table":
                        arguments["selection_id"] = context["current"]["selection"]
                    return json.dumps({"tool": tool, "arguments": arguments}), {}

                actor = SharedRoleActor(None, "scripted", max_tool_calls=1)
                actor._query = Mock(side_effect=reply)
                folder = Path(directory)
                runner = SharedTrajectoryRunner(router=router, actor=actor, max_calls=12)
                report = runner.run(manifest_path=path, adapter=live_adapter_for_manifest(load_manifest(path)),
                                    workspace=folder / "workspace", output_dir=folder, roles=roles)
                self.assertTrue(report["passed"], report)
                self.assertEqual(report["executor_calls"], 5)
                self.assertEqual(len(observed), 6)
                retry = observed[3]
                event = retry["recent_events"][-1]
                self.assertEqual(event["status"], "invalid_output")
                self.assertEqual(event["semantic_delta"], "no_state_change")
                self.assertNotIn("result", event)
                self.assertEqual(event["contract_error"]["field"], "arguments.issue")
                self.assertEqual(event["contract_error"], retry["current_state"]["last_error"])
                self.assertEqual(event["contract_error"], executor_contexts[3]["last_error"])
                self.assertEqual(retry["current_state"]["verification_status"]["status"], "not_checked")
                self.assertEqual(observed[4]["current_state"]["verification_status"]["status"], "passed")
                self.assertIsNone(observed[4]["current_state"]["last_error"])
                self.assertIn("STOP", retry["constraints"]["allowed_actions"])
                self.assertNotIn("private", {entry.name for entry in (folder / "workspace").iterdir()})
                if reference is None:
                    reference = observed
                else:
                    self.assertEqual(reference, observed)

    def test_executor_sees_selection_size_and_cited_older_inspection(self):
        path = PACKS / "tableqa-wtq-nu-6" / "manifest.json"
        adapter = live_adapter_for_manifest(load_manifest(path))
        with TemporaryDirectory(dir=PROJECT / "tests") as directory:
            env = adapter.materialize(path, Path(directory) / "workspace")
            inspection = adapter.execute_tool(env, "inspect_table", TABLE).output["artifact_id"]
            selection = adapter.execute_tool(env, "select_evidence", {
                **TABLE, "columns": ["c4"], "evidence_ids": [inspection],
                "filters": [{"column": "c4", "op": "eq", "value": "Kannada"}],
            }).output["artifact_id"]
            answer = adapter.execute_tool(env, "analyze_table", {
                **TABLE, "operation": "count", "selection_id": selection, "submit": True,
            }).output["artifact_id"]
            for offset in range(1, 5):
                adapter.execute_tool(env, "inspect_table", {**TABLE, "offset": offset})
            context = adapter.executor_context(env, "Evidence Verifier")
            self.assertIn(inspection, context["artifacts"])
            self.assertEqual(context["verification_status"]["answer_id"], answer)
            preview = context["artifacts"][selection]["data"]
            self.assertEqual((preview["shown_row_count"], preview["row_count"]), (6, 15))
            self.assertTrue(preview["rows_truncated"])

    def test_public_verifier_uses_explicit_percent_header_and_still_checks_value(self):
        path = PACKS / "tableqa-hitab-100" / "manifest.json"
        adapter = live_adapter_for_manifest(load_manifest(path))
        with TemporaryDirectory(dir=PROJECT / "tests") as directory:
            env = adapter.materialize(path, Path(directory) / "workspace")
            for value, expected in ((52.1, "passed"), (999, "failed")):
                submission = adapter.execute_tool(env, "submit_answer", {
                    "answer": [{"type": "number", "value": value, "unit": "%"}],
                    "preview_cells": [{**TABLE, "row_id": "r4", "column": "c3"}],
                })
                self.assertEqual(submission.status, "success")
                self.assertEqual(adapter.public_observation(env)["verification_status"]["status"], "not_checked")
                adapter.execute_tool(env, "verify_answer", {})
                verification = adapter.public_observation(env)["verification_status"]
                self.assertEqual(verification["status"], expected)
                self.assertEqual(verification["answer_id"], submission.output["artifact_id"])

    def test_all_reference_paths_through_public_tools_and_trusted_evaluator(self):
        index = read_json(ROOT / "packet_index.json")
        self.assertEqual(index["counts"], {"wtq": 22, "hitab": 14, "tablebench": 4})
        plans = {"wtq": {f"nu-{key}": value for key, value in WTQ.items()}, "hitab": HITAB, "tablebench": TABLEBENCH}
        with TemporaryDirectory(dir=PROJECT / "tests") as directory:
            for packet in index["packets"]:
                packet_id = packet["packet_id"]
                with self.subTest(packet=packet_id):
                    path = PACKS / packet_id / "manifest.json"
                    adapter = live_adapter_for_manifest(load_manifest(path))
                    folder = Path(directory) / packet_id
                    env = adapter.materialize(path, folder / "workspace")
                    self.assertFalse((env.workspace_dir / "private").exists())

                    def call(name, args):
                        result = adapter.execute_tool(env, name, args)
                        self.assertEqual(result.status, "success", result.output)
                        return result.output

                    plan = plans[packet["source"]][packet["source_id"]]
                    if isinstance(plan, tuple) and isinstance(plan[0], list):
                        columns, level = plan
                        evidence = call("resolve_headers", {**TABLE, "columns": columns})
                        answer = [typed_cell(column["header_path"][level]) for column in evidence["data"]["columns"]]
                        call("submit_answer", {"answer": answer, "evidence_ids": [evidence["artifact_id"]]})
                    else:
                        args = {**TABLE, "submit": True}
                        if isinstance(plan, tuple):
                            operation, filters = plan
                            columns = list(dict.fromkeys(operation[key] for key in ("column", "return_column") if key in operation))
                            selection = call("select_evidence", {**TABLE, "columns": columns, "filters": [
                                {"column": col, "op": op, "value": value} for col, op, value in filters
                            ]})
                            args["selection_id"] = selection["artifact_id"]
                        else:
                            operation = plan
                        call("analyze_table", {**args, **operation})
                    call("verify_answer", {})
                    self.assertTrue(adapter.terminal_status(env)["terminal"])
                    audit = audit_domain(folder, {"packet_root": str(ROOT), "domain": DOMAIN}, PROJECT)
                    self.assertTrue(audit["hard_pass"], audit)

    def test_shared_runner_recovers_after_verifier_rejects_wrong_scope(self):
        path = PACKS / "tableqa-hitab-100" / "manifest.json"
        adapter = live_adapter_for_manifest(load_manifest(path))
        roles = read_json(PROJECT / "config/shared_environment/table_qa_roles.json")
        actor = SharedRoleActor(None, "scripted")
        responses = iter([
            ("inspect_table", TABLE),
            ("resolve_headers", {**TABLE, "columns": ["c3"]}),
            ("select_evidence", {**TABLE, "columns": ["c3"], "row_ids": ["r1"]}),
            ("analyze_table", {**TABLE, "operation": "lookup", "column": "c3", "unit": "%", "submit": True}),
            ("verify_answer", {"issue": "Selected the input and service supply row instead of food service."}),
            ("diagnose_failure", {"step": "selection", "problem": "Wrong row for the sector in the question."}),
            ("select_evidence", {**TABLE, "columns": ["c3"], "row_ids": ["r4"]}),
            ("analyze_table", {**TABLE, "operation": "lookup", "column": "c3", "unit": "%", "submit": True}),
            ("verify_answer", {}),
        ])

        def reply(messages):
            prompt = json.loads(messages[-1]["content"].split("\n", 1)[1])
            self.assertNotIn("linked_cells", json.dumps(prompt))
            tool, arguments = next(responses)
            state = prompt["public_context"]
            if tool == "analyze_table":
                arguments["selection_id"] = state["current"]["selection"]
            return json.dumps({"tool": tool, "arguments": arguments}), {"finish_reason": "stop"}

        actor._query = Mock(side_effect=reply)
        with TemporaryDirectory(dir=PROJECT / "tests") as directory:
            folder = Path(directory) / path.parent.name
            runner = SharedTrajectoryRunner(router=TableQAFixedPolicy(), actor=actor, max_calls=12)
            report = runner.run(manifest_path=path, adapter=adapter, workspace=folder / "workspace",
                                output_dir=folder, roles=roles)
            self.assertTrue(report["passed"], report)
            self.assertEqual(report["executor_calls"], 9)
            logs = {
                key: [json.loads(line) for line in (folder / filename).read_text(encoding="utf-8").splitlines()]
                for key, filename in {
                    "states": "decision_states.jsonl", "decisions": "decisions.jsonl", "events": "trajectory.jsonl",
                }.items()
            }
            after_selection = logs["states"][7]["observation"]["current_state"]
            self.assertFalse(after_selection["artifacts"][after_selection["current"]["calculation"]]["fresh"])
            self.assertFalse(after_selection["terminal"])
            metrics = table_qa_metrics(**logs)
            self.assertTrue(metrics["ready_after_public_failure"])
            self.assertTrue(metrics["ready_with_budget"])
            self.assertEqual(metrics["calls_after_first_ready"], 0)
            audit = audit_domain(folder, {"packet_root": str(ROOT), "domain": DOMAIN}, PROJECT)
            self.assertTrue(audit["hard_pass"], audit)

    def test_answer_types_units_and_order(self):
        gold = {"answer": [{"type": "number", "value": "25", "unit": "%"}]}
        self.assertTrue(score_answer([{"type": "number", "value": 25, "unit": "%"}], gold))
        self.assertFalse(score_answer([{"type": "number", "value": 25}], gold))
        gold = {"answer": [typed_cell("A"), typed_cell("B")], "order": "sequence"}
        self.assertFalse(score_answer([typed_cell("B"), typed_cell("A")], gold))
        gold["order"] = "set"
        self.assertTrue(score_answer([typed_cell("B"), typed_cell("A")], gold))

    def test_direct_paths_can_skip_inspection_and_selection(self):
        path = PACKS / "tableqa-wtq-nu-1" / "manifest.json"
        roles = read_json(PROJECT / "config/shared_environment/table_qa_roles.json")
        for role, tool, arguments in (
            ("Numerical Analyst", "analyze_table", {**TABLE, **lookup("c3", "r2"), "submit": True}),
            ("Answer Composer", "submit_answer", {
                "answer": [typed_cell(100000)],
                "preview_cells": [{**TABLE, "row_id": "r2", "column": "c3"}],
            }),
        ):
            with self.subTest(role=role), TemporaryDirectory(dir=PROJECT / "tests") as directory:
                adapter = live_adapter_for_manifest(load_manifest(path))
                router = Mock()
                router.select.side_effect = [
                    RoutingDecision(role), RoutingDecision("Evidence Verifier"), RoutingDecision("STOP"),
                ]
                actor = SharedRoleActor(None, "scripted")
                actor._query = Mock(side_effect=[
                    (json.dumps({"tool": tool, "arguments": arguments}), {}),
                    (json.dumps({"tool": "verify_answer", "arguments": {}}), {}),
                ])
                folder = Path(directory)
                report = SharedTrajectoryRunner(router=router, actor=actor, max_calls=12).run(
                    manifest_path=path, adapter=adapter, workspace=folder / "workspace", output_dir=folder, roles=roles,
                )
                self.assertTrue(report["passed"], report)
                self.assertEqual(report["executor_calls"], 2)
