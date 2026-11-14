import json
from pathlib import Path
from tempfile import TemporaryDirectory
from textwrap import indent
from types import SimpleNamespace
import unittest
from unittest.mock import Mock

from routing_benchmark.routers import RoutingDecision
from routing_benchmark.v4.execution.adapters.spreadsheetbench_python_oj import SpreadsheetBenchPythonOJAdapter
from routing_benchmark.v4.execution.final_evaluation import audit_domain
from routing_benchmark.v4.execution.shared_actor import SharedRoleActor
from routing_benchmark.v4.execution.shared_runner import SharedTrajectoryRunner

PROJECT = Path(__file__).resolve().parents[1]
PACKET_ROOT = PROJECT / "data/RoutingBench/v4_beta_4d_82_20261102"
DOMAIN = "structured_data_processing"
PACK = PACKET_ROOT / "packs" / DOMAIN / "sdp-spreadsheetbench-250-20"
AUTHOR = "Table Reshaping Specialist"
TEST = "Workbook Test Executor"
REPAIR = "Workbook Repair Specialist"
VALIDATE = "Workbook Integrity Validator"
BODY = """sheet = workbook["RNM"]
first_rows, totals, duplicates = {}, {}, []
for row in range(2, sheet.max_row + 1):
    if not any(sheet.cell(row, col).value is not None for col in range(1, 12)):
        continue
    key = (sheet.cell(row, 2).value, sheet.cell(row, 3).value)
    value = sheet.cell(row, 10).value
    if key in first_rows:
        duplicates.append(row)
    else:
        first_rows[key] = row
    totals[key] = totals.get(key, 0) + (value if isinstance(value, (int, float)) else 0)
for key, row in first_rows.items():
    sheet.cell(row, 10).value = totals[key]
for row in reversed(duplicates):
    sheet.delete_rows(row)
"""


class StructuredDataContractV2Tests(unittest.TestCase):
    def test_shared_runner_payloads_and_repair(self):
        roles = json.loads((PACKET_ROOT / "agent_pools" / f"{DOMAIN}.json").read_text())
        adapter = SpreadsheetBenchPythonOJAdapter()
        runtime = {"candidate_authoring": "transformation_body_v2"}
        adapter.configure_runtime(runtime)
        router = Mock()
        actor = SharedRoleActor(None, "scripted")
        steps = iter((AUTHOR, TEST, REPAIR, TEST, VALIDATE, "STOP"))
        observations = []
        prompts = []
        authored = iter(("return", 'workbook["absent"]', "```python\n" + BODY + "```"))

        def route(observation, actions):
            self.assertEqual(actions, [role["agent_id"] for role in roles] + ["STOP"])
            observations.append(observation)
            return RoutingDecision(next(steps))

        def reply(messages):
            instruction, payload = messages[1]["content"].split("\n", 1)
            prompt = json.loads(payload)
            prompts.append((instruction, prompt, messages))
            context = prompt["public_context"]
            artifacts = context["runtime_state"]["artifacts"]
            if prompt["role"]["agent_id"] in (AUTHOR, REPAIR):
                return next(authored), {}
            arguments = {"candidate_artifact_id": artifacts["candidate_solution"]["artifact_id"]}
            tool = "run_solution_public"
            if prompt["role"]["agent_id"] == VALIDATE:
                tool = "validate_public_output"
                arguments["execution_artifact_id"] = artifacts["public_execution"]["artifact_id"]
            return json.dumps({"tool": tool, "arguments": arguments}), {}

        router.select.side_effect = route
        actor._query = reply
        with TemporaryDirectory(dir=PROJECT / "tests") as directory:
            episode = Path(directory) / PACK.name
            report = SharedTrajectoryRunner(router=router, actor=actor, max_calls=12).run(
                manifest_path=PACK / "manifest.json", adapter=adapter, workspace=episode / "workspace",
                output_dir=episode, roles=roles,
            )
            self.assertFalse((episode / "workspace/private").exists())
            self.assertTrue(report["passed"], report)
            audit = audit_domain(episode, {"domain": DOMAIN, "packet_root": str(PACKET_ROOT), "runtime": runtime}, PROJECT)
            self.assertTrue(audit.get("hard_pass"), audit)
            self.assertTrue(audit["same_candidate_hash_all_cases"])

        self.assertEqual([row["budget"]["calls_remaining"] for row in observations], list(range(12, 6, -1)))
        first, written, failed, revised, executed, terminal = [row["current_state"] for row in observations]
        self.assertIn("RNM", first["public_case_summary"]["sheets"])
        self.assertEqual(first["candidate_contract"]["submission"]["mode"], "transformation_body_v2")
        self.assertIn("def solve", written["artifacts"]["candidate_solution"]["content_preview"])
        self.assertIn("return", written["latest_candidate_rejection"]["error"])
        self.assertEqual(failed["latest_execution"]["status"], "failed")
        self.assertTrue(failed["latest_execution"]["fresh"])
        candidate = revised["artifacts"]["candidate_solution"]
        self.assertTrue(candidate["fresh"])
        self.assertEqual(candidate["base_candidate_artifact_id"], failed["artifacts"]["candidate_solution"]["artifact_id"])
        self.assertEqual(candidate["evidence_refs"], [failed["latest_execution"]["artifact_id"]])
        self.assertFalse(revised["latest_execution"]["fresh"])
        self.assertTrue(executed["latest_execution"]["fresh"])
        self.assertFalse(executed["terminal"])
        self.assertTrue(terminal["terminal"])

        instruction, author, _ = prompts[0]
        self.assertIn("Return Python directly", instruction)
        self.assertIn("RNM", [sheet["name"] for sheet in author["public_context"]["public_case_1"]["sheets"]])
        self.assertEqual(author["public_context"]["validation_contract"]["kind"], "deduplicate_sum")
        self.assertIn("transformation_body", author["allowed_tools"][0]["input_schema"]["required"])
        self.assertIn("return", prompts[1][2][-1]["content"])
        repair = next(prompt for _, prompt, _ in prompts if prompt["role"]["agent_id"] == REPAIR)
        evidence = repair["public_context"]["runtime_state"]["artifacts"]
        self.assertIn("absent", evidence["public_execution"]["content_preview"])
        self.assertIn("absent", evidence["candidate_solution"]["content_preview"])
        for payload in (*observations, *(prompt for _, prompt, _ in prompts)):
            self.assertFalse(payload["constraints"]["private_reference_visible"])
            self.assertNotIn("private/cases/", json.dumps(payload))
            self.assertNotIn("evaluation_target.json", json.dumps(payload))

    def test_legacy_full_source_still_accepted(self):
        adapter = SpreadsheetBenchPythonOJAdapter()
        with TemporaryDirectory(dir=PROJECT / "tests") as directory:
            env = adapter.materialize(PACK / "manifest.json", Path(directory) / "workspace")
            self.assertIsNone(adapter.plain_python_submission(env, {"write_solution"}))
            write = adapter.tool_spec(env, AUTHOR)[0]
            self.assertIn("python_source", write.input_schema["required"])
            result = adapter.execute_tool(env, write.name, {
                "python_source": "def solve(input_path, output_path):\n    load_workbook(input_path).save(output_path)",
                "input_artifact_refs": [],
            })
            self.assertEqual(result.status, "success")
            self.assertNotIn("submission", adapter.public_observation(env)["candidate_contract"])

    def test_direct_full_source_author_and_repair(self):
        adapter = SpreadsheetBenchPythonOJAdapter()
        adapter.configure_runtime({"candidate_authoring": "full_source_v1"})
        actor = SharedRoleActor(None, "scripted")
        header = "def solve(input_path, output_path):\n    workbook = load_workbook(input_path)\n"
        failing_source = header + "    workbook['absent']\n"
        repaired_source = header + indent(BODY, "    ") + "    workbook.save(output_path)\n"
        actor._query = Mock(side_effect=[
            (failing_source, {}),
            ("```python\n" + repaired_source + "```", {}),
        ])
        with TemporaryDirectory(dir=PROJECT / "tests") as directory:
            env = adapter.materialize(PACK / "manifest.json", Path(directory) / "workspace")
            author = actor.execute(env=env, adapter=adapter, action=AUTHOR, role_profile={}, task=env.manifest.task)
            self.assertEqual(author["status"], "success")
            self.assertEqual(author["tool_events"][0]["tool_call"]["arguments"]["python_source"], failing_source)
            candidate = adapter.public_observation(env)["artifacts"]["candidate_solution"]["artifact_id"]
            failed = adapter.execute_tool(env, "run_solution_public", {"candidate_artifact_id": candidate})
            self.assertEqual(failed.status, "failed")
            evidence = failed.output["execution_artifact"]["artifact_id"]

            repair = actor.execute(env=env, adapter=adapter, action=REPAIR, role_profile={}, task=env.manifest.task)
            self.assertEqual(repair["status"], "success")
            arguments = repair["tool_events"][0]["tool_call"]["arguments"]
            self.assertEqual(arguments["base_candidate_artifact_id"], candidate)
            self.assertEqual(arguments["evidence_refs"], [evidence])
            artifacts = adapter.public_observation(env)["artifacts"]
            self.assertEqual(artifacts["candidate_solution"]["content_preview"], repaired_source)
            candidate = artifacts["candidate_solution"]["artifact_id"]
            tested = adapter.execute_tool(env, "run_solution_public", {"candidate_artifact_id": candidate})
            self.assertEqual(tested.status, "success", tested.output)
            validated = adapter.execute_tool(env, "validate_public_output", {
                "candidate_artifact_id": candidate,
                "execution_artifact_id": tested.output["execution_artifact"]["artifact_id"],
            })
            self.assertEqual(validated.status, "success", validated.output)
