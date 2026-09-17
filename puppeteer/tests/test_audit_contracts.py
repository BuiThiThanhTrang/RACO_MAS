import copy
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

import torch
from audit_fixtures import setup, SilentLogs
from role_aware.audit_trace import AuditTrace
from role_aware.audit_analysis import load_events, validate_events, summarize
from role_aware.stop_probe import run_probe
from model import model_utils


class AuditContractTests(unittest.TestCase):
    def test_checkpoint_rejects_incompatible_resume_and_requires_transfer(self):
        with tempfile.TemporaryDirectory() as d:
            _, source, _, _ = setup(Path(d)/"source", mode="legacy_threshold", train=True)
            _, target, _, _ = setup(Path(d)/"target", mode="categorical_set_v2", train=True)
            checkpoint = copy.deepcopy(source.training_state_dict())
            with self.assertRaisesRegex(ValueError, "Routing"):
                target.load_training_state_dict(checkpoint)
            with self.assertRaisesRegex(ValueError, "explicit"):
                target.load_training_state_dict(checkpoint, load_optimizer=False)
            target.config["routing"]["allow_policy_transfer"] = True
            target.load_training_state_dict(checkpoint, load_optimizer=False)
            with self.assertRaisesRegex(ValueError, "Routing"):
                target.load_training_state_dict(checkpoint, load_optimizer=True)
            own = copy.deepcopy(target.training_state_dict())
            for key in ("runtime_version", "objective_version", "estimator_version", "routing_version"):
                corrupt = dict(own, **{key: "wrong"})
                with self.subTest(key=key), self.assertRaises(ValueError):
                    target.load_training_state_dict(corrupt)
            target.load_training_state_dict(own)

    def test_partial_episode_cannot_update(self):
        with tempfile.TemporaryDirectory() as d:
            r, p, _, _ = setup(d, mode="categorical_set_v2", train=True)
            r.start(None)
            self.assertEqual(len(p.trajectories), 2)
            p.trajectories[0][-1]["finalized"] = True
            before = copy.deepcopy(p.policy_network.state_dict())
            with self.assertRaisesRegex(RuntimeError, "incomplete"):
                p.update()
            for key, value in before.items():
                self.assertTrue(torch.equal(value, p.policy_network.state_dict()[key]))
            p.abort_task()
            self.assertEqual(p.trajectories, [])

    def test_joint_update_uses_each_decision_once_with_shared_prefix(self):
        with tempfile.TemporaryDirectory() as d:
            _, p, t, _ = setup(d, mode="categorical_set_v2", train=False)
            p.begin_task(t)
            root = torch.tensor(-.4, requires_grad=True)
            branch = torch.tensor(-.7, requires_grad=True)
            # Same root appears in each leaf; its sampling log likelihood occurs only once.
            p.trajectories = [
                [dict(log_prob=root, reward=-.2), dict(log_prob=branch, reward=1., finalized=True)],
                [dict(log_prob=root, reward=-.2), dict(log_prob=branch, reward=-1., finalized=True)],
            ]
            p.decisions = [dict(decision_id="root", log_prob=root),
                           dict(decision_id="branch", log_prob=branch)]
            expected_return = 2 * -.2 + p.gamma * (1. - 1.)
            metrics = p.update()
            self.assertAlmostEqual(metrics["task_return"], expected_return, places=6)
            self.assertAlmostEqual(metrics["policy_loss"], -(-.4-.7)*expected_return, places=6)

    def test_provider_retries_and_unknown_usage_are_audited(self):
        with tempfile.TemporaryDirectory() as d:
            trace = AuditTrace(d, "r", "t", "1")
            response = SimpleNamespace(
                usage=SimpleNamespace(total_tokens=17, prompt_tokens=12, completion_tokens=5),
                choices=[SimpleNamespace(message=SimpleNamespace(content="A"))])
            client = Mock()
            client.chat.completions.create.side_effect = [RuntimeError("retry"), response]
            with patch.object(model_utils, "CHAT_MAX_RETRY_TIMES", 2), patch.object(model_utils.time, "sleep"), trace.call_scope(purpose="agent", model_size=8):
                _, tokens = model_utils.chat_completion_request([{"role":"user","content":"Question"}], "fixture", client)
            self.assertEqual(tokens, 17)
            self.assertEqual(trace.call_totals(), dict(api_calls=2,total_tokens=17,unknown_usage_calls=1,model_cost=272))
            response.usage = None
            client.chat.completions.create.side_effect = [response]
            with trace.call_scope(purpose="agent", model_size=8):
                model_utils.chat_completion_request([{"role":"user","content":"Q"}], "fixture", client)
            self.assertEqual(trace.call_totals()["unknown_usage_calls"], 2)
            self.assertEqual(trace.call_totals()["total_tokens"], 17)

    def test_stop_and_continue_restore_identical_label_free_prefix(self):
        with tempfile.TemporaryDirectory() as d:
            r, _, _, calls = setup(Path(d)/"source", depth=1,
                                   schedule=lambda ids: [[ids[0]]], gold="DO_NOT_READ")
            r.start(None); r.n_step(1)
            source = next((Path(d)/"source").rglob("step_1.json"))
            snapshot = json.loads(source.read_text(encoding="utf-8"))
            original = copy.deepcopy(snapshot)
            self.assertNotIn("DO_NOT_READ", str(snapshot))
            outputs = []
            with patch("inference.reasoning.reasoning.LogManager", SilentLogs):
                for name, continuation in (("stop",False),("continue",True)):
                    result = run_probe(snapshot, r.registry, r.agent_graph, r.action_graph,
                                       r.runtime_config, Path(d)/name,
                                       r.registry.ordered_agents[0].hash, continuation)
                    outputs.append(result)
                    events = load_events(Path(d)/name/"events.jsonl")
                    self.assertEqual(validate_events(events)["errors"], [])
                    starts = [e for e in events if e["event_type"]=="action_started"]
                    self.assertEqual(len(starts), int(continuation))
                    candidates = json.loads((Path(d)/name/"candidates.json").read_text(encoding="utf-8"))
                    self.assertEqual(candidates["paths"][0]["steps"][0], snapshot["workflow"][0])
            self.assertEqual(snapshot, original)
            self.assertEqual([o["prediction"] for o in outputs], ["A","A"])
            self.assertGreater(outputs[1]["cost"]["api_calls"], outputs[0]["cost"]["api_calls"])
            self.assertNotIn("DO_NOT_READ", str(calls))

    def test_validator_detects_wrong_path_ghost_reward_and_nan(self):
        events = [
            dict(event_type="allocation", accepted=[dict(action_id="a",path_uid="p")]),
            dict(event_type="action_finished",action_id="a",path_uid="wrong"),
            dict(event_type="reward_assigned",action_ids=["ghost"]),
            dict(event_type="routing_decision",probabilities=[float("nan"),0.],p_stop=0.,allow_stop=False),
        ]
        errors = validate_events(events)["errors"]
        self.assertIn("wrong_path:a",errors)
        self.assertIn("unallocated_reward:ghost",errors)
        self.assertIn("invalid_probability",errors)

    def test_reasoning_and_aggregation_transitions_are_separate(self):
        task = dict(run_id="r",task_id="t",attempt_id="1",evaluation={"gold":"A"},
                    choices="AB",candidates=["B"],prediction="B",cost={},
                    paths=[dict(stop_reason="depth_limit",before_aggregation="A",prediction="B",
                                steps=[dict(action_id="a",result={"answer":"B"}),
                                       dict(action_id="b",result={"answer":"A"})])])
        report = summarize([task])
        self.assertEqual(report["leaf_transition_counts"], {"wrong->correct":1})
        self.assertEqual(report["path_aggregation_transition_counts"], {"correct->wrong":1})
        self.assertEqual(report["physical_agent_executions"], 2)

    def test_online_and_replay_use_same_raw_candidates(self):
        from role_aware.aggregation import aggregate_candidates
        with tempfile.TemporaryDirectory() as d:
            r, _, _, _ = setup(d, depth=1, schedule=lambda ids: [[ids[0]]])
            r.start(None); r.n_step(1)
            snapshot = json.loads((Path(d)/"candidates.json").read_text(encoding="utf-8"))
            self.assertEqual(snapshot["candidates"], ["FINAL ANSWER: A"])
            replay = aggregate_candidates(snapshot["candidates"], mode="majority", seed=42, task_id="question-1")
            self.assertEqual(replay["prediction"], snapshot["prediction"])
            self.assertEqual(replay["candidate_hash"], snapshot["candidate_hash"])

    def test_http_reward_calls_and_cache_count_once(self):
        from role_aware.reward_scorer import AuxiliaryRewardScorer
        with tempfile.TemporaryDirectory() as d:
            trace = AuditTrace(d,"r","t","1")
            scorer = AuxiliaryRewardScorer(dict(backend="http",model="fixture",
                                               base_url="http://fixture",cache={"enabled":True}))
            response = Mock()
            response.json.return_value = {"score":.7}
            with patch("requests.post",return_value=response) as post, trace.call_scope(purpose="reward_scorer"):
                self.assertEqual(scorer([{"role":"user","content":"Q"}]),.7)
                self.assertEqual(scorer([{"role":"user","content":"Q"}]),.7)
            self.assertEqual(post.call_count,1)
            self.assertEqual(trace.call_totals()["api_calls"],1)
            self.assertEqual(trace.call_totals()["unknown_usage_calls"],1)
