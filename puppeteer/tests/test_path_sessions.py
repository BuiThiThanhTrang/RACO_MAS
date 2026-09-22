import copy
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import torch

from audit_fixtures import setup
from agent.agent_session import PathRuntimeContext, bind_session
from inference.policy.role_aware_reinforce import RoleAwareREINFORCE

class PathSessionTests(unittest.TestCase):
    def test_same_teammate_interleaved_sessions_and_fork(self):
        with tempfile.TemporaryDirectory() as d:
            r,_,_,calls=setup(d)
            agent=r.registry.ordered_agents[0]
            a,b=PathRuntimeContext("a"),PathRuntimeContext("b")
            for context, marker in ((a,"ONLY_A"),(b,"ONLY_B"),(a,"A2"),(b,"B2")):
                session=context.session(agent.hash)
                with bind_session(agent,session):
                    agent.activate(r._new_info(0),session.initial_dialog_history)
                    agent._query(marker)
                    agent.deactivate()
            a_calls=[messages for _,messages in calls if "ONLY_A" in str(messages)]
            b_calls=[messages for _,messages in calls if "ONLY_B" in str(messages)]
            self.assertTrue(a_calls and b_calls)
            self.assertTrue(all("ONLY_B" not in str(m) for m in a_calls))
            self.assertTrue(all("ONLY_A" not in str(m) for m in b_calls))
            self.assertEqual(agent.dialog_history,[])
            child=a.fork("child")
            child.session(agent.hash).dialog_history[0]["content"]="CHANGED"
            self.assertNotIn("CHANGED",str(a.sessions))
            self.assertIn("ONLY_A",str(child.sessions))

    def test_orchestrator_receives_baseline_style_path_dialog_history(self):
        context = PathRuntimeContext("path")
        session = context.session("agent-1")
        session.dialog_history = [
            {"role": "system", "content": "SYSTEM"},
            {"role": "user", "content": "VISIBLE *MASKED*"},
            {"role": "assistant", "content": "FIRST_OUTPUT"},
        ]
        context.record_agent_turn("agent-1")
        context.record_agent_turn("agent-1")
        messages = context.baseline_orchestrator_messages("TASK")

        # Baseline loops over selected agents and appends each current,
        # simplified dialog history. Repeated agents therefore repeat it.
        self.assertEqual(len(messages), 6)
        self.assertEqual(messages[0]["content"], "SYSTEM")
        self.assertIn("VISIBLE", messages[1]["content"])
        self.assertNotIn("MASKED", messages[1]["content"])
        self.assertEqual(messages[2]["content"], "FIRST_OUTPUT")
        self.assertEqual(messages[3:], messages[:3])

        captured = []
        policy = object.__new__(RoleAwareREINFORCE)
        policy.device = torch.device("cpu")
        policy.state_representation = lambda received: (
            captured.append(copy.deepcopy(received)) or torch.ones(1, 4),
            0.0,
        )
        info = SimpleNamespace(task={"Question": "TASK"}, path_context=context)
        state = policy.get_state_representation(info)
        self.assertEqual(tuple(state.shape), (1, 4))
        self.assertEqual(captured, [messages])

    def test_reselected_agent_receives_all_successful_reasoning_results(self):
        with tempfile.TemporaryDirectory() as d:
            r, _, _, calls = setup(
                d, schedule=lambda ids: [[ids[0]], [ids[1]], [ids[1]]], depth=3
            )
            responses = iter([
                "REASONING RESULT: FIRST_OUTPUT\nFINAL ANSWER: A",
                "REASONING RESULT: SECOND_OUTPUT\nFINAL ANSWER: A",
                "REASONING RESULT: THIRD_OUTPUT\nFINAL ANSWER: A",
                "A",
            ])

            def query(messages, system_prompt=None):
                calls.append(("fixture", copy.deepcopy(messages)))
                return next(responses), 5

            for agent in r.registry.ordered_agents:
                agent.query_func = query
            r.start(None)
            r.n_step(3)

            self.assertGreaterEqual(len(calls), 3)
            third_text = str(calls[2][1])
            self.assertIn("FIRST_OUTPUT", third_text)
            self.assertIn("SECOND_OUTPUT", third_text)
            self.assertNotIn("THIRD_OUTPUT", third_text)

    def test_fork_prefix_and_artifact_are_independent(self):
        with tempfile.TemporaryDirectory() as d:
            r,_,_,_=setup(d,schedule=lambda ids:[[ids[0]]],depth=2)
            r.start(None)
            parent=r.reasoning_paths[0]
            parent.step()
            artifact=Path(parent.workspace_path)/"answer.txt"
            artifact.write_text("SHARED")
            parent.global_info.code_path=str(artifact)
            child=parent.fork(1,"child",parent.current_agent,r.workspace_path)
            Path(child.global_info.code_path).write_text("ONLY_CHILD")
            self.assertEqual(artifact.read_text(),"SHARED")
            self.assertEqual(parent.frontier,child.frontier)
            self.assertEqual(parent.completed_steps,child.completed_steps)
            self.assertIsNot(parent.context,child.context)

    def test_task_reset_has_no_conversation(self):
        with tempfile.TemporaryDirectory() as d:
            r,_,_,_=setup(d,schedule=lambda ids:[[ids[0]]],depth=1)
            r.start(None); r.n_step(1)
            self.assertTrue(all(a.dialog_history==[] for a in r.registry.ordered_agents))
