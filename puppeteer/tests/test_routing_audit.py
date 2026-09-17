import copy
import tempfile
import unittest
from pathlib import Path
import torch
from audit_fixtures import setup
from role_aware.audit_analysis import validate_events
from role_aware.audit_trace import redact

STOP = "__orchestrator_stop__"

class RoutingAuditTests(unittest.TestCase):
    def run_fixture(self, directory, **kwargs):
        reasoning, policy, trace, calls = setup(directory, **kwargs)
        reasoning.start(None)
        result = reasoning.n_step(reasoning.max_step_num)
        return reasoning, policy, trace, calls, result

    def test_stop_after_first_action(self):
        with tempfile.TemporaryDirectory() as d:
            r,p,t,c,_ = self.run_fixture(d, schedule=lambda ids:[[ids[0]],[STOP]], depth=2)
            self.assertEqual(r.reasoning_paths[0].completed_steps,1)
            self.assertEqual(r.reasoning_paths[0].stop_reason,"policy_stop")
            self.assertEqual(validate_events(t.events)["errors"], [])
            self.assertEqual(len([e for e in t.events if e["event_type"]=="action_started"]),1)

    def test_depth_limit_does_not_query_router_again(self):
        with tempfile.TemporaryDirectory() as d:
            r,p,t,c,_ = self.run_fixture(d, schedule=lambda ids:[[ids[0]],[ids[0]]], depth=2)
            reasons=[e for e in t.events if e["event_type"]=="path_finished"]
            self.assertEqual(reasons[0]["stop_reason"],"depth_limit")
            self.assertIsNone(reasons[0]["p_stop"])
            self.assertEqual(len([e for e in t.events if e["event_type"]=="routing_decision"]),2)
            self.assertEqual(validate_events(t.events)["errors"], [])

    def test_audit_on_off_preserves_outputs_weights_and_rng(self):
        with tempfile.TemporaryDirectory() as d:
            results=[]
            for enabled in (True,False):
                r,p,t,c,result=self.run_fixture(Path(d)/str(enabled),enabled=enabled,
                                                train=True,mode="categorical_set_v2",depth=2)
                results.append((result,copy.deepcopy(p.policy_network.state_dict()),torch.get_rng_state(),len(c)))
            self.assertEqual(results[0][0],results[1][0])
            self.assertEqual(results[0][3],results[1][3])
            self.assertTrue(torch.equal(results[0][2],results[1][2]))
            for k in results[0][1]:
                self.assertTrue(torch.equal(results[0][1][k],results[1][1][k]),k)

    def test_exception_aborts_without_update(self):
        with tempfile.TemporaryDirectory() as d:
            r,p,t,_=setup(d,schedule=lambda ids:[[ids[0]]])
            def fail(*args):
                raise RuntimeError("fixture")
            r.registry.ordered_agents[0].take_action=fail
            r.start(None)
            with self.assertRaisesRegex(RuntimeError,"fixture"):
                r.n_step(2)
            self.assertEqual(p.updates,0)
            self.assertEqual(r.reasoning_paths[0].stop_reason,"execution_error")
            self.assertEqual(validate_events(t.events)["errors"],[])

    def test_prediction_committed_before_reward_and_gold_never_in_requests(self):
        with tempfile.TemporaryDirectory() as d:
            r,p,t,c,_=self.run_fixture(d,mode="categorical_set_v2",train=True,gold="SECRET_GOLD")
            self.assertNotIn("SECRET_GOLD",str(c))
            self.assertEqual(validate_events(t.events)["errors"],[])
            self.assertTrue(any(e["event_type"]=="policy_update" for e in t.events))
            self.assertEqual(t.call_totals()["total_tokens"],5*len(c))

    def test_manifest_redacts_secrets(self):
        self.assertEqual(redact({"api_key":"secret","headers":{"Authorization":"secret"}}),
                         {"api_key":"[redacted]","headers":{"Authorization":"[redacted]"}})
