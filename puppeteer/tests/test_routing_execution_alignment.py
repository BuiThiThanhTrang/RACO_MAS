import tempfile
import unittest
from audit_fixtures import setup
from role_aware.audit_analysis import validate_events

class AlignmentTests(unittest.TestCase):
    def test_capacity_full_and_one_slot_left_no_ghost_trajectories(self):
        for mode in ("legacy_threshold","categorical_set_v2"):
            with self.subTest(mode=mode),tempfile.TemporaryDirectory() as d:
                r,p,t,_=setup(d,mode=mode,width=3,depth=3,train=True)
                r.start(None); r.n_step(3)
                self.assertLessEqual(len(r.reasoning_paths),3)
                self.assertEqual(validate_events(t.events)["errors"],[])
                accepted={a["action_id"] for e in t.events if e["event_type"]=="allocation" for a in e["accepted"]}
                rewarded={a for e in t.events if e["event_type"]=="reward_assigned" for a in e["action_ids"]}
                self.assertEqual(accepted,rewarded)
                for path in r.reasoning_paths:
                    self.assertLessEqual(path.completed_steps,3)

    def test_shared_prefix_is_single_execution(self):
        with tempfile.TemporaryDirectory() as d:
            r,p,t,_=setup(d,schedule=lambda ids:[[ids[0]],[ids[0],ids[1]]],width=2,depth=2)
            r.start(None); r.n_step(2)
            starts=[e for e in t.events if e["event_type"]=="action_started"]
            self.assertEqual(len(starts),3)
            self.assertEqual(sum(len(path.global_info.workflow.workflow) for path in r.reasoning_paths),4)
            self.assertEqual(validate_events(t.events)["errors"],[])

    def test_unknown_agent_is_not_silently_credited(self):
        with tempfile.TemporaryDirectory() as d:
            r,p,t,_=setup(d,schedule=lambda ids:[["unknown"]])
            with self.assertRaises(RuntimeError):
                r.start(None)
            self.assertEqual(p.updates,0)
