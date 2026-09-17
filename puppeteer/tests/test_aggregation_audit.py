import unittest
from role_aware.aggregation import aggregate_candidates,normalize_choice
from role_aware.audit_analysis import summarize
from scripts.replay_aggregation import replay
from role_aware.audit_trace import digest

class AggregationTests(unittest.TestCase):
    def test_legacy_tie_loses_available_correct_candidate(self):
        candidates=["F","i","J"]
        self.assertEqual(aggregate_candidates(candidates,mode="legacy")["prediction"],"J")
        task=dict(task_id="x",run_id="r",trace_path="x",evaluation={"gold":"F"},
                  candidates=candidates,prediction="J",paths=[])
        report=summarize([task])
        self.assertEqual(report["counts"]["lost_correct"],1)

    def test_majority_normalization_order_invariance_and_invalid(self):
        a=aggregate_candidates(["a","A","invalid","J"])
        self.assertEqual(a["prediction"],"A")
        self.assertEqual(a["invalid_count"],1)
        for values in (["F","I","J"],["J","F","I"]):
            self.assertEqual(aggregate_candidates(values,task_id="x")["prediction"],
                             aggregate_candidates(["F","I","J"],task_id="x")["prediction"])
        self.assertEqual(aggregate_candidates(["","???"])["prediction"],"")
        self.assertEqual(normalize_choice("The final answer is (b)."),"B")
        self.assertIsNone(normalize_choice("Z"))
        self.assertIsNone(normalize_choice("J",choices="ABCD"))

    def test_verifier_constrained_fallback_and_gold_is_not_input(self):
        calls=[]
        def verifier(prompt):
            calls.append(prompt)
            return "Z",7
        result=aggregate_candidates(["F","I","J"],mode="majority_verifier",question="Question?",verifier=verifier)
        self.assertIn(result["prediction"],["F","I","J"])
        self.assertEqual(result["verifier_status"],"invalid_fallback")
        self.assertEqual(result["verifier_tokens"],7)
        self.assertNotIn("gold",calls[0])

    def test_replay_same_candidates_different_gold_same_predictions(self):
        task=dict(task_id="x",run_id="r",candidates=["A","B"],candidate_hash=digest(["A","B"]),
                  question="Q",evaluation={"gold":"A"})
        first=replay([task],["legacy","majority"])
        task["evaluation"]["gold"]="B"
        second=replay([task],["legacy","majority"])
        self.assertEqual(first["rows"][0]["predictions"],second["rows"][0]["predictions"])
        task["candidates"]=["C"]
        with self.assertRaisesRegex(ValueError,"hash"):
            replay([task],["majority"])
