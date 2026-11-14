import json
from pathlib import Path
import tempfile
import unittest

from scripts.score_shared_reviews import score


class HumanScoringTests(unittest.TestCase):
    def test_nested_tool_infrastructure_is_distinct_from_proof_failure(self):
        from scripts.evaluate_shared_environment import infrastructure_failure
        def event(output): return [{'kind':'role_execution','result':{'tool_events':[{'tool_result':{'status':'failed','output':output}}]}}]
        self.assertFalse(infrastructure_failure(event({'result':{'returncode':1,'stdout':'unsolved goals'}})))
        self.assertTrue(infrastructure_failure(event({'kernel_probe':{'returncode':-1,'failure_origin':'tool_environment'}})))
        self.assertTrue(infrastructure_failure([{'kind':'router_error'}]))

    def test_only_confirmed_same_run_labels_and_executable_decisions_are_scored(self):
        with tempfile.TemporaryDirectory(dir=Path(__file__).resolve().parent) as root:
            run=Path(root)/'review-fixture';folder=run/'episodes/packet';folder.mkdir(parents=True)
            def write(path,rows): path.write_text(''.join(json.dumps(r)+'\n' for r in rows))
            states=[{'decision_id':str(i),'candidate_actions':['A','STOP']} for i in range(3)]
            decisions=[{'decision_id':'0','agent_id':'A','dispatch_status':'dispatched'},
                       {'decision_id':'1','agent_id':'STOP','dispatch_status':'stop_handled'},
                       {'decision_id':'2','agent_id':'A','dispatch_status':'suppressed_by_budget'}]
            write(folder/'decision_states.jsonl',states);write(folder/'decisions.jsonl',decisions)
            write(folder/'trajectory.jsonl',[]);(folder/'result.json').write_text('{}')
            labels=[{'run_id':run.name,'decision_id':str(i),'valid_actions':actions,'label_status':'human_verified','reviewer':'user'}
                    for i,actions in enumerate([['A'],['STOP'],[]])]
            file=run/'labels.jsonl';write(file,labels)
            before=(folder/'decision_states.jsonl').read_bytes()
            report=score(run,file)
            self.assertEqual(report['scored_decisions'],2)
            self.assertEqual(report['valid_action_accuracy'],1)
            self.assertEqual(report['human_review_status'],'human_verified')
            self.assertEqual((folder/'decision_states.jsonl').read_bytes(),before)
            labels[0]['label_status']='pending_human_review';write(file,labels)
            with self.assertRaises(ValueError):score(run,file)
            labels[0]['label_status']='human_verified';labels[0]['run_id']='other-run';write(file,labels)
            with self.assertRaises(ValueError):score(run,file)
