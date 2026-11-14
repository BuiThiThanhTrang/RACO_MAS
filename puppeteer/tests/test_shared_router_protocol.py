from types import SimpleNamespace
from unittest.mock import patch
import unittest

from scripts.run_shared_environment import build_router


class SharedRouterProtocolTests(unittest.TestCase):
    def select(self,configuration):
        with patch('routing_benchmark.routers.SystemOneDecisionClient') as client_class:
            client=client_class.return_value
            client.decide.return_value=SimpleNamespace(choice='B',confidence=None,probabilities={},input_tokens=0,latency_ms=0,model='test',model_version=None)
            router=build_router({'type':'decision','provider':'openrouter','model':'test',**configuration},42)
            observation={'benchmark_version':'4.4-shared-environment-v1','agent_profiles':[{'agent_id':'A'},{'agent_id':'B'}],
                'recent_events':[{'agent_id':'A','status':'ACTION_REJECTED','semantic_delta':'no_state_change'}]}
            decision=router.select(observation,['A','B','STOP'])
            sent,questions,_=client.decide.call_args.args
            return router,decision,sent,list(questions['next_agent']['criteria'])

    def test_default_keeps_rejected_action_in_model_choices(self):
        router,decision,sent,choices=self.select({})
        self.assertEqual(choices,['A','B','STOP'])
        self.assertFalse(router.observable_rejection_guard_enabled)
        self.assertEqual(decision.metadata['guard_blocked_actions'],[])
        self.assertNotIn('observable_guard',sent.get('constraints',{}))

    def test_explicit_false_is_forwarded_to_router_constructor(self):
        _,decision,_,choices=self.select({'observable_rejection_guard_enabled':False})
        self.assertFalse(decision.metadata['observable_rejection_guard_enabled'])
        self.assertEqual(choices,['A','B','STOP'])

    def test_optional_guard_requires_explicit_true(self):
        _,decision,_,choices=self.select({'observable_rejection_guard_enabled':True})
        self.assertEqual(choices,['B','STOP'])
        self.assertEqual(decision.metadata['guard_blocked_actions'],['A'])


if __name__=='__main__': unittest.main()
