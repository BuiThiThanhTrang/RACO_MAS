import itertools
import unittest
import torch
from role_aware.multiselect import sample_ordered

class MultiselectTests(unittest.TestCase):
    def outcomes(self,k):
        return [(3,)]+list(itertools.permutations(range(3),k))

    def test_normalization_and_conditional_likelihood(self):
        probs=torch.tensor([.2,.3,.4,.1],dtype=torch.float64)
        for k in (1,2,3):
            mass=0.
            for outcome in self.outcomes(k):
                result=sample_ordered(probs,k,3,forced=outcome)
                mass+=result.log_prob.exp().item()
            self.assertAlmostEqual(mass,1.,places=12)
        selected=sample_ordered(probs,2,3,forced=(0,1))
        self.assertAlmostEqual(selected.log_prob.exp().item(),.2*(.3/.7),places=12)

    def test_exact_expected_gradient_matches_score_estimator(self):
        for k in (1,2):
            logits=torch.tensor([.2,-.4,.6,.1],dtype=torch.float64,requires_grad=True)
            probs=logits.softmax(0)
            direct=0.; surrogate=0.
            for outcome in self.outcomes(k):
                result=sample_ordered(probs,k,3,forced=outcome)
                reward= float(sum(i+1 for i in outcome)) * (-1 if outcome==(3,) else 1)
                q=result.log_prob.exp()
                direct=direct+q*reward
                surrogate=surrogate+q.detach()*result.log_prob*reward
            expected=torch.autograd.grad(direct,logits,retain_graph=True)[0]
            score=torch.autograd.grad(surrogate,logits)[0]
            self.assertTrue(torch.allclose(expected,score,atol=1e-12))

    def test_mask_stop_and_no_replacement(self):
        p=torch.tensor([.5,0.,.5,0.])
        for invalid in ((1,),(3,),(0,0)):
            with self.assertRaises(ValueError):
                sample_ordered(p,len(invalid),3,forced=invalid)
        with self.assertRaises(ValueError):
            sample_ordered(torch.zeros(4),2,3)
