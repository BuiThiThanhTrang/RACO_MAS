"""Enumerate a tiny forked tree to check the complete-episode score gradient."""
import itertools
import unittest
import torch
from role_aware.multiselect import sample_ordered


class TreeEstimatorTests(unittest.TestCase):
    def test_expected_gradient_with_shared_prefix_and_sibling_rewards(self):
        logits = torch.tensor([.1,-.4,.2,.3], dtype=torch.float64, requires_grad=True)
        probs = logits.softmax(0)
        direct = 0.
        score = 0.
        mass = 0.
        # Root chooses one of three agents (STOP masked), then selects two ordered children
        # or STOP. Rewards depend on both prefix identity and sibling outcomes.
        root_probs = torch.cat((probs[:3], probs[3:] * 0))
        root_probs = root_probs / root_probs.sum()
        for first in range(3):
            prefix = sample_ordered(root_probs,1,3,forced=(first,))
            for children in [(3,)]+list(itertools.permutations(range(3),2)):
                fork = sample_ordered(probs,2,3,forced=children)
                joint = prefix.log_prob + fork.log_prob
                q = joint.exp()
                leaf_rewards = ([.5-first] if children==(3,) else
                                [(1 if child==first else -1)-.1*child for child in children])
                task_return = sum(-.2+.9*reward for reward in leaf_rewards)
                direct = direct + q*task_return
                score = score + q.detach()*joint*task_return
                mass += q.detach().item()
        self.assertAlmostEqual(mass,1.,places=12)
        expected = torch.autograd.grad(direct,logits,retain_graph=True)[0]
        actual = torch.autograd.grad(score,logits)[0]
        self.assertTrue(torch.allclose(expected,actual,atol=1e-12))
