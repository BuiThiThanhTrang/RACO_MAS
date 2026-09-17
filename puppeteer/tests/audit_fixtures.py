import copy
import logging
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
import torch
from agent.register.persona_loader import load_teammate_specs
from agent.register.register import AgentRegister
from inference.graph.agent_graph import AgentGraph
from inference.graph.action_graph import ActionGraph
from inference.policy.role_aware_reinforce import RoleAwareREINFORCE
from inference.reasoning.reasoning import GraphReasoning
from role_aware.audit_trace import AuditTrace, observe_provider_call
from role_aware.profile_store import ProfileStore

PROJECT = Path(__file__).resolve().parents[1]

class SilentLogs:
    def __init__(self, *args, folder_path=None, **kwargs):
        self.folder_path = str(folder_path)
        self.logger = logging.getLogger("audit-tests")
        self.logger.addHandler(logging.NullHandler())
    def create_logger(self, *args, **kwargs):
        pass
    def get_logger(self, *args):
        return self.logger

class ScheduledPolicy:
    def __init__(self, schedule):
        self.schedule = iter(schedule)
        self.updates = 0
        self.rewards = []
    def forward(self, info):
        return next(self.schedule)
    def finalize_task(self, transition, info):
        self.rewards.append(copy.deepcopy(transition))
    def update(self):
        self.updates += 1
        return {}
    def abort_task(self):
        self.aborted = True

def setup(directory, mode="legacy_threshold", width=3, depth=2, schedule=None,
          enabled=True, train=False, gold="A"):
    specs = load_teammate_specs(PROJECT/"personas/role_aware/s0_pool.jsonl")
    specs = [s for s in specs if s.role_card.role_name in {"Domain Reasoner", "General Reasoner"}][:3]
    registry = AgentRegister()
    registry.load(specs)
    requests = []
    for agent in registry.ordered_agents:
        def query(messages, system_prompt=None, name=agent.hash):
            messages = copy.deepcopy(messages)
            requests.append((name, messages))
            with observe_provider_call(messages, name, 8) as receipt:
                receipt["tokens"] = 5
                receipt["usage_source"] = "fixture"
                return "FINAL ANSWER: A", 5
        agent.query_func = query
    profiles = ProfileStore()
    profiles.initialize(specs)
    graph = AgentGraph(registry, profiles, allowed_tools=())
    action_graph = ActionGraph()
    config = dict(seed=42, dataset_mode="train" if train else "dev",
                  policy_mode="train" if train else "initialized",
                  device={"type":"cpu"}, training={"training":train, "entropy_coef":0.0},
                  routing={"mode":mode, "selection_count":2, "threshold_multiplier":0.0})
    runtime = dict(graph={"max_width":width,"max_depth":depth},
                   task_analyzer={"primary":{"dim":4}}, trajectory_reward={"enabled":False},
                   aggregation={"mode":"majority"}, audit_split="dev")
    if schedule is None:
        policy = RoleAwareREINFORCE(graph, action_graph, config, runtime)
        policy.get_state_representation = lambda info: torch.zeros(1,4)
    else:
        policy = ScheduledPolicy(schedule([a.hash for a in registry.ordered_agents]))
    trace = AuditTrace(directory, "fixture", "question-1", "attempt-0001", enabled)
    with patch("inference.reasoning.reasoning.LogManager", SilentLogs):
        reasoning = GraphReasoning(dict(id="question-1", type="MMLU-Pro", Question="Test question.",
                                        Answer=gold, choices="ABCDEFGHIJ"),
             graph, policy, action_graph, width, depth, runtime, registry, audit=trace)
    return reasoning, policy, trace, requests
