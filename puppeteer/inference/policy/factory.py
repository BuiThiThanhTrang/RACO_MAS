from inference.policy.decision_planner import DecisionPlannerPolicy
from inference.policy.frozen_llm_planner import FrozenLLMPlannerPolicy
from inference.policy.role_aware_reinforce import RoleAwareREINFORCE


def create_policy(
    agent_graph,
    action_graph,
    config,
    runtime_config,
    experience_store=None,
    legacy_policy_cls=RoleAwareREINFORCE,
):
    policy_type = str((config or {}).get("type", "role_aware_reinforce"))
    if policy_type == "role_aware_reinforce":
        return legacy_policy_cls(
            agent_graph=agent_graph,
            action_graph=action_graph,
            config=config,
            runtime_config=runtime_config,
        )
    if policy_type == "frozen_llm_planner":
        return FrozenLLMPlannerPolicy(
            agent_graph=agent_graph,
            action_graph=action_graph,
            config=config,
            runtime_config=runtime_config,
            experience_store=experience_store,
        )
    if policy_type == "decision_model_planner":
        return DecisionPlannerPolicy(
            agent_graph=agent_graph,
            action_graph=action_graph,
            config=config,
            runtime_config=runtime_config,
            experience_store=experience_store,
        )
    raise ValueError(f"Unknown policy.type: {policy_type}")
