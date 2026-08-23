from __future__ import annotations

ROLE_CAPABILITY_SCOPES = {
    "Planner / Decomposer": ("planning", "general_reasoning", "integration"),
    "General Reasoner": ("general_reasoning", "verification", "integration"),
    "Quantitative Reasoner": ("quantitative_reasoning", "general_reasoning", "verification"),
    "Domain Reasoner": ("domain_reasoning", "general_reasoning", "verification"),
    "Software Engineer": ("software_engineering", "planning", "verification", "repair", "integration"),
    "Commonsense Generator": ("commonsense_generation", "general_reasoning", "integration"),
    "Critic / Verifier": ("verification", "general_reasoning"),
    "Reflector": ("verification", "planning", "repair"),
    "Modifier / Repair": ("repair", "verification", "general_reasoning"),
    "Integrator / Concluder": ("integration", "general_reasoning", "verification"),
    "Python Tool Agent": ("tool_use", "software_engineering", "quantitative_reasoning", "verification"),
}

TASK_CAPABILITY_SCOPES = {
    "GSM-Hard": ("quantitative_reasoning", "general_reasoning", "planning", "verification", "integration", "tool_use"),
    "gsm-hard": ("quantitative_reasoning", "general_reasoning", "planning", "verification", "integration", "tool_use"),
    "MMLU-Pro": ("domain_reasoning", "general_reasoning", "planning", "verification", "integration"),
    "SRDD": ("software_engineering", "general_reasoning", "planning", "verification", "repair", "integration", "tool_use"),
    "CW": ("commonsense_generation", "general_reasoning", "verification", "repair", "integration"),
}

def relevant_capabilities(role_name: str, task_type: str) -> tuple[str, ...]:
    role_scope = ROLE_CAPABILITY_SCOPES.get(role_name)
    task_scope = TASK_CAPABILITY_SCOPES.get(task_type)
    if role_scope is None:
        raise KeyError(f"Unknown role capability scope: {role_name!r}")
    if task_scope is None:
        raise KeyError(f"Unknown task capability scope: {task_type!r}")
    task_set = set(task_scope)
    return tuple(capability for capability in role_scope if capability in task_set)
