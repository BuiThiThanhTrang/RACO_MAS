from __future__ import annotations

ROLE_CAPABILITY_SCOPES = {
    # Tool-backed research roles use the tool to acquire evidence, then reason
    # about and integrate it.  ``tool_use`` does not receive MMLU-Pro evidence
    # because it is outside that task's capability scope.
    "File Analyst": ("tool_use", "domain_reasoning", "general_reasoning", "integration"),
    "Academic Researcher": ("tool_use", "domain_reasoning", "general_reasoning", "integration"),
    "Web Researcher": ("tool_use", "domain_reasoning", "general_reasoning", "integration"),
    "Website Reader": ("tool_use", "domain_reasoning", "general_reasoning", "integration"),
    "Planner / Decomposer": ("planning", "general_reasoning", "integration"),
    "Problem Decomposer": ("planning", "general_reasoning", "verification"),
    "General Reasoner": ("general_reasoning", "verification", "integration"),
    "Quantitative Reasoner": ("quantitative_reasoning", "general_reasoning", "verification"),
    "Domain Reasoner": ("domain_reasoning", "general_reasoning", "verification"),
    "Software Engineer": ("software_engineering", "planning", "verification", "repair", "integration"),
    "Commonsense Generator": ("commonsense_generation", "general_reasoning", "integration"),
    "Critic / Verifier": ("verification", "general_reasoning"),
    "Reflector": ("verification", "planning", "repair"),
    "Summarizer": ("integration", "general_reasoning", "verification"),
    "Modifier / Repair": ("repair", "verification", "general_reasoning"),
    "Integrator / Concluder": ("integration", "general_reasoning", "verification"),
    "Python Tool Agent": ("tool_use", "software_engineering", "quantitative_reasoning", "verification"),
    "Stop Controller": ("planning", "verification", "integration"),
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
