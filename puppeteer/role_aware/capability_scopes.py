from __future__ import annotations

from role_aware.schemas import ACTIVE_CAPABILITY_SCHEMA

LEGACY_ROLE_CAPABILITY_SCOPES = {
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
    "Quantitative & Formal Reasoner": ("quantitative_reasoning", "general_reasoning", "verification", "planning"),
    "Natural & Life Science Specialist": ("domain_reasoning", "general_reasoning", "verification", "integration"),
    "Computing & Engineering Specialist": ("software_engineering", "quantitative_reasoning", "domain_reasoning", "general_reasoning", "verification"),
    "Social, Legal & Business Specialist": ("domain_reasoning", "general_reasoning", "quantitative_reasoning", "verification", "integration"),
    "Humanities & Behavioral Specialist": ("domain_reasoning", "general_reasoning", "commonsense_generation", "verification", "integration"),
    "Generalist Independent Solver": ("planning", "general_reasoning", "quantitative_reasoning", "domain_reasoning", "verification", "integration"),
    "Adversarial Verifier": ("verification", "general_reasoning", "quantitative_reasoning", "domain_reasoning", "integration"),
}
LEGACY_TASK_CAPABILITY_SCOPES = {
    "GSM-Hard": ("quantitative_reasoning", "general_reasoning", "planning", "verification", "integration", "tool_use"),
    "gsm-hard": ("quantitative_reasoning", "general_reasoning", "planning", "verification", "integration", "tool_use"),
    "MMLU-Pro": ("domain_reasoning", "general_reasoning", "planning", "quantitative_reasoning", "software_engineering", "commonsense_generation", "verification", "integration"),
    "SRDD": ("software_engineering", "general_reasoning", "planning", "verification", "repair", "integration", "tool_use"),
    "CW": ("commonsense_generation", "general_reasoning", "verification", "repair", "integration"),
}

MMLU_DOMAIN_ROLE_CAPABILITY_SCOPES = {
    "File Analyst": ("general_reasoning", "answer_integration", "evidence_verification"),
    "Academic Researcher": ("general_reasoning", "natural_science", "answer_integration"),
    "Web Researcher": ("general_reasoning", "answer_integration", "evidence_verification"),
    "Website Reader": ("general_reasoning", "answer_integration", "evidence_verification"),
    "Planner / Decomposer": ("task_planning", "general_reasoning", "answer_integration"),
    "Problem Decomposer": ("task_planning", "general_reasoning", "evidence_verification"),
    "General Reasoner": ("general_reasoning", "evidence_verification", "answer_integration"),
    "Quantitative Reasoner": ("formal_quantitative", "general_reasoning", "evidence_verification"),
    "Domain Reasoner": ("general_reasoning", "natural_science", "social_legal_business", "humanities_behavioral", "evidence_verification"),
    "Software Engineer": ("computing_engineering", "task_planning", "evidence_verification", "answer_integration"),
    "Commonsense Generator": ("humanities_behavioral", "general_reasoning", "answer_integration"),
    "Critic / Verifier": ("evidence_verification", "general_reasoning"),
    "Reflector": ("evidence_verification", "task_planning", "answer_integration"),
    "Summarizer": ("answer_integration", "general_reasoning", "evidence_verification"),
    "Modifier / Repair": ("computing_engineering", "evidence_verification", "general_reasoning"),
    "Integrator / Concluder": ("answer_integration", "general_reasoning", "evidence_verification"),
    "Python Tool Agent": ("computing_engineering", "formal_quantitative", "evidence_verification"),
    "Stop Controller": ("stop_decision", "task_planning", "evidence_verification", "answer_integration"),
    "Quantitative & Formal Reasoner": ("formal_quantitative", "general_reasoning", "evidence_verification", "task_planning"),
    "Natural & Life Science Specialist": ("natural_science", "general_reasoning", "evidence_verification", "answer_integration"),
    "Computing & Engineering Specialist": ("computing_engineering", "formal_quantitative", "general_reasoning", "evidence_verification"),
    "Social, Legal & Business Specialist": ("social_legal_business", "general_reasoning", "formal_quantitative", "evidence_verification", "answer_integration"),
    "Humanities & Behavioral Specialist": ("humanities_behavioral", "general_reasoning", "evidence_verification", "answer_integration"),
    "Generalist Independent Solver": ("task_planning", "formal_quantitative", "natural_science", "computing_engineering", "social_legal_business", "humanities_behavioral", "general_reasoning", "evidence_verification", "answer_integration"),
    "Adversarial Verifier": ("evidence_verification", "general_reasoning", "formal_quantitative", "natural_science", "computing_engineering", "social_legal_business", "humanities_behavioral", "answer_integration"),
}

MMLU_DOMAIN_TASK_CAPABILITY_SCOPES = {
    "GSM-Hard": ("formal_quantitative", "general_reasoning", "task_planning", "evidence_verification", "answer_integration", "stop_decision"),
    "gsm-hard": ("formal_quantitative", "general_reasoning", "task_planning", "evidence_verification", "answer_integration", "stop_decision"),
    "MMLU-Pro": ("task_planning", "formal_quantitative", "natural_science", "computing_engineering", "social_legal_business", "humanities_behavioral", "general_reasoning", "evidence_verification", "answer_integration", "stop_decision"),
    "SRDD": ("computing_engineering", "general_reasoning", "task_planning", "evidence_verification", "answer_integration", "stop_decision"),
    "CW": ("humanities_behavioral", "general_reasoning", "evidence_verification", "answer_integration", "stop_decision"),
}


if ACTIVE_CAPABILITY_SCHEMA == "legacy_v1":
    ROLE_CAPABILITY_SCOPES = LEGACY_ROLE_CAPABILITY_SCOPES
    TASK_CAPABILITY_SCOPES = LEGACY_TASK_CAPABILITY_SCOPES
else:
    ROLE_CAPABILITY_SCOPES = MMLU_DOMAIN_ROLE_CAPABILITY_SCOPES
    TASK_CAPABILITY_SCOPES = MMLU_DOMAIN_TASK_CAPABILITY_SCOPES

def relevant_capabilities(role_name: str, task_type: str) -> tuple[str, ...]:
    role_scope = ROLE_CAPABILITY_SCOPES.get(role_name)
    task_scope = TASK_CAPABILITY_SCOPES.get(task_type)
    if role_scope is None:
        raise KeyError(f"Unknown role capability scope: {role_name!r}")
    if task_scope is None:
        raise KeyError(f"Unknown task capability scope: {task_type!r}")
    task_set = set(task_scope)
    return tuple(capability for capability in role_scope if capability in task_set)
