from __future__ import annotations

from dataclasses import replace
from typing import Iterable

from role_aware.schemas import (
    CAPABILITY_DIMENSIONS,
    DecodingConfig,
    IOSchema,
    RoleCard,
    TeammateSpec,
)

ROLE_DEFINITIONS = (
    ("Planner / Decomposer", "Decompose the task and propose a structured solution plan.", "planning",
     {"planning": 0.95, "general_reasoning": 0.75, "integration": 0.70}),
    ("General Reasoner", "Solve reasoning subtasks using the supplied context.", "reasoning",
     {"general_reasoning": 0.95, "verification": 0.65, "integration": 0.65}),
    ("Quantitative Reasoner", "Solve quantitative reasoning subtasks accurately.", "reasoning",
     {"quantitative_reasoning": 0.95, "general_reasoning": 0.80, "verification": 0.70}),
    ("Domain Reasoner", "Apply task-domain knowledge while checking factual constraints.", "reasoning",
     {"domain_reasoning": 0.95, "general_reasoning": 0.75, "verification": 0.70}),
    ("Software Engineer", "Design and implement complete, executable software solutions.", "reasoning",
     {"software_engineering": 0.95, "planning": 0.75, "verification": 0.75, "repair": 0.80, "integration": 0.75}),
    ("Commonsense Generator", "Generate coherent text grounded in everyday commonsense.", "reasoning",
     {"commonsense_generation": 0.95, "general_reasoning": 0.75, "integration": 0.75}),
    ("Critic / Verifier", "Find errors and constraint violations in previous work.", "critique",
     {"verification": 0.95, "general_reasoning": 0.75}),
    ("Reflector", "Assess the current trajectory and identify improvements.", "reflect",
     {"verification": 0.80, "planning": 0.75, "repair": 0.75}),
    ("Modifier / Repair", "Repair incorrect work while preserving correct content.", "modify",
     {"repair": 0.95, "verification": 0.85, "general_reasoning": 0.70}),
    ("Integrator / Concluder", "Integrate valid evidence into the final task answer.", "conclude",
     {"integration": 0.95, "general_reasoning": 0.80, "verification": 0.75}),
    ("Python Tool Agent", "Perform deterministic computation and Python execution.", "run_python",
     {"tool_use": 0.95, "software_engineering": 0.85, "quantitative_reasoning": 0.85, "verification": 0.70}),
)

MODEL_PROVIDERS = {
    "qwen-2.5-7b": "huggingface_router",
    "qwen-2.5-14b": "huggingface_router",
    "llama-3.2-3b": "huggingface_router",
    "llama-3.1-8b": "huggingface_router",
    "mistralai/ministral-3b-2512": "openrouter",
    "mistral-nemo-12b": "openrouter",
}

SCENARIO_MODELS = {
    "s0": (
        ("qwen-2.5-7b", "llama-3.2-3b"),
        ("qwen-2.5-14b", "llama-3.1-8b"),
    ),
    "s1": (
        ("qwen-2.5-14b", "llama-3.1-8b"),
        ("qwen-2.5-7b", "llama-3.2-3b"),
    ),
    "s2_backbone": (
        ("qwen-2.5-14b", "llama-3.1-8b"),
        ("qwen-2.5-14b", "llama-3.1-8b"),
    ),
    "s3": (
        ("mistralai/ministral-3b-2512", "mistral-nemo-12b"),
        ("mistral-nemo-12b", "mistralai/ministral-3b-2512"),
    ),
}

def role_card(role_index: int) -> RoleCard:
    role_name, goal, action, overrides = ROLE_DEFINITIONS[role_index]
    prior = {name: 0.5 for name in CAPABILITY_DIMENSIONS}
    prior.update(overrides)
    is_tool = action == "run_python"
    forbidden = (
        "read_file", "search_arxiv", "search_bing", "access_website"
    ) + (() if is_tool else ("run_python",))
    return RoleCard(
        role_name=role_name,
        role_goal=goal,
        core_functions=(goal,),
        allowed_actions=(action,),
        forbidden_actions=forbidden,
        expected_input=IOSchema(
            type="task_context",
            requirements=("task", "interaction_history"),
        ),
        expected_output=IOSchema(
            type="action_result",
            fields=("result", "confidence"),
        ),
        tools=("run_python",) if is_tool else (),
        capability_prior=prior,
    )

def build_pool(scenario: str, teammates_per_role: int = 2) -> tuple[TeammateSpec, ...]:
    if teammates_per_role <= 0:
        raise ValueError("teammates_per_role must be positive")
    if scenario == "s3_mixed":
        model_rows = (
            ("qwen-2.5-7b", "mistralai/ministral-3b-2512"),
            ("llama-3.1-8b", "mistral-nemo-12b"),
        )
    else:
        base_scenario = {
            "s2_role": "s0",
            "s2_tools": "s0",
            "s2_cardinality": "s0",
        }.get(scenario, scenario)
        try:
            model_rows = SCENARIO_MODELS[base_scenario]
        except KeyError as exc:
            raise ValueError(f"Unknown scenario: {scenario!r}") from exc

    specs = []
    for role_index in range(len(ROLE_DEFINITIONS)):
        card_index = (role_index + 1) % len(ROLE_DEFINITIONS) if scenario == "s2_role" else role_index
        card = role_card(card_index)
        for teammate_index in range(teammates_per_role):
            row = model_rows[role_index % len(model_rows)]
            backbone = row[teammate_index % len(row)]
            instance_card = card
            if scenario == "s2_tools" and teammate_index == 0:
                if card.role_name == "Python Tool Agent":
                    instance_card = replace(
                        card,
                        allowed_actions=("reasoning",),
                        tools=(),
                        forbidden_actions=tuple(
                            dict.fromkeys(card.forbidden_actions + ("run_python",))
                        ),
                    )
                elif card.role_name == "Software Engineer":
                    instance_card = replace(
                        card,
                        allowed_actions=("run_python",),
                        tools=("run_python",),
                        forbidden_actions=tuple(
                            action for action in card.forbidden_actions
                            if action != "run_python"
                        ),
                    )
            specs.append(
                TeammateSpec(
                    teammate_id=f"{scenario}-r{role_index:02d}-t{teammate_index:02d}",
                    backbone=backbone,
                    provider_profile=MODEL_PROVIDERS[backbone],
                    decoding=DecodingConfig(temperature=0.1, max_tokens=4096),
                    role_card=instance_card,
                    metadata={
                        "scenario": scenario,
                        "role_slot": role_index,
                        "teammate_slot": teammate_index,
                    },
                )
            )
    return tuple(specs)
