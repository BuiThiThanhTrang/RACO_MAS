from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parents[1]
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

from model.model_config import model_registry
from role_aware.schemas import CAPABILITY_DIMENSIONS, DecodingConfig, IOSchema, RoleCard, TeammateSpec
from role_aware.validation import validate_teammate_specs


ROLE_BY_ACTION = {
    "read_file": ("File Analyst", "Extract relevant evidence from provided files."),
    "search_arxiv": ("Academic Researcher", "Find relevant academic literature."),
    "search_bing": ("Web Researcher", "Find relevant public web information."),
    "access_website": ("Website Reader", "Extract relevant information from a specified website."),
    "run_python": ("Python Tool Agent", "Perform deterministic computation and code execution."),
    "planning": ("Planner / Decomposer", "Decompose the task and propose a structured solution plan."),
    "reasoning": ("General Reasoner", "Solve reasoning subtasks using the supplied context."),
    "critique": ("Critic / Verifier", "Find errors and constraint violations in previous work."),
    "reflect": ("Reflector", "Assess the current trajectory and identify improvements."),
    "question": ("Problem Decomposer", "Identify useful subquestions and missing information."),
    "summarize": ("Summarizer", "Compress intermediate results without changing their meaning."),
    "conclude": ("Integrator / Concluder", "Integrate valid evidence into the final task answer."),
    "modify": ("Modifier / Repair", "Repair incorrect work while preserving correct content."),
    "terminate": ("Stop Controller", "Stop execution when the answer is ready or the budget is exhausted."),
}

CAPABILITIES_BY_ACTION = {
    "read_file": {"tool_use": 0.9, "domain_reasoning": 0.5},
    "search_arxiv": {"tool_use": 0.9, "domain_reasoning": 0.7},
    "search_bing": {"tool_use": 0.9, "domain_reasoning": 0.6},
    "access_website": {"tool_use": 0.9, "domain_reasoning": 0.6},
    "run_python": {"tool_use": 0.95, "quantitative_reasoning": 0.8, "software_engineering": 0.8},
    "planning": {"planning": 0.95, "general_reasoning": 0.7},
    "reasoning": {"general_reasoning": 0.9, "quantitative_reasoning": 0.6, "domain_reasoning": 0.6},
    "critique": {"verification": 0.95, "general_reasoning": 0.7},
    "reflect": {"verification": 0.75, "planning": 0.7},
    "question": {"planning": 0.8, "general_reasoning": 0.7},
    "summarize": {"integration": 0.75, "general_reasoning": 0.6},
    "conclude": {"integration": 0.95, "general_reasoning": 0.75},
    "modify": {"repair": 0.95, "verification": 0.8},
    "terminate": {"verification": 0.7, "integration": 0.6},
}

TOOL_ACTIONS = {"read_file", "search_arxiv", "search_bing", "access_website", "run_python"}


def _slug(value: str) -> str:
    value = re.sub(r"[^a-z0-9]+", "-", value.casefold()).strip("-")
    return value or "teammate"


def _teammate_id(pool_name: str, index: int, name: str, backbone: str) -> str:
    digest = hashlib.sha256(f"{pool_name}|{index}|{name}|{backbone}".encode("utf-8")).hexdigest()[:10]
    return f"{_slug(pool_name)}-{index:02d}-{digest}"


def migrate_record(record: dict, pool_name: str, index: int) -> TeammateSpec:
    actions = tuple(record.get("actions") or ())
    if len(actions) != 1:
        raise ValueError(f"Expected exactly one fixed action, got {actions!r}")
    action = actions[0]
    if action not in ROLE_BY_ACTION:
        raise ValueError(f"No v1 role mapping for action {action!r}")
    role_name, role_goal = ROLE_BY_ACTION[action]
    backbone = str(record["model_type"])
    model_config = model_registry.get_model_config(backbone)
    if model_config is None:
        raise ValueError(f"Unknown backbone {backbone!r}")

    capability_prior = {name: 0.5 for name in CAPABILITY_DIMENSIONS}
    capability_prior.update(CAPABILITIES_BY_ACTION[action])
    forbidden = tuple(sorted(TOOL_ACTIONS - {action})) if action not in TOOL_ACTIONS else ()
    tools = (action,) if action in TOOL_ACTIONS else ()
    card = RoleCard(
        role_name=role_name,
        role_goal=role_goal,
        core_functions=(role_goal,),
        allowed_actions=(action,),
        forbidden_actions=forbidden,
        expected_input=IOSchema(type="task_context", requirements=("task", "interaction_history")),
        expected_output=IOSchema(type="action_result", fields=("result", "confidence")),
        tools=tools,
        capability_prior=capability_prior,
    )
    return TeammateSpec(
        teammate_id=_teammate_id(pool_name, index, str(record.get("name", role_name)), backbone),
        backbone=backbone,
        provider_profile=model_config.api_profile,
        decoding=DecodingConfig(
            temperature=model_config.temperature,
            max_tokens=min(model_config.max_tokens, 4096),
        ),
        role_card=card,
        metadata={"legacy_name": record.get("name"), "policy": record.get("policy")},
    )


def migrate_file(source: Path, target: Path) -> None:
    records = [json.loads(line) for line in source.read_text(encoding="utf-8").splitlines() if line.strip()]
    specs = validate_teammate_specs(
        migrate_record(record, source.stem, index) for index, record in enumerate(records)
    )
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(
        "\n".join(json.dumps(spec.to_dict(), ensure_ascii=False) for spec in specs) + "\n",
        encoding="utf-8",
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Migrate legacy fixed-action personas to schema v1.")
    parser.add_argument("source")
    parser.add_argument("target")
    args = parser.parse_args()
    migrate_file(Path(args.source), Path(args.target))


if __name__ == "__main__":
    main()
