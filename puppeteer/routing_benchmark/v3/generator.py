from __future__ import annotations

import hashlib
import json
import random
from collections import Counter, deque
from dataclasses import replace
from pathlib import Path
from typing import Any, Iterable

from routing_benchmark.v2.environment import RoutingEnvironmentV2
from routing_benchmark.v2.generator import agent_catalog as v2_agent_catalog
from routing_benchmark.v2.generator_beta import build_beta_bundle
from routing_benchmark.v2.loader import BenchmarkBundleV2, validate_bundle
from routing_benchmark.v2.schemas import (
    FAILURE_STATUSES,
    ROUTING_REJECTION_STATUSES,
    AgentProfileV2,
    ScenarioSpecV2,
    TransitionSpecV2,
)

from .schemas import ProvenanceRecordV3, STOP_ACTION, V3_SCHEMA_VERSION, v3_schema_definitions


PROJECT_DIR = Path(__file__).resolve().parents[2]
SOURCE_RELATIVE_PATH = Path("data/RoutingBench/v2_beta/scenarios.jsonl")
SOURCE_PATH = PROJECT_DIR / SOURCE_RELATIVE_PATH
SELECTED_INSTANCE_NUMBERS = frozenset({1, 2, 4, 5, 8, 9, 10, 11})
DERIVATION_SCRIPT = "scripts/build_routingbench_v3.py"


def _profile(
    agent_id: str,
    role_name: str,
    role_goal: str,
    use_when: Iterable[str],
    avoid_when: Iterable[str],
    supported_inputs: Iterable[str],
    expected_outputs: Iterable[str],
    semantic_family: str,
) -> AgentProfileV2:
    # V3 counts every live actor call as exactly one call. It does not attach a
    # subjective utility or model-strength score to an agent.
    return AgentProfileV2(
        agent_id=agent_id,
        role_name=role_name,
        role_goal=role_goal,
        use_when=tuple(use_when),
        avoid_when=tuple(avoid_when),
        supported_inputs=tuple(supported_inputs),
        expected_outputs=tuple(expected_outputs),
        cost_class="standard",
        action_cost=1.0,
        semantic_family=semantic_family,
        distractor_only=False,
    )


def agent_catalog_v3() -> dict[str, AgentProfileV2]:
    catalog = {
        key: replace(value, action_cost=1.0, cost_class="standard")
        for key, value in v2_agent_catalog().items()
    }
    additions = (
        _profile(
            "agent_25",
            "Broad Corpus Evidence Resolver",
            "Resolve an explicit dependency from supplied textual evidence when several retrieval roles overlap.",
            ["a supplied-text target is observable", "the unresolved dependency can be grounded in the local corpus"],
            ["the task requires live web data", "no evidence target is observable"],
            ["supplied text", "evidence target"],
            ["grounded evidence packet"],
            "retrieval",
        ),
        _profile(
            "agent_26",
            "Artifact Synthesis Specialist",
            "Synthesize complete evidence branches into one supported candidate artifact.",
            ["all required evidence branches are available", "composition remains pending"],
            ["a dependency is unresolved", "a candidate is already complete"],
            ["evidence packets"],
            ["candidate artifact"],
            "composition",
        ),
        _profile(
            "agent_27",
            "Quality and Consistency Reviewer",
            "Check a candidate against available evidence and resolve a visible inconsistency.",
            ["a candidate exists", "verification or conflict handling is observable"],
            ["there is no candidate", "only final commitment remains"],
            ["candidate", "evidence"],
            ["verified candidate", "conflict resolution"],
            "verification",
        ),
        _profile(
            "agent_28",
            "Multimodal Attachment Analyst",
            "Inspect a visible media or structured attachment and expose task-relevant content.",
            ["an uninspected attachment is visible", "the attachment modality is supported"],
            ["the attachment was already inspected", "no attachment exists"],
            ["audio", "image", "file", "spreadsheet"],
            ["structured observation", "parsed file structure"],
            "tool",
        ),
        _profile(
            "agent_29",
            "Deterministic Processing Operator",
            "Execute an observable deterministic calculation or supplied program and return its trace.",
            ["parsed numeric data or executable input is available"],
            ["the required input has not been parsed", "the task is purely interpretive"],
            ["parsed table", "script bundle"],
            ["calculation trace", "execution trace"],
            "tool",
        ),
        _profile(
            "agent_30",
            "Workflow Completion Specialist",
            "Commit a complete supported candidate as the final artifact without adding new claims.",
            ["a supported candidate is complete", "only commitment remains"],
            ["evidence or conflict resolution is incomplete"],
            ["supported candidate"],
            ["committed final artifact"],
            "integration",
        ),
        _profile(
            "agent_31",
            "Current Information Retriever",
            "Retrieve an explicit current-information dependency from permitted external sources.",
            ["a live external evidence target is observable", "external retrieval is permitted"],
            ["the supplied corpus is sufficient", "a failed query has not been revised"],
            ["web query", "revised query"],
            ["external evidence packet"],
            "retrieval",
        ),
        _profile(
            "agent_32",
            "Adaptive Recovery Analyst",
            "Replace a visibly failed method with a materially different executable attempt.",
            ["an actor executed and failed or made no progress"],
            ["the action was rejected before execution", "no failure is visible"],
            ["failure event", "prior attempt"],
            ["revised method"],
            "recovery",
        ),
    )
    catalog.update({item.agent_id: item for item in additions})
    return catalog


ALTERNATIVE_ACTIONS = {
    "agent_02": "agent_25",
    "agent_03": "agent_31",
    "agent_06": "agent_26",
    "agent_07": "agent_27",
    "agent_08": "agent_30",
    "agent_09": "agent_28",
    "agent_10": "agent_28",
    "agent_11": "agent_29",
    "agent_12": "agent_32",
    "agent_21": "agent_29",
}


def _add_contract_equivalents(
    transitions: tuple[TransitionSpecV2, ...], scenario_id: str
) -> tuple[TransitionSpecV2, ...]:
    candidates: list[TransitionSpecV2] = []
    signatures: Counter[tuple[Any, ...]] = Counter()
    for item in transitions:
        alternative = ALTERNATIVE_ACTIONS.get(item.action)
        if not alternative:
            continue
        signature = (
            alternative,
            item.requires_all,
            item.requires_any,
            item.forbids,
            item.requires_artifacts,
            item.requires_last_statuses,
            item.requires_last_actions,
            item.minimum_steps_remaining,
            item.minimum_cost_remaining,
        )
        signatures[signature] += 1
        candidates.append(
            replace(
                item,
                transition_id=f"{scenario_id}:overlap:{item.transition_id}",
                action=alternative,
                public_effect=(
                    f"An overlapping role contract produced the same observable "
                    f"workflow artifact as {item.action}."
                ),
                cost_override=1.0,
            )
        )
    safe: list[TransitionSpecV2] = []
    for item in candidates:
        signature = (
            item.action,
            item.requires_all,
            item.requires_any,
            item.forbids,
            item.requires_artifacts,
            item.requires_last_statuses,
            item.requires_last_actions,
            item.minimum_steps_remaining,
            item.minimum_cost_remaining,
        )
        if signatures[signature] == 1:
            safe.append(item)
    return tuple((*transitions, *safe))


def _instance_number(scenario: ScenarioSpecV2) -> int:
    return int(str(scenario.instance_id).rsplit("-", 1)[-1])


def _source_hash() -> str:
    if not SOURCE_PATH.exists():
        raise FileNotFoundError(
            f"V3 provenance source is missing: {SOURCE_PATH}. Build V2 beta first."
        )
    return hashlib.sha256(SOURCE_PATH.read_bytes()).hexdigest()


def build_v3_bundle(seed: int = 42) -> tuple[BenchmarkBundleV2, dict[str, ProvenanceRecordV3]]:
    source_bundle = build_beta_bundle(seed)
    catalog = agent_catalog_v3()
    source_hash = _source_hash()
    scenarios: list[ScenarioSpecV2] = []
    provenance: dict[str, ProvenanceRecordV3] = {}
    family_indices: Counter[str] = Counter()
    for source in source_bundle.scenarios:
        number = _instance_number(source)
        if number not in SELECTED_INSTANCE_NUMBERS:
            continue
        family_indices[source.family] += 1
        local_index = family_indices[source.family]
        scenario_id = f"v3a-{source.family.replace('_', '-')}-{local_index:02d}"
        transitions = tuple(
            replace(
                item,
                transition_id=f"{scenario_id}:{item.transition_id}",
                cost_override=1.0,
                minimum_cost_remaining=0.0,
            )
            for item in source.transitions
        )
        transitions = _add_contract_equivalents(transitions, scenario_id)
        active = {item.action for item in transitions}
        pool = list(dict.fromkeys((*source.agent_pool, *sorted(active))))
        pool_mode = "overlap"
        if local_index % 2 == 0:
            pool = list(catalog)
            pool_mode = "stress"
        random.Random(f"v3:{seed}:{scenario_id}").shuffle(pool)
        distractors = tuple(item for item in pool if item not in active)
        provenance_id = f"prov-{scenario_id}"
        scenario = replace(
            source,
            scenario_id=scenario_id,
            template_id=f"v3a-template-{source.family}-{local_index:02d}",
            split_group_id=f"v3a-{source.family}-group-{local_index:02d}",
            counterfactual_group_id=(
                f"v3a:{source.counterfactual_group_id}"
                if source.counterfactual_group_id
                else None
            ),
            agent_pool=tuple(pool),
            distractor_ids=distractors,
            transitions=transitions,
            max_cost=float(source.max_steps),
            difficulty_tags=tuple(
                dict.fromkeys(
                    (
                        *(
                            tag
                            for tag in source.difficulty_tags
                            if not tag.startswith("pool:")
                        ),
                        "v3_live_ready",
                        "overlapping_role_contracts",
                        "set_valued_ground_truth",
                        f"pool:{pool_mode}",
                    )
                )
            ),
            schema_version=V3_SCHEMA_VERSION,
        )
        scenarios.append(scenario)
        provenance[provenance_id] = ProvenanceRecordV3(
            provenance_id=provenance_id,
            scenario_id=scenario_id,
            source_type="derived_project_benchmark",
            source_dataset="RoutingBench",
            source_version="2.0-beta",
            source_split=source.split,
            source_item_id=source.scenario_id,
            source_path=SOURCE_RELATIVE_PATH.as_posix(),
            source_sha256=source_hash,
            license="project-authored; governed by the repository data license",
            derivation_script=DERIVATION_SCRIPT,
            derivation_version=V3_SCHEMA_VERSION,
            transformation_steps=(
                "selected a documented V2-beta workflow specification",
                "converted primary labels to a set of formally valid next actions",
                "added overlapping role contracts with identical transition postconditions",
                "added stress-pool variants without exposing distractor labels",
                "normalized execution cost to one unit per live actor call",
            ),
            validation_method="executable transition preconditions plus release validator",
            human_review_status="pending",
        )
    bundle = BenchmarkBundleV2(
        metadata={
            "name": "RoutingBench",
            "version": V3_SCHEMA_VERSION,
            "release_stage": "alpha",
            "seed": seed,
            "workflow_count": len(scenarios),
            "family_count": len({item.family for item in scenarios}),
            "capability_scores": False,
            "backbone_identity_in_router_input": False,
            "preferred_actions": False,
            "action_utilities": False,
            "primary_ground_truth": "valid_actions",
            "valid_action_derivation": "formal_transition_preconditions",
            "track_b_execution": "live_llm_actor",
            "simulated_actor_outputs": False,
            "human_review_status": "pending",
        },
        agents=catalog,
        scenarios=tuple(scenarios),
    )
    validate_bundle(bundle)
    return bundle, provenance


def _state_provenance(env: RoutingEnvironmentV2) -> str:
    if env.state.steps_used == 0:
        return "initial"
    if env.goal_reached:
        return "terminal"
    if env.state.last_status in ROUTING_REJECTION_STATUSES:
        return "post_routing_rejection"
    if env.state.last_status in FAILURE_STATUSES:
        return "post_execution_failure"
    return "reachable"


def _reachable_states(
    scenario: ScenarioSpecV2,
    agents: dict[str, AgentProfileV2],
    limit: int = 96,
) -> list[RoutingEnvironmentV2]:
    initial = RoutingEnvironmentV2(scenario, agents)
    queue = deque([initial])
    seen = {initial.state.fingerprint()}
    states: list[RoutingEnvironmentV2] = []
    while queue and len(states) < limit:
        env = queue.popleft()
        states.append(env)
        if env.goal_reached:
            continue
        for action in env.admissible_actions():
            if action == STOP_ACTION:
                continue
            child = env.clone()
            child.step(action)
            fingerprint = child.state.fingerprint()
            if fingerprint not in seen and not child.done:
                seen.add(fingerprint)
                queue.append(child)
    for distractor in scenario.distractor_ids[:2]:
        injected = initial.clone()
        injected.step(distractor)
        if not injected.done and injected.admissible_actions():
            states.append(injected)
    return states


def _select_states(
    environments: list[RoutingEnvironmentV2], maximum: int = 14
) -> list[RoutingEnvironmentV2]:
    selected: list[RoutingEnvironmentV2] = []
    seen: set[str] = set()

    def add(env: RoutingEnvironmentV2) -> None:
        key = env.state.fingerprint()
        if key not in seen and len(selected) < maximum:
            seen.add(key)
            selected.append(env)

    ordered = sorted(
        environments,
        key=lambda item: (item.state.steps_used, item.state.fingerprint()),
    )
    groups = (
        ([item for item in ordered if _state_provenance(item) == "initial"], 1),
        ([item for item in ordered if len(item.admissible_actions()) >= 3], 4),
        ([item for item in ordered if len(item.admissible_actions()) == 2], 3),
        ([item for item in ordered if _state_provenance(item) == "post_execution_failure"], 2),
        ([item for item in ordered if _state_provenance(item) == "post_routing_rejection"], 1),
        ([item for item in ordered if item.goal_reached], 1),
    )
    for group, limit in groups:
        for env in group[:limit]:
            add(env)
    for env in ordered:
        if env.goal_reached and any(item.goal_reached for item in selected):
            continue
        add(env)
    return selected


def _describe_fact(scenario: ScenarioSpecV2, fact: str) -> str:
    return scenario.fact_descriptions.get(fact, fact)


def _describe_artifact(scenario: ScenarioSpecV2, artifact: str) -> str:
    return scenario.artifact_descriptions.get(artifact, artifact)


def _transition_conditions(
    scenario: ScenarioSpecV2,
    transition: TransitionSpecV2,
    env: RoutingEnvironmentV2,
) -> tuple[list[str], list[str]]:
    satisfied: list[str] = []
    violated: list[str] = []
    for fact in sorted(transition.requires_all):
        text = f"required fact is visible: {_describe_fact(scenario, fact)}"
        (satisfied if fact in env.state.facts else violated).append(text)
    if transition.requires_any:
        visible = sorted(transition.requires_any.intersection(env.state.facts))
        text = "at least one required alternative fact is visible"
        (satisfied if visible else violated).append(text)
    for fact in sorted(transition.forbids):
        text = f"forbidden condition is absent: {_describe_fact(scenario, fact)}"
        (violated if fact in env.state.facts else satisfied).append(text)
    for artifact in sorted(transition.requires_artifacts):
        text = f"required artifact is available: {_describe_artifact(scenario, artifact)}"
        (satisfied if artifact in env.state.artifacts else violated).append(text)
    if transition.requires_last_statuses:
        text = f"last status is one of {sorted(transition.requires_last_statuses)}"
        (satisfied if env.state.last_status in transition.requires_last_statuses else violated).append(text)
    if transition.requires_last_actions:
        text = f"last action is one of {sorted(transition.requires_last_actions)}"
        (satisfied if env.state.last_action in transition.requires_last_actions else violated).append(text)
    if transition.minimum_steps_remaining > env.remaining_steps:
        violated.append(
            f"requires {transition.minimum_steps_remaining} remaining calls but only {env.remaining_steps} remain"
        )
    else:
        satisfied.append("remaining call budget satisfies the transition contract")
    return satisfied, violated


def _action_rationales(env: RoutingEnvironmentV2) -> dict[str, dict[str, Any]]:
    valid = set(env.admissible_actions())
    result: dict[str, dict[str, Any]] = {}
    for action in env.candidate_actions():
        if action == STOP_ACTION:
            if env.goal_reached:
                result[action] = {
                    "status": "valid",
                    "rationale_source": "formal_transition_preconditions",
                    "satisfied_preconditions": [
                        "the committed final artifact satisfies all workflow goals"
                    ],
                    "violated_preconditions": [],
                }
            else:
                result[action] = {
                    "status": "invalid",
                    "rationale_source": "formal_transition_preconditions",
                    "satisfied_preconditions": [],
                    "violated_preconditions": [
                        "the workflow goal is incomplete and no committed final artifact is available"
                    ],
                }
            continue
        transitions = [item for item in env.scenario.transitions if item.action == action]
        analyses = [
            _transition_conditions(env.scenario, item, env) for item in transitions
        ]
        if action in valid:
            matched = next((pair for pair in analyses if not pair[1]), ([], []))
            satisfied = matched[0] or [
                "the role contract is applicable to the current observable state"
            ]
            result[action] = {
                "status": "valid",
                "rationale_source": "formal_transition_preconditions",
                "satisfied_preconditions": satisfied,
                "violated_preconditions": [],
            }
        else:
            best = min(analyses, key=lambda pair: len(pair[1]), default=([], []))
            violations = best[1] or [
                "the role has no executable transition contract for this workflow state"
            ]
            result[action] = {
                "status": "invalid",
                "rationale_source": "formal_transition_preconditions",
                "satisfied_preconditions": best[0],
                "violated_preconditions": violations,
            }
    return result


def _public_observation(env: RoutingEnvironmentV2) -> dict[str, Any]:
    observation = env.observation()
    profiles = []
    for profile in observation["agent_profiles"]:
        profiles.append(
            {
                key: value
                for key, value in profile.items()
                if key not in {"cost_class", "action_cost", "backbone"}
            }
        )
    return {
        "task": observation["task"],
        "task_context": observation.get("task_context", {}),
        "observed_state": observation["state"],
        "execution_memory": observation.get("execution_memory", {}),
        "recent_events": observation.get("recent_events", []),
        "budget": {
            "calls_remaining": env.remaining_steps,
            "reserved_terminal_stop": str(env.scenario.schema_version).startswith(
                "3.0-beta"
            ),
        },
        "agent_pool": profiles,
        "candidate_actions": list(env.candidate_actions()),
    }


def generate_decision_rows_v3(
    bundle: BenchmarkBundleV2,
    provenance: dict[str, ProvenanceRecordV3],
    schema_version: str = V3_SCHEMA_VERSION,
) -> list[dict[str, Any]]:
    provenance_by_scenario = {
        item.scenario_id: item for item in provenance.values()
    }
    rows: list[dict[str, Any]] = []
    for scenario in bundle.scenarios:
        environments = _select_states(_reachable_states(scenario, bundle.agents))
        for index, env in enumerate(environments):
            public = _public_observation(env)
            valid = list(env.admissible_actions())
            source = provenance_by_scenario[scenario.scenario_id]
            rows.append(
                {
                    "decision_id": f"{scenario.scenario_id}:state-{index:03d}",
                    "scenario_id": scenario.scenario_id,
                    "template_id": scenario.template_id,
                    "instance_id": scenario.instance_id,
                    "split": scenario.split,
                    "split_group_id": scenario.split_group_id,
                    "counterfactual_group_id": (
                        f"{scenario.counterfactual_group_id}:initial"
                        if scenario.counterfactual_group_id and env.state.steps_used == 0
                        else None
                    ),
                    "provenance_id": source.provenance_id,
                    "workflow_family": scenario.family,
                    "state_provenance": _state_provenance(env),
                    **public,
                    "valid_actions": valid,
                    "action_rationales": _action_rationales(env),
                    "terminal_valid": env.goal_reached,
                    "difficulty_tags": list(scenario.difficulty_tags),
                    "schema_version": schema_version,
                    "ground_truth_method": "formal_transition_preconditions",
                    "human_review_status": source.human_review_status,
                }
            )
    return rows


def quota_report_v3(
    rows: list[dict[str, Any]], scenarios: tuple[ScenarioSpecV2, ...]
) -> dict[str, Any]:
    nonterminal = [item for item in rows if not item["terminal_valid"]]
    singleton = [item for item in nonterminal if len(item["valid_actions"]) == 1]
    multi = [item for item in nonterminal if len(item["valid_actions"]) >= 2]
    three_plus = [item for item in nonterminal if len(item["valid_actions"]) >= 3]

    def rate(items: list[Any], denominator: list[Any]) -> float:
        return round(len(items) / len(denominator), 6) if denominator else 0.0

    return {
        "release_stage": "alpha",
        "workflow_count": len(scenarios),
        "decision_state_count": len(rows),
        "family_counts": dict(sorted(Counter(item.family for item in scenarios).items())),
        "split_counts": dict(sorted(Counter(item.split for item in scenarios).items())),
        "nonterminal_count": len(nonterminal),
        "singleton_valid_count": len(singleton),
        "singleton_valid_rate": rate(singleton, nonterminal),
        "multi_valid_count": len(multi),
        "multi_valid_rate": rate(multi, nonterminal),
        "three_plus_valid_count": len(three_plus),
        "three_plus_valid_rate": rate(three_plus, nonterminal),
        "terminal_count": sum(item["terminal_valid"] for item in rows),
        "post_execution_failure_count": sum(
            item["state_provenance"] == "post_execution_failure" for item in rows
        ),
        "post_routing_rejection_count": sum(
            item["state_provenance"] == "post_routing_rejection" for item in rows
        ),
        "preferred_actions_present": any("preferred_actions" in item for item in rows),
        "action_utilities_present": any("action_utilities" in item for item in rows),
        "targets": {
            "at_least_50_workflows": len(scenarios) >= 50,
            "singleton_nonterminal_at_most_40_percent": rate(singleton, nonterminal) <= 0.40,
            "multi_nonterminal_at_least_50_percent": rate(multi, nonterminal) >= 0.50,
            "all_eight_families": len({item.family for item in scenarios}) == 8,
        },
    }


def _write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")


def _write_jsonl(path: Path, values: Iterable[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for value in values:
            handle.write(json.dumps(value, ensure_ascii=False) + "\n")


def _checksums(paths: Iterable[Path]) -> dict[str, str]:
    return {
        path.name: hashlib.sha256(path.read_bytes()).hexdigest()
        for path in sorted(paths)
    }


def write_v3_release(output_dir: str | Path, seed: int = 42) -> dict[str, Any]:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    bundle, provenance = build_v3_bundle(seed)
    rows = generate_decision_rows_v3(bundle, provenance)
    report = quota_report_v3(rows, bundle.scenarios)
    metadata = dict(bundle.metadata)
    metadata["decision_state_count"] = len(rows)
    metadata["splits"] = dict(
        sorted(Counter(item.split for item in bundle.scenarios).items())
    )
    _write_json(output_dir / "metadata.json", metadata)
    _write_jsonl(
        output_dir / "agents.jsonl",
        (item.to_dict() for item in bundle.agents.values()),
    )
    scenario_rows = []
    provenance_by_scenario = {item.scenario_id: item for item in provenance.values()}
    for scenario in bundle.scenarios:
        value = scenario.to_dict()
        value["provenance_id"] = provenance_by_scenario[scenario.scenario_id].provenance_id
        scenario_rows.append(value)
    _write_jsonl(output_dir / "scenarios.jsonl", scenario_rows)
    _write_jsonl(
        output_dir / "provenance.jsonl",
        (item.to_dict() for item in provenance.values()),
    )
    _write_jsonl(output_dir / "decision_states.jsonl", rows)
    _write_json(output_dir / "quota_report.json", report)
    _write_json(
        output_dir / "split_manifest.json",
        {
            split: [item.scenario_id for item in bundle.scenarios if item.split == split]
            for split in ("development", "validation", "test")
        },
    )
    generated = [
        output_dir / "metadata.json",
        output_dir / "agents.jsonl",
        output_dir / "scenarios.jsonl",
        output_dir / "provenance.jsonl",
        output_dir / "decision_states.jsonl",
        output_dir / "quota_report.json",
        output_dir / "split_manifest.json",
    ]
    _write_json(output_dir / "checksums.json", _checksums(generated))
    specs = output_dir / "specs"
    specs.mkdir(parents=True, exist_ok=True)
    for name, schema in v3_schema_definitions().items():
        _write_json(specs / name, schema)
    return {"metadata": metadata, "quota_report": report}
