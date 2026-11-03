from __future__ import annotations

import hashlib
import json
import random
from collections import Counter
from dataclasses import replace
from pathlib import Path
from typing import Any, Iterable

from .generator import (
    ARTIFACTS,
    FACTS,
    agent_catalog,
    alpha_scenarios,
    generate_decision_rows,
)
from .loader import BenchmarkBundleV2, validate_bundle
from .schema_definitions import all_schemas
from .schemas import ScenarioSpecV2, TransitionSpecV2


BETA_SCHEMA_VERSION = "2.0-beta"
FAMILIES = (
    "sequential_dependency",
    "parallel_join",
    "conditional_branch",
    "conflict_resolution",
    "tool_routing",
    "failure_recovery",
    "terminal_control",
    "budget_sensitive",
)

TASK_SURFACES: dict[str, tuple[str, ...]] = {
    "sequential_dependency": (
        "Resolve the linked evidence dependencies in the order justified by the current state, then commit a concise result.",
        "Produce a supported final artifact from a chain of dependent evidence relations.",
        "Follow the unresolved evidence chain, combine the resulting claims, and return the committed artifact.",
    ),
    "parallel_join": (
        "Combine the independent evidence branches into one supported operational report.",
        "Resolve all currently open evidence branches and join them into a final artifact.",
        "Build one committed result from the independent source branches available to the workflow.",
    ),
    "conditional_branch": (
        "Inspect the available attachment with the appropriate tool and commit the requested result.",
        "Use the observable artifact type to extract its content and produce the final deliverable.",
        "Route the supplied operational artifact through the required inspection and integration steps.",
    ),
    "conflict_resolution": (
        "Finalize the evidence-grounded response after handling its currently observable evidence condition.",
        "Use the current candidate and evidence state to decide whether checking or commitment is needed.",
        "Resolve any visible support issue before committing the current candidate artifact.",
    ),
    "tool_routing": (
        "Use the supplied deterministic artifact to produce, validate, and commit the requested structured result.",
        "Route the available file or executable through the necessary deterministic processing stages.",
        "Create a validated final artifact from the observable local resource and its processing state.",
    ),
    "failure_recovery": (
        "Obtain the required evidence and adapt the method if an executed attempt fails or makes no progress.",
        "Recover from an observable execution failure, then compose and commit the result.",
        "Use execution feedback to revise the failed method before producing the final artifact.",
    ),
    "terminal_control": (
        "Return the answer artifact only when the observable workflow state is complete.",
        "Decide whether the current candidate needs further work, commitment, or a terminal stop.",
        "Complete the remaining terminal operation without repeating already completed work.",
    ),
    "budget_sensitive": (
        "Complete the supported candidate using a route feasible under the visible remaining budget.",
        "Choose the next admissible operation while respecting the stated execution budget.",
        "Commit the requested artifact without spending resources on an infeasible continuation.",
    ),
}


def _beta_status(status: str) -> str:
    return {
        "SUCCESS": "EXECUTED_SUCCESS",
        "NO_PROGRESS": "EXECUTED_NO_PROGRESS",
        "RETRYABLE_FAILURE": "EXECUTED_RETRYABLE_FAILURE",
        "INVALID_OUTPUT": "EXECUTED_INVALID_OUTPUT",
        "HARD_FAILURE": "EXECUTED_HARD_FAILURE",
    }.get(status, status)


def _clone_transition(
    transition: TransitionSpecV2, scenario_id: str
) -> TransitionSpecV2:
    return replace(
        transition,
        transition_id=f"{scenario_id}:{transition.transition_id}",
        status=_beta_status(transition.status),
        requires_last_statuses=frozenset(
            _beta_status(item) for item in transition.requires_last_statuses
        ),
    )


def _split_and_template(number: int) -> tuple[str, int]:
    if number <= 3:
        return "development", 1
    if number <= 7:
        return "validation", 2
    if number <= 9:
        return "test", 3
    if number <= 12:
        return "test", 4
    return "test", 5


def _counterfactual_pair(number: int) -> int | None:
    pairs = ((1, 2), (4, 5), (6, 7), (8, 9), (10, 11), (12, 13), (14, 15))
    for pair_index, pair in enumerate(pairs, start=1):
        if number in pair:
            return pair_index
    return None


def _pool_for(
    base: ScenarioSpecV2,
    family: str,
    number: int,
    seed: int,
) -> tuple[tuple[str, ...], tuple[str, ...], str]:
    catalog = agent_catalog()
    active = {item.action for item in base.transitions}
    pool = list(dict.fromkeys(base.agent_pool))
    mode = (number - 1) % 5
    mode_name = (
        "base",
        "irrelevant_noise",
        "adjacent_noise",
        "order_permutation",
        "stress_pool",
    )[mode]
    irrelevant = ["agent_15", "agent_16", "agent_17", "agent_23"]
    adjacent = ["agent_12", "agent_14", "agent_18", "agent_19", "agent_24"]
    if mode == 1:
        pool.extend(irrelevant)
    elif mode == 2:
        pool.extend(adjacent)
    elif mode == 4:
        pool = list(catalog)
    pool = list(dict.fromkeys(pool))
    random.Random(f"beta:{seed}:{family}:{number}").shuffle(pool)
    distractors = tuple(item for item in pool if item not in active)
    return tuple(pool), distractors, mode_name


def _public_context(scenario: ScenarioSpecV2) -> dict[str, Any]:
    facts = set(scenario.initial_facts)
    artifacts = set(scenario.initial_artifacts)
    resources: list[dict[str, str]] = []
    if "audio_attachment" in artifacts:
        resources.append({"type": "audio_attachment", "status": "available"})
    if "spreadsheet_attachment" in artifacts:
        resources.append({"type": "spreadsheet_attachment", "status": "available"})
    if "archive_attachment" in artifacts:
        resources.append({"type": "structured_file_bundle", "status": "available"})
    if "script_bundle" in artifacts:
        resources.append({"type": "deterministic_script_bundle", "status": "available"})
    if any("corpus" in fact or fact == "first_source_pending" for fact in facts):
        resources.append({"type": "supplied_text_collection", "status": "available"})
    if any("web" in fact or "query" in fact for fact in facts):
        resources.append({"type": "external_search", "status": "allowed"})
    return {
        "available_resources": resources,
        "task_constraints": {
            "use_only_observable_state": True,
            "hidden_source_location_provided": False,
        },
    }


def _make_source_observable(
    scenario: ScenarioSpecV2, second_action: str
) -> ScenarioSpecV2:
    descriptions = dict(scenario.fact_descriptions)
    transitions = list(scenario.transitions)
    if second_action == "agent_02":
        descriptions["second_source_pending"] = (
            "The resolved relation identifies an uninspected passage in the supplied text collection."
        )
        cue = "The later target and its supplied-text reference became observable."
    else:
        descriptions["second_source_pending"] = (
            "The resolved relation identifies a current-information dependency not covered by the supplied material."
        )
        cue = "The later target and its current-information requirement became observable."
    transitions = [
        replace(
            item,
            public_effect=cue,
        )
        if "relation-" in item.transition_id
        else item
        for item in transitions
    ]
    return replace(scenario, fact_descriptions=descriptions, transitions=tuple(transitions))


def _make_epistemic_source_variant(
    scenario: ScenarioSpecV2, successful_action: str
) -> ScenarioSpecV2:
    """Expose two reasonable source attempts without revealing which one works."""
    failed_action = "agent_03" if successful_action == "agent_02" else "agent_02"
    descriptions = dict(scenario.fact_descriptions)
    descriptions["second_source_pending"] = (
        "The later evidence location is not yet established; supplied text and external lookup are both available."
    )
    descriptions["alternate_source_pending"] = (
        "One inspected source produced no usable evidence; the other observable source remains untried."
    )
    transitions: list[TransitionSpecV2] = []
    success_transition: TransitionSpecV2 | None = None
    for item in scenario.transitions:
        if item.action == successful_action and "-second" in item.transition_id:
            success_transition = item
            transitions.append(item)
        elif "relation-" in item.transition_id:
            transitions.append(
                replace(
                    item,
                    public_effect=(
                        "The later target became explicit, but its evidence location remains uncertain."
                    ),
                )
            )
        else:
            transitions.append(item)
    if success_transition is None:
        return scenario
    transitions.append(
        replace(
            success_transition,
            transition_id=f"{scenario.scenario_id}:uncertain-source-failure",
            action=failed_action,
            add_facts=frozenset({"alternate_source_pending"}),
            remove_facts=frozenset({"second_source_pending"}),
            add_artifacts=frozenset(),
            status="EXECUTED_NO_PROGRESS",
            public_effect=(
                "The inspected source produced no usable evidence; the other available source remains untried."
            ),
        )
    )
    transitions.append(
        replace(
            success_transition,
            transition_id=f"{scenario.scenario_id}:uncertain-source-success",
            requires_all=frozenset({"relation_resolved", "alternate_source_pending"}),
        )
    )
    pool = tuple(dict.fromkeys((*scenario.agent_pool, failed_action)))
    return replace(
        scenario,
        transitions=tuple(transitions),
        fact_descriptions=descriptions,
        agent_pool=pool,
        distractor_ids=tuple(
            item for item in scenario.distractor_ids if item != failed_action
        ),
        difficulty_tags=tuple((*scenario.difficulty_tags, "epistemic_exploration")),
    )


def _add_scheduled_execution_failure(
    scenario: ScenarioSpecV2,
) -> ScenarioSpecV2:
    """Add an unavoidable but recoverable actor failure to one observable step."""
    initial = scenario.initial_state()
    candidates = [item for item in scenario.transitions if item.matches(initial)]
    if not candidates:
        return scenario
    original = candidates[0]
    trigger_candidates = list(original.remove_facts) or [
        item
        for item in original.requires_all
        if any(
            marker in item
            for marker in ("pending", "unresolved", "unmapped", "present", "ready")
        )
    ]
    if not trigger_candidates:
        return scenario
    trigger = sorted(trigger_candidates)[0]
    stalled = f"scheduled_stall_{scenario.family}_{scenario.instance_id}"
    retry_ready = f"scheduled_retry_{scenario.family}_{scenario.instance_id}"
    descriptions = dict(scenario.fact_descriptions)
    descriptions[stalled] = (
        "An executed method failed to produce usable output and now requires a materially revised attempt."
    )
    descriptions[retry_ready] = (
        "A materially revised method is ready for the previously attempted operation."
    )
    failure = replace(
        original,
        transition_id=f"{scenario.scenario_id}:scheduled-failure",
        add_facts=frozenset({stalled}),
        remove_facts=frozenset({trigger}),
        add_artifacts=frozenset(),
        remove_artifacts=frozenset(),
        status="EXECUTED_RETRYABLE_FAILURE",
        public_effect=(
            "The operation executed but failed to produce usable output; an unchanged retry is not appropriate."
        ),
    )
    recovery = TransitionSpecV2(
        transition_id=f"{scenario.scenario_id}:scheduled-recovery",
        action="agent_12",
        requires_all=frozenset({stalled}),
        add_facts=frozenset({retry_ready}),
        remove_facts=frozenset({stalled}),
        status="EXECUTED_SUCCESS",
        public_effect="A materially revised method is ready for the failed operation.",
    )
    retry = replace(
        original,
        transition_id=f"{scenario.scenario_id}:scheduled-retry",
        requires_all=frozenset(
            (set(original.requires_all) - {trigger}) | {retry_ready}
        ),
        remove_facts=frozenset(set(original.remove_facts) | {retry_ready}),
        add_facts=frozenset(
            set(original.add_facts)
            | ({trigger} if trigger not in original.remove_facts else set())
        ),
        status="EXECUTED_SUCCESS",
    )
    transitions = [
        failure if item.transition_id == original.transition_id else item
        for item in scenario.transitions
    ]
    transitions.extend((recovery, retry))
    pool = tuple(dict.fromkeys((*scenario.agent_pool, "agent_12")))
    return replace(
        scenario,
        transitions=tuple(transitions),
        agent_pool=pool,
        distractor_ids=tuple(
            item for item in scenario.distractor_ids if item != "agent_12"
        ),
        fact_descriptions=descriptions,
        max_steps=scenario.max_steps + 2,
        max_cost=scenario.max_cost + 2.0,
        difficulty_tags=tuple(
            (*scenario.difficulty_tags, "scheduled_execution_failure")
        ),
    )


def _add_equivalent_finalizer(scenario: ScenarioSpecV2) -> ScenarioSpecV2:
    """Add a qualitatively adjacent, equal-cost terminal route when justified."""
    finalizers = [
        item
        for item in scenario.transitions
        if item.action == "agent_08" and "final_committed" in item.add_facts
    ]
    if not finalizers:
        return scenario
    original = finalizers[0]
    alternative = replace(
        original,
        transition_id=f"{scenario.scenario_id}:concise-finalizer",
        action="agent_24",
        cost_override=1.0,
        public_effect=(
            "The supported candidate was concisely edited and committed as the final artifact."
        ),
    )
    return replace(
        scenario,
        transitions=tuple((*scenario.transitions, alternative)),
        agent_pool=tuple(dict.fromkeys((*scenario.agent_pool, "agent_24"))),
        distractor_ids=tuple(
            item for item in scenario.distractor_ids if item != "agent_24"
        ),
        difficulty_tags=tuple(
            (*scenario.difficulty_tags, "equivalent_terminal_routes")
        ),
    )


def beta_scenarios(seed: int = 42) -> tuple[ScenarioSpecV2, ...]:
    alpha_by_family: dict[str, list[ScenarioSpecV2]] = {family: [] for family in FAMILIES}
    for scenario in alpha_scenarios(seed):
        alpha_by_family[scenario.family].append(scenario)

    scenarios: list[ScenarioSpecV2] = []
    for family in FAMILIES:
        bases = alpha_by_family[family]
        for number in range(1, 16):
            base = bases[(number - 1) % len(bases)]
            split, topology = _split_and_template(number)
            scenario_id = f"v2b-{family.replace('_', '-')}-{number:02d}"
            transitions = tuple(
                _clone_transition(item, scenario_id) for item in base.transitions
            )
            pool, distractors, pool_mode = _pool_for(base, family, number, seed)
            pair = _counterfactual_pair(number)
            counterfactual = (
                f"v2b-cf-{family}-{split}-{pair:02d}"
                if pair is not None and family != "sequential_dependency"
                else None
            )
            surface_index = ((pair or number) - 1) % len(TASK_SURFACES[family])
            fact_descriptions = dict(FACTS)
            fact_descriptions.update(base.fact_descriptions)
            fact_descriptions["alternate_source_pending"] = (
                "One inspected source produced no usable evidence; another observable source remains available."
            )
            scenario = replace(
                base,
                scenario_id=scenario_id,
                template_id=f"v2b-template-{family}-{topology:02d}",
                instance_id=f"instance-{number:02d}",
                split=split,
                split_group_id=f"v2b-{family}-template-{topology:02d}",
                counterfactual_group_id=counterfactual,
                task=TASK_SURFACES[family][surface_index],
                agent_pool=pool,
                distractor_ids=distractors,
                fact_descriptions=fact_descriptions,
                artifact_descriptions=dict(ARTIFACTS),
                transitions=transitions,
                difficulty_tags=tuple((*base.difficulty_tags, f"pool:{pool_mode}")),
                schema_version=BETA_SCHEMA_VERSION,
                max_stagnant_repeats=3,
            )
            scenario = replace(scenario, public_context=_public_context(scenario))
            if family == "budget_sensitive":
                scenario = replace(
                    scenario,
                    transitions=tuple(
                        replace(item, minimum_steps_remaining=3)
                        if item.action == "agent_07"
                        else item
                        for item in scenario.transitions
                    ),
                )
            if family == "sequential_dependency":
                second_action = "agent_02" if base.instance_id == "instance-01" else "agent_03"
                scenario = _make_source_observable(scenario, second_action)
                if number >= 13:
                    scenario = _make_epistemic_source_variant(scenario, second_action)
                    scenario = replace(scenario, public_context={
                        "available_resources": [
                            {"type": "supplied_text_collection", "status": "available"},
                            {"type": "external_search", "status": "allowed"},
                        ],
                        "task_constraints": {
                            "evidence_location_known": False,
                            "use_only_observable_state": True,
                        },
                    })
            if family not in {"failure_recovery", "budget_sensitive"} and number % 3 == 0:
                scenario = _add_scheduled_execution_failure(scenario)
            scenario = _add_equivalent_finalizer(scenario)
            scenarios.append(scenario)
    return tuple(scenarios)


def build_beta_bundle(seed: int = 42) -> BenchmarkBundleV2:
    scenarios = beta_scenarios(seed)
    bundle = BenchmarkBundleV2(
        metadata={
            "name": "RoutingBench",
            "version": BETA_SCHEMA_VERSION,
            "release_stage": "beta",
            "seed": seed,
            "workflow_count": len(scenarios),
            "family_count": len(FAMILIES),
            "capability_scores": False,
            "backbone_identity": False,
            "live_llm_actors": False,
            "primary_ground_truth": "admissible_actions",
            "diagnostic_ground_truth": [
                "progress_actions",
                "oracle_efficient_actions",
            ],
            "observation_policy": "current_state_plus_compact_execution_memory",
        },
        agents=agent_catalog(),
        scenarios=scenarios,
    )
    validate_bundle(bundle)
    return bundle


def beta_quota_report(
    rows: list[dict[str, Any]], scenarios: tuple[ScenarioSpecV2, ...]
) -> dict[str, Any]:
    count = len(rows)
    provenance = Counter(row["state_provenance"] for row in rows)
    families = Counter(row["workflow_family"] for row in rows)
    splits = Counter(item.split for item in scenarios)
    stop = sum(bool(row["stop_ready"]) for row in rows)
    singleton = sum(
        not row["stop_ready"] and len(row["admissible_actions"]) == 1
        for row in rows
    )
    multi_equal = sum(
        not row["stop_ready"]
        and len(row["admissible_actions"]) >= 2
        and set(row["admissible_actions"]).issubset(row["oracle_efficient_actions"])
        for row in rows
    )
    tradeoff = sum(
        not row["stop_ready"]
        and len(row["admissible_actions"]) >= 2
        and not set(row["admissible_actions"]).issubset(
            row["oracle_efficient_actions"]
        )
        for row in rows
    )

    def rate(value: int) -> float:
        return round(value / count, 6) if count else 0.0

    return {
        "release_stage": "beta",
        "workflow_count": len(scenarios),
        "decision_state_count": count,
        "split_workflow_counts": dict(sorted(splits.items())),
        "singleton_nonterminal_count": singleton,
        "singleton_nonterminal_rate": rate(singleton),
        "multi_equivalent_count": multi_equal,
        "multi_equivalent_rate": rate(multi_equal),
        "multi_tradeoff_count": tradeoff,
        "multi_tradeoff_rate": rate(tradeoff),
        "stop_ready_count": stop,
        "stop_ready_rate": rate(stop),
        "post_failure_count": provenance.get("post_failure", 0),
        "post_failure_rate": rate(provenance.get("post_failure", 0)),
        "post_routing_rejection_count": provenance.get(
            "post_routing_rejection", 0
        ),
        "post_routing_rejection_rate": rate(
            provenance.get("post_routing_rejection", 0)
        ),
        "counterfactual_row_count": sum(
            bool(row.get("counterfactual_group_id")) for row in rows
        ),
        "counterfactual_row_rate": rate(
            sum(bool(row.get("counterfactual_group_id")) for row in rows)
        ),
        "family_counts": dict(sorted(families.items())),
        "provenance_counts": dict(sorted(provenance.items())),
        "target_assessment": {
            "singleton_nonterminal_35_45_percent": 0.35 <= rate(singleton) <= 0.45,
            "multi_equivalent_15_25_percent": 0.15 <= rate(multi_equal) <= 0.25,
            "multi_tradeoff_20_25_percent": 0.20 <= rate(tradeoff) <= 0.25,
            "stop_ready_10_15_percent": 0.10 <= rate(stop) <= 0.15,
            "combined_failure_or_rejection_at_least_15_percent": (
                rate(
                    provenance.get("post_failure", 0)
                    + provenance.get("post_routing_rejection", 0)
                )
                >= 0.15
            ),
        },
    }


def _write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")


def _write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def _checksums(paths: Iterable[Path]) -> dict[str, str]:
    return {
        path.name: hashlib.sha256(path.read_bytes()).hexdigest()
        for path in sorted(paths)
    }


def write_beta_release(output_dir: str | Path, seed: int = 42) -> dict[str, Any]:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    bundle = build_beta_bundle(seed)
    decisions = generate_decision_rows(bundle)
    report = beta_quota_report(decisions, bundle.scenarios)
    metadata = dict(bundle.metadata)
    metadata["decision_state_count"] = len(decisions)
    metadata["splits"] = dict(
        sorted(Counter(item.split for item in bundle.scenarios).items())
    )
    _write_json(output_dir / "metadata.json", metadata)
    _write_jsonl(
        output_dir / "agents.jsonl",
        (agent.to_dict() for agent in bundle.agents.values()),
    )
    _write_jsonl(
        output_dir / "scenarios.jsonl",
        (scenario.to_dict() for scenario in bundle.scenarios),
    )
    _write_jsonl(output_dir / "decision_states.jsonl", decisions)
    _write_json(output_dir / "quota_report.json", report)
    split_manifest = {
        split: [item.scenario_id for item in bundle.scenarios if item.split == split]
        for split in ("development", "validation", "test")
    }
    split_manifest["grouping_rule"] = (
        "All instances sharing a split_group_id remain in one split."
    )
    _write_json(output_dir / "split_manifest.json", split_manifest)
    generated = [
        output_dir / "metadata.json",
        output_dir / "agents.jsonl",
        output_dir / "scenarios.jsonl",
        output_dir / "decision_states.jsonl",
        output_dir / "quota_report.json",
        output_dir / "split_manifest.json",
    ]
    _write_json(output_dir / "checksums.json", _checksums(generated))
    specs_dir = output_dir / "specs"
    specs_dir.mkdir(parents=True, exist_ok=True)
    for filename, schema in all_schemas().items():
        _write_json(specs_dir / filename, schema)
    return {"metadata": metadata, "quota_report": report}
