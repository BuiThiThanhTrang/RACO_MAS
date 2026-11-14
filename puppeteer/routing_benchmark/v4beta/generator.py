from __future__ import annotations

import hashlib
import json
from collections import Counter, deque
from dataclasses import replace
from pathlib import Path
from typing import Any, Iterable, Mapping

from routing_benchmark.v2.loader import BenchmarkBundleV2, validate_bundle
from routing_benchmark.v2.environment import RoutingEnvironmentV2
from routing_benchmark.v2.schemas import STOP_ACTION, StepEventV2
from routing_benchmark.v3.generator import _checksums, _profile
from routing_benchmark.v3.schemas import ProvenanceRecordV3
from routing_benchmark.v4.generator import (
    VARIANTS,
    _build_scenario,
    _public_decision_id,
    agent_catalog_v4,
    generate_decision_rows_v4,
)
from routing_benchmark.v4.post_fault import generate_post_fault_track
from routing_benchmark.v4.public import build_public_observation

from .audit import audit_v3_sources, select_audited_rows
from .schemas import V4_BETA_SCHEMA_VERSION, v4_beta_schema_definitions


PROJECT_DIR = Path(__file__).resolve().parents[2]
DERIVATION_SCRIPT = "scripts/build_routingbench_v4_beta.py"
CORE_DOMAINS = (
    "transactional_service",
    "structured_data_processing",
    "cross_app_workflow",
    "code_debugging",
    "personal_device_tool",
    "scientific_procedure",
    "multi_hop_knowledge",
    "domain_qa_verification",
)
HOLDOUT_DOMAINS = (
    "document_media_processing",
    "resource_constrained_workflow",
)
WORKFLOW_PATTERNS = (
    "sequential_dependency",
    "conditional_branch",
    "conflict_resolution",
    "failure_recovery",
    "parallel_join",
    "terminal_control",
    "tool_routing",
    "budget_sensitive",
)


DOMAIN_CONTRACTS: dict[str, dict[str, str]] = {
    "transactional_service": {
        "input_artifact": "service record, request, and eligibility constraints",
        "candidate_artifact": "proposed transactional mutation",
        "downstream_artifact": "recorded transactional update",
        "final_artifact": "committed service confirmation",
        "primary_agent": "agent_04",
        "alternate_agent": "agent_36",
        "consumer_agent": "agent_29",
        "fault_signal": "The mutation conflicts with a visible eligibility rule, amount, or required term.",
    },
    "structured_data_processing": {
        "input_artifact": "structured source data and transformation contract",
        "candidate_artifact": "transformed structured dataset",
        "downstream_artifact": "schema-bound output dataset",
        "final_artifact": "published validated data artifact",
        "primary_agent": "agent_11",
        "alternate_agent": "agent_41",
        "consumer_agent": "agent_29",
        "fault_signal": "A derived value or required schema field conflicts with the visible transformation contract.",
    },
    "cross_app_workflow": {
        "input_artifact": "source record and destination application contract",
        "candidate_artifact": "cross-application transfer payload",
        "downstream_artifact": "destination application record",
        "final_artifact": "confirmed cross-application update",
        "primary_agent": "agent_10",
        "alternate_agent": "agent_29",
        "consumer_agent": "agent_06",
        "fault_signal": "The transfer payload has a visible field, type, or provenance mismatch.",
    },
    "code_debugging": {
        "input_artifact": "failing check, source snapshot, and patch constraints",
        "candidate_artifact": "candidate code patch",
        "downstream_artifact": "check result for the patched source",
        "final_artifact": "committed verified patch",
        "primary_agent": "agent_21",
        "alternate_agent": "agent_35",
        "consumer_agent": "agent_29",
        "fault_signal": "The patch conflicts with the failing check or produces an inconsistent test result.",
    },
    "personal_device_tool": {
        "input_artifact": "device state, requested setting, and postcondition",
        "candidate_artifact": "proposed device operation",
        "downstream_artifact": "updated device state",
        "final_artifact": "confirmed device configuration",
        "primary_agent": "agent_29",
        "alternate_agent": "agent_14",
        "consumer_agent": "agent_10",
        "fault_signal": "The operation parameters or visible postcondition do not match the requested setting.",
    },
    "scientific_procedure": {
        "input_artifact": "measurement record, method, and analysis constraints",
        "candidate_artifact": "derived scientific result",
        "downstream_artifact": "constraint-checked analysis record",
        "final_artifact": "published supported conclusion",
        "primary_agent": "agent_34",
        "alternate_agent": "agent_33",
        "consumer_agent": "agent_26",
        "fault_signal": "The result violates a visible constraint or conflicts with the supplied measurements.",
    },
    "multi_hop_knowledge": {
        "input_artifact": "supplied passages and a multi-hop question",
        "candidate_artifact": "composed evidence-grounded answer",
        "downstream_artifact": "cross-checked multi-hop answer",
        "final_artifact": "committed supported answer",
        "primary_agent": "agent_02",
        "alternate_agent": "agent_25",
        "consumer_agent": "agent_06",
        "fault_signal": "The answer skips an observable dependency or conflicts with a supplied passage.",
    },
    "domain_qa_verification": {
        "input_artifact": "domain question, candidate options, and visible context",
        "candidate_artifact": "domain-specific candidate answer",
        "downstream_artifact": "independently checked answer record",
        "final_artifact": "committed verified answer",
        "primary_agent": "agent_38",
        "alternate_agent": "agent_33",
        "consumer_agent": "agent_39",
        "fault_signal": "The candidate contradicts an observable premise, option constraint, or independent check.",
    },
    "document_media_processing": {
        "input_artifact": "document or media attachment and extraction contract",
        "candidate_artifact": "structured content extraction",
        "downstream_artifact": "validated document or media record",
        "final_artifact": "committed processed artifact",
        "primary_agent": "agent_28",
        "alternate_agent": "agent_09",
        "consumer_agent": "agent_29",
        "fault_signal": "The extraction omits a visible section, modality cue, or required schema field.",
    },
    "resource_constrained_workflow": {
        "input_artifact": "task contract, route alternatives, and explicit resource budget",
        "candidate_artifact": "budget-feasible route plan",
        "downstream_artifact": "postcondition-checked resource result",
        "final_artifact": "committed budget-compliant result",
        "primary_agent": "agent_13",
        "alternate_agent": "agent_42",
        "consumer_agent": "agent_29",
        "fault_signal": "The result exceeds an explicit budget or omits a required postcondition.",
    },
}


TASK_STEMS: dict[str, tuple[str, ...]] = {
    "transactional_service": (
        "Apply an eligible subscription tier change and verify the recorded renewal terms.",
        "Process a refundable order cancellation and verify the ledger adjustment.",
        "Update a shipping address only after validating the request constraints.",
        "Apply a permitted account ownership correction and confirm the audit record.",
        "Schedule an eligible service pause and verify its effective dates.",
        "Process a partial invoice dispute and confirm the adjusted balance.",
        "Apply a loyalty credit under the visible policy and verify the total.",
        "Change a billing cycle and confirm that the proration terms were recorded.",
        "Restore an eligible suspended service and verify the account state.",
        "Merge duplicate customer records while preserving the authoritative fields.",
        "Apply a permitted reservation change and verify the revised charges.",
        "Process an approved warranty replacement and confirm shipment creation.",
        "Close an eligible account and verify all required completion conditions.",
    ),
    "structured_data_processing": (
        "Normalize a supplied sales table, compute net totals, validate the schema, and publish it.",
        "Join two supplied tables on the declared key and validate unmatched records.",
        "Convert timestamps to the required timezone and validate ordering constraints.",
        "Aggregate measurements by group and validate the calculated summary columns.",
        "Deduplicate records under the supplied precedence rule and publish the result.",
        "Pivot a supplied event table and verify all required categories are represented.",
        "Impute only the permitted missing values and validate the resulting schema.",
        "Filter transactions under the declared rule and reconcile the retained total.",
        "Parse a nested record set into a flat table and validate field types.",
        "Calculate rolling statistics and verify boundary rows against the contract.",
        "Map source categories to a controlled vocabulary and flag unmapped values.",
        "Merge versioned records and retain the latest valid value for each key.",
        "Transform a survey table into the requested analysis format and validate it.",
    ),
    "cross_app_workflow": (
        "Transfer an approved contact update from a form system to the customer registry.",
        "Move a qualified lead into the sales tracker while preserving consent fields.",
        "Create a calendar event from an accepted request and confirm the linked record.",
        "Copy an approved issue into the release tracker and preserve its identifiers.",
        "Synchronize a completed order into the accounting system and verify totals.",
        "Transfer an onboarding record into the access system and confirm required fields.",
        "Publish an approved content record to the catalog and verify its canonical ID.",
        "Move a signed request into the case system while preserving attachment references.",
        "Synchronize an inventory adjustment across the warehouse and storefront records.",
        "Create a support follow-up from a resolved incident and verify the backlink.",
        "Transfer a reviewed application into the decision system with its provenance.",
        "Copy a finalized experiment record into the reporting workspace and verify links.",
        "Synchronize an approved vendor update across procurement and payment systems.",
    ),
    "code_debugging": (
        "Reproduce a failing parser test, patch the boundary condition, rerun checks, and commit.",
        "Fix a deterministic off-by-one defect and verify the regression suite.",
        "Repair a schema validation bug and confirm both positive and negative checks.",
        "Correct a stale-cache defect and verify invalidation behavior.",
        "Fix a deterministic serialization mismatch and validate round-trip output.",
        "Repair an incorrect retry condition and verify failure and success paths.",
        "Correct a timezone conversion defect and rerun the specified tests.",
        "Fix a null-handling crash and validate the required fallback behavior.",
        "Repair a sorting stability defect and verify equal-key ordering.",
        "Correct a permission-check regression and validate allowed and denied cases.",
        "Fix a deterministic aggregation error and reconcile the expected totals.",
        "Repair a malformed API payload and validate its generated schema.",
        "Correct a state-transition bug and verify the terminal postcondition.",
    ),
    "personal_device_tool": (
        "Enable a requested notification schedule and verify its active interval.",
        "Connect to the specified network and confirm the resulting connection state.",
        "Apply a display setting change and verify the effective device state.",
        "Create a permitted automation rule and confirm its trigger and action.",
        "Change an accessibility setting and verify the requested postcondition.",
        "Configure a backup target and confirm that readiness checks pass.",
        "Apply an application permission change and inspect the resulting access state.",
        "Set a focus schedule and verify its exception list.",
        "Configure a paired device and confirm the authenticated connection.",
        "Apply a storage cleanup rule and verify the retained categories.",
        "Change a privacy control and confirm the visible effective policy.",
        "Configure a default application and verify the declared file association.",
        "Apply a battery optimization rule and confirm its excluded applications.",
    ),
    "scientific_procedure": (
        "Calibrate a measurement series, validate the controls, and publish the supported result.",
        "Normalize assay readings against the declared baseline and verify constraints.",
        "Compute a concentration from supplied standards and validate the accepted range.",
        "Process sensor replicates, detect the stated quality violation, and report the result.",
        "Apply the supplied correction factor and verify uncertainty propagation.",
        "Compare two experimental conditions under the declared analysis procedure.",
        "Derive a rate from time-series measurements and validate unit consistency.",
        "Classify observed samples under the supplied decision threshold and audit counts.",
        "Estimate a parameter from the supplied observations and verify residual constraints.",
        "Process a dilution series and validate monotonicity before reporting.",
        "Reconcile instrument readings against a control record and publish the conclusion.",
        "Compute a material balance and verify conservation within the stated tolerance.",
        "Analyze an environmental sample record and validate the declared exclusion rule.",
    ),
    "document_media_processing": (
        "Extract a structured invoice from a document image, validate required fields, and commit it.",
        "Transcribe a short audio instruction, map it to a checklist, and validate completeness.",
        "Parse a signed form, preserve its section provenance, and publish a structured record.",
        "Inspect a chart image, extract the requested series, and verify axis interpretation.",
        "Process a multi-page report into section records and validate their ordering.",
        "Extract named items from a media transcript and verify all referenced timestamps.",
        "Convert a scanned table into structured rows and validate totals.",
        "Inspect an annotated image, extract labeled regions, and validate their identifiers.",
        "Parse a slide deck into titled sections and verify attachment references.",
        "Extract a form from a mixed document bundle and validate its schema.",
    ),
    "resource_constrained_workflow": (
        "Complete a two-tool lookup under the explicit call budget and verify the result.",
        "Choose a budget-feasible transformation route and confirm its postcondition.",
        "Resolve a record update under a strict remaining-call limit and commit it.",
        "Select a low-cost evidence path while preserving the required verification step.",
        "Process a structured artifact under a fixed operation budget and validate it.",
        "Complete a retryable workflow without exceeding the visible retry allowance.",
        "Route a multi-stage device task under the declared time budget and confirm state.",
        "Produce and validate a candidate while reserving one terminal decision.",
        "Recover from a failed method under the remaining-call budget and finish safely.",
        "Choose between equivalent executors under an explicit resource ceiling.",
    ),
}


def _split_for_core(index: int) -> str:
    return "development" if index < 3 else "validation" if index < 8 else "test"


def _base_workflows(v3_root: Path) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    manifest, audit_report = audit_v3_sources(v3_root)
    workflows: list[dict[str, Any]] = []
    for domain in CORE_DOMAINS[:6]:
        for index, task in enumerate(TASK_STEMS[domain]):
            workflows.append(
                {
                    "base_workflow_id": f"{domain}-{index + 1:02d}",
                    "task_domain": domain,
                    "workflow_pattern": WORKFLOW_PATTERNS[index % len(WORKFLOW_PATTERNS)],
                    "split": _split_for_core(index),
                    "task": task,
                    "source_kind": "project_authored",
                    "source_task_group": f"authored:{domain}:{index + 1:02d}",
                    "source_item_id": f"authored-{domain}-{index + 1:02d}",
                    "source_license": "project-authored",
                    **DOMAIN_CONTRACTS[domain],
                }
            )
    migrated = (
        ("multi_hop_knowledge", "multi_hop_evidence"),
        ("domain_qa_verification", "domain_reasoning_verification"),
    )
    for domain, family in migrated:
        for index, source in enumerate(select_audited_rows(manifest, family, 13)):
            workflows.append(
                {
                    "base_workflow_id": f"{domain}-{index + 1:02d}",
                    "task_domain": domain,
                    "workflow_pattern": WORKFLOW_PATTERNS[index % len(WORKFLOW_PATTERNS)],
                    "split": _split_for_core(index),
                    "task": source["task_seed"],
                    "source_kind": "v3_task_seed",
                    "source_task_group": f"v3:{source['v3_scenario_id']}",
                    "source_item_id": source["source_item_id"] or source["v3_scenario_id"],
                    "source_scenario_id": source["v3_scenario_id"],
                    "source_dataset": source["source_dataset"],
                    "source_license": source["source_license"] or "unknown",
                    "source_task_input": source.get("task_input") or {},
                    **DOMAIN_CONTRACTS[domain],
                }
            )
    for domain in HOLDOUT_DOMAINS:
        for index, task in enumerate(TASK_STEMS[domain]):
            workflows.append(
                {
                    "base_workflow_id": f"{domain}-{index + 1:02d}",
                    "task_domain": domain,
                    "workflow_pattern": WORKFLOW_PATTERNS[index % len(WORKFLOW_PATTERNS)],
                    "split": "test",
                    "task": task,
                    "source_kind": "project_authored_holdout",
                    "source_task_group": f"authored:{domain}:{index + 1:02d}",
                    "source_item_id": f"authored-{domain}-{index + 1:02d}",
                    "source_license": "project-authored",
                    **DOMAIN_CONTRACTS[domain],
                }
            )
    return workflows, manifest, audit_report


def build_v4_beta_bundle(v3_root: str | Path, seed: int = 42):
    del seed
    v3_root = Path(v3_root).resolve()
    workflows, audit_manifest, audit_report = _base_workflows(v3_root)
    catalog = agent_catalog_v4()
    catalog["agent_42"] = _profile(
        "agent_42",
        "Resource Route Optimizer",
        "Produce a feasible execution route under an explicit call, cost, or time budget.",
        [
            "the workflow exposes a resource ceiling and at least one unresolved operation",
            "multiple feasible routes or methods can be compared against the visible budget",
        ],
        [
            "no resource constraint is visible",
            "only downstream execution, verification, repair, or final commitment remains",
        ],
        ["task contract", "remaining budget", "route alternatives"],
        ["budget-feasible route plan", "resource-constraint trace"],
        "planning",
    )
    scenarios = []
    fixtures: list[dict[str, Any]] = []
    annotations: dict[str, dict[str, Any]] = {}
    task_inputs: dict[str, dict[str, Any]] = {}
    provenance: dict[str, ProvenanceRecordV3] = {}
    migration: list[dict[str, Any]] = []
    generator_path = Path(__file__)
    generator_hash = hashlib.sha256(generator_path.read_bytes()).hexdigest()
    v3_scenario_path = v3_root / "scenarios.jsonl"
    v3_hash = hashlib.sha256(v3_scenario_path.read_bytes()).hexdigest()

    for base in workflows:
        family_rows = []
        alpha_template = {**base, "domain": base["base_workflow_id"]}
        for variant in VARIANTS:
            scenario, outputs, annotation, task_input = _build_scenario(
                alpha_template, variant, catalog
            )
            scenario = replace(
                scenario,
                template_id=f"v4b-{base['base_workflow_id']}",
                instance_id=f"{base['base_workflow_id']}:{variant}",
                split=base["split"],
                split_group_id=f"v4b-source:{base['source_task_group']}",
                counterfactual_group_id=f"v4b-pair:{base['base_workflow_id']}",
                family=base["task_domain"],
                task=base["task"],
                difficulty_tags=(
                    "controlled_recovery",
                    f"task_domain:{base['task_domain']}",
                    f"workflow_pattern:{base['workflow_pattern']}",
                    f"fault:{variant}",
                    "history_ablation",
                    "set_valued_ground_truth",
                ),
                schema_version=V4_BETA_SCHEMA_VERSION,
                public_context={
                    "task_contract": {
                        "multiple_steps_required": True,
                        "commit_only_after_downstream_postcondition": True,
                    },
                    "source_observability": (
                        "Use only sources and artifacts made observable by the task and execution state."
                    ),
                },
            )
            annotation = {
                **annotation,
                "pair_id": f"v4b-pair:{base['base_workflow_id']}",
                "clean_scenario_id": f"v4-{base['base_workflow_id']}-clean",
                "task_domain": base["task_domain"],
                "workflow_pattern": base["workflow_pattern"],
                "base_workflow_id": base["base_workflow_id"],
            }
            task_input = {
                **task_input,
                "task_domain": base["task_domain"],
                "workflow_pattern": base["workflow_pattern"],
                "base_workflow_id": base["base_workflow_id"],
                "source_task_input": base.get("source_task_input", {}),
            }
            family_rows.append((scenario, outputs, annotation, task_input))

        active = {
            transition.action
            for scenario, _, _, _ in family_rows
            for transition in scenario.transitions
        }
        inactive = [agent for agent in catalog if agent not in active]
        selected = list(active) + inactive[: max(0, 18 - len(active))]
        selected.sort(
            key=lambda agent: hashlib.sha256(
                f"v4b-{base['base_workflow_id']}:{agent}".encode()
            ).hexdigest()
        )
        for scenario, outputs, annotation, task_input in family_rows:
            active_scenario = {transition.action for transition in scenario.transitions}
            scenario = replace(
                scenario,
                agent_pool=tuple(selected),
                distractor_ids=tuple(
                    agent for agent in selected if agent not in active_scenario
                ),
            )
            scenarios.append(scenario)
            fixtures.extend(outputs)
            annotations[scenario.scenario_id] = annotation
            task_inputs[scenario.scenario_id] = task_input
            authored = base["source_kind"].startswith("project_authored")
            provenance_id = f"prov-{scenario.scenario_id}"
            provenance[provenance_id] = ProvenanceRecordV3(
                provenance_id=provenance_id,
                scenario_id=scenario.scenario_id,
                source_type=base["source_kind"],
                source_dataset=(
                    "RoutingBench-V4-beta-authored-seeds"
                    if authored
                    else str(base.get("source_dataset") or "RoutingBench-V3-beta")
                ),
                source_version=(
                    f"sha256:{generator_hash}" if authored else f"sha256:{v3_hash}"
                ),
                source_split=scenario.split,
                source_item_id=str(base["source_item_id"]),
                source_path=(
                    Path(__file__).relative_to(PROJECT_DIR).as_posix()
                    if authored
                    else v3_scenario_path.relative_to(PROJECT_DIR).as_posix()
                ),
                source_sha256=generator_hash if authored else v3_hash,
                license=str(base["source_license"]),
                derivation_script=DERIVATION_SCRIPT,
                derivation_version=V4_BETA_SCHEMA_VERSION,
                transformation_steps=(
                    "audit the source task seed and provenance",
                    "exclude V3 role cards, graph transitions, labels, and utilities",
                    "rewrite the task into the V4 observable workflow contract",
                    "instantiate one clean and four post-execution fault counterfactuals",
                    "derive set-valued safe actions from executable transition contracts",
                ),
                validation_method=(
                    "source hash, split-group audit, executable transition checks, "
                    "counterfactual invariants, role-contract linter, and two-reviewer queue"
                ),
                human_review_status="pending",
            )
        migration.append(
            {
                "base_workflow_id": base["base_workflow_id"],
                "task_domain": base["task_domain"],
                "workflow_pattern": base["workflow_pattern"],
                "split": base["split"],
                "source_kind": base["source_kind"],
                "source_task_group": base["source_task_group"],
                "source_scenario_id": base.get("source_scenario_id"),
                "migration_status": "rewritten_to_v4beta_contract",
                "human_review_status": "pending",
            }
        )

    metadata = {
        "name": "RoutingBench",
        "version": V4_BETA_SCHEMA_VERSION,
        "release_stage": "beta",
        "base_workflow_count": len(workflows),
        "workflow_count": len(scenarios),
        "task_domains": list(CORE_DOMAINS + HOLDOUT_DOMAINS),
        "unseen_domain_holdouts": list(HOLDOUT_DOMAINS),
        "variants": list(VARIANTS),
        "primary_configurations": ["naive_state", "structured_state", "structured_history"],
        "diagnostic_configurations": [
            "structured_raw_history_development_only",
            "structured_history_guard_disabled_development_only",
        ],
        "evaluation_tracks": {
            "A": "isolated next-agent decisions",
            "B": "end-to-end router-induced trajectories",
            "C": "checkpointed post-fault recovery with matched clean controls",
        },
        "preferred_actions": False,
        "action_utilities": False,
        "capability_scores": False,
        "backbone_identity_in_router_input": False,
        "primary_ground_truth": "set_valued_safe_actions",
        "raw_history_policy": "development_only",
        "terminal_stop_budget_policy": "reserved_zero_cost_terminal_decision",
        "human_review_status": "pending",
        "release_blockers": ["two_independent_human_reviews_pending"],
    }
    bundle = BenchmarkBundleV2(metadata=metadata, agents=catalog, scenarios=tuple(scenarios))
    validate_bundle(bundle)
    return (
        bundle,
        provenance,
        fixtures,
        annotations,
        task_inputs,
        workflows,
        migration,
        audit_manifest,
        audit_report,
    )


def _write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")


def _write_jsonl(path: Path, rows: Iterable[Mapping[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(dict(row), ensure_ascii=False) + "\n")


def _workflow_pattern(row: Mapping[str, Any]) -> str:
    for tag in row.get("difficulty_tags") or ():
        if str(tag).startswith("workflow_pattern:"):
            return str(tag).split(":", 1)[1]
    return "unknown"


def _select_beta_decision_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Keep phase coverage while preventing the release from duplicating every BFS state.

    Track A is a curated state-evaluation set, not a replay log.  Track B still
    executes every reachable state online.  We retain up to two normal states
    and one state per other semantic phase for each scenario.
    """

    limits = {
        "initial": 1,
        "normal": 2,
        "verification": 1,
        "recovery": 1,
        "repair": 1,
        "contaminated": 1,
        "terminal": 1,
    }
    counts: Counter[str] = Counter()
    selected: list[dict[str, Any]] = []
    for row in rows:
        phase = str(row.get("state_provenance", "normal"))
        if counts[phase] >= limits.get(phase, 1):
            continue
        counts[phase] += 1
        selected.append(row)
    return selected


def _fast_enumeration_step(env: RoutingEnvironmentV2, action: str) -> None:
    """Apply one transition without V2 Pareto/utility look-ahead.

    V4 labels are derived by ``_safe_actions`` from formal transition contracts,
    so computing hidden V2-alpha success utilities here is redundant.  The
    emitted history fields used by V4 memory remain exact for action, status,
    effect and semantic state change.
    """

    before = env.state.fingerprint()
    before_progress = env.state.progress_fingerprint()
    applicable_before = env.applicable_actions()
    redundant = any(
        item.action == action and item.state_before == before for item in env.history
    )
    transition = env._transition_for(action) if action != STOP_ACTION else None
    cost = env._cost_for(action, transition)
    env._mutate(action, transition, cost)
    after = env.state.fingerprint()
    changed = before_progress != env.state.progress_fingerprint()
    stagnant_repeat = (
        not changed
        and any(
            item.action == action and not item.substantive_state_changed
            for item in reversed(env.history)
        )
    )
    env.history.append(
        StepEventV2(
            step=env.state.steps_used,
            action=action,
            status=str(env.state.last_status),
            applicable=action in applicable_before,
            valid=action in applicable_before,
            efficient=False,
            admissible=action in applicable_before,
            progress_action=changed,
            applicable_before=applicable_before,
            valid_before=applicable_before,
            efficient_before=(),
            admissible_before=applicable_before,
            progress_actions_before=applicable_before,
            state_before=before,
            state_after=after,
            substantive_state_changed=changed,
            redundant=redundant,
            stagnant_repeat=stagnant_repeat,
            cost=cost,
            effect=str(env.state.last_effect or ""),
        )
    )
    threshold = env.scenario.max_stagnant_repeats
    if threshold and not env.done:
        stagnant_tail = 0
        for previous in reversed(env.history):
            if previous.action != action or previous.substantive_state_changed:
                break
            stagnant_tail += 1
        if stagnant_tail >= threshold:
            env.done = True
            env.success = False
            env.terminal_reason = "LOOP_LIMIT"


def _reachable_states_fast(
    scenario, agents, limit: int = 96
) -> list[RoutingEnvironmentV2]:
    initial = RoutingEnvironmentV2(scenario, agents)
    queue = deque([initial])
    seen = {initial.state.fingerprint()}
    states: list[RoutingEnvironmentV2] = []
    while queue and len(states) < limit:
        env = queue.popleft()
        states.append(env)
        if env.done or env.goal_reached:
            continue
        for action in env.applicable_actions():
            if action == STOP_ACTION:
                continue
            child = env.clone()
            _fast_enumeration_step(child, action)
            fingerprint = child.state.fingerprint()
            if fingerprint not in seen:
                seen.add(fingerprint)
                queue.append(child)
    return states


def write_v4_beta_release(
    output_dir: str | Path, v3_root: str | Path, seed: int = 42
) -> dict[str, Any]:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    (
        bundle,
        provenance,
        fixtures,
        annotations,
        task_inputs,
        workflows,
        migration,
        audit_manifest,
        audit_report,
    ) = build_v4_beta_bundle(v3_root, seed)
    provenance_by_scenario = {item.scenario_id: item for item in provenance.values()}

    # Write static artifacts first, then stream Track A rows one scenario at a
    # time.  This bounds peak memory and makes the 620-scenario release usable
    # on ordinary research workstations.
    _write_jsonl(output_dir / "agents.jsonl", (item.to_dict() for item in bundle.agents.values()))
    _write_jsonl(
        output_dir / "scenarios.jsonl",
        ({**scenario.to_dict(), "provenance_id": provenance_by_scenario[scenario.scenario_id].provenance_id} for scenario in bundle.scenarios),
    )
    _write_jsonl(output_dir / "provenance.jsonl", (item.to_dict() for item in provenance.values()))
    _write_jsonl(output_dir / "task_inputs.jsonl", ({"scenario_id": key, "task_input": value} for key, value in sorted(task_inputs.items())))
    _write_jsonl(output_dir / "executor_outputs.jsonl", fixtures)

    decision_count = 0
    with (
        (output_dir / "decision_states.jsonl").open("w", encoding="utf-8") as decision_handle,
        (output_dir / "public_inputs.jsonl").open("w", encoding="utf-8") as public_handle,
        (output_dir / "private_labels.jsonl").open("w", encoding="utf-8") as private_handle,
    ):
        fixture_by_scenario: dict[str, list[dict[str, Any]]] = {}
        for fixture in fixtures:
            fixture_by_scenario.setdefault(str(fixture["scenario_id"]), []).append(fixture)
        import routing_benchmark.v4.generator as v4_generator_module

        original_reachable_states = v4_generator_module._reachable_states
        v4_generator_module._reachable_states = _reachable_states_fast
        try:
            scenario_iterator = enumerate(bundle.scenarios, start=1)
            for scenario_index, scenario in scenario_iterator:
                scenario_id = scenario.scenario_id
                sub_bundle = BenchmarkBundleV2(
                    metadata=bundle.metadata,
                    agents=bundle.agents,
                    scenarios=(scenario,),
                )
                candidate_rows = generate_decision_rows_v4(
                    sub_bundle,
                    {provenance_by_scenario[scenario_id].provenance_id: provenance_by_scenario[scenario_id]},
                    fixture_by_scenario.get(scenario_id, ()),
                    {scenario_id: annotations[scenario_id]},
                )
                selected_rows = _select_beta_decision_rows(candidate_rows)
                selected_ids = {str(row["decision_id"]) for row in selected_rows}
                annotations[scenario_id]["decision_labels"] = [
                    item
                    for item in annotations[scenario_id].get("decision_labels") or ()
                    if str(item["decision_id"]) in selected_ids
                ]
                label_index = {
                    str(item["decision_id"]): item
                    for item in annotations[scenario_id]["decision_labels"]
                }
                for row in selected_rows:
                    row["schema_version"] = V4_BETA_SCHEMA_VERSION
                    row["router_scenario_id"] = "v4b-" + hashlib.sha256(
                        str(row["template_id"]).encode()
                    ).hexdigest()[:16]
                    row["workflow_pattern"] = _workflow_pattern(row)
                    decision_handle.write(json.dumps(row, ensure_ascii=False) + "\n")
                    public_handle.write(
                        json.dumps(
                            {
                                "decision_id": _public_decision_id(row),
                                "split": row["split"],
                                "workflow_family": "dynamic_workflow",
                                "observations": {
                                    view: build_public_observation(row, view)
                                    for view in row["memory_views"]
                                },
                            },
                            ensure_ascii=False,
                        )
                        + "\n"
                    )
                    private = label_index.get(str(row["decision_id"]), {})
                    private_handle.write(
                        json.dumps(
                            {
                                "decision_id": row["decision_id"],
                                "public_decision_id": _public_decision_id(row),
                                "internal_scenario_id": scenario_id,
                                "task_domain": row["workflow_family"],
                                "workflow_pattern": row["workflow_pattern"],
                                "valid_actions": row["valid_actions"],
                                "terminal_valid": row["terminal_valid"],
                                "phase": private.get("phase", row["state_provenance"]),
                                "unsafe_actions": private.get("unsafe_actions", []),
                                "fault_variant": annotations[scenario_id]["variant"],
                                "action_rationales": row["action_rationales"],
                            },
                            ensure_ascii=False,
                        )
                        + "\n"
                    )
                    decision_count += 1
                decision_handle.flush()
                public_handle.flush()
                private_handle.flush()
                if scenario_index % 25 == 0 or scenario_index == len(bundle.scenarios):
                    print(
                        f"[V4-beta] materialized {scenario_index}/{len(bundle.scenarios)} "
                        f"scenarios and {decision_count} Track A states",
                        flush=True,
                    )
        finally:
            v4_generator_module._reachable_states = original_reachable_states
    post_fault_inputs, post_fault_runtime, post_fault_labels, post_fault_pairs = (
        generate_post_fault_track(
            scenarios=bundle.scenarios,
            agents=bundle.agents,
            fixtures=fixtures,
            annotations=annotations,
            benchmark_version=V4_BETA_SCHEMA_VERSION,
        )
    )
    label_by_checkpoint = {
        str(item["checkpoint_id"]): item for item in post_fault_labels
    }
    for collection in (post_fault_inputs, post_fault_runtime, post_fault_labels, post_fault_pairs):
        for item in collection:
            scenario_id = str(
                item.get("internal_scenario_id")
                or item.get("fault_scenario_id")
                or ""
            )
            if not scenario_id and item.get("fault_checkpoint_id"):
                scenario_id = str(
                    label_by_checkpoint.get(
                        str(item["fault_checkpoint_id"]), {}
                    ).get("internal_scenario_id", "")
                )
            annotation = annotations.get(scenario_id, {})
            item["task_domain"] = annotation.get(
                "task_domain", item.get("workflow_family")
            )
            item["workflow_pattern"] = annotation.get("workflow_pattern")

    metadata = {
        **bundle.metadata,
        "decision_state_count": decision_count,
        "post_fault_checkpoint_count": len(post_fault_runtime),
        "post_fault_pair_count": len(post_fault_pairs),
    }
    base_split_counts = Counter(item["split"] for item in workflows)
    quota = {
        "release_stage": "beta",
        "base_workflow_count": len(workflows),
        "scenario_count": len(bundle.scenarios),
        "decision_state_count": decision_count,
        "post_fault_pair_count": len(post_fault_pairs),
        "post_fault_episode_count": len(post_fault_runtime),
        "base_workflow_split_counts": dict(sorted(base_split_counts.items())),
        "scenario_split_counts": dict(sorted(Counter(item.split for item in bundle.scenarios).items())),
        "task_domain_base_counts": dict(sorted(Counter(item["task_domain"] for item in workflows).items())),
        "workflow_pattern_base_counts": dict(sorted(Counter(item["workflow_pattern"] for item in workflows).items())),
        "variant_counts": dict(sorted(Counter(item["variant"] for item in annotations.values()).items())),
        "human_review_status": "pending",
    }
    _write_json(output_dir / "metadata.json", metadata)
    _write_jsonl(output_dir / "recovery_annotations.jsonl", (value for _, value in sorted(annotations.items())))
    _write_jsonl(output_dir / "post_fault_inputs.jsonl", post_fault_inputs)
    _write_jsonl(output_dir / "post_fault_runtime.jsonl", post_fault_runtime)
    _write_jsonl(output_dir / "post_fault_labels.jsonl", post_fault_labels)
    _write_jsonl(output_dir / "post_fault_pair_manifest.jsonl", post_fault_pairs)
    _write_jsonl(output_dir / "base_workflows.jsonl", workflows)
    _write_jsonl(output_dir / "migration_manifest.jsonl", migration)
    _write_jsonl(output_dir / "v3_audit_manifest.jsonl", audit_manifest)
    _write_json(output_dir / "v3_audit_report.json", audit_report)
    _write_json(output_dir / "quota_report.json", quota)
    _write_json(
        output_dir / "split_manifest.json",
        {
            split: [item["base_workflow_id"] for item in workflows if item["split"] == split]
            for split in ("development", "validation", "test")
        },
    )
    review_checklist = [
        "task text and observable state provide sufficient information for all safe actions",
        "role-card use/avoid conditions match the transition preconditions and effects",
        "fault is post-execution and visible without exposing its private class",
        "clean and fault variants share task, pool, initial observation, budget, and terminal contract",
        "no V3 transition, valid-action label, utility, or capability score was copied",
        "source and license fields match the provenance record",
    ]
    _write_jsonl(
        output_dir / "human_review_queue.jsonl",
        (
            {
                "base_workflow_id": item["base_workflow_id"],
                "task_domain": item["task_domain"],
                "workflow_pattern": item["workflow_pattern"],
                "reviewer_1_status": "pending",
                "reviewer_2_status": "pending",
                "adjudication_status": "pending",
                "checklist": review_checklist,
            }
            for item in workflows
        ),
    )
    _write_jsonl(
        output_dir / "post_fault_human_review_queue.jsonl",
        (
            {
                "pair_id": item["pair_id"],
                "task_domain": item.get("task_domain"),
                "workflow_pattern": item.get("workflow_pattern"),
                "reviewer_1_status": "pending",
                "reviewer_2_status": "pending",
                "adjudication_status": "pending",
                "checklist": [
                    "fault and control checkpoints have matched prefix depth, cost, budget, pool, and task contract",
                    "different recovery behavior is justified by observable post-execution evidence",
                    "private fault labels and safe actions are absent from router inputs",
                    "at least one realistic continuation reaches the goal",
                ],
            }
            for item in post_fault_pairs
        ),
    )
    specs = output_dir / "specs"
    specs.mkdir(exist_ok=True)
    for name, schema in v4_beta_schema_definitions().items():
        _write_json(specs / name, schema)
    generated = [path for path in output_dir.iterdir() if path.is_file() and path.name != "checksums.json"]
    _write_json(output_dir / "checksums.json", _checksums(generated))
    return {"metadata": metadata, "quota_report": quota, "audit_report": audit_report}
