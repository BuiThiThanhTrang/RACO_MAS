from __future__ import annotations

import hashlib
import json
from collections import Counter, deque
from dataclasses import replace
from pathlib import Path
from typing import Any, Iterable, Mapping

from routing_benchmark.v2.environment import RoutingEnvironmentV2
from routing_benchmark.v2.loader import BenchmarkBundleV2, validate_bundle
from routing_benchmark.v2.schemas import (
    AgentProfileV2,
    ScenarioSpecV2,
    TransitionSpecV2,
    STOP_ACTION,
)
from routing_benchmark.v3.generator import _action_rationales, _checksums, _profile
from routing_benchmark.v3.schemas import ProvenanceRecordV3
from routing_benchmark.v3beta.generator import agent_catalog_v3_beta

from .memory import build_memory_views
from .post_fault import generate_post_fault_track
from .public import build_public_observation
from .schemas import V4_SCHEMA_VERSION, v4_schema_definitions


PROJECT_DIR = Path(__file__).resolve().parents[2]
TEMPLATE_RELATIVE_PATH = Path("routing_benchmark/v4/workflow_templates.json")
DERIVATION_SCRIPT = "scripts/build_routingbench_v4.py"
VARIANTS = (
    "clean",
    "explicit_invalid",
    "silent_corruption",
    "retryable_failure",
    "conflicting_outputs",
)
DETECTOR_ACTIONS = ("agent_07", "agent_19", "agent_27")
REPAIR_ACTIONS = ("agent_40",)
RECOVERY_ACTIONS = ("agent_12", "agent_32")


def _read_templates() -> list[dict[str, Any]]:
    path = PROJECT_DIR / TEMPLATE_RELATIVE_PATH
    return list(json.loads(path.read_text(encoding="utf-8")))


def agent_catalog_v4() -> dict[str, AgentProfileV2]:
    catalog = agent_catalog_v3_beta()
    # V4 uses agent_29 as the deterministic consumer in several domains.  The
    # inherited V3 card was narrower than the operations it actually performs,
    # which made correct routing depend on a hidden simulator contract.
    catalog["agent_29"] = _profile(
        "agent_29",
        "Deterministic Workflow Operator",
        "Execute a deterministic downstream operation from a complete candidate and return observable postcondition evidence.",
        [
            "a complete candidate artifact and operation contract are available",
            "the pending operation is deterministic and its postcondition can be inspected",
        ],
        [
            "the candidate is missing, contradictory, or still awaiting diagnosis",
            "the task only needs verification, repair, or final commitment",
        ],
        [
            "candidate artifact",
            "structured operation request",
            "parsed table or script bundle",
        ],
        ["downstream artifact", "execution trace", "postcondition evidence"],
        "tool",
    )
    # Recovery is for a failed attempt at the unresolved task operation, not
    # for a merely redundant planning call or a pre-execution rejection.
    catalog["agent_12"] = _profile(
        "agent_12",
        "Recovery Strategist",
        "Replace a failed task-execution method with a materially revised attempt.",
        [
            "an artifact-producing executor attempted the unresolved operation and returned no usable artifact",
            "a concrete execution failure report and prior method are available",
        ],
        [
            "the previous event was only redundant planning or inefficient analysis",
            "the action was rejected before execution",
            "no task-execution failure is observable",
        ],
        ["task-execution failure report", "prior execution method"],
        ["revised executable method"],
        "recovery",
    )
    catalog["agent_32"] = _profile(
        "agent_32",
        "Adaptive Recovery Analyst",
        "Design a materially different executable method after an observable task-executor failure.",
        [
            "a task executor ran and failed or produced no usable artifact",
            "the unresolved operation and prior execution attempt are visible",
        ],
        [
            "the previous event was only redundant planning or inefficient analysis",
            "the action was rejected before execution",
            "no task-execution failure is observable",
        ],
        ["task-execution failure event", "prior execution attempt"],
        ["materially revised executable method"],
        "recovery",
    )
    data_executor = _profile(
        "agent_41",
        "Structured Transformation Executor",
        "Execute a deterministic structured-data transformation from supplied data and an explicit transformation contract.",
        [
            "source data and an executable transformation contract are available",
            "a fresh or independent transformation result is required",
        ],
        [
            "the source data or transformation contract is missing",
            "only verification, repair, or final commitment remains",
        ],
        ["structured source data", "transformation contract"],
        ["transformed data artifact", "transformation trace"],
        "tool",
    )
    catalog[data_executor.agent_id] = data_executor
    repairer = _profile(
        "agent_40",
        "Artifact Repair Specialist",
        "Repair a diagnosed malformed, contradictory, or contaminated artifact without changing unrelated valid content.",
        [
            "a fault has been diagnosed",
            "a repairable source artifact or failure report is available",
        ],
        [
            "no observable diagnosis exists",
            "the original operation has not run",
            "the task only needs final commitment",
        ],
        ["fault diagnosis", "faulty artifact", "output contract"],
        ["corrected candidate artifact"],
        "repair",
    )
    catalog[repairer.agent_id] = repairer
    return catalog


def _transition(
    scenario_id: str,
    suffix: str,
    action: str,
    *,
    requires_all: Iterable[str] = (),
    requires_any: Iterable[str] = (),
    forbids: Iterable[str] = (),
    requires_artifacts: Iterable[str] = (),
    requires_last_statuses: Iterable[str] = (),
    add_facts: Iterable[str] = (),
    remove_facts: Iterable[str] = (),
    add_artifacts: Iterable[str] = (),
    remove_artifacts: Iterable[str] = (),
    status: str = "EXECUTED_SUCCESS",
    effect: str,
) -> TransitionSpecV2:
    return TransitionSpecV2(
        transition_id=f"{scenario_id}:{suffix}:{action}",
        action=action,
        requires_all=frozenset(requires_all),
        requires_any=frozenset(requires_any),
        forbids=frozenset(forbids),
        requires_artifacts=frozenset(requires_artifacts),
        requires_last_statuses=frozenset(requires_last_statuses),
        add_facts=frozenset(add_facts),
        remove_facts=frozenset(remove_facts),
        add_artifacts=frozenset(add_artifacts),
        remove_artifacts=frozenset(remove_artifacts),
        status=status,
        public_effect=effect,
        cost_override=1.0,
        minimum_cost_remaining=0.0,
    )


def _fixture(
    scenario_id: str,
    transition: TransitionSpecV2,
    raw_output: str,
    *,
    fault_signal: bool = False,
) -> dict[str, Any]:
    return {
        "output_id": f"out-{transition.transition_id}",
        "scenario_id": scenario_id,
        "transition_id": transition.transition_id,
        "action": transition.action,
        "status": transition.status,
        "structured_event": {
            "effect": transition.public_effect,
            "added_facts": sorted(transition.add_facts),
            "removed_facts": sorted(transition.remove_facts),
            "added_artifacts": sorted(transition.add_artifacts),
            "removed_artifacts": sorted(transition.remove_artifacts),
        },
        "raw_output": raw_output,
        "contains_fault_signal": fault_signal,
    }


def _build_scenario(
    template: Mapping[str, Any], variant: str, catalog: Mapping[str, AgentProfileV2]
) -> tuple[
    ScenarioSpecV2,
    list[dict[str, Any]],
    dict[str, Any],
    dict[str, Any],
]:
    domain = str(template["domain"])
    scenario_id = f"v4-{domain}-{variant}"
    primary = str(template["primary_agent"])
    alternate = str(template["alternate_agent"])
    consumer = str(template["consumer_agent"])
    transitions: list[TransitionSpecV2] = []
    fixtures: list[dict[str, Any]] = []
    unsafe: list[str] = []
    inefficient: list[str] = []
    fault_transition_ids: list[str] = []

    def add(
        transition: TransitionSpecV2,
        raw_output: str,
        *,
        fault: bool = False,
        unsafe_action: bool = False,
        inefficient_action: bool = False,
    ) -> None:
        transitions.append(transition)
        fixtures.append(
            _fixture(
                scenario_id,
                transition,
                raw_output,
                fault_signal=fault or unsafe_action,
            )
        )
        if fault:
            fault_transition_ids.append(transition.transition_id)
        if unsafe_action:
            unsafe.append(transition.transition_id)
        if inefficient_action:
            inefficient.append(transition.transition_id)

    initial_requires = ("task_available", "input_available", "work_pending")
    input_artifact = str(template["input_artifact"])
    candidate_artifact = str(template["candidate_artifact"])
    downstream_artifact = str(template["downstream_artifact"])
    final_artifact = str(template["final_artifact"])
    fault_signal = str(template["fault_signal"])

    if variant == "clean":
        for suffix, action in (("execute-primary", primary), ("execute-alternate", alternate)):
            add(
                _transition(
                    scenario_id,
                    suffix,
                    action,
                    requires_all=initial_requires,
                    forbids=("candidate_ready",),
                    add_facts=("candidate_ready", "candidate_clean"),
                    remove_facts=("work_pending",),
                    add_artifacts=("candidate_artifact",),
                    effect="A candidate artifact was produced with no observable anomaly.",
                ),
                f"SUCCESS: produced {candidate_artifact}. Contract fields and visible constraints are satisfied.",
            )
    elif variant == "explicit_invalid":
        add(
            _transition(
                scenario_id,
                "inject-invalid",
                primary,
                requires_all=initial_requires,
                forbids=("explicit_failure",),
                add_facts=("explicit_failure", "diagnosis_pending"),
                add_artifacts=("faulty_raw_output",),
                status="EXECUTED_INVALID_OUTPUT",
                effect="The executor ran, but its output failed the visible output contract.",
            ),
            f"INVALID_OUTPUT: attempted {candidate_artifact}; a required field is missing. {fault_signal}",
            fault=True,
        )
        add(
            _transition(
                scenario_id,
                "execute-alternate-clean",
                alternate,
                requires_all=initial_requires,
                forbids=("candidate_ready", "explicit_failure"),
                add_facts=("candidate_ready", "candidate_clean"),
                remove_facts=("work_pending",),
                add_artifacts=("candidate_artifact",),
                effect="The alternate executor produced a clean candidate without triggering the scheduled fault.",
            ),
            f"SUCCESS VIA ALTERNATE EXECUTOR: produced valid {candidate_artifact}.",
        )
        add(
            _transition(
                scenario_id,
                "inspect-invalid",
                "agent_19",
                requires_all=("explicit_failure", "diagnosis_pending"),
                requires_artifacts=("faulty_raw_output",),
                add_facts=("diagnosis_ready", "repair_pending"),
                remove_facts=("diagnosis_pending",),
                add_artifacts=("fault_diagnosis",),
                effect="The malformed output was inspected and a concrete contract violation was diagnosed.",
            ),
            f"DIAGNOSIS: {fault_signal} Repair only the invalid field and preserve valid content.",
        )
        for action in RECOVERY_ACTIONS:
            add(
                _transition(
                    scenario_id,
                    "plan-retry",
                    action,
                    requires_all=("explicit_failure",),
                    forbids=("retry_plan_ready", "diagnosis_ready"),
                    add_facts=("retry_plan_ready",),
                    add_artifacts=("revised_method",),
                    effect="A materially revised execution method is available after the invalid output.",
                ),
                "RECOVERY PLAN: validate required fields before committing the next execution result.",
            )
        add(
            _transition(
                scenario_id,
                "retry-primary",
                primary,
                requires_all=("explicit_failure", "retry_plan_ready", "work_pending"),
                add_facts=("candidate_ready", "candidate_clean"),
                remove_facts=("explicit_failure", "retry_plan_ready", "work_pending"),
                add_artifacts=("candidate_artifact",),
                effect="The revised execution produced a contract-compliant candidate artifact.",
            ),
            f"SUCCESS AFTER REVISED METHOD: produced valid {candidate_artifact}.",
        )
        add(
            _transition(
                scenario_id,
                "alternate-after-invalid",
                alternate,
                requires_all=("explicit_failure", "work_pending"),
                add_facts=("candidate_ready", "candidate_clean"),
                remove_facts=("explicit_failure", "diagnosis_pending", "work_pending"),
                add_artifacts=("candidate_artifact",),
                effect="An alternate executor replaced the invalid output with a valid candidate artifact.",
            ),
            f"SUCCESS VIA ALTERNATE EXECUTOR: produced valid {candidate_artifact}.",
        )
    elif variant == "silent_corruption":
        add(
            _transition(
                scenario_id,
                "inject-silent",
                primary,
                requires_all=initial_requires,
                add_facts=("candidate_ready", "candidate_suspect", "verification_required"),
                remove_facts=("work_pending",),
                add_artifacts=("candidate_artifact",),
                effect="The executor reported success, but the state exposes a consistency anomaly in the candidate.",
            ),
            f"SUCCESS: produced {candidate_artifact}. WARNING SIGNAL: {fault_signal}",
            fault=True,
        )
        add(
            _transition(
                scenario_id,
                "execute-alternate-clean",
                alternate,
                requires_all=initial_requires,
                forbids=("candidate_ready", "candidate_suspect"),
                add_facts=("candidate_ready", "candidate_clean"),
                remove_facts=("work_pending",),
                add_artifacts=("candidate_artifact",),
                effect="The alternate executor produced a clean candidate without triggering the scheduled fault.",
            ),
            f"SUCCESS VIA ALTERNATE EXECUTOR: produced valid {candidate_artifact}.",
        )
    elif variant == "retryable_failure":
        add(
            _transition(
                scenario_id,
                "inject-retryable",
                primary,
                requires_all=initial_requires,
                forbids=("retryable_failure",),
                add_facts=("retryable_failure",),
                add_artifacts=("failure_report",),
                status="EXECUTED_RETRYABLE_FAILURE",
                effect="The executor ran but made no progress because a retryable method failure was observed.",
            ),
            "RETRYABLE_FAILURE: the attempted method produced no usable artifact; a materially revised method is required.",
            fault=True,
        )
        add(
            _transition(
                scenario_id,
                "execute-alternate-clean",
                alternate,
                requires_all=initial_requires,
                forbids=("candidate_ready", "retryable_failure"),
                add_facts=("candidate_ready", "candidate_clean"),
                remove_facts=("work_pending",),
                add_artifacts=("candidate_artifact",),
                effect="The alternate executor completed the operation without triggering the scheduled method failure.",
            ),
            f"SUCCESS VIA ALTERNATE EXECUTOR: produced valid {candidate_artifact}.",
        )
        for action in RECOVERY_ACTIONS:
            add(
                _transition(
                    scenario_id,
                    "revise-method",
                    action,
                    requires_all=("retryable_failure",),
                    forbids=("retry_plan_ready",),
                    add_facts=("retry_plan_ready",),
                    add_artifacts=("revised_method",),
                    effect="A materially different method was prepared for the unresolved operation.",
                ),
                "REVISED METHOD: change the execution approach and validate its postcondition.",
            )
        add(
            _transition(
                scenario_id,
                "retry-with-plan",
                primary,
                requires_all=("retryable_failure", "retry_plan_ready", "work_pending"),
                add_facts=("candidate_ready", "candidate_clean"),
                remove_facts=("retryable_failure", "retry_plan_ready", "work_pending"),
                add_artifacts=("candidate_artifact",),
                effect="The revised retry produced a usable candidate artifact.",
            ),
            f"SUCCESS AFTER RETRY: produced {candidate_artifact} using the revised method.",
        )
        add(
            _transition(
                scenario_id,
                "alternate-after-failure",
                alternate,
                requires_all=("retryable_failure", "work_pending"),
                add_facts=("candidate_ready", "candidate_clean"),
                remove_facts=("retryable_failure", "work_pending"),
                add_artifacts=("candidate_artifact",),
                effect="An alternate executor completed the operation after the retryable failure.",
            ),
            f"SUCCESS VIA ALTERNATE EXECUTOR: produced {candidate_artifact}.",
        )
        add(
            _transition(
                scenario_id,
                "unchanged-retry",
                primary,
                requires_all=("retryable_failure", "work_pending"),
                forbids=("retry_plan_ready",),
                add_facts=("repeated_failure",),
                status="EXECUTED_RETRYABLE_FAILURE",
                effect="The unchanged retry repeated the same method failure without producing progress.",
            ),
            "RETRYABLE_FAILURE: unchanged retry repeated the prior failed approach.",
            unsafe_action=True,
        )
    elif variant == "conflicting_outputs":
        add(
            _transition(
                scenario_id,
                "first-candidate",
                primary,
                requires_all=initial_requires,
                add_facts=("candidate_a_ready", "independent_check_pending"),
                remove_facts=("work_pending",),
                add_artifacts=("candidate_artifact",),
                effect="A first candidate exists, while the visible contract still requires independent confirmation.",
            ),
            f"FIRST CANDIDATE: produced {candidate_artifact}; independent confirmation remains required.",
        )
        add(
            _transition(
                scenario_id,
                "execute-alternate-clean",
                alternate,
                requires_all=initial_requires,
                forbids=("candidate_ready", "candidate_a_ready"),
                add_facts=("candidate_ready", "candidate_clean"),
                remove_facts=("work_pending",),
                add_artifacts=("candidate_artifact",),
                effect="The alternate executor produced a clean candidate without creating a conflicting pair.",
            ),
            f"SUCCESS VIA ALTERNATE EXECUTOR: produced valid {candidate_artifact}.",
        )
        add(
            _transition(
                scenario_id,
                "independent-candidate",
                alternate,
                requires_all=("candidate_a_ready", "independent_check_pending"),
                add_facts=("conflict_observed", "verification_required"),
                remove_facts=("candidate_a_ready",),
                add_artifacts=("conflicting_candidate",),
                effect="Independent execution produced a materially conflicting candidate that requires adjudication.",
            ),
            f"INDEPENDENT CANDIDATE: result conflicts with the first {candidate_artifact}. {fault_signal}",
            fault=True,
        )
    else:
        raise ValueError(f"Unknown V4 variant: {variant}")

    # Planning is semantically applicable at the initial state, but this task
    # already exposes an ordered contract.  Treating it as a pre-execution
    # rejection conflates inefficient execution with an invalid route.
    add(
        _transition(
            scenario_id,
            "redundant-initial-planning",
            "agent_01",
            requires_all=initial_requires,
            status="EXECUTED_INEFFICIENT",
            effect="The planner restated the already visible workflow order but produced no new artifact or state progress.",
        ),
        "NO_PROGRESS: restated the visible workflow sequence without producing an executable artifact.",
        inefficient_action=True,
    )

    if variant in {"silent_corruption", "conflicting_outputs"}:
        trigger_facts = (
            ("candidate_suspect", "verification_required")
            if variant == "silent_corruption"
            else ("conflict_observed", "verification_required")
        )
        for action in ("agent_07", "agent_27"):
            add(
                _transition(
                    scenario_id,
                    "detect-fault",
                    action,
                    requires_all=trigger_facts,
                    add_facts=("diagnosis_ready", "repair_pending"),
                    remove_facts=(
                        "verification_required",
                        "candidate_ready",
                        "candidate_suspect",
                        "conflict_observed",
                        "independent_check_pending",
                    ),
                    add_artifacts=("fault_diagnosis",),
                    effect="A verifier detected and localized the observable inconsistency before final commitment.",
                ),
                f"FAULT DETECTED: {fault_signal}",
            )
        unsafe_requirements = (
            ("candidate_ready", "candidate_suspect")
            if variant == "silent_corruption"
            else ("candidate_a_ready", "independent_check_pending")
        )
        add(
            _transition(
                scenario_id,
                "unsafe-consume",
                consumer,
                requires_all=unsafe_requirements,
                add_facts=("contamination_level_1",),
                remove_facts=(
                    "candidate_ready",
                    "candidate_a_ready",
                    "candidate_suspect",
                    "verification_required",
                    "conflict_observed",
                    "independent_check_pending",
                ),
                add_artifacts=("contaminated_downstream_artifact",),
                effect="An unresolved candidate was consumed and contamination propagated into a downstream artifact.",
            ),
            f"DOWNSTREAM OUTPUT: consumed unresolved {candidate_artifact}; propagated inconsistency into {downstream_artifact}.",
            unsafe_action=True,
        )
        for action in ("agent_07", "agent_27"):
            add(
                _transition(
                    scenario_id,
                    "detect-contamination",
                    action,
                    requires_all=("contamination_level_1",),
                    requires_artifacts=("contaminated_downstream_artifact",),
                    add_facts=("diagnosis_ready", "repair_pending"),
                    remove_facts=(
                        "contamination_level_1",
                        "verification_required",
                        "candidate_suspect",
                        "conflict_observed",
                        "independent_check_pending",
                    ),
                    add_artifacts=("fault_diagnosis",),
                    remove_artifacts=("contaminated_downstream_artifact",),
                    effect="A verifier detected downstream contamination and isolated the faulty artifact.",
                ),
                f"CONTAMINATION DIAGNOSIS: {fault_signal}",
            )
        add(
            _transition(
                scenario_id,
                "unsafe-finalize",
                "agent_08",
                requires_all=("contamination_level_1",),
                requires_artifacts=("contaminated_downstream_artifact",),
                add_facts=("contamination_level_2",),
                status="EXECUTED_HARD_FAILURE",
                effect="The contaminated downstream artifact was committed and caused an irreversible terminal failure.",
            ),
            "HARD_FAILURE: an unverified contaminated artifact was committed.",
            unsafe_action=True,
        )

    if variant in {"explicit_invalid", "silent_corruption", "conflicting_outputs"}:
        add(
            _transition(
                scenario_id,
                "repair-artifact",
                "agent_40",
                requires_all=("diagnosis_ready", "repair_pending"),
                requires_artifacts=("fault_diagnosis",),
                add_facts=("candidate_ready", "candidate_clean"),
                remove_facts=(
                    "explicit_failure",
                    "diagnosis_ready",
                    "repair_pending",
                    "candidate_suspect",
                    "conflict_observed",
                ),
                add_artifacts=("candidate_artifact", "corrected_candidate_artifact"),
                remove_artifacts=("faulty_raw_output",),
                effect="The diagnosed defect was repaired and a corrected candidate artifact is available.",
            ),
            f"REPAIRED ARTIFACT: corrected {candidate_artifact} using the explicit diagnosis.",
        )

    # Optional verification is valid but not required when no anomaly remains.
    add(
        _transition(
            scenario_id,
            "optional-clean-check",
            "agent_27",
            requires_all=("candidate_ready", "candidate_clean"),
            forbids=("candidate_verified", "downstream_ready"),
            requires_artifacts=("candidate_artifact",),
            add_facts=("candidate_verified",),
            add_artifacts=("verification_record",),
            effect="The clean candidate received an optional consistency check.",
        ),
        f"VERIFIED: {candidate_artifact} is consistent with visible inputs.",
    )
    add(
        _transition(
            scenario_id,
            "consume-clean",
            consumer,
            requires_all=("candidate_ready", "candidate_clean"),
            forbids=("downstream_ready", "candidate_suspect", "conflict_observed"),
            requires_artifacts=("candidate_artifact",),
            add_facts=("downstream_ready",),
            remove_facts=("candidate_ready",),
            add_artifacts=("downstream_artifact",),
            effect="The clean candidate was consumed into the required downstream artifact.",
        ),
        f"DOWNSTREAM SUCCESS: created {downstream_artifact} from a clean candidate.",
    )
    for action in ("agent_08", "agent_24"):
        add(
            _transition(
                scenario_id,
                "finalize",
                action,
                requires_all=("downstream_ready",),
                forbids=("contamination_level_1", "contamination_level_2"),
                requires_artifacts=("downstream_artifact",),
                add_facts=("final_committed",),
                add_artifacts=("committed_final_artifact",),
                effect="A final artifact was committed after the required downstream state was reached.",
            ),
            f"FINAL COMMIT: produced {final_artifact}.",
        )

    active = {item.action for item in transitions}
    inactive = [item for item in catalog if item not in active]
    # Stable large pools test distractor robustness without exposing capability scores.
    selected = list(active) + inactive[: max(0, 18 - len(active))]
    selected.sort(key=lambda value: hashlib.sha256(f"{scenario_id}:{value}".encode()).hexdigest())
    scenario = ScenarioSpecV2(
        scenario_id=scenario_id,
        template_id=f"v4-{domain}",
        instance_id=f"{domain}:{variant}",
        split=str(template["split"]),
        split_group_id=f"v4-domain:{domain}",
        counterfactual_group_id=f"v4-pair:{domain}",
        family=domain,
        task=str(template["task"]),
        agent_pool=tuple(selected),
        distractor_ids=tuple(item for item in selected if item not in active),
        initial_facts=frozenset({"task_available", "input_available", "work_pending"}),
        initial_artifacts=frozenset({"task_input"}),
        fact_descriptions={
            "task_available": "The original task is available.",
            "input_available": f"The {input_artifact} is available.",
            "work_pending": "The main operation is still unresolved.",
            "candidate_ready": f"A {candidate_artifact} is available for downstream use.",
            "candidate_clean": "No unresolved observable anomaly remains in the candidate.",
            "candidate_verified": "The candidate received an optional consistency check.",
            "explicit_failure": "The executor returned an output that explicitly failed its contract.",
            "diagnosis_pending": "The explicit output failure has not been diagnosed.",
            "diagnosis_ready": "A concrete observable fault diagnosis is available.",
            "repair_pending": "The diagnosed artifact still requires repair.",
            "retryable_failure": "The previous executor attempt failed after execution and the operation remains unresolved.",
            "retry_plan_ready": "A materially revised retry method is available.",
            "repeated_failure": "The same failed method was repeated without meaningful revision.",
            "candidate_suspect": fault_signal,
            "verification_required": "A visible anomaly requires verification before downstream consumption.",
            "candidate_a_ready": "A first candidate exists but independent confirmation is still required.",
            "independent_check_pending": "The task contract requires an independent result before downstream use.",
            "conflict_observed": "Two candidate outputs materially conflict.",
            "contamination_level_1": "An unresolved faulty candidate has propagated into one downstream artifact.",
            "contamination_level_2": "A contaminated artifact was irreversibly committed.",
            "downstream_ready": f"The required {downstream_artifact} is available.",
            "final_committed": f"The {final_artifact} has been committed.",
        },
        artifact_descriptions={
            "task_input": input_artifact,
            "candidate_artifact": candidate_artifact,
            "corrected_candidate_artifact": f"corrected {candidate_artifact}",
            "faulty_raw_output": "raw executor output that failed its contract",
            "failure_report": "retryable execution failure report",
            "fault_diagnosis": "observable fault diagnosis",
            "revised_method": "materially revised execution method",
            "conflicting_candidate": f"conflicting alternative {candidate_artifact}",
            "verification_record": "optional consistency verification record",
            "contaminated_downstream_artifact": f"contaminated {downstream_artifact}",
            "downstream_artifact": downstream_artifact,
            "committed_final_artifact": final_artifact,
        },
        goal_facts=frozenset({"final_committed"}),
        goal_artifacts=frozenset({"committed_final_artifact"}),
        transitions=tuple(transitions),
        max_steps=9,
        max_cost=9.0,
        difficulty_tags=(
            "controlled_recovery",
            f"domain:{domain}",
            f"fault:{variant}",
            "history_ablation",
            "set_valued_ground_truth",
        ),
        schema_version=V4_SCHEMA_VERSION,
        public_context={
            "workflow_domain": domain,
            "task_contract": {
                "multiple_steps_required": True,
                "commit_only_after_downstream_postcondition": True,
            },
        },
        max_stagnant_repeats=2,
    )
    clean_id = f"v4-{domain}-clean"
    annotation = {
        "scenario_id": scenario_id,
        "pair_id": f"v4-pair:{domain}",
        "variant": variant,
        "clean_scenario_id": clean_id,
        "faults": (
            []
            if variant == "clean"
            else [
                {
                    "fault_id": f"fault-{scenario_id}",
                    "fault_class": variant,
                    "transition_ids": fault_transition_ids,
                    "injection_position": {
                        "explicit_invalid": "early",
                        "retryable_failure": "early",
                        "silent_corruption": "middle",
                        "conflicting_outputs": "late",
                    }.get(variant, "early"),
                    "observable": True,
                    "recoverable": True,
                    "requires_detection_agent": variant
                    in {"silent_corruption", "conflicting_outputs"},
                    "public_signal": fault_signal,
                }
            ]
        ),
        "unsafe_transition_ids": unsafe,
        "inefficient_transition_ids": inefficient,
        "detector_actions": list(DETECTOR_ACTIONS),
        "repair_actions": list(REPAIR_ACTIONS),
        "recovery_actions": list(RECOVERY_ACTIONS),
        "decision_labels": [],
    }
    task_input = {
        "domain": domain,
        "input_contract": input_artifact,
        "candidate_contract": candidate_artifact,
        "downstream_contract": downstream_artifact,
        "final_contract": final_artifact,
    }
    return scenario, fixtures, annotation, task_input


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
        if env.done or env.goal_reached:
            continue
        for action in env.applicable_actions():
            if action == STOP_ACTION:
                continue
            child = env.clone()
            child.step(action)
            fingerprint = child.state.fingerprint()
            if fingerprint not in seen:
                seen.add(fingerprint)
                queue.append(child)
    return states


def _phase(env: RoutingEnvironmentV2) -> str:
    facts = env.state.facts
    if env.goal_reached:
        return "terminal"
    if "contamination_level_1" in facts or "contamination_level_2" in facts:
        return "contaminated"
    if "repair_pending" in facts or "diagnosis_ready" in facts:
        return "repair"
    if "verification_required" in facts or "diagnosis_pending" in facts:
        return "verification"
    if "retryable_failure" in facts or "retry_plan_ready" in facts:
        return "recovery"
    if env.state.steps_used == 0:
        return "initial"
    return "normal"


def _selected_states(states: list[RoutingEnvironmentV2], maximum: int = 18):
    priority = {
        "initial": 0,
        "verification": 1,
        "recovery": 1,
        "repair": 2,
        "contaminated": 3,
        "normal": 4,
        "terminal": 5,
    }
    return sorted(
        states,
        key=lambda env: (
            priority[_phase(env)],
            env.state.steps_used,
            env.state.fingerprint(),
        ),
    )[:maximum]


def _safe_actions(
    env: RoutingEnvironmentV2, annotation: Mapping[str, Any]
) -> tuple[str, ...]:
    if env.goal_reached:
        return (STOP_ACTION,)
    unsafe = set(map(str, annotation.get("unsafe_transition_ids") or ()))
    inefficient = set(
        map(str, annotation.get("inefficient_transition_ids") or ())
    )
    result: list[str] = []
    for action in env.applicable_actions():
        if action == STOP_ACTION:
            continue
        transition = env._transition_for(action)
        if (
            transition is not None
            and transition.transition_id not in unsafe
            and transition.transition_id not in inefficient
        ):
            result.append(action)
    return tuple(result)


def _state_rationales(
    env: RoutingEnvironmentV2,
    annotation: Mapping[str, Any],
    valid: Iterable[str],
) -> dict[str, dict[str, Any]]:
    result = _action_rationales(env)
    valid_set = set(valid)
    unsafe = set(map(str, annotation.get("unsafe_transition_ids") or ()))
    inefficient = set(
        map(str, annotation.get("inefficient_transition_ids") or ())
    )
    for action in env.candidate_actions():
        transition = env._transition_for(action) if action != STOP_ACTION else None
        if transition is not None and transition.transition_id in unsafe:
            result[action] = {
                "status": "invalid",
                "rationale_source": "formal_transition_preconditions",
                "satisfied_preconditions": [
                    "the action is executable in the simulator"
                ],
                "violated_preconditions": [
                    "the action would propagate or repeat an observable unresolved fault"
                ],
            }
        elif transition is not None and transition.transition_id in inefficient:
            result[action] = {
                "status": "applicable_but_inefficient",
                "rationale_source": "formal_transition_contract",
                "satisfied_preconditions": [
                    "the role can execute on the current observable state"
                ],
                "violated_preconditions": [
                    "execution produces no artifact or substantive state progress"
                ],
            }
        elif action in valid_set:
            result[action]["status"] = "valid"
            result[action]["violated_preconditions"] = []
        else:
            result[action]["status"] = "invalid"
    return result


def build_v4_bundle(seed: int = 42):
    del seed
    catalog = agent_catalog_v4()
    scenarios: list[ScenarioSpecV2] = []
    fixtures: list[dict[str, Any]] = []
    annotations: dict[str, dict[str, Any]] = {}
    task_inputs: dict[str, dict[str, Any]] = {}
    provenance: dict[str, ProvenanceRecordV3] = {}
    template_path = PROJECT_DIR / TEMPLATE_RELATIVE_PATH
    source_hash = hashlib.sha256(template_path.read_bytes()).hexdigest()
    for template in _read_templates():
        family_rows = []
        for variant in VARIANTS:
            scenario, outputs, annotation, task_input = _build_scenario(
                template, variant, catalog
            )
            family_rows.append((scenario, outputs, annotation, task_input))

        # Clean and fault counterfactuals expose exactly the same agent pool.
        # The union is derived only from the workflow family, never the hidden
        # fault variant, and then padded with deterministic distractors.
        domain = str(template["domain"])
        active_family = {
            transition.action
            for scenario, _, _, _ in family_rows
            for transition in scenario.transitions
        }
        inactive = [item for item in catalog if item not in active_family]
        selected = list(active_family) + inactive[: max(0, 18 - len(active_family))]
        selected.sort(
            key=lambda value: hashlib.sha256(
                f"v4-{domain}:{value}".encode()
            ).hexdigest()
        )
        for scenario, outputs, annotation, task_input in family_rows:
            scenario_active = {item.action for item in scenario.transitions}
            scenario = replace(
                scenario,
                agent_pool=tuple(selected),
                distractor_ids=tuple(
                    item for item in selected if item not in scenario_active
                ),
            )
            scenarios.append(scenario)
            fixtures.extend(outputs)
            annotations[scenario.scenario_id] = annotation
            task_inputs[scenario.scenario_id] = task_input
            provenance_id = f"prov-{scenario.scenario_id}"
            provenance[provenance_id] = ProvenanceRecordV3(
                provenance_id=provenance_id,
                scenario_id=scenario.scenario_id,
                source_type="project_authored_controlled_template",
                source_dataset="RoutingBench-V4-controlled-templates",
                source_version=f"sha256:{source_hash}",
                source_split=scenario.split,
                source_item_id=str(template["domain"]),
                source_path=TEMPLATE_RELATIVE_PATH.as_posix(),
                source_sha256=source_hash,
                license="project-authored",
                derivation_script=DERIVATION_SCRIPT,
                derivation_version=V4_SCHEMA_VERSION,
                transformation_steps=(
                    "instantiate one clean and four fault variants per domain",
                    "derive set-valued safe actions from executable transition contracts",
                    "store fault and recovery labels only in an evaluation sidecar",
                    "generate deterministic raw and structured executor fixtures",
                ),
                validation_method="template hash, executable transition validation, paired recovery checks, and pending human review",
                human_review_status="pending",
            )
    metadata = {
        "name": "RoutingBench",
        "version": V4_SCHEMA_VERSION,
        "release_stage": "alpha",
        "workflow_count": len(scenarios),
        "domain_count": len({item.family for item in scenarios}),
        "variants": list(VARIANTS),
        "primary_configurations": [
            "naive_state",
            "structured_state",
            "structured_history",
        ],
        "evaluation_tracks": {
            "A": "isolated next-agent decisions",
            "B": "end-to-end router-induced trajectories",
            "C": "checkpointed post-fault recovery with matched clean controls",
        },
        "diagnostic_configuration": "structured_raw_history_development_only",
        "memory_policy": {
            "recent_event_limit": 5,
            "persistent_memory": "full_structured_trajectory",
            "summarizer": "deterministic_only",
            "preserve_failures": True,
            "preserve_evidence_refs": True,
        },
        "preferred_actions": False,
        "action_utilities": False,
        "capability_scores": False,
        "backbone_identity_in_router_input": False,
        "primary_ground_truth": "set_valued_safe_actions",
        "router_input_artifact": "public_inputs.jsonl",
        "private_label_artifact": "private_labels.jsonl",
        "combined_internal_artifact": "decision_states.jsonl",
        "post_fault_public_artifact": "post_fault_inputs.jsonl",
        "post_fault_private_artifacts": [
            "post_fault_runtime.jsonl",
            "post_fault_labels.jsonl",
            "post_fault_pair_manifest.jsonl",
        ],
        "public_private_policy": (
            "routers consume only public_inputs.jsonl; private_labels.jsonl and "
            "decision_states.jsonl are evaluator/audit artifacts"
        ),
        "terminal_stop_budget_policy": "reserved_zero_cost_terminal_decision",
        "human_review_status": "pending",
    }
    bundle = BenchmarkBundleV2(
        metadata=metadata,
        agents=catalog,
        scenarios=tuple(scenarios),
    )
    validate_bundle(bundle)
    return bundle, provenance, fixtures, annotations, task_inputs


def generate_decision_rows_v4(
    bundle: BenchmarkBundleV2,
    provenance: Mapping[str, ProvenanceRecordV3],
    fixtures: Iterable[Mapping[str, Any]],
    annotations: dict[str, dict[str, Any]],
) -> list[dict[str, Any]]:
    fixtures = list(fixtures)
    provenance_by_scenario = {item.scenario_id: item for item in provenance.values()}
    rows: list[dict[str, Any]] = []
    for scenario in bundle.scenarios:
        annotation = annotations[scenario.scenario_id]
        decision_labels: list[dict[str, Any]] = []
        for index, env in enumerate(
            _selected_states(_reachable_states(scenario, bundle.agents))
        ):
            valid = list(_safe_actions(env, annotation))
            if not valid:
                continue
            decision_id = f"{scenario.scenario_id}:state-{index:03d}"
            views = build_memory_views(
                env,
                fixtures,
                include_raw=scenario.split == "development",
            )
            phase = _phase(env)
            row = {
                "decision_id": decision_id,
                "scenario_id": scenario.scenario_id,
                "router_scenario_id": f"v4-{scenario.family}-case",
                "template_id": scenario.template_id,
                "instance_id": scenario.instance_id,
                "split": scenario.split,
                "split_group_id": scenario.split_group_id,
                "counterfactual_group_id": scenario.counterfactual_group_id,
                "provenance_id": provenance_by_scenario[
                    scenario.scenario_id
                ].provenance_id,
                "workflow_family": scenario.family,
                "state_provenance": phase,
                "task": scenario.task,
                "task_context": dict(scenario.public_context),
                "observed_state": views["state"]["observed_state"],
                "execution_memory": views["structured"]["execution_memory"],
                "recent_events": views["structured"]["recent_events"],
                "memory_views": views,
                "budget": {
                    "calls_remaining": env.remaining_steps,
                    "reserved_terminal_stop": True,
                },
                "agent_pool": [
                    bundle.agents[item].public_profile() for item in scenario.agent_pool
                ],
                "candidate_actions": list(env.candidate_actions()),
                "valid_actions": valid,
                "action_rationales": _state_rationales(env, annotation, valid),
                "terminal_valid": env.goal_reached,
                "difficulty_tags": list(scenario.difficulty_tags),
                "schema_version": V4_SCHEMA_VERSION,
                "ground_truth_method": "formal_safe_transition_contracts",
                "human_review_status": "pending",
            }
            rows.append(row)
            applicable = set(env.applicable_actions())
            unsafe_actions = []
            unsafe_ids = set(annotation["unsafe_transition_ids"])
            for action in applicable:
                if action == STOP_ACTION:
                    continue
                transition = env._transition_for(action)
                if transition and transition.transition_id in unsafe_ids:
                    unsafe_actions.append(action)
            decision_labels.append(
                {
                    "decision_id": decision_id,
                    "phase": phase,
                    "fault_active": phase
                    in {"verification", "recovery", "repair", "contaminated"},
                    "unsafe_actions": sorted(unsafe_actions),
                    "detector_actions_available": sorted(
                        set(valid).intersection(annotation["detector_actions"])
                    ),
                    "repair_actions_available": sorted(
                        set(valid).intersection(annotation["repair_actions"])
                    ),
                    "recovery_actions_available": sorted(
                        set(valid).intersection(annotation["recovery_actions"])
                    ),
                }
            )
        annotation["decision_labels"] = decision_labels
    return rows


def _write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")


def _write_jsonl(path: Path, rows: Iterable[Mapping[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(dict(row), ensure_ascii=False) + "\n")


def _public_decision_id(row: Mapping[str, Any]) -> str:
    payload = build_public_observation(row, "structured")
    digest = hashlib.sha256(
        json.dumps(payload, ensure_ascii=False, sort_keys=True).encode("utf-8")
    ).hexdigest()[:20]
    return f"pub-{digest}"


def _quota(
    rows: list[dict[str, Any]],
    scenarios: tuple[ScenarioSpecV2, ...],
    annotations: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    nonterminal = [item for item in rows if not item["terminal_valid"]]
    recovery = [
        item
        for item in rows
        if item["state_provenance"]
        in {"verification", "recovery", "repair", "contaminated"}
    ]
    return {
        "release_stage": "alpha",
        "workflow_count": len(scenarios),
        "decision_state_count": len(rows),
        "domain_counts": dict(sorted(Counter(item.family for item in scenarios).items())),
        "variant_counts": dict(
            sorted(Counter(item["variant"] for item in annotations.values()).items())
        ),
        "split_counts": dict(sorted(Counter(item.split for item in scenarios).items())),
        "recovery_state_count": len(recovery),
        "contaminated_state_count": sum(
            item["state_provenance"] == "contaminated" for item in rows
        ),
        "multi_valid_nonterminal_rate": (
            sum(len(item["valid_actions"]) >= 2 for item in nonterminal)
            / len(nonterminal)
            if nonterminal
            else 0.0
        ),
        "raw_history_split": "development_only",
        "preferred_actions_present": any("preferred_actions" in item for item in rows),
        "action_utilities_present": any("action_utilities" in item for item in rows),
    }


def write_v4_release(output_dir: str | Path, seed: int = 42) -> dict[str, Any]:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    bundle, provenance, fixtures, annotations, task_inputs = build_v4_bundle(seed)
    rows = generate_decision_rows_v4(
        bundle, provenance, fixtures, annotations
    )
    (
        post_fault_inputs,
        post_fault_runtime,
        post_fault_labels,
        post_fault_pairs,
    ) = generate_post_fault_track(
        scenarios=bundle.scenarios,
        agents=bundle.agents,
        fixtures=fixtures,
        annotations=annotations,
        benchmark_version=V4_SCHEMA_VERSION,
    )
    quota = _quota(rows, bundle.scenarios, annotations)
    metadata = {
        **bundle.metadata,
        "decision_state_count": len(rows),
        "post_fault_checkpoint_count": len(post_fault_runtime),
        "post_fault_pair_count": len(post_fault_pairs),
    }
    quota = {
        **quota,
        "post_fault_checkpoint_count": len(post_fault_runtime),
        "post_fault_pair_count": len(post_fault_pairs),
        "post_fault_condition_counts": dict(
            sorted(
                Counter(item["condition"] for item in post_fault_labels).items()
            )
        ),
        "post_fault_split_counts": dict(
            sorted(
                Counter(item["split"] for item in post_fault_runtime).items()
            )
        ),
    }
    _write_json(output_dir / "metadata.json", metadata)
    _write_jsonl(output_dir / "agents.jsonl", (item.to_dict() for item in bundle.agents.values()))
    by_scenario = {item.scenario_id: item for item in provenance.values()}
    scenario_rows = []
    for scenario in bundle.scenarios:
        value = scenario.to_dict()
        value["provenance_id"] = by_scenario[scenario.scenario_id].provenance_id
        scenario_rows.append(value)
    _write_jsonl(output_dir / "scenarios.jsonl", scenario_rows)
    _write_jsonl(output_dir / "provenance.jsonl", (item.to_dict() for item in provenance.values()))
    _write_jsonl(output_dir / "task_inputs.jsonl", (
        {"scenario_id": key, "task_input": value}
        for key, value in sorted(task_inputs.items())
    ))
    _write_jsonl(output_dir / "executor_outputs.jsonl", fixtures)
    _write_jsonl(output_dir / "recovery_annotations.jsonl", (
        value for _, value in sorted(annotations.items())
    ))
    _write_jsonl(output_dir / "decision_states.jsonl", rows)
    label_index = {
        str(item["decision_id"]): item
        for annotation in annotations.values()
        for item in annotation.get("decision_labels") or ()
    }
    _write_jsonl(
        output_dir / "public_inputs.jsonl",
        (
            {
                "decision_id": _public_decision_id(row),
                "split": row["split"],
                "workflow_family": row["workflow_family"],
                "observations": {
                    view_name: build_public_observation(row, view_name)
                    for view_name in row["memory_views"]
                },
            }
            for row in rows
        ),
    )
    _write_jsonl(
        output_dir / "private_labels.jsonl",
        (
            {
                "decision_id": row["decision_id"],
                "public_decision_id": _public_decision_id(row),
                "internal_scenario_id": row["scenario_id"],
                "valid_actions": row["valid_actions"],
                "terminal_valid": row["terminal_valid"],
                "phase": label_index.get(row["decision_id"], {}).get(
                    "phase", row["state_provenance"]
                ),
                "unsafe_actions": label_index.get(
                    row["decision_id"], {}
                ).get("unsafe_actions", []),
                "fault_variant": annotations[row["scenario_id"]]["variant"],
                "action_rationales": row["action_rationales"],
            }
            for row in rows
        ),
    )
    _write_jsonl(output_dir / "post_fault_inputs.jsonl", post_fault_inputs)
    _write_jsonl(output_dir / "post_fault_runtime.jsonl", post_fault_runtime)
    _write_jsonl(output_dir / "post_fault_labels.jsonl", post_fault_labels)
    _write_jsonl(
        output_dir / "post_fault_pair_manifest.jsonl", post_fault_pairs
    )
    _write_json(output_dir / "quota_report.json", quota)
    _write_json(
        output_dir / "split_manifest.json",
        {
            split: [item.scenario_id for item in bundle.scenarios if item.split == split]
            for split in ("development", "validation", "test")
        },
    )
    _write_jsonl(
        output_dir / "human_review_queue.jsonl",
        (
            {
                "scenario_id": item.scenario_id,
                "review_status": "pending",
                "checklist": [
                    "observable state is sufficient to justify every safe action",
                    "unsafe downstream actions are executable but correctly excluded from valid_actions",
                    "fault signal is visible without exposing the private fault label",
                    "recovery action removes or isolates the faulty artifact",
                    "clean and fault variants preserve the same task contract",
                ],
            }
            for item in bundle.scenarios
        ),
    )
    _write_jsonl(
        output_dir / "post_fault_human_review_queue.jsonl",
        (
            {
                "pair_id": item["pair_id"],
                "fault_checkpoint_id": item["fault_checkpoint_id"],
                "control_checkpoint_id": item["control_checkpoint_id"],
                "review_status": "pending",
                "checklist": [
                    "fault and control checkpoints expose the same task contract and agent pool",
                    "the observable history justifies different recovery behavior without exposing the private fault class",
                    "remaining step and cost budgets are equal across the pair",
                    "valid actions admit at least one realistic continuation to the goal",
                    "unsafe actions represent propagation of the observed fault rather than arbitrary rejection",
                ],
            }
            for item in post_fault_pairs
        ),
    )
    generated = [
        output_dir / name
        for name in (
            "metadata.json",
            "agents.jsonl",
            "scenarios.jsonl",
            "provenance.jsonl",
            "task_inputs.jsonl",
            "executor_outputs.jsonl",
            "recovery_annotations.jsonl",
            "decision_states.jsonl",
            "public_inputs.jsonl",
            "private_labels.jsonl",
            "post_fault_inputs.jsonl",
            "post_fault_runtime.jsonl",
            "post_fault_labels.jsonl",
            "post_fault_pair_manifest.jsonl",
            "quota_report.json",
            "split_manifest.json",
            "human_review_queue.jsonl",
            "post_fault_human_review_queue.jsonl",
        )
    ]
    _write_json(output_dir / "checksums.json", _checksums(generated))
    specs = output_dir / "specs"
    specs.mkdir(parents=True, exist_ok=True)
    for name, schema in v4_schema_definitions().items():
        _write_json(specs / name, schema)
    return {"metadata": metadata, "quota_report": quota}
