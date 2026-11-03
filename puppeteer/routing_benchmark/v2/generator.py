from __future__ import annotations

import hashlib
import json
import random
from collections import Counter, deque
from pathlib import Path
from typing import Any, Iterable

from .environment import RoutingEnvironmentV2
from .loader import BenchmarkBundleV2, validate_bundle
from .schemas import (
    FAILURE_STATUSES,
    ROUTING_REJECTION_STATUSES,
    AgentProfileV2,
    ScenarioSpecV2,
    TransitionSpecV2,
    STOP_ACTION,
)
from .schema_definitions import all_schemas


SCHEMA_VERSION = "2.0-alpha"


FACTS = {
    "dependencies_unmapped": "The linked requirements have not yet been ordered into an execution plan.",
    "plan_ready": "An ordered dependency plan is available.",
    "first_source_pending": "The first planned source has not been examined.",
    "first_evidence_ready": "Evidence for the first dependency is available.",
    "relation_unresolved": "The relation between the intermediate entity and the target is unresolved.",
    "relation_resolved": "The intermediate relation has been resolved.",
    "second_source_pending": "A later evidence dependency remains unresolved.",
    "second_evidence_ready": "Evidence for the later dependency is available.",
    "composition_pending": "Available evidence has not yet been combined into a candidate.",
    "candidate_ready": "A candidate response has been produced.",
    "candidate_supported": "The candidate is supported by the currently available evidence.",
    "candidate_verified": "The candidate has passed an explicit evidence check.",
    "conflict_present": "The candidate and evidence contain a material conflict.",
    "conflict_resolved": "The material evidence conflict has been resolved.",
    "finalization_pending": "A supported candidate still needs to be committed as the final artifact.",
    "final_committed": "A final artifact has been committed.",
    "registry_pending": "A structured registry lookup remains unresolved.",
    "web_pending": "A current external-source lookup remains unresolved.",
    "registry_evidence": "The structured registry evidence is available.",
    "web_evidence": "The current external-source evidence is available.",
    "temporal_pending": "A date-sensitive evidence dependency remains unresolved.",
    "temporal_evidence": "Date-sensitive evidence is available.",
    "audio_pending": "An attached audio artifact has not been inspected.",
    "spreadsheet_pending": "An attached spreadsheet has not been inspected.",
    "archive_pending": "An attached structured file bundle has not been inspected.",
    "script_pending": "A supplied script has not been executed in the deterministic sandbox.",
    "media_content_ready": "Task-relevant media content has been extracted.",
    "file_structure_ready": "The file structure and usable table are available.",
    "table_analysis_pending": "A calculation over the parsed table remains unresolved.",
    "table_result_ready": "The requested table calculation is available.",
    "execution_result_ready": "The deterministic script execution result is available.",
    "format_check_pending": "The produced artifact still requires structural validation.",
    "format_validated": "The artifact satisfies the requested structural format.",
    "primary_query_ready": "An initial external query is ready for execution.",
    "primary_corpus_method_ready": "An initial corpus extraction method is ready.",
    "search_blocked": "The initial external query failed and should not be repeated unchanged.",
    "corpus_method_stalled": "The initial corpus method made no useful progress.",
    "revised_query_ready": "A materially revised external query is ready.",
    "revised_corpus_method_ready": "A materially revised corpus method is ready.",
    "retrieved_evidence": "The required source-grounded evidence is available.",
}


ARTIFACTS = {
    "audio_attachment": "an audio attachment",
    "spreadsheet_attachment": "a spreadsheet attachment",
    "archive_attachment": "a structured file bundle",
    "script_bundle": "a deterministic script bundle",
    "dependency_plan": "an ordered dependency plan",
    "evidence_packet": "a source-grounded evidence packet",
    "candidate_artifact": "a candidate answer artifact",
    "final_artifact": "a committed final artifact",
    "media_transcript": "a task-focused media transcript",
    "parsed_table": "a parsed table",
    "calculation_trace": "a deterministic calculation trace",
    "execution_trace": "a deterministic execution trace",
}


def _profile(
    agent_id: str,
    role_name: str,
    goal: str,
    use_when: list[str],
    avoid_when: list[str],
    inputs: list[str],
    outputs: list[str],
    family: str,
    cost: float = 1.0,
    cost_class: str = "standard",
    distractor_only: bool = False,
) -> AgentProfileV2:
    return AgentProfileV2(
        agent_id=agent_id,
        role_name=role_name,
        role_goal=goal,
        use_when=tuple(use_when),
        avoid_when=tuple(avoid_when),
        supported_inputs=tuple(inputs),
        expected_outputs=tuple(outputs),
        cost_class=cost_class,
        action_cost=cost,
        semantic_family=family,
        distractor_only=distractor_only,
    )


def agent_catalog() -> dict[str, AgentProfileV2]:
    rows = [
        _profile("agent_01", "Workflow Planner", "Order unresolved dependencies into an executable plan.", ["dependencies have no execution order"], ["a usable plan already exists"], ["task specification"], ["dependency plan"], "planning"),
        _profile("agent_02", "Corpus Evidence Analyst", "Resolve a concrete evidence dependency from supplied text.", ["a supplied-corpus dependency is explicit"], ["the unresolved source is live web, media, or a registry"], ["text corpus", "evidence target"], ["grounded evidence packet"], "retrieval"),
        _profile("agent_03", "External Source Analyst", "Resolve a concrete current-information dependency from external sources.", ["current external evidence or a revised query is required"], ["a failed query has not been revised", "the supplied corpus is sufficient"], ["web query", "revised query"], ["external evidence packet"], "retrieval", 1.5, "elevated"),
        _profile("agent_04", "Registry Analyst", "Resolve structured records and registry lookups.", ["a structured-record lookup is pending"], ["the source is unstructured text or media"], ["record key", "registry"], ["registry evidence"], "retrieval"),
        _profile("agent_05", "Relational Inference Specialist", "Resolve an intermediate relation from available evidence.", ["an intermediate entity is known but its target relation is unresolved"], ["raw evidence is still absent"], ["evidence packet", "relation target"], ["resolved relation"], "reasoning"),
        _profile("agent_06", "Evidence Composer", "Combine complete evidence branches into a candidate artifact.", ["all required evidence branches are available"], ["a dependency or conflict is unresolved"], ["evidence packets"], ["candidate artifact"], "composition"),
        _profile("agent_07", "Conflict Auditor", "Check a candidate against evidence and resolve material conflicts.", ["a candidate conflicts with evidence", "explicit verification is useful"], ["there is no candidate", "only final commitment remains"], ["candidate", "evidence"], ["verified candidate", "conflict resolution"], "verification"),
        _profile("agent_08", "Final Artifact Editor", "Commit a supported candidate as the final deliverable.", ["a supported candidate is ready"], ["evidence, computation, or conflict resolution is missing"], ["supported candidate"], ["committed final artifact"], "integration"),
        _profile("agent_09", "Media Inspector", "Extract task-relevant information from audio or images.", ["a media attachment is pending inspection"], ["media content is already available"], ["audio", "image"], ["media transcript", "structured observation"], "tool" ,1.5,"elevated"),
        _profile("agent_10", "File Structure Inspector", "Open a supplied file and expose its usable structure.", ["a file or archive has not been parsed"], ["a parsed structure is already available"], ["file", "archive", "spreadsheet"], ["parsed file structure"], "tool"),
        _profile("agent_11", "Table Calculation Specialist", "Compute a requested value from an already parsed table.", ["a parsed table and calculation target are available"], ["the file has not been parsed"], ["parsed table"], ["table result", "calculation trace"], "tool"),
        _profile("agent_12", "Recovery Strategist", "Replace a failed method with a materially revised attempt.", ["the previous action failed or made no progress"], ["no failure occurred", "an unchanged retry is proposed"], ["failure status", "prior attempt"], ["revised method"], "recovery"),
        _profile("agent_13", "Budget Controller", "Choose a feasible execution plan under a strict resource budget.", ["multiple feasible routes differ in resource use"], ["the route is already determined"], ["budget", "route alternatives"], ["budget-feasible route"], "planning"),
        _profile(
            "agent_14",
            "General Problem Solver",
            "Resolve or synthesize evidence-backed intermediate steps when the required inputs are already available and specialist boundaries overlap.",
            [
                "a dependency plan identifies a concrete evidence target",
                "an evidence packet is available for intermediate reasoning",
                "completed evidence branches need to be combined",
            ],
            [
                "the task still requires initial planning",
                "required evidence inputs are not yet available",
                "the state requires a specialist-only tool, verification, or final commitment",
            ],
            [
                "dependency plan with an evidence target",
                "grounded evidence packet",
                "completed evidence branches",
            ],
            [
                "grounded evidence packet",
                "resolved intermediate relation",
                "synthesized candidate artifact",
            ],
            "general",
            2.0,
            "high",
        ),
        _profile("agent_15", "Quantitative Specialist", "Solve formal mathematical derivations.", ["a mathematical derivation is required"], ["the state requires retrieval, media inspection, or finalization"], ["equations", "numeric data"], ["quantitative derivation"], "domain", distractor_only=True),
        _profile("agent_16", "Legal Analyst", "Interpret legal rules and legal texts.", ["legal doctrine must be interpreted"], ["the state has no legal dependency"], ["legal text"], ["legal analysis"], "domain", distractor_only=True),
        _profile("agent_17", "Creative Writer", "Produce stylistically constrained creative prose.", ["the requested artifact is creative prose"], ["the workflow is evidence-grounded or computational"], ["style brief"], ["creative text"], "domain", distractor_only=True),
        _profile("agent_18", "Source Cross-checker", "Cross-check a candidate using an independent evidence source.", ["independent corroboration is warranted"], ["no candidate exists", "budget only permits direct finalization"], ["candidate", "secondary source"], ["cross-checked candidate"], "verification", 1.2, "elevated"),
        _profile("agent_19", "Format Validator", "Validate that a structured artifact follows its required schema.", ["content exists but its structure is unchecked"], ["content generation is incomplete"], ["structured artifact", "schema"], ["validated structure"], "verification"),
        _profile("agent_20", "Alternative Hypothesis Analyst", "Develop an independent explanation when evidence supports competing hypotheses.", ["multiple plausible hypotheses remain"], ["the workflow has one explicit unresolved operation"], ["conflicting hypotheses"], ["alternative explanation"], "reasoning", 1.5, "elevated", True),
        _profile("agent_21", "Deterministic Code Executor", "Execute supplied deterministic code in a sandbox and return its trace.", ["a supplied script must be executed"], ["no executable artifact exists"], ["script bundle"], ["execution result", "execution trace"], "tool", 1.5, "elevated"),
        _profile("agent_22", "Temporal Evidence Analyst", "Resolve date-sensitive evidence dependencies.", ["a fact depends on a date or time interval"], ["the dependency is not temporal"], ["dated records"], ["temporal evidence"], "retrieval"),
        _profile("agent_23", "Geographic Specialist", "Resolve geographic relations and locations.", ["a geographic dependency is explicit"], ["the state has no geographic relation"], ["location data"], ["geographic result"], "domain", distractor_only=True),
        _profile("agent_24", "Concise Artifact Finalizer", "Edit and commit a complete supported artifact without changing its meaning.", ["a supported candidate is complete and needs concise final presentation"], ["dependencies, evidence, or conflict resolution remain unresolved"], ["supported candidate"], ["concise committed artifact"], "integration"),
    ]
    return {item.agent_id: item for item in rows}


def _tr(
    transition_id: str,
    action: str,
    requires: Iterable[str] = (),
    *,
    requires_any: Iterable[str] = (),
    forbids: Iterable[str] = (),
    artifacts: Iterable[str] = (),
    last_statuses: Iterable[str] = (),
    add: Iterable[str] = (),
    remove: Iterable[str] = (),
    add_artifacts: Iterable[str] = (),
    remove_artifacts: Iterable[str] = (),
    status: str = "SUCCESS",
    effect: str = "State updated.",
    cost: float | None = None,
) -> TransitionSpecV2:
    return TransitionSpecV2(
        transition_id=transition_id,
        action=action,
        requires_all=frozenset(requires),
        requires_any=frozenset(requires_any),
        forbids=frozenset(forbids),
        requires_artifacts=frozenset(artifacts),
        requires_last_statuses=frozenset(last_statuses),
        add_facts=frozenset(add),
        remove_facts=frozenset(remove),
        add_artifacts=frozenset(add_artifacts),
        remove_artifacts=frozenset(remove_artifacts),
        status=status,
        public_effect=effect,
        cost_override=cost,
    )


def _scenario(
    family: str,
    number: int,
    task: str,
    pool: list[str],
    initial_facts: list[str],
    transitions: list[TransitionSpecV2],
    *,
    initial_artifacts: list[str] | None = None,
    max_steps: int = 9,
    max_cost: float = 14.0,
    tags: list[str] | None = None,
    counterfactual_group: str | None = None,
    seed: int = 42,
) -> ScenarioSpecV2:
    family_code = family.replace("_", "-")
    scenario_id = f"v2a-{family_code}-{number:02d}"
    shuffled_pool = list(dict.fromkeys(pool))
    random.Random(f"{seed}:{scenario_id}").shuffle(shuffled_pool)
    catalog = agent_catalog()
    distractors = tuple(
        item for item in shuffled_pool if catalog[item].distractor_only
    )
    return ScenarioSpecV2(
        scenario_id=scenario_id,
        template_id=f"template-{family_code}",
        instance_id=f"instance-{number:02d}",
        split="validation",
        split_group_id=f"alpha-{family_code}",
        counterfactual_group_id=counterfactual_group,
        family=family,
        task=task,
        agent_pool=tuple(shuffled_pool),
        distractor_ids=distractors,
        initial_facts=frozenset(initial_facts),
        initial_artifacts=frozenset(initial_artifacts or ()),
        fact_descriptions=dict(FACTS),
        artifact_descriptions=dict(ARTIFACTS),
        goal_facts=frozenset({"final_committed"}),
        goal_artifacts=frozenset({"final_artifact"}),
        transitions=tuple(transitions),
        max_steps=max_steps,
        max_cost=max_cost,
        difficulty_tags=tuple(tags or ()),
        schema_version=SCHEMA_VERSION,
    )


def _final_transition(prefix: str) -> TransitionSpecV2:
    return _tr(
        f"{prefix}-final",
        "agent_08",
        ["candidate_ready", "candidate_supported", "finalization_pending"],
        add=["final_committed"],
        remove=["finalization_pending"],
        add_artifacts=["final_artifact"],
        effect="The supported candidate was committed as the final artifact.",
    )


def alpha_scenarios(seed: int = 42) -> tuple[ScenarioSpecV2, ...]:
    scenarios: list[ScenarioSpecV2] = []

    for number, second_agent in ((1, "agent_02"), (2, "agent_03")):
        p = f"seq{number}"
        transitions = [
            _tr(f"{p}-plan", "agent_01", ["dependencies_unmapped"], add=["plan_ready", "first_source_pending"], remove=["dependencies_unmapped"], add_artifacts=["dependency_plan"], effect="An ordered dependency plan and first evidence target are available."),
            _tr(f"{p}-first-specialist", "agent_02", ["plan_ready", "first_source_pending"], add=["first_evidence_ready", "relation_unresolved"], remove=["first_source_pending"], add_artifacts=["evidence_packet"], effect="The first evidence dependency was resolved."),
            _tr(f"{p}-first-general", "agent_14", ["plan_ready", "first_source_pending"], add=["first_evidence_ready", "relation_unresolved"], remove=["first_source_pending"], add_artifacts=["evidence_packet"], effect="A general solver resolved the first evidence dependency at higher cost."),
            _tr(f"{p}-relation-specialist", "agent_05", ["first_evidence_ready", "relation_unresolved"], add=["relation_resolved", "second_source_pending"], remove=["relation_unresolved"], effect="The intermediate relation was resolved and the later evidence target became explicit."),
            _tr(f"{p}-relation-general", "agent_14", ["first_evidence_ready", "relation_unresolved"], add=["relation_resolved", "second_source_pending"], remove=["relation_unresolved"], effect="A general solver resolved the intermediate relation at higher cost."),
            _tr(f"{p}-second", second_agent, ["relation_resolved", "second_source_pending"], add=["second_evidence_ready", "composition_pending"], remove=["second_source_pending"], effect="The later evidence dependency was resolved."),
            _tr(f"{p}-compose-specialist", "agent_06", ["first_evidence_ready", "second_evidence_ready", "composition_pending"], add=["candidate_ready", "candidate_supported", "finalization_pending"], remove=["composition_pending"], add_artifacts=["candidate_artifact"], effect="The evidence branches were combined into a supported candidate."),
            _tr(f"{p}-compose-general", "agent_14", ["first_evidence_ready", "second_evidence_ready", "composition_pending"], add=["candidate_ready", "candidate_supported", "finalization_pending"], remove=["composition_pending"], add_artifacts=["candidate_artifact"], effect="A general solver combined the evidence at higher cost."),
            _tr(f"{p}-optional-verify", "agent_07", ["candidate_ready", "candidate_supported"], forbids=["candidate_verified"], add=["candidate_verified"], effect="The supported candidate was additionally verified."),
            _final_transition(p),
        ]
        scenarios.append(
            _scenario(
                "sequential_dependency",
                number,
                "Identify the governing body associated with an activity discovered through two dependent evidence relations, then return a concise final artifact.",
                ["agent_01", "agent_02", "agent_03", "agent_05", "agent_06", "agent_07", "agent_08", "agent_14", "agent_15", "agent_16"],
                ["dependencies_unmapped"],
                transitions,
                max_steps=10,
                tags=["multi_step", "overlapping_generalist", "optional_verification"],
                seed=seed,
            )
        )

    for number, pair in ((1, ("agent_04", "agent_03")), (2, ("agent_02", "agent_22"))):
        p = f"join{number}"
        left_action, right_action = pair
        left_pending = "registry_pending" if number == 1 else "first_source_pending"
        left_ready = "registry_evidence" if number == 1 else "first_evidence_ready"
        right_pending = "web_pending" if number == 1 else "temporal_pending"
        right_ready = "web_evidence" if number == 1 else "temporal_evidence"
        transitions = [
            _tr(f"{p}-left", left_action, [left_pending], add=[left_ready], remove=[left_pending], add_artifacts=["evidence_packet"], effect="One independent evidence branch was resolved."),
            _tr(f"{p}-right", right_action, [right_pending], add=[right_ready], remove=[right_pending], add_artifacts=["evidence_packet"], effect="The other independent evidence branch was resolved."),
            _tr(f"{p}-compose", "agent_06", [left_ready, right_ready], forbids=["candidate_ready"], add=["candidate_ready", "candidate_supported", "finalization_pending"], add_artifacts=["candidate_artifact"], effect="Both independent evidence branches were joined into a candidate."),
            _tr(f"{p}-compose-general", "agent_14", [left_ready, right_ready], forbids=["candidate_ready"], add=["candidate_ready", "candidate_supported", "finalization_pending"], add_artifacts=["candidate_artifact"], effect="A general solver joined the evidence branches at higher cost."),
            _tr(f"{p}-optional-verify", "agent_07", ["candidate_ready", "candidate_supported"], forbids=["candidate_verified"], add=["candidate_verified"], effect="The joined candidate was additionally verified."),
            _final_transition(p),
        ]
        scenarios.append(
            _scenario(
                "parallel_join",
                number,
                "Combine two independent sources into one supported operational report.",
                [left_action, right_action, "agent_01", "agent_06", "agent_07", "agent_08", "agent_14", "agent_17", "agent_23"],
                [left_pending, right_pending],
                transitions,
                max_steps=7,
                tags=["order_invariant", "set_valued", "overlapping_generalist"],
                seed=seed,
            )
        )

    conditional_common_pool = ["agent_09", "agent_10", "agent_11", "agent_06", "agent_08", "agent_15", "agent_17", "agent_21"]
    scenarios.append(
        _scenario(
            "conditional_branch",
            1,
            "Inspect the attached operational artifact, extract its task-relevant content, and commit the requested result.",
            conditional_common_pool,
            ["audio_pending"],
            [
                _tr("conditional1-media", "agent_09", ["audio_pending"], artifacts=["audio_attachment"], add=["media_content_ready", "composition_pending"], remove=["audio_pending"], add_artifacts=["media_transcript"], effect="Task-relevant content was extracted from the media artifact."),
                _tr("conditional1-compose", "agent_06", ["media_content_ready", "composition_pending"], add=["candidate_ready", "candidate_supported", "finalization_pending"], remove=["composition_pending"], add_artifacts=["candidate_artifact"], effect="Extracted media content was converted into a supported candidate."),
                _final_transition("conditional1"),
            ],
            initial_artifacts=["audio_attachment"],
            max_steps=5,
            tags=["counterfactual", "artifact_routing"],
            counterfactual_group="cf-artifact-route-01",
            seed=seed,
        )
    )
    scenarios.append(
        _scenario(
            "conditional_branch",
            2,
            "Inspect the attached operational artifact, extract its task-relevant content, and commit the requested result.",
            conditional_common_pool,
            ["spreadsheet_pending"],
            [
                _tr("conditional2-file", "agent_10", ["spreadsheet_pending"], artifacts=["spreadsheet_attachment"], add=["file_structure_ready", "table_analysis_pending"], remove=["spreadsheet_pending"], add_artifacts=["parsed_table"], effect="The spreadsheet structure and usable table were exposed."),
                _tr("conditional2-table", "agent_11", ["file_structure_ready", "table_analysis_pending"], artifacts=["parsed_table"], add=["table_result_ready", "candidate_ready", "candidate_supported", "finalization_pending"], remove=["table_analysis_pending"], add_artifacts=["calculation_trace", "candidate_artifact"], effect="The requested table calculation produced a supported candidate."),
                _final_transition("conditional2"),
            ],
            initial_artifacts=["spreadsheet_attachment"],
            max_steps=5,
            tags=["counterfactual", "artifact_routing"],
            counterfactual_group="cf-artifact-route-01",
            seed=seed,
        )
    )

    scenarios.append(
        _scenario(
            "conflict_resolution",
            1,
            "Finalize the current evidence-grounded response after handling any unresolved evidence condition.",
            ["agent_02", "agent_06", "agent_07", "agent_08", "agent_18", "agent_20", "agent_24"],
            ["candidate_ready", "conflict_present", "finalization_pending"],
            [
                _tr("conflict1-audit", "agent_07", ["candidate_ready", "conflict_present"], add=["conflict_resolved", "candidate_verified", "candidate_supported"], remove=["conflict_present"], effect="The candidate-evidence conflict was resolved."),
                _tr("conflict1-crosscheck", "agent_18", ["candidate_ready", "conflict_present"], add=["conflict_resolved", "candidate_verified", "candidate_supported"], remove=["conflict_present"], effect="Independent cross-checking resolved the conflict at higher cost."),
                _final_transition("conflict1"),
            ],
            initial_artifacts=["candidate_artifact", "evidence_packet"],
            max_steps=4,
            tags=["counterfactual", "overlapping_verifiers"],
            counterfactual_group="cf-conflict-01",
            seed=seed,
        )
    )
    scenarios.append(
        _scenario(
            "conflict_resolution",
            2,
            "Finalize the current evidence-grounded response after handling any unresolved evidence condition.",
            ["agent_02", "agent_06", "agent_07", "agent_08", "agent_18", "agent_20", "agent_24"],
            ["candidate_ready", "candidate_supported", "finalization_pending"],
            [
                _tr("conflict2-optional-audit", "agent_07", ["candidate_ready", "candidate_supported"], forbids=["candidate_verified"], add=["candidate_verified"], effect="The already supported candidate was additionally verified."),
                _final_transition("conflict2"),
            ],
            initial_artifacts=["candidate_artifact", "evidence_packet"],
            max_steps=4,
            tags=["counterfactual", "optional_verification"],
            counterfactual_group="cf-conflict-01",
            seed=seed,
        )
    )

    scenarios.append(
        _scenario(
            "tool_routing",
            1,
            "Inspect the supplied structured bundle, compute its requested aggregate, validate the result format, and commit it.",
            ["agent_09", "agent_10", "agent_11", "agent_19", "agent_08", "agent_15", "agent_17", "agent_21"],
            ["archive_pending"],
            [
                _tr("tool1-file", "agent_10", ["archive_pending"], artifacts=["archive_attachment"], add=["file_structure_ready", "table_analysis_pending"], remove=["archive_pending"], add_artifacts=["parsed_table"], effect="The structured bundle was parsed into a usable table."),
                _tr("tool1-table", "agent_11", ["file_structure_ready", "table_analysis_pending"], add=["table_result_ready", "candidate_ready", "candidate_supported", "format_check_pending", "finalization_pending"], remove=["table_analysis_pending"], add_artifacts=["calculation_trace", "candidate_artifact"], effect="The requested aggregate and its calculation trace were produced."),
                _tr("tool1-format", "agent_19", ["candidate_ready", "format_check_pending"], add=["format_validated"], remove=["format_check_pending"], effect="The candidate structure passed format validation."),
                _tr("tool1-final", "agent_08", ["candidate_ready", "candidate_supported", "format_validated", "finalization_pending"], add=["final_committed"], remove=["finalization_pending"], add_artifacts=["final_artifact"], effect="The validated aggregate was committed."),
            ],
            initial_artifacts=["archive_attachment"],
            max_steps=6,
            tags=["multi_tool", "format_contract"],
            seed=seed,
        )
    )
    scenarios.append(
        _scenario(
            "tool_routing",
            2,
            "Execute the supplied deterministic program, validate its structured output, and commit the requested result.",
            ["agent_09", "agent_10", "agent_11", "agent_19", "agent_08", "agent_15", "agent_17", "agent_21"],
            ["script_pending"],
            [
                _tr("tool2-execute", "agent_21", ["script_pending"], artifacts=["script_bundle"], add=["execution_result_ready", "candidate_ready", "candidate_supported", "format_check_pending", "finalization_pending"], remove=["script_pending"], add_artifacts=["execution_trace", "candidate_artifact"], effect="The deterministic program produced a result and execution trace."),
                _tr("tool2-format", "agent_19", ["candidate_ready", "format_check_pending"], add=["format_validated"], remove=["format_check_pending"], effect="The program output passed structural validation."),
                _tr("tool2-final", "agent_08", ["candidate_ready", "candidate_supported", "format_validated", "finalization_pending"], add=["final_committed"], remove=["finalization_pending"], add_artifacts=["final_artifact"], effect="The validated program result was committed."),
            ],
            initial_artifacts=["script_bundle"],
            max_steps=5,
            tags=["multi_tool", "execution_contract"],
            seed=seed,
        )
    )

    scenarios.append(
        _scenario(
            "failure_recovery",
            1,
            "Obtain current external evidence; revise the method after a failed attempt; then produce and commit the result.",
            ["agent_01", "agent_03", "agent_06", "agent_07", "agent_08", "agent_12", "agent_14", "agent_17"],
            ["primary_query_ready"],
            [
                _tr("recovery1-fail", "agent_03", ["primary_query_ready"], add=["search_blocked"], remove=["primary_query_ready"], status="RETRYABLE_FAILURE", effect="The initial query failed and cannot be repeated unchanged."),
                _tr("recovery1-plan", "agent_12", ["search_blocked"], last_statuses=["RETRYABLE_FAILURE"], add=["revised_query_ready"], remove=["search_blocked"], effect="A materially revised external query is ready."),
                _tr("recovery1-retry", "agent_03", ["revised_query_ready"], add=["retrieved_evidence", "composition_pending"], remove=["revised_query_ready"], add_artifacts=["evidence_packet"], effect="The revised query produced usable external evidence."),
                _tr("recovery1-compose", "agent_06", ["retrieved_evidence", "composition_pending"], add=["candidate_ready", "candidate_supported", "finalization_pending"], remove=["composition_pending"], add_artifacts=["candidate_artifact"], effect="Recovered evidence was converted into a supported candidate."),
                _final_transition("recovery1"),
            ],
            max_steps=7,
            tags=["post_failure", "retry_after_state_change"],
            seed=seed,
        )
    )
    scenarios.append(
        _scenario(
            "failure_recovery",
            2,
            "Extract the required evidence from a supplied corpus; replace a stalled method before retrying; then commit the result.",
            ["agent_01", "agent_02", "agent_06", "agent_07", "agent_08", "agent_12", "agent_14", "agent_16"],
            ["primary_corpus_method_ready"],
            [
                _tr("recovery2-stall", "agent_02", ["primary_corpus_method_ready"], add=["corpus_method_stalled"], remove=["primary_corpus_method_ready"], status="NO_PROGRESS", effect="The initial corpus method produced no usable evidence."),
                _tr("recovery2-plan", "agent_12", ["corpus_method_stalled"], last_statuses=["NO_PROGRESS"], add=["revised_corpus_method_ready"], remove=["corpus_method_stalled"], effect="A materially revised corpus method is ready."),
                _tr("recovery2-retry", "agent_02", ["revised_corpus_method_ready"], add=["retrieved_evidence", "composition_pending"], remove=["revised_corpus_method_ready"], add_artifacts=["evidence_packet"], effect="The revised method produced usable corpus evidence."),
                _tr("recovery2-compose", "agent_06", ["retrieved_evidence", "composition_pending"], add=["candidate_ready", "candidate_supported", "finalization_pending"], remove=["composition_pending"], add_artifacts=["candidate_artifact"], effect="Recovered evidence was converted into a supported candidate."),
                _final_transition("recovery2"),
            ],
            max_steps=7,
            tags=["post_no_progress", "retry_after_state_change"],
            seed=seed,
        )
    )

    terminal_pool = ["agent_01", "agent_02", "agent_06", "agent_07", "agent_08", "agent_18", "agent_17", "agent_24"]
    scenarios.append(
        _scenario(
            "terminal_control",
            1,
            "Return the supported answer artifact when the workflow state is complete enough to do so.",
            terminal_pool,
            ["candidate_ready", "candidate_supported", "finalization_pending"],
            [
                _tr("terminal1-verify", "agent_07", ["candidate_ready", "candidate_supported"], forbids=["candidate_verified"], add=["candidate_verified"], effect="The supported candidate was additionally verified."),
                _final_transition("terminal1"),
            ],
            initial_artifacts=["candidate_artifact", "evidence_packet"],
            max_steps=4,
            tags=["counterfactual", "optional_verification"],
            counterfactual_group="cf-terminal-01",
            seed=seed,
        )
    )
    scenarios.append(
        _scenario(
            "terminal_control",
            2,
            "Return the supported answer artifact when the workflow state is complete enough to do so.",
            terminal_pool,
            ["candidate_ready", "candidate_supported", "final_committed"],
            [],
            initial_artifacts=["candidate_artifact", "evidence_packet", "final_artifact"],
            max_steps=3,
            tags=["counterfactual", "stop_ready"],
            counterfactual_group="cf-terminal-01",
            seed=seed,
        )
    )

    for number, max_steps in ((1, 4), (2, 2)):
        p = f"budget{number}"
        scenarios.append(
            _scenario(
                "budget_sensitive",
                number,
                "Commit the supported candidate while respecting the remaining execution budget.",
                ["agent_07", "agent_08", "agent_13", "agent_14", "agent_18", "agent_15", "agent_17", "agent_24"],
                ["candidate_ready", "candidate_supported", "finalization_pending"],
                [
                    _tr(f"{p}-verify", "agent_07", ["candidate_ready", "candidate_supported"], forbids=["candidate_verified"], add=["candidate_verified"], effect="The supported candidate was additionally verified."),
                    _final_transition(p),
                ],
                initial_artifacts=["candidate_artifact", "evidence_packet"],
                max_steps=max_steps,
                max_cost=4.0,
                tags=["counterfactual", "budget_sensitive", "valid_vs_efficient"],
                counterfactual_group="cf-budget-01",
                seed=seed,
            )
        )

    return tuple(scenarios)


def build_alpha_bundle(seed: int = 42) -> BenchmarkBundleV2:
    bundle = BenchmarkBundleV2(
        metadata={
            "name": "RoutingBench",
            "version": SCHEMA_VERSION,
            "release_stage": "alpha",
            "seed": seed,
            "workflow_count": 16,
            "family_count": 8,
            "capability_scores": False,
            "backbone_identity": False,
            "live_llm_actors": False,
            "ground_truth": ["valid_actions", "efficient_actions"],
        },
        agents=agent_catalog(),
        scenarios=alpha_scenarios(seed),
    )
    validate_bundle(bundle)
    return bundle


def _state_provenance(env: RoutingEnvironmentV2) -> str:
    if env.state.steps_used == 0:
        return "initial"
    if env.goal_reached:
        return "terminal"
    if env.state.last_status in ROUTING_REJECTION_STATUSES:
        return (
            "post_routing_rejection"
            if env.scenario.schema_version.startswith("2.0-beta")
            else "post_invalid"
        )
    if env.state.last_status in FAILURE_STATUSES:
        return "post_failure"
    return "reachable"


def _reachable_states(
    scenario: ScenarioSpecV2,
    agents: dict[str, AgentProfileV2],
    limit: int = 64,
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
        actions = (
            env.admissible_actions()
            if scenario.schema_version.startswith("2.0-beta")
            else env.valid_actions()
        )
        for action in actions:
            if action == STOP_ACTION:
                continue
            child = env.clone()
            child.step(action)
            fingerprint = child.state.fingerprint()
            if fingerprint not in seen and not child.done:
                seen.add(fingerprint)
                queue.append(child)

    for distractor in scenario.distractor_ids:
        if distractor not in initial.applicable_actions():
            injected = initial.clone()
            injected.step(distractor)
            continuation = (
                injected.admissible_actions()
                if scenario.schema_version.startswith("2.0-beta")
                else injected.valid_actions()
            )
            if not injected.done and continuation:
                states.append(injected)
            break
    return states


def _select_alpha_states(
    environments: list[RoutingEnvironmentV2], maximum: int = 8
) -> list[RoutingEnvironmentV2]:
    selected: list[RoutingEnvironmentV2] = []
    seen: set[str] = set()

    def add(env: RoutingEnvironmentV2) -> None:
        fingerprint = env.state.fingerprint()
        if fingerprint not in seen and len(selected) < maximum:
            seen.add(fingerprint)
            selected.append(env)

    ordered = sorted(
        environments, key=lambda env: (env.state.steps_used, env.state.fingerprint())
    )
    initial = [env for env in ordered if _state_provenance(env) == "initial"]
    post_failure = [
        env for env in ordered if _state_provenance(env) == "post_failure"
    ]
    post_invalid = [
        env
        for env in ordered
        if _state_provenance(env)
        in {"post_invalid", "post_routing_rejection"}
    ]
    terminal = [env for env in ordered if _state_provenance(env) == "terminal"]
    tradeoff = [
        env
        for env in ordered
        if len(env.valid_actions()) > len(env.efficient_actions())
    ]
    multi_equivalent = [
        env
        for env in ordered
        if len(env.valid_actions()) > 1
        and len(env.valid_actions()) == len(env.efficient_actions())
    ]
    for group, limit in (
        (initial, 1),
        (post_failure, 1),
        (post_invalid, 1),
        (tradeoff, 2),
        (multi_equivalent, 2),
        (terminal, 1),
    ):
        for env in group[:limit]:
            add(env)
    for env in ordered:
        add(env)
        if len(selected) >= maximum:
            break
    return selected


def _select_beta_states(
    environments: list[RoutingEnvironmentV2], maximum: int = 12
) -> list[RoutingEnvironmentV2]:
    """Select a balanced, deterministic slice of reachable beta states."""
    selected: list[RoutingEnvironmentV2] = []
    seen: set[str] = set()

    def add(env: RoutingEnvironmentV2) -> None:
        fingerprint = env.state.fingerprint()
        if fingerprint not in seen and len(selected) < maximum:
            seen.add(fingerprint)
            selected.append(env)

    ordered = sorted(
        environments, key=lambda env: (env.state.steps_used, env.state.fingerprint())
    )
    groups = {
        "initial": [env for env in ordered if _state_provenance(env) == "initial"],
        "post_failure": [
            env for env in ordered if _state_provenance(env) == "post_failure"
        ],
        "post_rejection": [
            env
            for env in ordered
            if _state_provenance(env) == "post_routing_rejection"
        ],
        "multi": [env for env in ordered if len(env.admissible_actions()) > 1],
        "tradeoff": [
            env
            for env in ordered
            if len(env.admissible_actions()) > len(env.efficient_actions())
        ],
        "terminal": [env for env in ordered if env.goal_reached],
    }
    for name, limit in (
        ("initial", 1),
        ("post_failure", 2),
        ("post_rejection", 1),
        ("multi", 3),
        ("tradeoff", 2),
        ("terminal", 1),
    ):
        for env in groups[name][:limit]:
            add(env)
    for env in ordered:
        if env.goal_reached and any(item.goal_reached for item in selected):
            continue
        add(env)
        if len(selected) >= maximum:
            break
    return selected


def generate_decision_rows(bundle: BenchmarkBundleV2) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for scenario in bundle.scenarios:
        reachable = _reachable_states(scenario, bundle.agents)
        environments = (
            _select_beta_states(reachable)
            if scenario.schema_version.startswith("2.0-beta")
            else _select_alpha_states(reachable)
        )
        for index, env in enumerate(environments):
            outcomes = env.action_outcomes()
            observation = env.observation()
            provenance = _state_provenance(env)
            admissible = env.admissible_actions()
            progress = env.progress_actions()
            efficient = env.efficient_actions()
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
                        if scenario.counterfactual_group_id
                        and env.state.steps_used == 0
                        else None
                    ),
                    "workflow_family": scenario.family,
                    "state_provenance": provenance,
                    "state_variant_id": "canonical",
                    "task": scenario.task,
                    "observed_state": observation["state"],
                    "task_context": observation.get("task_context", {}),
                    "execution_memory": observation.get("execution_memory", {}),
                    "recent_events": observation.get("recent_events", []),
                    "budget": observation["budget"],
                    "agent_pool": observation["agent_profiles"],
                    "candidate_actions": list(env.candidate_actions()),
                    "applicable_actions": list(env.applicable_actions()),
                    "executable_actions": list(env.applicable_actions()),
                    "admissible_actions": list(admissible),
                    "progress_actions": list(progress),
                    "oracle_efficient_actions": list(efficient),
                    # Backward-compatible aliases used by V2-alpha tooling.
                    "valid_actions": list(
                        admissible
                        if scenario.schema_version.startswith("2.0-beta")
                        else progress
                    ),
                    "efficient_actions": list(efficient),
                    "stop_ready": env.goal_reached,
                    "distractor_actions": list(scenario.distractor_ids),
                    "action_outcomes": {
                        action: [
                            {"remaining_steps": steps, "remaining_cost": cost}
                            for steps, cost in points
                        ]
                        for action, points in outcomes.items()
                    },
                    "difficulty_tags": list(scenario.difficulty_tags),
                    "schema_version": scenario.schema_version,
                    "oracle_version": "pareto-v1",
                }
            )
    return rows


def quota_report(rows: list[dict[str, Any]], workflow_count: int) -> dict[str, Any]:
    count = len(rows)
    singleton = sum(len(row["valid_actions"]) == 1 for row in rows)
    multi = sum(len(row["valid_actions"]) >= 2 for row in rows)
    tradeoff = sum(
        len(row["valid_actions"]) >= 2
        and len(row["efficient_actions"]) < len(row["valid_actions"])
        for row in rows
    )
    provenance = Counter(row["state_provenance"] for row in rows)
    families = Counter(row["workflow_family"] for row in rows)
    counterfactual = sum(bool(row["counterfactual_group_id"]) for row in rows)

    def rate(value: int) -> float:
        return round(value / count, 6) if count else 0.0

    return {
        "release_stage": "alpha",
        "workflow_count": workflow_count,
        "decision_state_count": count,
        "singleton_valid_count": singleton,
        "singleton_valid_rate": rate(singleton),
        "multi_valid_count": multi,
        "multi_valid_rate": rate(multi),
        "valid_but_efficiency_tradeoff_count": tradeoff,
        "valid_but_efficiency_tradeoff_rate": rate(tradeoff),
        "stop_ready_count": sum(row["stop_ready"] for row in rows),
        "stop_ready_rate": rate(sum(row["stop_ready"] for row in rows)),
        "post_failure_count": provenance.get("post_failure", 0),
        "post_invalid_count": provenance.get("post_invalid", 0),
        "counterfactual_initial_count": counterfactual,
        "provenance_counts": dict(sorted(provenance.items())),
        "family_counts": dict(sorted(families.items())),
        "v2_release_targets_enforced": False,
        "note": "Alpha measures generator behavior; final V2 quotas are enforced after template expansion.",
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


def write_alpha_release(output_dir: str | Path, seed: int = 42) -> dict[str, Any]:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    bundle = build_alpha_bundle(seed)
    decisions = generate_decision_rows(bundle)
    report = quota_report(decisions, len(bundle.scenarios))

    metadata = dict(bundle.metadata)
    metadata["decision_state_count"] = len(decisions)
    metadata["splits"] = {"validation": len(bundle.scenarios)}
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
    _write_json(
        output_dir / "split_manifest.json",
        {
            "validation": [item.scenario_id for item in bundle.scenarios],
            "grouping_rule": "All alpha instances are validation-only; V2 beta splits by template group.",
        },
    )
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
