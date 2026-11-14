from __future__ import annotations

import hashlib
import json
import random
import re
from collections import Counter
from dataclasses import replace
from pathlib import Path
from typing import Any, Iterable, Mapping

from routing_benchmark.v2.loader import BenchmarkBundleV2, validate_bundle
from routing_benchmark.v2.schemas import AgentProfileV2, ScenarioSpecV2, TransitionSpecV2
from routing_benchmark.v3.generator import (
    _checksums,
    _profile,
    _write_json,
    _write_jsonl,
    agent_catalog_v3,
    build_v3_bundle,
    generate_decision_rows_v3,
)
from routing_benchmark.v3.schemas import ProvenanceRecordV3

from .schemas import V3_BETA_SCHEMA_VERSION, v3_beta_schema_definitions


PROJECT_DIR = Path(__file__).resolve().parents[2]
MUSIQUE_RELATIVE_PATH = Path("data/MuSiQue/validation.parquet")
MMLU_RELATIVE_PATH = Path("data/MMLU-Pro/test.parquet")
DERIVATION_SCRIPT = "scripts/build_routingbench_v3_beta.py"
RELEASE_REVISION = "role-card-r1"


def agent_catalog_v3_beta() -> dict[str, AgentProfileV2]:
    catalog = agent_catalog_v3()
    additions = (
        _profile(
            "agent_33",
            "Quantitative and Formal Reasoner",
            "Solve quantitative, mathematical, physical, and formal-logic dependencies.",
            ["the task contains a quantitative or formal dependency"],
            ["the unresolved work is retrieval, media inspection, or finalization"],
            ["question", "equations", "numeric data", "candidate options"],
            ["candidate answer", "formal derivation"],
            "domain_reasoning",
        ),
        _profile(
            "agent_34",
            "Natural and Life Science Specialist",
            "Solve biology, chemistry, health, and natural-science dependencies.",
            ["the question requires natural or life science knowledge"],
            ["the question is primarily legal, historical, or computational"],
            ["question", "scientific context", "candidate options"],
            ["candidate answer", "scientific reasoning"],
            "domain_reasoning",
        ),
        _profile(
            "agent_35",
            "Computing and Engineering Specialist",
            "Solve computing, systems, algorithms, and engineering dependencies.",
            ["the question requires computing or engineering expertise"],
            ["the question is primarily biomedical, legal, or historical"],
            ["question", "technical context", "candidate options"],
            ["candidate answer", "technical reasoning"],
            "domain_reasoning",
        ),
        _profile(
            "agent_36",
            "Social, Legal, and Business Specialist",
            "Solve business, economics, law, and social-institutional dependencies.",
            ["the question concerns institutions, markets, organizations, or law"],
            ["the question is primarily mathematical or natural-scientific"],
            ["question", "institutional context", "candidate options"],
            ["candidate answer", "domain reasoning"],
            "domain_reasoning",
        ),
        _profile(
            "agent_37",
            "Humanities and Behavioral Specialist",
            "Solve history, philosophy, psychology, and behavioral dependencies.",
            ["the question requires humanities or behavioral-science expertise"],
            ["the question is primarily computational or engineering-focused"],
            ["question", "humanities context", "candidate options"],
            ["candidate answer", "interpretive reasoning"],
            "domain_reasoning",
        ),
        _profile(
            "agent_38",
            "Generalist Independent Solver",
            "Produce an independent answer when several domain boundaries plausibly overlap.",
            ["the task is answerable from the visible input", "domain boundaries overlap"],
            ["a tool-only operation or final commitment is the sole unresolved step"],
            ["question", "candidate options", "supplied evidence"],
            ["candidate answer", "independent reasoning"],
            "domain_reasoning",
        ),
        _profile(
            "agent_39",
            "Adversarial Answer Verifier",
            "Test a candidate answer against the original question and visible evidence.",
            ["a candidate answer exists and has not been checked"],
            ["no candidate exists", "only final commitment remains"],
            ["candidate answer", "original question", "visible evidence"],
            ["verified candidate", "error diagnosis"],
            "verification",
        ),
    )
    catalog.update({item.agent_id: item for item in additions})
    return catalog


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _stable_rows(frame, key: str, seed: int):
    return sorted(
        (row for _, row in frame.iterrows()),
        key=lambda row: hashlib.sha256(
            f"{seed}:{row[key]}".encode("utf-8")
        ).hexdigest(),
    )


def _plain_sequence(value: Any) -> list[Any]:
    if value is None:
        return []
    if hasattr(value, "tolist"):
        value = value.tolist()
    return list(value) if isinstance(value, (list, tuple)) else [value]


def _transition(
    scenario_id: str,
    suffix: str,
    action: str,
    *,
    requires_all: Iterable[str] = (),
    forbids: Iterable[str] = (),
    requires_artifacts: Iterable[str] = (),
    add_facts: Iterable[str] = (),
    remove_facts: Iterable[str] = (),
    add_artifacts: Iterable[str] = (),
    effect: str,
) -> TransitionSpecV2:
    return TransitionSpecV2(
        transition_id=f"{scenario_id}:{suffix}:{action}",
        action=action,
        requires_all=frozenset(requires_all),
        forbids=frozenset(forbids),
        requires_artifacts=frozenset(requires_artifacts),
        add_facts=frozenset(add_facts),
        remove_facts=frozenset(remove_facts),
        add_artifacts=frozenset(add_artifacts),
        public_effect=effect,
        cost_override=1.0,
        minimum_cost_remaining=0.0,
    )


def _pool(
    catalog: Mapping[str, AgentProfileV2],
    active: set[str],
    scenario_id: str,
    stress: bool,
) -> tuple[tuple[str, ...], tuple[str, ...], str]:
    if stress:
        values = list(catalog)
        mode = "stress"
    else:
        inactive = [item for item in catalog if item not in active]
        random.Random(f"v3b-pool:{scenario_id}").shuffle(inactive)
        values = [*active, *inactive[: max(0, 16 - len(active))]]
        mode = "overlap"
    random.Random(f"v3b-order:{scenario_id}").shuffle(values)
    distractors = tuple(item for item in values if item not in active)
    return tuple(values), distractors, mode


def _musique_scenarios(
    catalog: Mapping[str, AgentProfileV2], seed: int
) -> tuple[
    list[ScenarioSpecV2],
    dict[str, ProvenanceRecordV3],
    dict[str, dict[str, Any]],
    dict[str, dict[str, Any]],
]:
    import pandas as pd

    path = PROJECT_DIR / MUSIQUE_RELATIVE_PATH
    if not path.exists():
        raise FileNotFoundError(
            f"MuSiQue source missing at {path}. Download the documented validation snapshot first."
        )
    frame = pd.read_parquet(path)
    frame = frame[(frame["answerable"] == True) & frame["answer"].notna()]  # noqa: E712
    by_hop: dict[int, list[Any]] = {}
    for row in _stable_rows(frame, "id", seed):
        # Official MuSiQue IDs use both ``2hop__...`` and forms such as
        # ``3hop1__...``/``4hop2__...``.  The leading integer is the public
        # composition depth; the optional digit identifies a construction
        # pattern rather than an additional hop.
        match = re.match(r"(\d+)hop\d*__", str(row["id"]))
        if match:
            by_hop.setdefault(int(match.group(1)), []).append(row)
    selected: list[tuple[int, Any]] = []
    for hop in (2, 3, 4):
        selected.extend((hop, row) for row in by_hop.get(hop, [])[:8])
    if len(selected) < 24:
        raise ValueError("MuSiQue source does not contain the required 24 stratified items")
    split_pattern = {
        2: ("development", "validation", "test", "development", "validation", "test", "development", "validation"),
        3: ("test", "development", "validation", "test", "development", "validation", "test", "development"),
        4: ("validation", "test", "development", "validation", "test", "development", "validation", "test"),
    }
    source_hash = _sha256(path)
    scenarios: list[ScenarioSpecV2] = []
    provenance: dict[str, ProvenanceRecordV3] = {}
    task_inputs: dict[str, dict[str, Any]] = {}
    evaluation: dict[str, dict[str, Any]] = {}
    hop_indices: Counter[int] = Counter()
    for ordinal, (hop, row) in enumerate(selected, start=1):
        hop_indices[hop] += 1
        split = split_pattern[hop][hop_indices[hop] - 1]
        scenario_id = f"v3b-musique-{hop}hop-{hop_indices[hop]:02d}"
        question = str(row["question"]).strip()
        transitions: list[TransitionSpecV2] = []
        for action in ("agent_01",):
            transitions.append(
                _transition(
                    scenario_id,
                    "plan",
                    action,
                    requires_all=("task_available", "evidence_unresolved"),
                    forbids=("plan_ready", "evidence_ready"),
                    add_facts=("plan_ready",),
                    add_artifacts=("dependency_plan",),
                    effect="A dependency plan for the multi-hop question is available.",
                )
            )
        for action in ("agent_02", "agent_25"):
            transitions.append(
                _transition(
                    scenario_id,
                    "retrieve",
                    action,
                    requires_all=("task_available", "corpus_available", "evidence_unresolved"),
                    forbids=("evidence_ready",),
                    add_facts=("evidence_ready",),
                    remove_facts=("evidence_unresolved",),
                    add_artifacts=("evidence_packet",),
                    effect="A grounded evidence packet was extracted from the supplied corpus.",
                )
            )
        for action in ("agent_05", "agent_14"):
            transitions.append(
                _transition(
                    scenario_id,
                    "infer",
                    action,
                    requires_all=("evidence_ready",),
                    forbids=("relation_resolved",),
                    requires_artifacts=("evidence_packet",),
                    add_facts=("relation_resolved",),
                    add_artifacts=("resolved_relation",),
                    effect="The multi-hop relation needed by the original question was resolved.",
                )
            )
        for action in ("agent_06", "agent_26"):
            transitions.append(
                _transition(
                    scenario_id,
                    "compose",
                    action,
                    requires_all=("evidence_ready", "relation_resolved"),
                    forbids=("candidate_ready",),
                    requires_artifacts=("evidence_packet", "resolved_relation"),
                    add_facts=("candidate_ready",),
                    add_artifacts=("candidate_answer",),
                    effect="A candidate answer to the original question is available.",
                )
            )
        for action in ("agent_07", "agent_27", "agent_39"):
            transitions.append(
                _transition(
                    scenario_id,
                    "verify",
                    action,
                    requires_all=("candidate_ready", "evidence_ready"),
                    forbids=("candidate_verified",),
                    requires_artifacts=("candidate_answer", "evidence_packet"),
                    add_facts=("candidate_verified",),
                    add_artifacts=("verified_candidate",),
                    effect="The candidate was checked against the original question and evidence.",
                )
            )
        for action in ("agent_08", "agent_24", "agent_30"):
            transitions.append(
                _transition(
                    scenario_id,
                    "finalize",
                    action,
                    requires_all=("candidate_verified",),
                    requires_artifacts=("verified_candidate",),
                    add_facts=("final_committed",),
                    add_artifacts=("committed_final_artifact",),
                    effect="A short final answer was committed for evaluation.",
                )
            )
        active = {item.action for item in transitions}
        pool, distractors, pool_mode = _pool(
            catalog, active, scenario_id, stress=ordinal % 2 == 0
        )
        paragraphs = []
        supporting_ids: list[int] = []
        for raw in _plain_sequence(row["paragraphs"]):
            item = dict(raw)
            paragraph_id = int(item.get("idx", len(paragraphs)))
            paragraphs.append(
                {
                    "idx": paragraph_id,
                    "title": str(item.get("title", "")),
                    "text": str(item.get("paragraph_text", "")),
                }
            )
            if bool(item.get("is_supporting", False)):
                supporting_ids.append(paragraph_id)
        scenario = ScenarioSpecV2(
            scenario_id=scenario_id,
            template_id=f"v3b-musique-{hop}hop",
            instance_id=str(row["id"]),
            split=split,
            split_group_id=f"musique:{row['id']}",
            counterfactual_group_id=None,
            family="multi_hop_evidence",
            task=f"Answer the multi-hop question using the supplied corpus: {question}",
            agent_pool=pool,
            distractor_ids=distractors,
            initial_facts=frozenset(
                {"task_available", "corpus_available", "evidence_unresolved"}
            ),
            initial_artifacts=frozenset({"question", "supplied_corpus"}),
            fact_descriptions={
                "task_available": "The original multi-hop question is available.",
                "corpus_available": "A supplied text corpus is available to evidence agents.",
                "evidence_unresolved": "Evidence dependencies for the original question remain unresolved.",
                "plan_ready": "A dependency plan is available.",
                "evidence_ready": "A grounded evidence packet is available.",
                "relation_resolved": "The required multi-hop relation has been resolved.",
                "candidate_ready": "A candidate answer to the original question is available.",
                "candidate_verified": "The candidate has been checked against visible evidence.",
                "final_committed": "A final answer has been committed.",
            },
            artifact_descriptions={
                "question": "original question",
                "supplied_corpus": "supplied paragraph corpus",
                "dependency_plan": "dependency plan",
                "evidence_packet": "grounded evidence packet",
                "resolved_relation": "resolved multi-hop relation",
                "candidate_answer": "candidate answer",
                "verified_candidate": "verified candidate",
                "committed_final_artifact": "committed final answer",
            },
            goal_facts=frozenset({"final_committed"}),
            goal_artifacts=frozenset({"committed_final_artifact"}),
            transitions=tuple(transitions),
            max_steps=8,
            max_cost=8.0,
            difficulty_tags=(
                "real_task",
                "semantic_gold",
                f"hop_count:{hop}",
                "source:musique",
                f"pool:{pool_mode}",
                "set_valued_ground_truth",
            ),
            schema_version=V3_BETA_SCHEMA_VERSION,
            public_context={
                "available_resources": [
                    {"type": "supplied_corpus", "status": "available", "item_count": len(paragraphs)}
                ],
                "task_constraints": {
                    "answer_original_question": True,
                    "hidden_source_location_provided": False,
                },
            },
            max_stagnant_repeats=2,
        )
        scenarios.append(scenario)
        provenance_id = f"prov-{scenario_id}"
        provenance[provenance_id] = ProvenanceRecordV3(
            provenance_id=provenance_id,
            scenario_id=scenario_id,
            source_type="licensed_public_benchmark",
            source_dataset="MuSiQue",
            source_version=f"sha256:{source_hash}",
            source_split="validation",
            source_item_id=str(row["id"]),
            source_path=MUSIQUE_RELATIVE_PATH.as_posix(),
            source_sha256=source_hash,
            license="CC-BY-4.0",
            derivation_script=DERIVATION_SCRIPT,
            derivation_version=V3_BETA_SCHEMA_VERSION,
            transformation_steps=(
                "selected answerable items deterministically and stratified by hop count",
                "removed supporting-paragraph labels from actor input",
                "stored official answers only in the evaluation sidecar",
                "derived valid action sets from executable artifact preconditions",
            ),
            validation_method="source hash and item lookup plus executable transition validation",
            human_review_status="pending",
        )
        task_inputs[scenario_id] = {
            "question": question,
            "paragraphs": paragraphs,
        }
        evaluation[scenario_id] = {
            "scenario_id": scenario_id,
            "metric": "normalized_exact_match",
            "answer": str(row["answer"]),
            "aliases": [str(item) for item in _plain_sequence(row["answer_aliases"])],
            "supporting_ids": supporting_ids,
            "source_item_id": str(row["id"]),
        }
    return scenarios, provenance, task_inputs, evaluation


MMLU_SPLIT_CATEGORIES = {
    "development": ("biology", "business", "computer science", "math"),
    "validation": ("chemistry", "economics", "psychology", "other"),
    "test": ("engineering", "health", "history", "law", "philosophy", "physics"),
}

MMLU_SPECIALISTS = {
    "biology": ("agent_34",),
    "chemistry": ("agent_34", "agent_33"),
    "health": ("agent_34",),
    "computer science": ("agent_35", "agent_33"),
    "engineering": ("agent_35", "agent_33"),
    "math": ("agent_33",),
    "physics": ("agent_33", "agent_34"),
    "business": ("agent_36",),
    "economics": ("agent_36", "agent_33"),
    "law": ("agent_36",),
    "history": ("agent_37",),
    "philosophy": ("agent_37", "agent_33"),
    "psychology": ("agent_37", "agent_34"),
    "other": ("agent_38",),
}


def _mmlu_scenarios(
    catalog: Mapping[str, AgentProfileV2], seed: int
) -> tuple[
    list[ScenarioSpecV2],
    dict[str, ProvenanceRecordV3],
    dict[str, dict[str, Any]],
    dict[str, dict[str, Any]],
]:
    import pandas as pd

    path = PROJECT_DIR / MMLU_RELATIVE_PATH
    if not path.exists():
        raise FileNotFoundError(f"MMLU-Pro source missing at {path}")
    frame = pd.read_parquet(path)
    source_hash = _sha256(path)
    scenarios: list[ScenarioSpecV2] = []
    provenance: dict[str, ProvenanceRecordV3] = {}
    task_inputs: dict[str, dict[str, Any]] = {}
    evaluation: dict[str, dict[str, Any]] = {}
    ordinal = 0
    for split, categories in MMLU_SPLIT_CATEGORIES.items():
        for category in categories:
            category_rows = frame[frame["category"].astype(str).str.casefold() == category]
            if category == "other" and category_rows.empty:
                category_rows = frame[frame["category"].astype(str).str.casefold() == "others"]
            selected = _stable_rows(category_rows, "question_id", seed)[:2]
            if len(selected) < 2:
                raise ValueError(f"MMLU-Pro category {category!r} has fewer than two items")
            for category_index, row in enumerate(selected, start=1):
                ordinal += 1
                scenario_id = f"v3b-mmlu-{category.replace(' ', '-')}-{category_index:02d}"
                options = [str(item) for item in _plain_sequence(row["options"])]
                labels = [chr(ord("A") + index) for index in range(len(options))]
                rendered_options = "\n".join(
                    f"{label}. {option}" for label, option in zip(labels, options)
                )
                question = str(row["question"]).strip()
                task = (
                    "Select the best answer to this multiple-choice question and commit "
                    f"only its option letter.\nQuestion: {question}\n{rendered_options}"
                )
                specialists = tuple(
                    dict.fromkeys((*MMLU_SPECIALISTS[category], "agent_38"))
                )
                transitions: list[TransitionSpecV2] = []
                for action in specialists:
                    transitions.append(
                        _transition(
                            scenario_id,
                            "reason",
                            action,
                            requires_all=("question_available", "reasoning_pending"),
                            forbids=("candidate_ready",),
                            add_facts=("candidate_ready",),
                            remove_facts=("reasoning_pending",),
                            add_artifacts=("candidate_answer",),
                            effect="A reasoned candidate option is available.",
                        )
                    )
                for action in ("agent_39", "agent_27"):
                    transitions.append(
                        _transition(
                            scenario_id,
                            "verify",
                            action,
                            requires_all=("candidate_ready",),
                            forbids=("candidate_verified",),
                            requires_artifacts=("candidate_answer",),
                            add_facts=("candidate_verified",),
                            add_artifacts=("verified_candidate",),
                            effect="The candidate option was adversarially checked against the question.",
                        )
                    )
                for action in ("agent_08", "agent_24", "agent_30"):
                    transitions.append(
                        _transition(
                            scenario_id,
                            "finalize",
                            action,
                            requires_all=("candidate_verified",),
                            requires_artifacts=("verified_candidate",),
                            add_facts=("final_committed",),
                            add_artifacts=("committed_final_artifact",),
                            effect="A final option letter was committed for evaluation.",
                        )
                    )
                active = {item.action for item in transitions}
                pool, distractors, pool_mode = _pool(
                    catalog, active, scenario_id, stress=ordinal % 2 == 0
                )
                scenario = ScenarioSpecV2(
                    scenario_id=scenario_id,
                    template_id="v3b-mmlu-domain-reasoning",
                    instance_id=str(row["question_id"]),
                    split=split,
                    split_group_id=f"mmlu-pro:{row['question_id']}",
                    counterfactual_group_id=None,
                    family="domain_reasoning_verification",
                    task=task,
                    agent_pool=pool,
                    distractor_ids=distractors,
                    initial_facts=frozenset({"question_available", "reasoning_pending"}),
                    initial_artifacts=frozenset({"question", "candidate_options"}),
                    fact_descriptions={
                        "question_available": "The original multiple-choice question and options are available.",
                        "reasoning_pending": "No reasoned candidate option exists yet.",
                        "candidate_ready": "A reasoned candidate option is available.",
                        "candidate_verified": "The candidate option has been adversarially checked.",
                        "final_committed": "A final option letter has been committed.",
                    },
                    artifact_descriptions={
                        "question": "multiple-choice question",
                        "candidate_options": "candidate options",
                        "candidate_answer": "candidate option",
                        "verified_candidate": "verified option",
                        "committed_final_artifact": "committed option letter",
                    },
                    goal_facts=frozenset({"final_committed"}),
                    goal_artifacts=frozenset({"committed_final_artifact"}),
                    transitions=tuple(transitions),
                    max_steps=5,
                    max_cost=5.0,
                    difficulty_tags=(
                        "real_task",
                        "semantic_gold",
                        "source:mmlu-pro",
                        f"domain:{category}",
                        "unseen_domain_split" if split == "test" else "seen_domain_split",
                        f"pool:{pool_mode}",
                        "set_valued_ground_truth",
                    ),
                    schema_version=V3_BETA_SCHEMA_VERSION,
                    public_context={
                        "available_resources": [{"type": "closed_book_question", "status": "available"}],
                        "task_constraints": {
                            "commit_option_letter_only": True,
                            "category_label_visible": False,
                        },
                    },
                    max_stagnant_repeats=2,
                )
                scenarios.append(scenario)
                provenance_id = f"prov-{scenario_id}"
                provenance[provenance_id] = ProvenanceRecordV3(
                    provenance_id=provenance_id,
                    scenario_id=scenario_id,
                    source_type="licensed_public_benchmark",
                    source_dataset="MMLU-Pro",
                    source_version=f"sha256:{source_hash}",
                    source_split="test",
                    source_item_id=str(row["question_id"]),
                    source_path=MMLU_RELATIVE_PATH.as_posix(),
                    source_sha256=source_hash,
                    license="MIT",
                    derivation_script=DERIVATION_SCRIPT,
                    derivation_version=V3_BETA_SCHEMA_VERSION,
                    transformation_steps=(
                        "selected two items per declared domain using a deterministic hash order",
                        "held out declared domain groups across development, validation, and test",
                        "stored official option labels only in the evaluation sidecar",
                        "derived valid action sets from domain-overlap and artifact preconditions",
                    ),
                    validation_method="source hash and question_id lookup plus executable transition validation",
                    human_review_status="pending",
                )
                task_inputs[scenario_id] = {
                    "question": question,
                    "options": dict(zip(labels, options)),
                }
                evaluation[scenario_id] = {
                    "scenario_id": scenario_id,
                    "metric": "exact_choice",
                    "answer": str(row["answer"]),
                    "aliases": [],
                    "supporting_ids": [],
                    "source_item_id": str(row["question_id"]),
                }
    return scenarios, provenance, task_inputs, evaluation


def build_v3_beta_bundle(seed: int = 42):
    alpha_bundle, alpha_provenance = build_v3_bundle(seed)
    catalog = agent_catalog_v3_beta()
    alpha_scenarios = [
        replace(
            item,
            schema_version=V3_BETA_SCHEMA_VERSION,
            agent_pool=tuple(
                value for value in item.agent_pool if value in catalog
            ),
            difficulty_tags=tuple(
                dict.fromkeys((*item.difficulty_tags, "source:routingbench-v2"))
            ),
        )
        for item in alpha_bundle.scenarios
    ]
    musique = _musique_scenarios(catalog, seed)
    mmlu = _mmlu_scenarios(catalog, seed)
    scenarios = tuple((*alpha_scenarios, *musique[0], *mmlu[0]))
    provenance = {**alpha_provenance, **musique[1], **mmlu[1]}
    task_inputs = {**musique[2], **mmlu[2]}
    evaluation = {**musique[3], **mmlu[3]}
    metadata = {
        "name": "RoutingBench",
        "version": V3_BETA_SCHEMA_VERSION,
        "release_revision": RELEASE_REVISION,
        "release_stage": "beta",
        "seed": seed,
        "workflow_count": len(scenarios),
        "family_count": len({item.family for item in scenarios}),
        "real_task_workflow_count": len(evaluation),
        "capability_scores": False,
        "backbone_identity_in_router_input": False,
        "preferred_actions": False,
        "action_utilities": False,
        "primary_ground_truth": "valid_actions",
        "valid_action_derivation": "formal_transition_preconditions",
        "track_b_execution": "live_llm_actor",
        "simulated_actor_outputs": False,
        "semantic_gold_visibility": "evaluation_sidecar_only",
        "human_review_status": "pending",
        "terminal_stop_budget_policy": "reserved_zero_cost_terminal_decision",
    }
    bundle = BenchmarkBundleV2(metadata=metadata, agents=catalog, scenarios=scenarios)
    validate_bundle(bundle)
    return bundle, provenance, task_inputs, evaluation


def _quota(rows: list[dict[str, Any]], scenarios: tuple[ScenarioSpecV2, ...]) -> dict[str, Any]:
    nonterminal = [item for item in rows if not item["terminal_valid"]]
    multi = [item for item in nonterminal if len(item["valid_actions"]) >= 2]
    singleton = [item for item in nonterminal if len(item["valid_actions"]) == 1]
    real = [item for item in scenarios if "real_task" in item.difficulty_tags]
    return {
        "release_stage": "beta",
        "workflow_count": len(scenarios),
        "decision_state_count": len(rows),
        "real_task_workflow_count": len(real),
        "family_counts": dict(sorted(Counter(item.family for item in scenarios).items())),
        "split_counts": dict(sorted(Counter(item.split for item in scenarios).items())),
        "source_counts": dict(
            sorted(
                Counter(
                    next(
                        (tag.split(":", 1)[1] for tag in item.difficulty_tags if tag.startswith("source:")),
                        "unknown",
                    )
                    for item in scenarios
                ).items()
            )
        ),
        "nonterminal_count": len(nonterminal),
        "singleton_valid_rate": len(singleton) / len(nonterminal) if nonterminal else 0.0,
        "multi_valid_rate": len(multi) / len(nonterminal) if nonterminal else 0.0,
        "preferred_actions_present": any("preferred_actions" in item for item in rows),
        "action_utilities_present": any("action_utilities" in item for item in rows),
        "targets": {
            "at_least_100_workflows": len(scenarios) >= 100,
            "at_least_50_real_tasks": len(real) >= 50,
            "at_least_two_licensed_real_sources": len(
                {
                    tag
                    for item in real
                    for tag in item.difficulty_tags
                    if tag.startswith("source:")
                }
            ) >= 2,
            "multi_nonterminal_at_least_50_percent": (
                len(multi) / len(nonterminal) >= 0.50 if nonterminal else False
            ),
        },
    }


SOURCE_REGISTRY = {
    "MuSiQue": {
        "included": True,
        "redistributable": True,
        "license": "CC-BY-4.0",
        "upstream": "https://github.com/StonyBrookNLP/musique",
        "local_path": MUSIQUE_RELATIVE_PATH.as_posix(),
    },
    "MMLU-Pro": {
        "included": True,
        "redistributable": True,
        "license": "MIT",
        "upstream": "https://huggingface.co/datasets/TIGER-Lab/MMLU-Pro",
        "local_path": MMLU_RELATIVE_PATH.as_posix(),
    },
    "GAIA": {
        "included": False,
        "redistributable": False,
        "local_only": True,
        "reason": "The gated dataset card prohibits resharing validation/test data in crawlable form.",
    },
    "SRDD": {
        "included": False,
        "reason": "The local CSV has no pinned upstream revision and dataset license record yet.",
    },
}


def write_v3_beta_release(output_dir: str | Path, seed: int = 42) -> dict[str, Any]:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    bundle, provenance, task_inputs, evaluation = build_v3_beta_bundle(seed)
    rows = generate_decision_rows_v3(
        bundle, provenance, schema_version=V3_BETA_SCHEMA_VERSION
    )
    quota = _quota(rows, bundle.scenarios)
    metadata = dict(bundle.metadata)
    metadata["decision_state_count"] = len(rows)
    metadata["splits"] = dict(
        sorted(Counter(item.split for item in bundle.scenarios).items())
    )
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
    _write_jsonl(
        output_dir / "task_inputs.jsonl",
        (
            {"scenario_id": scenario_id, "task_input": value}
            for scenario_id, value in sorted(task_inputs.items())
        ),
    )
    _write_jsonl(
        output_dir / "evaluation.jsonl",
        (value for _, value in sorted(evaluation.items())),
    )
    _write_jsonl(output_dir / "decision_states.jsonl", rows)
    _write_json(output_dir / "quota_report.json", quota)
    _write_json(output_dir / "source_registry.json", SOURCE_REGISTRY)
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
                "provenance_id": by_scenario[item.scenario_id].provenance_id,
                "source_item_id": by_scenario[item.scenario_id].source_item_id,
                "review_status": "pending",
                "checklist": [
                    "public state is sufficient to justify every valid action",
                    "no invalid action is defensible from the public state",
                    "gold and source-only annotations are absent from router and actor inputs",
                    "role overlap represents genuine interchangeable routing choices",
                ],
            }
            for item in bundle.scenarios
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
            "evaluation.jsonl",
            "decision_states.jsonl",
            "quota_report.json",
            "source_registry.json",
            "split_manifest.json",
            "human_review_queue.jsonl",
        )
    ]
    _write_json(output_dir / "checksums.json", _checksums(generated))
    specs = output_dir / "specs"
    specs.mkdir(parents=True, exist_ok=True)
    for name, schema in v3_beta_schema_definitions().items():
        _write_json(specs / name, schema)
    return {"metadata": metadata, "quota_report": quota}
