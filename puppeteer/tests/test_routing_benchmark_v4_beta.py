from __future__ import annotations

from collections import Counter
from pathlib import Path

from routing_benchmark.v4beta.audit import audit_v3_sources
from routing_benchmark.v4.generator import _reachable_states
from routing_benchmark.v4.post_fault import generate_post_fault_track
from routing_benchmark.v4beta.generator import (
    _reachable_states_fast,
    build_v4_beta_bundle,
)
from routing_benchmark.v4beta.reporting import wilson_interval
from routing_benchmark.v4beta.schemas import V4_BETA_SCHEMA_VERSION


PROJECT_DIR = Path(__file__).resolve().parents[1]
V3_ROOT = PROJECT_DIR / "data/RoutingBench/v3_beta"


def test_v3_audit_excludes_counterfactual_duplicates() -> None:
    manifest, report = audit_v3_sources(V3_ROOT)
    assert report["status_counts"]["eligible"] == 52
    assert all(
        row["audit_status"] == "eligible"
        for row in manifest
        if row["v3_family"] in {"multi_hop_evidence", "domain_reasoning_verification"}
    )


def test_v4_beta_quota_and_counterfactual_contract() -> None:
    bundle, _, _, annotations, _, workflows, *_ = build_v4_beta_bundle(V3_ROOT)
    assert bundle.metadata["version"] == V4_BETA_SCHEMA_VERSION
    assert len(workflows) == 124
    assert len(bundle.scenarios) == 620
    assert Counter(row["split"] for row in workflows) == Counter(
        {"development": 24, "validation": 40, "test": 60}
    )
    grouped = Counter(value["base_workflow_id"] for value in annotations.values())
    assert set(grouped.values()) == {5}


def test_holdout_domains_are_test_only() -> None:
    *_, workflows, _, _, _ = build_v4_beta_bundle(V3_ROOT)
    holdouts = {"document_media_processing", "resource_constrained_workflow"}
    assert all(row["split"] == "test" for row in workflows if row["task_domain"] in holdouts)


def test_wilson_interval_is_bounded() -> None:
    result = wilson_interval(7, 10)
    assert result["ci95_low"] < result["rate"] < result["ci95_high"]
    assert 0.0 <= result["ci95_low"] <= result["ci95_high"] <= 1.0


def test_fast_track_a_enumerator_preserves_reachable_states() -> None:
    bundle, *_ = build_v4_beta_bundle(V3_ROOT)
    for scenario in bundle.scenarios[:5]:
        reference = {
            env.state.fingerprint()
            for env in _reachable_states(scenario, bundle.agents)
        }
        optimized = {
            env.state.fingerprint()
            for env in _reachable_states_fast(scenario, bundle.agents)
        }
        assert optimized == reference


def test_every_domain_variant_has_unambiguous_transitions() -> None:
    bundle, *_ = build_v4_beta_bundle(V3_ROOT)
    seen = set()
    for scenario in bundle.scenarios:
        variant = scenario.scenario_id.rsplit("-", 1)[-1]
        key = (scenario.family, variant)
        if key in seen:
            continue
        seen.add(key)
        assert _reachable_states_fast(scenario, bundle.agents)
    assert len(seen) == 50


def test_fast_track_c_replay_preserves_checkpoint_labels() -> None:
    bundle, _, fixtures, annotations, *_ = build_v4_beta_bundle(V3_ROOT)
    scenarios = bundle.scenarios[:5]
    scenario_ids = {item.scenario_id for item in scenarios}
    selected_fixtures = [
        item for item in fixtures if item["scenario_id"] in scenario_ids
    ]
    selected_annotations = {
        key: value for key, value in annotations.items() if key in scenario_ids
    }
    slow = generate_post_fault_track(
        scenarios=scenarios,
        agents=bundle.agents,
        fixtures=selected_fixtures,
        annotations=selected_annotations,
        benchmark_version="4.0-alpha.2",
    )
    fast = generate_post_fault_track(
        scenarios=scenarios,
        agents=bundle.agents,
        fixtures=selected_fixtures,
        annotations=selected_annotations,
        benchmark_version=V4_BETA_SCHEMA_VERSION,
    )
    fields = lambda rows: [
        (
            item["checkpoint_state_fingerprint"],
            item["valid_actions"],
            item["oracle_min_remaining_calls"],
        )
        for item in rows
    ]
    assert fields(slow[2]) == fields(fast[2])
    assert len({item["pair_id"] for item in fast[3]}) == 4
