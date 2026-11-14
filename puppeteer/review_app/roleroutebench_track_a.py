from __future__ import annotations

import argparse
import copy
import html
import json
import sys
from pathlib import Path
from typing import Any

import streamlit as st

PROJECT = Path(__file__).resolve().parents[1]
if str(PROJECT) not in sys.path:
    sys.path.insert(0, str(PROJECT))

from review_app.track_a_presentation import (
    artifact_items,
    candidate_actions,
    contract_badges,
    current_facts,
    decision_step,
    discover_labeling_dirs,
    domain_label,
    load_observation,
    role_cards,
    role_pool_is_complete,
    terminal_summary,
)
from routing_benchmark.roleroutebench.track_a.canonicalize import read_jsonl
from routing_benchmark.roleroutebench.track_a.review import append_review, latest_reviews, make_review_record
from routing_benchmark.roleroutebench.track_a.schema import ACTION_CLASSES, STATE_PHASES


def parse_args() -> tuple[Path, str]:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--root", type=Path, default=PROJECT / "data" / "RoleRouteBench" / "track_a_4d_live_v1")
    parser.add_argument("--labeling-dir", default="all")
    args, _ = parser.parse_known_args()
    return args.root.resolve(), args.labeling_dir


def load_review_data(root: Path, labeling_dir: str) -> tuple[dict[str, dict[str, Any]], dict[str, dict[str, Any]], dict[str, str]]:
    states: dict[str, dict[str, Any]] = {}
    prelabels: dict[str, dict[str, Any]] = {}
    cohorts: dict[str, str] = {}
    for cohort_root in discover_labeling_dirs(root, labeling_dir):
        cohort = cohort_root.name
        for row in read_jsonl(cohort_root / "sample_states.jsonl"):
            decision_id = row["decision_id"]
            if decision_id in states:
                raise ValueError(f"duplicate decision_id across cohorts: {decision_id}")
            states[decision_id] = row
            cohorts[decision_id] = cohort
        for row in read_jsonl(cohort_root / "llm_prelabels.jsonl"):
            decision_id = row["decision_id"]
            if decision_id in prelabels:
                raise ValueError(f"duplicate pre-label across cohorts: {decision_id}")
            prelabels[decision_id] = row
    return states, prelabels, cohorts


def esc(value: Any) -> str:
    return html.escape(str(value), quote=True)


def badge_html(values: list[str]) -> str:
    return "".join(f'<span class="rrb-chip">{esc(value)}</span>' for value in values)


def render_header(state: dict[str, Any], prelabel: dict[str, Any], observation: dict[str, Any]) -> None:
    actions = candidate_actions(state, observation)
    roles = role_cards(observation)
    terminal, terminal_note = terminal_summary(state["domain"], observation)
    budget = observation.get("budget") or {}
    title = decision_step(state["decision_id"])
    pool_complete = role_pool_is_complete(state, observation)
    st.markdown(
        f"""
        <section class="rrb-heading">
          <h1>Decision brief · {esc(title)}</h1>
          <code>{esc(state['packet_id'])}</code>
          <div class="rrb-chips">
            <span class="rrb-chip">{esc(domain_label(state['domain']))}</span>
            <span class="rrb-chip">{esc(prelabel.get('state_phase', 'unlabeled phase'))}</span>
          </div>
          <p>Role pool luôn hiển thị đủ; Terra không được dùng để lọc role card hoặc candidate action.</p>
        </section>
        <section class="rrb-metrics" aria-label="Decision summary">
          <div class="rrb-metric"><span>Budget</span><strong>{esc(budget.get('calls_remaining', state.get('budget_remaining')))} calls</strong><small>{'STOP terminal được reserve' if budget.get('reserved_terminal_stop') else 'Không reserve terminal STOP'}</small></div>
          <div class="rrb-metric"><span>Role pool</span><strong>{len(roles)} + STOP</strong><small>{len(roles)} role cards {'đầy đủ' if pool_complete else 'không khớp'} + {len(actions) - len(roles)} terminal action</small></div>
          <div class="rrb-metric"><span>Terminal</span><strong>{'Đã đạt' if terminal else 'Chưa đạt'}</strong><small>{esc(terminal_note)}</small></div>
        </section>
        """,
        unsafe_allow_html=True,
    )


def render_decision_brief(state: dict[str, Any], observation: dict[str, Any]) -> None:
    task = observation.get("task") or "No public task contract is present."
    st.markdown(
        f"""
        <section class="rrb-panel">
          <h2>Task contract</h2>
          <p class="rrb-task">{esc(task)}</p>
          <div class="rrb-chips">{badge_html(contract_badges(state['domain'], observation))}</div>
        </section>
        """,
        unsafe_allow_html=True,
    )
    st.markdown("## Current facts")
    st.caption("Mỗi fact giữ đường dẫn về field nguồn. Nội dung đầy đủ vẫn nằm trong Evidence và Audit JSON.")
    rows = []
    for fact in current_facts(state["domain"], observation):
        rows.append(
            f"""
            <div class="rrb-fact-row">
              <span class="rrb-fact-label">{esc(fact['label'])}</span>
              <span class="rrb-fact-value">{esc(fact['value'])}<code>{esc(fact['path'])}</code></span>
            </div>
            """
        )
    st.markdown(f'<section class="rrb-panel rrb-facts">{"".join(rows)}</section>', unsafe_allow_html=True)


def render_role_cards(observation: dict[str, Any]) -> None:
    cards = role_cards(observation)
    st.markdown("## Full router-visible role pool")
    st.caption("Các card được hiển thị theo đúng thứ tự router nhận. Không dùng pre-label để ẩn hoặc sắp hạng role.")
    known = {"agent_id", "role_name", "semantic_family", "role_goal", "use_when", "avoid_when", "expected_outputs"}
    for index, card in enumerate(cards, start=1):
        role_id = card.get("agent_id", f"role-{index}")
        role_name = card.get("role_name")
        heading = f"{index:02d} · {role_id}" + (f" — {role_name}" if role_name and role_name != role_id else "")
        with st.expander(heading, expanded=index == 1):
            if card.get("semantic_family"):
                st.caption(f"Semantic family: {card['semantic_family']}")
            st.markdown(f"**Role goal**  \n{card.get('role_goal', 'Not declared')}")
            left, right = st.columns(2)
            with left:
                st.markdown("**Use when**")
                for item in card.get("use_when") or []:
                    st.markdown(f"- {item}")
                st.markdown("**Expected outputs**")
                for item in card.get("expected_outputs") or []:
                    st.markdown(f"- {item}")
            with right:
                st.markdown("**Avoid when**")
                for item in card.get("avoid_when") or []:
                    st.markdown(f"- {item}")
            remaining = {key: value for key, value in card.items() if key not in known}
            if remaining:
                st.markdown("**Other router-visible fields**")
                st.json(remaining, expanded=False)


def render_action_editor(
    state: dict[str, Any],
    prelabel: dict[str, Any],
    observation: dict[str, Any],
    reviewer: str,
    reviews_path: Path,
) -> None:
    classification = copy.deepcopy(prelabel["classification"])
    actions = candidate_actions(state, observation)
    labeled_actions = [item["action"] for item in classification["actions"]]
    if labeled_actions != actions:
        st.error("Pre-label action order does not match the full router-visible candidate pool.")
        st.stop()
    st.markdown("## Review every candidate action")
    st.caption("Terra proposal chỉ là điểm bắt đầu. Human reviewer phải kiểm tra đủ role pool và STOP.")
    edited_actions = []
    for index, item in enumerate(classification["actions"]):
        with st.expander(f"{index + 1:02d} · {item['action']} — {item['classification']}", expanded=False):
            chosen = st.selectbox(
                "Class",
                ACTION_CLASSES,
                index=ACTION_CLASSES.index(item["classification"]),
                key=f"class-{state['decision_id']}-{index}",
            )
            rationale = st.text_area(
                "Rationale grounded in router-visible evidence",
                value=item["rationale"],
                key=f"why-{state['decision_id']}-{index}",
                height=100,
            )
            edited_actions.append({"action": item["action"], "classification": chosen, "rationale": rationale})

    st.markdown("### State-level label")
    one, two, three = st.columns(3)
    with one:
        phase = st.selectbox(
            "State phase",
            STATE_PHASES,
            index=STATE_PHASES.index(classification["state_phase"]),
            key=f"phase-{state['decision_id']}",
        )
    with two:
        confidence = st.selectbox(
            "Confidence",
            ["low", "medium", "high"],
            index=["low", "medium", "high"].index(classification["confidence"]),
            key=f"confidence-{state['decision_id']}",
        )
    with three:
        terminal = st.checkbox(
            "Terminal valid",
            value=classification["terminal_valid"],
            key=f"terminal-{state['decision_id']}",
        )
    summary = st.text_area("Summary", value=classification["summary"], key=f"summary-{state['decision_id']}")
    correction = st.text_input("Correction category (optional)", key=f"correction-{state['decision_id']}")
    edited = {
        "actions": edited_actions,
        "terminal_valid": terminal,
        "state_phase": phase,
        "confidence": confidence,
        "summary": summary,
    }

    col1, col2, col3 = st.columns(3)
    action = None
    if col1.button("Accept Terra proposal", type="primary", use_container_width=True):
        action = "accepted"
        edited = classification
    if col2.button("Save corrections", use_container_width=True):
        action = "corrected"
    if col3.button("Send to adjudication", use_container_width=True):
        action = "rejected_for_adjudication"
    if action:
        try:
            record = make_review_record(
                state,
                prelabel,
                edited,
                reviewer_id=reviewer,
                decision=action,
                correction_category=correction or None,
            )
            append_review(reviews_path, record)
            st.success("Review appended. Lịch sử cũ không bị ghi đè.")
            st.rerun()
        except ValueError as exc:
            st.error(str(exc))


def render_evidence(observation: dict[str, Any]) -> None:
    st.markdown("## Public evidence available to the router")
    st.caption("Các summary giúp đọc nhanh; mở từng mục để xem exact public record, không cắt nội dung.")
    artifacts = artifact_items(observation)
    st.markdown(f"### Artifacts · {len(artifacts)}")
    if not artifacts:
        st.info("No artifact is present at this decision state.")
    for name, artifact in artifacts:
        status = artifact.get("status") or ("fresh" if artifact.get("fresh") else "available")
        with st.expander(f"{name} · {status}", expanded=False):
            if artifact.get("content") is not None:
                st.code(str(artifact["content"]), language=None)
            st.json(artifact, expanded=False)

    events = observation.get("recent_events") or []
    st.markdown(f"### Recent events · {len(events)}")
    if not events:
        st.info("No prior role execution is visible at this decision state.")
    for index, event in enumerate(events):
        agent = event.get("agent_id", "unknown agent")
        status = event.get("status", "unknown")
        with st.expander(f"{index:02d} · {agent} · {status}", expanded=False):
            st.json(event, expanded=False)

    st.markdown("### Full current_state")
    st.caption("Bao gồm execution contracts, checks, issues, requirements, versions và mọi field không đưa vào decision brief.")
    st.json(observation.get("current_state") or {}, expanded=False)


def render_audit(state: dict[str, Any], prelabel: dict[str, Any], observation: dict[str, Any]) -> None:
    st.markdown("## Exact audit payload")
    st.caption("Đây là exact `router_input_json` đã lưu từ Track B. Không field nào bị loại khỏi tab này.")
    st.download_button(
        "Download exact router input",
        data=state["router_input_json"].encode("utf-8"),
        file_name=f"{state['decision_id'].replace(':', '__')}.router-input.json",
        mime="application/json",
    )
    st.code(state["router_input_json"], language="json", line_numbers=False)
    with st.expander("Track A row metadata", expanded=False):
        st.json(state, expanded=False)
    with st.expander("Terra proposal and provenance", expanded=False):
        st.json(prelabel, expanded=False)
    with st.expander("Router visibility contract", expanded=False):
        st.json(
            {
                "benchmark_version": observation.get("benchmark_version"),
                "budget": observation.get("budget"),
                "constraints": observation.get("constraints"),
            },
            expanded=False,
        )


ROOT, LABELING_DIR = parse_args()
REVIEWS_PATH = ROOT / "reviews" / "human_reviews.jsonl"

st.set_page_config(page_title="RoleRouteBench Track A Review", layout="wide")
st.markdown(
    """
    <style>
      .block-container { max-width: 1180px; padding-top: 2rem; }
      .rrb-heading { display: grid; gap: .55rem; margin-bottom: 1.25rem; }
      .rrb-heading h1 { margin: 0; }
      .rrb-heading code { width: fit-content; background: rgba(127,127,127,.14); padding: .15rem .5rem; border-radius: .5rem; }
      .rrb-heading p { color: #8b8b94; margin: 0; }
      .rrb-chips { display: flex; flex-wrap: wrap; gap: .45rem; }
      .rrb-chip { background: rgba(127,127,127,.12); border-radius: .7rem; padding: .25rem .65rem; color: inherit; }
      .rrb-metrics { display: grid; grid-template-columns: repeat(3,minmax(0,1fr)); gap: .9rem; margin-bottom: 1.15rem; }
      .rrb-metric, .rrb-panel { background: rgba(127,127,127,.09); border-radius: 1.4rem; padding: 1.1rem; }
      .rrb-metric { display: grid; gap: .25rem; }
      .rrb-metric span, .rrb-metric small, .rrb-fact-label { color: #8b8b94; }
      .rrb-metric strong { font-size: 1.75rem; font-weight: 500; }
      .rrb-panel h2 { margin-top: 0; }
      .rrb-task { white-space: pre-wrap; font-size: 1.05rem; line-height: 1.7; }
      .rrb-facts { display: grid; gap: 1rem; }
      .rrb-fact-row { display: grid; grid-template-columns: minmax(160px,.35fr) minmax(0,1fr); gap: 1rem; }
      .rrb-fact-value { display: grid; gap: .35rem; overflow-wrap: anywhere; }
      .rrb-fact-value code { display: block; background: rgba(127,127,127,.16); color: #8b8b94; padding: .15rem .55rem; border-radius: .55rem; overflow-wrap: anywhere; }
      @media (max-width: 760px) {
        .rrb-metrics { grid-template-columns: 1fr; }
        .rrb-fact-row { grid-template-columns: 1fr; gap: .25rem; }
      }
    </style>
    """,
    unsafe_allow_html=True,
)

try:
    STATES, PRELABELS, COHORTS = load_review_data(ROOT, LABELING_DIR)
except ValueError as exc:
    st.error(str(exc))
    st.stop()

latest = latest_reviews(REVIEWS_PATH)
available = [decision_id for decision_id in sorted(PRELABELS) if decision_id in STATES]
if not available:
    st.warning("No successful Track A pre-label is available yet.")
    st.stop()

st.sidebar.title("Track A human review")
reviewer = st.sidebar.text_input("Reviewer ID", value="reviewer-01")
domain_options = sorted({STATES[decision_id]["domain"] for decision_id in available})
selected_domain = st.sidebar.selectbox("Domain", ["all", *domain_options], format_func=lambda value: "All domains" if value == "all" else domain_label(value))
status_filter = st.sidebar.selectbox("Review status", ["pending", "reviewed", "all"])
filtered = [
    decision_id
    for decision_id in available
    if (selected_domain == "all" or STATES[decision_id]["domain"] == selected_domain)
    and (
        status_filter == "all"
        or (status_filter == "pending" and decision_id not in latest)
        or (status_filter == "reviewed" and decision_id in latest)
    )
]
if not filtered:
    st.sidebar.info("No sample matches the current filters.")
    st.stop()

st.sidebar.metric("Pre-labels", len(available))
st.sidebar.metric("Reviewed", sum(decision_id in latest for decision_id in available))
st.sidebar.metric("Pending", sum(decision_id not in latest for decision_id in available))
decision_id = st.sidebar.selectbox(
    "Sample",
    filtered,
    format_func=lambda value: f"{STATES[value]['packet_id']} · {decision_step(value)}",
)
st.sidebar.caption(f"Cohort: {COHORTS[decision_id]}")
if decision_id in latest:
    st.sidebar.success(f"Latest review: {latest[decision_id]['decision']}")

state = STATES[decision_id]
prelabel = PRELABELS[decision_id]
observation = load_observation(state)
render_header(state, prelabel, observation)

cards = role_cards(observation)
tab_brief, tab_roles, tab_actions, tab_evidence, tab_audit = st.tabs(
    ["Decision brief", f"{len(cards)} role cards", "Action review", "Evidence", "Audit JSON"]
)
with tab_brief:
    render_decision_brief(state, observation)
with tab_roles:
    render_role_cards(observation)
with tab_actions:
    render_action_editor(state, prelabel, observation, reviewer, REVIEWS_PATH)
with tab_evidence:
    render_evidence(observation)
with tab_audit:
    render_audit(state, prelabel, observation)
