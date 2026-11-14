from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Any

import streamlit as st


PROJECT_DIR = Path(__file__).resolve().parents[1]
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

from review_app.annotation_store import (  # noqa: E402
    REVIEW_LABELS,
    AnnotationStore,
)
from review_app.data_loader import (  # noqa: E402
    RoutingBenchReviewDataset,
    public_leakage_keys,
)
from review_app.presentation import safe_text, wrapped_json_html  # noqa: E402
from review_app.workflow_graph import workflow_to_dot  # noqa: E402


DEFAULT_DATASET = PROJECT_DIR / "data" / "RoutingBench" / "v4_beta"
DEFAULT_DB = DEFAULT_DATASET / "reviews" / "routingbench_v4_reviews.sqlite"
LABEL_OPTIONS = ("Unlabeled", "Approved", "Needs revision", "Reject")
LABEL_TO_VALUE = {
    "Approved": "approved",
    "Needs revision": "needs_revision",
    "Reject": "reject",
}
VALUE_TO_LABEL = {value: label for label, value in LABEL_TO_VALUE.items()}


st.set_page_config(
    page_title="RoutingBench V4 Review",
    page_icon="🧭",
    layout="wide",
    initial_sidebar_state="expanded",
)
st.markdown(
    """
    <style>
      .block-container {padding-top: 1.6rem; padding-bottom: 4rem; max-width: 1540px;}
      .rb-head {margin: 0 0 1.4rem; padding: 1.35rem 1.5rem; border-radius: 20px;
        border: 1px solid #d7e2ea; background: linear-gradient(135deg,#ffffff 0%,#eef7f8 72%,#e3f1f1 100%);
        box-shadow: 0 12px 30px rgba(16,36,62,.06);}
      .rb-head h1 {font-size: 1.9rem; line-height: 1.2; margin: .22rem 0 .42rem; color:#10243e; letter-spacing:-.025em;}
      .rb-head p {color:#526274; margin:0; max-width:82ch; line-height:1.55;}
      .rb-kicker {font-size:.72rem; text-transform:uppercase; letter-spacing:.11em; color:#276d75; font-weight:800;}
      .rb-brand {padding:.8rem .35rem 1.1rem;}
      .rb-brand-mark {display:inline-grid; place-items:center; width:2.45rem; height:2.45rem;
        border-radius:13px; background:linear-gradient(145deg,#2f77d0,#15939a); color:white;
        font-size:1rem; font-weight:800; box-shadow:0 8px 20px rgba(21,147,154,.25);}
      .rb-brand-title {font-size:1.13rem; font-weight:780; color:#f8fafc; margin-top:.72rem;}
      .rb-brand-subtitle {font-size:.82rem; color:#aac0d4; margin-top:.18rem; line-height:1.4;}
      .rb-card {border:1px solid #dce4ee; border-radius:15px; padding:15px 16px;
        background:linear-gradient(145deg,#fff,#f8fafc); margin-bottom:10px;}
      .rb-card-title {font-size:1.03rem; font-weight:750; color:#10243e; margin:.16rem 0;}
      .rb-muted {font-size:.9rem; color:#5b6b7d; line-height:1.5;}
      .rb-json-wrap {box-sizing:border-box; width:100%; max-width:100%; margin:0; padding:1rem;
        border:1px solid #dfe7ef; border-radius:13px; background:#f7f9fc; color:#25364a;
        font:.82rem/1.56 ui-monospace,SFMono-Regular,Menlo,Monaco,Consolas,monospace;
        white-space:pre-wrap; overflow-wrap:anywhere; word-break:break-word; overflow-x:hidden;}
      .rb-pill {display:inline-block; padding:.25rem .55rem; border-radius:999px; margin:.12rem .2rem .12rem 0;
        background:#e6f3f5; color:#1b6269; font-size:.78rem; font-weight:700;}
      div[data-testid="stMetric"] {background:#fff; border:1px solid #dce4ee; padding:14px 16px;
        border-radius:15px; box-shadow:0 6px 18px rgba(15,23,42,.035);}
      div[data-testid="stExpander"] {border-color:#dce4ee; border-radius:14px; overflow:hidden; background:#fff;}
      div[data-testid="stVerticalBlockBorderWrapper"] {border-color:#dce4ee !important; border-radius:15px !important;}
      div[data-testid="stSidebar"] hr {border-color:#334155;}
      .stButton button {font-weight:700;}
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_resource(show_spinner="Đang lập chỉ mục dữ liệu V4-beta…")
def load_dataset(path: str) -> RoutingBenchReviewDataset:
    return RoutingBenchReviewDataset(path)


@st.cache_resource(show_spinner=False)
def load_store(path: str) -> AnnotationStore:
    return AnnotationStore(path)


def render_json(value: Any) -> None:
    st.html(wrapped_json_html(value))


def page_header(kicker: str, title: str, description: str) -> None:
    st.html(
        "<section class='rb-head'>"
        f"<div class='rb-kicker'>{safe_text(kicker)}</div>"
        f"<h1>{safe_text(title)}</h1>"
        f"<p>{safe_text(description)}</p>"
        "</section>"
    )


def render_role_card(card: dict[str, Any]) -> None:
    st.html(
        "<div class='rb-card'>"
        f"<div class='rb-kicker'>{safe_text(card.get('agent_id', ''))}</div>"
        f"<div class='rb-card-title'>{safe_text(card.get('role_name', 'Unknown role'))}</div>"
        f"<div class='rb-muted'>{safe_text(card.get('role_goal', ''))}</div>"
        "</div>"
    )
    left, right = st.columns(2)
    with left:
        st.markdown("**Use when**")
        for item in card.get("use_when", []):
            st.markdown(f"- {item}")
        st.markdown("**Supported inputs**")
        for item in card.get("supported_inputs", []):
            st.markdown(f"- {item}")
    with right:
        st.markdown("**Avoid when**")
        for item in card.get("avoid_when", []):
            st.markdown(f"- {item}")
        st.markdown("**Expected outputs**")
        for item in card.get("expected_outputs", []):
            st.markdown(f"- {item}")


def sidebar() -> tuple[RoutingBenchReviewDataset, AnnotationStore, str, str, str]:
    st.sidebar.html(
        "<div class='rb-brand'><div class='rb-brand-mark'>V4</div>"
        "<div class='rb-brand-title'>RoutingBench Review</div>"
        "<div class='rb-brand-subtitle'>Base workflows · Track C pairs<br>Two independent reviewers</div></div>"
    )
    reviewer_id = st.sidebar.text_input("Reviewer ID", placeholder="reviewer_01").strip()
    role = st.sidebar.selectbox("Vai trò", ("Independent reviewer", "Adjudicator"))
    if role == "Independent reviewer":
        pages = ("Tổng quan", "Base workflow", "Track C pair", "Export")
    else:
        pages = ("Tổng quan", "Adjudication", "Export")
    page = st.sidebar.radio("Không gian làm việc", pages)
    with st.sidebar.expander("Cấu hình dữ liệu"):
        dataset_path = st.text_input(
            "Dataset directory",
            value=os.environ.get("ROUTINGBENCH_DATA_DIR", str(DEFAULT_DATASET)),
        )
        db_path = st.text_input(
            "Review database",
            value=os.environ.get("ROUTINGBENCH_REVIEW_DB", str(DEFAULT_DB)),
        )
    try:
        dataset = load_dataset(str(Path(dataset_path).resolve()))
        store = load_store(str(Path(db_path).resolve()))
    except Exception as exc:
        st.error(f"Không thể mở gói review V4: {exc}")
        st.stop()
    st.sidebar.divider()
    st.sidebar.caption(f"Dataset `{dataset.version}`")
    st.sidebar.caption(
        f"{len(dataset.base_workflows)} base workflows · "
        f"{len(dataset.post_fault_pairs)} Track C pairs"
    )
    return dataset, store, reviewer_id, role, page


def require_reviewer(reviewer_id: str) -> None:
    if not reviewer_id:
        st.warning("Nhập Reviewer ID trong sidebar trước khi lưu.")
        st.stop()


def render_overview(
    dataset: RoutingBenchReviewDataset, store: AnnotationStore, reviewer_id: str
) -> None:
    page_header(
        "RoutingBench V4-beta",
        "Human review control room",
        "Theo dõi hai luồng bắt buộc: review 124 base workflow và 496 cặp Track C. "
        "Mỗi đối tượng cần hai reviewer độc lập; bất đồng được chuyển sang adjudication.",
    )
    if not reviewer_id:
        st.info("Nhập Reviewer ID để xem tiến độ cá nhân. Dữ liệu hiện vẫn ở chế độ chỉ đọc.")
    else:
        base = store.progress(
            dataset.version, reviewer_id, "base_workflow", len(dataset.base_workflows)
        )
        pairs = store.progress(
            dataset.version, reviewer_id, "track_c_pair", len(dataset.post_fault_pairs)
        )
        st.subheader("Tiến độ của reviewer")
        left, right = st.columns(2)
        with left:
            c1, c2, c3 = st.columns(3)
            c1.metric("Base submitted", f"{base['submitted']}/{base['total']}")
            c2.metric("Draft + submitted", base["saved"])
            c3.metric("Flagged", base["flagged"])
            st.progress(float(base["percent"]) / 100.0, text=f"Base workflows · {base['percent']:.1f}%")
        with right:
            c1, c2, c3 = st.columns(3)
            c1.metric("Track C submitted", f"{pairs['submitted']}/{pairs['total']}")
            c2.metric("Draft + submitted", pairs["saved"])
            c3.metric("Flagged", pairs["flagged"])
            st.progress(float(pairs["percent"]) / 100.0, text=f"Track C pairs · {pairs['percent']:.1f}%")

    st.subheader("Quy tắc review")
    cols = st.columns(3)
    rules = (
        (
            "01 · Independent",
            "Hai reviewer làm độc lập, không xem nhãn của nhau trong trang review.",
        ),
        (
            "02 · Evidence first",
            "Đối chiếu task, role cards, 5 counterfactual variants, public observations và provenance.",
        ),
        (
            "03 · Resolve",
            "Chỉ adjudicate khi nhãn khác nhau; sau khi sửa dữ liệu, hai reviewer xác nhận lại.",
        ),
    )
    for column, (title, body) in zip(cols, rules):
        column.html(
            "<div class='rb-card'>"
            f"<div class='rb-kicker'>{safe_text(title)}</div>"
            f"<div class='rb-muted'>{safe_text(body)}</div></div>"
        )

    meta = dataset.metadata
    st.subheader("Phạm vi dataset")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Task domains", len(meta.get("task_domains", [])))
    c2.metric("Counterfactual scenarios", meta.get("workflow_count", "—"))
    c3.metric("Decision states", meta.get("decision_state_count", "—"))
    c4.metric("Post-fault checkpoints", meta.get("post_fault_checkpoint_count", "—"))


def _filter_select(label: str, values: list[str], key: str) -> str | None:
    selected = st.selectbox(label, ["Tất cả", *values], key=key)
    return None if selected == "Tất cả" else selected


def base_workflow_picker(dataset: RoutingBenchReviewDataset) -> str:
    values = dataset.filter_values()
    columns = st.columns(4)
    with columns[0]:
        split = _filter_select("Split", values["splits"], "base-split")
    with columns[1]:
        domain = _filter_select("Domain", values["domains"], "base-domain")
    with columns[2]:
        pattern = _filter_select("Pattern", values["patterns"], "base-pattern")
    with columns[3]:
        source = _filter_select("Source", values["sources"], "base-source")
    ids = dataset.base_workflow_ids(
        split=split, domain=domain, pattern=pattern, source_kind=source
    )
    if not ids:
        st.info("Không có base workflow khớp bộ lọc.")
        st.stop()
    return st.selectbox("Base workflow", ids, key="base-workflow-id")


def track_c_picker(dataset: RoutingBenchReviewDataset) -> str:
    values = dataset.filter_values()
    columns = st.columns(4)
    with columns[0]:
        split = _filter_select("Split", values["splits"], "track-split")
    with columns[1]:
        domain = _filter_select("Domain", values["domains"], "track-domain")
    with columns[2]:
        fault_class = _filter_select(
            "Fault class", values["fault_classes"], "track-fault"
        )
    with columns[3]:
        pattern = _filter_select("Pattern", values["patterns"], "track-pattern")
    ids = dataset.track_c_pair_ids(
        split=split, domain=domain, fault_class=fault_class, pattern=pattern
    )
    if not ids:
        st.info("Không có Track C pair khớp bộ lọc.")
        st.stop()
    return st.selectbox("Track C pair", ids, key="track-pair-id")


def render_observation(payload: dict[str, Any]) -> None:
    state = payload.get("state", {})
    has_raw_outputs = "raw_executor_outputs" in payload
    raw_outputs = payload.get("raw_executor_outputs", [])
    st.markdown(f"**{payload.get('task', 'Task unavailable')}**")
    c1, c2, c3 = st.columns(3)
    c1.metric("Step", state.get("step_index", "—"))
    c2.metric("Calls left", payload.get("budget", {}).get("calls_remaining", "—"))
    c3.metric("Goal", state.get("goal_status", "—"))
    left, right = st.columns(2)
    with left:
        st.markdown("**Observable facts**")
        for item in state.get("facts", []):
            st.markdown(f"- {item}")
    with right:
        st.markdown("**Open requirements**")
        for item in state.get("open_requirements", []):
            st.markdown(f"- {item}")
    with st.expander("Event, memory và artifacts"):
        render_json(
            {
                "last_event": state.get("last_event"),
                "artifact_inventory": state.get("artifact_inventory", []),
                "recent_events": payload.get("recent_events", []),
                "execution_memory": payload.get("execution_memory", {}),
            }
        )
    if has_raw_outputs:
        st.markdown("#### Raw executor outputs")
        st.caption(
            "Output nguyên văn của executor theo từng bước; chỉ xuất hiện trong raw view."
        )
        if not raw_outputs:
            st.info("Chưa có executor output trước decision state này.")
        for index, output in enumerate(raw_outputs, start=1):
            step = output.get("step", index)
            agent_id = output.get("agent_id", "unknown agent")
            status = output.get("status", "unknown status")
            with st.expander(
                f"Step {step} · {agent_id} · {status}", expanded=True
            ):
                st.code(str(output.get("raw_output", "")), wrap_lines=True)


def render_checklist_form(
    *,
    store: AnnotationStore,
    dataset_version: str,
    reviewer_id: str,
    object_kind: str,
    object_id: str,
    checklist: list[str],
) -> None:
    st.divider()
    st.subheader("Checklist độc lập")
    st.caption(
        "Chọn một nhãn cho từng mục. `Needs revision` và `Reject` bắt buộc có ghi chú cụ thể."
    )
    existing = (
        store.get_review(dataset_version, reviewer_id, object_kind, object_id)
        if reviewer_id
        else None
    )
    if existing:
        st.info(
            f"Đã lưu `{existing['status']}` · revision {existing['revision']} · "
            f"{existing['reviewed_at']}"
        )
    labels: dict[str, str] = {}
    notes: dict[str, str] = {}
    saved_labels = (existing or {}).get("checklist_labels", {})
    saved_notes = (existing or {}).get("checklist_notes", {})
    for index, item in enumerate(checklist, start=1):
        with st.container(border=True):
            st.markdown(f"**{index:02d}. {item}**")
            current = VALUE_TO_LABEL.get(saved_labels.get(item), "Unlabeled")
            selected = st.radio(
                f"Nhãn mục {index}",
                LABEL_OPTIONS,
                index=LABEL_OPTIONS.index(current),
                horizontal=True,
                label_visibility="collapsed",
                key=f"review-label::{reviewer_id}::{object_kind}::{object_id}::{index}",
            )
            if selected != "Unlabeled":
                labels[item] = LABEL_TO_VALUE[selected]
            note = st.text_input(
                "Ghi chú cho mục này",
                value=saved_notes.get(item, ""),
                placeholder="Bắt buộc nếu cần sửa hoặc reject",
                key=f"review-note::{reviewer_id}::{object_kind}::{object_id}::{index}",
            )
            if note.strip():
                notes[item] = note.strip()
    summary = st.text_area(
        "Ghi chú tổng hợp",
        value=(existing or {}).get("summary_notes", ""),
        key=f"review-summary::{reviewer_id}::{object_kind}::{object_id}",
    )

    def save(status: str) -> None:
        require_reviewer(reviewer_id)
        store.save_review(
            dataset_version=dataset_version,
            reviewer_id=reviewer_id,
            object_kind=object_kind,
            object_id=object_id,
            checklist_items=checklist,
            checklist_labels=labels,
            checklist_notes=notes,
            summary_notes=summary,
            status=status,
        )

    left, right = st.columns(2)
    if left.button("Lưu draft", width="stretch", key=f"draft::{object_kind}::{object_id}"):
        try:
            save("draft")
            st.success("Đã lưu draft vào database V4.")
        except ValueError as exc:
            st.error(str(exc))
    if right.button(
        "Submit review", type="primary", width="stretch", key=f"submit::{object_kind}::{object_id}"
    ):
        try:
            save("submitted")
            st.success("Đã submit review độc lập.")
        except ValueError as exc:
            st.error(str(exc))


def render_base_workflow(
    dataset: RoutingBenchReviewDataset, store: AnnotationStore, reviewer_id: str
) -> None:
    page_header(
        "Independent review · Base workflow",
        "Review workflow family và 5 counterfactual variants",
        "Đối chiếu task contract, role cards, transitions, public observations, safe-action labels và provenance trước khi gán nhãn checklist.",
    )
    base_id = base_workflow_picker(dataset)
    bundle = dataset.base_bundle(base_id)
    base = bundle["base"]
    st.markdown(f"### {base.get('task', base_id)}")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Domain", base.get("task_domain", "—"))
    c2.metric("Split", base.get("split", "—"))
    c3.metric("Pattern", base.get("workflow_pattern", "—"))
    c4.metric("Variants", len(bundle["scenarios"]))
    st.caption(f"`{base_id}` · source `{base.get('source_kind', '—')}` · license `{base.get('source_license', '—')}`")

    tab_variants, tab_decisions, tab_roles, tab_source = st.tabs(
        ("Counterfactual variants", "Public state ↔ gold", "Role cards", "Provenance")
    )
    scenario_options = [str(row["scenario_id"]) for row in bundle["scenarios"]]
    scenario_lookup = {str(row["scenario_id"]): row for row in bundle["scenarios"]}

    with tab_variants:
        scenario_id = st.selectbox(
            "Variant", scenario_options, key=f"variant::{base_id}"
        )
        scenario = scenario_lookup[scenario_id]
        names = {
            agent_id: str(card.get("role_name", agent_id))
            for agent_id, card in dataset.agents.items()
        }
        st.graphviz_chart(workflow_to_dot(scenario, names), width="stretch")
        rows = []
        for transition in scenario.get("transitions", []):
            rows.append(
                {
                    "transition": transition.get("transition_id"),
                    "action": names.get(transition.get("action"), transition.get("action")),
                    "status": transition.get("status"),
                    "requires": ", ".join(
                        transition.get("requires_all", [])
                        + transition.get("requires_artifacts", [])
                    ),
                    "effect": transition.get("public_effect"),
                }
            )
        st.dataframe(rows, width="stretch", hide_index=True)
        with st.expander("Formal scenario JSON"):
            render_json(scenario)

    with tab_decisions:
        scenario_id = st.selectbox(
            "Variant", scenario_options, key=f"decision-variant::{base_id}"
        )
        decisions = dataset.decisions_for_scenario(scenario_id)
        if not decisions:
            st.warning("Variant này không có decision state.")
        else:
            decision_lookup = {
                str(item["private"]["decision_id"]): item for item in decisions
            }
            decision_id = st.selectbox(
                "Decision state",
                list(decision_lookup),
                key=f"decision-state::{base_id}::{scenario_id}",
            )
            item = decision_lookup[decision_id]
            views = list(item["public"].get("observations", {}))
            view = st.selectbox(
                "Observation view",
                views,
                index=views.index("structured") if "structured" in views else 0,
                key=f"observation-view::{base_id}::{decision_id}",
            )
            public_payload = item["public"]["observations"][view]
            leaked = public_leakage_keys(public_payload)
            if leaked:
                st.error(f"Public payload chứa key private: {', '.join(leaked)}")
            else:
                st.success("Public payload không chứa key evaluator-side bị cấm.")
            left, right = st.columns([1.25, 1], gap="large")
            with left:
                st.markdown("#### Router-visible observation")
                render_observation(public_payload)
            with right:
                st.markdown("#### Evaluator-side safe actions")
                private = item["private"]
                st.markdown(
                    " ".join(
                        f"<span class='rb-pill'>{safe_text(action)}</span>"
                        for action in private.get("valid_actions", [])
                    ),
                    unsafe_allow_html=True,
                )
                st.caption(
                    f"phase `{private.get('phase', '—')}` · terminal `{private.get('terminal_valid', False)}`"
                )
                rationale_rows = [
                    {
                        "action": action,
                        "status": rationale.get("status"),
                        "why": "; ".join(rationale.get("violated_preconditions", []))
                        or "; ".join(rationale.get("satisfied_preconditions", [])),
                    }
                    for action, rationale in private.get("action_rationales", {}).items()
                ]
                st.dataframe(rationale_rows, width="stretch", hide_index=True)

    with tab_roles:
        used_agents = sorted(
            {
                str(agent_id)
                for scenario in bundle["scenarios"]
                for agent_id in scenario.get("agent_pool", [])
            }
        )
        selected_agent = st.selectbox(
            "Agent", used_agents, key=f"role-card::{base_id}"
        )
        render_role_card(dataset.agents[selected_agent])

    with tab_source:
        st.markdown("#### Base workflow seed")
        render_json(base)
        st.markdown("#### Row-level provenance for five variants")
        for row in bundle["provenance"]:
            with st.expander(str(row.get("scenario_id", "provenance"))):
                render_json(row)

    render_checklist_form(
        store=store,
        dataset_version=dataset.version,
        reviewer_id=reviewer_id,
        object_kind="base_workflow",
        object_id=base_id,
        checklist=list(bundle["queue"].get("checklist", [])),
    )


def render_track_c(
    dataset: RoutingBenchReviewDataset, store: AnnotationStore, reviewer_id: str
) -> None:
    page_header(
        "Independent review · Track C",
        "Review matched post-fault / control checkpoint pair",
        "Kiểm tra matching contract, bằng chứng hậu thực thi quan sát được, leakage boundary và continuation tới goal cho từng cặp checkpoint.",
    )
    pair_id = track_c_picker(dataset)
    bundle = dataset.track_c_bundle(pair_id)
    manifest = bundle["manifest"]
    st.markdown(f"### {bundle['base_workflow'].get('task', pair_id)}")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Domain", manifest.get("task_domain", "—"))
    c2.metric("Split", manifest.get("split", "—"))
    c3.metric("Fault class", manifest.get("fault_class", "—"))
    c4.metric("Injection", manifest.get("injection_position", "—"))
    st.caption(f"`{pair_id}` · Nhãn fault/control dưới đây là audit chrome, không thuộc router payload.")

    match = dataset.pair_match_report(pair_id)
    st.subheader("Matched-pair contract")
    columns = st.columns(len(match))
    for column, (name, passed) in zip(columns, match.items()):
        column.metric(name.replace("same_", "").replace("_", " ").title(), "✓" if passed else "✕")
    if not all(match.values()):
        st.error("Cặp checkpoint vi phạm ít nhất một matching invariant.")

    tab_public, tab_private, tab_runtime = st.tabs(
        ("Router-visible comparison", "Private evaluation audit", "Replay continuation")
    )
    with tab_public:
        views = list(bundle["fault"]["public"].get("observations", {}))
        view = st.selectbox(
            "Observation view",
            views,
            index=views.index("structured") if "structured" in views else 0,
            key=f"track-view::{pair_id}",
        )
        fault_payload = bundle["fault"]["public"]["observations"][view]
        control_payload = bundle["control"]["public"]["observations"][view]
        leaks = {
            "fault": public_leakage_keys(fault_payload),
            "control": public_leakage_keys(control_payload),
        }
        if leaks["fault"] or leaks["control"]:
            st.error(f"Phát hiện public leakage: {leaks}")
        else:
            st.success("Cả hai router payload đều không chứa key evaluator-side bị cấm.")
        left, right = st.columns(2, gap="large")
        with left:
            st.markdown("#### Fault checkpoint")
            render_observation(fault_payload)
        with right:
            st.markdown("#### Matched control")
            render_observation(control_payload)

    with tab_private:
        left, right = st.columns(2, gap="large")
        with left:
            st.markdown("#### Fault label")
            render_json(bundle["fault"]["label"])
        with right:
            st.markdown("#### Control label")
            render_json(bundle["control"]["label"])

    with tab_runtime:
        left, right = st.columns(2, gap="large")
        with left:
            st.markdown("#### Fault replay state + history")
            render_json(bundle["fault"]["runtime"])
        with right:
            st.markdown("#### Control replay state + history")
            render_json(bundle["control"]["runtime"])

    render_checklist_form(
        store=store,
        dataset_version=dataset.version,
        reviewer_id=reviewer_id,
        object_kind="track_c_pair",
        object_id=pair_id,
        checklist=list(bundle["queue"].get("checklist", [])),
    )


def render_adjudication(
    dataset: RoutingBenchReviewDataset, store: AnnotationStore, adjudicator_id: str
) -> None:
    page_header(
        "Third-party resolution",
        "Adjudicate reviewer disagreements",
        "So sánh hai review đã submit, chỉ xử lý các checklist có nhãn khác nhau và ghi lại quyết định cuối cùng kèm rationale.",
    )
    require_reviewer(adjudicator_id)
    reviewers = [
        value for value in store.reviewer_ids(dataset.version) if value != adjudicator_id
    ]
    if len(reviewers) < 2:
        st.info("Cần ít nhất hai independent reviewer đã lưu annotation.")
        return
    left, right = st.columns(2)
    reviewer_a = left.selectbox("Reviewer A", reviewers, index=0)
    reviewer_b_options = [value for value in reviewers if value != reviewer_a]
    reviewer_b = right.selectbox("Reviewer B", reviewer_b_options, index=0)
    report = store.agreement_report(dataset.version, reviewer_a, reviewer_b)
    c1, c2, c3 = st.columns(3)
    c1.metric("Compared objects", report["compared"])
    c2.metric(
        "Exact agreement",
        "—" if report["exact_agreement"] is None else f"{100 * report['exact_agreement']:.1f}%",
    )
    c3.metric("Disagreements", report["disagreement_count"])

    disagreements = store.disagreements(dataset.version, reviewer_a, reviewer_b)
    adjudicated = store.adjudicated_ids(dataset.version)
    unresolved = [
        row
        for row in disagreements
        if (row["object_kind"], row["object_id"]) not in adjudicated
    ]
    if not unresolved:
        st.success("Không còn bất đồng chưa adjudicate cho cặp reviewer này.")
        return
    selected_key = st.selectbox(
        "Unresolved object",
        [f"{row['object_kind']} · {row['object_id']}" for row in unresolved],
    )
    selected = unresolved[
        [f"{row['object_kind']} · {row['object_id']}" for row in unresolved].index(selected_key)
    ]
    if selected["object_kind"] == "base_workflow":
        checklist = list(
            dataset.review_queue[selected["object_id"]].get("checklist", [])
        )
    else:
        checklist = list(
            dataset.post_fault_review_queue[selected["object_id"]].get("checklist", [])
        )

    final_labels: dict[str, str] = {}
    for index, item in enumerate(checklist, start=1):
        label_a = selected["reviewer_a_labels"].get(item)
        label_b = selected["reviewer_b_labels"].get(item)
        with st.container(border=True):
            st.markdown(f"**{index:02d}. {item}**")
            a, b = st.columns(2)
            a.markdown(f"**{reviewer_a}:** `{label_a}`")
            a.caption(selected["reviewer_a_notes"].get(item, "Không có ghi chú"))
            b.markdown(f"**{reviewer_b}:** `{label_b}`")
            b.caption(selected["reviewer_b_notes"].get(item, "Không có ghi chú"))
            if label_a == label_b and label_a in REVIEW_LABELS:
                options = [VALUE_TO_LABEL[label_a], *[
                    label for label in LABEL_OPTIONS[1:] if label != VALUE_TO_LABEL[label_a]
                ]]
            else:
                options = list(LABEL_OPTIONS)
            final = st.selectbox(
                "Final label",
                options,
                key=f"adj::{selected['object_kind']}::{selected['object_id']}::{index}",
            )
            if final != "Unlabeled":
                final_labels[item] = LABEL_TO_VALUE[final]
    rationale = st.text_area(
        "Adjudication rationale",
        placeholder="Nêu bằng chứng quyết định và thay đổi dataset cần thực hiện nếu có.",
    )
    if st.button("Lưu adjudication", type="primary", width="stretch"):
        try:
            store.save_adjudication(
                dataset_version=dataset.version,
                object_kind=selected["object_kind"],
                object_id=selected["object_id"],
                checklist_items=checklist,
                reviewer_a=reviewer_a,
                reviewer_b=reviewer_b,
                reviewer_a_labels=selected["reviewer_a_labels"],
                reviewer_b_labels=selected["reviewer_b_labels"],
                final_labels=final_labels,
                rationale=rationale,
                adjudicator_id=adjudicator_id,
            )
            st.success("Đã lưu adjudication và append audit event.")
        except ValueError as exc:
            st.error(str(exc))


def render_export(
    dataset: RoutingBenchReviewDataset, store: AnnotationStore, reviewer_id: str
) -> None:
    page_header(
        "Review artifacts",
        "Export auditable JSONL",
        "SQLite là working store; export tạo các artifact review V4 để validator và release pipeline sử dụng.",
    )
    require_reviewer(reviewer_id)
    output = st.text_input(
        "Output directory", value=str(dataset.root / "reviews" / "exports")
    )
    if st.button("Export review artifacts", type="primary"):
        counts = store.export(Path(output), dataset.version)
        st.success("Đã export review artifacts.")
        st.json(counts)


dataset, store, reviewer_id, role, page = sidebar()
if page == "Tổng quan":
    render_overview(dataset, store, reviewer_id)
elif page == "Base workflow":
    render_base_workflow(dataset, store, reviewer_id)
elif page == "Track C pair":
    render_track_c(dataset, store, reviewer_id)
elif page == "Adjudication":
    render_adjudication(dataset, store, reviewer_id)
else:
    render_export(dataset, store, reviewer_id)
