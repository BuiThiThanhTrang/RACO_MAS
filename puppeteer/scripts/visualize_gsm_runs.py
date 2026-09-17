"""Create a reproducible GSM-Hard evaluation report from sequential run folders.

Example
-------
python scripts/visualize_gsm_runs.py \
  --runs gsm-hard_train_seed42_20260826_230144 \
          gsm-hard_train_seed42_20260827_045805 \
  --log-group GSM-Hard \
  --label "Evolved W3D3 + Qwen3-Embedding-8B"

The script mirrors the chart families used by
``notebooks/gsm_hard_role_aware_report.ipynb`` while treating sequential run
folders as parts of one evaluation rather than independent experiments.
"""

from __future__ import annotations

import argparse
import ast
import hashlib
import html
import json
import math
import re
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
import seaborn as sns
import yaml


TIMESTAMP_FORMAT = "%Y-%m-%d-%H-%M-%S"
TIMESTAMP_LINE_RE = re.compile(r"^\[\d{4}-\d{2}-\d{2}\s")
RAW_TASK_RE = re.compile(r"'task': (\{.*?\}), 'workflow':", re.DOTALL)
VIS_RE_TEMPLATE = r"{name}\s*=\s*new vis\.DataSet\((\[.*?\])\);"
ROLE_ORDER = [
    "Planner / Decomposer",
    "General Reasoner",
    "Quantitative Reasoner",
    "Domain Reasoner",
    "Software Engineer",
    "Commonsense Generator",
    "Critic / Verifier",
    "Reflector",
    "Modifier / Repair",
    "Integrator / Concluder",
    "Python Tool Agent",
]
COLORS = ["#3366CC", "#E8710A", "#109618", "#7B1FA2", "#0097A7"]


@dataclass(frozen=True)
class RunSpec:
    folder: str
    path: Path
    data_start: int
    data_limit: int | None
    policy_mode: str
    dataset_mode: str
    seed: int | None
    max_width: int | None
    max_depth: int | None
    analyzer_model: str
    analyzer_dim: int | None
    analyzer_max_tokens: int
    personas_path: str | None
    segment: str = ""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Visualize sequential GSM-Hard runs and generate an HTML report."
    )
    parser.add_argument("--runs", nargs="+", required=True, help="Run folders inside runs/.")
    parser.add_argument(
        "--log-group",
        default="GSM-Hard",
        help="Folder inside logs/ containing per-item artifacts.",
    )
    parser.add_argument(
        "--label",
        default="Evolved GSM-Hard evaluation",
        help="Human-readable experiment label.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output directory. Default: <workspace>/output/<generated-name>.",
    )
    parser.add_argument("--rolling-window", type=int, default=40)
    parser.add_argument("--expected-final-size", type=int, default=909)
    parser.add_argument("--workers", type=int, default=8)
    return parser.parse_args()


def find_puppeteer_root(start: Path | None = None) -> Path:
    start = (start or Path(__file__).resolve()).resolve()
    for candidate in (start, *start.parents):
        if (candidate / "runs").is_dir() and (candidate / "logs").is_dir():
            return candidate
        if (candidate / "puppeteer" / "runs").is_dir():
            return candidate / "puppeteer"
    raise FileNotFoundError("Could not locate the Puppeteer project root.")


def normalize_text(value: Any) -> str:
    return re.sub(r"\s+", " ", str(value or "")).strip()


def safe_slug(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "_", value).strip("_").lower()


def read_jsonl(path: Path) -> list[dict]:
    if not path.is_file():
        return []
    rows: list[dict] = []
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        if not line.strip():
            continue
        try:
            value = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(value, dict):
            rows.append(value)
    return rows


def read_action_file(path: Path) -> list[dict]:
    text = path.read_text(encoding="utf-8", errors="replace").strip()
    if not text:
        return []
    try:
        value = json.loads(text)
        if isinstance(value, list):
            return [item for item in value if isinstance(item, dict)]
        return [value] if isinstance(value, dict) else []
    except json.JSONDecodeError:
        return read_jsonl(path)


def parse_vis_dataset(html_path: Path, name: str) -> list[dict]:
    if not html_path.is_file():
        return []
    text = html_path.read_text(encoding="utf-8", errors="replace")
    match = re.search(VIS_RE_TEMPLATE.format(name=re.escape(name)), text, re.DOTALL)
    if not match:
        return []
    try:
        value = json.loads(match.group(1))
    except json.JSONDecodeError:
        return []
    return value if isinstance(value, list) else []


def parse_folder_timestamp(name: str) -> datetime | None:
    try:
        return datetime.strptime(name, TIMESTAMP_FORMAT)
    except ValueError:
        return None


def parse_meta(meta_path: Path) -> tuple[str, dict]:
    if not meta_path.is_file():
        return "", {}
    lines = meta_path.read_text(encoding="utf-8", errors="replace").splitlines()
    runtime_config: dict = {}
    for line in lines:
        stripped = line.strip()
        if stripped.startswith("{") and "'task_analyzer'" in stripped:
            try:
                value = ast.literal_eval(stripped)
                if isinstance(value, dict):
                    runtime_config = value
                break
            except (SyntaxError, ValueError):
                pass
    question_lines: list[str] = []
    for index, line in enumerate(lines):
        if line.strip() != "Task:":
            continue
        for following in lines[index + 1 :]:
            if TIMESTAMP_LINE_RE.match(following):
                break
            question_lines.append(following)
        break
    return "\n".join(question_lines).strip(), runtime_config


def action_fingerprint(action: dict) -> str:
    payload = json.dumps(action, ensure_ascii=False, sort_keys=True, default=str)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def workflow_state(actions: list[dict]) -> str:
    states: list[str] = []
    for item in actions:
        action = item.get("action") or {}
        result = item.get("result") or {}
        states.append(
            f"{action.get('action')}({action.get('parameter')}) - "
            f"{result.get('step_data')} - {result.get('answer')}"
        )
    return "\n".join(states) if states else "None"


def lowess_numpy(x: pd.Series, y: pd.Series, frac: float = 0.22) -> tuple[np.ndarray, np.ndarray]:
    x_values = np.asarray(x, dtype=float)
    y_values = np.asarray(y, dtype=float)
    valid = np.isfinite(x_values) & np.isfinite(y_values)
    x_values, y_values = x_values[valid], y_values[valid]
    if len(x_values) < 3:
        return x_values, y_values
    order = np.argsort(x_values)
    x_values, y_values = x_values[order], y_values[order]
    neighbors = max(3, min(len(x_values), int(math.ceil(frac * len(x_values)))))
    fitted: list[float] = []
    for x0 in x_values:
        distance = np.abs(x_values - x0)
        bandwidth = np.partition(distance, neighbors - 1)[neighbors - 1]
        if bandwidth <= 0:
            fitted.append(float(np.mean(y_values[distance == 0])))
            continue
        u = np.clip(distance / bandwidth, 0, 1)
        weights = (1 - u**3) ** 3
        design = np.column_stack([np.ones_like(x_values), x_values - x0])
        root_weight = np.sqrt(weights)[:, None]
        beta, *_ = np.linalg.lstsq(
            design * root_weight, y_values * root_weight.ravel(), rcond=None
        )
        fitted.append(float(beta[0]))
    return x_values, np.asarray(fitted)


def wilson_interval(successes: int, total: int, z: float = 1.959963984540054) -> tuple[float, float]:
    if total <= 0:
        return math.nan, math.nan
    proportion = successes / total
    denominator = 1 + z * z / total
    center = (proportion + z * z / (2 * total)) / denominator
    half = z * math.sqrt(
        proportion * (1 - proportion) / total + z * z / (4 * total * total)
    ) / denominator
    return center - half, center + half


def load_run_specs(runs_root: Path, folders: list[str]) -> list[RunSpec]:
    specs: list[RunSpec] = []
    for folder in folders:
        run_path = runs_root / folder
        config_path = run_path / "resolved_config.yaml"
        if not config_path.is_file():
            raise FileNotFoundError(f"Missing resolved config: {config_path}")
        config = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
        dataset = config.get("dataset", {}) or {}
        policy = config.get("policy", {}) or {}
        global_config = config.get("global_config", {}) or {}
        graph = global_config.get("graph", {}) or {}
        analyzer = (global_config.get("task_analyzer", {}) or {}).get("primary", {}) or {}
        specs.append(
            RunSpec(
                folder=folder,
                path=run_path,
                data_start=int(dataset.get("data_start") or 0),
                data_limit=dataset.get("data_limit"),
                policy_mode=str(policy.get("policy_mode") or "unknown"),
                dataset_mode=str(dataset.get("mode") or "unknown"),
                seed=config.get("seed"),
                max_width=graph.get("max_width"),
                max_depth=graph.get("max_depth"),
                analyzer_model=str(analyzer.get("model") or "unknown"),
                analyzer_dim=analyzer.get("dim"),
                analyzer_max_tokens=int(analyzer.get("max_input_tokens") or 512),
                personas_path=config.get("personas_path"),
            )
        )
    specs.sort(key=lambda item: (item.data_start, item.folder))
    return [
        RunSpec(**{**spec.__dict__, "segment": f"Part {index + 1} · start {spec.data_start}"})
        for index, spec in enumerate(specs)
    ]


def load_results(specs: list[RunSpec]) -> tuple[pd.DataFrame, pd.DataFrame]:
    frames: list[pd.DataFrame] = []
    inventory: list[dict] = []
    for spec in specs:
        paths = [
            path
            for path in sorted((spec.path / "results").glob("*.jsonl"))
            if path.stat().st_size and "uncommitted_after" not in path.name
        ]
        rows: list[dict] = []
        for path in paths:
            for row in read_jsonl(path):
                row["source_result"] = str(path)
                rows.append(row)
        frame = pd.DataFrame(rows)
        if frame.empty:
            raise ValueError(f"No result rows found for {spec.folder}")
        if "original_index" not in frame:
            frame["original_index"] = frame.get("id")
        frame["original_index"] = pd.to_numeric(frame["original_index"], errors="coerce")
        frame["batch_index"] = pd.to_numeric(frame.get("batch_index"), errors="coerce")
        frame = frame.dropna(subset=["original_index", "batch_index"]).copy()
        frame["original_index"] = frame["original_index"].astype(int)
        frame["batch_index"] = frame["batch_index"].astype(int)
        frame["correct"] = frame["correct"].fillna(False).astype(bool)
        frame["source_run_folder"] = spec.folder
        frame["segment"] = spec.segment
        inventory.append(
            {
                "run_folder": spec.folder,
                "segment": spec.segment,
                "data_start": spec.data_start,
                "result_files": len(paths),
                "raw_result_rows": len(frame),
                "batch_min": int(frame.batch_index.min()),
                "batch_max": int(frame.batch_index.max()),
            }
        )
        frames.append(frame)
    combined = pd.concat(frames, ignore_index=True)
    combined = combined.sort_values(["batch_index", "source_run_folder"])
    combined = combined.drop_duplicates("original_index", keep="last")
    combined = combined.drop_duplicates("batch_index", keep="last").reset_index(drop=True)
    return combined, pd.DataFrame(inventory)


def build_question_lookup(dataset_path: Path) -> dict[str, int]:
    if not dataset_path.is_file():
        raise FileNotFoundError(f"Missing GSM-Hard dataset: {dataset_path}")
    dataset = pd.read_parquet(dataset_path).reset_index().rename(columns={"index": "original_index"})
    return {
        normalize_text("Solve this math problem and return the final numerical answer:\n" + row.input): int(
            row.original_index
        )
        for row in dataset[["original_index", "input"]].itertuples(index=False)
    }


def graph_payload(log_dir: Path, has_actions: bool) -> tuple[dict, list[dict]]:
    nodes = parse_vis_dataset(log_dir / "agent_graph.html", "nodes")
    edges = parse_vis_dataset(log_dir / "agent_graph.html", "edges")
    label_by_id = {
        node.get("id"): str(node.get("label") or "Unknown").split("\n", 1)[0] for node in nodes
    }
    edge_pairs = [
        (edge.get("from"), edge.get("to"))
        for edge in edges
        if edge.get("from") is not None and edge.get("to") is not None
    ]
    active_nodes = {value for pair in edge_pairs for value in pair}
    active_count = len(active_nodes) if active_nodes else int(has_actions)
    graph = nx.DiGraph()
    graph.add_nodes_from(active_nodes)
    graph.add_edges_from(edge_pairs)
    without_loops = nx.DiGraph((u, v) for u, v in graph.edges() if u != v)
    without_loops.add_nodes_from(graph.nodes())
    density = nx.density(without_loops) if without_loops.number_of_nodes() > 1 else 0.0
    try:
        cycles = list(nx.simple_cycles(graph, length_bound=4))
    except TypeError:
        cycles = [cycle for cycle in nx.simple_cycles(graph) if len(cycle) <= 4]
    cycle_counts = Counter(len(cycle) for cycle in cycles)
    transitions = [
        {
            "source_role": label_by_id.get(source, "Unknown"),
            "target_role": label_by_id.get(target, "Unknown"),
        }
        for source, target in edge_pairs
    ]
    metrics = {
        "graph_pool_nodes": len(nodes),
        "graph_active_nodes": active_count,
        "graph_edges": len(edge_pairs),
        "graph_unique_edges": graph.number_of_edges(),
        "edge_node_ratio": len(edge_pairs) / max(active_count, 1),
        "directed_density": density,
        **{f"cycles_len_{length}": cycle_counts.get(length, 0) for length in range(1, 5)},
    }
    return metrics, transitions


def identify_task(log_dir: Path, question: str, question_lookup: dict[str, int]) -> int | None:
    original_index = question_lookup.get(normalize_text(question)) if question else None
    if original_index is not None:
        return int(original_index)
    fallback = ""
    meta_path = log_dir / "meta.log"
    if meta_path.is_file():
        fallback += meta_path.read_text(encoding="utf-8", errors="replace")
    for path in sorted(log_dir.glob("path*.log")):
        fallback += "\n" + path.read_text(encoding="utf-8", errors="replace")
    for raw in reversed(RAW_TASK_RE.findall(fallback)):
        try:
            task = ast.literal_eval(raw)
        except (SyntaxError, ValueError):
            continue
        if isinstance(task, dict):
            value = task.get("original_index", task.get("id"))
            try:
                return int(value)
            except (TypeError, ValueError):
                continue
    return None


def parse_log_folder(
    log_dir: Path,
    question_lookup: dict[str, int],
    fallback_analyzer_max_tokens: int,
    fallback_max_depth: int,
) -> dict:
    question, runtime_config = parse_meta(log_dir / "meta.log")
    original_index = identify_task(log_dir, question, question_lookup)
    path_files = sorted(log_dir.glob("path_*.jsonl"))
    raw_actions: list[dict] = []
    paths: list[list[dict]] = []
    for path_file in path_files:
        actions = read_action_file(path_file)
        paths.append(actions)
        occurrences: Counter = Counter()
        for action_index, item in enumerate(actions, start=1):
            fingerprint = action_fingerprint(item)
            occurrences[fingerprint] += 1
            raw_actions.append(
                {
                    "log_folder": log_dir.name,
                    "path_file": path_file.name,
                    "action_index": action_index,
                    "agent": item.get("agent", "Unknown"),
                    "action": (item.get("action") or {}).get("action"),
                    "tokens": pd.to_numeric(item.get("tokens", 0), errors="coerce"),
                    "cost": pd.to_numeric(item.get("cost", 0), errors="coerce"),
                    "model_size": pd.to_numeric(item.get("model_size", np.nan), errors="coerce"),
                    "success": item.get("success"),
                    "action_fingerprint": fingerprint,
                    "fingerprint_occurrence": occurrences[fingerprint],
                }
            )
    raw_frame = pd.DataFrame(raw_actions)
    if raw_frame.empty:
        unique_frame = raw_frame.copy()
    else:
        unique_frame = (
            raw_frame.sort_values(["path_file", "action_index"])
            .drop_duplicates(["action_fingerprint", "fingerprint_occurrence"], keep="first")
            .reset_index(drop=True)
        )
        unique_frame["tokens"] = unique_frame.tokens.fillna(0).astype(int)
        unique_frame["cost"] = unique_frame.cost.fillna(0).astype(float)
    reward_rows = read_jsonl(log_dir / "trajectory_rewards.jsonl")
    skywork_statuses = Counter(str(row.get("skywork_status") or "missing") for row in reward_rows)
    graph_metrics, transitions = graph_payload(log_dir, has_actions=not unique_frame.empty)

    analyzer = (runtime_config.get("task_analyzer", {}) or {}).get("primary", {}) or {}
    graph_config = runtime_config.get("graph", {}) or {}
    max_input_tokens = int(analyzer.get("max_input_tokens") or fallback_analyzer_max_tokens)
    max_depth = int(graph_config.get("max_depth") or fallback_max_depth)
    requests: dict[str, dict] = {}
    initial_text = question or ""
    requests[hashlib.sha256(initial_text.encode("utf-8")).hexdigest()] = {
        "source_path": "initial",
        "prefix_length": 0,
        "input_text": initial_text,
    }
    for path_file, actions in zip(path_files, paths):
        last_prefix = len(actions) if len(actions) < max_depth else max(0, len(actions) - 1)
        for prefix_length in range(1, last_prefix + 1):
            prefix = actions[:prefix_length]
            key = hashlib.sha256(
                json.dumps(prefix, ensure_ascii=False, sort_keys=True, default=str).encode("utf-8")
            ).hexdigest()
            requests.setdefault(
                key,
                {
                    "source_path": path_file.name,
                    "prefix_length": prefix_length,
                    "input_text": initial_text + "\n" + workflow_state(prefix),
                },
            )
    analyzer_rows: list[dict] = []
    for request_index, request in enumerate(requests.values(), start=1):
        uncapped = max(1, math.ceil(len(request["input_text"]) / 4))
        analyzer_rows.append(
            {
                "log_folder": log_dir.name,
                "request_index": request_index,
                "source_path": request["source_path"],
                "prefix_length": request["prefix_length"],
                "characters": len(request["input_text"]),
                "estimated_uncapped_tokens": uncapped,
                "configured_max_input_tokens": max_input_tokens,
                "estimated_tokens_after_cap": min(uncapped, max_input_tokens),
                "would_be_truncated": uncapped > max_input_tokens,
            }
        )
    path_lengths = [len(actions) for actions in paths]
    summary = {
        "log_folder": log_dir.name,
        "log_timestamp": parse_folder_timestamp(log_dir.name),
        "original_index": original_index,
        "question": question,
        "path_count": max(len(path_files), len(reward_rows)),
        "agent_activations": len(unique_frame),
        "unique_roles": int(unique_frame.agent.nunique()) if not unique_frame.empty else 0,
        "mean_chain_length": float(np.mean(path_lengths)) if path_lengths else math.nan,
        "max_chain_length": max(path_lengths, default=math.nan),
        "total_tokens": int(unique_frame.tokens.sum()) if not unique_frame.empty else 0,
        "total_cost": float(unique_frame.cost.sum()) if not unique_frame.empty else 0.0,
        "raw_path_tokens": int(raw_frame.tokens.fillna(0).sum()) if not raw_frame.empty else 0,
        "copied_prefix_rows_removed": len(raw_frame) - len(unique_frame),
        "failed_actions": int((unique_frame.success != "Success").sum()) if not unique_frame.empty else 0,
        "embedding_calls_estimated": len(analyzer_rows),
        "embedding_tokens_estimated": int(
            sum(row["estimated_tokens_after_cap"] for row in analyzer_rows)
        ),
        "embedding_requests_truncated_estimated": int(
            sum(row["would_be_truncated"] for row in analyzer_rows)
        ),
        "skywork_statuses": dict(skywork_statuses),
        **graph_metrics,
    }
    return {
        "summary": summary,
        "actions": unique_frame.to_dict("records"),
        "analyzer_requests": analyzer_rows,
        "transitions": transitions,
    }


def load_logs(
    group_path: Path,
    question_lookup: dict[str, int],
    selected_ids: set[int],
    analyzer_max_tokens: int,
    max_depth: int,
    workers: int,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, dict]:
    if not group_path.is_dir():
        raise FileNotFoundError(f"Missing log group: {group_path}")
    folders = sorted(path for path in group_path.iterdir() if path.is_dir())

    def parse_one(path: Path) -> dict:
        return parse_log_folder(
            path,
            question_lookup,
            fallback_analyzer_max_tokens=analyzer_max_tokens,
            fallback_max_depth=max_depth,
        )

    with ThreadPoolExecutor(max_workers=max(1, min(workers, len(folders)))) as pool:
        parsed = list(pool.map(parse_one, folders))
    matched = [
        item
        for item in parsed
        if item["summary"]["original_index"] is not None
        and int(item["summary"]["original_index"]) in selected_ids
    ]
    matched.sort(key=lambda item: item["summary"]["log_timestamp"] or datetime.min)
    latest_by_id: dict[int, dict] = {}
    for item in matched:
        latest_by_id[int(item["summary"]["original_index"])] = item
    chosen = list(latest_by_id.values())
    summaries = pd.DataFrame(item["summary"] for item in chosen)
    actions: list[dict] = []
    analyzer_requests: list[dict] = []
    transitions: list[dict] = []
    for item in chosen:
        original_index = int(item["summary"]["original_index"])
        for row in item["actions"]:
            actions.append({"original_index": original_index, **row})
        for row in item["analyzer_requests"]:
            analyzer_requests.append({"original_index": original_index, **row})
        for row in item["transitions"]:
            transitions.append({"original_index": original_index, **row})
    diagnostics = {
        "log_folders_scanned": len(folders),
        "matched_log_rows_before_dedup": len(matched),
        "unique_matched_logs": len(chosen),
        "duplicate_matched_logs_removed": len(matched) - len(chosen),
    }
    return (
        summaries,
        pd.DataFrame(actions),
        pd.DataFrame(analyzer_requests),
        pd.DataFrame(transitions),
        diagnostics,
    )


def accuracy_summary(items: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict] = []
    for label, frame in list(items.groupby("segment", sort=False)) + [("Combined", items)]:
        total = len(frame)
        correct = int(frame.correct.sum())
        low, high = wilson_interval(correct, total)
        rows.append(
            {
                "label": label,
                "n": total,
                "correct": correct,
                "incorrect": total - correct,
                "accuracy": correct / total if total else math.nan,
                "ci_low": low,
                "ci_high": high,
            }
        )
    return pd.DataFrame(rows)


def efficiency_summary(items: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict] = []
    for label, frame in list(items.groupby("segment", sort=False)) + [("Combined", items)]:
        metric_frame = frame.dropna(subset=["total_tokens"])
        rows.append(
            {
                "label": label,
                "n": len(frame),
                "n_with_logs": len(metric_frame),
                "accuracy": float(frame.correct.mean()),
                "mean_tokens": float(metric_frame.total_tokens.mean()),
                "mean_internal_cost": float(metric_frame.total_cost.mean()),
                "mean_agent_activations": float(metric_frame.agent_activations.mean()),
                "mean_chain_length": float(metric_frame.mean_chain_length.mean()),
                "mean_path_count": float(metric_frame.path_count.mean()),
                "mean_unique_roles": float(metric_frame.unique_roles.mean()),
                "correct_per_million_tokens": float(
                    metric_frame.correct.sum() / max(metric_frame.total_tokens.sum(), 1) * 1_000_000
                ),
            }
        )
    return pd.DataFrame(rows)


def setup_plotting() -> None:
    sns.set_theme(style="whitegrid", context="talk")
    plt.rcParams.update(
        {
            "figure.dpi": 120,
            "savefig.dpi": 180,
            "axes.titleweight": "bold",
            "figure.facecolor": "white",
        }
    )


def save_figure(fig: plt.Figure, output_dir: Path, filename: str) -> str:
    path = output_dir / filename
    fig.savefig(path, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return filename


def make_figures(
    items: pd.DataFrame,
    actions: pd.DataFrame,
    analyzer_requests: pd.DataFrame,
    transitions: pd.DataFrame,
    accuracy: pd.DataFrame,
    efficiency: pd.DataFrame,
    output_dir: Path,
    label: str,
    rolling_window: int,
) -> list[tuple[str, str, str]]:
    figures: list[tuple[str, str, str]] = []
    palette = COLORS[: len(accuracy)]

    fig, axes = plt.subplots(1, 2, figsize=(18, 6.5), gridspec_kw={"width_ratios": [1.15, 1]})
    x = np.arange(len(accuracy))
    values = accuracy.accuracy.to_numpy()
    errors = np.vstack([values - accuracy.ci_low.to_numpy(), accuracy.ci_high.to_numpy() - values])
    axes[0].bar(x, values, yerr=errors, capsize=5, color=palette, alpha=0.9)
    axes[0].set_xticks(x, [f"{row.label}\n(n={row.n})" for row in accuracy.itertuples()], rotation=12, ha="right")
    axes[0].set_ylabel("Accuracy")
    axes[0].set_ylim(0, min(1.0, max(0.75, float(accuracy.ci_high.max()) + 0.08)))
    axes[0].set_title("GSM-Hard accuracy (Wilson 95% CI)")
    for index, value in enumerate(values):
        axes[0].text(index, value + 0.025, f"{value:.3f}", ha="center", fontsize=11)
    axes[1].barh(accuracy.label, accuracy.correct, color=palette, label="Correct")
    axes[1].barh(
        accuracy.label,
        accuracy.incorrect,
        left=accuracy.correct,
        color="#D9D9D9",
        label="Incorrect",
    )
    axes[1].set_xlabel("Samples")
    axes[1].set_title("Coverage and outcomes")
    axes[1].legend()
    for index, row in enumerate(accuracy.itertuples()):
        if row.correct:
            axes[1].text(row.correct / 2, index, str(row.correct), ha="center", va="center", color="white", fontweight="bold")
        if row.incorrect:
            axes[1].text(row.correct + row.incorrect / 2, index, str(row.incorrect), ha="center", va="center", color="#333333")
    fig.suptitle(label, y=1.02, fontsize=19, fontweight="bold")
    fig.tight_layout()
    name = save_figure(fig, output_dir, "01_accuracy_and_coverage.png")
    figures.append((name, "Accuracy and coverage", "Wilson confidence intervals, correct counts, and incorrect counts for each sequential part and their combined result."))

    ordered = items.sort_values("batch_index").copy()
    ordered["evaluation_step"] = np.arange(1, len(ordered) + 1)
    ordered["cumulative_accuracy"] = ordered.correct.astype(float).expanding().mean()
    min_periods = max(5, min(rolling_window // 4, len(ordered)))
    ordered["rolling_accuracy"] = ordered.correct.astype(float).rolling(rolling_window, min_periods=min_periods).mean()
    fig, axes = plt.subplots(1, 3, figsize=(21, 6.3))
    axes[0].scatter(ordered.evaluation_step, ordered.correct.astype(float), s=13, alpha=0.18, color="#9E9E9E", label="Item outcome")
    axes[0].plot(ordered.evaluation_step, ordered.cumulative_accuracy, color="#3366CC", linewidth=2.8, label="Cumulative accuracy")
    axes[0].plot(ordered.evaluation_step, ordered.rolling_accuracy, color="#E8710A", linewidth=2.2, label=f"Rolling accuracy ({rolling_window})")
    axes[0].set_ylim(-0.05, 1.05)
    axes[0].set_xlabel("Evaluation item order")
    axes[0].set_ylabel("Accuracy")
    axes[0].set_title("Outcome progression")
    axes[0].legend(fontsize=9)
    for column, axis, ylabel, title, color in [
        ("total_tokens", axes[1], "Exact agent tokens / task", "Token consumption", "#7B1FA2"),
        ("agent_activations", axes[2], "Agent activations / task", "Agent activation", "#109618"),
    ]:
        valid = ordered.dropna(subset=[column])
        axis.scatter(valid.evaluation_step, valid[column], s=15, alpha=0.25, color=color)
        gx, gy = lowess_numpy(valid.evaluation_step, valid[column])
        axis.plot(gx, gy, color=color, linewidth=3)
        axis.set_xlabel("Evaluation item order")
        axis.set_ylabel(ylabel)
        axis.set_title(title)
    for boundary in ordered.groupby("segment", sort=False).evaluation_step.max().iloc[:-1]:
        for axis in axes:
            axis.axvline(boundary + 0.5, color="#616161", linestyle="--", linewidth=1)
    fig.suptitle("Evaluation-order diagnostics (not a learning curve)", y=1.03, fontsize=19, fontweight="bold")
    fig.tight_layout()
    name = save_figure(fig, output_dir, "02_evaluation_dynamics.png")
    figures.append((name, "Evaluation dynamics", "Cumulative and rolling accuracy plus LOWESS trends for token use and agent activations. Dashed lines mark the run boundary."))

    panels = [
        ("mean_tokens", "Mean exact agent tokens / task", "Token consumption"),
        ("mean_internal_cost", "Mean internal cost / task", "Internal cost proxy"),
        ("mean_agent_activations", "Mean activations / task", "Agent activations"),
        ("mean_chain_length", "Mean nodes / path", "Reasoning-chain length"),
    ]
    fig, axes = plt.subplots(2, 2, figsize=(17, 12))
    for axis, (column, ylabel, title) in zip(axes.flat, panels):
        values = efficiency[column].to_numpy()
        axis.bar(efficiency.label, values, color=palette, alpha=0.9)
        axis.set_ylabel(ylabel)
        axis.set_title(title)
        axis.tick_params(axis="x", rotation=12)
        for index, value in enumerate(values):
            axis.text(index, value * 1.02 if value else 0.02, f"{value:,.1f}", ha="center", fontsize=10)
    fig.suptitle("Operational metrics across sequential evaluation parts", y=1.01, fontsize=19, fontweight="bold")
    fig.tight_layout()
    name = save_figure(fig, output_dir, "03_efficiency_summary.png")
    figures.append((name, "Efficiency summary", "Mean token volume, internal cost proxy, activations, and chain length for each part and the combined run."))

    ordered["rolling_tokens"] = ordered.total_tokens.rolling(rolling_window, min_periods=min_periods).mean()
    ordered["rolling_correct"] = ordered.correct.astype(float).rolling(rolling_window, min_periods=min_periods).sum()
    ordered["rolling_token_sum"] = ordered.total_tokens.rolling(rolling_window, min_periods=min_periods).sum()
    ordered["rolling_correct_per_million_tokens"] = ordered.rolling_correct / ordered.rolling_token_sum.clip(lower=1) * 1_000_000
    fig, axes = plt.subplots(1, 2, figsize=(17, 6.5))
    segment_efficiency = efficiency.loc[efficiency.label != "Combined"]
    axes[0].scatter(segment_efficiency.mean_tokens, segment_efficiency.accuracy, s=240, c=palette[: len(segment_efficiency)], edgecolor="black")
    for row in segment_efficiency.itertuples():
        axes[0].annotate(f"{row.label}\n{row.correct_per_million_tokens:.1f} correct/M tokens", (row.mean_tokens, row.accuracy), xytext=(10, 8), textcoords="offset points", fontsize=10)
    axes[0].set_xlabel("Mean exact agent tokens / task")
    axes[0].set_ylabel("Accuracy")
    axes[0].set_title("Performance-cost trade-off by part")
    axes[1].plot(ordered.evaluation_step, ordered.rolling_correct_per_million_tokens, color="#BF360C", linewidth=2.6)
    axes[1].set_xlabel("Evaluation item order")
    axes[1].set_ylabel(f"Correct / million tokens ({rolling_window}-item window)")
    axes[1].set_title("Rolling performance-cost ratio")
    fig.tight_layout()
    name = save_figure(fig, output_dir, "04_performance_cost_tradeoff.png")
    figures.append((name, "Performance-cost trade-off", "Per-part accuracy versus mean agent-token volume and the rolling number of correct answers per million exact agent tokens."))

    if not actions.empty:
        role_summary = (
            actions.groupby("agent", as_index=False)
            .agg(calls=("tokens", "size"), total_tokens=("tokens", "sum"), mean_tokens=("tokens", "mean"), samples_active=("original_index", "nunique"))
            .sort_values("calls", ascending=True)
        )
        fig, axes = plt.subplots(1, 2, figsize=(18, max(7, 0.55 * len(role_summary))))
        axes[0].barh(role_summary.agent, role_summary.calls, color="#3366CC")
        axes[0].set_xlabel("Unique activations")
        axes[0].set_title("Role activation frequency")
        axes[1].barh(role_summary.agent, role_summary.total_tokens, color="#7B1FA2")
        axes[1].set_xlabel("Exact agent tokens")
        axes[1].set_title("Token use by role")
        for axis, column in [(axes[0], "calls"), (axes[1], "total_tokens")]:
            for index, value in enumerate(role_summary[column]):
                axis.text(value, index, f"  {value:,.0f}", va="center", fontsize=9)
        fig.tight_layout()
        name = save_figure(fig, output_dir, "05_role_usage.png")
        figures.append((name, "Role usage", "Deduplicated agent activations and exact token consumption by semantic role."))

        order = actions.groupby("agent").tokens.sum().sort_values(ascending=False).index
        fig, ax = plt.subplots(figsize=(16, 7))
        sns.boxplot(data=actions, x="agent", y="tokens", order=order, color="#8AB4F8", showfliers=False, ax=ax)
        ax.set_xlabel("")
        ax.set_ylabel("Tokens / agent action")
        ax.set_title("Distribution of tokens per agent activation")
        ax.tick_params(axis="x", rotation=35)
        fig.tight_layout()
        name = save_figure(fig, output_dir, "06_agent_tokens_per_activation.png")
        figures.append((name, "Token distribution per activation", "Per-call token distributions; outliers are hidden in the box plot but retained in exported CSV files."))

    metric_items = items.dropna(subset=["mean_chain_length"]).copy()
    if not metric_items.empty:
        fig, axes = plt.subplots(1, 3, figsize=(20, 6.5))
        sns.histplot(data=metric_items, x="mean_chain_length", hue="correct", discrete=True, multiple="dodge", shrink=0.8, palette={True: "#109618", False: "#D9534F"}, ax=axes[0])
        axes[0].set_xlabel("Mean actions / path")
        axes[0].set_title("Chain length by outcome")
        path_counts = metric_items.path_count.value_counts().sort_index()
        axes[1].bar(path_counts.index.astype(str), path_counts.values, color="#3366CC")
        axes[1].set_xlabel("Paths / task")
        axes[1].set_ylabel("Tasks")
        axes[1].set_title("Exploration width realized")
        depth_counts = metric_items.max_chain_length.value_counts().sort_index()
        axes[2].bar(depth_counts.index.astype(int).astype(str), depth_counts.values, color="#E8710A")
        axes[2].set_xlabel("Maximum path depth")
        axes[2].set_ylabel("Tasks")
        axes[2].set_title("Reasoning depth realized")
        fig.tight_layout()
        name = save_figure(fig, output_dir, "07_path_structure.png")
        figures.append((name, "Path structure", "Realized exploration width, reasoning depth, and chain-length distribution split by correctness."))

    if not transitions.empty:
        matrix = pd.crosstab(transitions.source_role, transitions.target_role)
        role_order = [role for role in ROLE_ORDER if role in matrix.index or role in matrix.columns]
        matrix = matrix.reindex(index=role_order, columns=role_order, fill_value=0)
        fig, ax = plt.subplots(figsize=(14, 11))
        sns.heatmap(matrix, cmap="Blues", linewidths=0.3, annot=True, fmt="g", cbar_kws={"label": "Transition count"}, ax=ax)
        ax.set_xlabel("Next role")
        ax.set_ylabel("Current role")
        ax.set_title("Role-to-role transition heatmap")
        fig.tight_layout()
        name = save_figure(fig, output_dir, "08_role_transition_heatmap.png")
        figures.append((name, "Role transition heatmap", "Directed role transitions reconstructed from each saved agent graph."))

    if not analyzer_requests.empty:
        analyzer_by_item = analyzer_requests.groupby("original_index", as_index=False).agg(calls=("request_index", "size"), estimated_tokens=("estimated_tokens_after_cap", "sum"))
        fig, axes = plt.subplots(1, 2, figsize=(17, 6.3))
        sns.histplot(analyzer_by_item.calls, discrete=True, color="#E8710A", ax=axes[0])
        axes[0].set_xlabel("Reconstructed Task Analyzer calls / task")
        axes[0].set_title("Task Analyzer call distribution")
        sorted_tokens = analyzer_by_item.estimated_tokens.sort_values().reset_index(drop=True)
        axes[1].plot(np.arange(1, len(sorted_tokens) + 1), sorted_tokens.cumsum(), color="#BF360C", linewidth=2.5)
        axes[1].set_xlabel("Tasks sorted by estimated analyzer input tokens")
        axes[1].set_ylabel("Cumulative estimated input tokens")
        axes[1].set_title("Task Analyzer input-token estimate")
        fig.tight_layout()
        name = save_figure(fig, output_dir, "09_task_analyzer_estimates.png")
        figures.append((name, "Task Analyzer estimates", "Reconstructed embedding-call count and capped input-token estimates. Token counts use ceil(characters/4), not remote API usage."))

    topology = items.dropna(subset=["edge_node_ratio"]).sort_values("batch_index").copy()
    if len(topology) >= 8:
        quartile = max(1, len(topology) // 4)
        early = topology.head(quartile).copy()
        early["phase"] = "Early evaluation (first 25%)"
        late = topology.tail(quartile).copy()
        late["phase"] = "Late evaluation (last 25%)"
        phase_data = pd.concat([early, late], ignore_index=True)
        phase_order = ["Early evaluation (first 25%)", "Late evaluation (last 25%)"]
        phase_palette = {phase_order[0]: "#90CAF9", phase_order[1]: "#3366CC"}
        fig, axes = plt.subplots(1, 2, figsize=(18, 6.5))
        sns.histplot(data=phase_data, x="edge_node_ratio", hue="phase", hue_order=phase_order, palette=phase_palette, stat="density", common_norm=False, element="step", fill=False, linewidth=2.5, ax=axes[0])
        axes[0].set_xlabel("Edge-node ratio E/N_active")
        axes[0].set_title("Graph compaction distribution")
        cycle_columns = [f"cycles_len_{length}" for length in range(1, 5)]
        cycle_means = phase_data.groupby("phase")[cycle_columns].mean().reindex(phase_order)
        cycle_means.columns = ["Length 1", "Length 2", "Length 3", "Length 4"]
        cycle_means.T.plot(kind="bar", ax=axes[1], color=[phase_palette[value] for value in phase_order], width=0.8)
        axes[1].set_xlabel("Cycle length")
        axes[1].set_ylabel("Mean cycles / task")
        axes[1].set_title("Cyclicality by cycle length")
        axes[1].tick_params(axis="x", rotation=0)
        axes[1].legend(fontsize=9)
        fig.tight_layout()
        name = save_figure(fig, output_dir, "10_topology_early_vs_late.png")
        figures.append((name, "Topology: early versus late evaluation", "Graph compaction and cyclicality diagnostics. Because the policy is frozen, this is an order-based stability check rather than evidence of learning."))
    return figures


def dataframe_html(frame: pd.DataFrame, formats: dict[str, str] | None = None) -> str:
    display_frame = frame.copy()
    for column, formatter in (formats or {}).items():
        if column in display_frame:
            display_frame[column] = display_frame[column].map(
                lambda value: formatter.format(value) if pd.notna(value) else "—"
            )
    return display_frame.to_html(index=False, classes="data-table", border=0, escape=True)


def write_html_report(
    output_dir: Path,
    label: str,
    specs: list[RunSpec],
    items: pd.DataFrame,
    accuracy: pd.DataFrame,
    efficiency: pd.DataFrame,
    figures: list[tuple[str, str, str]],
    quality: dict,
    expected_final_size: int,
) -> Path:
    combined_accuracy = accuracy.loc[accuracy.label == "Combined"].iloc[0]
    combined_efficiency = efficiency.loc[efficiency.label == "Combined"].iloc[0]
    coverage = len(items) / expected_final_size if expected_final_size else math.nan
    top_role = quality.get("top_role_by_calls", "n/a")
    run_rows = pd.DataFrame(
        [
            {
                "run_folder": spec.folder,
                "part": spec.segment,
                "data_start": spec.data_start,
                "policy_mode": spec.policy_mode,
                "dataset_mode": spec.dataset_mode,
                "graph": f"W{spec.max_width}D{spec.max_depth}",
                "task_analyzer": spec.analyzer_model,
                "dimension": spec.analyzer_dim,
                "personas": spec.personas_path,
            }
            for spec in specs
        ]
    )
    figure_cards = "\n".join(
        f'<section class="figure-card"><h2>{html.escape(title)}</h2><p>{html.escape(caption)}</p><a href="{html.escape(filename)}"><img src="{html.escape(filename)}" alt="{html.escape(title)}"></a></section>'
        for filename, title, caption in figures
    )
    warnings: list[str] = []
    if len(items) < expected_final_size:
        warnings.append(
            f"Đây là report tạm thời trên {len(items)}/{expected_final_size} item; còn {expected_final_size - len(items)} item final chưa có kết quả trong hai run này."
        )
    if len({spec.policy_mode for spec in specs}) > 1:
        warnings.append("Các run không có cùng policy_mode.")
    if quality.get("missing_log_metrics", 0):
        warnings.append(f"Có {quality['missing_log_metrics']} item không nối được metric từ log.")
    if not warnings:
        warnings.append("Không phát hiện mismatch cấu hình hoặc thiếu log trong phạm vi đã chạy.")
    warning_html = "".join(f"<li>{html.escape(value)}</li>" for value in warnings)
    document = f"""<!doctype html>
<html lang="vi">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>{html.escape(label)} — GSM-Hard report</title>
<style>
:root {{ --ink:#172033; --muted:#667085; --blue:#3366cc; --paper:#f5f7fb; --card:#ffffff; }}
* {{ box-sizing:border-box; }} body {{ margin:0; background:var(--paper); color:var(--ink); font:15px/1.55 Inter,Segoe UI,Arial,sans-serif; }}
main {{ max-width:1240px; margin:0 auto; padding:42px 24px 70px; }}
h1 {{ font-size:34px; margin:0 0 8px; }} h2 {{ margin:0 0 8px; font-size:22px; }}
.subtitle {{ color:var(--muted); margin:0 0 28px; }}
.cards {{ display:grid; grid-template-columns:repeat(auto-fit,minmax(190px,1fr)); gap:14px; margin:22px 0; }}
.card,.figure-card,.table-card,.notice {{ background:var(--card); border:1px solid #e5e9f2; border-radius:16px; box-shadow:0 8px 24px rgba(31,42,68,.05); }}
.card {{ padding:18px; }} .card strong {{ display:block; font-size:28px; color:var(--blue); }} .card span {{ color:var(--muted); }}
.notice {{ padding:17px 22px; border-left:5px solid #e8710a; margin:22px 0; }}
.table-card {{ padding:20px; margin:22px 0; overflow-x:auto; }}
.data-table {{ border-collapse:collapse; width:100%; font-size:14px; }} .data-table th,.data-table td {{ padding:10px 12px; border-bottom:1px solid #e9edf4; text-align:left; }} .data-table th {{ background:#f8faff; }}
.figure-card {{ padding:22px; margin:22px 0; }} .figure-card p {{ color:var(--muted); margin-top:0; }} .figure-card img {{ width:100%; height:auto; border-radius:10px; border:1px solid #edf0f5; }}
code {{ background:#eef2f8; padding:2px 5px; border-radius:5px; }} footer {{ color:var(--muted); margin-top:32px; }}
</style>
</head>
<body><main>
<h1>{html.escape(label)}</h1>
<p class="subtitle">Hai run liên tiếp được ghép theo <code>batch_index</code>. Chế độ evolved được xem là evaluation với policy đóng băng.</p>
<div class="cards">
  <div class="card"><strong>{combined_accuracy.accuracy:.3f}</strong><span>Combined accuracy</span></div>
  <div class="card"><strong>{int(combined_accuracy.correct)}/{int(combined_accuracy.n)}</strong><span>Correct / evaluated</span></div>
  <div class="card"><strong>{coverage:.1%}</strong><span>Final split coverage</span></div>
  <div class="card"><strong>{combined_efficiency.mean_tokens:,.0f}</strong><span>Mean exact agent tokens / task</span></div>
  <div class="card"><strong>{combined_efficiency.mean_agent_activations:.2f}</strong><span>Mean activations / task</span></div>
  <div class="card"><strong>{html.escape(str(top_role))}</strong><span>Most activated role</span></div>
</div>
<div class="notice"><strong>Quality notes</strong><ul>{warning_html}</ul></div>
<section class="table-card"><h2>Run configuration</h2>{dataframe_html(run_rows)}</section>
<section class="table-card"><h2>Accuracy summary</h2>{dataframe_html(accuracy, {'accuracy':'{:.3f}','ci_low':'{:.3f}','ci_high':'{:.3f}'})}</section>
<section class="table-card"><h2>Efficiency summary</h2>{dataframe_html(efficiency, {'accuracy':'{:.3f}','mean_tokens':'{:,.1f}','mean_internal_cost':'{:,.1f}','mean_agent_activations':'{:.2f}','mean_chain_length':'{:.2f}','mean_path_count':'{:.2f}','mean_unique_roles':'{:.2f}','correct_per_million_tokens':'{:.2f}'})}</section>
{figure_cards}
<footer>Generated by <code>scripts/visualize_gsm_runs.py</code>. CSV files beside this report contain the traceable item-level data and quality checks.</footer>
</main></body></html>"""
    report_path = output_dir / "report.html"
    report_path.write_text(document, encoding="utf-8")
    return report_path


def main() -> None:
    args = parse_args()
    puppeteer_root = find_puppeteer_root()
    workspace_root = puppeteer_root.parent
    runs_root = puppeteer_root / "runs"
    logs_root = puppeteer_root / "logs"
    dataset_path = puppeteer_root / "data" / "GSM-Hard" / "test.parquet"
    specs = load_run_specs(runs_root, args.runs)
    generated_name = safe_slug(args.label) + "_" + datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = (args.output or (workspace_root / "output" / generated_name)).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    results, run_inventory = load_results(specs)
    question_lookup = build_question_lookup(dataset_path)
    log_summaries, actions, analyzer_requests, transitions, log_diagnostics = load_logs(
        logs_root / args.log_group,
        question_lookup,
        set(results.original_index.astype(int)),
        analyzer_max_tokens=specs[0].analyzer_max_tokens,
        max_depth=int(specs[0].max_depth or 3),
        workers=args.workers,
    )
    items = results.merge(log_summaries, on="original_index", how="left", validate="one_to_one")
    segment_by_id = items.set_index("original_index").segment.to_dict()
    if not actions.empty:
        actions["segment"] = actions.original_index.map(segment_by_id)
        actions["tokens"] = pd.to_numeric(actions.tokens, errors="coerce").fillna(0).astype(int)
    if not analyzer_requests.empty:
        analyzer_requests["segment"] = analyzer_requests.original_index.map(segment_by_id)
    if not transitions.empty:
        transitions["segment"] = transitions.original_index.map(segment_by_id)

    accuracy = accuracy_summary(items)
    efficiency = efficiency_summary(items)
    setup_plotting()
    figures = make_figures(
        items,
        actions,
        analyzer_requests,
        transitions,
        accuracy,
        efficiency,
        output_dir,
        args.label,
        max(5, args.rolling_window),
    )

    batch_values = sorted(items.batch_index.astype(int).tolist())
    expected_batch = list(range(min(batch_values), max(batch_values) + 1)) if batch_values else []
    missing_batch = sorted(set(expected_batch) - set(batch_values))
    role_summary = (
        actions.groupby("agent", as_index=False)
        .agg(calls=("tokens", "size"), total_tokens=("tokens", "sum"), mean_tokens=("tokens", "mean"), samples_active=("original_index", "nunique"))
        .sort_values("calls", ascending=False)
        if not actions.empty
        else pd.DataFrame(columns=["agent", "calls", "total_tokens", "mean_tokens", "samples_active"])
    )
    transition_summary = (
        transitions.groupby(["source_role", "target_role"], as_index=False).size().rename(columns={"size": "count"}).sort_values("count", ascending=False)
        if not transitions.empty
        else pd.DataFrame(columns=["source_role", "target_role", "count"])
    )
    quality = {
        **log_diagnostics,
        "result_rows": len(items),
        "unique_original_indices": int(items.original_index.nunique()),
        "batch_min": int(items.batch_index.min()),
        "batch_max": int(items.batch_index.max()),
        "missing_batch_indices": missing_batch,
        "missing_log_metrics": int(items.total_tokens.isna().sum()),
        "copied_prefix_rows_removed": int(items.copied_prefix_rows_removed.fillna(0).sum()),
        "exact_agent_tokens": int(actions.tokens.sum()) if not actions.empty else 0,
        "estimated_task_analyzer_calls": len(analyzer_requests),
        "estimated_task_analyzer_input_tokens": int(analyzer_requests.estimated_tokens_after_cap.sum()) if not analyzer_requests.empty else 0,
        "top_role_by_calls": role_summary.iloc[0].agent if len(role_summary) else None,
        "policy_modes": sorted({spec.policy_mode for spec in specs}),
        "dataset_modes": sorted({spec.dataset_mode for spec in specs}),
        "graph_settings": sorted({f"W{spec.max_width}D{spec.max_depth}" for spec in specs}),
        "task_analyzer_models": sorted({spec.analyzer_model for spec in specs}),
    }

    items.to_csv(output_dir / "item_metrics.csv", index=False)
    actions.to_csv(output_dir / "agent_actions_deduplicated.csv", index=False)
    analyzer_requests.to_csv(output_dir / "task_analyzer_request_estimates.csv", index=False)
    transitions.to_csv(output_dir / "role_transitions.csv", index=False)
    role_summary.to_csv(output_dir / "role_usage_summary.csv", index=False)
    transition_summary.to_csv(output_dir / "role_transition_summary.csv", index=False)
    accuracy.to_csv(output_dir / "run_accuracy.csv", index=False)
    efficiency.to_csv(output_dir / "run_efficiency.csv", index=False)
    run_inventory.to_csv(output_dir / "run_inventory.csv", index=False)
    (output_dir / "quality_checks.json").write_text(
        json.dumps(quality, ensure_ascii=False, indent=2, default=str), encoding="utf-8"
    )
    report_path = write_html_report(
        output_dir,
        args.label,
        specs,
        items,
        accuracy,
        efficiency,
        figures,
        quality,
        args.expected_final_size,
    )
    print(f"Report: {report_path}")
    print(f"Items: {len(items)}; accuracy: {items.correct.mean():.4f}")
    print(f"Log coverage: {len(items) - quality['missing_log_metrics']}/{len(items)}")
    print(f"Figures: {len(figures)}")


if __name__ == "__main__":
    main()
