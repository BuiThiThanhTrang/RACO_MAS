import json
from pathlib import Path


OUT = Path(r"D:\SideProject\ChatDev\puppeteer\notebooks\log_token_usage_by_agent_and_bge.ipynb")


def src(text):
    return text.strip("\n").splitlines(keepends=True)


def markdown(cell_id, text):
    return {"cell_type": "markdown", "id": cell_id, "metadata": {}, "source": src(text)}


def code(cell_id, text):
    return {
        "cell_type": "code",
        "execution_count": None,
        "id": cell_id,
        "metadata": {},
        "outputs": [],
        "source": src(text),
    }


cells = [
    markdown("title", r'''
# Token usage theo agent và BGE trong một log folder

Notebook này đọc **một group folder** trong `puppeteer/logs`, ví dụ `GSM-Hard-24-08-evolved`, rồi:

- cộng token chính xác của từng agent từ các action trong `path_*.jsonl`;
- loại các action prefix bị copy khi reasoning path split;
- tái dựng input của từng lần gọi BGE Task Analyzer từ task và workflow prefix;
- thống kê theo agent, sample và toàn bộ log group;
- xuất bảng CSV và biểu đồ PNG.

> **Giới hạn dữ liệu BGE:** `TaskAnalyzerRepresentation` hiện bỏ qua `response.usage`, vì vậy log cũ không chứa số token BGE chính xác. Notebook ưu tiên tokenizer BGE cục bộ; nếu tokenizer chưa có trong cache, nó dùng ước lượng `ceil(characters / 4)`. Cả số token trước giới hạn và sau `max_input_tokens` đều được giữ để không che giấu giả định này.
'''),
    code("configuration", r'''
from __future__ import annotations

import ast
import hashlib
import json
import math
import re
import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

try:
    from IPython.display import display
except ImportError:
    display = print

warnings.filterwarnings("ignore", category=FutureWarning)
sns.set_theme(style="whitegrid", context="talk")
plt.rcParams.update({"figure.dpi": 120, "savefig.dpi": 180, "axes.titleweight": "bold"})


def find_project_root(start: Path | None = None) -> Path:
    start = (start or Path.cwd()).resolve()
    for candidate in (start, *start.parents):
        if (candidate / "puppeteer" / "logs").is_dir():
            return candidate
    raise FileNotFoundError("Không tìm thấy project root chứa puppeteer/logs")


PROJECT_ROOT = find_project_root()
LOGS_ROOT = PROJECT_ROOT / "puppeteer" / "logs"

# Có thể điền tên folder bên trong puppeteer/logs hoặc một absolute path.
LOG_FOLDER = "GSM-Hard-24-08-evolved"

# Nếu đã tải tokenizer về máy, có thể đặt path local ở đây.
# None = dùng model name đọc từ config của log và chỉ tìm trong local cache.
BGE_TOKENIZER_PATH = None
BGE_LOCAL_FILES_ONLY = True

# Số sample hiển thị trong stacked chart; CSV vẫn chứa toàn bộ sample.
TOP_SAMPLES_TO_PLOT = 30

candidate = Path(LOG_FOLDER)
LOG_GROUP_PATH = candidate if candidate.is_absolute() else LOGS_ROOT / candidate
if not LOG_GROUP_PATH.is_dir():
    raise FileNotFoundError(f"Log folder không tồn tại: {LOG_GROUP_PATH}")

safe_group_name = re.sub(r"[^A-Za-z0-9._-]+", "_", LOG_GROUP_PATH.name)
REPORT_DIR = PROJECT_ROOT / "output" / "token_usage" / safe_group_name
REPORT_DIR.mkdir(parents=True, exist_ok=True)

print(f"Log group: {LOG_GROUP_PATH}")
print(f"Output:    {REPORT_DIR}")
'''),
    markdown("definitions", r'''
## 1. Parser và định nghĩa

`tokens` trong `path_*.jsonl` là tổng token mà action của agent đã dùng, đúng với `Action.set_cost(tokens=...)` trong code. Với BGE, mỗi lần `policy.forward()` tạo một embedding cho task cộng workflow hiện tại. Các workflow prefix giống nhau giữa các path split chỉ được tính một lần.
'''),
    code("parsers", r'''
TIMESTAMP_PATTERN = re.compile(r"^\[\d{4}-\d{2}-\d{2}\s")


def read_action_file(path: Path) -> list[dict]:
    text = path.read_text(encoding="utf-8", errors="replace").strip()
    if not text:
        return []
    try:
        value = json.loads(text)
        return value if isinstance(value, list) else [value]
    except json.JSONDecodeError:
        rows = []
        for line in text.splitlines():
            line = line.strip()
            if line:
                rows.append(json.loads(line))
        return rows


def parse_meta(meta_path: Path) -> tuple[str, dict]:
    if not meta_path.is_file():
        return "", {}
    lines = meta_path.read_text(encoding="utf-8", errors="replace").splitlines()
    runtime_config = {}
    for line in lines:
        stripped = line.strip()
        if stripped.startswith("{") and "'task_analyzer'" in stripped:
            try:
                runtime_config = ast.literal_eval(stripped)
                break
            except (SyntaxError, ValueError):
                pass

    question_lines = []
    for i, line in enumerate(lines):
        if line.strip() != "Task:":
            continue
        for following in lines[i + 1:]:
            if TIMESTAMP_PATTERN.match(following):
                break
            question_lines.append(following)
        break
    return "\n".join(question_lines).strip(), runtime_config


def action_fingerprint(action: dict) -> str:
    payload = json.dumps(action, ensure_ascii=False, sort_keys=True, default=str)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def workflow_state(actions: list[dict]) -> str:
    states = []
    for item in actions:
        action = item.get("action") or {}
        result = item.get("result") or {}
        states.append(
            f"{action.get('action')}({action.get('parameter')}) - "
            f"{result.get('step_data')} - {result.get('answer')}"
        )
    return "\n".join(states) if states else "None"


def task_analyzer_config(runtime_config: dict) -> dict:
    analyzer = runtime_config.get("task_analyzer", {}) or {}
    return dict(analyzer.get("primary", {}) or {})


def build_token_counter(model_name: str, local_path: str | None = None, local_only: bool = True):
    tokenizer_source = local_path or model_name
    try:
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained(
            tokenizer_source,
            local_files_only=local_only,
            trust_remote_code=False,
            use_fast=True,
        )

        def count(text: str) -> int:
            return len(tokenizer.encode(text, add_special_tokens=True, truncation=False))

        return count, f"tokenizer:{tokenizer_source}"
    except Exception as exc:
        print(f"Không tải được tokenizer BGE cục bộ ({type(exc).__name__}); dùng ceil(characters/4).")

        def count(text: str) -> int:
            return max(1, math.ceil(len(text) / 4))

        return count, "heuristic:ceil(characters/4)"


def save_figure(fig, filename: str):
    path = REPORT_DIR / filename
    fig.savefig(path, bbox_inches="tight", facecolor="white")
    print(f"Saved: {path.name}")
'''),
    markdown("agent-heading", r'''
## 2. Token chính xác theo agent

Mỗi row trong `agent_actions` là một action thực sự. Fingerprint kèm occurrence index giúp loại prefix được copy sang path mới nhưng vẫn giữ các lần gọi lặp hợp lệ trong cùng một path.
'''),
    code("load-agent-actions", r'''
sample_dirs = sorted(path for path in LOG_GROUP_PATH.iterdir() if path.is_dir())
raw_action_rows = []
sample_metadata = []

for sample_dir in sample_dirs:
    question, runtime_config = parse_meta(sample_dir / "meta.log")
    analyzer_cfg = task_analyzer_config(runtime_config)
    graph_cfg = runtime_config.get("graph", {}) or {}
    sample_metadata.append({
        "sample_folder": sample_dir.name,
        "question": question,
        "bge_model": analyzer_cfg.get("model", "BAAI/bge-large-en-v1.5"),
        "bge_max_input_tokens": int(analyzer_cfg.get("max_input_tokens", 512)),
        "max_depth": int(graph_cfg.get("max_depth", 4)),
        "path_files": len(list(sample_dir.glob("path_*.jsonl"))),
    })

    for path_file in sorted(sample_dir.glob("path_*.jsonl")):
        occurrence_by_fingerprint = {}
        for action_index, item in enumerate(read_action_file(path_file), start=1):
            fingerprint = action_fingerprint(item)
            occurrence_by_fingerprint[fingerprint] = occurrence_by_fingerprint.get(fingerprint, 0) + 1
            raw_action_rows.append({
                "sample_folder": sample_dir.name,
                "path_file": path_file.name,
                "action_index": action_index,
                "agent": item.get("agent", "Unknown agent"),
                "action": (item.get("action") or {}).get("action"),
                "tokens": pd.to_numeric(item.get("tokens", 0), errors="coerce"),
                "cost": pd.to_numeric(item.get("cost", 0), errors="coerce"),
                "model_size": pd.to_numeric(item.get("model_size", np.nan), errors="coerce"),
                "success": item.get("success"),
                "action_fingerprint": fingerprint,
                "fingerprint_occurrence": occurrence_by_fingerprint[fingerprint],
            })

metadata = pd.DataFrame(sample_metadata)
raw_actions = pd.DataFrame(raw_action_rows)
if raw_actions.empty:
    raise ValueError(f"Không tìm thấy action nào trong {LOG_GROUP_PATH}")

agent_actions = (
    raw_actions
    .sort_values(["sample_folder", "path_file", "action_index"])
    .drop_duplicates(["sample_folder", "action_fingerprint", "fingerprint_occurrence"], keep="first")
    .reset_index(drop=True)
)
agent_actions["tokens"] = agent_actions.tokens.fillna(0).astype(int)
copied_prefix_rows_removed = len(raw_actions) - len(agent_actions)

agent_summary = (
    agent_actions.groupby("agent", as_index=False)
    .agg(
        total_tokens=("tokens", "sum"),
        calls=("tokens", "size"),
        mean_tokens_per_call=("tokens", "mean"),
        median_tokens_per_call=("tokens", "median"),
        p95_tokens_per_call=("tokens", lambda values: values.quantile(0.95)),
        max_tokens_per_call=("tokens", "max"),
        total_internal_cost=("cost", "sum"),
        samples_active=("sample_folder", "nunique"),
    )
    .sort_values("total_tokens", ascending=False)
    .reset_index(drop=True)
)
agent_summary["share_of_agent_tokens"] = agent_summary.total_tokens / agent_summary.total_tokens.sum()

print(f"Sample folders: {len(sample_dirs):,}")
print(f"Unique agent actions: {len(agent_actions):,}")
print(f"Copied-prefix rows removed: {copied_prefix_rows_removed:,}")
display(agent_summary.style.format({
    "total_tokens": "{:,.0f}",
    "mean_tokens_per_call": "{:,.1f}",
    "median_tokens_per_call": "{:,.1f}",
    "p95_tokens_per_call": "{:,.1f}",
    "total_internal_cost": "{:,.0f}",
    "share_of_agent_tokens": "{:.1%}",
}))
'''),
    markdown("bge-heading", r'''
## 3. Tái dựng token input của BGE

Một request ban đầu chỉ chứa task. Sau mỗi action, policy gọi BGE với task cộng trạng thái workflow; nếu path chạm `max_depth`, code kết thúc trước lần gọi tiếp theo. Workflow prefix giống nhau giữa các branch được khử trùng lặp.
'''),
    code("reconstruct-bge", r'''
default_model = metadata.bge_model.dropna().mode().iloc[0] if len(metadata) else "BAAI/bge-large-en-v1.5"
count_bge_tokens, bge_count_method = build_token_counter(
    model_name=default_model,
    local_path=BGE_TOKENIZER_PATH,
    local_only=BGE_LOCAL_FILES_ONLY,
)

bge_rows = []
for sample_dir in sample_dirs:
    meta_row = metadata.loc[metadata.sample_folder == sample_dir.name].iloc[0]
    question = meta_row.question
    max_depth = int(meta_row.max_depth)
    max_input_tokens = int(meta_row.bge_max_input_tokens)
    model_name = meta_row.bge_model

    unique_requests = {
        hashlib.sha256(question.encode("utf-8")).hexdigest(): {
            "source_path": "initial",
            "prefix_length": 0,
            "input_text": question,
        }
    }

    for path_file in sorted(sample_dir.glob("path_*.jsonl")):
        actions = read_action_file(path_file)
        # Khi len(actions) < max_depth, request sau action cuối là request chọn STOP.
        last_prefix = len(actions) if len(actions) < max_depth else max(0, len(actions) - 1)
        for prefix_length in range(1, last_prefix + 1):
            prefix = actions[:prefix_length]
            prefix_key = hashlib.sha256(
                json.dumps(prefix, ensure_ascii=False, sort_keys=True, default=str).encode("utf-8")
            ).hexdigest()
            unique_requests.setdefault(prefix_key, {
                "source_path": path_file.name,
                "prefix_length": prefix_length,
                "input_text": question + "\n" + workflow_state(prefix),
            })

    for request_index, request in enumerate(unique_requests.values(), start=1):
        uncapped_tokens = int(count_bge_tokens(request["input_text"]))
        bge_rows.append({
            "sample_folder": sample_dir.name,
            "request_index": request_index,
            "source_path": request["source_path"],
            "prefix_length": request["prefix_length"],
            "bge_model": model_name,
            "count_method": bge_count_method,
            "characters": len(request["input_text"]),
            "estimated_uncapped_tokens": uncapped_tokens,
            "configured_max_input_tokens": max_input_tokens,
            "estimated_tokens_after_cap": min(uncapped_tokens, max_input_tokens),
            "would_be_truncated": uncapped_tokens > max_input_tokens,
        })

bge_requests = pd.DataFrame(bge_rows)
bge_summary = pd.DataFrame([{
    "component": f"BGE Task Analyzer ({default_model})",
    "total_tokens": int(bge_requests.estimated_tokens_after_cap.sum()),
    "total_uncapped_tokens": int(bge_requests.estimated_uncapped_tokens.sum()),
    "calls": len(bge_requests),
    "mean_tokens_per_call": bge_requests.estimated_tokens_after_cap.mean(),
    "median_tokens_per_call": bge_requests.estimated_tokens_after_cap.median(),
    "p95_tokens_per_call": bge_requests.estimated_tokens_after_cap.quantile(0.95),
    "truncated_requests": int(bge_requests.would_be_truncated.sum()),
    "count_method": bge_count_method,
    "is_exact_from_log": False,
}])

display(bge_summary.style.format({
    "total_tokens": "{:,.0f}",
    "total_uncapped_tokens": "{:,.0f}",
    "mean_tokens_per_call": "{:,.1f}",
    "median_tokens_per_call": "{:,.1f}",
    "p95_tokens_per_call": "{:,.1f}",
}))
print("Lưu ý: BGE total là estimate input token, không phải response.usage lấy từ log.")
'''),
    markdown("combined-heading", r'''
## 4. Bảng tổng hợp và biểu đồ

Trong bảng kết hợp, các hàng agent là số liệu chính xác từ artefact; hàng BGE là ước lượng input token theo phương pháp được ghi trong `count_method`.
'''),
    code("summaries-and-charts", r'''
agent_components = agent_summary.rename(columns={"agent": "component"}).copy()
agent_components["component_type"] = "Agent (exact)"
agent_components["count_method"] = "path_*.jsonl tokens"
agent_components["is_exact_from_log"] = True

bge_component = bge_summary.copy()
bge_component["component_type"] = "BGE (estimated)"

combined_summary = pd.concat([
    agent_components[["component", "component_type", "total_tokens", "calls", "mean_tokens_per_call", "median_tokens_per_call", "p95_tokens_per_call", "count_method", "is_exact_from_log"]],
    bge_component[["component", "component_type", "total_tokens", "calls", "mean_tokens_per_call", "median_tokens_per_call", "p95_tokens_per_call", "count_method", "is_exact_from_log"]],
], ignore_index=True).sort_values("total_tokens", ascending=False).reset_index(drop=True)
display(combined_summary.style.format({
    "total_tokens": "{:,.0f}",
    "mean_tokens_per_call": "{:,.1f}",
    "median_tokens_per_call": "{:,.1f}",
    "p95_tokens_per_call": "{:,.1f}",
}))

sample_agent_tokens = agent_actions.pivot_table(
    index="sample_folder", columns="agent", values="tokens", aggfunc="sum", fill_value=0
)
sample_bge_tokens = bge_requests.groupby("sample_folder").estimated_tokens_after_cap.sum().rename("BGE Task Analyzer (estimated)")
sample_totals = sample_agent_tokens.join(sample_bge_tokens, how="outer").fillna(0)
sample_totals["all_components_total"] = sample_totals.sum(axis=1)

fig, ax = plt.subplots(figsize=(13, max(6, 0.48 * len(combined_summary))))
plot_data = combined_summary.sort_values("total_tokens", ascending=True)
colors = np.where(plot_data.component_type.eq("BGE (estimated)"), "#E8710A", "#3366CC")
ax.barh(plot_data.component, plot_data.total_tokens, color=colors)
ax.set_xlabel("Total tokens")
ax.set_title(f"Token usage by agent and BGE — {LOG_GROUP_PATH.name}")
for i, value in enumerate(plot_data.total_tokens):
    ax.text(value, i, f"  {value:,.0f}", va="center", fontsize=10)
fig.tight_layout()
save_figure(fig, "01_total_tokens_by_agent_and_bge.png")
plt.show()

fig, ax = plt.subplots(figsize=(15, 7))
order = agent_summary.agent.tolist()
sns.boxplot(
    data=agent_actions,
    x="agent",
    y="tokens",
    order=order,
    color="#8AB4F8",
    showfliers=False,
    ax=ax,
)
ax.set_xlabel("")
ax.set_ylabel("Tokens / agent action")
ax.set_title("Distribution of tokens per agent activation")
ax.tick_params(axis="x", rotation=35)
fig.tight_layout()
save_figure(fig, "02_agent_tokens_per_activation.png")
plt.show()

top_samples = sample_totals.nlargest(min(TOP_SAMPLES_TO_PLOT, len(sample_totals)), "all_components_total")
stack_columns = [column for column in top_samples.columns if column != "all_components_total"]
fig, ax = plt.subplots(figsize=(16, 8))
top_samples[stack_columns].plot(kind="bar", stacked=True, ax=ax, colormap="tab20")
ax.set_xlabel("Sample folder")
ax.set_ylabel("Tokens")
ax.set_title(f"Token composition for top {len(top_samples)} samples")
ax.tick_params(axis="x", rotation=65, labelsize=8)
ax.legend(title="Component", bbox_to_anchor=(1.02, 1), loc="upper left", fontsize=9)
fig.tight_layout()
save_figure(fig, "03_sample_token_composition.png")
plt.show()

fig, axes = plt.subplots(1, 2, figsize=(16, 6))
sns.histplot(bge_requests.estimated_tokens_after_cap, bins=25, color="#E8710A", ax=axes[0])
axes[0].set_xlabel("Estimated BGE input tokens after configured cap")
axes[0].set_title("BGE tokens per request")
bge_by_sample = bge_requests.groupby("sample_folder").estimated_tokens_after_cap.sum().sort_values().reset_index(drop=True)
axes[1].plot(np.arange(1, len(bge_by_sample) + 1), bge_by_sample.cumsum(), color="#BF360C", linewidth=2.5)
axes[1].set_xlabel("Samples sorted by BGE token usage")
axes[1].set_ylabel("Cumulative estimated BGE tokens")
axes[1].set_title("Cumulative BGE token usage")
fig.tight_layout()
save_figure(fig, "04_bge_token_estimates.png")
plt.show()
'''),
    markdown("export-heading", r'''
## 5. Xuất dữ liệu và quality checks

Các file chi tiết cho phép truy ngược từ tổng token về sample, path và action. `quality_checks.csv` ghi rõ số action prefix bị loại, phương pháp đếm BGE và tỷ lệ request có thể bị truncate.
'''),
    code("exports", r'''
agent_actions.to_csv(REPORT_DIR / "agent_actions_deduplicated.csv", index=False)
raw_actions.to_csv(REPORT_DIR / "agent_actions_raw.csv", index=False)
agent_summary.to_csv(REPORT_DIR / "agent_token_summary.csv", index=False)
bge_requests.to_csv(REPORT_DIR / "bge_request_token_estimates.csv", index=False)
bge_summary.to_csv(REPORT_DIR / "bge_token_summary.csv", index=False)
combined_summary.to_csv(REPORT_DIR / "combined_agent_bge_token_summary.csv", index=False)
sample_totals.to_csv(REPORT_DIR / "sample_token_totals.csv")
metadata.to_csv(REPORT_DIR / "sample_log_metadata.csv", index=False)

quality_checks = pd.DataFrame([{
    "log_group": LOG_GROUP_PATH.name,
    "sample_folders": len(sample_dirs),
    "samples_with_path_actions": agent_actions.sample_folder.nunique(),
    "raw_action_rows": len(raw_actions),
    "unique_action_rows": len(agent_actions),
    "copied_prefix_rows_removed": copied_prefix_rows_removed,
    "agent_total_tokens_exact": int(agent_actions.tokens.sum()),
    "bge_request_count_reconstructed": len(bge_requests),
    "bge_total_tokens_estimated_after_cap": int(bge_requests.estimated_tokens_after_cap.sum()),
    "bge_total_tokens_estimated_uncapped": int(bge_requests.estimated_uncapped_tokens.sum()),
    "bge_requests_above_configured_cap": int(bge_requests.would_be_truncated.sum()),
    "bge_count_method": bge_count_method,
    "bge_exact_usage_available_in_log": False,
}])
quality_checks.to_csv(REPORT_DIR / "quality_checks.csv", index=False)
display(quality_checks.T)

print("Hoàn tất. Files đã tạo:")
for path in sorted(REPORT_DIR.iterdir()):
    print(f"- {path.name}")
'''),
]

notebook = {
    "cells": cells,
    "metadata": {
        "kernelspec": {
            "display_name": "Python 3",
            "language": "python",
            "name": "python3",
        },
        "language_info": {
            "name": "python",
            "version": "3.11",
        },
    },
    "nbformat": 4,
    "nbformat_minor": 5,
}

OUT.parent.mkdir(parents=True, exist_ok=True)
OUT.write_text(json.dumps(notebook, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
