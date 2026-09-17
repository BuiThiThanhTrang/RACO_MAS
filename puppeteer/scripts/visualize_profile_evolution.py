"""Compare capability profiles at prior, probe, and evolved checkpoints.

The default command compares every Domain Reasoner in the S0 pool:

python scripts/visualize_profile_evolution.py \
  --personas personas/role_aware/s0_pool.jsonl \
  --probe profiles/artifacts/gsm-hard_s0_seed42_probe_profiles.json \
  --checkpoint runs/gsm-hard_train_seed42_20260826_train/checkpoints/latest.pt \
  --role "Domain Reasoner"
"""

from __future__ import annotations

import argparse
import html
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch


STAGES = ("Prior", "Probe", "Evolved")
STAGE_COLORS = {
    "Prior": "#7A869A",
    "Probe": "#E8710A",
    "Evolved": "#3366CC",
}
CAPABILITY_LABELS = {
    "planning": "Planning",
    "general_reasoning": "General\nreasoning",
    "quantitative_reasoning": "Quantitative\nreasoning",
    "domain_reasoning": "Domain\nreasoning",
    "software_engineering": "Software\nengineering",
    "commonsense_generation": "Commonsense\ngeneration",
    "verification": "Verification",
    "repair": "Repair",
    "integration": "Integration",
    "tool_use": "Tool use",
}


@dataclass(frozen=True)
class Teammate:
    teammate_id: str
    backbone: str
    role_name: str
    capability_prior: dict[str, float]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Visualize prior → probe → evolved capability profiles."
    )
    parser.add_argument(
        "--personas",
        type=Path,
        default=Path("personas/role_aware/s0_pool.jsonl"),
    )
    parser.add_argument(
        "--probe",
        type=Path,
        default=Path("profiles/artifacts/gsm-hard_s0_seed42_probe_profiles.json"),
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=Path(
            "runs/gsm-hard_train_seed42_20260826_train/checkpoints/latest.pt"
        ),
    )
    parser.add_argument("--role", default="Domain Reasoner")
    parser.add_argument(
        "--teammate-id",
        action="append",
        default=None,
        help="Optional teammate ID. Repeat to select multiple teammates.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("../output/domain_reasoner_profile_evolution"),
    )
    return parser.parse_args()


def resolve(path: Path, project_root: Path) -> Path:
    return path.resolve() if path.is_absolute() else (project_root / path).resolve()


def read_json(path: Path) -> dict:
    if not path.is_file():
        raise FileNotFoundError(path)
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(f"Expected a JSON object in {path}")
    return value


def load_teammates(path: Path, role_name: str, selected_ids: list[str] | None) -> list[Teammate]:
    if not path.is_file():
        raise FileNotFoundError(path)
    teammates: list[Teammate] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        card = row.get("role_card", {})
        teammate_id = str(row.get("teammate_id"))
        if card.get("role_name") != role_name:
            continue
        if selected_ids and teammate_id not in selected_ids:
            continue
        teammates.append(
            Teammate(
                teammate_id=teammate_id,
                backbone=str(row.get("backbone")),
                role_name=str(card.get("role_name")),
                capability_prior={
                    str(name): float(value)
                    for name, value in (card.get("capability_prior") or {}).items()
                },
            )
        )
    if not teammates:
        suffix = f" with IDs {selected_ids}" if selected_ids else ""
        raise ValueError(f"No teammates found for role {role_name!r}{suffix}")
    return teammates


def load_checkpoint(path: Path) -> dict:
    if not path.is_file():
        raise FileNotFoundError(path)
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(checkpoint, dict) or not isinstance(checkpoint.get("profiles"), dict):
        raise ValueError(f"Checkpoint does not contain a profile store: {path}")
    return checkpoint


def prior_profile(teammate: Teammate, capability_names: list[str]) -> dict[str, Any]:
    return {
        "capability_names": capability_names,
        "mean": [float(teammate.capability_prior.get(name, 0.5)) for name in capability_names],
        "uncertainty": [1.0] * len(capability_names),
        "observation_count": [0] * len(capability_names),
        "role_adherence_mean": 1.0,
        "role_adherence_uncertainty": 1.0,
        "role_adherence_count": 0,
        "global_reliability_mean": 0.5,
        "global_reliability_uncertainty": 1.0,
        "global_reliability_count": 0,
        "update_step": 0,
    }


def validate_profile(profile: dict, teammate_id: str, stage: str) -> None:
    required = ("capability_names", "mean", "uncertainty", "observation_count")
    missing = [name for name in required if name not in profile]
    if missing:
        raise ValueError(f"{stage} profile for {teammate_id} is missing {missing}")
    size = len(profile["capability_names"])
    if any(len(profile[name]) != size for name in ("mean", "uncertainty", "observation_count")):
        raise ValueError(f"{stage} profile vectors have different lengths for {teammate_id}")


def build_frames(
    teammates: list[Teammate], probe_profiles: dict, checkpoint_profiles: dict
) -> tuple[pd.DataFrame, pd.DataFrame]:
    capability_rows: list[dict] = []
    summary_rows: list[dict] = []
    for teammate in teammates:
        if teammate.teammate_id not in probe_profiles:
            raise KeyError(f"Probe profile missing teammate {teammate.teammate_id}")
        if teammate.teammate_id not in checkpoint_profiles:
            raise KeyError(f"Checkpoint profile missing teammate {teammate.teammate_id}")
        probe = probe_profiles[teammate.teammate_id]
        evolved = checkpoint_profiles[teammate.teammate_id]
        capability_names = [str(name) for name in probe["capability_names"]]
        prior = prior_profile(teammate, capability_names)
        profiles = {"Prior": prior, "Probe": probe, "Evolved": evolved}
        for stage, profile in profiles.items():
            validate_profile(profile, teammate.teammate_id, stage)
            if list(profile["capability_names"]) != capability_names:
                raise ValueError(
                    f"Capability order mismatch for {teammate.teammate_id} at {stage}"
                )
            for index, capability in enumerate(capability_names):
                capability_rows.append(
                    {
                        "teammate_id": teammate.teammate_id,
                        "backbone": teammate.backbone,
                        "role_name": teammate.role_name,
                        "stage": stage,
                        "stage_order": STAGES.index(stage),
                        "capability": capability,
                        "capability_order": index,
                        "mean": float(profile["mean"][index]),
                        "uncertainty": float(profile["uncertainty"][index]),
                        "observation_count": int(profile["observation_count"][index]),
                    }
                )
            summary_rows.append(
                {
                    "teammate_id": teammate.teammate_id,
                    "backbone": teammate.backbone,
                    "role_name": teammate.role_name,
                    "stage": stage,
                    "stage_order": STAGES.index(stage),
                    "global_reliability_mean": float(
                        profile.get("global_reliability_mean", 0.5)
                    ),
                    "global_reliability_uncertainty": float(
                        profile.get("global_reliability_uncertainty", 1.0)
                    ),
                    "global_reliability_count": int(
                        profile.get("global_reliability_count", 0)
                    ),
                    "role_adherence_mean": float(profile.get("role_adherence_mean", 1.0)),
                    "role_adherence_uncertainty": float(
                        profile.get("role_adherence_uncertainty", 1.0)
                    ),
                    "role_adherence_count": int(profile.get("role_adherence_count", 0)),
                    "update_step": int(profile.get("update_step", 0)),
                }
            )
    return pd.DataFrame(capability_rows), pd.DataFrame(summary_rows)


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


def save_figure(fig: plt.Figure, output_dir: Path, filename: str) -> Path:
    path = output_dir / filename
    fig.savefig(path, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return path


def plot_capability_profiles(capabilities: pd.DataFrame, output_dir: Path) -> Path:
    teammate_ids = capabilities.teammate_id.drop_duplicates().tolist()
    figure_height = max(6.5, 5.4 * len(teammate_ids))
    fig, axes = plt.subplots(
        len(teammate_ids),
        1,
        figsize=(17, figure_height),
        sharex=True,
        squeeze=False,
    )
    for axis, teammate_id in zip(axes.flat, teammate_ids):
        frame = capabilities.loc[capabilities.teammate_id == teammate_id]
        metadata = frame.iloc[0]
        for stage in STAGES:
            stage_frame = frame.loc[frame.stage == stage].sort_values("capability_order")
            axis.plot(
                stage_frame.capability_order,
                stage_frame["mean"],
                marker="o",
                markersize=7,
                linewidth=2.4,
                color=STAGE_COLORS[stage],
                label=stage,
            )
        axis.set_ylim(0, 1.04)
        axis.set_ylabel("Capability mean")
        axis.set_title(f"{teammate_id} · {metadata.backbone}")
        axis.legend(ncol=3, loc="lower left", fontsize=10)
    last_frame = capabilities.loc[
        capabilities.teammate_id == teammate_ids[0]
    ].sort_values("capability_order")
    labels = [CAPABILITY_LABELS.get(value, value) for value in last_frame.capability.drop_duplicates()]
    axes[-1, 0].set_xticks(range(len(labels)), labels, rotation=20, ha="right")
    axes[-1, 0].set_xlabel("Capability dimension")
    fig.suptitle(
        "Domain Reasoner capability profile: prior → probe → evolved",
        y=1.01,
        fontsize=19,
        fontweight="bold",
    )
    fig.tight_layout()
    return save_figure(fig, output_dir, "01_capability_profile_evolution.png")


def plot_reliability_and_evidence(
    capabilities: pd.DataFrame, summaries: pd.DataFrame, output_dir: Path
) -> Path:
    teammate_ids = summaries.teammate_id.drop_duplicates().tolist()
    fig, axes = plt.subplots(1, 2, figsize=(18, 7))
    markers = ("o", "s", "^")
    for marker, teammate_id in zip(markers, teammate_ids):
        frame = summaries.loc[summaries.teammate_id == teammate_id].sort_values("stage_order")
        label = f"{teammate_id} · {frame.iloc[0].backbone}"
        axes[0].plot(
            frame.stage,
            frame.global_reliability_mean,
            marker=marker,
            markersize=8,
            linewidth=2.5,
            label=label,
        )
        for x, value in zip(frame.stage, frame.global_reliability_mean):
            axes[0].annotate(f"{value:.3f}", (x, value), xytext=(0, 9), textcoords="offset points", ha="center", fontsize=10)
    axes[0].set_ylim(0, 1.04)
    axes[0].set_ylabel("Global reliability mean")
    axes[0].set_xlabel("Profile stage")
    axes[0].set_title("Global reliability evolution")
    axes[0].legend(fontsize=10)

    observed = capabilities.loc[capabilities.observation_count > 0].copy()
    observed["label"] = observed.capability.map(
        lambda value: CAPABILITY_LABELS.get(value, value).replace("\n", " ")
    )
    observed["series"] = observed.teammate_id + " · " + observed.stage
    pivot = observed.pivot_table(
        index="label",
        columns="series",
        values="observation_count",
        aggfunc="max",
        fill_value=0,
    )
    pivot.plot(kind="bar", ax=axes[1], width=0.78, colormap="tab20")
    axes[1].set_xlabel("Updated capability")
    axes[1].set_ylabel("Observation count")
    axes[1].set_title("Evidence accumulated by capability")
    axes[1].tick_params(axis="x", rotation=0)
    axes[1].legend(title="Teammate · stage", fontsize=8, title_fontsize=9)
    fig.tight_layout()
    return save_figure(fig, output_dir, "02_reliability_and_evidence.png")


def plot_profile_deltas(capabilities: pd.DataFrame, output_dir: Path) -> Path:
    rows: list[dict] = []
    for teammate_id, frame in capabilities.groupby("teammate_id", sort=False):
        pivot = frame.pivot(index="capability", columns="stage", values="mean")
        metadata = frame.iloc[0]
        for transition, start, end in (
            ("Probe − Prior", "Prior", "Probe"),
            ("Evolved − Probe", "Probe", "Evolved"),
            ("Evolved − Prior", "Prior", "Evolved"),
        ):
            for capability, value in (pivot[end] - pivot[start]).items():
                rows.append(
                    {
                        "row": f"{teammate_id} · {metadata.backbone}\n{transition}",
                        "capability": capability,
                        "delta": float(value),
                    }
                )
    delta = pd.DataFrame(rows)
    capability_order = (
        capabilities.sort_values("capability_order").capability.drop_duplicates().tolist()
    )
    row_order = delta.row.drop_duplicates().tolist()
    matrix = delta.pivot(index="row", columns="capability", values="delta").reindex(
        index=row_order, columns=capability_order
    )
    bound = max(0.05, float(np.nanmax(np.abs(matrix.to_numpy()))))
    fig, ax = plt.subplots(figsize=(17, max(6, 0.85 * len(matrix))))
    sns.heatmap(
        matrix,
        cmap="RdBu_r",
        center=0,
        vmin=-bound,
        vmax=bound,
        annot=True,
        fmt=".3f",
        linewidths=0.5,
        cbar_kws={"label": "Change in capability mean"},
        ax=ax,
    )
    ax.set_xlabel("Capability dimension")
    ax.set_ylabel("")
    ax.set_xticklabels(
        [CAPABILITY_LABELS.get(value, value).replace("\n", " ") for value in capability_order],
        rotation=30,
        ha="right",
    )
    ax.set_title("Capability changes between profile stages")
    fig.tight_layout()
    return save_figure(fig, output_dir, "03_capability_delta_heatmap.png")


def table_html(frame: pd.DataFrame) -> str:
    return frame.to_html(index=False, border=0, classes="data-table", escape=True)


def write_report(
    output_dir: Path,
    teammates: list[Teammate],
    capabilities: pd.DataFrame,
    summaries: pd.DataFrame,
    checkpoint: dict,
    figure_paths: list[Path],
) -> Path:
    selected_capabilities = capabilities.loc[
        (capabilities.observation_count > 0)
        | capabilities.capability.eq("domain_reasoning")
    ].copy()
    selected_capabilities["mean"] = selected_capabilities["mean"].map(lambda value: f"{value:.4f}")
    selected_capabilities["uncertainty"] = selected_capabilities["uncertainty"].map(
        lambda value: f"{value:.4f}"
    )
    selected_capabilities = selected_capabilities[
        [
            "teammate_id",
            "backbone",
            "stage",
            "capability",
            "mean",
            "uncertainty",
            "observation_count",
        ]
    ]
    summary_table = summaries.copy()
    for column in (
        "global_reliability_mean",
        "global_reliability_uncertainty",
        "role_adherence_mean",
        "role_adherence_uncertainty",
    ):
        summary_table[column] = summary_table[column].map(lambda value: f"{value:.4f}")
    summary_table = summary_table[
        [
            "teammate_id",
            "backbone",
            "stage",
            "global_reliability_mean",
            "global_reliability_uncertainty",
            "global_reliability_count",
            "role_adherence_mean",
            "role_adherence_count",
            "update_step",
        ]
    ]
    progress = checkpoint.get("progress", {}) or {}
    metadata = checkpoint.get("metadata", {}) or {}
    images = "\n".join(
        f'<section><h2>{html.escape(path.stem.replace("_", " ").title())}</h2><img src="{html.escape(path.name)}" alt="{html.escape(path.stem)}"></section>'
        for path in figure_paths
    )
    document = f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Domain Reasoner profile evolution</title>
<style>
body{{margin:0;background:#f6f7fb;color:#172033;font:15px/1.5 Segoe UI,Arial,sans-serif}}main{{max-width:1250px;margin:auto;padding:36px 24px 70px}}h1{{margin-bottom:5px}}p{{color:#667085}}section,.table-wrap{{background:white;border:1px solid #e5e9f2;border-radius:14px;padding:20px;margin:20px 0}}img{{width:100%;height:auto}}.data-table{{border-collapse:collapse;width:100%;font-size:13px}}th,td{{padding:8px 10px;border-bottom:1px solid #e9edf4;text-align:left}}th{{background:#f8faff}}.table-wrap{{overflow-x:auto}}code{{background:#eef2f8;padding:2px 5px;border-radius:4px}}
</style></head><body><main>
<h1>Domain Reasoner profile evolution</h1>
<p>Prior from S0 RoleCard → 10-item probe profile → evolved profile after {html.escape(str(progress.get('completed_items', 'unknown')))} training items. Checkpoint saved at {html.escape(str(metadata.get('saved_at', 'unknown')))}.</p>
<div class="table-wrap"><h2>Global profile summary</h2>{table_html(summary_table)}</div>
<div class="table-wrap"><h2>Updated and primary capability dimensions</h2>{table_html(selected_capabilities)}</div>
{images}
</main></body></html>"""
    report_path = output_dir / "report.html"
    report_path.write_text(document, encoding="utf-8")
    return report_path


def main() -> None:
    args = parse_args()
    project_root = Path(__file__).resolve().parents[1]
    personas_path = resolve(args.personas, project_root)
    probe_path = resolve(args.probe, project_root)
    checkpoint_path = resolve(args.checkpoint, project_root)
    output_dir = resolve(args.output, project_root)
    output_dir.mkdir(parents=True, exist_ok=True)

    teammates = load_teammates(personas_path, args.role, args.teammate_id)
    probe_profiles = read_json(probe_path)
    checkpoint = load_checkpoint(checkpoint_path)
    capabilities, summaries = build_frames(
        teammates, probe_profiles, checkpoint["profiles"]
    )

    setup_plotting()
    figure_paths = [
        plot_capability_profiles(capabilities, output_dir),
        plot_reliability_and_evidence(capabilities, summaries, output_dir),
        plot_profile_deltas(capabilities, output_dir),
    ]
    capabilities.to_csv(output_dir / "capability_profile_evolution.csv", index=False)
    summaries.to_csv(output_dir / "profile_summary_evolution.csv", index=False)
    run_metadata = {
        "personas": str(personas_path),
        "probe_profiles": str(probe_path),
        "checkpoint": str(checkpoint_path),
        "role": args.role,
        "teammate_ids": [item.teammate_id for item in teammates],
        "checkpoint_metadata": checkpoint.get("metadata", {}),
        "checkpoint_progress": checkpoint.get("progress", {}),
    }
    (output_dir / "sources.json").write_text(
        json.dumps(run_metadata, ensure_ascii=False, indent=2, default=str),
        encoding="utf-8",
    )
    report_path = write_report(
        output_dir,
        teammates,
        capabilities,
        summaries,
        checkpoint,
        figure_paths,
    )
    print(f"Report: {report_path}")
    print("Teammates:", ", ".join(item.teammate_id for item in teammates))
    print("Figures:", len(figure_paths))


if __name__ == "__main__":
    main()
