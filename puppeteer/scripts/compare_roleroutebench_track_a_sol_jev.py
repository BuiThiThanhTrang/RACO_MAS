from __future__ import annotations

import argparse
import json
import math
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


PROJECT = Path(__file__).resolve().parents[1]


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def main() -> int:
    parser = argparse.ArgumentParser(description="Paired Sol/JEV comparison using Terra Track A labels")
    parser.add_argument(
        "--root",
        type=Path,
        default=PROJECT / "data" / "RoleRouteBench" / "track_a_4d_live_v1",
    )
    parser.add_argument(
        "--sol-predictions",
        type=Path,
        default=None,
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=PROJECT / "report" / "roleroutebench_track_a_sol_jev_terra_comparison.json",
    )
    args = parser.parse_args()
    sol_path = args.sol_predictions or args.root / "router_runs" / "sol_gpt6_openrouter_v1" / "predictions.jsonl"

    states = {row["decision_id"]: row for row in read_jsonl(args.root / "track_a_states.jsonl")}
    sol = {row["decision_id"]: row for row in read_jsonl(sol_path)}
    jev = {row["decision_id"]: row for row in read_jsonl(args.root / "track_a_provenance.jsonl")}
    labels: dict[str, dict[str, Any]] = {}
    for cohort in sorted(args.root.glob("labeling*")):
        if (cohort / "sample_manifest.json").exists():
            labels.update({row["decision_id"]: row for row in read_jsonl(cohort / "llm_prelabels.jsonl")})

    ids = sorted(set(states) & set(sol) & set(jev) & set(labels))
    paired = Counter()
    by_domain: dict[str, Counter[str]] = defaultdict(Counter)
    by_phase: dict[str, Counter[str]] = defaultdict(Counter)
    stop = Counter()
    same_action = 0
    records: list[dict[str, Any]] = []
    for decision_id in ids:
        valid_actions = set(labels[decision_id]["valid_actions"])
        sol_action = sol[decision_id]["selected_action"]
        jev_action = jev[decision_id]["source_action_as_run"]
        sol_valid = sol_action in valid_actions
        jev_valid = jev_action in valid_actions
        if sol_valid and jev_valid:
            paired["both_correct"] += 1
        elif sol_valid:
            paired["sol_only"] += 1
        elif jev_valid:
            paired["jev_only"] += 1
        else:
            paired["both_wrong"] += 1
        same_action += int(sol_action == jev_action)
        domain = states[decision_id]["domain"]
        phase = labels[decision_id]["state_phase"]
        for bucket, key in ((by_domain, domain), (by_phase, phase)):
            bucket[key].update(total=1, sol=int(sol_valid), jev=int(jev_valid))
        terminal = bool(labels[decision_id]["terminal_valid"])
        stop.update(
            terminal=int(terminal),
            sol_stop=int(sol_action == "STOP"),
            sol_true=int(sol_action == "STOP" and terminal),
            jev_stop=int(jev_action == "STOP"),
            jev_true=int(jev_action == "STOP" and terminal),
        )
        records.append(
            {
                "decision_id": decision_id,
                "domain": domain,
                "state_phase": phase,
                "sol_action": sol_action,
                "jev_action": jev_action,
                "valid_actions": sorted(valid_actions),
                "sol_valid": sol_valid,
                "jev_valid": jev_valid,
            }
        )

    discordant = paired["sol_only"] + paired["jev_only"]
    smaller = min(paired["sol_only"], paired["jev_only"])
    mcnemar_p = (
        min(1.0, 2 * sum(math.comb(discordant, x) for x in range(smaller + 1)) / (2**discordant))
        if discordant
        else 1.0
    )

    def groups(source: dict[str, Counter[str]]) -> dict[str, Any]:
        return {
            key: {
                **dict(value),
                "sol_accuracy": value["sol"] / value["total"],
                "jev_accuracy": value["jev"] / value["total"],
                "delta_percentage_points": 100 * (value["sol"] - value["jev"]) / value["total"],
            }
            for key, value in sorted(source.items())
        }

    report = {
        "coverage": {
            "primary_states": len(states),
            "matched_states": len(ids),
            "missing_sol_predictions": len(set(states) - set(sol)),
        },
        "paired": dict(paired),
        "same_action_count": same_action,
        "same_action_rate": same_action / len(ids) if ids else None,
        "mcnemar_exact_two_sided_p": mcnemar_p,
        "by_domain": groups(by_domain),
        "by_state_phase": groups(by_phase),
        "stop": {
            **dict(stop),
            "sol_precision": stop["sol_true"] / stop["sol_stop"] if stop["sol_stop"] else None,
            "sol_recall": stop["sol_true"] / stop["terminal"] if stop["terminal"] else None,
            "jev_precision": stop["jev_true"] / stop["jev_stop"] if stop["jev_stop"] else None,
            "jev_recall": stop["jev_true"] / stop["terminal"] if stop["terminal"] else None,
        },
        "records": records,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({key: value for key, value in report.items() if key != "records"}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
