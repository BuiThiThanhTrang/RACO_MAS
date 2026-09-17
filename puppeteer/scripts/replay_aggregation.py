"""Replay label-free aggregation on identical saved dev candidates."""
import argparse
from collections import defaultdict
import json
from pathlib import Path
import random
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from role_aware.aggregation import aggregate_candidates, normalize_choice
from role_aware.audit_analysis import collect_tasks
from role_aware.audit_trace import AuditTrace, digest, write_json

def paired_bootstrap(rows, baseline, treatment, seed=42, repeats=2000):
    clusters = defaultdict(list)
    for row in rows:
        clusters[row["task_id"]].append(int(row[treatment]) - int(row[baseline]))
    values = [sum(v)/len(v) for v in clusters.values()]
    if not values:
        return None
    rng = random.Random(seed)
    samples = sorted(sum(rng.choices(values, k=len(values)))/len(values) for _ in range(repeats))
    return dict(delta=sum(values)/len(values), ci95=[samples[int(.025*repeats)], samples[int(.975*repeats)]],
                task_clusters=len(values), observations=len(rows))

def replay(tasks, modes, seed=42, verifier=None):
    rows = []
    for task in tasks:
        if digest(task["candidates"]) != task["candidate_hash"]:
            raise ValueError("Candidate snapshot hash mismatch")
        # Gold is read only after every method has committed its prediction.
        predictions = {mode: aggregate_candidates(task["candidates"], mode=mode, seed=seed,
            task_id=task["task_id"], choices=task.get("choices", "ABCDEFGHIJ"),
            question=task["question"], verifier=verifier) for mode in modes}
        gold = normalize_choice(task.get("evaluation", {}).get("gold"), task.get("choices", "ABCDEFGHIJ"))
        if gold is None:
            raise ValueError("Missing valid evaluation label")
        rows.append(dict(task_id=task["task_id"], run_id=task["run_id"],
            candidate_hash=task["candidate_hash"], predictions=predictions,
            **{m: normalize_choice(predictions[m]["prediction"]) == gold for m in modes}))
    comparisons = {}
    for mode in modes[1:]:
        comparisons[mode] = paired_bootstrap(rows, modes[0], mode, seed)
    return dict(rows=rows, comparisons=comparisons, seed=seed,
                accuracy={m: sum(r[m] for r in rows)/len(rows) if rows else None for m in modes},
                paired_changes={m: dict(wrong_to_correct=sum(not r[modes[0]] and r[m] for r in rows),
                                        correct_to_wrong=sum(r[modes[0]] and not r[m] for r in rows))
                                for m in modes[1:]})

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("inputs", nargs="+")
    parser.add_argument("--output", required=True)
    parser.add_argument("--modes", nargs="+", default=["legacy", "majority"],
                        choices=["legacy", "majority", "majority_verifier"])
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--verifier-model", help="Explicit model; this option enables live API calls on ties")
    parser.add_argument("--allow-non-dev-diagnostic", action="store_true")
    args = parser.parse_args()
    tasks = collect_tasks(args.inputs)
    if not tasks:
        parser.error("No completed candidate snapshots found")
    if any(t["split"] != "dev" for t in tasks) and not args.allow_non_dev_diagnostic:
        parser.error("Method selection requires dev snapshots; non-dev is diagnostic only")
    verifier = None
    trace = None
    if "majority_verifier" in args.modes:
        if not args.verifier_model:
            parser.error("--verifier-model is required for majority_verifier")
        from model.query_manager import query_manager
        from model.model_config import model_registry
        trace = AuditTrace(Path(args.output).with_suffix("").with_name(Path(args.output).stem + "_calls"),
                           "aggregation_replay", "ties", "attempt-1")
        def verifier(prompt):
            with trace.call_scope(purpose="tie_verifier", model_size=model_registry.get_model_size(args.verifier_model) or 0):
                return query_manager.query(args.verifier_model, prompt)
    result = replay(tasks, args.modes, args.seed, verifier)
    result["purpose"] = "non-dev diagnostic" if any(t["split"] != "dev" for t in tasks) else "dev selection"
    result["verifier_cost"] = trace.call_totals() if trace else {}
    write_json(args.output, result)
    print(json.dumps(dict(tasks=len(tasks), accuracy=result["accuracy"], output=args.output)))

if __name__ == "__main__":
    main()
