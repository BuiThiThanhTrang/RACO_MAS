"""Offline trace validation and outcome analysis; no model access."""
from collections import Counter
import json
import math
from pathlib import Path
from role_aware.aggregation import normalize_choice

def load_events(path):
    rows = []
    for number, line in enumerate(Path(path).read_text(encoding="utf-8").splitlines(), 1):
        if line.strip():
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as error:
                raise ValueError(f"Invalid JSONL at {path}:{number}") from error
    return rows

def validate_events(events):
    errors = []
    completed = any(e["event_type"] == "task_finished" for e in events)
    allocations, receipts, paths, finished = {}, Counter(), set(), Counter()
    commitment = None
    rewarded, updated = set(), False
    for event in events:
        kind = event["event_type"]
        if kind == "allocation":
            for a in event.get("accepted", []):
                if a["action_id"] in allocations:
                    errors.append("duplicate_allocation:" + a["action_id"])
                allocations[a["action_id"]] = a
        elif kind == "action_finished":
            aid = event["action_id"]
            receipts[aid] += 1
            if aid not in allocations:
                errors.append("unallocated_execution:" + aid)
            elif allocations[aid]["path_uid"] != event["path_uid"]:
                errors.append("wrong_path:" + aid)
        elif kind == "path_created":
            paths.add(event["path_uid"])
        elif kind == "path_finished":
            finished[event["path_uid"]] += 1
            if event.get("stop_reason") == "depth_limit" and event.get("p_stop") is not None:
                errors.append("invented_stop_probability:" + event["path_uid"])
        elif kind == "prediction_committed":
            commitment = event["event_seq"]
        elif kind in {"reward_assigned", "policy_update"} and commitment is None:
            errors.append("scoring_before_prediction")
        if kind == "reward_assigned":
            for aid in event.get("action_ids", []):
                rewarded.add(aid)
                if aid not in allocations:
                    errors.append("unallocated_reward:" + aid)
        if kind == "policy_update":
            updated = True
        if kind == "routing_decision" and event.get("probabilities") is not None:
            probs = event["probabilities"]
            if any(not math.isfinite(p) or p < 0 for p in probs):
                errors.append("invalid_probability")
            if abs(sum(probs) - 1) > 1e-5:
                errors.append("distribution_not_normalized")
            if not event.get("allow_stop") and event.get("p_stop") != 0:
                errors.append("stop_not_masked")
    for aid in allocations:
        if receipts[aid] > 1 or (completed and receipts[aid] != 1):
            errors.append("receipt_count:" + aid)
    for uid in paths:
        if finished[uid] > 1 or (completed and finished[uid] != 1):
            errors.append("stop_reason_count:" + uid)
    if completed and updated and rewarded != set(allocations):
        errors.append("reward_execution_mismatch")
    if completed and commitment is None:
        errors.append("missing_prediction_commit")
    return dict(status="completed" if completed else "incomplete", errors=errors,
                actions=len(allocations), receipts=sum(receipts.values()), paths=len(paths))

def collect_tasks(inputs):
    """Latest completed attempt per run/task; interrupted attempts remain separately auditable."""
    records = {}
    for value in inputs:
        source = Path(value)
        files = [source] if source.name == "candidates.json" else sorted(source.rglob("candidates.json"))
        for file in files:
            snapshot = json.loads(file.read_text(encoding="utf-8"))
            event_file = file.with_name("events.jsonl")
            events = load_events(event_file) if event_file.exists() else []
            if not any(e["event_type"] == "task_finished" for e in events):
                continue
            evaluation = file.with_name("evaluation.json")
            snapshot["evaluation"] = json.loads(evaluation.read_text(encoding="utf-8")) if evaluation.exists() else {}
            snapshot["events"] = events
            snapshot["trace_path"] = str(file.parent)
            key = (snapshot["run_id"], snapshot["task_id"])
            if key not in records or snapshot["attempt_id"] > records[key]["attempt_id"]:
                records[key] = snapshot
    return list(records.values())

def summarize(tasks):
    counts = Counter()
    stop = Counter()
    transitions = Counter()
    aggregation_transitions = Counter()
    physical_steps = set()
    strata = {}
    cases = []
    costs = Counter()
    complete_costs = Counter()
    for task in tasks:
        gold = normalize_choice(task.get("evaluation", {}).get("gold"), task.get("choices", "ABCDEFGHIJ"))
        if gold is None:
            counts["missing_gold"] += 1
            continue
        counts["tasks"] += 1
        parsed = [normalize_choice(v, task.get("choices", "ABCDEFGHIJ")) for v in task["candidates"]]
        correct_any = gold in parsed
        correct_final = normalize_choice(task["prediction"], task.get("choices", "ABCDEFGHIJ")) == gold
        counts["any_path_correct"] += int(correct_any)
        counts["final_correct"] += int(correct_final)
        counts["lost_correct"] += int(correct_any and not correct_final)
        counts["candidate_count"] += len(parsed)
        counts["invalid_candidates"] += sum(v is None for v in parsed)
        votes = Counter(v for v in parsed if v is not None)
        counts["tasks_with_valid_candidate"] += bool(votes)
        counts["all_invalid_tasks"] += not bool(votes)
        counts["ties"] += bool(votes) and sum(v == max(votes.values()) for v in votes.values()) > 1
        for path in task["paths"]:
            stop[path["stop_reason"]] += 1
            choices = task.get("choices", "ABCDEFGHIJ")
            def state(value):
                parsed_value = normalize_choice(value, choices)
                return "missing" if parsed_value is None else "correct" if parsed_value == gold else "wrong"
            values = [step.get("result", {}).get("answer") for step in path["steps"]]
            states = [state(v) for v in values]
            for previous, current in zip(states, states[1:]):
                transitions[previous + "->" + current] += 1
            aggregation_transitions[state(path.get("before_aggregation")) + "->" + state(path.get("prediction"))] += 1
            for step in path["steps"]:
                aid = step.get("action_id")
                if aid is not None:
                    physical_steps.add((task["run_id"], task["task_id"], task["attempt_id"], aid))
        for field, bucket in (("path_count", str(len(task["paths"]))),
                              ("max_depth", str(max((len(p["steps"]) for p in task["paths"]), default=0)))):
            group = strata.setdefault(field, {}).setdefault(bucket, Counter())
            group["tasks"] += 1
            group["final_correct"] += correct_final
            group["any_path_correct"] += correct_any
            group["lost_correct"] += correct_any and not correct_final
        for field in ("api_calls", "total_tokens", "unknown_usage_calls", "model_cost"):
            costs[field] += task.get("cost", {}).get(field, 0)
        terminal = next((e for e in reversed(task.get("events", [])) if e["event_type"] == "task_finished"), {})
        for field in ("api_calls", "total_tokens", "unknown_usage_calls", "model_cost"):
            complete_costs[field] += terminal.get(field, 0)
        if correct_any and not correct_final:
            cases.append(dict(task_id=task["task_id"], run_id=task["run_id"],
                              candidates=task["candidates"], prediction=task["prediction"],
                              trace_path=task["trace_path"]))
    ratio = lambda n, d: counts[n] / counts[d] if counts[d] else None
    return dict(counts=dict(counts), final_accuracy=ratio("final_correct", "tasks"),
                any_path_correct_rate=ratio("any_path_correct", "tasks"),
                lost_correct_rate=ratio("lost_correct", "any_path_correct"),
                lost_correct_all_tasks=ratio("lost_correct", "tasks"),
                tie_rate=ratio("ties", "tasks_with_valid_candidate"),
                invalid_answer_rate=ratio("invalid_candidates", "candidate_count"),
                stop_reasons=dict(stop), leaf_transition_counts=dict(transitions),
                path_aggregation_transition_counts=dict(aggregation_transitions),
                physical_agent_executions=len(physical_steps),
                strata={field:{bucket:dict(v) for bucket,v in groups.items()} for field,groups in strata.items()},
                cost=dict(costs), cost_including_scoring=dict(complete_costs), lost_correct_cases=cases)
