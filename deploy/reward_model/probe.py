"""GPU probes and CPU comparison. Passing is necessary, not a policy-quality claim."""
import json
import math
import time
from pathlib import Path
from .artifacts import LOCK, PACKAGE, fingerprint, atomic_json
from .schemas import validate_request


def cases():
    rows = json.loads((PACKAGE / "probe_cases.json").read_text(encoding="utf-8"))
    if len(rows) < 2 or len({r["id"] for r in rows}) != len(rows):
        raise ValueError("Probe case IDs must be unique")
    for row in rows:
        validate_request({"schema_version": 1, "messages": row["messages"]})
    return rows


def run_q5(root):
    from .backend import RewardBackend
    backend = RewardBackend(root)
    try:
        rows = []
        for case in cases():
            start = time.monotonic()
            result = backend.score(case["messages"])
            rows.append({"id": case["id"], "token_ids": backend.token_ids(case["messages"]),
                "reward": result["reward"], "state": result["last_hidden_state"],
                "seconds": time.monotonic() - start})
        # A/B/.../A detects KV-state leakage across independent conversations.
        repeated = backend.score(cases()[0]["messages"])
        repeat_error = max(abs(a-b) for a,b in zip(rows[0]["state"], repeated["last_hidden_state"]))
        repeat_error = max(repeat_error, abs(rows[0]["reward"] - repeated["reward"]))
        report = {"fingerprint": fingerprint(), "rows": rows, "repeat_max_error": repeat_error}
        atomic_json(Path(root) / "q5_probe.json", report)
        return {"cases": len(rows), "repeat_max_error": repeat_error}
    finally:
        backend.close()


def run_reference(root):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    location = Path(root) / "reference" / LOCK["model_revision"]
    tokenizer = AutoTokenizer.from_pretrained(location, local_files_only=True)
    model = AutoModelForCausalLM.from_pretrained(location, local_files_only=True,
        torch_dtype=torch.bfloat16, device_map="auto")
    model.eval()
    rows = []
    for case in cases():
        inputs = tokenizer.apply_chat_template(case["messages"], tokenize=True,
            add_generation_prompt=False, return_tensors="pt", return_dict=True)
        if inputs["input_ids"].shape[1] > 4096:
            raise ValueError("Reference case exceeds 4096 tokens")
        ids = inputs["input_ids"][0].tolist()
        inputs = {k:v.to(model.get_input_embeddings().weight.device) for k,v in inputs.items()}
        with torch.inference_mode():
            output = model.generate(**inputs, max_new_tokens=1, do_sample=False,
                return_dict_in_generate=True, output_scores=True, output_hidden_states=True,
                pad_token_id=tokenizer.eos_token_id)
        rows.append({"id": case["id"], "token_ids": ids,
            "reward": float(output.scores[0][0][0]),
            "state": output.hidden_states[0][-1][0,-1,:].float().cpu().tolist()})
        del output
    atomic_json(Path(root) / "reference_probe.json", {"fingerprint": fingerprint(), "rows": rows})
    return {"cases": len(rows)}


def compare(reference, candidate, *, min_cosine, max_reward_error, max_norm_relative_error):
    import numpy as np
    if not 0 < min_cosine <= 1 or not math.isfinite(min_cosine):
        raise ValueError("min_cosine must be in (0, 1]")
    if not all(math.isfinite(x) and x >= 0 for x in (max_reward_error, max_norm_relative_error)):
        raise ValueError("Error thresholds must be finite and nonnegative")
    identity = fingerprint()
    if reference.get("fingerprint") != identity or candidate.get("fingerprint") != identity:
        raise ValueError("Probe code/model fingerprint is stale; rerun both probes")
    expected = [row["id"] for row in cases()]
    if any([r["id"] for r in report["rows"]] != expected for report in (reference, candidate)):
        raise ValueError("Probe case sets differ")
    metrics = []
    for ref, quant in zip(reference["rows"], candidate["rows"]):
        a, b = np.asarray(ref["state"], dtype=np.float64), np.asarray(quant["state"], dtype=np.float64)
        if a.shape != (8192,) or b.shape != (8192,) or not np.isfinite(a).all() or not np.isfinite(b).all():
            raise ValueError("Invalid probe hidden state")
        if not all(math.isfinite(r["reward"]) for r in (ref, quant)):
            raise ValueError("Invalid probe reward")
        na, nb = np.linalg.norm(a), np.linalg.norm(b)
        if na == 0 or nb == 0:
            raise ValueError("Zero-norm hidden state")
        cosine = float(np.dot(a,b)/(na*nb))
        norm_error = float(abs(nb-na)/na)
        reward_error = abs(ref["reward"]-quant["reward"])
        token_match = ref["token_ids"] == quant["token_ids"]
        passed = token_match and cosine >= min_cosine and norm_error <= max_norm_relative_error and reward_error <= max_reward_error
        metrics.append({"id": ref["id"], "tokens_match": token_match, "cosine": cosine,
            "norm_relative_error": norm_error, "reward_absolute_error": reward_error, "passed": passed})
    rank_agreements = []
    rows_by_id = [{r["id"]:r for r in report["rows"]} for report in (reference, candidate)]
    pairs = {}
    for row in cases():
        if "pair" in row:
            pairs.setdefault(row["pair"], []).append(row["id"])
    for pair, ids in pairs.items():
        if len(ids) != 2:
            raise ValueError("Each ranking pair must have exactly two cases")
        deltas = [r[ids[0]]["reward"]-r[ids[1]]["reward"] for r in rows_by_id]
        rank_agreements.append({"pair": pair, "agrees": bool(np.sign(deltas[0]) == np.sign(deltas[1]))})
    repeat_error = candidate.get("repeat_max_error", float("inf"))
    passed = all(m["passed"] for m in metrics) and all(r["agrees"] for r in rank_agreements) and math.isfinite(repeat_error) and repeat_error <= 1e-4
    return {"fingerprint": identity, "passed": bool(passed), "metrics": metrics,
        "ranking": rank_agreements, "repeat_max_error": repeat_error,
        "thresholds": {"min_cosine": min_cosine, "max_reward_error": max_reward_error,
            "max_norm_relative_error": max_norm_relative_error}}
