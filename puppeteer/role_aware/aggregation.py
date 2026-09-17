"""Label-free multiple-choice aggregation and reproducible tie breaking."""
from collections import Counter
import hashlib
import random
import re
from role_aware.audit_trace import digest

def normalize_choice(value, choices="ABCDEFGHIJ"):
    text = str(value or "").strip()
    if text.upper() in choices and len(text) == 1:
        return text.upper()
    patterns = [r"(?i)(?:final\s+answer|answer|option|choice)\s*(?:is\s*)?[:=]?\s*\(?([A-J])\b",
                r"\(([A-Ja-j])\)", r"^([A-Ja-j])\s*[:.)]"]
    for pattern in patterns:
        matches = list(re.finditer(pattern, text))
        if matches:
            result = matches[-1].group(1).upper()
            return result if result in choices else None
    return None

def aggregate_candidates(candidates, *, mode="majority", choices="ABCDEFGHIJ",
                         seed=42, task_id="", question="", verifier=None):
    if mode not in {"legacy", "majority", "majority_verifier"}:
        raise ValueError(f"Unknown aggregation mode: {mode}")
    if mode == "legacy":
        from tasks.evaluator import BenchmarkEvaluator
        normalized = [str(BenchmarkEvaluator.extract_choice_answer(v)).strip() for v in candidates if v is not None]
    else:
        normalized = [normalize_choice(v, choices) for v in candidates]
    votes = Counter(v for v in normalized if v is not None)
    ties = [v for v, count in votes.items() if count == max(votes.values())] if votes else []
    rng_seed = int(hashlib.sha256(f"{seed}:{task_id}".encode()).hexdigest(), 16)
    prediction = (ties[-1] if mode == "legacy" else random.Random(rng_seed).choice(sorted(ties))) if ties else ""
    result = dict(mode=mode, candidates=list(candidates), candidate_hash=digest(candidates),
                  normalized=normalized, votes=dict(votes), tie=len(ties) > 1, tied_answers=sorted(ties),
                  invalid_count=sum(v is None for v in normalized), prediction=prediction,
                  verifier_status="not_needed", verifier_tokens=0)
    if mode == "majority_verifier" and len(ties) > 1:
        if verifier is None:
            raise ValueError("majority_verifier requires an explicit verifier")
        prompt = ("Choose one of the tied answers. Return only its letter.\n"
                  f"Question:\n{question}\nCandidates:\n" + "\n".join(map(str, candidates)) +
                  "\nAllowed tied answers: " + ", ".join(sorted(ties)))
        try:
            response, tokens = verifier(prompt)
            result["verifier_tokens"] = tokens
            choice = normalize_choice(response, choices)
            result["verifier_status"] = "ok" if choice in ties else "invalid_fallback"
            if choice in ties:
                result["prediction"] = choice
        except Exception as error:
            result["verifier_status"] = "error_fallback"
            result["verifier_error_type"] = type(error).__name__
    return result
