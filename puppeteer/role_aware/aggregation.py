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


def select_srdd_artifact(candidates):
    """Select a code artifact with deterministic, label-free runtime checks."""
    from pathlib import Path
    from tasks.evaluator import BenchmarkEvaluator
    from utils.file_utils import read_code

    records = []
    for index, candidate in enumerate(candidates):
        exists = bool(candidate) and Path(str(candidate)).is_file()
        executable = False
        complete = False
        if exists:
            code = read_code(str(candidate))
            complete = BenchmarkEvaluator.srdd_completeness(code) >= 1.0
            executable, _ = BenchmarkEvaluator.srdd_executability(str(candidate))
        records.append({
            "index": index,
            "candidate": candidate,
            "exists": bool(exists),
            "executable": bool(executable),
            "complete": bool(complete),
        })
    if not records:
        return {"mode": "srdd_artifact_selector", "prediction": "",
                "selected_index": None, "records": []}
    selected = max(records, key=lambda item: (
        int(item["exists"]), int(item["executable"]), int(item["complete"]),
        -item["index"],
    ))
    return {"mode": "srdd_artifact_selector",
            "prediction": selected["candidate"] or "",
            "selected_index": selected["index"], "records": records}


def select_gaia_path_answer(answers):
    """Return the last non-empty answer produced on a GAIA reasoning path."""
    for answer in reversed(list(answers or [])):
        if answer is not None and str(answer).strip():
            return answer
    return None


def aggregate_gaia_candidates(
    candidates, *, mode="majority", seed=42, task_id="", question="", verifier=None
):
    """Aggregate open GAIA answers without treating them as MCQ choices."""
    from tasks.evaluator import BenchmarkEvaluator

    cleaned = [
        BenchmarkEvaluator.extract_gaia_answer(candidate)
        for candidate in candidates
        if candidate is not None and str(candidate).strip()
    ]
    if not cleaned:
        return {
            "mode": mode,
            "candidates": list(candidates),
            "prediction": "",
            "normalized": [],
            "votes": {},
            "tie": False,
            "verifier_status": "not_needed",
            "verifier_tokens": 0,
        }
    if mode == "legacy":
        return {
            "mode": mode,
            "candidates": list(candidates),
            "prediction": cleaned[-1],
            "normalized": [],
            "votes": {},
            "tie": False,
            "verifier_status": "not_needed",
            "verifier_tokens": 0,
        }

    normalized = [
        BenchmarkEvaluator.normalize_gaia_string(candidate) for candidate in cleaned
    ]
    votes = Counter(normalized)
    maximum = max(votes.values())
    tied_keys = sorted(key for key, count in votes.items() if count == maximum)
    rng_seed = int(
        hashlib.sha256(f"{seed}:{task_id}".encode()).hexdigest(), 16
    )
    selected_key = random.Random(rng_seed).choice(tied_keys)
    selected_index = max(
        index for index, value in enumerate(normalized) if value == selected_key
    )
    result = {
        "mode": mode,
        "candidates": list(candidates),
        "prediction": cleaned[selected_index],
        "normalized": normalized,
        "votes": dict(votes),
        "tie": len(tied_keys) > 1,
        "verifier_status": "not_needed",
        "verifier_tokens": 0,
    }
    if mode == "majority_verifier" and len(tied_keys) > 1:
        if verifier is None:
            raise ValueError("majority_verifier requires an explicit verifier")
        tied_indices = [
            index for index, value in enumerate(normalized) if value in tied_keys
        ]
        prompt = (
            "Select the most defensible candidate answer to the GAIA question. "
            "Return only CANDIDATE: <number>; do not solve with hidden gold data.\n"
            f"Question:\n{question}\nCandidates:\n"
            + "\n".join(
                f"{index + 1}. {cleaned[index]}" for index in tied_indices
            )
        )
        try:
            response, tokens = verifier(prompt)
            result["verifier_tokens"] = tokens
            match = re.search(r"(?i)candidate\s*:\s*(\d+)", str(response))
            choice = int(match.group(1)) - 1 if match else -1
            if choice in tied_indices:
                result["prediction"] = cleaned[choice]
                result["verifier_status"] = "ok"
            else:
                result["verifier_status"] = "invalid_fallback"
        except Exception as error:
            result["verifier_status"] = "error_fallback"
            result["verifier_error_type"] = type(error).__name__
    return result


def aggregate_musique_candidates(
    candidates, *, mode="majority", seed=42, task_id="", question="", verifier=None,
    fallback_extractor=None,
):
    """Aggregate open MuSiQue answers while preserving candidate-only selection."""
    from tasks.evaluator import BenchmarkEvaluator

    cleaned = []
    original_indices = []
    fallback_extractions = []
    for index, candidate in enumerate(candidates):
        if candidate is None or not str(candidate).strip():
            continue
        canonical = BenchmarkEvaluator.extract_musique_answer(candidate)
        if not canonical and fallback_extractor is not None:
            extracted = fallback_extractor(candidate, index)
            if isinstance(extracted, dict):
                canonical = str(extracted.get("answer") or "").strip()
                fallback_extractions.append(
                    {
                        "candidate_index": index,
                        "status": extracted.get("status"),
                        "answer": canonical,
                        "model": extracted.get("model"),
                        "tokens": int(extracted.get("tokens") or 0),
                    }
                )
            else:
                canonical = str(extracted or "").strip()
        if canonical:
            cleaned.append(canonical)
            original_indices.append(index)
    if not cleaned:
        return {
            "mode": mode,
            "candidates": list(candidates),
            "prediction": "",
            "normalized": [],
            "votes": {},
            "tie": False,
            "selected_candidate_index": None,
            "verifier_status": "not_needed",
            "verifier_tokens": 0,
            "fallback_extractions": fallback_extractions,
        }
    normalized = [
        BenchmarkEvaluator.normalize_qa_answer(candidate) for candidate in cleaned
    ]
    if mode == "legacy":
        selected_index = len(cleaned) - 1
        return {
            "mode": mode,
            "candidates": list(candidates),
            "prediction": cleaned[selected_index],
            "normalized": normalized,
            "votes": {},
            "tie": False,
            "selected_candidate_index": original_indices[selected_index],
            "verifier_status": "not_needed",
            "verifier_tokens": 0,
            "fallback_extractions": fallback_extractions,
        }

    votes = Counter(normalized)
    maximum = max(votes.values())
    tied_keys = sorted(key for key, count in votes.items() if count == maximum)
    rng_seed = int(hashlib.sha256(f"{seed}:{task_id}".encode()).hexdigest(), 16)
    selected_key = random.Random(rng_seed).choice(tied_keys)
    selected_index = max(
        index for index, value in enumerate(normalized) if value == selected_key
    )
    result = {
        "mode": mode,
        "candidates": list(candidates),
        "prediction": cleaned[selected_index],
        "normalized": normalized,
        "votes": dict(votes),
        "tie": len(tied_keys) > 1,
        "selected_candidate_index": original_indices[selected_index],
        "verifier_status": "not_needed",
        "verifier_tokens": 0,
        "fallback_extractions": fallback_extractions,
    }
    if mode == "majority_verifier" and len(tied_keys) > 1:
        if verifier is None:
            raise ValueError("majority_verifier requires an explicit verifier")
        tied_indices = [
            index for index, value in enumerate(normalized) if value in tied_keys
        ]
        prompt = (
            "Select the candidate best supported by its cited MuSiQue paragraphs. "
            "Return only CANDIDATE: <number>. Do not introduce a new answer.\n"
            f"Question:\n{question}\nCandidates:\n"
            + "\n".join(
                f"{index + 1}. {cleaned[index]}" for index in tied_indices
            )
        )
        try:
            response, tokens = verifier(prompt)
            result["verifier_tokens"] = tokens
            match = re.search(r"(?i)candidate\s*:\s*(\d+)", str(response))
            choice = int(match.group(1)) - 1 if match else -1
            if choice in tied_indices:
                result["prediction"] = cleaned[choice]
                result["selected_candidate_index"] = original_indices[choice]
                result["verifier_status"] = "ok"
            else:
                result["verifier_status"] = "invalid_fallback"
        except Exception as error:
            result["verifier_status"] = "error_fallback"
            result["verifier_error_type"] = type(error).__name__
    return result
