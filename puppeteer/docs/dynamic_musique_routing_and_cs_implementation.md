# Dynamic MuSiQue Routing and Collaboration Evaluation

## Objective

Keep the existing MuSiQue answer, support, and collaboration metrics unchanged,
while adding an experimental routing mode that lets the orchestrator choose roles
from the current semantic state instead of following a hand-authored stage graph.
The same state and feasibility rules are used for Jev and Sol so their routing
quality can be compared fairly.

## Implementation plan and completed changes

### 1. Preserve the existing benchmark

- `musique_stage_v1` remains available and retains its previous fixed-stage
  behavior for reproducibility.
- Existing answer EM/F1, trace-union support precision/recall/F1, and the current
  MuSiQue collaboration metrics are not redefined. A separate
  `paper_compatible_support_*` family evaluates only the final selected path's
  terminal support prediction.
- The new configuration is opt-in through `musique_dynamic_v2`.

### 2. Route dynamically from semantic state

`musique_dynamic_v2` exposes all technically feasible roles. It hard-masks only:

1. a role that exceeded its per-path call budget;
2. a repeated role when the usable semantic state did not change;
3. Verifier or Integrator before any candidate answer exists.

It does not force Retriever → Bridge → Composition → Verifier → Integrator.
The router may revisit a role after paragraph references, intermediate answers,
candidate answers, or final answers change. Merely rephrasing the same output is
not counted as progress.

The dynamic routing input excludes MuSiQue `hop_count` and `category`, because
these are benchmark annotations that would not be known to a real router.

### 3. Jev and Sol routing protocols

Jev uses two typed Decisions calls on non-root states:

1. **Semantic diagnosis:** direct-answer alignment, evidence sufficiency,
   dependency resolution based only on the generated plan, candidate conflict,
   recent progress, and the dominant next capability need.
2. **Role selection:** choose one role or STOP from the feasible candidate set,
   with the diagnosis attached to the route state.

Root path creation remains sequential so Jev never needs to enumerate an
exponential bundle over the full pool. Width is an upper bound.

Sol receives the same filtered task, routing state, role profiles, and feasible
candidate set in one structured planner call. Its prompt asks it to perform the
same semantic diagnosis internally before choosing.

### 4. Integrator and candidate parsing

- The Answer Integrator is constrained to selecting or normalizing an answer
  already present in accumulated state.
- It returns JSON with `status`, `supporting_paragraph_ids`, `final_answer`, and
  `missing_subquestion`.
- `NEEDS_EVIDENCE` and `CONFLICT` explicitly return no final answer, allowing the
  dynamic router to recover with another role.
- A second Integrator call is allowed after meaningful evidence/candidate
  progress, but is masked on an unchanged semantic state.
- Fenced JSON after `FINAL ANSWER:` that contains only an intermediate answer can
  no longer be canonicalized to the literal answer `json`.
- Structured answer fields and deterministic parsing remain the fast path. If
  both fail for an answer-capable role, a cached, gold-blind Luna extractor may
  return `FINAL`, `CANDIDATE`, or `NO_ANSWER`. Any extracted answer must be an
  exact contiguous quote from the actor output; the extractor cannot solve or
  rewrite the answer.

### 5. One post-task call for collaboration and semantic outcome

- Official MuSiQue EM/F1 remain unchanged and are written as
  `official_correct`, `answer_em`, and `answer_f1`.
- After `prediction_committed`, `router_cs_semantic_v2` makes one structured
  Luna call. It returns both router collaboration scores and an answer verdict
  of `EQUIVALENT`, `NOT_EQUIVALENT`, or `UNCERTAIN`.
- The prompt presents `collaboration_trace` and `answer_evaluation` as separate
  sections. Collaboration rationales must ignore correctness; semantic judging
  may compare only the committed prediction with accepted answers and may not
  select another answer from an intermediate trace.
- Gold is never exposed to routing or actors. The semantic verdict may be used
  by route experience, profile evidence, and training reward only after the task
  has ended.
- `UNCERTAIN` and provider errors fail closed to official EM unless the config
  explicitly requests `failure_policy: raise`.
- Frozen naive runs therefore change only their reporting. Evolving or trained
  runs avoid storing false route/profile failures caused solely by answer
  surface form. When used for training/profile evidence, this is deliberately a
  task-level outcome rather than per-agent causal attribution.

### 6. MultiAgentBench-style router collaboration score

The same combined judge is executed online after answer commitment so future
runs write CS and semantic outcome together. The offline evaluator remains for
old audit logs; it reads `candidates.json`, `events.jsonl`, and `evaluation.json`
without rerunning actors.

It produces:

- **Routing Planning Score:** 1–5;
- **State-Handoff Communication Score:** 0–5, strictly 0 when no cross-role
  handoff exists;
- **CS raw:** arithmetic mean of the two components;
- **CS100:** `CS raw × 20`.

This is inspired by MultiAgentBench but adapted to a router-only system: it
scores task-state routing and handoff quality, not explicit natural-language task
assignment among autonomous agents. Final correctness remains excluded from CS
scoring even though the same API response also contains a semantic verdict.
Online results are stored under `final_metrics.router_cs`; offline results are
stored under `analysis/` as a sidecar and appear in a separate
`multiagentbench_cs` namespace in the aggregate report.

## Experiment configurations

- Jev: `config/experiments/decision_musique_dynamic_jev_gemini.yaml`
- Sol: `config/experiments/frozen_musique_dynamic_sol_gemini.yaml`

Both use the Gemini 2.5 Flash actor pool, W2D5, the same dynamic guard, and the
same answer aggregation, gold-blind extraction fallback, and semantic outcome
judge. Only the router changes.

Run Jev:

```powershell
python main.py MuSiQue validation `
  --config config/experiments/decision_musique_dynamic_jev_gemini.yaml `
  --policy_mode frozen `
  --run_id musique_dynamic_jev_gemini
```

Run Sol:

```powershell
python main.py MuSiQue validation `
  --config config/experiments/frozen_musique_dynamic_sol_gemini.yaml `
  --policy_mode frozen `
  --run_id musique_dynamic_sol_gemini
```

Evaluate router collaboration and semantic correctness from an existing run.
Use `--overwrite` to replace legacy `router_cs_v1` sidecars:

```powershell
python scripts/evaluate_multiagentbench_cs.py runs/musique_dynamic_jev_gemini `
  --model gpt-6-luna-openrouter `
  --reasoning_effort low `
  --overwrite
```

Inspect judge inputs without making API calls:

```powershell
python scripts/evaluate_multiagentbench_cs.py runs/musique_dynamic_jev_gemini `
  --dry_run --limit 1
```

Aggregate all MuSiQue metrics plus the optional CS sidecar:

```powershell
python scripts/analyze_musique_run.py runs/musique_dynamic_jev_gemini
```

## Comparison protocol

Run both routers over the identical ordered sample range and actor pool. Compare:

1. answer EM/F1, trace-union support F1, and paper-compatible support F1;
2. the existing deterministic MuSiQue collaboration metrics;
3. router CS and its two components;
4. number of calls, repeated-role masks, fallback rate, and planner tokens;
5. failure slices where the actor output was weak but the router either recovered
   or stopped prematurely.

This separation avoids using answer correctness twice and makes it possible to
distinguish actor capability from routing quality.
