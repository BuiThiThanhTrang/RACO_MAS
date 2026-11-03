# Adaptive no-train routing harness: implementation plan

## 1. Objective

Improve the frozen Jev/Sol orchestrators without training a routing policy. The
router must remain state-adaptive, must transfer across multi-step benchmarks,
and must remain usable when unrelated or noisy agents are added to the pool.

The harness is deliberately **not** a fixed role graph. It only exposes agents
whose declared inputs and capabilities are useful for the current state; Jev or
Sol still chooses the next role and may revisit a role after substantive state
change.

## 2. Baseline checkpoint

The complete no-train benchmark implementation before this change is preserved
on branch `RACO_no_train` at commit `2280d25`.

## 3. Implementation phases

### Phase A — typed, benchmark-neutral blackboard

Add a compact planner state with stable fields:

- objective and task type;
- accumulated evidence and cited resource identifiers;
- intermediate, candidate, and final answers;
- last action status, failure/stall state, and role history;
- remaining depth/width;
- diagnosed next need and terminal readiness.

Raw previous outputs remain available for backward compatibility, but can be
bounded to the most recent calls. This reduces prompt growth and lets the same
planner prompt operate on MuSiQue, GAIA, and future multi-step benchmarks.

### Phase B — capability and precondition gating

Create a reusable `CapabilityRoutingHarness` shared by decision-model and LLM
planners. It applies three identity-safe filters:

1. availability and the existing task-specific feasibility guard;
2. explicit benchmark-scope mismatch filtering (generic agents are retained);
3. need-aware capability shortlisting using only role cards and routing profiles.

The shortlist always has a safe fallback and never reads gold answers, gold hop
counts, model identity, or benchmark labels excluded from the routing input.

### Phase C — deterministic terminal contract

Expose STOP only when the current artifact satisfies the benchmark-neutral
terminal contract. For MuSiQue this means a committed final answer, cited
evidence when configured, and no failed last action. For GAIA it means a valid
candidate artifact and no failed last action. Semantic diagnosis remains
advisory and cannot override a failed deterministic contract.

### Phase D — conditional width and branching

Treat `max_width` as capacity, not a target:

- start with one root path when `initial_paths: 1`;
- continue with one agent during normal progress;
- expose two-agent continuation/fork bundles only after conflict, failure/stall,
  or diagnosed no-progress;
- cap branch choices so Jev never exceeds its option limit.

GraphReasoning already interprets additional non-root selections as forks that
inherit the current path state, so no fixed subgraph is introduced.

### Phase E — noisy-pool robustness analysis

Add an offline comparison script for a clean-pool run and a mixed/noisy-pool
run. Report task-score degradation, noisy-agent selection rate, call inflation,
useful-call ratio, redundant-transition rate, and recovery success. Noise roles
are supplied explicitly to avoid guessing from model names.

### Phase F — configuration and verification

Enable the harness in the dynamic MuSiQue Jev and Sol configurations. Add unit
tests for scope filtering, generic-role preservation, need-aware shortlisting,
typed blackboard construction, terminal gating, and conditional branch options.
Existing configurations remain unchanged unless they opt in.

## 4. Expected effects and evaluation criteria

These are hypotheses to test on identical task IDs, not guaranteed gains.

| Change | Expected direct effect | Primary measurement |
|---|---|---|
| Typed blackboard | Less prompt noise; more consistent state interpretation | planner tokens, invalid decisions |
| Scope gate | Fewer unrelated agents selected in mixed pools | noise selection rate |
| Capability shortlist | Smaller Jev choice space; fewer low-value calls | option count, useful-call ratio |
| Terminal gate | Fewer premature STOP decisions | unsupported/empty final rate |
| Conditional branching | Lower call cost on easy tasks while preserving recovery paths | calls/task, fork rate, answer score |
| State-delta retry rule | Less repeated work without blocking useful retries | redundant transition rate |

A change is accepted only if it does not materially reduce semantic answer
accuracy on the clean pool and improves at least one of cost, collaboration, or
robustness. Recommended evaluation:

1. same 104 MuSiQue validation IDs, Jev, clean pool;
2. same IDs and settings, Sol;
3. same two runs after mixing MuSiQue with MMLU-Pro/SRDD agents;
4. report answer EM/F1 and semantic correctness alongside collaboration score,
   paper-compatible support F1, calls/task, fork rate, and noise selection rate.

## 5. Compatibility and risks

- The harness is opt-in under `policy.routing_harness`.
- Dataset-scope filtering acts only on explicit role-card text; unscoped generic
  roles remain eligible.
- Capability matching uses conservative fallbacks so incomplete role cards do
  not make the action space empty.
- A stricter terminal contract can increase depth usage. Compare both accuracy
  and cost rather than accuracy alone.
- Conditional branching may miss benefits from unconditional self-consistency;
  keep the old configuration as the ablation baseline.

## 6. Implementation status

- [x] Baseline checkpoint pushed.
- [x] Typed blackboard and shared capability harness.
- [x] Deterministic terminal gate.
- [x] Conditional branching and adaptive root width.
- [x] Robustness comparison script.
- [x] Jev/Sol config updates.
- [x] Unit and regression verification (59 focused tests passed).

## 7. Running the evaluation

Run the updated Jev condition on the same MuSiQue slice used by the baseline:

```powershell
python main.py MuSiQue validation `
  --config config/experiments/decision_musique_dynamic_jev_gemini.yaml `
  --policy_mode frozen `
  --data_limit 104 `
  --run_id "musique_adaptive_jev_gemini"
```

Run the Sol condition by replacing the config with
`config/experiments/frozen_musique_dynamic_sol_gemini.yaml`.

After producing a clean-pool and a mixed-pool run over identical task IDs:

```powershell
python scripts/analyze_routing_robustness.py `
  runs/musique_adaptive_jev_gemini `
  runs/musique_adaptive_jev_gemini_mixed `
  --noise-role "Quantitative & Formal Reasoner" `
  --noise-role "Scientific Claim Verifier" `
  --output runs/musique_adaptive_jev_gemini_mixed/analysis/robustness.json
```

Repeat `--noise-role` for every intentionally injected role. The script matches
only shared task IDs and reports noisy-minus-clean deltas, so different task
subsets cannot silently distort the comparison.

## 8. Verification record

- Both updated Jev and Sol YAML files pass runtime config validation.
- 59 focused routing, planner, MuSiQue, and analyzer tests pass.
- The full repository suite runs 215 tests: 205 pass and 10 pre-existing
  train/baseline assertions fail. Those failures are unchanged from the baseline
  checkpoint and concern legacy REINFORCE test fixtures, an old split-size lock,
  missing evidence-scope entries for newly added roles, and a source-text count
  assertion. They are outside the frozen no-train routing path changed here.
- The robustness analyzer passed a 100-task self-comparison: every delta was
  zero and call inflation was exactly 1.0, validating task alignment and metric
  aggregation without any API calls.
