# Implementation plan: MBPP+ dynamic executor routing

## 1. Scope and research contract

Build an independent experiment package for comparing fixed and dynamic multi-agent
workflows on MBPP+. The router has one responsibility: select one executor from the
pool or select `STOP`.

Fixed router action space at every step:

- `coder`
- `logic_reviewer`
- `edge_case_analyst`
- `repair_agent`
- `STOP`

`RUN_BASE_TESTS` is deliberately not an action. The canonical EvalPlus prompt exposes
one sample assertion. The controller runs the full Base suite after every real code
change and exposes only sanitized feedback. Plus Tests are private evaluation data
and run exactly once on the final submitted artifact.

## 2. Implementation phases

1. **Isolation and contracts**
   - Create the standalone `routerlab` package with no legacy-framework dependency.
   - Define typed task, code, feedback, routing-state, decision, budget, trace, and
     result contracts.
   - Validate the exact executor/action set.

2. **Expert backbone and executor pool**
   - Use one shared `Qwen/Qwen2.5-Coder-3B-Instruct` checkpoint for every expert.
   - Implement role prompts for Coder, Logic Reviewer, Edge-case Analyst, and Repair
     Agent.
   - Provide deterministic replay clients for tests and OpenAI-compatible and local
     Transformers clients for experiments.

3. **Evaluation boundary**
   - Load MBPP+ through EvalPlus and record its dataset hash.
   - Expose only the canonical prompt (description plus one sample assertion) and
     entry point to agents.
   - Keep canonical solutions, inputs, expected outputs, and Plus results inside the
     evaluator.
   - Log public routing traces separately from private final Plus evaluation.

4. **Comparable workflows**
   - E1: single agent with Base-Test-driven self-repair.
   - E2: Coder, Logic Reviewer, Edge-case Analyst, Repair Agent in a fixed sequence.
   - E3: router-first execution with the full fixed action space at every step.
   - Do not mask Repair or STOP before code: they become a charged no-op and a failed
     no-submission respectively.
   - Implement heuristic, generative-model, and typed decision-model routers through
     the same routing interface. The registered comparison set includes JEV, Luna and
     pinned local Laya; DREX adapters remain available for later experiments.

5. **Budget and controller policy**
   - Apply the same seven-step executor budget, Base-Test, wall-time, and
     per-executor-call output limits. Router calls are measured separately.
   - Charge one step for every non-STOP executor action and zero steps for STOP.
   - Keep the full action space at zero remaining steps; log an unexecutable executor
     choice as a budget violation and terminate without running it.
   - Reserve evaluation capacity for the final artifact when one exists.
   - Let E3 stop through router `STOP`; allow controller-forced termination only when
     a hard budget prevents another valid decision or execution.
   - Submit exactly the current code at termination. Do not use Base Tests to roll
     back to an older version or let the harness rescue a routing regression.

6. **Metrics and reproducibility**
   - Record initial/final Base pass rates, final Plus pass rate, calls, tokens, Base
     executions, latency, cost, executor utilization, invalid router rate,
     no-op/no-submission rates, steps to first code, and termination reasons.
   - Generate one task manifest and reuse it for every method.
   - Save config, manifest, traces, final solutions, errors, and incremental metrics.

7. **Verification and pilot**
   - Unit-test action contracts, automatic Base Tests, router `STOP`, Plus-data
     isolation, logging, and metrics.
   - Run smoke tasks for environment validation, then the shared 30-task pilot.
   - Inspect failures before scaling JEV, Luna and pinned Laya to the locked 378-task
     `manifests/mbpp_plus_full.json` manifest under protocol v5.

## 3. Acceptance criteria

- `routerlab` contains no import or runtime dependency on the previous framework.
- A router cannot request tests or access Plus-Test information.
- Every changed code hash is Base-tested automatically.
- E1, E2, E3-heuristic, E3-generative, and the JEV/Luna/Laya E3 decision routers run
  from separate versioned YAML configs.
- All methods can use the same manifest and expert checkpoint.
- Unit tests pass before any benchmark run.
