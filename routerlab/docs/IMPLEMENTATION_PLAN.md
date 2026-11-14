# Implementation plan: MBPP+ dynamic executor routing

## 1. Scope and research contract

Build an independent experiment package for comparing fixed and dynamic multi-agent
workflows on MBPP+. The router has one responsibility: select one executor from the
pool or select `STOP`.

Legal router actions:

- `coder`
- `logic_reviewer`
- `edge_case_analyst`
- `repair_agent`
- `STOP`

`RUN_BASE_TESTS` is deliberately not an action. The controller runs Base Tests after
every real code change. Plus Tests are private evaluation data and run only on the
initial and final artifacts.

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
   - Expose only prompt and entry point to agents.
   - Keep canonical solutions, inputs, expected outputs, and Plus results inside the
     evaluator.
   - Log public routing traces separately from private initial/final evaluation.

4. **Comparable workflows**
   - E1: single agent with Base-Test-driven self-repair.
   - E2: Coder, Logic Reviewer, Edge-case Analyst, Repair Agent in a fixed sequence.
   - E3: mandatory Coder bootstrap followed by dynamic executor/`STOP` decisions.
   - Implement heuristic and model-based routers through the same routing interface.

5. **Budget and controller policy**
   - Apply the same model-call, token, Base-Test, and wall-time limits.
   - Reserve evaluation capacity for the final artifact.
   - Let E3 stop through router `STOP`; allow controller-forced termination only when
     a hard budget prevents another valid decision or execution.
   - Select fallback code using Base Tests only, never Plus Tests.

6. **Metrics and reproducibility**
   - Record initial/final Base and Plus pass rates, absolute and normalized Plus
     improvement, repair recovery, calls, tokens, Base executions, latency, cost,
     executor utilization, invalid router rate, and termination reasons.
   - Generate one task manifest and reuse it for every method.
   - Save config, manifest, traces, final solutions, errors, and incremental metrics.

7. **Verification and pilot**
   - Unit-test action contracts, automatic Base Tests, router `STOP`, Plus-data
     isolation, logging, and metrics.
   - Run 5 tasks for environment validation, then 50 tasks for the first pilot.
   - Inspect failures before scaling to the full selected manifest.

## 3. Acceptance criteria

- `routerlab` contains no import or runtime dependency on the previous framework.
- A router cannot request tests or access Plus-Test information.
- Every changed code hash is Base-tested automatically.
- E1, E2, E3-heuristic, and E3-model run from separate versioned YAML configs.
- All methods can use the same manifest and expert checkpoint.
- Unit tests pass before any benchmark run.
