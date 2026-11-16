# Dynamic Router Lab

`routerlab` is a standalone research codebase for measuring dynamic executor routing
in a multi-agent code-generation system.

The initial benchmark is MBPP+ through EvalPlus. All expert roles use the same SLM
backbone by default: `Qwen/Qwen2.5-Coder-3B-Instruct`.

## System contract

The dynamic workflow is router-first:

```text
MBPP+ canonical prompt (task description + one sample assertion)
   |
Router sees the fixed five-action space
   |
selected executor (or STOP)
   |
first code -> automatic Base Tests -> public feedback
   |
public state -> Router -> {one executor, STOP}
                         |
                         +-> reviewer: update public findings
                         +-> code edit: automatic Base Tests -> new public state
                         +-> STOP: one private final Base+Plus evaluation
```

The router selects only:

```text
coder | logic_reviewer | edge_case_analyst | repair_agent | STOP
```

It cannot select a test action. Base Tests are a deterministic controller transition
after a code change. This removes test scheduling as a confounder when comparing
routers.

## Experimental methods

| ID | Method | Control policy |
|---|---|---|
| E1 | Single-agent self-repair | Generate, inspect Base feedback, repair until pass or budget |
| E2 | Fixed multi-agent | Coder → Logic Reviewer → Edge-case Analyst → Repair Agent |
| E3-H | Dynamic heuristic | Same fixed action space; deterministic policy chooses executor/STOP |
| E3-M | Dynamic generative router | An SLM chooses executor/STOP from public state |
| E3-DM | Typed decision router | JEV, Luna, Laya or another registered model answers the same typed choice question |

E3 has no mandatory Coder bootstrap and no action mask. Every routing state exposes the
same five actions. `repair_agent` before code is a charged no-op, and `STOP` before code
ends with no submission. Every non-STOP executor action consumes one of seven shared
steps; `STOP` consumes no step. At zero remaining steps the full action space is still
shown: selecting an executor is logged as a budget violation and is not executed. The
controller also enforces Base-Test and wall-time safety limits. Token usage is measured
but is not an episode-level stopping condition; executor output is bounded per call.

## Expert pool

- **Coder:** creates a complete candidate implementation.
- **Logic Reviewer:** reports algorithmic, contract, and complexity problems without
  editing code.
- **Edge-case Analyst:** reports supported boundary and structural cases without
  inventing requirements.
- **Repair Agent:** edits code using the task, current code, Base feedback, and reviewer
  findings.

All experts share one model client and checkpoint. Differences therefore come from
their role prompts and routing sequence rather than different backbone capability.
Routers receive a separate, normalized capability card for every action. These cards
describe inputs, effects, availability, and step cost without exposing executor output-
format instructions. See [`docs/SYSTEM_ARCHITECTURE.md`](docs/SYSTEM_ARCHITECTURE.md).

## Evaluation isolation

| Information | Experts/router | Evaluator/private log |
|---|---:|---:|
| Canonical task prompt, including one sample assertion, and entry point | Yes | Yes |
| Current code | Yes | Yes |
| Sanitized Base status/counts | Yes | Yes |
| Base inputs or expected outputs | No | Yes |
| Plus inputs/results | No | Yes |
| Canonical solution | No | Yes |

EvalPlus already places one example assertion in every canonical MBPP+ prompt. The
full Base suite runs automatically after each changed code artifact and exposes only
sanitized status/count feedback. Plus Tests run exactly once, after routing has
terminated, on the final submitted artifact. If several versions exist, the newest
current version is submitted even when its Base score is not higher. The harness never
rolls back to a Base-best version or rescues a poor routing decision.

## Metrics

Quality metrics:

- initial/final Base pass rate;
- final Plus pass rate.

Because Plus Tests are final-only, initial Plus pass rate, Plus improvement, and
Plus-based repair recovery are intentionally emitted as `null`; the harness does not
infer metrics from unobserved private evaluations.

Efficiency and routing metrics:

- model calls, input/output tokens, Base-Test executions, wall time, and reported cost;
- cost per solved task;
- executor utilization;
- router decisions, invalid-decision and budget-violation rates, no-op and
  no-submission rates, steps to first code, and termination reasons.

## Output of each run

```text
runs/<run_name>/<run_id>/
  config.json          resolved configuration
  manifest.json        dataset hash and ordered task IDs
  public_trace.jsonl   actions, public Base outcomes, usage, budgets
  private_eval.jsonl   initial Base-only and final Base+Plus evaluation
  solutions.jsonl      final submitted code
  metrics.json         aggregate metrics, updated after each task
  errors.json          recoverable per-task failures
```

`public_trace.jsonl` contains no Plus-Test fields. Keep `private_eval.jsonl` unavailable
to routers and any online training loop.

## Installation on Windows PowerShell

From the repository root:

```powershell
cd D:\SideProject\ChatDev\routerlab
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -e .
```

The reported full v5 runs keep the router on the client machine and serve the shared
Qwen executor through one authenticated, scale-to-zero Modal T4 endpoint. The Modal
image pins Qwen revision `d3a7bedd480bc27342f4336c71e86858b25df935`.
EvalPlus executes generated code in isolated worker processes; run only code produced
within the controlled experiment environment.

On Windows, RouterLab automatically uses its EvalPlus compatibility backend because
EvalPlus 0.3.1 directly depends on Unix-only `resource` and `SIGALRM`. The benchmark
inputs, expected outputs, and special oracles remain unchanged; worker timeouts and
memory limits are enforced with Windows processes, timers, and `psutil`. Use this same
backend for every compared method and record the operating system in reported results.
When loading a local ModernBERT-based Laya checkpoint, RouterLab also scopes the
documented Windows Transformers initialization workaround to `laya.load()`; the
original Transformers method is restored immediately afterward.

## Full v5 protocol and commands

The locked full manifest contains 378 tasks and dataset hash
`92743def42b30b354a30898e4fa33fb0`. The three primary router runs use
`prompt_version: v5-capability-cards-current-code`, the same seven-step executor
budget, 20 Base executions, 900-second wall-time limit, Modal Qwen expert, and final
current-code submission rule.

```powershell
routerlab run --config configs\mbpp_e3_jev_modal.yaml --manifest manifests\mbpp_plus_full.json --run-id full-jev-v5-s42
routerlab run --config configs\mbpp_e3_luna_openrouter_modal.yaml --manifest manifests\mbpp_plus_full.json --run-id full-luna-v5-s42
routerlab run --config configs\mbpp_e3_laya_modal.yaml --manifest manifests\mbpp_plus_full.json --run-id full-laya-typed-v5-pinned-s42
```

JEV and Luna use `OPENROUTER_API_KEY`; all three configs use `MODAL_PROXY_TOKEN`.
Laya is pinned to revision `989e28febe5886389496fe13a8036c62b2a20ae0`.
Before the Laya run, select the cache that contains that revision:

```powershell
$env:HF_HOME="D:\ModelCache\huggingface"
$env:HF_HUB_CACHE="D:\ModelCache\huggingface\hub"
$env:HF_HUB_OFFLINE="1"
$env:HF_HUB_DISABLE_SYMLINKS_WARNING="1"
```

To test whether CPU RAM can hold both Laya and the Qwen expert without running a
task, close memory-heavy applications and use the BF16 local configuration:

```powershell
routerlab check-models --config configs\mbpp_e3_laya_local.yaml
```

If both models load, run one smoke task before attempting a larger manifest:

```powershell
routerlab run --config configs\mbpp_e3_laya_local.yaml --limit 1 --run-id local-smoke1 --fast-check --fail-fast
```

For machines that cannot hold both models, the low-cost Modal deployment keeps
the selected router local or remote and serves the same Qwen checkpoint on one
scale-to-zero T4. See
[`docs/MODAL_QWEN.md`](docs/MODAL_QWEN.md).

Typed routers now share one `decision_model` interface and registry. Laya, JEV,
OpenAI Decisions and DREX use the same state and action contract; only their wire
adapters differ. See [`docs/DECISION_MODELS.md`](docs/DECISION_MODELS.md).

For a quick environment check, create a 5-task manifest or override the limit:

```powershell
routerlab run --config configs\mbpp_e3_heuristic.yaml --limit 5 --run-id smoke5
```

Use `--fast-check` only for debugging. The normal protocol evaluates every Base/Plus
case so that per-task passed-test counts remain comparable. Use `--fail-fast` while
debugging; omit it for long runs so one failed task does not discard the rest.

## Fair-comparison checklist

For every reported table, keep constant:

- task manifest and EvalPlus dataset hash;
- expert checkpoint, decoding settings, role prompts, and runtime environment;
- seven executor steps, Base-Test and wall-time budgets, plus the same per-call
  executor output-token limit; router overhead is measured separately;
- initial generation and final submission rules;
- random seed and hardware class.

Run at least three seeds if stochastic decoding is introduced. The supplied configs
use greedy decoding (`temperature: 0`) to support a deterministic first pilot.

## Development verification

```powershell
$env:PYTHONPATH='D:\SideProject\ChatDev\routerlab\src'
python -m unittest discover -s tests -v
python -m compileall -q src tests
```

Implementation phases and acceptance criteria are in
[`docs/IMPLEMENTATION_PLAN.md`](docs/IMPLEMENTATION_PLAN.md).
The full normative harness, information-flow, router-input, executor-input, evaluation,
logging, and submission contract is in
[`docs/SYSTEM_ARCHITECTURE.md`](docs/SYSTEM_ARCHITECTURE.md).

## Upstream components

- EvalPlus: <https://github.com/evalplus/evalplus>
- Qwen2.5-Coder-3B-Instruct: <https://huggingface.co/Qwen/Qwen2.5-Coder-3B-Instruct>
- Laya: <https://github.com/NandhaKishorM/laya>
