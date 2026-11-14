# Dynamic Router Lab

`routerlab` is a standalone research codebase for measuring dynamic executor routing
in a multi-agent code-generation system. It is independent of the legacy Puppeteer
package in the parent repository.

The initial benchmark is MBPP+ through EvalPlus. All expert roles use the same SLM
backbone by default: `Qwen/Qwen2.5-Coder-3B-Instruct`.

## System contract

The dynamic workflow is:

```text
MBPP+ task
   |
mandatory Coder bootstrap
   |
controller runs Base Tests automatically
   |
public state -> Router -> {one executor, STOP}
                         |
                         +-> reviewer: update public findings
                         +-> code edit: automatic Base Tests -> new public state
                         +-> STOP: private final evaluation
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
| E3-H | Dynamic heuristic | Same expert pool; deterministic heuristic chooses executor/STOP |
| E3-M | Dynamic model router | A decision model chooses executor/STOP from public state |

E3 always begins with the same mandatory Coder bootstrap. After that, only the router
chooses normal termination through `STOP`. The controller may force termination when
a hard model-call, token, Base-Test, or wall-time budget is exhausted.

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

## Evaluation isolation

| Information | Experts/router | Evaluator/private log |
|---|---:|---:|
| Task prompt and entry point | Yes | Yes |
| Current code | Yes | Yes |
| Sanitized Base status/counts | Yes | Yes |
| Base inputs or expected outputs | No | Yes |
| Plus inputs/results | No | Yes |
| Canonical solution | No | Yes |

Plus Tests run only for the initial and final code artifacts. The initial Plus result
is never used to choose an action or select a fallback artifact. If several versions
exist, fallback selection uses public Base performance only.

## Metrics

Quality metrics:

- initial/final Base pass rate;
- initial/final Plus pass rate;
- absolute and normalized Plus improvement;
- repair recovery rate for tasks initially failing Plus Tests.

Efficiency and routing metrics:

- model calls, input/output tokens, Base-Test executions, wall time, and reported cost;
- cost per solved task;
- executor utilization;
- router decisions, invalid-decision count/rate, and termination reasons.

## Output of each run

```text
runs/<run_name>/<run_id>/
  config.json          resolved configuration
  manifest.json        dataset hash and ordered task IDs
  public_trace.jsonl   actions, public Base outcomes, usage, budgets
  private_eval.jsonl   initial/final Base+Plus evaluation
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

The first local-model run downloads the Qwen checkpoint. A CUDA-capable GPU is strongly
recommended. EvalPlus executes generated code in isolated worker processes; run only
code produced within the controlled experiment environment.

## Shared protocol and commands

Create one 50-task manifest before comparing methods:

```powershell
routerlab manifest --config configs\mbpp_e1.yaml --limit 50 --output manifests\mbpp_plus_pilot50_s42.json
```

Run all four methods against exactly that manifest:

```powershell
routerlab run --config configs\mbpp_e1.yaml --manifest manifests\mbpp_plus_pilot50_s42.json --run-id pilot50-s42
routerlab run --config configs\mbpp_e2.yaml --manifest manifests\mbpp_plus_pilot50_s42.json --run-id pilot50-s42
routerlab run --config configs\mbpp_e3_heuristic.yaml --manifest manifests\mbpp_plus_pilot50_s42.json --run-id pilot50-s42
routerlab run --config configs\mbpp_e3_model.yaml --manifest manifests\mbpp_plus_pilot50_s42.json --run-id pilot50-s42
```

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
- total model-call, token, Base-Test, and wall-time budgets;
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

## Upstream components

- EvalPlus: <https://github.com/evalplus/evalplus>
- Qwen2.5-Coder-3B-Instruct: <https://huggingface.co/Qwen/Qwen2.5-Coder-3B-Instruct>
