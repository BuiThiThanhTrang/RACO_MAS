# Common decision-model router

All typed decision models use `workflow.router: decision_model`. The workflow,
state payload, five actions, action descriptions, invalid-output policy and logging
are shared. An adapter may only translate the canonical contract to a provider's
wire format and normalize its typed answer.

The action descriptions are canonical router-facing capability cards. Each card states
the action's inputs, state effect, availability, and step cost. Reviewer and Analyst
cards explicitly allow use before code and after either Base pass or Base fail; this
describes capability without requiring a review subgraph. Raw executor role prompts
are not sent to decision models because output-format instructions are irrelevant to
routing and would create unequal option lengths. See `docs/SYSTEM_ARCHITECTURE.md` for
the exact router and executor information boundaries.

The common routing question explicitly treats both Base pass and Base fail as
incomplete evidence rather than routing rules. A Base pass is not proof of full
correctness. Routers remain
free to choose any executor or `STOP` at every step. Review is optional and evaluated
by expected value; no mandatory review subgraph, repeated-review rule, stopping rule,
or action mask is added.

List registered pairs:

```powershell
routerlab decision-models
```

## Minimal configuration

```yaml
workflow:
  method: e3
  router: decision_model

router_model:
  name: jev
  provider: openrouter
```

Known pairs and defaults:

| Name | Provider | Adapter | Default model or endpoint |
|---|---|---|---|
| `laya` | `local` | Laya SDK | `convaiinnovations/laya-typed-decisions` |
| `jev` | `openrouter` | OpenRouter Decisions API | `typesafe/jev-1.13` |
| `jev` | `typesafe` | System One HTTP | `jev-1.13.0` |
| `jev` | `huggingface` | Transformers decision pipeline | `autotrust/JEV-9B` |
| `jev` | `http` | System One HTTP | URL in `JEV_BASE_URL` |
| `openai_decisions` | `openai` | OpenAI Decisions HTTP | `gpt-6-luna` |
| `openai_decisions` | `openrouter` | OpenRouter Decisions API | `openai/gpt-6-luna-decisions` |
| `drex` | `local` | System One HTTP | `drex-v1.5` at localhost:8000 |
| `drex` | `http`/`modal` | System One HTTP | URL in `DREX_BASE_URL` |

Checkpoint, endpoint, API-key variable, context length and device can be overridden
under `router_model` for a pinned or self-hosted deployment. Such overrides are
written to each run's `config.json`.

Local Laya configs also support an immutable Hugging Face `revision`. Pin this field
for every reported run so a change to the repository's `main` branch cannot silently
download different weights:

```yaml
router_model:
  name: laya
  provider: local
  checkpoint: convaiinnovations/laya-typed-decisions
  revision: 989e28febe5886389496fe13a8036c62b2a20ae0
```

The pinned checkpoint warns that its calibration value for `choice:11+` is outside
the runtime range and is clamped. RouterLab has five actions and therefore uses the
valid `choice:3-5` calibration bucket. This warning does not change the selected
argmax action for the current experiment.

## Credentials

```powershell
$env:OPENROUTER_API_KEY="..."     # jev/openai_decisions @ openrouter
$env:TYPESAFE_API_KEY="..."       # jev@typesafe
$env:OPENAI_API_KEY="..."         # openai_decisions@openai
$env:JEV_BASE_URL="https://.../v1/systemone"
$env:JEV_API_KEY="..."            # jev@http
$env:DREX_BASE_URL="https://.../v1/systemone"
$env:DREX_API_KEY="..."           # drex@http or drex@modal
$env:MODAL_PROXY_TOKEN="..."      # shared Qwen executor endpoint
```

Local DREX does not require a key. Start its official server on port 8000 before
running RouterLab. Local Hugging Face JEV requires the experimental Transformers
decision pipeline:

```powershell
python -m pip install -e ".[decision-local]"
```

## Full MBPP+ v5 configuration

The primary comparison uses:

| Router | Config | Deployment |
|---|---|---|
| JEV | `configs/mbpp_e3_jev_modal.yaml` | `typesafe/jev-1.13` via OpenRouter |
| Luna | `configs/mbpp_e3_luna_openrouter_modal.yaml` | `openai/gpt-6-luna-decisions` via OpenRouter |
| Laya | `configs/mbpp_e3_laya_modal.yaml` | pinned local CPU checkpoint |

All three use the same Modal Qwen executor, full 378-task manifest, v5 capability
cards, seven executor steps, 20 Base-execution safety limit, 900-second episode limit,
and final current-code submission rule.

JEV and Luna aliases are resolved by OpenRouter. Every trace records
`resolved_model`; report the full-run value rather than assuming the alias stayed on
the pilot version. Pilot v5 observed `typesafe/jev-1.13-20260917` and
`openai/gpt-6-luna-decisions-20261006`.

For Laya on this Windows machine:

```powershell
$env:HF_HOME="D:\ModelCache\huggingface"
$env:HF_HUB_CACHE="D:\ModelCache\huggingface\hub"
$env:HF_HUB_OFFLINE="1"
$env:HF_HUB_DISABLE_SYMLINKS_WARNING="1"
```

## Full MBPP+ v5 commands

```powershell
routerlab run --config configs\mbpp_e3_jev_modal.yaml --manifest manifests\mbpp_plus_full.json --run-id full-jev-v5-s42
routerlab run --config configs\mbpp_e3_luna_openrouter_modal.yaml --manifest manifests\mbpp_plus_full.json --run-id full-luna-v5-s42
routerlab run --config configs\mbpp_e3_laya_modal.yaml --manifest manifests\mbpp_plus_full.json --run-id full-laya-typed-v5-pinned-s42
```

Use one fixed model version, provider endpoint and task manifest for every reported
run. Run the three configurations sequentially and do not use `--fast-check` for final
results. `metrics.json` includes provider-reported OpenRouter cost but not Modal GPU
billing because the expert endpoint reports token usage without a dollar cost.
