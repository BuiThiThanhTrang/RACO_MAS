---
license: other
language:
- en
pretty_name: RoutingBench V3-beta
size_categories:
- n<1K
tags:
- multi-agent
- agent-routing
- orchestration
- decision-making
- live-llm-agents
- benchmark
configs:
- config_name: decision
  default: true
  data_files:
  - split: development
    path: data/decision/development.parquet
  - split: validation
    path: data/decision/validation.parquet
  - split: test
    path: data/decision/test.parquet
- config_name: interactive
  data_files:
  - split: development
    path: data/interactive/development.parquet
  - split: validation
    path: data/interactive/validation.parquet
  - split: test
    path: data/interactive/test.parquet
- config_name: actor_inputs
  data_files:
  - split: development
    path: data/actor_inputs/development.parquet
  - split: validation
    path: data/actor_inputs/validation.parquet
  - split: test
    path: data/actor_inputs/test.parquet
- config_name: evaluation
  data_files:
  - split: development
    path: data/evaluation/development.parquet
  - split: validation
    path: data/evaluation/validation.parquet
  - split: test
    path: data/evaluation/test.parquet
- config_name: provenance
  data_files:
  - split: development
    path: data/provenance/development.parquet
  - split: validation
    path: data/provenance/validation.parquet
  - split: test
    path: data/provenance/test.parquet
- config_name: human_review
  data_files:
  - split: development
    path: data/human_review/development.parquet
  - split: validation
    path: data/human_review/validation.parquet
  - split: test
    path: data/human_review/test.parquet
- config_name: agent_catalog
  data_files:
  - split: train
    path: data/agent_catalog/train.parquet
---

# RoutingBench V3-beta

RoutingBench V3-beta is a public research preview for evaluating whether an
orchestrator can dynamically select the next appropriate agent from the task,
current observable state, execution memory, budget, and qualitative role cards.

**Human review and adjudication remain pending for all 116 workflows.** Formal
validation has passed, but this release must not be described as human-validated
or as the final RoutingBench V3 benchmark.

## What is included

- 116 workflows across 10 workflow families.
- 1,156 isolated next-action decisions with set-valued valid actions.
- 52 workflows derived from licensed real tasks: 24 MuSiQue and 28 MMLU-Pro.
- 39 qualitative agent profiles without capability scores or backbone identity.
- Track A for isolated next-agent selection.
- Track B specifications for live actor execution and separate routing,
  execution, completion, semantic-accuracy, and failure-origin metrics.
- Public provenance, source registry, review queue, validation report, schemas,
  checksums, and reproducible baseline configurations.

## Configurations and splits

| Configuration | Development | Validation | Test | Purpose |
|---|---:|---:|---:|---|
| `decision` | 320 | 322 | 514 | Track A routing states and public oracle labels |
| `interactive` | 32 | 32 | 52 | Track B workflow specifications |
| `actor_inputs` | 16 | 16 | 20 | Inputs exposed only after an actor is selected |
| `evaluation` | 16 | 16 | 20 | Gold sidecar; never a router or actor input |
| `provenance` | 32 | 32 | 52 | Source ID, hash, license, and derivation |
| `human_review` | 32 | 32 | 52 | Pending review checklist |
| `agent_catalog` | 39 (`train`) | — | — | Qualitative role contracts |

The test split and labels are public in this beta. It is not a sealed
leaderboard split.

## Loading

```python
from datasets import load_dataset

decisions = load_dataset("TrangBui/RoutingBench-V3", "decision")
workflows = load_dataset("TrangBui/RoutingBench-V3", "interactive")
agents = load_dataset("TrangBui/RoutingBench-V3", "agent_catalog")
```

The default configuration is `decision`.

### Nested state fields

Dynamic nested objects are stored as canonical UTF-8 JSON strings in columns
ending in `_json`. This keeps the Parquet schemas stable across workflow
families and compatible with the Dataset Viewer without changing their
contents. Decode them after loading, for example:

```python
import json

state = json.loads(decisions["development"][0]["observed_state_json"])
transitions = json.loads(workflows["development"][0]["transitions_json"])
```

## Track A: isolated routing

The router receives the task, observable workflow state, recent events,
execution memory, qualitative agent profiles, candidate actions, and remaining
call budget. The primary metric is whether the selected action belongs to the
set-valued `valid_actions` label.

Do not provide `action_rationales`, `valid_actions`, evaluation gold, transition
tables, or provenance-only fields to the router.

## Router visibility contract

Release revision `role-card-r1` keeps the `3.0-beta` schema and task labels
unchanged while aligning the General Problem Solver role card with its actual
evidence-backed scope.

The router can observe the task, public task context, facts, artifact inventory,
open requirements, last event, recent public events, compact execution memory,
remaining budget, qualitative role cards, and candidate actions. It cannot
observe formal transition preconditions, `valid_actions`, action rationales,
hidden simulator state, evaluation gold, supporting IDs, raw actor
contributions, actor candidate answers, or `prior_live_contributions`.

In Track B, full actor contributions are retained in run traces and are passed
to later actors when needed. The router receives only the corresponding public
effect, status, artifact availability, and state-change signal. This beta is
therefore a state-aware routing benchmark rather than a raw-history-selection
benchmark.

## Track B: live trajectory evaluation

Track B starts from the initial state, calls the selected actor, updates public
state from the observed result, and routes again. Actors are real model calls;
their outputs are not simulated. Report these dimensions separately:

- routing quality;
- actor contract success;
- workflow completion;
- semantic end-task accuracy;
- router/executor/mixed failure origin.

`STOP` is a reserved zero-cost terminal decision outside the actor-call budget.
If a final artifact is committed on the last allowed actor call, the router
still receives one explicit terminal decision. Selecting a non-STOP action in
that state produces `TERMINAL_ACTION_ERROR`.

## Reference baselines

GPT-6 Luna Decisions was evaluated through OpenRouter. Track B used
Gemini 2.5 Flash for every actor.

| Track/split | Router profile | Primary routing score | Completion | End-task accuracy |
|---|---|---:|---:|---:|
| A/development | naive | 81.56% valid action | — | — |
| A/development | structured | 80.94% valid action | — | — |
| B/validation real tasks | naive | 78.05% valid transition | 62.50% | 62.50% |
| B/validation real tasks | structured | 66.33% valid transition | 68.75% | 68.75% |

Structured routing substantially improves STOP precision on Track A (94.12%
versus 71.11%) but does not improve overall valid-action accuracy. These results
are intentionally reported without a subjective weighted score. Exact summaries
and resolved configurations are included under `baselines/`.

## Construction and validation

- Formal transition preconditions derive set-valued valid actions.
- No preferred actions, numeric action utilities, capability scores, or actor
  backbone identities are exposed to the router.
- Official answers are isolated in the `evaluation` configuration.
- Split groups do not cross partitions; selected MMLU-Pro domains are disjoint
  across splits.
- Formal validation reports zero observable-state label collisions.
- Oracle development Track A reaches 100%; random reaches 11.875%.
- All 116 review records remain `pending`.

See `validation_report.json`, `quota_report.json`, `source_registry.json`,
`release_manifest.json`, and `checksums.sha256`.

## Known limitations

- This is a small beta intended to validate benchmark meaning and mechanics,
  not a conference-ready final benchmark.
- Human review and adjudication are pending.
- `v3b-musique-3hop-06` is a priority review item because the supplied corpus
  may not express one bridge fact directly enough.
- Live Track B scores depend on both the router and actor model. Gemini may
  occasionally fail structured JSON serialization on LaTeX-heavy tasks.
- The public test labels make this release unsuitable for a sealed leaderboard.

## Licensing and attribution

This is a mixed-license release:

- project-authored workflow specifications, annotations, schemas, and
  documentation: CC BY 4.0;
- MuSiQue-derived records: upstream CC BY 4.0;
- MMLU-Pro-derived records: upstream MIT.

Per-record source identifiers and licenses are provided in the `provenance`
configuration. See `LICENSE_DATA` and `source_registry.json`.

Reference implementation:
[BuiThiThanhTrang/RACO_MAS](https://github.com/BuiThiThanhTrang/RACO_MAS)

## Citation

Until a paper citation is available, cite this versioned dataset release and
the applicable upstream datasets. A machine-readable citation is included in
`CITATION.cff`.
