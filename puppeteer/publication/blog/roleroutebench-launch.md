---
title: "RoleRouteBench: Evaluating the Decisions Behind Multi-Agent Success"
thumbnail: null
authors:
- user: TrangBui
---

# RoleRouteBench: Evaluating the Decisions Behind Multi-Agent Success

Multi-agent benchmarks usually end with a simple question: did the system solve
the task? That matters, but it does not isolate the capability of the component
that selected and sequenced the agents.

**RoleRouteBench** is a reproducible environment for evaluating dynamic role
routing on multi-step tasks. A router observes the public state, selects one
role or `STOP`, receives the observable consequence of that action, and decides
again. The packet, role cards, adapters, executor models, call budgets, and
evaluator are frozen so that competing decision models can be compared under
the same environment.

The `4.0.0-beta.2` release is available on
[Hugging Face](https://huggingface.co/datasets/TrangBui/RoleRouteBench) with DOI
[`10.57967/hf/11430`](https://doi.org/10.57967/hf/11430).

## Why final task success is not enough

Final success is jointly produced by the router and the fixed executor. A weak
executor can fail after a sensible route; a strong executor can sometimes
recover from a poor route. Reporting only the terminal answer hides several
important behaviors:

- whether the router gathered evidence before authoring an artifact;
- whether it reacted to a failed execution or repeated the same action;
- whether it verified a fresh candidate before stopping;
- whether it exhausted its budget without selecting `STOP`; and
- whether a successful artifact was produced through a coherent trajectory or
  by chance.

RoleRouteBench therefore reports outcome, coordination, safety, and efficiency
signals separately. It does not collapse them into one weighted score.

## Four live domains

Track B contains **82 canonical live trajectories** across four domains:

| Domain | Packets | What the roles coordinate |
| --- | ---: | --- |
| Database migration and integrity | 12 | schema inspection, profiling, migration authoring, execution, recovery, and validation |
| Code debugging | 20 | reproduction, diagnosis, patch authoring, regression verification, and finalization |
| Formal proof | 30 | proof-state inspection, lemma retrieval, synthesis, compilation, repair, and kernel checks |
| Structured data processing | 20 | workbook inspection, transformation authoring, execution, semantic validation, and repair |

Every domain exposes the complete role pool plus `STOP` at every routing step.
The role cards describe implementation boundaries, not a required workflow.
Routers may skip roles, revisit them, branch on new evidence, or perform an
explicit recovery action.

## Reference JEV results

JEV 1.13 is included as a reference system. These results demonstrate that the
environment generates non-trivial signals; they are not presented as an oracle.

| Domain | Trajectory/public pass | Independent trusted pass | Primary routing decisions |
| --- | ---: | ---: | ---: |
| Database migration and integrity | 6/12 | 7/12 | 84 |
| Code debugging | 12/20 | 12/20 | 143 |
| Formal proof | 17/30 | 18/30 | 185 |
| Structured data processing | 2/20 | 1/20 | 156 |

The structured-data cohort deliberately remains a difficult research preview.
Its traces contain survey, authoring, testing, validation, repair, precondition
rejection, missing `STOP`, and budget exhaustion, even though hard task success
has a strong floor effect.

## Public/private isolation

The repository publishes the evaluator artifacts for transparent reproduction,
but the runtime enforces a protocol boundary:

```text
router-visible                         evaluator-only
┌──────────────────────────────┐       ┌──────────────────────────────┐
│ task + public artifacts      │       │ gold patches / proofs        │
│ role cards + STOP            │       │ answer workbooks / oracles   │
│ observable action outcomes   │       │ trusted result checks        │
│ remaining call budget        │       │ Track A labels               │
└──────────────┬───────────────┘       └──────────────┬───────────────┘
               │ route → execute → observe            │ after termination
               └──────────────────────────────────────►│ evaluate only
```

Sentinel tests certify that router and executor contexts do not include packet
`private/` paths, Track A labels, or Track B evaluator results. Failed and
invalid actions remain in the trace. A transaction-level rollback may occur
inside a domain contract, but orchestrator recovery is a separate routed action.

## Reproduce the frozen release

From the tagged code release, install the locked dependencies and run:

```bash
python scripts/reproduce_roleroutebench.py
```

The wrapper downloads the pinned dataset revision, runs the private-isolation
sentinel tests, and checks the frozen runtime checksum and all packet contexts.

## Submit another router

We invite reproducible results from alternative decision models. A submission
does not require giving us the router weights or API access. Publish the result
bundle and traces, validate the standard result JSON, and open an
[Add a router result](https://github.com/BuiThiThanhTrang/RACO_MAS/issues/new?template=router-result.yml)
issue.

JEV is the current reference system. Comparative router studies, additional
domains, state-level human routing labels, and the redesigned recovery track are
planned for later releases.

## Links

- [Dataset and DOI](https://huggingface.co/datasets/TrangBui/RoleRouteBench)
- [Reference implementation](https://github.com/BuiThiThanhTrang/RACO_MAS)
- [Trajectory explorer](https://huggingface.co/spaces/TrangBui/RoleRouteBench-Explorer)
- [Router submission protocol](https://github.com/BuiThiThanhTrang/RACO_MAS/blob/main/puppeteer/CONTRIBUTING_ROUTER_RESULTS.md)
