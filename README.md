# Dynamic Router Lab

This repository contains an independent experimental framework for evaluating dynamic
executor routing in multi-agent code-generation systems.

The current benchmark is MBPP+ through EvalPlus. The router observes public task state,
current code, sanitized Base Test feedback, action history, and remaining budget. It can
select one executor or `STOP`:

```text
coder | logic_reviewer | edge_case_analyst | repair_agent | STOP
```

Base Tests are executed automatically by the controller whenever code changes. Plus
Tests are private and run only for the initial and final code artifacts.

## Project

The complete implementation, configurations, tests, and documentation are in
[`routerlab/`](routerlab/README.md).

```text
routerlab/
  configs/                 E1, E2, E3 heuristic, and E3 model configurations
  docs/                    implementation plan
  src/routerlab/           experiment runtime
  tests/                   unit tests for contracts, isolation, routing, and metrics
```

## Installation

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -e .\routerlab
```

## Quick verification

```powershell
$env:PYTHONPATH="$PWD\routerlab\src"
python -m unittest discover -s routerlab\tests -v
```

## Pilot

```powershell
cd routerlab
routerlab manifest --config configs\mbpp_e1.yaml --limit 50 --output manifests\mbpp_plus_pilot50_s42.json
routerlab run --config configs\mbpp_e3_heuristic.yaml --manifest manifests\mbpp_plus_pilot50_s42.json --run-id pilot50-s42
```

See [`routerlab/README.md`](routerlab/README.md) for the full protocol and commands for
all experiment configurations.
