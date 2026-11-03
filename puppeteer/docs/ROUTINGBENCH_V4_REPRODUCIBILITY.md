# RoutingBench V4 Track B: Reproduction and Runtime Certification

This document reproduces the four-domain Track B beta release while preserving
the exact runtime that produced the published live trajectories.  The release
adds certification around the runtime; it does not modify routing policy,
adapter behavior, prompts, tool contracts, state construction, or evaluator
logic.

## Revisions and provenance

Keep three revisions in every reported experiment:

1. **Dataset revision**: the immutable Hugging Face commit containing packets,
   role cards, trajectories, results and evaluator artifacts.
2. **Execution revision**: the Git revision whose runtime files match
   `reproducibility/runtime_core.sha256.json`.
3. **Certification revision**: the Git revision that adds tests, CI, downloader,
   locks and this documentation.  This revision may advance without changing
   the execution revision, provided the runtime checksum remains identical.

The downloader is pinned by default to dataset commit
`cdc67bad1458f6b939f759e3f764f2e93612f424` in
`TrangBui/RoutingBench-V4`.  Record a newer immutable commit explicitly if the
published release is updated.

## What is and is not frozen

The byte-identity boundary includes:

- `routing_benchmark/routers.py`;
- every Python module under `routing_benchmark/v4`;
- the v2/v3 compatibility modules imported by the frozen v4 package loader;
- `scripts/run_shared_environment.py` and
  `scripts/evaluate_shared_environment.py`;
- the database, code-debugging, formal-proof and structured-data as-run
  configuration files;
- the four frozen agent-pool files that are included in router state and
  executor payloads.

The exact path patterns, file count and aggregate SHA-256 are stored in
`reproducibility/runtime_core.sha256.json`.  CI recalculates this value before
testing packets.  Regression gates, sentinel tests, the downloader, dependency
lock, CI workflow and documentation are intentionally outside the runtime
boundary.  They observe the frozen implementation but are not imported by the
live runner.

## Private-artifact isolation

Packets may publicly distribute evaluator artifacts for transparency.  Public
distribution does not make them model-visible during an episode.

The frozen live adapters materialize only entries in `manifest.public_files`.
Router observations come from `adapter.public_observation(...)`.  Executor
requests are constructed from `adapter.executor_context(...)`, the selected
role card and typed tool definitions.  Private artifacts are consumed only by
the trusted evaluator after execution.

The release gate checks all 82 packets and every role context.  The sentinel
test additionally copies one packet from each domain, injects a unique secret
under `private/`, then verifies that the secret is absent from:

- the materialized workspace;
- the router observation;
- every tested executor context;
- the actual message payload passed to the executor client.

Historical canonical trajectories can also be supplied to the certificate
script.  It scans recorded router observations and executor requests without
changing those trajectories.

## Environment setup

The closest as-run environment is Windows x86-64 with Python 3.13.9 and SQLite
3.50.2.  Exact Python package versions are recorded in
`requirements-routingbench-v4-release.lock.txt` and machine-readable metadata
is in `reproducibility/runtime-lock.json`.

From the `puppeteer` directory:

```powershell
py -3.13 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements-routingbench-v4-release.lock.txt
```

Formal-proof packets additionally require the Lean/Mathlib toolchain declared
by each packet's public execution contract.  Code-debugging packets require the
pinned repository mirrors/revisions declared by their runtime profiles.

## Download the dataset

The downloader refuses to overwrite an existing directory.  From `puppeteer`:

```powershell
python scripts/download_routingbench_v4.py `
  --repo-id TrangBui/RoutingBench-V4 `
  --revision cdc67bad1458f6b939f759e3f764f2e93612f424 `
  --output data/RoutingBench/v4_beta_4d_82_20261102
```

It converts the published `track_b_packets` tree and `track_b_agents` table to
the unchanged runner layout (`packs/` and `agent_pools/`).  No packet content is
edited.

## Run release gates

```powershell
python -m unittest tests.test_routingbench_v4_private_isolation -v
python scripts/certify_routingbench_v4_release.py `
  --release-root data/RoutingBench/v4_beta_4d_82_20261102 `
  --runtime-manifest reproducibility/runtime_core.sha256.json `
  --report reproducibility/certification_report.json
```

To audit the canonical local traces as well, repeat `--run-dir` for each run:

```powershell
python scripts/certify_routingbench_v4_release.py `
  --run-dir runs/database_shared_jev_16k_full_20261031 `
  --run-dir runs/code_shared_jev_gemini_v3_20261101 `
  --run-dir runs/formal_shared_jev_prover_v4_20261031 `
  --run-dir runs/formal_minif2f_20_jev_live_20261101 `
  --run-dir runs/structured_data_10_jev_qwen35_9b_live_20261102_b `
  --run-dir runs/structured_data_250_20_jev_qwen35_9b_live_20261102_c `
  --run-dir runs/structured_data_final10_jev_qwen35_9b_live_20261102_b `
  --run-dir runs/structured_data_formula2_jev_qwen35_9b_live_20261102_c
```

## Run live trajectories

The existing as-run commands remain unchanged.  For example:

```powershell
python scripts/run_shared_environment.py --config config/shared_environment/database_jev_16k.yaml
python scripts/run_shared_environment.py --config config/shared_environment/code_jev_16k.yaml
python scripts/run_shared_environment.py --config config/shared_environment/formal_jev_16k.yaml
python scripts/run_shared_environment.py --config config/shared_environment/structured_data_250_20_jev_qwen35_9b_16k.yaml
```

Provider credentials are read from the environment variables named in each
config.  Hosted model responses and service latency are not guaranteed to be
byte-identical across time.  The frozen environment makes the protocol,
visible information, action space, adapters and scoring reproducible; exact
trajectory replay requires the archived raw service responses or equivalent
provider-side determinism.

## License boundary

The harness, adapters, runner, evaluator, CI and project-authored documentation
are released under the Apache License 2.0 in the repository root. The dataset's
`LICENSE_DATA` governs dataset artifacts separately. Bundled third-party source
materials retain their original licenses and provenance records.
