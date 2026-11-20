# Qwen2.5-Coder-3B on Modal

This deployment serves the shared executor model used by the JEV, Luna, and Laya full
v5 runs. Laya stays on the local CPU; JEV and Luna are called through OpenRouter. All
three route to the same authenticated Qwen endpoint on one NVIDIA T4. T4 is the
lowest-price configured GPU and its 16 GB VRAM is sufficient for this 3B checkpoint in
FP16 with the experiment's 8K context limit.

The deployment pins:

```text
model: Qwen/Qwen2.5-Coder-3B-Instruct
revision: d3a7bedd480bc27342f4336c71e86858b25df935
torch: 2.6.0
transformers: 4.49.0
attention: sdpa
```

The deployment is deliberately limited to one container and one request at a time.
It scales to zero and retains an idle container for at most 30 seconds, avoiding
horizontal cost multiplication while reducing repeated cold starts between Base
Test executions.

## Install and authenticate

```powershell
conda activate routerlab
cd D:\SideProject\ChatDev\routerlab
python -m pip install -e ".[modal]"
modal setup
```

## Deploy and protect the endpoint

```powershell
modal deploy deploy\modal_qwen25_coder.py
modal workspace proxy-tokens create --name routerlab-qwen
```

The deployed endpoint URL is stored in the `*_modal.yaml` experiment configs,
including E1 single-agent, E2 fixed workflow, JEV, Luna and Laya. Save the token
ID/secret printed by Modal; the
secret is shown only once. Configure the current PowerShell session:

```powershell
$env:MODAL_PROXY_TOKEN="wk-YOUR-ID.ws-YOUR-SECRET"
```

## Smoke test and full v5 benchmark

```powershell
routerlab run --config configs\mbpp_e3_laya_modal.yaml --limit 1 --run-id modal-smoke1 --fast-check --fail-fast
routerlab run --config configs\mbpp_e1_modal.yaml --manifest manifests\mbpp_plus_full.json --run-id full-single-v5-s42
routerlab run --config configs\mbpp_e2_modal.yaml --manifest manifests\mbpp_plus_full.json --run-id full-fixed-v5-s42
routerlab run --config configs\mbpp_e3_jev_modal.yaml --manifest manifests\mbpp_plus_full.json --run-id full-jev-v5-s42
routerlab run --config configs\mbpp_e3_luna_openrouter_modal.yaml --manifest manifests\mbpp_plus_full.json --run-id full-luna-v5-s42
routerlab run --config configs\mbpp_e3_laya_modal.yaml --manifest manifests\mbpp_plus_full.json --run-id full-laya-typed-v5-pinned-s42
```

Do not use `--fast-check` for the reported full experiment. The first request after
scale-to-zero includes a cold start; later requests in a continuous run reuse the
same container.

The endpoint returns OpenAI-compatible token usage but no dollar-cost field. Therefore
RouterLab's `metrics.json` does not include Modal GPU billing in `total_cost_usd`;
calculate that component separately from Modal usage/billing data.
