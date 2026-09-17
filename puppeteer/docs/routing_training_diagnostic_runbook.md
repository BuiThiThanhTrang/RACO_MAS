# Routing training diagnostic runbook

Run every command below from `D:\SideProject\ChatDev\puppeteer` in PowerShell.

```powershell
$Python = "..\puppeteer_env\Scripts\python.exe"
```

## 1. Check the REINFORCE update direction offline

This test makes no API calls.

```powershell
& $Python scripts/sanity_check_policy_update.py `
  --output runs/audit_validation/policy_update_sanity.json
```

The command must exit with code 0 and print `"passed": true`.

## 2. Preserve the current initial versus final comparison

This analysis makes no API calls.

```powershell
& $Python scripts/compare_routing_runs.py `
  runs/mmlu_diag_initial_s42/audit `
  runs/mmlu_diag_final_s42/audit `
  --output runs/audit_validation/dev_protocol/initial_vs_final_comparison.json
```

Check these fields in the output JSON:

- `manifest_check.comparable`
- `runs.left.root.normalized_entropy_mean`
- `runs.right.root.normalized_entropy_mean`
- `runs.*.task_sensitivity`
- `paired.selected_step_agreement`
- `paired.discordant_tasks`

## 3. Run the entropy ablations

The three configs use the same first 20 MMLU-Pro training items, seed, profiles,
cost weights, topology, and routing mode. Only `entropy_coef` changes.

Run one experiment at a time:

```powershell
& $Python main.py --config config/diagnostics/mmlu_train_entropy_0.yaml
& $Python main.py --config config/diagnostics/mmlu_train_entropy_0001.yaml
& $Python main.py --config config/diagnostics/mmlu_train_entropy_001.yaml
```

The output directories are:

- `runs/mmlu_train_diag_entropy_0_s42`
- `runs/mmlu_train_diag_entropy_0001_s42`
- `runs/mmlu_train_diag_entropy_001_s42`

Do not reuse a directory after a partial or failed run. Give the replacement
config a new `run_id` so traces from separate attempts cannot mix.

## 4. Analyze each training run

These commands make no API calls.

```powershell
& $Python scripts/analyze_training_audit.py `
  runs/mmlu_train_diag_entropy_0_s42/audit `
  --output runs/mmlu_train_diag_entropy_0_s42/training_health.json

& $Python scripts/analyze_training_audit.py `
  runs/mmlu_train_diag_entropy_0001_s42/audit `
  --output runs/mmlu_train_diag_entropy_0001_s42/training_health.json

& $Python scripts/analyze_training_audit.py `
  runs/mmlu_train_diag_entropy_001_s42/audit `
  --output runs/mmlu_train_diag_entropy_001_s42/training_health.json
```

Reject a run if:

- `updates.optimizer_step_rate` is not 1.0.
- a gradient is missing, zero for every update, NaN, or infinite.
- the run contains incomplete traces.

Compare these fields across runs:

- `routing.normalized_entropy`
- `routing.top1_top2_margin`
- `routing.probability_range`
- `routing.decisions_with_threshold_candidates`
- `rewards.accuracy`
- `rewards.step_penalty`
- `rewards.token_cost_penalty`

### Resume after an interrupted provider call

Resume only from the latest committed checkpoint in the same run. For a run
that failed before its first completed item, use `checkpoint_initial.pt`:

```powershell
& $Python main.py `
  --config config/diagnostics/mmlu_train_entropy_0.yaml `
  --policy_mode train `
  --checkpoint runs/mmlu_train_diag_entropy_0_s42/checkpoints/checkpoint_initial.pt
```

The resume path restores the policy, optimizer, profiles, progress, and random
number generator state. A failed audit attempt remains in the run for diagnosis,
while the retried task receives a new attempt ID. Do not resume a checkpoint
after changing the persona pool, provider/model mapping, chat-context budget, or
training configuration; start a new run ID in that case.

## 5. Evaluate the initial and final checkpoint of an ablation

The following example evaluates `entropy_coef=0` on the same first 20 dev
items. The explicit run IDs prevent overwriting earlier diagnostics.

```powershell
$TrainRun = "mmlu_train_diag_entropy_0_s42"
$EvalPrefix = "mmlu_entropy_0"

& $Python main.py `
  --config runs/audit_validation/dev_protocol/mmlu_audit_legacy_initial.yaml `
  --run_id "${EvalPrefix}_initial_eval_s42" `
  --policy_mode evolved `
  --checkpoint "runs/$TrainRun/checkpoints/checkpoint_initial.pt" `
  --profile_source probe `
  --profile_path profiles/artifacts/mmlu-pro_s0_seed42_probe_profiles.json `
  --data_limit 20

& $Python main.py `
  --config runs/audit_validation/dev_protocol/mmlu_audit_legacy_final.yaml `
  --run_id "${EvalPrefix}_final_eval_s42" `
  --policy_mode evolved `
  --checkpoint "runs/$TrainRun/checkpoints/latest.pt" `
  --profile_source probe `
  --profile_path profiles/artifacts/mmlu-pro_s0_seed42_probe_profiles.json `
  --data_limit 20

& $Python scripts/analyze_routing_audit.py `
  "runs/${EvalPrefix}_initial_eval_s42/audit" `
  --output "runs/${EvalPrefix}_initial_eval_s42/audit_report.json"

& $Python scripts/analyze_routing_audit.py `
  "runs/${EvalPrefix}_final_eval_s42/audit" `
  --output "runs/${EvalPrefix}_final_eval_s42/audit_report.json"

& $Python scripts/compare_routing_runs.py `
  "runs/${EvalPrefix}_initial_eval_s42/audit" `
  "runs/${EvalPrefix}_final_eval_s42/audit" `
  --output "runs/${EvalPrefix}_initial_vs_final.json"
```

Repeat section 5 with these substitutions:

| Entropy | `$TrainRun` | `$EvalPrefix` |
|---|---|---|
| 0 | `mmlu_train_diag_entropy_0_s42` | `mmlu_entropy_0` |
| 0.001 | `mmlu_train_diag_entropy_0001_s42` | `mmlu_entropy_0001` |
| 0.01 | `mmlu_train_diag_entropy_001_s42` | `mmlu_entropy_001` |

## 6. Gate before the 140-item dev run

Advance an ablation only if all of the following hold:

1. The optimizer and gradient checks pass.
2. Final normalized entropy is lower than initial normalized entropy.
3. Final top-1/top-2 margin and task sensitivity are higher than initial.
4. Routing sequences change on more than an isolated task.
5. Accuracy gains occur on tasks whose routing changed, and remain visible over
   repeated routing seeds.

If none of the three entropy settings pass this gate, do not run 140 items.
The next change should add a reward baseline or a batched advantage estimator;
changing the step and token cost weights would not address uniform routing.
