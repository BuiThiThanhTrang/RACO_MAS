# S0 backbone and capability-prior basis

This document records the model-card-role-blend v2 prior used by
`personas/role_aware/s0_pool.jsonl`. The numbers are engineering priors for
the router; they are neither calibrated probabilities nor measured project
performance.

## Backbones

| Registry key | Hugging Face API model |
| --- | --- |
| `qwen-3.5-9b` | `Qwen/Qwen3.5-9B:featherless-ai` |
| `qwen-3.5-4b` | `Qwen/Qwen3.5-4B:featherless-ai` |
| `gemma-3-12b-it` | `google/gemma-3-12b-it:featherless-ai` |
| `llama-3.1-8b` | `meta-llama/Llama-3.1-8B-Instruct:novita` |

## Construction

Each profile begins with an ordinal backbone base. It is derived from benchmark
families reported in the official model card, rather than by copying raw
accuracy into the profile.

| Backbone | Model-card evidence used |
| --- | --- |
| Qwen3.5-9B | MMLU-Pro 82.5; HMMT, LiveCodeBench, IFEval, BFCL, TAU2 |
| Qwen3.5-4B | MMLU-Pro 79.1; HMMT, LiveCodeBench, IFEval, BFCL, TAU2 |
| Gemma-3-12B-IT | Gemma 3 12B's MMLU, MMLU-Pro CoT, GSM8K, MATH, MBPP, HumanEval, HellaSwag, BoolQ, PIQA |
| Llama-3.1-8B-Instruct | MMLU-Pro CoT, CommonSenseQA, HumanEval, MBPP, GSM8K, MATH, API-Bank, BFCL, IFEval |

The final agent profile is:

```text
agent_prior[c] =
  clip(
    backbone_base[c]
    + 0.60 × (role_template[c] − mean_role_template[c]),
    0.01,
    0.99
  )
```

The role term represents the expected behavior of an agent under that role's
prompt and allowed actions. It does not claim that the underlying backbone
changes its intrinsic ability. Verification and repair have no directly
comparable model-card benchmark, so their backbone bases are neutral; only the
role term distinguishes them.

The Gemma evidence is reported as Gemma 3 12B PT family evaluation on the
official card. It is indirect evidence for the IT endpoint, so Gemma receives
no unsupported general-reasoning advantage.

## Interpretation

These priors describe **effective agent capability conditional on both the
backbone and role**. The router also receives role features separately, so this
pool is suitable for an operational role-aware system. It is not a clean
backbone-only ablation. Use the crossed profile-aware pool when comparing
backbones within the same role.

## Sources

- https://huggingface.co/Qwen/Qwen3.5-9B
- https://huggingface.co/Qwen/Qwen3.5-4B
- https://huggingface.co/google/gemma-3-12b-it
- https://huggingface.co/meta-llama/Llama-3.1-8B-Instruct

All S0 records now use `prior_profile_version:
model-card-role-blend-v2`. Any checkpoint, probe artifact, or audit manifest
built with the previous S0 fingerprint must not be reused.
