# Prior profiles for the Gemma 3/Qwen Puppeteer pool

This document records how priors are constructed for
`personas/role_aware/puppeteer_model_card_role_aware_pool.jsonl` after the
backbone replacement experiment.

## Pool changes

The role cards, actions, decoding settings, and provider profiles are retained.
Only the following backbone substitutions are applied while generating the
role-aware pool:

| Original backbone | Replacement | Affected role cards |
| --- | --- | --- |
| `qwen-2.5-7b` | `qwen-3.5-4b` | File Analyst; Summarizer |
| `llama-3.1-8b` | `gemma-3-12b-it` | Web Researcher; Critic / Verifier; Stop Controller |
| `llama-3.2-3b` | `qwen-2.5-7b` | Website Reader; Problem Decomposer |

All replacement backbones use the Hugging Face Router profile via Featherless
AI. Their registered API model names are `Qwen/Qwen3.5-4B:featherless-ai`,
`google/gemma-3-12b-it:featherless-ai`, and
`Qwen/Qwen2.5-7B-Instruct:featherless-ai`.

## Evidence translated into backbone bases

The values are ordinal router priors in `[0.01, 0.99]`. They are neither raw
benchmark scores nor calibrated probabilities. A higher value only states that,
before task-specific evidence, the router should regard one capability as
relatively more plausible for that model.

| Backbone | Published evidence used | Profile emphasis |
| --- | --- | --- |
| Qwen 3.5 4B | The official card reports MMLU-Pro 79.1, HMMT 74.0, LiveCodeBench 55.8, IFEval 89.8, BFCL 50.3, and TAU2-Bench 79.9. | General, quantitative, and domain reasoning; integration; instruction following; moderate tool use. Software engineering remains conservative relative to the larger Qwen 3.5 9B. |
| Gemma 3 12B IT | The official Gemma 3 card reports PT-12B MMLU 74.5, MMLU-Pro CoT 45.3, GSM8K 71.0, MATH 43.3, HumanEval 45.7, MBPP 60.4, and HellaSwag 84.2. | General, quantitative, domain reasoning, commonsense generation, and integration. Tool use is conservative because no directly comparable function-calling metric is used. |
| Qwen2.5 7B Instruct | The official Qwen card describes improved coding, mathematics, structured output, and instruction following; Qwen's published report provides the detailed benchmark table. | Quantitative reasoning, software engineering, structured reasoning, and tool use. |

The Gemma benchmark table reports **pretrained (PT)** family results, while the
pool invokes an instruction-tuned (IT) endpoint. It is used only as family-level
evidence for ordinal initialization. Verification and repair do not have a
comparable public backbone benchmark, so their backbone bases stay neutral at
`0.70`.

## Formula

Let `b[m,c]` be a backbone base for model `m` and capability `c`; let
`r[i,c]` be the old role-template prior for agent `i`; and let `mean_r[c]` be
the mean template prior over the fourteen agents. The generated prior is:

```text
prior[i,c] = round(clip(b[m,c] + 0.60 * (r[i,c] - mean_r[c]), 0.01, 0.99), 6)
```

The backbone base captures published model evidence. The centered role term
keeps the role-specific expected advantage without changing the pool-wide
average of a capability. Consequently, one backbone can have different agent
priors when it is paired with different roles, but its intrinsic base remains
the same.

## Generation and checks

The generator reads the unchanged role template in
`personas/role_aware/mimas_pool.jsonl`, applies the replacement map, and writes
the target pool. It records the original-to-new mapping in every agent's
`metadata.backbone_replacement` field.

```powershell
& $Python scripts/generate_puppeteer_model_card_pool.py
```

Any checkpoint, profile artifact, or audit manifest made with a previous pool
has a different fingerprint. Start a new run and initialize profiles from
`priors`; do not resume it with this pool.

## Sources

- https://huggingface.co/Qwen/Qwen3.5-4B
- https://huggingface.co/google/gemma-3-12b-it
- https://huggingface.co/Qwen/Qwen2.5-7B-Instruct
- https://qwenlm.github.io/blog/qwen2.5-llm/
