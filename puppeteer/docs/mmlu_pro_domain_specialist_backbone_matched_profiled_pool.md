# MMLU-Pro domain-specialist pool: backbone-matched profiles v2

Pool file: `personas/role_aware/mmlu_pro_domain_specialist_backbone_matched_profiled_pool.jsonl`.

## Design

The policy has a fixed 10-dimensional capability vector. Therefore every role
must retain every dimension. A low value means that the capability is not a
meaningful reason to route an MMLU-Pro item to that role; it does not grant the
role a tool or action. In particular, every reasoning-only role has
`tool_use = 0.05` because its role card grants no tool action.

For each non-stop role and capability `c`, the prior is:

```text
prior(role, backbone, c) = role_template[c], if role_template[c] < 0.45
otherwise prior(role, backbone, c) = clip(0.55 × backbone_base[c] + 0.45 × role_template[c], 0.05, 0.95)
```

`backbone_base` is an ordinal assessment from official model-card benchmark
results, and `role_template` represents relevance of the capability to the
role. The relevance gate prevents a strong backbone from inflating an
irrelevant capability. Neither is an accuracy estimate or a calibrated
probability. Stop
Controller has its own fixed relevance template because its `terminate` action
is local and never calls its assigned backbone.

## Backbone assignment

| Role | Backbone | Action |
| --- | --- | --- |
| Quantitative & Formal Reasoner | `qwen-3.5-9b` | `reasoning` |
| Natural & Life Science Specialist | `qwen-3.5-9b` | `reasoning` |
| Computing & Engineering Specialist | `llama-3.1-8b` | `reasoning` |
| Social, Legal & Business Specialist | `gemma-3-12b-it` | `reasoning` |
| Humanities & Behavioral Specialist | `mistral-nemo-12b` | `reasoning` |
| Generalist Independent Solver | `qwen-3.5-9b` | `reasoning` |
| Adversarial Verifier | `qwen-3.5-9b` | `critique` |
| Stop Controller | `llama-3.1-8b` | `terminate` |

- **Qwen3.5-9B** is assigned to quantitative, natural/life science, generalist,
  and verification work. Its model card reports MMLU-Pro 82.5 and GPQA Diamond
  81.7, so it is the strongest available choice for the science-heavy roles.
- **Llama-3.1-8B** is assigned to computing and engineering. Its model card
  reports HumanEval 72.6, MATH 51.9, and GSM8K 84.5, which support this choice.
- **Gemma-3-12B-it** is assigned to social, legal, and business work. Its model
  card reports SocialIQA 53.4 but notably lower MATH 43.3 and GPQA 25.4 than
  the models used for formal and natural-science roles.
- **Mistral-Nemo-12B** remains the humanities and behavioral specialist. Its
  card reports strong commonsense-oriented results, including HellaSwag 83.5
  and CommonSenseQA 70.4, but no comparable strong math or science result is
  reported there.

## Sources

- https://huggingface.co/Qwen/Qwen3.5-9B
- https://huggingface.co/meta-llama/Llama-3.1-8B-Instruct
- https://huggingface.co/google/gemma-3-12b-it
- https://huggingface.co/mistralai/Mistral-Nemo-Instruct-2407
