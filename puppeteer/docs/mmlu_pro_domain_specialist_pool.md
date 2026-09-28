# MMLU-Pro domain-specialist split-backbone pool

Pool file: `personas/role_aware/mmlu_pro_domain_specialist_split_backbone_pool.jsonl`.

| Role | Backbone | Public action |
| --- | --- | --- |
| Quantitative & Formal Reasoner | `qwen-3.5-9b` | `reasoning` |
| Natural & Life Science Specialist | `gemma-3-12b-it` | `reasoning` |
| Computing & Engineering Specialist | `qwen-3.5-9b` | `reasoning` |
| Social, Legal & Business Specialist | `llama-3.1-8b` | `reasoning` |
| Humanities & Behavioral Specialist | `mistral-nemo-12b` | `reasoning` |
| Generalist Independent Solver | `gemma-3-12b-it` | `reasoning` |
| Adversarial Verifier | `qwen-3.5-9b` | `critique` |
| Stop Controller | `llama-3.1-8b` | `terminate` |

The pool has eight agents. Backbones are assigned by the expected task fit,
rather than making every role share one model: Qwen 3.5 9B serves formal,
computing, and adversarial work; Gemma 3 12B IT serves natural science and
mixed-domain independent solving; Llama 3.1 8B serves social/legal/business
reasoning and carries the local Stop Controller action; and Mistral Nemo serves
humanities and behavioral questions. Stop Controller does not query its assigned
backbone when it terminates a path.

For each capability `c`, the prior is:

```text
clip(backbone_base[c] + 0.60 * (role_template[c] - mean_role_template[c]), 0.01, 0.99)
```

The bases are ordinal values from the model-card evidence. They are not raw
benchmark scores or calibrated selection probabilities. MMLU-Pro profile
updates now cover quantitative reasoning, software engineering, and commonsense
in addition to its existing general/domain/planning/verification/integration
scope. Tool use and repair remain outside the MMLU-Pro update scope.

Sources:

- https://huggingface.co/Qwen/Qwen3.5-9B
- https://huggingface.co/google/gemma-3-12b-it
- https://huggingface.co/meta-llama/Llama-3.1-8B-Instruct
- https://huggingface.co/mistralai/Mistral-Nemo-Instruct-2407
