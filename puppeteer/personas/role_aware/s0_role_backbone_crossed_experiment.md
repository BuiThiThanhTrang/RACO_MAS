# S0 crossed role-backbone experiment

The ordinary S0 pool couples roles and backbones, so a routing preference for a
role cannot be interpreted as a preference for a backbone or its profile.

The profile-aware pool has 44 agents: eleven roles times Qwen3.5-9B,
Qwen3.5-4B, Gemma-3-12B-IT, and Llama-3.1-8B. Role name, prompt, action,
tools, and decoding are identical within a role. Only the capability profile
varies by backbone.

## Prior construction

The values are ordinal priors in the range [0, 1], not raw benchmark
accuracies and not probe measurements. They follow official model-card results:

| Backbone | Direct evidence used |
| --- | --- |
| Qwen3.5-9B | MMLU-Pro 82.5; HMMT, LiveCodeBench, IFEval, BFCL, TAU2 |
| Qwen3.5-4B | MMLU-Pro 79.1; HMMT, LiveCodeBench, IFEval, BFCL, TAU2 |
| Gemma-3-12B-IT | Gemma 3 12B's published MMLU, MMLU-Pro CoT, GSM8K, MATH, MBPP, HumanEval, HellaSwag, BoolQ, and PIQA |
| Llama-3.1-8B-Instruct | MMLU-Pro CoT, CommonSenseQA, HumanEval, MBPP, GSM8K, MATH, API-Bank, BFCL, and IFEval |

The Gemma table is reported for Gemma 3 12B PT in the official card, so it is
used as indirect evidence for the IT checkpoint and is intentionally not given
a large general-reasoning advantage. Verification and repair have no direct
reported benchmark for all four backbones; their model-level bases remain
neutral. Role specialization is then added equally to every backbone in that
role.

Sources:
- https://huggingface.co/Qwen/Qwen3.5-9B
- https://huggingface.co/Qwen/Qwen3.5-4B
- https://huggingface.co/google/gemma-3-12b-it
- https://huggingface.co/meta-llama/Llama-3.1-8B-Instruct

Run the profile-aware configuration only if the objective is to measure whether
routing uses these fixed, model-card-derived profile distinctions.
