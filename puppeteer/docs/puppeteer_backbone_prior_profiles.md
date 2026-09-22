# Puppeteer baseline: role cards and model-card priors

`personas/role_aware/puppeteer_pool.jsonl` is the original legacy baseline and
is preserved unchanged.  Its role-aware counterpart is
`personas/role_aware/puppeteer_model_card_role_aware_pool.jsonl`.

The counterpart has the exact same fourteen agents, backbones, actions,
provider profiles, decoding settings, and autonomous policies.  It adds a
schema-v1 role card per agent and replaces the old generic priors in
`mimas_pool.jsonl` with `puppeteer-model-card-role-blend-v1` priors.

## Role cards

| Legacy agent | Role card | Public action |
| --- | --- | --- |
| FileAgent | File Analyst | `read_file` |
| ArxivAgent | Academic Researcher | `search_arxiv` |
| BingAgent | Web Researcher | `search_bing` |
| WebsiteAgent | Website Reader | `access_website` |
| PythonAgent | Python Tool Agent | `run_python` |
| PlannerAgent | Planner / Decomposer | `planning` |
| ReasoningAgent | General Reasoner | `reasoning` |
| CriticAgent | Critic / Verifier | `critique` |
| ReflectAgent | Reflector | `reflect` |
| QuestionAgent | Problem Decomposer | `question` |
| SummarizerAgent | Summarizer | `summarize` |
| ConcluderAgent | Integrator / Concluder | `conclude` |
| ModifierAgent | Modifier / Repair | `modify` |
| TerminatorAgent | Stop Controller | `terminate` |

Each card restricts its public action and describes its input/output as task
context and an action result.  The router never receives model or provider
identity from these cards.

## Model-card evidence and ordinal bases

The profile values are router priors, not benchmark scores or calibrated
probabilities.  Higher values only encode ordinal evidence across the ten
capability dimensions.

| Backbone in the pool | Evidence used | Resulting emphasis |
| --- | --- | --- |
| Qwen2.5-7B-Instruct | The official card reports stronger coding, mathematics, structured output, and instruction following than Qwen2; Qwen's published Qwen2.5 report gives 75.5 on MATH and 84.8 on HumanEval for the 7B instruct model. | Quantitative reasoning and software engineering |
| Qwen2.5-14B-Instruct | The same official Qwen2.5 family evidence; it is the larger 14.7B instruct model, so it receives a moderate ordinal increase over the 7B model rather than copied raw benchmark values. | General, quantitative, domain reasoning, and software engineering |
| Llama-3.1-8B-Instruct | Official card: MMLU 69.4, GSM8K 84.5, MATH 51.9, HumanEval 72.6, MBPP 72.8, BFCL 76.1, IFEval 80.4. | Quantitative reasoning, software engineering, and tool use |
| Llama-3.2-3B-Instruct | Official Meta card: MMLU 63.4, GSM8K 77.7, MATH 48.0 and lower function-calling results than Llama-3.1-8B. | Useful small model; lower general and code base than 8B |
| Ministral 3B 2512 | The official Ministral 3 family card reports 70.7 MMLU for the 3B base model, 60.1 MATH CoT, and 0.548 LiveCodeBench for the 3B reasoning model; the 3B instruct row reports 0.830 MATH Maj@1.  This is family-level evidence because the OpenRouter registry name is abbreviated. | Quantitative reasoning and software engineering, with conservative general ability |
| Mistral Nemo 12B Instruct | Official card: MMLU 68.0, HellaSwag 83.5, CommonSenseQA 70.4; it also documents function calling. | Planning, commonsense generation, integration, and tool use |

Verification and repair have no comparable published backbone benchmark in
these cards.  Their backbone bases are deliberately neutral, allowing only the
role card to distinguish them.

## Construction

The old `mimas_pool.jsonl` priors serve only as a role template.  For capability
dimension `c`, the generated prior is:

```text
clip(backbone_base[c] + 0.60 × (role_template[c] − mean_role_template[c]), 0.01, 0.99)
```

Thus the profile describes the **expected effective capability of an agent
under its role prompt and action**, rather than claiming that the same backbone
has a different intrinsic ability in each role.  This matches the S0
`model-card-role-blend-v2` convention.  Since the router also receives a
separate role feature, this is suitable for an operational profile-aware run;
it is not a clean backbone-only experiment.

## Reproducibility

Regenerate the pool after changing the documented bases:

```powershell
& $Python scripts/generate_puppeteer_model_card_pool.py
```

Do not reuse a checkpoint, profile artifact, or audit manifest created with a
different pool fingerprint.

## Sources

- https://huggingface.co/Qwen/Qwen2.5-7B-Instruct
- https://huggingface.co/Qwen/Qwen2.5-14B-Instruct
- https://qwenlm.github.io/blog/qwen2.5-llm/
- https://huggingface.co/meta-llama/Llama-3.1-8B-Instruct
- https://github.com/meta-llama/llama-models/blob/main/models/llama3_2/MODEL_CARD.md
- https://huggingface.co/mistralai/Ministral-3-3B-Instruct-2512
- https://huggingface.co/mistralai/Mistral-Nemo-Instruct-2407
