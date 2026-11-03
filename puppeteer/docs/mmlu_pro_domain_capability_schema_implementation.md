# Implementation plan: MMLU-Pro domain capability schema

## Goal

Replace generic agent-operation capabilities with a capability schema that
matches the MMLU-Pro domain-specialist pool. The orchestrator should learn
which specialist fits a question domain, rather than receive misleading signal
from unavailable tools or code-repair capabilities.

## Schema

1. Define ten fixed dimensions: `task_planning`, `formal_quantitative`,
   `natural_science`, `computing_engineering`, `social_legal_business`,
   `humanities_behavioral`, `general_reasoning`, `evidence_verification`,
   `answer_integration`, and `stop_decision`.
2. Keep a fixed length of ten because the policy network consumes the profile
   mean, uncertainty, and observation count as feature vectors.
3. Give Stop Controller its own `stop_decision` dimension. Its terminate action
   is local, so its profile is role-based rather than backbone-based.

## Implementation

1. Replace `CAPABILITY_DIMENSIONS` and map old persona-card fields conservatively
   when legacy personas load.
2. Reject trained profile artifacts and checkpoints from the former schema.
   Their ten positions have different meanings and cannot be resumed safely.
3. Replace role/task evidence scopes with domain-oriented scopes.
4. Update the pool factory so generated legacy pools use the new schema.
5. Create a new MMLU pool whose priors are calculated from a backbone base and
   role template. Low-relevance dimensions remain low; a strong backbone cannot
   inflate them.
6. Add regression tests for the new schema, role scopes, legacy-persona loading,
   and Stop Controller discovery.

## Validation

- Load the new pool with `AgentRegister` and assert eight agents and a
  terminator index.
- Assert every profile has exactly the new ten dimensions.
- Assert all MMLU specialist scopes are subsets of the MMLU-Pro task scope.
- Assert legacy capability names are converted only for persona cards.
- Run role-aware foundation, pipeline, and new schema tests.

## Experiment consequence

Train from a fresh checkpoint and use `profiles.initialization.source: priors`.
Do not load a checkpoint, probe profile, reference profile, or saved profile
artifact created under the former capability schema.
