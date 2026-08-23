# Role-aware persona pools

These JSONL files use schema version 1.0. Model and provider identity remain in the
internal `TeammateSpec`; routing receives only the public `RoleCard` and current
`CapabilityProfile` view.

`default_pool.jsonl`, `mimas_pool.jsonl`, `test_pool.jsonl`, and
`gsm_local_pool.jsonl` are deterministic migrations of their legacy counterparts.
`gsm_pool.jsonl` is the active role-aware GSM pool.

The legacy Titan pool is intentionally not migrated because every backbone name in
that file is absent from `model/model_config.py`. Add explicit registry entries or
choose a verified replacement model before generating a Titan schema-v1 pool.
