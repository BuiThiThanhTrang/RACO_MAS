# Submit router results to RoleRouteBench

RoleRouteBench accepts reproducible router-result submissions for the frozen
`4.0.0-beta.2` environment. The submitter keeps control of the router and
publishes only the result bundle and enough provenance to reproduce the run.

## Comparison contract

1. Use dataset revision `cdc67bad1458f6b939f759e3f764f2e93612f424`.
2. Use code revision `ee21943c3594c97f51f2082d0671dde44b7d2eb2`.
3. Do not change packet artifacts, role cards, adapters, executor backbones,
   decoding settings, evaluator, or per-domain call budgets.
4. The router may observe only router-facing public state and the complete role
   pool plus `STOP`. Do not expose `private/`, Track A labels, or
   `track_b_results` during routing.
5. Preserve every routed action and observable consequence. Do not erase failed
   actions or silently restore state to improve a trajectory.
6. Report trajectory/public success separately from independent end-task
   success. Do not publish a weighted composite as the primary score.

## Submission files

- A JSON result matching [`schemas/router_result.schema.json`](schemas/router_result.schema.json).
- A stable public artifact URL containing the run configuration, raw event
  traces, generated summary, dependency versions, and a short reproduction
  command.
- A disclosure of any infrastructure-affected episodes and reruns.

Start from [`submissions/jev-1.13.reference.json`](submissions/jev-1.13.reference.json),
validate the JSON against the schema, then open the **Add a router result** issue
template. A maintainer checks schema validity and freeze compliance before a
result is added to the community table. Acceptance does not imply endorsement.
