# Decision-model planner: Jev and Clef

This policy is the no-training System One alternative to the existing generative
`FrozenLLMPlannerPolicy`. It keeps graph execution, W3D2 capacity management,
agent sessions, aggregation, audit, and route experience unchanged.

Unlike the frozen LLM baseline, the decision-model planner accepts either a
homogeneous or heterogeneous actor pool. Backbone and provider identity remain
hidden from the planner's public role-card view, so heterogeneous runs still test
routing from task context and task-specific role profiles rather than model-name
shortcuts.

## Routing contract

At the root, the default planner receives one finite `route_bundle` choice. For a
pool of `n` available roles and W3, the action space contains every unordered role
subset of size one through three. The task-specific pools therefore expose 63
options for the seven-role MMLU-Pro pool and 41 options for the six-role SRDD pool.

The 13-agent baseline pool uses `decision.root_selection_mode: sequential`
because flat W3 enumeration would require 377 options, above Jev's 255-option
limit. Sequential mode chooses the first root agent from all 13 candidates, then
chooses a complementary unselected agent or `finish_bundle` until it reaches W3.
The three decisions contain at most 13, 13, and 12 options respectively. This
keeps all 13 teammates eligible, prevents duplicate root selections, and lets the
planner choose an effective width of one, two, or three.

On an existing path, the planner receives one `next_action` choice containing all
available roles plus `stop`. The decision probability distribution is written to
the audit trace only. It is not used as an agent-profile feature, an uncertainty
estimate, or a path-count threshold.

Jev and Clef do not generate free-form assignments. `subtask` and
`expected_contribution` are built deterministically from the selected role's
static routing profile.

## OpenRouter credentials

Both Jev and Clef configurations use the same OpenRouter API key:

```powershell
$env:OPENROUTER_API_KEY = "sk-or-v1-..."
```

Optional app-attribution headers can be set without changing the experiment files:

```powershell
$env:OPENROUTER_SITE_URL = "https://your-project.example"
$env:OPENROUTER_APP_NAME = "RACO MAS"
```

Secrets are never placed in experiment YAML files or audit events.
Both models are called through OpenRouter's typed Decisions endpoint,
`https://openrouter.ai/api/alpha/decisions`; they are not sent to the chat
completions endpoint.

## Local web search through SearXNG

The default provider is now a private SearXNG instance, so the Web Researcher
does not require a paid search API key. Start it from the `puppeteer` directory:

```powershell
docker compose -f infra/searxng/compose.yaml up -d
Invoke-RestMethod "http://127.0.0.1:8080/search?q=agent+orchestration&format=json"
```

The provider URL, result limit, timeout, retries, and disk cache are configured
in `config/global.yaml` under `web_search`. If a different local port is used,
override the endpoint for the current PowerShell process:

```powershell
$env:SEARXNG_URL = "http://127.0.0.1:8888/search"
```

New personas should use the provider-neutral `search_web` action. The existing
`search_bing` action is retained as an alias and also calls SearXNG, so baseline
role cards and experiment configs do not need to be rewritten. With
`retrieval_only: true`, a search role returns evidence to the graph without a
second actor-model call; a downstream agent reasons over those results.

The full baseline-pool configuration enables all five tool actions:

```yaml
tools:
  allowed:
    - read_file
    - search_arxiv
    - search_bing
    - search_web
    - access_website
    - run_python
```

## Smoke tests

Run commands from the `puppeteer` directory using the project's
`puppeteer-role-aware` Python environment.

```powershell
python main.py MMLU-Pro test `
  --config config/experiments/decision_mmlu_pro_naive_jev.yaml `
  --policy_mode frozen `
  --data_limit 1 `
  --run_id mmlu_decision_jev_smoke
```

```powershell
python main.py MMLU-Pro test `
  --config config/experiments/decision_mmlu_pro_naive_clef.yaml `
  --policy_mode frozen `
  --data_limit 1 `
  --run_id mmlu_decision_clef_smoke
```

Replace `MMLU-Pro` with `SRDD` and use the matching SRDD configuration for a
software-development smoke test.

## Experiment files

For each dataset, four explicit configurations are provided:

- Naive + Jev
- Evolving Experience + Jev
- Naive + Clef (`cloudflare/clef`)
- Evolving Experience + Clef (`cloudflare/clef`)

The only difference between Naive and Evolving is `experience.mode`. Static role
cards remain immutable in both cases, with probing, profile updates, and profile
uncertainty disabled.

## Provider behavior

- OpenRouter `429`, `5xx`, `524`, and `529` responses are retried with bounded
  exponential backoff. The experiment configs use five retries because the
  Decisions endpoint is currently alpha and its upstream providers can be
  temporarily overloaded.
- Invalid or out-of-contract decision output falls back at bounded cost: one root
  path, or `STOP` on an existing path that already has output.
- Transport, authentication, quota, and provider errors are not hidden by a naive
  fallback. The run fails with the provider error so experimental results cannot
  silently mix routing policies.
- Jev is pinned to `typesafe/jev-1.13` so a paper run does not silently change
  when a rolling alias advances. Clef uses the exact `cloudflare/clef` identifier
  from OpenRouter.
