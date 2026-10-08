# Local SearXNG for RACO-MAS

This private SearXNG instance supplies web results to both `search_web` and the
legacy `search_bing` action. It binds only to `127.0.0.1`, requires no paid
search API key, and exposes JSON results for the agent tools.

From the `puppeteer` directory:

```powershell
docker compose -f infra/searxng/compose.yaml up -d
Invoke-RestMethod "http://127.0.0.1:8080/search?q=agent+orchestration&format=json"
```

The Python client uses `http://127.0.0.1:8080/search` by default. Override it
for the current process when the port or host differs:

```powershell
$env:SEARXNG_URL = "http://127.0.0.1:8888/search"
```

Stop the service without deleting its cache:

```powershell
docker compose -f infra/searxng/compose.yaml down
```

For any non-local deployment, copy `.env.example` to `.env`, generate a strong
secret, configure a reverse proxy, and review SearXNG's limiter settings. The
included configuration is intentionally local-only.
