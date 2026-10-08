# GAIA benchmark integration

The integration uses the gated official `gaia-benchmark/GAIA` snapshot. Dataset
files remain under `data/GAIA`, which is ignored by Git. Do not republish either
the validation or test examples.

## Download

1. Sign in to Hugging Face and accept the access terms at
   `https://huggingface.co/datasets/gaia-benchmark/GAIA`.
2. Create a read-only Hugging Face token.
3. In PowerShell, from the `puppeteer` directory:

```powershell
$env:HF_TOKEN = "<your-read-token>"
python -m scripts.download_gaia --split all
```

The downloader pins the resolved Hub commit in `data/GAIA/.snapshot.json` and
does not print questions or answers. Validation can be scored locally. Test has
private answers and produces a separate `*_submission.jsonl` file for the
official leaderboard.

## Recommended smoke run

Keep SearXNG running, then execute five Level-1 validation tasks:

```powershell
python main.py GAIA validation `
  --config config/experiments/decision_gaia_naive_jev.yaml `
  --policy_mode frozen `
  --level 1 `
  --data_limit 5 `
  --run_id "gaia_l1_jev_naive_smoke"
```

## Experiment ladder

The default GAIA topology is W2D4 for the paired Jev/Sol Level-1 comparison.
Width is only an upper bound: root selection may open one or two paths, and every
path may stop before the configured depth.

| Run | Level | Topology | Experience | Purpose |
|---|---:|---:|---|---|
| Smoke | 1 | W1D3 | none | Validate credentials, file tools, SearXNG, and output format |
| Main L1 | 1 | W2D4 | none / route_outcome | Mostly short tool chains |
| Main L2 | 2 | W2D5 | none / route_outcome | Recommended primary comparison |
| Main L3 | 3 | W2D6 | none / route_outcome | Hard multi-tool tasks |
| Upper-bound | 3 | W3D6 | route_outcome | Expensive ablation, not the default |

For a clean Naive versus Evolving comparison, keep the actor pool, decision
model, W/D, aggregation, data order, and seed fixed. Change only
`experience.mode` from `none` to `route_outcome`. Use validation for both modes;
test outcomes are private and cannot update route experience.

The shipped configs cover Jev and Clef in both modes, plus the paired Sol
baseline. They use:

- eleven homogeneous Qwen 3.5 9B actors;
- either flat-bundle or sequential root selection as declared by the config;
- local SearXNG through `search_web`;
- document/archive reading, media inspection, spreadsheet inspection, web
  search, website access, and Python execution;
- Luna only as a final tie verifier when two paths disagree;
- static role cards, no probing, no profile updates, and no uncertainty.

## Attachment and tool coverage

The local 2023 snapshot contains 109 attachment tasks across validation and
test: 29 XLSX, 18 PNG, 15 PDF, 13 TXT, 7 JPG, 7 MP3, 6 CSV, 2 PPTX, 2 ZIP,
2 DOCX, 2 XML, and one each of MOV, M4A, JSON-LD, JSON, Python, and PDB.

The pool routes these formats as follows:

| Attachment/capability | Primary role and action | Notes |
|---|---|---|
| PDF, DOCX, PPTX, TXT, JSON/XML, ZIP, PDB/source | File Analyst / `read_file` | ZIP extraction is bounded and path-safe |
| PNG/JPG, MP3/M4A/WAV, MOV/MP4 | Media Analyst / `inspect_media` | local faster-whisper ASR; question-specific vision for images/video frames |
| CSV/XLSX | Spreadsheet Analyst / `inspect_spreadsheet` | preserves coordinates, formulas, cached values, fills, and merged ranges |
| calculation or targeted parsing | Python Data Analyst / `run_python` | attachment path is available in `GAIA_ATTACHMENT` |
| public evidence discovery | Web Researcher / `search_web` | repeated search is allowed only after substantive state change; exact duplicate queries are blocked |

The stage guard uses an evidence fingerprint rather than banning a role for the
whole path. A role cannot be called twice for the same state. A successful step
that adds new evidence changes the fingerprint, so a second Web Researcher call
with a new query is valid. Failed or empty steps do not unlock an immediate
repeat of the same role.

To run a different level, pass `--level 2` or `--level 3` and copy the config to
adjust `global_config.graph.max_depth` according to the table above.
