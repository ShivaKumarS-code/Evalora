# evalora

Evalora generates a test suite from your documents, runs it against any RAG pipeline over HTTP, scores retrieval and hallucination, attributes each failure to a cause, and fails CI when quality regresses.

## Quickstart

From a clean machine to a first report in under five minutes:

```bash
# 1. Install (Python 3.11+)
uv tool install git+https://github.com/ShivaKumarS-code/Evalora@main

# 2. Configure: point evalora at your documents and your pipeline endpoint
evalora init
# then edit evalora.yaml: set docs.paths and pipeline.endpoint

# 3. Generate a test set from your documents (needs GEMINI_API_KEY)
export GEMINI_API_KEY=...
evalora generate --config evalora.yaml --out test_set.jsonl --limit 4

# 4. Run it against your pipeline and build the report
evalora run --config evalora.yaml --tests test_set.jsonl --run-name demo
evalora report --run demo --out report.html

# 5. Store the result as the baseline and gate future runs against it
evalora baseline set --run demo --as main
evalora run --config evalora.yaml --tests test_set.jsonl --run-name candidate
evalora gate --run candidate --baseline main     # exit 1 on regression
```

Your pipeline only needs one HTTP endpoint; the exact request and response
contract is documented in
[`docs/ADAPTER_CONTRACT.md`](./docs/ADAPTER_CONTRACT.md).

## Architecture

```text
documents ──► ingest (loaders, chunker) ──► generate (LLM: 4 question types,
                                              dedupe, answerability filter)
                                                 │
                                                 ▼
test_set.jsonl ──► run (httpx, concurrency, retries, response mapping)
                     │                │
                     ▼                ▼
              retrieval metrics   judges (faithfulness, correctness,
              (deterministic)     abstention)  ──► attribution ──► SQLite store
                                                                        │
                                                                        ▼
                                    report (self-contained HTML) ◄── aggregate
                                                                        │
                                                                        ▼
                                    gate (paired bootstrap diff vs baseline) ──► CI
```

All LLM traffic goes through one `LLMClient` wrapper that adds caching, rate
limiting and retries. Retrieval metrics, attribution, aggregation, the
report and the gate never call an LLM.

## What works now

- Pydantic v2 schemas for chunks, test cases, pipeline responses, claim
  verdicts, case results, and failure categories.
- Strict YAML config loading with actionable validation errors, plus
  `evalora init` to write a starter `evalora.yaml`.
- Gemini over HTTP, behind one `LLMClient` wrapper that
  applies caching, rate limiting, retries with exponential backoff and jitter,
  and typed errors.
- SQLite response cache keyed by provider, model, messages, temperature and
  response schema.
- `complete_json` for structured output, with validation-error repair retries
  and a typed failure error.
- Per-call token usage and approximate cost tracking, with aggregation helpers.
- A deterministic `FakeProvider` for tests.
- Markdown, text and PDF loaders that preserve document paths and PDF page
  numbers.
- A boundary-aware chunker with configurable size and overlap, plus chunk IDs
  that stay stable for unchanged content.
- `evalora generate` for single-hop, multi-hop, unanswerable and adversarial
  questions, with exact/near duplicate removal, a separate answerability check,
  seeded sampling, a `--limit` dev flag, and a generation report with drop
  reasons and token usage.
- `evalora run` with bounded concurrency, per-request timeouts, retries on
  transient failures, configurable response mapping, and per-case error capture.
  It judges each case with Gemini by default; `--no-judge` stores retrieval-only
  results without LLM calls.
- Deterministic retrieval metrics: hit rate, MRR and context recall, plus a
  separate count of retrieval returning anything for unanswerable cases.
- Faithfulness judging with claim extraction and per-claim verification,
  correctness judging with a stored three-way label, and abstention detection
  with rule-based matching plus an LLM fallback for ambiguous answers.
- Rule-based failure attribution covering pipeline errors, failed and wrong
  abstention, retrieval failure, hallucination and generation failure.
- Run aggregation with per-type means, failure counts, latency p50/p95, token
  and cost totals, and seeded bootstrap confidence intervals.
- A self-contained HTML report with summary cards, breakdown tables, filterable
  case drill-down, unsupported claims highlighted, and an optional baseline diff.
  It can embed judge-validation agreement numbers when a validation summary is
  available.
- Judge validation tooling: `evalora export-labels` writes a local HTML claim
  labelling page; `evalora validate-judge` reports precision, recall, F1,
  accuracy, Cohen's kappa and a confusion matrix with bootstrap CIs.
- `evalora ragtruth-sample` loads MIT-licensed RAGTruth data and draws a
  reproducible labelled sample with a fixed seed and manifest.
- SQLite storage for runs, results and baselines. Runs resume by skipping case
  IDs already stored under the same run name.
- Baselines: `evalora baseline set` registers a named alias, and
  `baseline export`/`import` move a baseline as a JSON artifact so CI can
  commit or download it.
- `evalora gate` compares a run to the baseline per configured metric with a
  paired-by-`case_id` bootstrap confidence interval, so noise inside the CI
  does not fail the build. Drops only fail when they exceed the configured
  threshold and the CI excludes zero. Absolute floors are also supported.
  Exit codes: 0 pass, 1 regression, 2 tool errors.
- Gate markdown summaries for pull-request comments: metrics with deltas and
  arrows, new failures by category, and the top regressed cases.
- Cost and time controls on `evalora run`: `--max-cases` caps the run,
  `--dry-run` prints a call and cost estimate without running, and
  `--max-budget-usd` aborts before or during a run when projected spend
  exceeds the budget.
- A composite GitHub Action in [`action/`](./action/) that installs evalora,
  reuses or generates a test set, runs the eval, imports a baseline, gates,
  uploads the report artifact and posts the PR summary.
- The adapter contract is documented in
  [`docs/ADAPTER_CONTRACT.md`](./docs/ADAPTER_CONTRACT.md).
- GitHub Actions CI for lint, format, strict type checks and tests.

## Requirements

- Python 3.11 or newer
- [uv](https://docs.astral.sh/uv/)

## Development quickstart

```bash
uv sync
uv run pytest
uv run ruff check .
uv run ruff format --check .
uv run mypy src
```

Try the CLI:

```bash
uv run evalora version
uv run evalora init
uv run evalora generate --config evalora.yaml --out test_set.jsonl --limit 4
uv run evalora run --config evalora.yaml --tests test_set.jsonl --run-name demo
uv run evalora report --run demo --out report.html
uv run evalora baseline set --run demo --as main
uv run evalora gate --run demo --baseline main
uv run evalora export-labels --run demo --out labels.html
uv run evalora validate-judge --labels labels.jsonl --dataset manual
```

`generate` reads the documents under `docs.paths` and writes the JSONL test set
plus `generation_report.json` in the output directory. It makes real LLM calls,
so `GEMINI_API_KEY` must be set and the configured `docs.paths` must contain
documents.

`run` judges each case with Gemini by default, so it also needs
`GEMINI_API_KEY`. It stores results in `.evalora/runs.db`, prints a retrieval
summary, and skips cases already completed under the same run name. Set
`EVALORA_DB_PATH` to move the database, or pass `--no-judge` for a
retrieval-only run that makes no LLM calls.

`report` reads stored results and writes one self-contained HTML file. It
supports `--baseline <run-name>` for a metric diff and pass-to-fail flips. It
never calls an LLM.

`export-labels` builds a self-contained HTML page for hand-labelling the claims
from a judged run. `validate-judge` runs the judge on those labels and reports
agreement; it needs `GEMINI_API_KEY`. The public RAGTruth workflow and the
hand-labelled-set procedure are documented in
[`docs/JUDGE_VALIDATION.md`](./docs/JUDGE_VALIDATION.md).

`baseline set` points a named baseline at a stored run; `baseline export`
writes a portable JSON artifact and `baseline import` recreates the run in a
fresh database and registers the alias. `gate` never calls an LLM: it exits 0
when quality holds, 1 on a regression, and 2 on tool errors. Pass
`--markdown summary.md` to write a pull-request comment body.

`run` cost controls: `--dry-run` prints the call and cost estimate for the
pending cases and exits; `--max-cases N` caps how many pending cases run;
`--max-budget-usd X` aborts when the projected judge spend exceeds X (the
model must have verified pricing in `llm/costs.py`, and the run stops
scheduling new cases once actual spend crosses the budget).

## CI: gating a pull request

Commit a baseline JSON to the repo (or fetch it from a workflow artifact), then
use the composite action from [`action/`](./action/action.yml):

```yaml
name: evalora-gate
on:
  pull_request:

jobs:
  quality:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4

      # Your RAG pipeline must be reachable at the configured endpoint.
      - uses: ShivaKumarS-code/Evalora/action@main
        with:
          api-key: ${{ secrets.GEMINI_API_KEY }}
          config-path: evalora.yaml
          test-set: test_set.jsonl      # reused if committed, generated otherwise
          run-name: ci-${{ github.run_id }}
          baseline: main
          baseline-file: baseline.json  # committed baseline artifact
          max-budget-usd: "1.00"
```

The action installs evalora, imports the baseline, runs the eval, builds
`report.html`, uploads it as an artifact, posts the gate summary as a PR
comment, and fails the job when the gate fails. The baseline JSON comes from
a good run on `main`:

```bash
evalora baseline export --run main-run --out baseline.json
git add baseline.json
```

The gate only fails when a drop exceeds the configured threshold and the
bootstrap confidence interval of the paired difference excludes zero, so
ordinary run-to-run noise does not block pull requests.

## Results

The dogfooding evidence — regression experiments against a deliberately broken
pipeline, judge validation numbers, costs and runtimes, and an honest
limitations list — lives in [`docs/RESULTS.md`](./docs/RESULTS.md).

## API keys

API keys are read from environment variables only and are never accepted from
the YAML config:

```bash
export GEMINI_API_KEY=...
```

The response cache lives at `~/.evalora/llm_cache.db` by default. Set
`EVALORA_CACHE_PATH` to move it, or pass `--no-cache` before a command that uses
an LLM.

## Configuration

`evalora init` writes a documented starter config. The full schema lives in
[`src/evalora/config.py`](./src/evalora/config.py), and intentional deviations
and clarifications are recorded in
[`docs/DECISIONS.md`](./docs/DECISIONS.md).

Config reference:

```yaml
docs:
  paths: ["./docs"]       # folders or files to ingest (md, txt, pdf)
  chunk_size: 800         # characters per chunk
  chunk_overlap: 100       # overlap between neighbouring chunks

generation:
  model: gemini-3.5-flash-lite
  cases_per_type: {single_hop: 30, multi_hop: 15, unanswerable: 15, adversarial: 15}
  seed: 42                # seeded sampling keeps generation reproducible

pipeline:
  endpoint: http://localhost:8000/query   # your RAG endpoint (POST {"question": ...})
  headers: {}             # e.g. auth headers; values are redacted in snapshots
  timeout_s: 30
  concurrency: 4
  match_threshold: 0.8    # text-overlap fallback when the pipeline returns no chunk ids
  response_mapping: {answer: answer, contexts: contexts}

judge:
  model: gemini-3.5-flash-lite
  temperature: 0

gate:
  baseline: main           # default baseline alias for `evalora gate`
  thresholds:
    faithfulness: {max_drop: 0.05, min_absolute: null}
    context_recall: {max_drop: 0.05}
    correctness: {max_drop: 0.05}
  ci_confidence: 0.95
```

`max_drop` is the tolerated mean drop for a metric; the gate fails only when
the drop exceeds it and the bootstrap CI of the paired difference excludes
zero. `min_absolute` is an optional floor: the gate fails when the current
value falls below it.

## Repository layout

```text
src/evalora/
  cli.py          Typer app
  config.py       YAML config loading and validation
  schemas.py      shared Pydantic models
  ingest/
    loaders.py    markdown, text and PDF loading
    chunker.py    boundary-aware chunks with stable IDs
  generate/
    prompts.py    versioned generation and answerability prompts
    generator.py  case generation, sampling and report writing
    filters.py    dedupe and answerability filtering
  run/
    adapter.py    response mapping and contract validation
    runner.py     async runner, test-set loading and error capture
    estimate.py   call and cost estimates for dry runs and budgets
  evaluate/
    prompts.py    versioned judge prompts
    retrieval.py  hit rate, MRR, context recall and aggregation
    faithfulness.py claim extraction and verification
    correctness.py gold-answer grading
    abstention.py rule-based detection with LLM fallback
    attribution.py rule-based failure labels
    aggregate.py  per-type metrics and bootstrap CIs
    judges.py     per-case judge orchestration
  gate/
    compare.py    paired bootstrap comparison and gate decisions
    baseline.py  baseline export/import as JSON artifacts
    markdown.py   pull-request summary rendering
  report/
    builder.py    self-contained HTML report
    templates/report.html.j2
  validate/
    labels.py     label file IO and HTML labelling sheet
    judge_eval.py agreement metrics and validation runs
    ragtruth.py   RAGTruth loader and seeded sampler
  store/
    db.py         runs, results and baselines in SQLite
  llm/
    base.py       protocol, response types, errors, LLMClient wrapper
    gemini.py
    fake.py       deterministic test provider
    factory.py    configured client construction
    cache.py      SQLite response cache
    limiter.py    concurrency and token-bucket limiting
    json_mode.py  structured output with repair retries
    costs.py      token usage and approximate pricing
tests/
action/
  action.yml     composite GitHub Action: run, gate, report artifact, PR comment
docs/
  DECISIONS.md
  ADAPTER_CONTRACT.md
  JUDGE_VALIDATION.md
```

## License

Not chosen yet.
