# evalora

Evalora generates a test suite from your documents, runs it against any RAG pipeline over HTTP, scores retrieval and hallucination, attributes each failure to a cause, and fails CI when quality regresses.

This repository is being built phase by phase from [`SPEC.md`](./SPEC.md).

## Status

Phase 0, Phase 1 and Phase 2 are implemented and tested. Later phases are not built yet.

| Phase | Status |
| --- | --- |
| 0. Scaffolding and LLM layer | Complete |
| 1. Ingestion and test generation | Complete |
| 2. Runner and retrieval metrics | Complete |
| 3. Judges, attribution and report | Not started |
| 4. Judge validation | Not started |
| 5. CI gate, GitHub Action and release | Not started |
| 6. Dogfooding and write-up | Not started |

No evaluation numbers are reported in this README yet. Every metric that
eventually appears must come from a real stored run.

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
- Deterministic retrieval metrics: hit rate, MRR and context recall, plus a
  separate count of retrieval returning anything for unanswerable cases.
- SQLite storage for runs, results and baselines. Runs resume by skipping case
  IDs already stored under the same run name.
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
```

`generate` reads the documents under `docs.paths` and writes the JSONL test set
plus `generation_report.json` in the output directory. It makes real LLM calls,
so `GEMINI_API_KEY` must be set and the configured `docs.paths` must contain
documents.

`run` does not call an LLM. It stores results in `.evalora/runs.db`, prints a
retrieval summary, and skips cases already completed under the same run name.
Set `EVALORA_DB_PATH` to move the database.

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
  evaluate/
    retrieval.py  hit rate, MRR, context recall and aggregation
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
docs/
  DECISIONS.md
  ADAPTER_CONTRACT.md
```

## License

Not chosen yet.
