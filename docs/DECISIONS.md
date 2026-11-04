# Decisions

Decisions are appended as phases are built. Each entry records the choice and
why it was made.

## D001 Project name

The project is named `evalora` (matching the repository root). The package is
`evalora`, the CLI command is `evalora`, and the starter config is
`evalora.yaml`. `SPEC.md` was updated to use the same name.

## D002 Provider uses HTTP directly

Gemini is called through its REST API with `httpx` instead of a vendor SDK.
This keeps the dependency surface small, makes requests easy to mock with
`respx`, and avoids a vendor SDK error model leaking into the rest of the
codebase.

## D003 One LLM client wrapper

`LLMClient` in `llm/base.py` is the only path used to call a model. It composes
cache lookup, the async rate limiter, retry with exponential backoff and jitter,
the provider call, and cache write. No other module calls a provider directly.
Provider classes only translate between `Message`/`LLMResponse` and their HTTP
API.

## D004 Cache key and location

The cache key is a SHA-256 hash of the provider, model, messages, temperature
and response schema, as specified. `max_tokens` is deliberately not part of the
key. Cached payloads are validated before reuse; a corrupt entry is treated as a
miss.

The cache is stored at `~/.evalora/llm_cache.db` by default so it survives
between repositories. `EVALORA_CACHE_PATH` overrides it. The CLI root option
`--no-cache` disables cache reads and writes for the command.

## D005 Cached usage accounting

A cached response keeps the token usage and cost it originally reported and
sets `cached=True`. Phase 2 must count only fresh (`cached=False`) calls when
reporting run spend, so cache hits do not inflate cost totals.

## D006 Structured output retries

`complete_json` makes at most 3 calls by default (2 repair attempts). On a
parse or schema-validation failure it appends the failed assistant response and
a repair instruction containing the validation error, then retries. Exhaustion
raises `StructuredOutputError` with the attempt count and last error.

`response_schema` asks the provider for JSON mode (`responseMimeType` for
Gemini). Schema enforcement happens locally through Pydantic, so provider
variance cannot change the accepted shape.

## D007 Config additions

- `pipeline.match_threshold` (default `0.8`) is included now because the
  adapter contract requires a configurable text-overlap threshold for pipelines
  that do not return chunk IDs.
- `gate.thresholds.<metric>.min_absolute` (default `None`) is included for the
  Phase 5 absolute-floor requirement.

## D008 Fake provider behaviour

`FakeProvider` accepts either a queue of strings/exception instances or a
mapping from a stable prompt hash to a response. When the queue is empty it
returns a configurable default. It records `call_count` and the exact message
lists used for each call. Exceptions in the queue are raised, which lets tests
simulate 429s, 5xx responses and non-retryable errors without patching.

## D009 Pricing

Approximate USD-per-million-token prices live in `llm/costs.py` for the
configured Gemini model. Unknown models cost zero rather than guessing. Prices
are list prices, not invoices; the cost figure is an estimate for budgeting and
should be described that way in reports.

## D010 Type-check scope

`mypy --strict` is configured for `src/`. Tests are run through ruff and pytest.
If type-checking tests becomes cheap enough to maintain, the mypy file list will
be widened.

## D011 CLI scaffold

`evalora init` and `evalora version` exist in Phase 0; the eval commands arrive
with their phases. The root `--no-cache` flag is wired into `AppState` now so
LLM-using commands added in Phase 1 inherit it without changing the CLI shape.

## D012 Gemini is the only provider

The Groq provider and its configuration option were removed. Gemini is the only
supported provider for generation and judging; the `provider` fields accept only
`gemini`, the default judge model is `gemini-2.0-flash`, and only
`GEMINI_API_KEY` is required. The Groq pricing entry was removed with it.
