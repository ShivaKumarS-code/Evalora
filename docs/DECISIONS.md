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

## D013 PDF pages and chunk provenance

`Chunk` gains an optional `page` field (the spec's core schema did not include
one) so PDF page numbers survive chunking and can be written into test-case
metadata. `doc_id` is the document path relative to the current working
directory when possible, otherwise an absolute POSIX path. Ordinals keep
incrementing across all pages of the same document so chunk IDs stay unique.

## D014 Chunking algorithm

Chunks are built from paragraphs first, then sentences inside oversized
paragraphs, then fixed-size character slices for a single overlong sentence.
Segments are packed greedily up to `chunk_size`, carrying the last
`chunk_overlap` characters into the next chunk. A chunk ID is
`sha1(doc_id + ordinal + text)[:12]`, matching the spec.

## D015 Prompt versions

Every generation and answerability prompt is a `PromptTemplate` with a name and
version. Each generated `TestCase` records `prompt` and `prompt_version` in
metadata, and the generation report records the version of every prompt used.
Prompts are sent through `complete_json`, so provider JSON mode plus local
Pydantic validation both apply.

## D016 `--limit` allocation

`--limit` caps the total number of cases requested, not the number per type. It
is allocated round-robin in the fixed order single-hop, multi-hop, unanswerable,
adversarial, so a small limit still produces a mix. The generation report shows
the post-limit requested counts.

## D017 Duplicate detection

Questions are normalized to lowercase alphanumeric tokens. Exact normalized
matches are dropped as `exact_duplicate`; otherwise token-set Jaccard similarity
at or above `0.9` is dropped as `near_duplicate`. The threshold is a module
constant for now because the spec did not require it in config.

## D018 Answerability and generation behaviour

Answerable case types are checked against their gold chunks only. Unanswerable
cases are checked against the source chunk plus up to two neighbors, which is
the context the question generator saw. A mismatch with the expected verdict is
dropped as `answerability_disagreement`; missing context is
`insufficient_context`; LLM failures after retries are `generation_failed` and
do not abort the run. Pair-based types are skipped for documents that produce
no related chunk pair and counted as `insufficient_context`.

Generation runs calls sequentially for deterministic ordering. The report is
written as `generation_report.json` next to `--out`; test-set lines are compact
JSON with sorted keys and LF line endings.

## D019 LLM client construction

`llm/factory.py` builds the Gemini provider, SQLite cache (unless `--no-cache`)
and rate limiter used by generation. Defaults are 4 concurrent calls and 60
requests per minute; these are constants until a config need appears.

## D020 Pipeline response mapping

`run/adapter.py` resolves the configured `answer` and `contexts` paths as dot
paths with an optional leading `$.` and list indices (`result.0.contexts`).
Every resolved context must be an object with a string `text` and an optional
string `id`. Missing paths and wrong types raise `ContractError` naming the
mapping path. The contract is documented in `docs/ADAPTER_CONTRACT.md`.

## D021 Context matching

Returned context ids are authoritative: an id matches only when it is one of
the case's gold chunk ids, and the runner does not fall back to text matching in
that case. Contexts without ids are matched by normalized-text containment or
by gold-token coverage at or above `pipeline.match_threshold` (default `0.8`).

Text matching needs the gold chunk text, so generated test cases now include a
`gold_chunk_texts` mapping in metadata. This keeps the test set self-contained
for CI and avoids re-chunking documents at run time. Test sets without that
metadata cannot text-match contexts that lack ids.

## D022 Retrieval applicability

`RetrievalScores.applicable` is false for unanswerable cases and for answerable
cases with no gold chunks. Those cases are excluded from hit rate, MRR and
context recall means. The retrieval summary still reports how many unanswerable
cases returned any context, and reports `n/a` rates when no applicable cases
exist rather than inventing a zero.

## D023 Runner behaviour

The runner shares one `httpx.AsyncClient`, bounds in-flight requests with a
semaphore from `pipeline.concurrency`, and applies `pipeline.timeout_s` per
request. It retries timeouts, transport errors, HTTP 429 and 5xx up to two times
with exponential backoff and no jitter; other 4xx responses are not retried.
Every failure is stored as a `PipelineResponse` with `error` and the
`pipeline_error` failure category, so one bad case never aborts a run.

## D024 Run storage and resume

SQLite tables store runs, per-case results and named baselines. The database is
`.evalora/runs.db` by default; `EVALORA_DB_PATH` overrides it. Each run records
the config snapshot, test-set hash, git SHA when available, timestamps and case
counts. Header values in the saved config snapshot are replaced with `***` so
auth tokens are not persisted.

Results are saved as each case completes. Resuming a run name loads completed
case ids and skips them. Reusing a run name with a different test-set hash is
rejected with a clear error instead of mixing results from different test sets.

## D025 Toy RAG

`examples/toy_rag` uses a pure-Python BM25 index over the sample docs. Answers
are extractive by default so tests and demos work offline; `TOY_RAG_USE_LLM=1`
switches to Gemini through the same `LLMClient` wrapper. Environment toggles
cover tiny chunks, disabled documents, retrieval disabled, a weak prompt and
the LLM model. `examples/evalora.yaml` points at these docs and the local
endpoint. It contains no evaluation results.

## D026 CLI run

`evalora run` requires `--run-name` and defaults `--tests` to
`test_set.jsonl`. It shows a Rich progress bar, saves each result, prints a
retrieval summary table, and reports the database path. The command makes no
LLM calls.

## D027 Toy RAG is local-only

The `examples/` tree (toy RAG app, sample docs and example config) and the two
toy-specific integration tests are intentionally gitignored. They stay available
for local development but are not part of the published repository. FastAPI and
uvicorn remain in the dev dependency group so the local demo still runs. The
committed test suite covers the runner, adapter, store and retrieval metrics
independently of the toy app.



