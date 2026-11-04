# evalora: Build Spec for Claude Code

Working name: `evalora`. A Python CLI and CI tool that generates a test suite from a user's documents, runs it against any RAG pipeline over HTTP, scores retrieval and hallucination, attributes each failure to a cause, and fails the build when quality regresses.

## How to use this spec

Build phase by phase. Do not start a phase until the previous phase's acceptance criteria pass. After each phase, run the full test suite, update the README, and commit. If a requirement is ambiguous, pick the simplest option, note it in `docs/DECISIONS.md`, and move on.

## Ground rules

1. **Never fabricate results.** No hardcoded metrics, no sample numbers in the README unless they were produced by an actual run. Placeholders must be clearly marked `TODO: fill from real run`.
2. **Deterministic where possible.** Retrieval metrics, attribution and the CI gate must not call an LLM.
3. **Every LLM call goes through one provider wrapper** with caching, retries and rate limiting. No direct SDK calls elsewhere.
4. **Everything is typed.** Pydantic v2 models for all data crossing a module boundary. Run `mypy --strict` or `pyright` clean.
5. **Tests ship with the code.** Each module gets unit tests. LLM calls are mocked in tests via a fake provider. At most a few tests hit a real API, and they are marked `@pytest.mark.live` and skipped by default.
6. **Small commits**, one logical change each, conventional commit messages.

## Tech stack

* Python 3.11+, `uv` for env and dependency management
* CLI: Typer
* Schemas and config: Pydantic v2, PyYAML
* HTTP: httpx (async)
* Storage: SQLite via the stdlib `sqlite3` (no ORM)
* LLM provider: Gemini behind a thin interface
* Stats: numpy, scipy, scikit-learn
* PDF: pypdf
* Report: Jinja2 producing one self-contained HTML file
* Tests: pytest, pytest-asyncio, respx (HTTP mocking)
* Lint and format: ruff

## Repo layout

```
evalora/
  pyproject.toml
  README.md
  SPEC.md
  docs/
    DECISIONS.md
    JUDGE_VALIDATION.md
    ADAPTER_CONTRACT.md
  src/evalora/
    __init__.py
    cli.py                  # Typer app, thin, no logic
    config.py               # YAML config loading and validation
    schemas.py              # all Pydantic models
    ingest/
      loaders.py            # md, txt, pdf
      chunker.py            # stable chunk IDs
    llm/
      base.py               # LLMProvider protocol
      gemini.py
      fake.py               # deterministic provider for tests
      cache.py              # SQLite response cache
      limiter.py            # async rate limiter and retry
      json_mode.py          # structured output with schema validation and retry
    generate/
      generator.py
      prompts.py
      filters.py            # dedupe, answerability check
    run/
      runner.py             # async HTTP runner
      adapter.py            # endpoint contract and validation
    evaluate/
      retrieval.py          # hit rate, MRR, context recall
      faithfulness.py       # claim split and verify
      correctness.py
      abstention.py
      attribution.py        # rule based failure labels
      aggregate.py          # scores, breakdowns, bootstrap CIs
    store/
      db.py                 # runs, results, baselines
    report/
      builder.py
      templates/report.html.j2
    gate/
      compare.py            # baseline diff and thresholds
    validate/
      labels.py             # label file IO
      judge_eval.py         # precision, recall, F1, kappa
  examples/
    toy_rag/                # tiny FastAPI RAG app for demos and integration tests
    evalora.yaml
  tests/
  .github/
    workflows/ci.yml
  action/                   # GitHub Action (Phase 5)
```

## Core schemas (put in `schemas.py`)

```python
class Chunk(BaseModel):
    chunk_id: str  # stable: sha1(doc_path + ordinal + text)[:12]
    doc_id: str
    ordinal: int
    text: str


class QuestionType(str, Enum):
    SINGLE_HOP = "single_hop"
    MULTI_HOP = "multi_hop"
    UNANSWERABLE = "unanswerable"
    ADVERSARIAL = "adversarial"


class TestCase(BaseModel):
    case_id: str
    type: QuestionType
    question: str
    gold_answer: str | None  # None for unanswerable
    gold_chunk_ids: list[str]  # empty for unanswerable
    source_model: str  # which model generated it
    answerability_verdict: str  # from the filter step
    metadata: dict = {}


class RetrievedContext(BaseModel):
    id: str | None  # chunk id if the pipeline returns one
    text: str


class PipelineResponse(BaseModel):
    answer: str
    contexts: list[RetrievedContext]
    latency_ms: float
    error: str | None = None


class ClaimVerdict(BaseModel):
    claim: str
    supported: bool
    evidence_context_index: int | None
    rationale: str


class CaseResult(BaseModel):
    case: TestCase
    response: PipelineResponse
    retrieval: RetrievalScores  # hit, rr, context_recall
    faithfulness: float | None  # supported claims / total claims
    claim_verdicts: list[ClaimVerdict]
    correct: bool | None
    abstained: bool
    failure_category: FailureCategory | None


class FailureCategory(str, Enum):
    RETRIEVAL_FAILURE = "retrieval_failure"
    GENERATION_FAILURE = "generation_failure"
    HALLUCINATION = "hallucination"
    FAILED_ABSTENTION = "failed_abstention"
    WRONG_ABSTENTION = "wrong_abstention"
    PIPELINE_ERROR = "pipeline_error"
```

## Adapter contract (the user's pipeline)

The user's RAG system only needs to expose one HTTP endpoint:

```
POST {endpoint}
Request:  {"question": "..."}
Response: {"answer": "...", "contexts": [{"id": "optional", "text": "..."}]}
```

Document this in `docs/ADAPTER_CONTRACT.md` with a minimal FastAPI example. Support configurable headers (for auth) and a configurable JSONPath style mapping for `answer` and `contexts` so non-conforming APIs can be adapted without code changes. If the pipeline does not return chunk IDs, fall back to text overlap matching (normalized text containment or high token overlap) to map retrieved contexts back to gold chunks. Make the matching threshold configurable.

## Config file (`evalora.yaml`)

```yaml
docs:
  paths: ["./docs"]
  chunk_size: 800
  chunk_overlap: 100
generation:
  provider: gemini
  model: gemini-2.0-flash
  cases_per_type: {single_hop: 30, multi_hop: 15, unanswerable: 15, adversarial: 15}
  seed: 42
pipeline:
  endpoint: http://localhost:8000/query
  headers: {}
  timeout_s: 30
  concurrency: 4
  response_mapping: {answer: "answer", contexts: "contexts"}
judge:
  provider: gemini
  model: gemini-2.0-flash
  temperature: 0
gate:
  baseline: main
  thresholds:
    faithfulness: {max_drop: 0.05}
    context_recall: {max_drop: 0.05}
    correctness: {max_drop: 0.05}
  ci_confidence: 0.95
```

Validate with Pydantic. Fail fast with clear messages on bad config. Read API keys from environment variables only, never from the YAML.

## CLI surface

```
evalora init                         # write a starter evalora.yaml
evalora generate --config evalora.yaml --out test_set.jsonl
evalora run --config evalora.yaml --tests test_set.jsonl --run-name <name>
evalora report --run <name> --out report.html
evalora gate --run <name> --baseline <name>   # exit 1 on regression
evalora baseline set --run <name> --as main
evalora validate-judge --labels labels.jsonl  # Phase 4
```

The CLI layer stays thin: parse args, call library functions, print results.

---

## Phase 0: Scaffolding and LLM layer

**Goal:** a working repo and a reliable, cacheable, testable LLM interface. Everything else depends on this.

**Tasks**
1. Initialize the project with `uv`, `pyproject.toml`, ruff, pytest, mypy or pyright, pre-commit.
2. Implement `schemas.py` with the models above.
3. Implement `LLMProvider` protocol: `async def complete(messages, *, temperature, max_tokens, response_schema=None) -> LLMResponse`.
4. Implement the Gemini provider. Implement `FakeProvider` that returns scripted responses keyed by prompt hash or a queue.
5. Implement the SQLite cache. Cache key is a hash of provider, model, messages, temperature and schema. Add a `--no-cache` flag.
6. Implement the limiter: async semaphore for concurrency, token bucket for requests per minute, exponential backoff with jitter on 429 and 5xx, a max retry count, and clear errors on exhaustion.
7. Implement `json_mode.py`: request JSON, parse and validate against a Pydantic model, retry up to N times with the validation error appended to the prompt, raise a typed error if all retries fail.
8. Track token usage and approximate cost per call. Aggregate per run.
9. Set up GitHub Actions CI running lint, type check and tests.

**Acceptance criteria**
* Calling the same prompt twice results in one network call and one cache hit (tested with the fake provider and a call counter).
* A simulated 429 triggers backoff and then succeeds (tested).
* Malformed JSON triggers a repair retry and then succeeds (tested). Persistent failure raises the typed error.
* `pytest` and the type checker pass in CI.

---

## Phase 1: Ingestion and test generation

**Goal:** `evalora generate` turns a folder of documents into a quality-filtered `test_set.jsonl`.

**Tasks**
1. Loaders for `.md`, `.txt` and `.pdf`. Preserve doc path and page where available.
2. Chunker with configurable size and overlap, splitting on paragraph and sentence boundaries before falling back to characters. Chunk IDs must be stable across runs for unchanged content.
3. Generator, per question type:
   * **single_hop:** one chunk in, a question, a gold answer and the supporting quote out.
   * **multi_hop:** pick related chunks (same doc neighbors or embedding-free heuristics such as shared keywords or entities), generate a question that needs facts from two or more.
   * **unanswerable:** give the LLM a chunk and ask for a plausible question on the same topic whose answer is NOT in the chunk or the provided neighbors. Gold answer is null.
   * **adversarial:** generate questions where near-miss details exist (similar names, numbers, versions, dates) so a sloppy retriever grabs the wrong chunk.
4. Put every prompt in `prompts.py` as named constants with version strings. Include the prompt version in each TestCase's metadata.
5. Filters in `filters.py`:
   * exact and near-duplicate removal (normalized text plus a simple similarity threshold)
   * **answerability check:** a separate LLM call, given only the gold chunks, decides whether the question is answerable from them. Answerable types must pass. Unanswerable cases must fail to be answerable even when given the source chunk plus neighbors. Drop disagreements and log how many were dropped and why.
6. Seeded sampling so generation is reproducible. Write `test_set.jsonl` plus a `generation_report.json` (counts requested vs kept per type, drop reasons, token usage).
7. Add a `--limit` dev flag to generate a handful of cases cheaply.

**Acceptance criteria**
* Running generate on a small sample folder produces a valid JSONL where every line parses into `TestCase`.
* Re-running with the same seed and warm cache produces an identical file.
* The generation report shows dropped counts by reason.
* Unit tests cover chunker stability, dedupe, and the answerability filter using the fake provider.

---

## Phase 2: Runner and retrieval metrics

**Goal:** run the test set against a live endpoint and compute all deterministic retrieval metrics.

**Tasks**
1. Async runner using httpx with a concurrency limit, per request timeout, retries on transient failures, and per case error capture (a failure becomes `PipelineResponse.error`, never crashes the run).
2. Response mapping per config. Validate the response shape and produce a clear message when the pipeline response does not match the contract.
3. Context to chunk matching: use returned IDs when present, otherwise text matching as described in the adapter section.
4. Retrieval metrics in `retrieval.py`, all without LLMs:
   * **hit rate@k:** at least one gold chunk retrieved
   * **MRR:** reciprocal rank of the first gold chunk
   * **context recall:** fraction of gold chunks retrieved
   * skip these for unanswerable cases (report separately how often retrieval returned anything)
5. SQLite store (`store/db.py`): tables for runs, results and baselines. Save the config snapshot, test set hash, git SHA if available, and timestamps with each run. Make runs resumable: skip cases already stored for the same run name.
6. Build `examples/toy_rag`: a small FastAPI RAG app (simple embedding or BM25 retrieval plus an LLM answer) with environment toggles to degrade it on purpose (tiny chunk size, retrieval disabled for some docs, weak prompt). It is used for integration tests and the dogfooding story.
7. CLI: `evalora run` with a progress bar and a final summary table.

**Acceptance criteria**
* Against a mocked endpoint (respx), the runner handles success, timeout, 500, and malformed response without crashing, and records each.
* Retrieval metrics are unit tested with hand-computed examples, including edge cases (no gold chunks, duplicate retrievals, ties).
* Interrupting and resuming a run does not redo completed cases.
* `evalora run` against `toy_rag` produces stored results and a printed retrieval summary.

---

## Phase 3: Judges, attribution and report

**Goal:** faithfulness, correctness and abstention scoring, failure attribution, and the HTML report.

**Tasks**
1. **Faithfulness judge** in two steps:
   * *Claim extraction:* an LLM splits the answer into atomic, self-contained factual claims (resolve pronouns). Ignore hedges, greetings and refusals.
   * *Claim verification:* for each claim, an LLM decides whether it is supported by the retrieved contexts only, citing which context supports it. Output a `ClaimVerdict`.
   * Faithfulness = supported claims divided by total claims. If there are zero claims (for example, a pure refusal), the score is null, not 1.0.
   * Keep judge temperature at 0. Version the judge prompts.
2. **Correctness judge:** compare the answer to the gold answer for answerable cases. Output correct, partially correct or incorrect with a short reason. Store the three-way label and derive a boolean from it.
3. **Abstention detection:** a rule-based pass first (configurable phrase patterns such as "I don't know", "not in the provided context"), with an optional LLM fallback for ambiguous cases. Record `abstained`.
4. **Failure attribution** in `attribution.py`, pure logic, in this priority order:
   1. pipeline error recorded: `PIPELINE_ERROR`
   2. unanswerable case and the system answered with substantive claims: `FAILED_ABSTENTION`
   3. answerable case and the system abstained although a gold chunk was retrieved: `WRONG_ABSTENTION`
   4. answerable case, no gold chunk retrieved, and the answer is wrong: `RETRIEVAL_FAILURE`
   5. gold chunk retrieved and the answer contains unsupported claims: `HALLUCINATION`
   6. gold chunk retrieved, claims supported, but the answer is wrong or ignores the gold content: `GENERATION_FAILURE`
   * Document the exact decision table in `docs/DECISIONS.md` and unit test every branch.
5. **Aggregation** (`aggregate.py`): overall and per question type means for each metric, failure category counts, latency p50 and p95, token and cost totals. Compute bootstrap confidence intervals (default 1000 resamples, seeded) for the key scores.
6. **HTML report** (`report/builder.py`): one self-contained file, no external CDN dependencies.
   * Header with run metadata and config snapshot.
   * Summary cards with scores and confidence intervals.
   * Breakdown tables by question type and by failure category.
   * Filterable per-case drill down: question, gold answer, retrieved contexts, answer, claim verdicts with unsupported claims highlighted, and the failure label.
   * If a baseline is supplied, a diff section showing metric changes and cases that flipped from pass to fail.
7. CLI: `evalora report`.

**Acceptance criteria**
* Faithfulness, correctness and abstention modules are unit tested with the fake provider including: all claims supported, mixed, zero claims, and judge JSON failure.
* The attribution table has a test for each branch.
* The report opens in a browser with no network access and renders correctly for a run containing every failure category.
* End to end against `toy_rag` in healthy mode and in degraded mode produces visibly different reports.

---

## Phase 4: Judge validation (the credibility phase)

**Goal:** measure how much the LLM judge can be trusted, and publish the numbers honestly.

**Tasks**
1. **Label tooling:** a small CLI or local HTML page for labelling `(contexts, answer)` pairs as supported or unsupported at the claim level, saving to `labels.jsonl`. Include a way to export a labelling sheet from an existing run.
2. **Own labelled set:** target 100 to 150 labelled items covering clearly supported, clearly unsupported, partially supported and subtly wrong (number, date or entity swapped) cases. Document how the items were sampled in `docs/JUDGE_VALIDATION.md`. The author labels these by hand. Do not synthesize labels with an LLM.
3. **`validate-judge` command:** run the faithfulness judge on the labelled items and report precision, recall, F1, accuracy and Cohen's kappa for the "unsupported" class, plus a confusion matrix. Bootstrap confidence intervals on each.
4. **Public benchmark:** add a loader for a public hallucination dataset (for example RAGTruth, using the license-permitted subset) and report the same metrics on a sampled subset. Record which subset, seed and sample size.
5. **Judge comparison:** run the same evaluation across at least 2 judge prompts and 2 models. Store results in a comparison table and pick the default judge based on the data, not preference.
6. **Error analysis:** list the judge's characteristic failure modes (for example, missing numeric swaps or over-flagging paraphrase) with example cases. Write this honestly in `docs/JUDGE_VALIDATION.md`, including weaknesses.
7. Add a `--judge-validation-summary` section to the report that shows the validated agreement numbers for the judge in use, so readers know how much to trust the scores.

**Acceptance criteria**
* `validate-judge` runs end to end on a labelled file and outputs all metrics with confidence intervals.
* Metrics code is unit tested against hand-computed confusion matrices and against `sklearn` reference values.
* `docs/JUDGE_VALIDATION.md` contains only numbers produced by real runs, with the dataset sizes stated, and a section on known weaknesses.

---

## Phase 5: CI gate, GitHub Action and release

**Goal:** make the tool usable in a pull request workflow and ready to publish.

**Tasks**
1. **Baselines:** `evalora baseline set` stores a named baseline run. Support exporting and importing a baseline as a JSON file so CI can keep it as an artifact or commit it.
2. **Gate** (`gate/compare.py`): compare the current run to the baseline for each configured metric. Use bootstrap confidence intervals on the difference (paired by `case_id` where possible) so that noise does not trigger false failures. Fail only when the drop exceeds the configured threshold and the confidence interval excludes zero. Also support absolute minimum thresholds (for example, faithfulness must stay above a floor). Exit code 1 on failure, 0 on pass, 2 on tool errors.
3. **Markdown summary:** produce a PR comment body with a metrics table, deltas with arrows, the count of new failures by category, and the top 5 regressed cases.
4. **GitHub Action** in `action/`: a composite action that installs the package, runs `generate` (cached) or reuses a committed test set, runs `run` and `gate`, uploads the HTML report as an artifact, and posts the markdown as a PR comment. Provide an example workflow in the README.
5. **Cost and time controls:** a `--max-cases` option, a cost estimate printed before a run (`--dry-run`), and a hard budget flag that aborts if projected spend exceeds a limit.
6. **Packaging:** publish-ready `pyproject.toml`, console entry point, version command, and a clean install test in a fresh venv via `uvx`.
7. **Docs:** README with a quickstart (under 5 minutes), architecture diagram, adapter contract, config reference, and a CI section.

**Acceptance criteria**
* Gate tests: identical runs pass; a controlled drop beyond threshold fails; a drop inside the noise band passes; exit codes are correct.
* The example GitHub workflow runs successfully on a demo repo with the toy pipeline.
* A fresh machine can go from `pip install` to a first report by following only the README.

---

## Phase 6: Dogfooding and write-up

**Goal:** produce the real evidence that goes on a resume.

**Tasks**
1. Point `evalora` at a real RAG pipeline (the author's TechMart FAQ retrieval or the `toy_rag` app).
2. Run a healthy baseline and store it.
3. Introduce deliberate regressions one at a time: smaller chunk size, a weaker retriever, a weaker system prompt, a stale index missing some docs. For each, run the gate and record which failure category dominates and whether the gate caught it.
4. Write `docs/RESULTS.md` containing: dataset and test set sizes, per-type scores, the regression experiment table, judge validation numbers, cost and runtime of a full run, and an honest limitations section.
5. Record a short demo (terminal plus report) showing: break the pipeline, watch the gate fail, read the failure attribution.

**Acceptance criteria**
* Every number in `docs/RESULTS.md` traces to a stored run in the SQLite DB or a committed artifact.
* The limitations section lists at least the known judge weaknesses and the limits of synthetic test sets.

---

## Definition of done (v1)

* All six phases' acceptance criteria pass in CI.
* README quickstart works on a clean environment.
* Judge validation and dogfooding results are documented with real numbers.
* No hardcoded or invented metrics anywhere in the repo.

## Out of scope for v1

Web UI or hosted service, multi-user auth, streaming responses, non-text modalities, support for evaluating agents with multi-turn conversations, fine-tuned judges.

## Notes for Claude Code

* Start Phase 0 by creating the repo skeleton, then implement and test the LLM layer before touching anything else.
* At the end of each phase, print a short summary: what was built, what tests cover it, and any decisions logged.
* Prefer small, well-tested functions over clever abstractions.
* When something requires the author's real data or manual labelling (Phase 4 labels, Phase 6 runs), stop and ask for it rather than generating substitutes.
