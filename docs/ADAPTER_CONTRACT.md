# Adapter contract

Evalora drives a RAG pipeline over one HTTP endpoint.

## Request

```http
POST {endpoint}
Content-Type: application/json

{"question": "What year was TechMart founded?"}
```

Headers configured under `pipeline.headers` are sent with every request, so
authentication tokens and custom headers work without code changes.

## Response

```json
{
  "answer": "TechMart was founded in 1994.",
  "contexts": [
    {"id": "abc123456789", "text": "TechMart was founded in 1994 in Portland."},
    {"text": "The first store opened on Alder Street."}
  ]
}
```

- `answer` is a string.
- `contexts` is a list of objects with a required string `text` and an optional
  string `id`.
- Empty answer strings are allowed; malformed shapes are recorded as pipeline
  errors instead of crashing the run.

## Minimal FastAPI example

```python
from fastapi import FastAPI
from pydantic import BaseModel

app = FastAPI()


class QueryRequest(BaseModel):
    question: str


class Context(BaseModel):
    id: str | None = None
    text: str


class QueryResponse(BaseModel):
    answer: str
    contexts: list[Context]


@app.post("/query", response_model=QueryResponse)
def query(request: QueryRequest) -> QueryResponse:
    chunks = retrieve(request.question)
    return QueryResponse(
        answer=answer(request.question, chunks),
        contexts=[Context(id=chunk.id, text=chunk.text) for chunk in chunks],
    )
```

The full runnable example is kept locally and is not part of the published
repository.

## Response mapping

Non-conforming APIs can be mapped without code changes using dot paths:

```yaml
pipeline:
  response_mapping:
    answer: "data.answer"
    contexts: "data.documents"
```

Paths support an optional leading `$.`, nested object keys, and list indices:

```yaml
response_mapping:
  answer: "$.result.answer"
  contexts: "result.0.contexts"
```

Each resolved context must still contain a string `text` field and may contain a
string `id` field. A missing path or wrong type is reported with the offending
mapping path.

## Matching contexts to gold chunks

When a returned context has an `id`, evalora treats it as authoritative: it
matches only when that id is one of the case's `gold_chunk_ids`.

When `id` is absent, evalora falls back to text matching. Test cases produced by
`evalora generate` include a `gold_chunk_texts` mapping in metadata. A context
matches a gold chunk when the normalized texts contain one another, or when the
fraction of gold tokens found in the context reaches
`pipeline.match_threshold` (default `0.8`). If a test set has no
`gold_chunk_texts` metadata, contexts without ids cannot be matched.

Unanswerable cases do not receive hit rate, MRR or context recall scores.
Instead, runs report how often retrieval returned any context for them.

## Error handling

The runner applies a per-request timeout and retries transient failures
(timeouts, transport errors, HTTP 429 and 5xx) before giving up. Every failure
is stored as a `PipelineResponse` with an `error` string and a
`pipeline_error` failure category; one failing case never aborts the run.
