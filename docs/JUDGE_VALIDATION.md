# Judge validation

This document measures how much the faithfulness judge can be trusted. Every
number below must come from a real run. Items that are still waiting on real
data are marked `TODO: fill from real run`.

## Judge under validation

- Claim extraction prompt: `claim_extraction` v1
- Claim verification prompt: `claim_verification` v1 (default), with
  `claim_verification_strict` v2 available for comparison
- Model: `gemini-3.5-flash-lite`, temperature 0
- Positive class: `unsupported`

## Public benchmark: RAGTruth

RAGTruth is MIT-licensed. The license was read from the repository before using
the data:

- Repository: <https://github.com/ParticleMedia/RAGTruth>
- License: <https://github.com/ParticleMedia/RAGTruth/blob/main/LICENSE>
- License text: MIT License, Copyright (c) 2023 Particle Media

Citation:

```text
Wu, Yuanhao; Zhu, Juno; Xu, Siliang; Shum, Kashun; Niu, Cheng;
Zhong, Randy; Song, Juntong; Zhang, Tong. RAGTruth: A Hallucination Corpus
for Developing Trustworthy Retrieval-Augmented Language Models. 2023.
https://arxiv.org/abs/2401.00396
```

### Selection

The loader uses `dataset/response.jsonl` and `dataset/source_info.jsonl` from
the repository, filters to `split == "test"` and `quality == "good"` with a
matching source record and a non-empty response, then samples with a fixed seed.

- Eligible responses: 2675
- Sample size: 250
- Sample seed: 42
- Split: `test`
- Unsupported (any annotated hallucination span): 97
- Supported: 153
- Task mix: Summary 88, Data2txt 81, QA 81

RAGTruth is word-level and response-level, not claim-level. For this benchmark an
item is the whole response and its source context. The item is labelled
`unsupported` when it has at least one annotated hallucination span; spans marked
`implicit_true` are also treated as unsupported because this judge is allowed to
use only the provided context. The judge predicts `unsupported` when at least one
extracted claim is unsupported. This adaptation is recorded here so the numbers
are not mistaken for claim-level RAGTruth scores.

The raw dataset and the sampled JSONL are kept local under `data/` and are
gitignored. Reproduce the sample with:

```bash
evalora ragtruth-sample \
  --responses data/ragtruth/response.jsonl \
  --sources data/ragtruth/source_info.jsonl \
  --out data/ragtruth_sample.jsonl \
  --manifest data/ragtruth_sample_manifest.json \
  --sample-size 250 \
  --seed 42 \
  --split test
```

### Results

`TODO: fill from real run` after
`evalora validate-judge --labels data/ragtruth_sample.jsonl --dataset RAGTruth`
finishes.

| Metric | Value | Bootstrap CI |
| --- | --- | --- |
| Precision (unsupported) | TODO | TODO |
| Recall (unsupported) | TODO | TODO |
| F1 (unsupported) | TODO | TODO |
| Accuracy | TODO | TODO |
| Cohen's kappa | TODO | TODO |

Confusion matrix, positive class `unsupported`:

| | Predicted unsupported | Predicted supported |
| --- | --- | --- |
| Actually unsupported | TODO | TODO |
| Actually supported | TODO | TODO |

`TODO: fill from real run` for token usage, cost, and wall-clock time.

## Own hand-labelled set

Target: 100 to 150 claim items from a judged run, covering clearly supported,
clearly unsupported, partially supported, and subtly wrong claims (number, date
or entity swaps).

Export a local labelling page from a judged run:

```bash
evalora export-labels --run <run-name> --out labels.html --max-items 150
```

The page shows each claim with its question, answer and contexts. Label each
claim by hand as supported or unsupported, download `labels.jsonl`, then run:

```bash
evalora validate-judge --labels labels.jsonl --dataset manual
```

The labels must be authored by hand. No labels are generated with an LLM.

Status: `TODO: awaiting hand-labelled set and real run`. Sampling method and
results will be added here; no numbers are claimed yet.

## Judge comparison

The comparison plan is at least two verification prompts (`v1` and `v2`) across
at least two Gemini models. The selection criterion is F1 for the unsupported
class, with precision, recall and kappa considered before changing the default.

| Model | Prompt | Precision | Recall | F1 | Accuracy | Kappa |
| --- | --- | --- | --- | --- | --- | --- |
| TODO | v1 | TODO | TODO | TODO | TODO | TODO |
| TODO | v2 | TODO | TODO | TODO | TODO | TODO |

Status: `TODO: fill from real comparison runs`. The default model remains
`gemini-3.5-flash-lite` until the data says otherwise.

## Known weaknesses and error analysis

`TODO: fill from real run`. This section will list the judge's characteristic
failure modes with example items, including any missed numeric/date/entity
swaps, over-flagging of paraphrases, and claim-extraction misses. No weaknesses
will be invented before inspecting the real predictions.

## Operational notes

- The free-tier Gemini key is limited to 15 requests per minute for this model.
  The client paces requests to that budget and retries transient 429/5xx
  responses with exponential backoff.
- The model occasionally returns 503 "high demand". Cached responses make a
  rerun cheaper; a failed run can be restarted without redoing completed calls.
- Raw datasets and generated label sheets are local artifacts, not committed.
