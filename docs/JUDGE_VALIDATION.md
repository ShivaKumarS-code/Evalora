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

Measured on 2026-11-18 with `evalora validate-judge` (judge `gemini-3.5-flash-lite`,
prompt `claim_verification` v1, temperature 0).

The planned validation set was the first 60 of the 250-item sample. The free-tier
Gemini daily quota (500 requests per day) was exhausted partway through — the
dogfooding runs of the same day consumed most of it — so the run stopped at
HTTP 429. The 23 items whose extraction and verification calls had already
completed form the measured subset; those items are committed as
`artifacts/phase6/ragtruth_prefix_labels.jsonl` (7 unsupported, 16 supported)
and the metrics below were computed over exactly that file.

| Metric | Value | Bootstrap CI |
| --- | --- | --- |
| Precision (unsupported) | 0.625 | [0.250, 1.000] |
| Recall (unsupported) | 0.714 | [0.333, 1.000] |
| F1 (unsupported) | 0.667 | [0.333, 0.909] |
| Accuracy | 0.783 | [0.609, 0.957] |
| Cohen's kappa | 0.506 | [0.107, 0.862] |

Confusion matrix, positive class `unsupported`:

| | Predicted unsupported | Predicted supported |
| --- | --- | --- |
| Actually unsupported | 5 | 2 |
| Actually supported | 3 | 13 |

Token usage for the measured run: 0 fresh calls — every judgment was served
from the response cache, which is what made a same-day measurement possible
after the quota cut the 60-item run short. The earlier 5-item smoke run
(2026-11-05, also real) had reported precision/recall/F1 of 0.5 with 7 calls.

The remaining 37 items of the 60-item subset are pending a quota reset; a
rerun of the full 60 completes the already-judged 23 from cache and only pays
for the rest:

```bash
evalora validate-judge \
  --labels data/dogfooding/ragtruth_first60.jsonl \
  --out artifacts/phase6/ragtruth_validation.json \
  --dataset RAGTruth
```

With n = 23 the confidence intervals are wide; the point estimates say the
judge is usable but imperfect: it over-flags (3 false positives) and misses
some hallucinations (2 false negatives). Treat the numbers as an initial
measurement, not a final characterization.

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

From the 23-item RAGTruth measurement above (each example is a stored item in
`artifacts/phase6/ragtruth_validation_prefix.json`):

- **Over-flagging summaries (3 false positives).** All three items the judge
  wrongly marked unsupported — `ragtruth:55`, `ragtruth:1480`,
  `ragtruth:2169` — are summary-style answers ("Here is a summary of the
  article in 200 words or less…"). Summaries compress and paraphrase several
  source facts into fewer sentences, and strict claim-by-claim verification
  against the source passages tends to mark that compression as unsupported.
  `ragtruth:55` was also flagged as a false positive in the earlier 5-item
  smoke run, so the pattern is stable across runs.
- **Missing subtle details (2 false negatives).** `ragtruth:2246` adds a
  casualty count ("Three of the militants are killed") that RAGTruth
  annotates as hallucinated, but the judge accepted it as supported;
  `ragtruth:122` is a tabloid-style narrative whose annotated hallucination
  the judge likewise missed. Numeric and narrative-embellishment errors are
  the judge's weak spot.
- **Wide uncertainty at n = 23.** The bootstrap CIs span most of the unit
  interval. The numbers above are an initial measurement, not a final
  characterization; the full 60-item subset and the 100–150-item hand-labelled
  set are still pending (quota reset and manual labelling respectively).

## Operational notes

- The free-tier Gemini key is limited to 15 requests per minute for this model.
  The client paces requests to that budget and retries transient 429/5xx
  responses with exponential backoff.
- The model occasionally returns 503 "high demand". Cached responses make a
  rerun cheaper; a failed run can be restarted without redoing completed calls.
- Raw datasets and generated label sheets are local artifacts, not committed.
