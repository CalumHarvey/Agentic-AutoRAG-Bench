# Plan: `stagebo`, a stage-aware Bayesian search method

*Updated 12 November 2026. The working copy of this plan is also kept as a claude.ai doc;
this file is the versioned one.*

## Goal and contribution

This fork adds one thing: a Bayesian proposer for RAG configurations that uses each stage's
measurements and their uncertainty. It goes in as one more search method in
`agentic_autorag_bench/methods/` and is compared head-to-head with the methods already
there.

**Contribution in one sentence:** a RAG optimiser whose model is a stage-decomposed,
judge-noise-aware Bayesian model (function-network structure, IRT question difficulty, a
judge-error layer calibrated by an audit), with cheap retrieval-only evaluations, compared
with random, greedy, MO-TPE and Agentic AutoRAG on public and enterprise benchmarks.

**Why fork Agentic AutoRAG:**

- It is the only RAG optimiser that already measures stages separately: a failure mode per
  question (`retrieval_miss`, `retrieval_partial`, `generation_wrong`, `retrieval_complete`).
  Its own ablation shows the LLM proposer gets little out of that signal, which is the gap
  this work fills.
- This benchmark repo already ships the baselines to beat (random, MO-TPE, qLogNEHVI,
  kb-greedy, the LLM proposer with and without diagnosis), all scored the same way, with
  the paper's runs committed.
- MIT-licensed, uv, LiteLLM, local Sentence Transformers models and LanceDB.

**Out of scope:** entity graphs, representation metrics and new pipeline components, except
as search-space levers a comparison needs.

## Where it stands

**M0 is done.**

- Two forks, both on a `stagebo` branch cut from the `emnlp-neurips-camera-ready` tag
  ([ADR 1](adr/0001-two-forks.md)). Issues live on this fork.
- The paper's offline reproduction gives Table 2, the Section 5.2 significance numbers and
  `hypervolume.json` byte-identical to the committed files (7 November).
- CI; CUDA torch on Windows, CPU torch on Linux; tests on a local qwen2.5:7b.
- Every method (random, MO-TPE, warm MO-TPE, qLogNEHVI, kb-greedy, cost-aware agentic)
  runs end to end on MultiHop-RAG: 1 seed, 3 trials, a 20-question exam, a 10-question
  hold-out (`experiment-0/`).
- **Local models, charged a hosted provider's price** ([ADR 2](adr/0002-shadow-prices-for-local-models.md)):
  generator qwen2.5:7b, judge gemma3:4b, proposer qwen3:8b, through Ollama. The whole
  method slate cost about $0.04 at those prices.

**What the smoke runs found** (`experiment-0/`, 9 November; the run outputs are local and
not committed):

| Finding | Evidence | What it changes |
|---|---|---|
| Noise is larger than the gaps between methods | MO-TPE and warm MO-TPE tried the same config: 0.65 and 0.80 on the exam, 0.4 and 0.6 on the hold-out (temperature 1.0, 10–20 questions) | Temperature 0 where models allow it, larger exams and hold-outs, the paper's 10 seeds; motivates the judge-noise layer (M5) |
| Exam score doesn't carry over | Random search's best trial: 0.70 on the exam, 0.0 on the hold-out | Report held-out scores; tune on more questions |
| Retrieval is the main failure | All gold evidence retrieved for 0–35% of questions; the generator declines 20–50% of the time | Retrieval levers matter most, which favours retrieval-only trials |
| Generation takes most of a trial | Index on a cache hit about 1 s, rebuild 40–170 s; retrieval 0.02–6 s a question (mostly the reranker); generation 10–80 s a question; a 20-question trial 3–9 min | Retrieval-only trials (M1) cut a trial to seconds; the index rebuild then becomes the cost to model |
| The agentic proposer is slow on local models | qwen3:8b took 6, 10 and 15 min per proposal: 31 of the run's 55 min | A surrogate model (seconds per step) is far cheaper per step |
| Local beat free Gemini as generator | Random search hold-out: qwen2.5:7b 0.55 judge accuracy, Gemini flash-lite 0.39 (it declined up to 60%) | Stay local |

**Hardware.** A GTX 1660 SUPER (6 GB, no tensor cores). An RTX 5060 (8 GB, tensor cores)
should make generation and embedding several times faster (an estimate, not measured), but
still can't hold the generator and the proposer at once; a 16 GB card would.

**Next: M1.**

## How Agentic AutoRAG works

A run has three phases (`../Agentic-AutoRAG/docs/how_it_works.md`):

1. **Exam.** An examiner LLM writes questions anchored to verbatim spans, or
   `examiner.custom_exam_path` loads your own.
2. **Loop.** Per trial: build or load the index (chunks and embeddings cached per chunking
   and embedder pair), answer every exam question, diagnose, propose the next config.
3. **Select.** The objective is answer accuracy; in cost-aware mode, the Pareto frontier of
   accuracy against LLM cost per query.

**Search space** (`TrialConfig`, `../Agentic-AutoRAG/agentic_autorag/config/models.py`):

| Stage | Levers | Rebuilds the index? |
|---|---|---|
| Chunking | strategy, `chunk_token_size`, `chunk_token_overlap` | yes |
| Embedding | Sentence Transformers model | yes |
| Retrieval | `index_type`, `top_k`, `hybrid_alpha`, fusion, `long_context_reorder` | no |
| Query expansion | none, hyde, multi_query, query_decompose | no |
| Rerank | cross-encoder or none, `reranker_top_n` | no |
| Compression | none, tree_summarize, refine | no |
| Generator | LLM, reasoning, temperature | no |

**Where `stagebo` plugs in.** A class in `agentic_autorag_bench/methods/` with
`async search(evaluator, budget, *, seed) -> SearchResult`, where
`Evaluator = Callable[[TrialConfig], Awaitable[TrialResult]]` (`types.py`). `random.py` shows
the pattern.

**What the hook lacks.** `TrialResult` carries only aggregates, and the evaluator always runs
the whole pipeline. A stage-aware model needs per-question, per-stage outcomes and a
retrieval-only evaluation.

## Additions

| # | Addition | Where | What it does | Grounded in |
|---|---|---|---|---|
| 1 | Per-question, per-stage results | optimiser `ExamResult` → bench `TrialResult` | Each question's failure mode, gold docs or spans retrieved, and correctness | Agentic AutoRAG's `failure_mode()`; RAGChecker, eRAG |
| 2 | Retrieval-only evaluation | a new evaluator path | Scores a config's retrieval without generating or judging | BO of Function Networks with Partial Evaluations (Buathong et al., ICML 2024) |
| 3 | Stage-decomposed surrogate | `methods/stagebo.py` | One model per stage, chained, so a retrieval trial informs the full prediction | BO of Function Networks (Astudillo & Frazier, NeurIPS 2021) |
| 4 | Cost-aware choice of depth | same | Each step picks a config and a depth (retrieval-only or full) by expected gain per unit cost | Partial-evaluations follow-up (2025) |
| 5 | Judge-noise and question-difficulty layers | same | Judge sensitivity and specificity (Dawid-Skene), calibrated on a human-graded audit; per-question difficulty (IRT) | Bayesian Dawid-Skene (Gao et al., 2024), PPI/ARES, tinyBenchmarks |

**Benchmark adapter.** `benchmarks/enterprise_rag_bench.py`, implementing the optimiser's
`BenchmarkAdapter.prepare(...)`. EnterpriseRAG-Bench questions have gold document ids, so
they fit the custom-exam format as tier B (`supporting_doc_ids`). Use the `rep_10k` subset
from RAG_Optimisation.

**Every addition can be switched off**, so each is an ablation. The surrogate sees only what
the other methods see from a trial, plus the retrieval-only trials it pays for out of the
same budget.

## Comparisons

The claim to test: `stagebo` reaches a target accuracy in fewer trials, or at lower cost,
than the existing methods, on the same search space and budget.

| Method | What it is | Why it is in |
|---|---|---|
| `random` | Uniform random search | Simple search is a strong baseline (IBM HPO study) |
| `kb_greedy` | Greedy from the knowledge base | The greedy baseline |
| `motpe`, `motpe_warm` | Optuna MO-TPE, cold and warm-started | syftr's optimiser |
| `qlognehvi` | BoTorch qLogNEHVI through Ax | A stronger black-box Bayesian baseline |
| `agentic_score`, `agentic_cost` | LLM diagnoser and proposer | The paper's method |
| `agentic_nokb_nodiag` | LLM proposer, score only | How much the diagnosis adds |
| `stagebo` and its ablations | This work | The method under test |

**Benchmarks:** HotpotQA, MuSiQue and MultiHop-RAG (the paper's), plus EnterpriseRAG-Bench
on `rep_10k`.

**Reported:**

- Best accuracy found against trials, and against cost or time (the curve, not only the end
  point), with the paper's @10 and @20 checkpoints.
- Held-out accuracy of each method's chosen config.
- 10 seeds per method; differences as paired intervals across seeds. The smoke runs show
  fewer seeds or questions can't separate methods.
- For the judge layer: agreement with a human-graded audit (Cohen's kappa), and whether
  correcting for it changes which config is picked.
- Before relying on retrieval-only trials: whether retrieval quality predicts answer
  accuracy on each benchmark (eRAG's warning that it may not).

## What carries over from RAG_Optimisation

| Carry over | Why |
|---|---|
| The `rep_10k` subset build and its notes | 10k documents that rank pipelines as the full corpus does |
| The balanced 50-question sample (seed 0) | 5 per question type, for judged comparisons |
| The official-evaluator set-up (LiteLLM proxy, judge-effort hook, integrity check) | The benchmark's own judge, with unjudged questions re-run |
| The paired, stratified bootstrap | Comparing two configs on the same questions |
| Measured facts: retrieval baselines on `rep_10k`, embedding throughput | Priors and budget |

On `rep_10k`, BM25 + cross-encoder rerank was the best retriever (recall@4 0.653).
`hybrid_alpha = 0.0` is pure BM25, so this search space can reach it.

## Milestones

Each milestone ends in a result you can stop at. M3 is the minimum defensible result; M4
and M5 are what no prior work has.

| Milestone | What | Stop point | Status |
|---|---|---|---|
| M0 Fork and reproduce | Forks on the paper tag, tests pass, paper tables regenerated, live trials on every method | A working fork and the measured cost of a trial | **Done** (9 Nov) |
| M1 Stage-level plumbing | Per-question failure modes and gold hits in `TrialResult`; a retrieval-only evaluator | Retrieval-only trials in seconds, no LLM calls | Next |
| M2 EnterpriseRAG-Bench adapter | `rep_10k` as a tier-B exam; check retrieval quality predicts answer accuracy | An enterprise benchmark in the suite, and evidence for or against cheap trials | |
| M3 Stage-decomposed surrogate, full trials only | `methods/stagebo.py` against every baseline, 10 seeds | The first head-to-head result | |
| M4 Partial evaluations | Each step chooses a config and a depth by expected gain per unit cost | Accuracy against cost or time for every method | |
| M5 Judge-noise and difficulty layers | Dawid-Skene judge model on a human-graded audit; IRT difficulty | Whether correcting for judge error changes the pick | |

## Risks

- **Simple search wins.** The IBM study found random and greedy search already efficient.
  Report fewer trials or lower cost to reach a target, not only a higher final score.
- **Retrieval doesn't predict answers.** eRAG found span-based retrieval metrics correlate
  weakly with end-to-end quality. Check in M2; if it fails, use eRAG-style utility labels.
- **Noise swamps the comparison.** Seen in the smoke runs. Use temperature 0 where models
  allow it, larger exams and hold-outs, and 10 seeds; budget the time for it.
- **Time.** Local models remove the quota, but a full trial takes 3–9 minutes on the GTX
  1660 SUPER. Partial evaluations and a faster GPU are the answers.
- **Upstream drift.** Stay on the tag; merge upstream on purpose.
- **Scope creep.** New pipeline components go in only as search-space levers a comparison
  needs.
