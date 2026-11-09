# Agentic AutoRAG Bench (fork)

A benchmark for RAG pipeline optimisers, forked to add a stage-aware Bayesian search method and compare it with the methods the paper already runs, on the same search space and budget.

## Search

**Search method**:
A strategy that picks which configs to evaluate within a budget (random, MO-TPE, the agentic methods, `stagebo`); a class with `search(evaluator, budget, *, seed)` in `agentic_autorag_bench/methods/`.
_Avoid_: Optimiser (for the method alone), algorithm

**Config**:
One full setting of every lever, a `TrialConfig`.
_Avoid_: Pipeline (for the setting), candidate

**Lever**:
One tunable setting of the pipeline, such as `chunk_token_size`, `top_k` or the generator model.
_Avoid_: Hyperparameter, knob, parameter

**Search space**:
The values each lever may take, set in the project config.

**Trial**:
One evaluation of one config on the exam, producing a `TrialResult`.
_Avoid_: Run (a run is many trials), experiment

**Budget**:
The trials (or wall-clock time) a search method may spend, the same for every method being compared.

**Run**:
Every method and seed in one bench config, from start to held-out scoring.

## Pipeline

**Stage**:
One step of the RAG pipeline whose levers are tuned together: chunking, embedding, retrieval, query expansion, rerank, compression, generator.
_Avoid_: Step, phase

**Generator**:
The LLM that writes the answer from the question and the retrieved chunks; a stage, so its model, reasoning setting and temperature are levers.
_Avoid_: Answer model, reader

**Judge**:
The LLM that marks a generated answer right or wrong against the gold answer.
_Avoid_: Grader, evaluator (the evaluator runs the whole trial)

## Evaluation

**Exam**:
The fixed questions every trial is scored on during search; for the paper datasets, the committed validation exam.
_Avoid_: Test set, benchmark (for the questions)

**Hold-out**:
Questions the search never sees, used once at the end to score each method's chosen config.
_Avoid_: Test split

**Failure mode**:
Why a question was answered wrongly, per question: `retrieval_miss`, `retrieval_partial`, `generation_wrong`, or `retrieval_complete` when retrieval found everything needed.

**Full evaluation**:
A trial that runs every stage, including the generator and the judge.

**Retrieval-only evaluation**:
A trial that stops after retrieval and scores it against the gold documents or spans, with no generator or judge calls.
_Avoid_: Partial trial, cheap trial

## Model

**Surrogate**:
The statistical model `stagebo` fits to trial results to predict how good an untried config would be, with its uncertainty.
_Avoid_: Proxy model, predictor

**Audit**:
A small set of judge verdicts checked by a person, used to measure how often the judge is wrong.
