# 1. Two forks, pinned to the paper's tag

Status: accepted (2026-11-09)

## Context

The work adds a stage-aware Bayesian search method (`stagebo`) to Agentic AutoRAG and
compares it with the methods already there. The code lives in two upstream repos: the
optimiser ([Agentic-Systems-Lab/Agentic-AutoRAG](https://github.com/Agentic-Systems-Lab/Agentic-AutoRAG))
and the paper's benchmark suite
([lassebaerlandstrand/Agentic-AutoRAG-Bench](https://github.com/lassebaerlandstrand/Agentic-AutoRAG-Bench)),
which installs the optimiser from `../Agentic-AutoRAG`. Search methods plug in on the
bench side (`agentic_autorag_bench/methods/`), next to random, MO-TPE and the agentic
methods.

The options were one fork per upstream repo, or a third repo of our own depending on
both.

## Decision

**Two forks, no third repo**: [CalumHarvey/Agentic-AutoRAG](https://github.com/CalumHarvey/Agentic-AutoRAG)
and [CalumHarvey/Agentic-AutoRAG-Bench](https://github.com/CalumHarvey/Agentic-AutoRAG-Bench),
cloned side by side under their upstream names so the path dependency resolves.

- New methods and benchmark adapters go in the bench fork.
- Optimiser changes (per-question, per-stage results; a retrieval-only evaluation) stay
  small and could go back upstream.
- Both forks work on a `stagebo` branch cut from the `emnlp-neurips-camera-ready` tag,
  the state behind the paper's committed runs, not from `main`, which has moved on.
  Upstream is merged in only on purpose.
- Issues live on the bench fork.

The paper's offline reproduction (README, "Reproduce the paper") was run on this state
on 2026-11-07: Table 2, the Section 5.2 significance numbers and `hypervolume.json`
came out byte-identical to the committed files.

## Consequences

Upstream fixes are a `git merge upstream/<branch>` away in each fork. A change that
spans both repos lands as two commits, one per fork, on branches of the same name; the
bench's CI checks out the optimiser branch with the same name when there is one.
