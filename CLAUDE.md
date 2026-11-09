## Agent skills

### Issue tracker

Issues are tracked in GitHub Issues on CalumHarvey/Agentic-AutoRAG-Bench, via the `gh` CLI. This is a fork, so `gh` must target the fork, not the upstream parent: the clone has `gh repo set-default CalumHarvey/Agentic-AutoRAG-Bench`. See `docs/agents/issue-tracker.md`.

### Triage labels

The five default triage labels (needs-triage, needs-info, ready-for-agent, ready-for-human, wontfix), used as-is. See `docs/agents/triage-labels.md`.

### Domain docs

Single-context: one root `GLOSSARY.md` plus `docs/adr/`. See `docs/agents/domain.md`.

## Layout

Two forks cloned side by side: this one (the benchmark, where new search methods go, in `agentic_autorag_bench/methods/`) and `../Agentic-AutoRAG` (the optimiser, changed only in small, upstreamable ways). Both work on the `stagebo` branch, cut from the `emnlp-neurips-camera-ready` tag. See `docs/adr/0001-two-forks.md`.

## Tests

`uv run pytest`. The method tests build configs with `ollama/qwen2.5:7b`; with Ollama running and that model pulled they make a one-token call to it, otherwise `tests/conftest.py` stubs the check. Lint with `uv run ruff check agentic_autorag_bench tests` (`scripts/` carries upstream lint debt).
