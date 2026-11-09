# 2. Local models are charged a hosted provider's price

Status: accepted (2026-11-09)

## Context

The free Gemini tier allows about 500 requests a day and 15 a minute; a 20-question
smoke trial uses about 32 (20 answers, about 12 judge calls). Running the generator
locally through Ollama removes most of that limit, but the bench prices every call
through LiteLLM's price list, where Ollama models cost $0 or are not listed at all.
With every local config at $0, a cost-aware search has no cost axis to trade
against, and "lower cost to reach a target accuracy" cannot be measured.

The options were token counts only, a shadow price per local model, or measured
compute (GPU time or electricity).

## Decision

**A shadow price.** Each local model is charged what the same model costs from a
hosted provider, taken from LiteLLM's own price list:

- `../Agentic-AutoRAG/scripts/build_local_model_prices.py` maps each Ollama tag to the
  LiteLLM listings of the same model (same family, size and instruct variant) and
  writes `agentic_autorag/local_model_prices.json`. The price is the listing at the
  median input price, the lower one when the count is even; input decides because RAG
  prompts are mostly retrieved context (about 750 input tokens per output token in the
  smoke trial). Every listing is kept in the file.
- `configure_litellm_runtime()` installs the table: `ollama/<tag>` and
  `ollama_chat/<tag>` are registered with LiteLLM, and `acompletion_with_cost` charges
  them from their token counts. Every LLM call in the optimiser and the bench goes
  through that function.
- A shadow-priced model still gets the live check when a project config is built: the
  price says nothing about whether the model is pulled.
- Configs name local models by full tag (`ollama/qwen2.5:7b`, not `ollama/qwen2.5`).
  A local model not in the table stays at $0.

Token counts stay the primary cost measure; the shadow price turns them into dollars
that are comparable across local and API models. GPU time is reported alongside, not
optimised: it depends on this machine and includes rate-limit waits.

## Consequences

Dollar figures for local runs are what the same work would cost hosted, not what it
cost here (close to nothing in electricity). The absolute numbers depend on the
provider picked, which the median rule fixes; the ratios between configs do not.
Rebuild the table after a LiteLLM upgrade, and add a tag to the script before using a
new local model.
