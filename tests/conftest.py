"""Shared test setup.

Building a ``ProjectConfig`` checks that every LLM in it can be called: names that are
not in LiteLLM's catalog (such as ``ollama/qwen2.5:7b``, used by the method tests) get
a live one-token call. On a machine running Ollama with that model, the call is made
for real. Where Ollama is not reachable (CI), the check is stubbed so the tests still
run without network.
"""

from __future__ import annotations

import socket

import pytest


def _ollama_reachable(host: str = "127.0.0.1", port: int = 11434, timeout_s: float = 0.5) -> bool:
    try:
        with socket.create_connection((host, port), timeout=timeout_s):
            return True
    except OSError:
        return False


_OLLAMA = _ollama_reachable()


@pytest.fixture(autouse=True)
def _stub_llm_probe_without_ollama(monkeypatch: pytest.MonkeyPatch) -> None:
    if _OLLAMA:
        return
    from agentic_autorag.config import models

    monkeypatch.setattr(models, "_probe_model", lambda model: (True, None))
