"""Shared test setup.

Building a ``ProjectConfig`` checks that every LLM in it can be called: names that are
not in LiteLLM's catalog (such as ``ollama/qwen2.5:7b``, used by the method tests) get
a live one-token call. On a machine running Ollama with that model, the call is made
for real. Where Ollama is not reachable (CI), the check is stubbed so the tests still
run without network.
"""

from __future__ import annotations

import json
import socket
import urllib.request

import pytest

# The Ollama model the method tests name (as ``ollama/<model>``).
TEST_OLLAMA_MODEL = "qwen2.5:7b"


def _ollama_reachable(host: str = "127.0.0.1", port: int = 11434, timeout_s: float = 0.5) -> bool:
    try:
        with socket.create_connection((host, port), timeout=timeout_s):
            return True
    except OSError:
        return False


_OLLAMA = _ollama_reachable()


@pytest.fixture(scope="session", autouse=True)
def _load_ollama_model() -> None:
    """Load the test model before the first check calls it.

    The check gives a model 10 s to answer; loading it cold onto the GPU can take
    longer, which failed whichever test happened to call it first.
    """
    if not _OLLAMA:
        return
    body = json.dumps({"model": TEST_OLLAMA_MODEL, "keep_alive": "15m"}).encode()
    request = urllib.request.Request(
        "http://127.0.0.1:11434/api/generate", data=body, headers={"Content-Type": "application/json"}
    )
    try:
        with urllib.request.urlopen(request, timeout=180):
            pass
    except OSError:
        pass  # the checks will report a missing model themselves


@pytest.fixture(autouse=True)
def _stub_llm_probe_without_ollama(monkeypatch: pytest.MonkeyPatch) -> None:
    if _OLLAMA:
        return
    from agentic_autorag.config import models

    monkeypatch.setattr(models, "_probe_model", lambda model, **_: (True, None))
