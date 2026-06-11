"""
Simplified streaming LLM client for LLM Wiki.
Supports OpenAI and Ollama (matching FW_RAG backends).
Derived from src/lib/llm-client.ts
"""

import json
from typing import Callable

import requests

from llm_wiki.wiki_models import LlmConfig


def stream_chat(
    config: LlmConfig,
    messages: list[dict[str, str]],
    on_token: Callable[[str], None],
    on_done: Callable[[], None],
    on_error: Callable[[Exception], None],
    max_tokens: int = 4096,
    temperature: float = 0.1,
) -> None:
    """
    Stream chat completion.
    Calls on_token for each content chunk, on_done at end, on_error on failure.
    """
    try:
        if config.provider == "openai":
            _stream_openai(config, messages, on_token, on_done, on_error, max_tokens, temperature)
        elif config.provider == "ollama":
            _stream_ollama(config, messages, on_token, on_done, on_error, max_tokens, temperature)
        else:
            on_error(ValueError(f"Unsupported provider: {config.provider}"))
    except Exception as e:
        on_error(e)


def _stream_openai(
    config: LlmConfig,
    messages: list[dict[str, str]],
    on_token: Callable[[str], None],
    on_done: Callable[[], None],
    on_error: Callable[[Exception], None],
    max_tokens: int,
    temperature: float,
) -> None:
    url = config.base_url.rstrip("/") + "/chat/completions" if config.base_url else "https://api.openai.com/v1/chat/completions"
    headers = {"Content-Type": "application/json"}
    if config.api_key:
        headers["Authorization"] = f"Bearer {config.api_key}"
    payload = {
        "model": config.model,
        "messages": messages,
        "stream": True,
        "max_tokens": max_tokens,
        "temperature": temperature,
    }
    response = requests.post(url, headers=headers, json=payload, stream=True, timeout=1800)
    if not response.ok:
        on_error(RuntimeError(f"OpenAI HTTP {response.status_code}: {response.text}"))
        return
    for line in response.iter_lines():
        if not line:
            continue
        decoded = line.decode("utf-8")
        if decoded.startswith("data: "):
            data = decoded[6:]
            if data == "[DONE]":
                break
            try:
                chunk = json.loads(data)
                delta = chunk.get("choices", [{}])[0].get("delta", {})
                token = delta.get("content", "")
                if token:
                    on_token(token)
            except Exception:
                continue
    on_done()


def _stream_ollama(
    config: LlmConfig,
    messages: list[dict[str, str]],
    on_token: Callable[[str], None],
    on_done: Callable[[], None],
    on_error: Callable[[Exception], None],
    max_tokens: int,
    temperature: float,
) -> None:
    url = config.base_url.rstrip("/") + "/api/chat" if config.base_url else "http://localhost:11434/api/chat"
    headers = {"Content-Type": "application/json"}
    payload = {
        "model": config.model,
        "messages": messages,
        "stream": True,
        "options": {
            "num_predict": max_tokens,
            "temperature": temperature,
        },
    }
    response = requests.post(url, headers=headers, json=payload, stream=True, timeout=1800)
    if not response.ok:
        on_error(RuntimeError(f"Ollama HTTP {response.status_code}: {response.text}"))
        return
    for line in response.iter_lines():
        if not line:
            continue
        try:
            chunk = json.loads(line.decode("utf-8"))
            token = chunk.get("message", {}).get("content", "")
            if token:
                on_token(token)
            if chunk.get("done"):
                break
        except Exception:
            continue
    on_done()
