"""LLM provider abstraction.

The library is provider-agnostic: anything that implements a single-method
``generate(prompt, system=None) -> str`` interface works. Thin adapters for OpenAI,
Anthropic, Google Gemini, DeepSeek, and Ollama are included and gated behind optional
dependencies so the core package has zero required deps.

Install the extra for whichever provider you plan to use — see the README for the
installation command. Each provider reads sensible defaults from environment variables
(see ``.env.template``); constructor arguments always win over env vars.
"""

from __future__ import annotations

import contextlib
import hashlib
import json
import os
from pathlib import Path
from typing import Any, Protocol, runtime_checkable


@runtime_checkable
class LLMProvider(Protocol):
    """Any object with a ``generate(prompt, system=None) -> str`` method."""

    def generate(self, prompt: str, system: str | None = None) -> str: ...


class CachedProvider:
    """Decorator that adds an on-disk JSON cache to any ``LLMProvider`` instance.

    Each ``(model, system, prompt)`` triple is hashed to a filename under ``cache_dir``.
    Cache hits return instantly without calling the underlying provider; misses fall
    through, record the response, and return it. Cache files are plain JSON so you can
    inspect them with ``cat``.

    The same decorator works with every provider in this package — just wrap any
    instance before passing it to ``Scheduler``::

        # Decorate different providers with the same cache logic:
        provider = CachedProvider(OpenAIProvider(model="gpt-4o-mini"),  cache_dir=".cache")
        provider = CachedProvider(AnthropicProvider(),                   cache_dir=".cache")
        provider = CachedProvider(GeminiProvider(),                      cache_dir=".cache")
        provider = CachedProvider(DeepSeekProvider(),                    cache_dir=".cache")
        provider = CachedProvider(OllamaProvider(),                      cache_dir=".cache")

    The cache is write-through and thread-unsafe; intended for single-process notebook
    use. Re-running the notebook with the same seeds, model, and prompts will produce
    identical trajectories without any API calls.
    """

    def __init__(self, provider: Any, cache_dir: str | Path = ".cache") -> None:
        self._provider = provider
        self._cache_dir = Path(cache_dir)
        self._cache_dir.mkdir(parents=True, exist_ok=True)
        self.hits = 0
        self.misses = 0

    @property
    def model(self) -> str:
        return getattr(self._provider, "model", "unknown")

    def _key(self, prompt: str, system: str | None) -> str:
        content = f"{self.model}\x1e{system or ''}\x1e{prompt}"
        return hashlib.sha256(content.encode("utf-8")).hexdigest()

    def generate(self, prompt: str, system: str | None = None) -> str:
        key = self._key(prompt, system)
        path = self._cache_dir / f"{key}.json"
        if path.exists():
            try:
                data = json.loads(path.read_text())
                self.hits += 1
                return str(data["response"])
            except (json.JSONDecodeError, KeyError, OSError):
                pass  # corrupt cache entry — fall through and overwrite
        response: str = self._provider.generate(prompt, system=system)
        self.misses += 1
        with contextlib.suppress(OSError):
            path.write_text(
                json.dumps(
                    {
                        "model": self.model,
                        "system": system,
                        "prompt": prompt,
                        "response": response,
                    },
                    indent=2,
                )
            )
        return response


class MockProvider:
    """Deterministic stub for tests and offline development.

    Records every call it receives and returns a fixed response (or a sequence of responses).
    """

    def __init__(self, response: str | list[str] = "OK") -> None:
        self._fixed: str | None = response if isinstance(response, str) else None
        self._queue: list[str] | None = list(response) if isinstance(response, list) else None
        self.calls: list[tuple[str, str | None]] = []

    def generate(self, prompt: str, system: str | None = None) -> str:
        self.calls.append((prompt, system))
        if self._queue is not None:
            if not self._queue:
                raise RuntimeError("MockProvider response queue exhausted")
            return self._queue.pop(0)
        assert self._fixed is not None
        return self._fixed


class OpenAIProvider:
    """OpenAI chat.completions wrapper.

    Install with the ``openai`` extra (see README). Reads ``OPENAI_API_KEY``,
    ``OPENAI_MODEL``, and ``OPENAI_BASE_URL`` from the environment if not passed in.
    Works with any OpenAI-compatible endpoint (Azure, OpenRouter, local servers) via
    ``base_url=``.
    """

    def __init__(
        self,
        model: str | None = None,
        api_key: str | None = None,
        base_url: str | None = None,
        temperature: float = 0.0,
    ) -> None:
        try:
            from openai import OpenAI
        except ImportError as e:
            raise ImportError(
                "openai not installed. Install the `openai` extra (see README)."
            ) from e
        self._client = OpenAI(
            api_key=api_key or os.environ.get("OPENAI_API_KEY"),
            base_url=base_url or os.environ.get("OPENAI_BASE_URL") or None,
        )
        self.model = model or os.environ.get("OPENAI_MODEL") or "gpt-4o-mini"
        self.temperature = temperature

    def generate(self, prompt: str, system: str | None = None) -> str:
        messages: list[dict[str, str]] = []
        if system is not None:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})
        kwargs: dict[str, Any] = {
            "model": self.model,
            "messages": messages,
        }
        if self.temperature is not None:
            kwargs["temperature"] = self.temperature
        try:
            resp = self._client.chat.completions.create(**kwargs)
        except Exception as exc:
            # Newer OpenAI models (o-series, gpt-5-*) reject a custom temperature.
            # Retry once without it so the demo still runs.
            if "temperature" in str(exc) and "temperature" in kwargs:
                kwargs.pop("temperature", None)
                resp = self._client.chat.completions.create(**kwargs)
            else:
                raise
        return resp.choices[0].message.content or ""


class DeepSeekProvider(OpenAIProvider):
    """DeepSeek via its OpenAI-compatible API.

    Install with the ``openai`` extra (reuses the openai SDK; see README). Reads
    ``DEEPSEEK_API_KEY`` and ``DEEPSEEK_MODEL`` from the environment if not passed in.
    """

    DEFAULT_BASE_URL = "https://api.deepseek.com"

    def __init__(
        self,
        model: str | None = None,
        api_key: str | None = None,
        base_url: str | None = None,
        temperature: float = 0.0,
    ) -> None:
        super().__init__(
            model=model or os.environ.get("DEEPSEEK_MODEL") or "deepseek-chat",
            api_key=api_key or os.environ.get("DEEPSEEK_API_KEY"),
            base_url=base_url or self.DEFAULT_BASE_URL,
            temperature=temperature,
        )


class AnthropicProvider:
    """Anthropic Messages API wrapper.

    Install with the ``anthropic`` extra (see README). Reads ``ANTHROPIC_API_KEY`` and
    ``ANTHROPIC_MODEL`` from the environment if not passed in.
    """

    def __init__(
        self,
        model: str | None = None,
        api_key: str | None = None,
        max_tokens: int = 1024,
    ) -> None:
        try:
            from anthropic import Anthropic
        except ImportError as e:
            raise ImportError(
                "anthropic not installed. Install the `anthropic` extra (see README)."
            ) from e
        self._client = Anthropic(api_key=api_key or os.environ.get("ANTHROPIC_API_KEY"))
        self.model = model or os.environ.get("ANTHROPIC_MODEL") or "claude-sonnet-4-6"
        self.max_tokens = max_tokens

    def generate(self, prompt: str, system: str | None = None) -> str:
        kwargs: dict[str, Any] = {
            "model": self.model,
            "max_tokens": self.max_tokens,
            "messages": [{"role": "user", "content": prompt}],
        }
        if system is not None:
            kwargs["system"] = system
        resp = self._client.messages.create(**kwargs)
        parts = [block.text for block in resp.content if getattr(block, "type", "") == "text"]
        return "".join(parts)


class GeminiProvider:
    """Google Gemini wrapper (google-genai SDK).

    Install with the ``gemini`` extra (see README). Reads ``GEMINI_API_KEY`` and
    ``GEMINI_MODEL`` from the environment if not passed in.
    """

    def __init__(
        self,
        model: str | None = None,
        api_key: str | None = None,
        temperature: float = 0.0,
    ) -> None:
        try:
            from google import genai
        except ImportError as e:
            raise ImportError(
                "google-genai not installed. Install the `gemini` extra (see README)."
            ) from e
        self._client = genai.Client(api_key=api_key or os.environ.get("GEMINI_API_KEY"))
        self.model = model or os.environ.get("GEMINI_MODEL") or "gemini-2.0-flash"
        self.temperature = temperature

    def generate(self, prompt: str, system: str | None = None) -> str:
        from google.genai import types

        config_kwargs: dict[str, Any] = {"temperature": self.temperature}
        if system is not None:
            config_kwargs["system_instruction"] = system
        resp = self._client.models.generate_content(
            model=self.model,
            contents=prompt,
            config=types.GenerateContentConfig(**config_kwargs),
        )
        return resp.text or ""


class OllamaProvider:
    """Local Ollama HTTP wrapper.

    Install with the ``ollama`` extra (see README). Reads ``OLLAMA_ENDPOINT`` and
    ``OLLAMA_MODEL`` from the environment if not passed in.
    """

    def __init__(
        self,
        model: str | None = None,
        endpoint: str | None = None,
        temperature: float = 0.0,
        timeout: float = 120.0,
    ) -> None:
        try:
            import httpx  # noqa: F401
        except ImportError as e:
            raise ImportError(
                "httpx not installed. Install the `ollama` extra (see README)."
            ) from e
        self.model = model or os.environ.get("OLLAMA_MODEL") or "qwen3:14b"
        self.endpoint = (endpoint or os.environ.get("OLLAMA_ENDPOINT") or "http://localhost:11434").rstrip("/")
        self.temperature = temperature
        self.timeout = timeout

    def generate(self, prompt: str, system: str | None = None) -> str:
        import httpx

        payload: dict[str, Any] = {
            "model": self.model,
            "prompt": prompt,
            "stream": False,
            "options": {"temperature": self.temperature},
        }
        if system is not None:
            payload["system"] = system
        resp = httpx.post(f"{self.endpoint}/api/generate", json=payload, timeout=self.timeout)
        resp.raise_for_status()
        return str(resp.json().get("response", ""))
