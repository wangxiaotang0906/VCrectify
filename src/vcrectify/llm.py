"""Fixed language-model clients with explicit, validated JSON outputs.

No client downloads weights or contacts a remote service on import. Configure a
local checkpoint or an explicit chat-completions endpoint to perform inference.
"""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Any, Dict, Mapping, Optional, Protocol, Sequence
from urllib.error import HTTPError, URLError
from urllib.parse import urlsplit
from urllib.request import Request, urlopen


class StructuredOutputError(ValueError):
    """The model returned an unusable answer; no label is silently substituted."""


class JsonClient(Protocol):
    model_id: str

    def complete_json(self, messages: Sequence[Mapping[str, str]]) -> Dict[str, Any]: ...


def parse_json_object(text: str) -> Dict[str, Any]:
    """Accept JSON (optionally a fenced JSON block), never infer a class from prose."""
    text = text.strip()
    if text.startswith("```") and text.endswith("```"):
        lines = text.splitlines()
        text = "\n".join(lines[1:-1]).strip()
    try:
        result = json.loads(text)
    except (TypeError, json.JSONDecodeError) as exc:
        raise StructuredOutputError("LLM answer is not a JSON object") from exc
    if not isinstance(result, dict):
        raise StructuredOutputError("LLM answer must be a JSON object")
    return result


class _CachedClient:
    """Content-addressed caching includes model settings and the full evidence prompt."""

    def __init__(self, cache_dir: Optional[str] = None):
        self.cache_dir = Path(cache_dir) if cache_dir else None
        if self.cache_dir:
            self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.calls = 0
        self.cache_hits = 0

    def complete_json(self, messages: Sequence[Mapping[str, str]]) -> Dict[str, Any]:
        payload = {"settings": self.cache_identity(), "messages": list(messages)}
        key = hashlib.sha256(json.dumps(payload, sort_keys=True, ensure_ascii=False).encode()).hexdigest()
        cache = self.cache_dir / (key + ".json") if self.cache_dir else None
        if cache is not None and cache.exists():
            self.cache_hits += 1
            return parse_json_object(cache.read_text(encoding="utf-8"))
        self.calls += 1
        answer = parse_json_object(self._complete_text(messages))
        if cache is not None:
            temporary = cache.with_suffix(".tmp")
            temporary.write_text(json.dumps(answer, ensure_ascii=False, indent=2), encoding="utf-8")
            temporary.replace(cache)
        return answer

    def cache_identity(self) -> Mapping[str, Any]:
        raise NotImplementedError

    def _complete_text(self, messages: Sequence[Mapping[str, str]]) -> str:
        raise NotImplementedError


class HTTPJsonClient(_CachedClient):
    """Client for an explicitly configured OpenAI-compatible /chat/completions API.

    Credentials are read from an environment variable only, excluded from cache
    keys and exceptions. Transport errors never trigger a different model.
    """

    def __init__(self, base_url: str, model: str, api_key_env: str = "VCRECTIFY_API_KEY",
                 max_new_tokens: int = 384, temperature: float = 0.0,
                 timeout: float = 120.0, cache_dir: Optional[str] = None,
                 json_mode: bool = True,
                 chat_template_kwargs: Optional[Mapping[str, Any]] = None):
        super().__init__(cache_dir)
        parsed = urlsplit(base_url)
        if parsed.scheme not in ("http", "https") or not parsed.netloc:
            raise ValueError("base_url must be an explicit HTTP(S) endpoint")
        if parsed.username or parsed.password or parsed.query or parsed.fragment:
            raise ValueError("Pass credentials through api_key_env, not the URL")
        if not model or max_new_tokens < 1 or temperature < 0 or timeout <= 0:
            raise ValueError("Invalid language-model generation configuration")
        self.endpoint = base_url.rstrip("/")
        if not self.endpoint.endswith("/chat/completions"):
            self.endpoint += "/chat/completions"
        self.model_id = model
        self.api_key_env = api_key_env
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature
        self.timeout = timeout
        self.json_mode = json_mode
        if chat_template_kwargs is not None and not isinstance(chat_template_kwargs, Mapping):
            raise ValueError("chat_template_kwargs must be a mapping or null")
        self.chat_template_kwargs = (json.loads(json.dumps(dict(chat_template_kwargs)))
                                     if chat_template_kwargs is not None else None)

    def cache_identity(self) -> Mapping[str, Any]:
        return {"backend": "http", "endpoint": self.endpoint, "model": self.model_id,
                "max_new_tokens": self.max_new_tokens, "temperature": self.temperature,
                "json_mode": self.json_mode, "chat_template_kwargs": self.chat_template_kwargs}

    def _complete_text(self, messages: Sequence[Mapping[str, str]]) -> str:
        payload = {"model": self.model_id, "messages": list(messages),
                   "temperature": self.temperature, "max_tokens": self.max_new_tokens}
        if self.json_mode:
            payload["response_format"] = {"type": "json_object"}
        if self.chat_template_kwargs is not None:
            # A vLLM/SGLang extension, not a universal chat-completions field.
            # Omitted unless explicitly requested by the user's configuration.
            payload["chat_template_kwargs"] = self.chat_template_kwargs
        headers = {"Content-Type": "application/json"}
        key = os.environ.get(self.api_key_env)
        if key:
            headers["Authorization"] = "Bearer " + key
        request = Request(self.endpoint, data=json.dumps(payload).encode(), headers=headers, method="POST")
        try:
            with urlopen(request, timeout=self.timeout) as response:
                body = json.loads(response.read().decode("utf-8"))
        except HTTPError as exc:
            raise RuntimeError("Configured LLM endpoint returned HTTP %d" % exc.code) from None
        except (URLError, TimeoutError):
            raise RuntimeError("Configured LLM endpoint could not be reached") from None
        try:
            answer = body["choices"][0]["message"]["content"]
        except (KeyError, IndexError, TypeError) as exc:
            raise StructuredOutputError("Endpoint response has no chat-completion content") from exc
        if not isinstance(answer, str):
            raise StructuredOutputError("Endpoint chat-completion content is not text")
        return answer


class TransformersJsonClient(_CachedClient):
    """Frozen local causal LM; Qwen3 thinking is disabled for compact JSON answers.

    ``local_files_only=True`` is the default. Explicitly set it false if fetching
    the named model is intended. No task-specific fitting is performed.
    """

    def __init__(self, model: str, device: str = "auto", max_new_tokens: int = 384,
                 temperature: float = 0.0, local_files_only: bool = True,
                 revision: Optional[str] = None, cache_dir: Optional[str] = None,
                 seed: int = 0):
        super().__init__(cache_dir)
        if max_new_tokens < 1 or temperature < 0:
            raise ValueError("Invalid language-model generation configuration")
        self.model_id = model
        self.device = device
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature
        self.local_files_only = local_files_only
        self.revision = revision
        self.seed = seed
        self._model = None
        self._tokenizer = None

    def cache_identity(self) -> Mapping[str, Any]:
        return {"backend": "transformers", "model": self.model_id, "revision": self.revision,
                "max_new_tokens": self.max_new_tokens, "temperature": self.temperature,
                "seed": self.seed, "thinking": False}

    def _load(self):
        if self._model is not None:
            return
        try:
            import torch
            from transformers import AutoModelForCausalLM, AutoTokenizer
        except ImportError as exc:
            raise ImportError("Local SUMMER requires the optional transformers dependency") from exc
        device = "cuda" if self.device == "auto" and torch.cuda.is_available() else self.device
        if device == "auto":
            device = "cpu"
        options = {"local_files_only": self.local_files_only, "trust_remote_code": False}
        if self.revision:
            options["revision"] = self.revision
        self._tokenizer = AutoTokenizer.from_pretrained(self.model_id, **options)
        dtype = torch.float16 if str(device).startswith("cuda") else torch.float32
        self._model = AutoModelForCausalLM.from_pretrained(self.model_id, torch_dtype=dtype, **options)
        self._model.to(device).eval()
        self._model.requires_grad_(False)

    def _complete_text(self, messages: Sequence[Mapping[str, str]]) -> str:
        self._load()
        import torch
        text = self._tokenizer.apply_chat_template(
            list(messages), tokenize=False, add_generation_prompt=True, enable_thinking=False)
        batch = self._tokenizer(text, return_tensors="pt")
        length = batch["input_ids"].shape[-1]
        max_context = getattr(self._model.config, "max_position_embeddings", None)
        if max_context is not None and length + self.max_new_tokens > max_context:
            raise ValueError("SUMMER prompt exceeds model context; reduce retrieval/evidence budgets")
        batch = {key: value.to(self._model.device) for key, value in batch.items()}
        generation = {"max_new_tokens": self.max_new_tokens, "do_sample": self.temperature > 0,
                      "pad_token_id": self._tokenizer.eos_token_id}
        if self.temperature > 0:
            generation.update(temperature=self.temperature, top_p=0.9)
        devices = [self._model.device.index or 0] if self._model.device.type == "cuda" else []
        with torch.inference_mode(), torch.random.fork_rng(devices=devices):
            torch.manual_seed(self.seed)
            output = self._model.generate(**batch, **generation)
        return self._tokenizer.decode(output[0, length:], skip_special_tokens=True)


def client_from_config(config: Mapping[str, Any]) -> JsonClient:
    backend = config.get("backend", "http")
    common = {"model": config.get("model", "Qwen/Qwen3-8B"),
              "max_new_tokens": int(config.get("max_new_tokens", 384)),
              "temperature": float(config.get("temperature", 0.0)),
              "cache_dir": config.get("cache_dir")}
    if backend == "http":
        if not config.get("base_url"):
            raise ValueError("HTTP reasoning requires an explicit base_url")
        return HTTPJsonClient(base_url=config["base_url"],
                              api_key_env=config.get("api_key_env", "VCRECTIFY_API_KEY"),
                              timeout=float(config.get("timeout", 120)),
                              json_mode=bool(config.get("json_mode", True)),
                              chat_template_kwargs=config.get("chat_template_kwargs"), **common)
    if backend == "transformers":
        return TransformersJsonClient(device=config.get("device", "auto"),
                                      local_files_only=bool(config.get("local_files_only", True)),
                                      revision=config.get("revision"), seed=int(config.get("seed", 0)),
                                      **common)
    raise ValueError("Unknown LLM backend: %s (expected transformers or http)" % backend)
