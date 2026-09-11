"""Unified OpenAI-compatible LLM backend.

Single interface for local vLLM and remote OpenAI-compatible APIs
(e.g. Volcano Ark). All agents and judges go through LLMBackend.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any

import openai
import yaml


@dataclass
class LLMConfig:
    base_url: str
    api_key: str = "EMPTY"
    model: str = "qwen2.5-7b"
    temperature: float = 0.0
    max_tokens: int = 1024
    timeout: float = 300.0


def load_llm_config(config: dict, role: str = "agent") -> LLMConfig:
    """role: 'agent' (provider from llm.provider) | 'judge' (remote if configured)."""
    llm_cfg = config["llm"]
    if role == "judge":
        provider = "remote" if llm_cfg["remote"].get("base_url") else "local"
    else:
        provider = llm_cfg.get("provider", "local")
    src = llm_cfg[provider]
    return LLMConfig(
        base_url=src["base_url"],
        api_key=src.get("api_key", "EMPTY"),
        model=src["model"],
        temperature=llm_cfg.get("temperature", 0.0),
        max_tokens=llm_cfg.get("max_tokens", 1024),
    )


def _chat_template_kwargs() -> dict[str, Any] | None:
    """Per-model chat-template overrides, e.g.
    AGENTIC_RAG_CHAT_KWARGS='{"enable_thinking": false}'. Used to turn off
    Qwen3-8B's default thinking mode (the think block burns the payload
    generator's max_tokens=256 budget and starves the candidate pool;
    incident 2026-09-06, see research/monitor/experiment_ledger.md)."""
    import os

    raw = os.environ.get("AGENTIC_RAG_CHAT_KWARGS")
    if not raw:
        return None
    try:
        return json.loads(raw)
    except Exception:
        return None


def _top_level_kwargs() -> dict[str, Any] | None:
    """Top-level request-body overrides, e.g.
    AGENTIC_RAG_TOP_LEVEL_KWARGS='{"reasoning_effort": "low"}'.

    Needed for gpt-oss-20b: vLLM renders it through the harmony path, which
    ignores chat_template_kwargs and reads `reasoning_effort` only as a
    top-level request field. Without it the reasoning channel eats the whole
    payload-generation budget and authority/bio candidates come back empty
    (incident 2026-09-10, see research/issues/known_issues.md)."""
    import os

    raw = os.environ.get("AGENTIC_RAG_TOP_LEVEL_KWARGS")
    if not raw:
        return None
    try:
        return json.loads(raw)
    except Exception:
        return None


class LLMBackend:
    def __init__(self, cfg: LLMConfig):
        self.cfg = cfg
        self.client = openai.OpenAI(
            base_url=cfg.base_url, api_key=cfg.api_key, timeout=cfg.timeout
        )
        self.extra_chat_template_kwargs = _chat_template_kwargs()
        self.extra_top_level_kwargs = _top_level_kwargs()

    def chat(
        self,
        messages: list[dict[str, Any]],
        temperature: float | None = None,
        max_tokens: int | None = None,
        tools: list[dict] | None = None,
    ) -> tuple[str | None, list[dict[str, Any]]]:
        """Returns (text, tool_calls). tool_calls is a list of dicts when the model
        requests tools, else []."""
        kwargs: dict[str, Any] = dict(
            model=self.cfg.model,
            messages=messages,
            temperature=self.cfg.temperature if temperature is None else temperature,
            max_tokens=self.cfg.max_tokens if max_tokens is None else max_tokens,
        )
        if tools:
            kwargs["tools"] = tools
        extra_body: dict[str, Any] = {}
        if self.extra_chat_template_kwargs:
            extra_body["chat_template_kwargs"] = self.extra_chat_template_kwargs
        if self.extra_top_level_kwargs:
            extra_body.update(self.extra_top_level_kwargs)
        if extra_body:
            kwargs["extra_body"] = extra_body
        resp = self.client.chat.completions.create(**kwargs)
        msg = resp.choices[0].message
        if msg.tool_calls:
            calls = [
                {
                    "id": c.id,
                    "function": {"name": c.function.name, "arguments": c.function.arguments},
                }
                for c in msg.tool_calls
            ]
            return msg.content, calls
        return msg.content, []

    def complete(self, prompt: str, **kw) -> str:
        text, _ = self.chat([{"role": "user", "content": prompt}], **kw)
        return text or ""

    def json_complete(self, prompt: str, **kw) -> dict:
        text = self.complete(prompt, **kw)
        text = text.strip()
        if text.startswith("```"):
            text = text.split("\n", 1)[1].rsplit("```", 1)[0].strip()
        start = text.find("{")
        end = text.rfind("}")
        return json.loads(text[start : end + 1])