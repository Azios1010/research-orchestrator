"""One OpenAI-compatible chat call to DeepSeek that turns a log tail into diagnosis.md.

Only the redacted log tail and status.json are sent: no data, no images, no configs.
Any failure produces "diagnosis unavailable" and the watcher carries on.
"""

from __future__ import annotations

import json
import os
from collections.abc import Callable, Mapping
from pathlib import Path

import requests

from labcore.redact import redact_for_model

DEFAULT_BASE_URL = "https://api.deepseek.com"
# UNVERIFIED: model name. api-docs.deepseek.com listed "deepseek-flash" on 2026-11-26
# ("deepseek-chat" was retired on 2026-07-24). Override with DEEPSEEK_MODEL.
DEFAULT_MODEL = "deepseek-flash"
# UNVERIFIED: thinking mode is documented as on by default; this turns it off to keep the call small.
THINKING_OFF = {"type": "disabled"}
MAX_LINES = 8
MAX_LINE_CHARS = 300
MAX_LOG_CHARS = 12000
STATUS_KEYS = ("stage", "ok", "error_type", "error_message", "traceback_tail")
UNAVAILABLE = "diagnosis unavailable"
PROMPT_PATH = Path(__file__).resolve().parent / "prompt_template.md"


def build_messages(log_tail: str, status: Mapping | None) -> list[dict]:
    safe_status = {
        key: redact_for_model(value) if isinstance(value, str) else value
        for key, value in (status or {}).items()
        if key in STATUS_KEYS
    }
    user = (
        "status.json:\n"
        + json.dumps(safe_status, indent=2)
        + "\n\nLog tail:\n"
        + redact_for_model(log_tail)[-MAX_LOG_CHARS:]
    )
    return [
        {"role": "system", "content": PROMPT_PATH.read_text(encoding="utf-8")},
        {"role": "user", "content": user},
    ]


def diagnose(
    log_tail: str,
    status: Mapping | None,
    *,
    env: Mapping[str, str] | None = None,
    post: Callable = requests.post,
    timeout: int = 90,
) -> str:
    env = os.environ if env is None else env
    key = env.get("DEEPSEEK_API_KEY")
    if not key:
        return f"{UNAVAILABLE} (DEEPSEEK_API_KEY is not set)"
    base = (env.get("DEEPSEEK_BASE_URL") or DEFAULT_BASE_URL).rstrip("/")
    body = {
        "model": env.get("DEEPSEEK_MODEL") or DEFAULT_MODEL,
        "messages": build_messages(log_tail, status),
        "max_tokens": 600,
        "stream": False,
    }
    if "api.deepseek.com" in base:
        body["thinking"] = THINKING_OFF
    try:
        response = post(
            f"{base}/chat/completions",
            headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json"},
            json=body,
            timeout=timeout,
        )
        response.raise_for_status()
        text = response.json()["choices"][0]["message"]["content"]
    except Exception as exc:  # never let the optional summary break the watcher
        # Only the exception type: messages can echo request details.
        return f"{UNAVAILABLE} (request failed: {type(exc).__name__})"
    lines = [line.rstrip()[:MAX_LINE_CHARS] for line in str(text or "").splitlines() if line.strip()]
    return "\n".join(lines[:MAX_LINES]) or f"{UNAVAILABLE} (empty reply)"
