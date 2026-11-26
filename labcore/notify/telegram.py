"""Optional Telegram message. Silent when TELEGRAM_BOT_TOKEN / TELEGRAM_CHAT_ID are unset."""

from __future__ import annotations

import os
from collections.abc import Callable, Mapping

import requests


def send(text: str, *, env: Mapping[str, str] | None = None, post: Callable = requests.post) -> bool:
    env = os.environ if env is None else env
    token, chat = env.get("TELEGRAM_BOT_TOKEN"), env.get("TELEGRAM_CHAT_ID")
    if not token or not chat:
        return False
    try:
        response = post(
            f"https://api.telegram.org/bot{token}/sendMessage",
            json={"chat_id": chat, "text": text[:4000]},
            timeout=20,
        )
        response.raise_for_status()
    except Exception as exc:
        # The request URL contains the token, so never print the exception message.
        print(f"notify: telegram failed: {type(exc).__name__}")
        return False
    return True
