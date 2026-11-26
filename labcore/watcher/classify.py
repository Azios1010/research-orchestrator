"""Decide whether a failure is infrastructure ('infra', retried once) or code ('code', never).

The recorded final exception decides when there is one, so a harmless network warning earlier
in a log cannot turn a code bug into a retry. When in doubt the answer is 'code': a wrong
'infra' costs GPU hours.
"""

from __future__ import annotations

import re
from collections.abc import Mapping
from dataclasses import dataclass


@dataclass(frozen=True)
class Classification:
    failure_class: str  # "infra" | "code"
    reason: str


INFRA_EXCEPTIONS = {
    "ConnectionError", "ConnectTimeout", "ReadTimeout", "ConnectTimeoutError", "ReadTimeoutError",
    "SSLError", "ChunkedEncodingError", "ProtocolError", "RemoteDisconnected", "IncompleteRead",
    "LocalEntryNotFoundError", "ConnectionResetError", "ConnectionRefusedError",
    "ConnectionAbortedError", "gaierror", "URLError",
}

_INFRA = re.compile(
    r"temporary failure in name resolution|name or service not known|getaddrinfo failed|"
    r"max retries exceeded|connection (?:reset|refused|aborted|timed out|broken)|"
    r"\b(?:ConnectionError|ConnectTimeout|ReadTimeout|ChunkedEncodingError|LocalEntryNotFoundError)\b|"
    r"read timed out|remote end closed connection|network is unreachable|"
    r"\b(?:429|500|502|503|504) (?:server|client) error|"
    r"(?:http error|status code)[ :=]+(?:429|5\d\d)\b|"
    r"too many requests|service unavailable|bad gateway|gateway time-?out|internal server error|"
    r"download failed|consistency check failed|"
    r"session (?:error|terminated|expired|was killed|crashed)|"
    r"no (?:gpu|accelerator)s? (?:is |are )?(?:currently )?available",
    re.IGNORECASE,
)
# These look like download problems but will fail again until someone changes something.
_CONFIG = re.compile(
    r"\b(?:401|403) client error|gated ?repo|unauthorized|forbidden|repository not found|"
    r"permission denied|invalid (?:token|credentials)",
    re.IGNORECASE,
)
_OUT_OF_MEMORY = re.compile(r"out of memory|\bMemoryError\b|oom[- ]?kill", re.IGNORECASE)
_TRACEBACK = "Traceback (most recent call last)"


def classify(
    *,
    run_status: str,
    status_json: Mapping | None = None,
    log_text: str = "",
    failure_message: str | None = None,
    check_failures: list[str] | None = None,
    last_seen: str | None = None,
) -> Classification:
    """`run_status` is ERROR, TIMEOUT, CANCELLED or COMPLETE_INVALID."""
    if run_status == "CANCELLED":
        return Classification("code", "cancelled: never retried automatically")

    error_type = (status_json or {}).get("error_type")
    if error_type:
        text = f"{error_type}: {status_json.get('error_message') or ''}"
        if _CONFIG.search(text):
            return Classification("code", f"access or configuration problem ({error_type})")
        if error_type in INFRA_EXCEPTIONS:
            return Classification("infra", f"network exception {error_type}")
        match = _INFRA.search(text)
        if match:
            return Classification("infra", f"error message mentions '{match.group()}'")
        return Classification("code", f"exception {error_type} raised by the run")

    if check_failures:
        text = "\n".join(check_failures)
        match = None if _CONFIG.search(text) else _INFRA.search(text)
        if match:
            return Classification("infra", f"failed check mentions '{match.group()}'")
        return Classification("code", "sanity checks failed")

    text = f"{failure_message or ''}\n{log_text}"
    if _OUT_OF_MEMORY.search(text):
        return Classification("code", "out of memory")
    if _TRACEBACK in log_text:
        text = log_text.rsplit(_TRACEBACK, 1)[1]
        match = None if _CONFIG.search(text) else _INFRA.search(text)
        if match:
            return Classification("infra", f"last traceback mentions '{match.group()}'")
        return Classification("code", "uncaught exception in the log")
    if _CONFIG.search(text):
        return Classification("code", "access or configuration problem")
    match = _INFRA.search(text)
    if match:
        return Classification("infra", f"log mentions '{match.group()}'")
    if status_json is None:
        if run_status == "ERROR":
            return Classification("infra", "the kernel failed before our code started (Kaggle side)")
        if run_status == "TIMEOUT" and last_seen in (None, "QUEUED", "UNKNOWN"):
            return Classification("infra", "the kernel never started running (Kaggle queue)")
    if run_status == "TIMEOUT":
        return Classification("code", "the run exceeded its time limit")
    return Classification("code", "no sign of an infrastructure problem")
