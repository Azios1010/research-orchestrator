"""Remove secrets from logs before they are committed or posted, and data references
before anything is sent to an external model."""

from __future__ import annotations

import re

_SECRETS = [
    (re.compile(r"\bhf_[A-Za-z0-9]{20,}"), "[REDACTED]"),
    (re.compile(r"\bsk-[A-Za-z0-9_-]{16,}"), "[REDACTED]"),
    (re.compile(r"\b(?:gh[pousr]_[A-Za-z0-9]{20,}|github_pat_[A-Za-z0-9_]{20,})"), "[REDACTED]"),
    (re.compile(r"\bKGAT_[A-Za-z0-9_-]{10,}"), "[REDACTED]"),
    (re.compile(r"\b\d{6,12}:[A-Za-z0-9_-]{30,}"), "[REDACTED]"),  # Telegram bot token
    (re.compile(r"\beyJ[A-Za-z0-9_-]{8,}\.[A-Za-z0-9_-]{8,}\.[A-Za-z0-9_-]{8,}"), "[REDACTED]"),
    (re.compile(r"(?i)\bbearer\s+[A-Za-z0-9._~+/=-]{8,}"), "Bearer [REDACTED]"),
    # name=value or name: value, where the name looks secret and the value contains a digit
    (
        re.compile(
            r"(?i)([A-Za-z0-9_]*(?:api[_-]?key|token|secret|password|passwd|kaggle_key)[A-Za-z0-9_]*"
            r"['\"]?\s*[=:]\s*['\"]?)(?=[^\s'\",}()]*\d)[^\s'\",}()]{8,}"
        ),
        r"\1[REDACTED]",
    ),
    # signed download URLs
    (
        re.compile(r"(?i)([?&](?:x-amz-signature|x-goog-signature|signature|token|key)=)[^&\s'\"]+"),
        r"\1[REDACTED]",
    ),
]

_DATA_EXT = r"(?:parquet|nii\.gz|nii|npz|npy|dcm|mha|nrrd|h5|hdf5|csv)"
_DATA_REFERENCES = [
    (re.compile(r"/kaggle/(?:input|temp)(?:/[^\s'\"),:]*)?"), "[DATA_PATH]"),
    (re.compile(rf"(?:[A-Za-z]:)?(?:[\\/][^\s'\"\\/:,()]+)+\.{_DATA_EXT}\b"), "[DATA_PATH]"),
    (re.compile(rf"\b[\w.-]+\.{_DATA_EXT}\b"), "[DATA_FILE]"),
    (re.compile(r"\bsub[-_][A-Za-z]*\d+[\w-]*"), "[SUBJECT]"),
]


def redact_secrets(text: str) -> str:
    for pattern, replacement in _SECRETS:
        text = pattern.sub(replacement, text)
    return text


def redact_for_model(text: str) -> str:
    """Secrets, data paths, data file names and subject identifiers. Source file paths stay."""
    text = redact_secrets(text)
    for pattern, replacement in _DATA_REFERENCES:
        text = pattern.sub(replacement, text)
    return text
