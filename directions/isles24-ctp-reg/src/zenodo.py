"""Read single files of the official ISLES'24 release without downloading all of it.

The release is one 7z of about 99 GB made of solid LZMA2 blocks of about 4 GB. A file can only be
decompressed by reading its block from the start, so each needed block is streamed from its first
byte up to the last wanted file; everything else is thrown away. zenodo_manifest.json (written by
scripts/make_manifest.py) gives every file's block, offset, size and CRC32. Every extracted file is
checked against its CRC32.
"""

from __future__ import annotations

import http.client
import io
import json
import lzma
import threading
import time
import urllib.error
import urllib.request
import zlib
from collections.abc import Callable
from pathlib import Path

MANIFEST = Path(__file__).resolve().parent / "zenodo_manifest.json"
CHUNK = 1 << 20
KINDS = ("cbf", "cbv", "mtt", "tmax", "cta", "ctp", "lesion_mask")


class SourceError(RuntimeError):
    """The archive could not give a file: network given up, or the bytes are not what was listed."""


class Stopped(SourceError):
    """The caller asked all streams to stop."""


def load_manifest(path: str | Path = "") -> dict:
    return json.loads(Path(path or MANIFEST).read_text(encoding="utf-8"))


def select_subjects(manifest: dict, n: int) -> list[str]:
    """The first n subjects in a fixed order that keeps the download small.

    Subjects whose files all sit in one block come first: the earliest-ending one of every block
    (blocks in archive order), then the second of every block, and so on. Subjects split over two
    blocks follow, by name. The order depends only on where the archive's packer put the files.
    """
    subjects = manifest["subjects"]
    if not 0 < n <= len(subjects):
        raise ValueError(f"n_cases must be between 1 and {len(subjects)}, got {n}")
    in_block: dict[int, list] = {}
    split = []
    for name, files in subjects.items():
        blocks = {entry[0] for entry in files.values()}
        if len(blocks) == 1:
            end = max(start + size for _, start, size, _ in files.values())
            in_block.setdefault(blocks.pop(), []).append((end, name))
        else:
            split.append(name)
    ranked = sorted(
        (position, block, name)
        for block, members in in_block.items()
        for position, (_, name) in enumerate(sorted(members))
    )
    return ([name for _, _, name in ranked] + sorted(split))[:n]


def plan(manifest: dict, subjects: list[str]) -> dict[int, list[tuple[int, int, int, str, str]]]:
    """{block: [(start, size, crc, subject, kind), ...] in stream order} for the wanted subjects."""
    blocks: dict[int, list] = {}
    for name in subjects:
        files = manifest["subjects"][name]
        for kind in KINDS:
            block, start, size, crc = files[kind]
            blocks.setdefault(block, []).append((start, size, crc, name, kind))
    return {block: sorted(files) for block, files in blocks.items()}


def expected_bytes(wanted: dict[int, list]) -> int:
    """About how much must be downloaded: each block up to the end of its last wanted file.

    The blocks compress by a few percent only, so decompressed bytes slightly overstate this.
    """
    return sum(max(start + size for start, size, *_ in files) for files in wanted.values())


# ---------------------------------------------------------------- byte ranges


def open_url(url: str) -> Callable:
    def open_range(start: int, end: int):
        headers = {"Range": f"bytes={start}-{end}", "User-Agent": "research-lab"}
        reply = urllib.request.urlopen(urllib.request.Request(url, headers=headers), timeout=120)
        if reply.status != 206 or not reply.headers.get("Content-Range", "").startswith(f"bytes {start}-"):
            reply.close()
            raise SourceError(f"the server did not honour a byte range (status {reply.status})")
        return reply

    return open_range


def open_file(path: str | Path) -> Callable:
    """The same interface over a local file: offline testing only."""

    def open_range(start: int, end: int):
        with open(path, "rb") as handle:
            handle.seek(start)
            return io.BytesIO(handle.read(end - start + 1))

    return open_range


# ---------------------------------------------------------------- one block


def stream_block(
    open_range: Callable,
    block: dict,
    files: list[tuple[int, int, int, str, str]],
    dest: Path,
    on_file: Callable[[str, str, Path], None],
    *,
    stop: threading.Event,
    count: Callable[[int], None] = lambda n: None,
    retries: int = 8,
    sleep: Callable[[float], None] = time.sleep,
) -> None:
    """Decompress one block up to its last wanted file, writing the wanted files into `dest`.

    A dropped connection is reopened at the first compressed byte not yet decompressed; the
    decompressor keeps its state, so nothing is downloaded twice. `retries` counts consecutive
    failures: any data received resets it.
    """
    decoder = lzma.LZMADecompressor(
        format=lzma.FORMAT_RAW, filters=[{"id": lzma.FILTER_LZMA2, "dict_size": block["dict_size"]}]
    )
    last_byte = block["offset"] + block["packed"] - 1
    fed = produced = failures = index = crc = 0
    reply = handle = None
    try:
        while index < len(files):
            if stop.is_set():
                raise Stopped("stopped")
            try:
                if reply is None:
                    reply = open_range(block["offset"] + fed, last_byte)
                data = reply.read(CHUNK)
                if not data:
                    raise ConnectionError("the stream ended early")
            except (OSError, http.client.HTTPException) as exc:
                if reply is not None:
                    reply.close()
                    reply = None
                code = getattr(exc, "code", None) if isinstance(exc, urllib.error.HTTPError) else None
                if code is not None and 400 <= code < 500 and code not in (408, 429):
                    raise SourceError(f"the server refused the request: HTTP {code}") from exc
                failures += 1
                if failures > retries:
                    reason = f"{type(exc).__name__}: {exc}"
                    raise SourceError(f"gave up after {failures} attempts: {reason}") from exc
                sleep(min(60.0, 5.0 * failures))
                continue
            failures = 0
            fed += len(data)
            count(len(data))
            try:
                out = memoryview(decoder.decompress(data))
            except (lzma.LZMAError, EOFError) as exc:
                raise SourceError(f"the block cannot be decompressed: {exc}") from exc
            while index < len(files):
                start, size, expected_crc, subject, kind = files[index]
                low, high = max(start - produced, 0), min(start + size - produced, len(out))
                if high <= low:
                    break  # this file begins in a later piece
                path = Path(dest) / f"{subject}_{kind}.nii.gz"
                if handle is None:
                    handle, crc = open(path, "wb"), 0
                handle.write(out[low:high])
                crc = zlib.crc32(out[low:high], crc)
                if produced + high < start + size:
                    break  # the rest follows
                handle.close()
                handle = None
                if crc != expected_crc:
                    raise SourceError(f"CRC mismatch in {subject} {kind}")
                index += 1
                on_file(subject, kind, path)
            produced += len(out)
            if decoder.eof and index < len(files):
                raise SourceError("the block ended before the last wanted file")
    finally:
        if reply is not None:
            reply.close()
        if handle is not None:
            handle.close()
