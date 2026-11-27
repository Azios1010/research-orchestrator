"""Write src/zenodo_manifest.json from the index of the official ISLES'24 archive.

    python -m pip install py7zr     # needed for this script only, never inside the kernel
    python directions/isles24-ctp-reg/scripts/make_manifest.py

Only the archive's index is fetched (a few MB, through HTTP range requests), no image. The
manifest says where each needed file sits: its solid block, its offset in the decompressed block,
its size and CRC32. src/ is hash-locked, so the manifest also pins which bytes a run may read.
"""

from __future__ import annotations

import io
import json
import re
import time
import urllib.error
import urllib.request
from pathlib import Path

import py7zr

RECORD = 17652035
URL = f"https://zenodo.org/api/records/{RECORD}/files/train.7z/content"
OUTPUT = Path(__file__).resolve().parents[1] / "src" / "zenodo_manifest.json"
CHUNK = 1 << 20
LZMA2 = b"\x21"
WANTED = re.compile(
    r"^train/derivatives/(?P<subject>sub-stroke\d+)/.*_space-ncct_"
    r"(?P<kind>cbf|cbv|mtt|tmax|cta|ctp|lesion-msk)\.nii\.gz$"
)
KINDS = {"lesion-msk": "lesion_mask"}
N_KINDS = 7


class RangeFile(io.RawIOBase):
    """A read-only, seekable view of a remote file, fetched in 1 MiB pieces."""

    def __init__(self, url: str):
        self.url, self.pos, self.cache = url, 0, {}
        self.size = int(self._get(0, 0)[1].split("/")[1])

    def _get(self, start: int, end: int) -> tuple[bytes, str]:
        headers = {"Range": f"bytes={start}-{end}", "User-Agent": "research-lab"}
        request = urllib.request.Request(self.url, headers=headers)
        for attempt in range(6):
            try:
                with urllib.request.urlopen(request, timeout=120) as reply:
                    if reply.status != 206:
                        raise RuntimeError(f"expected a partial reply, got {reply.status}")
                    return reply.read(), reply.headers["Content-Range"]
            except urllib.error.HTTPError as exc:
                if exc.code not in (429, 500, 502, 503, 504) or attempt == 5:
                    raise
                time.sleep(int(exc.headers.get("Retry-After") or 30))
        raise AssertionError("unreachable")

    def readable(self):
        return True

    def seekable(self):
        return True

    def tell(self):
        return self.pos

    def seek(self, offset, whence=0):
        self.pos = offset + {0: 0, 1: self.pos, 2: self.size}[whence]
        return self.pos

    def read(self, n=-1):
        if n is None or n < 0:
            n = self.size - self.pos
        n = max(0, min(n, self.size - self.pos))
        if n > 200 * CHUNK:
            raise RuntimeError(f"refusing to read {n} bytes: only the index is wanted")
        out = bytearray()
        while len(out) < n:
            index, inside = divmod(self.pos, CHUNK)
            if index not in self.cache:
                end = min((index + 1) * CHUNK, self.size) - 1
                self.cache[index] = self._get(index * CHUNK, end)[0]
            piece = self.cache[index][inside : inside + n - len(out)]
            out += piece
            self.pos += len(piece)
        return bytes(out)


def dictionary_size(prop: int) -> int:
    """The LZMA2 dictionary size encoded in the one property byte."""
    return 0xFFFFFFFF if prop == 40 else (2 | (prop & 1)) << (prop // 2 + 11)


def build(url: str = URL) -> dict:
    remote = RangeFile(url)
    with py7zr.SevenZipFile(remote, "r") as archive:
        header = archive.header
        streams = header.main_streams
        folders = streams.unpackinfo.folders
        sizes = streams.substreamsinfo.unpacksizes
        digests = streams.substreamsinfo.digests
        counts = streams.substreamsinfo.num_unpackstreams_folders
        blocks, offset, cursor = [], 32 + streams.packinfo.packpos, 0
        for folder in folders:
            n_packed = len(folder.packed_indices)
            packed = sum(streams.packinfo.packsizes[cursor : cursor + n_packed])
            blocks.append({"offset": offset, "packed": packed, "coders": folder.coders})
            offset += packed
            cursor += n_packed
        subjects: dict[str, dict] = {}
        used_blocks: dict[str, dict] = {}
        block, in_block, start, stream = 0, 0, 0, 0
        for item in header.files_info.files:
            if item.get("emptystream"):
                continue
            while in_block >= counts[block]:
                block, in_block, start = block + 1, 0, 0
            match = WANTED.match(item["filename"])
            if match:
                coders = blocks[block]["coders"]
                if len(coders) != 1 or coders[0]["method"] != LZMA2:
                    raise RuntimeError(f"block {block} is not a single LZMA2 stream")
                kind = KINDS.get(match["kind"], match["kind"])
                subject = subjects.setdefault(match["subject"], {})
                if kind in subject:
                    raise RuntimeError(f"two {kind} files for {match['subject']}")
                subject[kind] = [block, start, sizes[stream], digests[stream]]
                used_blocks[str(block)] = {
                    "offset": blocks[block]["offset"],
                    "packed": blocks[block]["packed"],
                    "dict_size": dictionary_size(coders[0]["properties"][0]),
                }
            start += sizes[stream]
            in_block += 1
            stream += 1
    incomplete = sorted(s for s, files in subjects.items() if len(files) != N_KINDS)
    if incomplete:
        raise RuntimeError(f"subjects without all {N_KINDS} files: {incomplete}")
    return {
        "source": f"Zenodo record {RECORD}, train.7z",
        "url": url,
        "archive_size": remote.size,
        "blocks": {key: used_blocks[key] for key in sorted(used_blocks, key=int)},
        "subjects": {name: subjects[name] for name in sorted(subjects)},
    }


def main() -> None:
    manifest = build()
    lines = ['{', f' "source": {json.dumps(manifest["source"])},', f' "url": {json.dumps(manifest["url"])},']
    lines.append(f' "archive_size": {manifest["archive_size"]},')
    for key in ("blocks", "subjects"):
        rows = [f"  {json.dumps(name)}: {json.dumps(value)}" for name, value in manifest[key].items()]
        lines.append(f' "{key}": {{\n' + ",\n".join(rows) + "\n }" + ("," if key == "blocks" else ""))
    OUTPUT.write_text("\n".join(lines) + "\n}\n", encoding="utf-8", newline="\n")
    print(f"{len(manifest['subjects'])} subjects in {len(manifest['blocks'])} blocks -> {OUTPUT}")


if __name__ == "__main__":
    main()
