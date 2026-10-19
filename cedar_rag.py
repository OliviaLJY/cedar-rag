"""Shared document loading utilities for the Cedar Lab teaching project."""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent
DOCS_DIR = PROJECT_ROOT / "docs"
CHUNK_PATTERN = re.compile(r"^\[([A-Z]\d+)\]\s+(.+)$", re.MULTILINE)


@dataclass(frozen=True)
class Chunk:
    chunk_id: str
    filename: str
    text: str


def load_chunks(docs_dir: Path = DOCS_DIR) -> list[Chunk]:
    """Load labeled paragraphs while keeping labels out of searchable text."""
    chunks: list[Chunk] = []
    for filename in ("access.txt", "training.txt", "booking.txt"):
        content = (docs_dir / filename).read_text(encoding="utf-8")
        for match in CHUNK_PATTERN.finditer(content):
            chunks.append(Chunk(match.group(1), filename, match.group(2).strip()))

    if len(chunks) != 6:
        raise ValueError(f"Expected exactly 6 chunks, found {len(chunks)}")
    return chunks

