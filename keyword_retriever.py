"""Transparent word-overlap retrieval baseline."""

from __future__ import annotations

import argparse
import re
from dataclasses import dataclass

from cedar_rag import Chunk, load_chunks


STOPWORDS = {
    "a", "an", "the", "is", "are", "to", "of", "in", "at", "for", "and",
    "or", "may", "can", "i", "it", "be",
}


@dataclass(frozen=True)
class KeywordResult:
    chunk: Chunk
    score: int
    matched_words: tuple[str, ...]


def searchable_words(text: str) -> set[str]:
    """Return distinct lowercase alphanumeric words minus the exact stopword list."""
    return set(re.findall(r"[a-z0-9]+", text.lower())) - STOPWORDS


def retrieve(question: str, k: int = 2) -> tuple[list[KeywordResult], list[KeywordResult]]:
    if k < 1:
        raise ValueError("k must be at least 1")

    chunks = load_chunks()  # Load: each paragraph becomes a chunk with separate metadata.
    query_words = searchable_words(question)
    scored = []
    for chunk in chunks:  # Score: count distinct words shared by question and paragraph.
        matched = tuple(sorted(query_words & searchable_words(chunk.text)))
        scored.append(KeywordResult(chunk, len(matched), matched))

    ranked = sorted(scored, key=lambda item: (-item.score, item.chunk.chunk_id))  # Rank.
    selected = [item for item in ranked if item.score > 0][:k]
    return ranked, selected


def format_run(question: str, k: int = 2) -> str:
    ranked, selected = retrieve(question, k)
    lines = [f"Question: {question}", f"k: {k}", "", "All scores:"]
    for result in ranked:
        words = ", ".join(result.matched_words) if result.matched_words else "(none)"
        lines.append(f"{result.chunk.chunk_id}: score={result.score}; matched={words}")

    lines.append("")
    if not selected:
        lines.append("No matching passages.")
    else:
        lines.append("Selected passages:")
        for result in selected:
            chunk = result.chunk
            lines.extend(
                [f"[{chunk.chunk_id}] {chunk.filename}", chunk.text, ""]
            )
    return "\n".join(lines).rstrip()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("question", help="question to retrieve evidence for")
    parser.add_argument("-k", "--k", type=int, default=2, help="positive chunks to select")
    args = parser.parse_args()
    print(format_run(args.question, args.k))


if __name__ == "__main__":
    main()
