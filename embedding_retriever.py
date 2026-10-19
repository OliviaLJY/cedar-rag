"""Semantic retrieval with all-MiniLM-L6-v2 sentence embeddings."""

from __future__ import annotations

import argparse
import importlib.metadata
from dataclasses import dataclass

import numpy as np
from sentence_transformers import SentenceTransformer

from cedar_rag import Chunk, load_chunks


MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"


@dataclass(frozen=True)
class EmbeddingResult:
    chunk: Chunk
    score: float
    index: int


def package_versions() -> dict[str, str]:
    return {
        name: importlib.metadata.version(name)
        for name in ("sentence-transformers", "numpy", "torch", "transformers")
    }


class EmbeddingRetriever:
    """Encode six documents once and reuse their vectors for many questions."""

    def __init__(self) -> None:
        self.chunks = load_chunks()
        self.model = SentenceTransformer(MODEL_NAME, device="cpu")
        texts = [chunk.text for chunk in self.chunks]
        self.document_vectors = self.model.encode_document(
            texts, normalize_embeddings=True, convert_to_numpy=True
        )

    def retrieve(
        self, question: str, k: int = 2
    ) -> tuple[np.ndarray, list[EmbeddingResult], list[EmbeddingResult]]:
        if not 1 <= k <= len(self.chunks):
            raise ValueError(f"k must be between 1 and {len(self.chunks)}")
        query_vector = self.model.encode_query(
            question, normalize_embeddings=True, convert_to_numpy=True
        )
        scores = self.document_vectors @ query_vector
        ranked = sorted(
            (
                EmbeddingResult(chunk, float(scores[index]), index)
                for index, chunk in enumerate(self.chunks)
            ),
            key=lambda item: (-item.score, item.chunk.chunk_id),
        )
        return query_vector, ranked, ranked[:k]


def _vector(values: np.ndarray, count: int = 12) -> str:
    return "[" + ", ".join(f"{value:.4f}" for value in values[:count]) + "]"


def format_run(retriever: EmbeddingRetriever, question: str, k: int = 2, inspect: bool = False) -> str:
    query_vector, ranked, selected = retriever.retrieve(question, k)
    versions = package_versions()
    lines = [
        f"Model: {MODEL_NAME}",
        "Device: CPU",
        "Versions: " + ", ".join(f"{name}={value}" for name, value in versions.items()),
        f"Document matrix shape: {retriever.document_vectors.shape}",
        f"Question vector shape: {query_vector.shape}",
        "Displayed coordinates: first 12; retrieval uses all 384 coordinates.",
        "",
        "Document vectors (first 12 coordinates and full-vector length):",
    ]
    for chunk, vector in zip(retriever.chunks, retriever.document_vectors):
        lines.append(f"{chunk.chunk_id}: {_vector(vector)} length={np.linalg.norm(vector):.6f}")
    lines.extend(
        [
            "",
            "Question vector:",
            f"{_vector(query_vector)} length={np.linalg.norm(query_vector):.6f}",
            "",
            "All scores (descending):",
        ]
    )
    for result in ranked:
        lines.append(f"{result.chunk.chunk_id}: {result.score:.6f}")

    lines.extend(["", f"Selected top {k} passages:"])
    for result in selected:
        lines.extend(
            [
                f"[{result.chunk.chunk_id}] {result.chunk.filename} score={result.score:.6f}",
                result.chunk.text,
                "",
            ]
        )

    if inspect:
        top = selected[0]
        document_vector = retriever.document_vectors[top.index]
        products = query_vector * document_vector
        subtotal_first = float(np.sum(products[:12]))
        subtotal_remaining = float(np.sum(products[12:]))
        full_similarity = float(np.sum(products))
        direct_cosine = float(
            np.dot(query_vector, document_vector)
            / (np.linalg.norm(query_vector) * np.linalg.norm(document_vector))
        )
        displayed = np.concatenate(
            [retriever.document_vectors[:, :12].reshape(-1), query_vector[:12]]
        )
        lines.extend(
            [
                f"Q1 coordinate-product inspection for top passage {top.chunk.chunk_id}:",
                f"First 12 products: {_vector(products)}",
                f"First-12 subtotal: {subtotal_first:.8f}",
                f"Remaining-372 subtotal: {subtotal_remaining:.8f}",
                f"Full dot-product similarity: {full_similarity:.8f}",
                f"Direct cosine from vector lengths: {direct_cosine:.8f}",
                f"Agreement within tolerance (1e-6): {np.isclose(full_similarity, direct_cosine, atol=1e-6)}",
                f"Largest displayed coordinate: {float(np.max(displayed)):.4f}",
                f"Smallest displayed coordinate: {float(np.min(displayed)):.4f}",
                "A coordinate alone has no dependable semantic label; meaning and similarity depend on the vector as a whole.",
            ]
        )
    return "\n".join(lines).rstrip()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("question")
    parser.add_argument("-k", "--k", type=int, default=2)
    parser.add_argument("--inspect", action="store_true", help="show coordinate math for the top result")
    args = parser.parse_args()
    retriever = EmbeddingRetriever()
    print(format_run(retriever, args.question, args.k, args.inspect))


if __name__ == "__main__":
    main()
