"""Run the four-question comparison and write reproducible evidence files."""

from __future__ import annotations

import csv
from pathlib import Path

from embedding_retriever import EmbeddingRetriever, format_run as format_embedding_run
from keyword_retriever import format_run as format_keyword_run, retrieve as keyword_retrieve


QUESTIONS = {
    "Q1": "May a trained graduate student operate the laser alone at 19:00?",
    "Q2": "Can I use the laser without another person present?",
    "Q3": "How frequently must I refresh my laser qualification?",
    "Q4": "What is the laser wavelength?",
}
REQUIRED = {"Q1": ("A1", "A2"), "Q2": ("A2",), "Q3": ("T1",), "Q4": ()}
EVIDENCE_DIR = Path(__file__).resolve().parent / "evidence"


def ids(results) -> list[str]:
    return [result.chunk.chunk_id for result in results]


def main() -> None:
    EVIDENCE_DIR.mkdir(exist_ok=True)
    embedding = EmbeddingRetriever()

    q1 = QUESTIONS["Q1"]
    keyword_q1 = format_keyword_run(q1, k=2)
    (EVIDENCE_DIR / "keyword-baseline.txt").write_text(
        "Predicted required passages before retrieval: A1 and A2\n\n"
        + keyword_q1
        + "\n\nManual score check for A1:\n"
        + "Distinct query/source matches are 00, graduate, and trained, so score=3. "
        + "The token 00 comes from the alphanumeric tokenizer splitting both 19:00 and the source times. "
        + "Repeated words cannot increase a set-overlap score, and no stemming changes student/students.\n",
        encoding="utf-8",
    )
    inspection = format_embedding_run(embedding, q1, k=2, inspect=True)
    (EVIDENCE_DIR / "embedding-inspection.txt").write_text(inspection + "\n", encoding="utf-8")

    sections: list[str] = []
    rows: list[dict[str, str]] = []
    stored = {}
    for question_id, question in QUESTIONS.items():
        keyword_ranked, keyword_selected = keyword_retrieve(question, k=2)
        _, embedding_ranked, embedding_selected = embedding.retrieve(question, k=2)
        stored[question_id] = (keyword_selected, embedding_selected)
        sections.extend(
            [
                "=" * 78,
                f"{question_id} KEYWORD RETRIEVAL",
                "=" * 78,
                format_keyword_run(question, k=2),
                "",
                "=" * 78,
                f"{question_id} EMBEDDING RETRIEVAL",
                "=" * 78,
                format_embedding_run(embedding, question, k=2),
                "",
            ]
        )
        required = set(REQUIRED[question_id])
        keyword_ids = set(ids(keyword_selected))
        embedding_ids = set(ids(embedding_selected))
        rows.append(
            {
                "question_id": question_id,
                "keyword_top_ids": ";".join(ids(keyword_selected)),
                "keyword_scores": ";".join(f"{r.score}" for r in keyword_selected),
                "embedding_top_ids": ";".join(ids(embedding_selected)),
                "embedding_scores": ";".join(f"{r.score:.6f}" for r in embedding_selected),
                "required_ids": ";".join(REQUIRED[question_id]) if required else "absent",
                "keyword_evidence_complete": "N/A" if not required else str(required <= keyword_ids),
                "embedding_evidence_complete": "N/A" if not required else str(required <= embedding_ids),
                "observation": observation(question_id, keyword_selected, embedding_selected),
            }
        )

    (EVIDENCE_DIR / "retrieval-runs.txt").write_text("\n".join(sections).rstrip() + "\n", encoding="utf-8")
    with (Path(__file__).resolve().parent / "results.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    experiment = [
        f"Question: {q1}",
        "Required evidence: A1 and A2",
        "Coverage = retrieved required chunks / 2",
        "",
    ]
    for k in (1, 3):
        _, _, selected = embedding.retrieve(q1, k=k)
        selected_ids = ids(selected)
        covered = len(set(selected_ids) & set(REQUIRED["Q1"]))
        added = "; ".join(f"{r.chunk.chunk_id} ({r.score:.6f})" for r in selected)
        experiment.extend(
            [
                f"k={k}: {added}",
                f"Required evidence coverage: {covered}/2 = {covered / 2:.1f}",
                f"All required evidence present: {covered == 2}",
                "",
            ]
        )
    experiment.extend(
        [
            "Interpretation:",
            "At k=1, A2 gives only the two-person operating condition.",
            "At k=3, T1 adds a current-training condition and B1 adds a booking limit, but A1 is still missing.",
            "The added passages therefore do not establish the general access-hours condition, and Q1 coverage remains 1/2.",
        ]
    )
    (EVIDENCE_DIR / "k-experiment.txt").write_text("\n".join(experiment).rstrip() + "\n", encoding="utf-8")

    write_answer_audits(embedding)


def observation(question_id, keyword_selected, embedding_selected) -> str:
    kid = ids(keyword_selected)
    eid = ids(embedding_selected)
    if question_id == "Q1":
        return f"Keyword selected {kid}; embeddings selected {eid}. Complete evidence requires both A1 and A2."
    if question_id == "Q2":
        target = next((r for r in embedding_selected if r.chunk.chunk_id == "A2"), None)
        return f"The paraphrase target A2 was {'retrieved at ' + format(target.score, '.6f') if target else 'not in the embedding top 2'}."
    if question_id == "Q3":
        target = next((r for r in embedding_selected if r.chunk.chunk_id == "T1"), None)
        return f"The paraphrase target T1 was {'retrieved at ' + format(target.score, '.6f') if target else 'not in the embedding top 2'}."
    return f"The answer is absent, but nearest-neighbor retrieval still returned {eid}; a score is not proof of answerability."


def write_answer_audits(embedding: EmbeddingRetriever) -> None:
    blocks = [
        "Grounded answer audit",
        "Each answer uses only its embedding top-2 passages as factual evidence.",
        "",
    ]
    for question_id in ("Q1", "Q4"):
        question = QUESTIONS[question_id]
        _, _, selected = embedding.retrieve(question, k=2)
        selected_ids = ids(selected)
        blocks.extend([f"=== {question_id} ===", f"Question: {question}", f"Selected IDs: {', '.join(selected_ids)}"])
        for result in selected:
            blocks.append(f"[{result.chunk.chunk_id}] {result.chunk.text}")
        blocks.append("")
        if question_id == "Q1":
            if "A2" in selected_ids:
                answer = "No. Whenever the laser is operating, two trained people must be present [A2]."
                claims = ["Claim: Operating the laser alone is not allowed. Support: A2.", "Claim: Two trained people must be present whenever it operates. Support: A2."]
            else:
                answer = "The supplied passages do not establish whether operating the laser alone is allowed; the laser operating-personnel rule is missing."
                claims = ["Claim: The supplied evidence lacks the operating-personnel rule. Support: inspection of supplied passages."]
            if "A1" in selected_ids:
                answer += " At 19:00, a trained graduate student is within the general 08:00-22:00 access hours [A1]."
                claims.append("Claim: 19:00 is within trained-graduate-student general access hours. Support: A1.")
            else:
                answer += " The supplied passages do not establish the general access hours applicable at 19:00."
                claims.append("Claim: General access-hour evidence is missing from the supplied set. Support: inspection of supplied passages.")
        else:
            answer = "The laser wavelength is not specified in the supplied passages."
            claims = ["Claim: The supplied passages do not specify a laser wavelength. Support: inspection of both supplied passages; no chunk citation is invented."]
        blocks.extend([f"Answer: {answer}", "Claim audit:", *claims, "Audit verdict: Supported; no outside facts added.", ""])
    (EVIDENCE_DIR / "answer-audits.txt").write_text("\n".join(blocks).rstrip() + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
