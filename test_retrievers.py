"""Small deterministic checks for document loading and keyword scoring."""

import unittest

from cedar_rag import load_chunks
from keyword_retriever import retrieve, searchable_words


Q1 = "May a trained graduate student operate the laser alone at 19:00?"


class CedarRagTests(unittest.TestCase):
    def test_exactly_six_chunks(self):
        self.assertEqual(6, len(load_chunks()))

    def test_repeated_words_do_not_increase_score(self):
        self.assertEqual(searchable_words("laser laser LASER"), {"laser"})

    def test_q1_a1_manual_score(self):
        ranked, _ = retrieve(Q1, k=2)
        a1 = next(result for result in ranked if result.chunk.chunk_id == "A1")
        # "19:00" and "08:00" each yield the alphanumeric token "00";
        # "student" does not match the unstemmed source word "students".
        self.assertEqual(("00", "graduate", "trained"), a1.matched_words)
        self.assertEqual(3, a1.score)

    def test_ties_break_by_chunk_id(self):
        ranked, _ = retrieve("nonsense", k=2)
        self.assertEqual(["A1", "A2", "B1", "B2", "T1", "T2"], [r.chunk.chunk_id for r in ranked])


if __name__ == "__main__":
    unittest.main()
