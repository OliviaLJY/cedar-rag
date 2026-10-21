# Cedar Lab RAG teaching project

This project compares a transparent keyword baseline with real normalized embeddings for six fictional Cedar Lab passages. It records the evidence returned by each method, audits whether that evidence is complete, and produces two answers constrained to the retrieved passages.

The documents are classroom material, not real laboratory safety guidance.

## Submission details

- Personal reflection: replace the guidance in `REFLECTION.md` with your own response (150 words maximum per partner).
- Python used for this run: 3.12
- Embedding model: `sentence-transformers/all-MiniLM-L6-v2`, CPU
- Recorded package versions: `sentence-transformers 6.1.0`, `numpy 2.5.4`, `torch 2.14.1`, `transformers 5.20.1`

## Setup and exact run commands

From this directory on macOS or Linux:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python -m unittest -v
python keyword_retriever.py "May a trained graduate student operate the laser alone at 19:00?" -k 2
python embedding_retriever.py "May a trained graduate student operate the laser alone at 19:00?" -k 2 --inspect
python run_experiments.py
```

The first embedding run downloads the pretrained model and therefore needs internet access. Later runs can use the local model cache. No API key or GPU is required.

## What the code does

`cedar_rag.py` reads the three files in `docs/`. A regular expression separates each bracketed ID from its paragraph, so IDs and filenames remain metadata and are never searched or embedded. It also fails loudly unless exactly six chunks load.

`keyword_retriever.py` lowercases text, extracts alphanumeric words, removes exactly the assigned stopword list, and converts the remaining words to sets. The score is the size of the intersection of the question set and passage set, so repeating a word cannot raise a score. Results sort by negative score and then ascending chunk ID. Only positive-scoring passages are eligible, up to `k`.

The relevant code statements are marked directly in `retrieve()`:

- load: `chunks = load_chunks()`;
- score: the loop intersects the query and chunk word sets and counts the matched set;
- rank: `sorted(..., key=lambda item: (-item.score, item.chunk.chunk_id))`.

Before retrieval, the predicted Q1 evidence was A1 for general access at 19:00 and A2 for the two-person laser-operating rule. The actual baseline selected both. A manual A1 check gives score 3 from the distinct shared tokens `00`, `graduate`, and `trained`. The exact alphanumeric tokenizer splits both clock times and creates the shared token `00`; the specified no-stemming rule means `student` and `students` do not match.

`embedding_retriever.py` loads the same six paragraph texts into `encode_document()` once. It sends each question through `encode_query()`. Both calls use the same model, request NumPy arrays, and normalize vectors to length 1. The matrix multiplication `document_vectors @ query_vector` computes six dot products at once. Because the vectors are normalized, each dot product equals cosine similarity. Sorting those six numbers and slicing the first `k` turns them into the selected passages. There is intentionally no relevance threshold.

The document matrix is `(6, 384)` and each question vector is `(384,)`. Both sides need 384 coordinates in the same learned space so coordinates can be multiplied pairwise and summed. The inspection file shows only 12 coordinates for readability, but retrieval uses all 384. Among those displayed values, the largest is `0.1357` and the smallest is `-0.1167`; neither number alone says what a passage means because individual coordinates have no reliable human-readable label.

## Results and audit

The complete actual runs are under `evidence/`, and the compact comparison is in `results.csv`.

- Q1: keyword retrieval selected A1 (3) and A2 (2), complete evidence. Embeddings selected A2 (0.686446) and T1 (0.581953), so A1 was missing and coverage was 1/2.
- Q2: both methods retrieved A2. Its embedding score was 0.585193.
- Q3: both methods retrieved T1. Its embedding score was 0.742027.
- Q4: the corpus contains no wavelength. Embeddings still returned T1 (0.347721) and A2 (0.320827) because nearest-neighbor retrieval always returns the requested top `k`; positive similarity does not prove answerability.

For Q1, changing embedding `k` from 1 to 3 did not recover A1. The additional T1 and B1 passages supplied training and booking conditions, but not the missing access-hours rule. This is a concrete example of larger context adding plausible but incomplete material.

The grounded Q1 answer therefore says the user cannot operate alone based on A2, while explicitly noting that the retrieved passages do not establish the 19:00 access-hours condition. The Q4 answer states that wavelength is unspecified. `evidence/answer-audits.txt` checks every claim and does not invent a citation for missing information.

## Files

- `docs/`: six verbatim source chunks in the three assigned files
- `keyword_retriever.py`: exact word-overlap baseline
- `embedding_retriever.py`: MiniLM retrieval and vector inspection
- `run_experiments.py`: reproducible Q1-Q4 runs, CSV, k experiment, and answer audit
- `test_retrievers.py`: checks chunk count, set behavior, manual score, and tie-breaking
- `results.csv`: ranked selections, scores, required evidence, and judgments
- `evidence/keyword-baseline.txt`: Q1 baseline output and manual score audit
- `evidence/embedding-inspection.txt`: shapes, visible coordinates, norms, and cosine check
- `evidence/retrieval-runs.txt`: actual detailed output for Q1-Q4 from both methods
- `evidence/k-experiment.txt`: Q1 at embedding k=1 and k=3
- `evidence/answer-audits.txt`: evidence-only answers and claim-by-claim checks

## GitHub handoff

After adding names and original reflection, create an empty GitHub repository named `cedar-rag`, then run:

```bash
git add .
git commit -m "Complete Cedar Lab RAG assignment"
git branch -M main
git remote add origin https://github.com/YOUR-USER/cedar-rag.git
git push -u origin main
```

No remote is configured here because a GitHub account and repository URL were not supplied.
