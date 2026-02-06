# Offline RAG Chatbot

This repository contains a fully offline Retrieval-Augmented Generation (RAG) prototype. It demonstrates an end-to-end local pipeline for ingesting documents, building a vector index, and answering user questions with a local GGUF LLM. No paid APIs or cloud-hosted models are required.

## 1. Project Overview

What the system does

- Ingests documents from a local folder, splits them into chunks, and embeds them with an open-source sentence-transformer.
- Builds a FAISS index of embeddings and saves the index and text chunks to disk.
- Runs a local LLM (GGUF format via GPT4All / llama.cpp) to generate answers constrained to retrieved context.
- Serves a minimal Streamlit UI for interactive querying.

Why an offline RAG approach was chosen

- Reproducibility and privacy: all data and models remain on the host machine.
- Cost control: no external API or paid LLM dependency.
- Portability for take-home or interview work where internet or paid accounts may not be available.

## 2. System Architecture

High-level pipeline

ingestion → chunking → embedding → index build → retrieval → generation (inference)

Component responsibilities

- Ingest: `src/ingest.py` loads PDF and TXT files from `data/docs/` and splits text into overlapping chunks.
- Embed / Index: `src/embed.py` encodes chunks using SentenceTransformers and writes a FAISS index (`data/index/faiss.index`) and a pickled chunks file (`data/index/chunks.pkl`). This step is explicit and run offline prior to inference.
- Retrieve: `src/retrieve.py` loads the FAISS index and the same SentenceTransformer to embed queries, performs k-NN search and returns top chunks.
- Generate: `src/llm.py` loads a local GGUF model via GPT4All and generates an answer. The prompt forces a context-only response; if the answer is not present in the context the LLM should respond with “I don't know.”
- UI: `app.py` provides a Streamlit front-end for interactive prompts.
- Evaluation: `src/evaluate.py` contains simple metrics (Recall@K and Context Precision) to assess retrieval quality.

## 3. Technology Stack

- SentenceTransformers (`all-MiniLM-L6-v2`) — lightweight, high-quality embeddings for small/medium scale tasks.
- FAISS (`faiss-cpu`) — efficient nearest-neighbour search locally.
- GPT4All / llama.cpp (GGUF) — runs a local LLM from a GGUF file for offline generation.
- PyPDF2 — PDF text extraction.
- Streamlit — minimal web UI for interaction.
- NumPy / scikit-learn — numerical utilities used by embedding / retrieval code.

Rationale: the stack is chosen to keep the whole pipeline local, lightweight, and easy to reproduce in a personal environment.

## 4. Project Structure (key files)

- `app.py` — Streamlit app, minimal UI for queries.
- `src/ingest.py` — document loading and chunking utilities.
- `src/embed.py` — builds embeddings and writes FAISS index + chunks to `data/index/`.
- `src/retrieve.py` — loads index, encodes query, returns top-k chunks.
- `src/llm.py` — loads local GGUF model (via GPT4All) and generates answers constrained to context.
- `src/rag.py` — small orchestrator: retrieve + generate.
- `src/evaluate.py` — simple retrieval evaluation functions: `recall_at_k` and `context_precision`.
- `data/docs/` — input documents (PDF/TXT). Replace or add files here before indexing.
- `data/index/` — produced artifacts: `faiss.index`, `chunks.pkl`.
- `models/llm/` — place the GGUF model file here (see next section).

## 5. Installation & Setup

1. Python environment

- Recommended: create and activate a virtual environment (zsh example):

  python -m venv .venv
  source .venv/bin/activate

2. Install dependencies

- Install required packages:

  pip install -r requirements.txt

3. Model placement

- Put your GGUF model file into `models/llm/`.
- Update `src/llm.py` if you need to change the model filename or path. The repo expects the model to be available offline and does not attempt to download models automatically (the code sets allow_download=False).

4. Data

- Add documents (PDF or TXT) to `data/docs/` before building the index.

## 6. How to Run

Indexing (one-time / offline step)

- Build embeddings and the FAISS index:

  python src/embed.py

- Successful completion writes `data/index/faiss.index` and `data/index/chunks.pkl`.

Inference (interactive)

- Start the Streamlit UI:

  streamlit run app.py

- Enter a question in the UI. The system will retrieve top-k chunks and prompt the local LLM to answer using only the provided context.

Notes

- The embedding model is re-used at inference time in `src/retrieve.py` to encode queries. This keeps the index and query embedding model consistent.
- If you change the embedding model, rebuild the index.

## 7. Evaluation Metrics

Two simple metrics included in `src/evaluate.py`:

- Recall@K: fraction of ground-truth passages that appear in the top-K retrieved chunks. Useful to measure whether relevant pieces of information are present in retrieval results.
- Context Precision: fraction of the top-K retrieved chunks that contain any of the ground-truth tokens/phrases. Gives an estimate of how much of the retrieved context is actually relevant to the expected answer.

These metrics are retrieval-focused; they do not directly measure end-to-end answer quality. For RAG systems, retrieval quality is critical because the generative step is constrained to retrieved context.

## 8. Design Decisions

- Minimal external dependencies: avoid cloud APIs and paid services; use widely available open-source components.
- Separation of indexing vs inference: embedding and index-building are explicit offline steps to decouple expensive work from low-latency query-time inference.
- Hallucination control: the generation prompt requires the model to answer only from the supplied context and to return an explicit "I don't know." when the context lacks an answer. This is a simple but effective operational constraint to reduce unsupported model responses.
- Simplicity over complexity: FAISS flat index + SentenceTransformers is easy to understand and sufficient for small to medium datasets typical in take-home assignments.

## 9. Limitations & Future Improvements

Limitations

- The current pipeline uses a memory-backed FAISS IndexFlatL2; it may not scale well to very large corpora without switching to on-disk/index-shard approaches.
- Retrieval quality depends on chunking heuristics and the embedding model; current chunking is simple fixed-size token windows with overlap.
- No re-ranker is used; the LLM is only fed the raw top-k chunks in order.
- No metadata management (source attribution, timestamps) for returned chunks.
- No automated tests or CI configured in this prototype.

Suggested next steps

- Add a lexical/semantic re-ranker (e.g., cross-encoder) to improve top-k ordering.
- Use an on-disk or IVF+PQ FAISS index for larger datasets.
- Add metadata tracking and provenance for every chunk returned to support traceability.
- Improve chunking (heuristic or content-aware split) and consider sentence/paragraph boundaries.
- Add unit tests for ingestion, retrieval, and prompt templates; add small-scale integration tests.



