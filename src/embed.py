def main():
    from sentence_transformers import SentenceTransformer
    import faiss
    import pickle
    import numpy as np
    from ingest import load_docs, chunk

    print("Loading embedding model...")
    model = SentenceTransformer("all-MiniLM-L6-v2")

    print("Loading documents...")
    docs = load_docs()
    chunks = sum([chunk(d) for d in docs], [])

    print(f"Total chunks: {len(chunks)}")

    # 🔒 SAFETY CHECK (ADD IT HERE)
    if len(chunks) == 0:
        raise ValueError(
            "No document chunks found. "
            "Ensure files exist in data/docs and contain extractable text."
        )

    print("Generating embeddings...")
    embeddings = model.encode(chunks, show_progress_bar=True)

    index = faiss.IndexFlatL2(embeddings.shape[1])
    index.add(np.array(embeddings))

    faiss.write_index(index, "data/index/faiss.index")
    pickle.dump(chunks, open("data/index/chunks.pkl", "wb"))

    print("FAISS index built successfully.")

if __name__ == "__main__":
    main()
