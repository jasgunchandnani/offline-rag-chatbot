import faiss, pickle, numpy as np
from sentence_transformers import SentenceTransformer

model = SentenceTransformer("all-MiniLM-L6-v2")
index = faiss.read_index("data/index/faiss.index")
chunks = pickle.load(open("data/index/chunks.pkl", "rb"))

def retrieve(query, k=4):
    q_emb = model.encode([query])
    _, idx = index.search(np.array(q_emb), k)
    return [chunks[i] for i in idx[0]]
