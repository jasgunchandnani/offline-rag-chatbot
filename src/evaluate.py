from retrieve import retrieve

def recall_at_k(query, ground_truth, k=4):
    retrieved = retrieve(query, k)
    hits = sum(1 for g in ground_truth if g in " ".join(retrieved))
    return hits / len(ground_truth)

def context_precision(query, ground_truth, k=4):
    retrieved = retrieve(query, k)
    relevant = sum(1 for r in retrieved if any(g in r for g in ground_truth))
    return relevant / k
