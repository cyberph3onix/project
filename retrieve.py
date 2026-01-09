import json
import numpy as np
from sentence_transformers import SentenceTransformer
from numpy.linalg import norm

# Load embedding model
model = SentenceTransformer("all-MiniLM-L6-v2")

# Load chunks
chunks = []
with open("chunks.jsonl", "r") as f:
    for line in f:
        obj = json.loads(line)
        chunks.append(obj["chunk"])

# Embed chunks once
chunk_embeddings = model.encode(chunks, show_progress_bar=True)


def cosine_sim(a, b):
    return np.dot(a, b) / (norm(a) * norm(b))


def retrieve_evidence(claim, top_k=3):
    claim_emb = model.encode([claim])[0]

    scores = []
    for i, emb in enumerate(chunk_embeddings):
        score = cosine_sim(claim_emb, emb)
        scores.append((score, chunks[i]))

    scores.sort(reverse=True, key=lambda x: x[0])

    return scores[:top_k]


# 🔍 Test run
if __name__ == "__main__":
    claim = "The character avoids commitment due to childhood experiences."
    results = retrieve_evidence(claim)

    for score, text in results:
        print(f"\nScore: {score:.3f}")
        print(text)