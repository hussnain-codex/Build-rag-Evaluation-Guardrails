
import json

import faiss
import numpy as np

from config import FAISS_INDEX_PATH, CHUNKS_METADATA_PATH
from embeddings import create_embeddings

INDEX_PATH = FAISS_INDEX_PATH
METADATA_PATH = CHUNKS_METADATA_PATH

_vector_store_cache = None


def load_vector_store():
    """Load the saved FAISS index and chunk metadata (cached after first call)."""

    global _vector_store_cache

    if _vector_store_cache is not None:
        return _vector_store_cache

    if not INDEX_PATH.exists() or not METADATA_PATH.exists():
        raise FileNotFoundError(
            "Vector store not found. Run: python src/vector_store.py"
        )

    index = faiss.read_index(str(INDEX_PATH))

    with open(METADATA_PATH, "r", encoding="utf-8") as file:
        chunks = json.load(file)

    if index.ntotal != len(chunks):
        raise ValueError("Index and chunk metadata do not match.")

    _vector_store_cache = (index, chunks)
    return _vector_store_cache


def retrieve_chunks(question, top_k=4):
    """Find the most relevant chunks for a question."""

    index, chunks = load_vector_store()

    # Convert the question into an embedding
    query_vector = create_embeddings([question])
    query_vector = np.asarray(query_vector, dtype="float32")

    # Avoid requesting more results than the index contains
    k = min(top_k, index.ntotal)

    if k == 0:
        return []

    scores, indices = index.search(query_vector, k)

    results = []

    for score, chunk_index in zip(scores[0], indices[0]):
        if chunk_index == -1:
            continue

        chunk = chunks[int(chunk_index)]

        results.append({
            "text": chunk["text"],
            "source": chunk["metadata"]["source"],
            "page": chunk["metadata"]["page"],
            "chunk_index": chunk["metadata"]["chunk_index"],
            "score": float(score)
        })

    return results


if __name__ == "__main__":
    question = input("Ask a question about your PDFs: ").strip()

    if not question:
        print("Please enter a question.")
    else:
        results = retrieve_chunks(question)

        if not results:
            print("No chunks are available.")
        else:
            for rank, result in enumerate(results, start=1):
                print(f"\n--- Result {rank} ---")
                print("Source:", result["source"])
                print("Page:", result["page"])
                print("Similarity:", round(result["score"], 4))
                print("Text:", result["text"][:500])
