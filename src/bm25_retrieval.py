"""Keyword retrieval over the same chunks used by FAISS, using BM25."""

import json
import re

from rank_bm25 import BM25Okapi

from config import CHUNKS_METADATA_PATH, DEFAULT_TOP_K

_bm25_cache = None
_chunks_cache = None


def _tokenize(text):
    return re.findall(r"[a-z0-9]+", text.lower())


def load_bm25_index():
    """Build (and cache) a BM25 index over vector_store/chunks.json."""

    global _bm25_cache, _chunks_cache

    if _bm25_cache is not None:
        return _bm25_cache, _chunks_cache

    if not CHUNKS_METADATA_PATH.exists():
        raise FileNotFoundError(
            "Chunk metadata not found. Run: python src/vector_store.py"
        )

    with open(CHUNKS_METADATA_PATH, "r", encoding="utf-8") as file:
        chunks = json.load(file)

    tokenized_corpus = [_tokenize(chunk["text"]) for chunk in chunks]
    bm25 = BM25Okapi(tokenized_corpus)

    _bm25_cache, _chunks_cache = bm25, chunks
    return _bm25_cache, _chunks_cache


def retrieve_chunks_bm25(question, top_k=DEFAULT_TOP_K):
    """Find the most relevant chunks for a question using BM25 keyword scoring."""

    bm25, chunks = load_bm25_index()

    if not chunks:
        return []

    query_tokens = _tokenize(question)
    scores = bm25.get_scores(query_tokens)

    ranked_indices = sorted(
        range(len(scores)), key=lambda i: scores[i], reverse=True
    )[:top_k]

    results = []
    for chunk_index in ranked_indices:
        if scores[chunk_index] <= 0:
            continue
        chunk = chunks[chunk_index]
        results.append({
            "text": chunk["text"],
            "source": chunk["metadata"]["source"],
            "page": chunk["metadata"]["page"],
            "chunk_index": chunk["metadata"]["chunk_index"],
            "score": float(scores[chunk_index]),
            "corpus_index": chunk_index,
        })

    return results


if __name__ == "__main__":
    question = input("Ask a question (BM25 keyword search): ").strip()

    if not question:
        print("Please enter a question.")
    else:
        results = retrieve_chunks_bm25(question)

        if not results:
            print("No matching chunks found.")
        else:
            for rank, result in enumerate(results, start=1):
                print(f"\n--- Result {rank} ---")
                print("Source:", result["source"])
                print("Page:", result["page"])
                print("BM25 score:", round(result["score"], 4))
                print("Text:", result["text"][:300])
