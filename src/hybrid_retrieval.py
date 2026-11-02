"""Hybrid retrieval: fuse FAISS semantic search with BM25 keyword search
using Reciprocal Rank Fusion (RRF)."""

from bm25_retrieval import retrieve_chunks_bm25
from retrieval import retrieve_chunks
from config import DEFAULT_TOP_K

RRF_K = 60  # standard RRF smoothing constant


def _chunk_key(chunk):
    return (chunk["source"], chunk["page"], chunk["chunk_index"])


def retrieve_chunks_hybrid(
    question,
    top_k=DEFAULT_TOP_K,
    candidate_pool=20,
    semantic_weight=1.0,
    bm25_weight=1.0,
):
    """Combine semantic and BM25 rankings with weighted Reciprocal Rank Fusion.

    Each retriever's results are scored by 1 / (RRF_K + rank). Scores from
    the two lists are summed per chunk, duplicates are merged, and the top_k
    chunks by fused score are returned.
    """

    semantic_results = retrieve_chunks(question, top_k=candidate_pool)
    bm25_results = retrieve_chunks_bm25(question, top_k=candidate_pool)

    fused = {}

    for rank, chunk in enumerate(semantic_results, start=1):
        key = _chunk_key(chunk)
        rrf_score = semantic_weight * (1 / (RRF_K + rank))
        if key not in fused:
            fused[key] = {"chunk": chunk, "score": 0.0}
        fused[key]["score"] += rrf_score

    for rank, chunk in enumerate(bm25_results, start=1):
        key = _chunk_key(chunk)
        rrf_score = bm25_weight * (1 / (RRF_K + rank))
        if key not in fused:
            fused[key] = {"chunk": chunk, "score": 0.0}
        fused[key]["score"] += rrf_score

    ranked = sorted(fused.values(), key=lambda item: item["score"], reverse=True)

    results = []
    for item in ranked[:top_k]:
        chunk = dict(item["chunk"])
        chunk["score"] = item["score"]
        results.append(chunk)

    return results


if __name__ == "__main__":
    question = input("Ask a question (hybrid search): ").strip()

    if not question:
        print("Please enter a question.")
    else:
        results = retrieve_chunks_hybrid(question)

        if not results:
            print("No matching chunks found.")
        else:
            for rank, result in enumerate(results, start=1):
                print(f"\n--- Result {rank} ---")
                print("Source:", result["source"])
                print("Page:", result["page"])
                print("Fused RRF score:", round(result["score"], 5))
                print("Text:", result["text"][:300])
