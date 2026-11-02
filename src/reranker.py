"""Cross-encoder reranking: retrieve a larger candidate pool with the fast
retrievers, then rerank by actual query-document relevance.

Loaded lazily and cached, since the cross-encoder model is only needed when
reranking is actually requested (keeps memory usage down on an 8GB machine).
"""

from sentence_transformers import CrossEncoder

from config import RERANKER_MODEL_NAME, DEFAULT_TOP_K, RERANK_CANDIDATE_POOL
from retrieval import retrieve_chunks

_reranker_cache = None
_reranker_failed = False


def get_reranker():
    """Load the cross-encoder once and reuse it. Returns None if loading fails."""

    global _reranker_cache, _reranker_failed

    if _reranker_cache is not None:
        return _reranker_cache

    if _reranker_failed:
        return None

    try:
        _reranker_cache = CrossEncoder(RERANKER_MODEL_NAME)
    except Exception as error:
        print(f"Warning: could not load reranker model ({error}). "
              f"Falling back to un-reranked results.")
        _reranker_failed = True
        return None

    return _reranker_cache


def retrieve_chunks_reranked(
    question,
    top_k=DEFAULT_TOP_K,
    candidate_pool=RERANK_CANDIDATE_POOL,
    retrieve_fn=retrieve_chunks,
):
    """Retrieve a larger candidate set, then rerank with a cross-encoder.

    retrieve_fn lets this be reused on top of semantic, BM25 or hybrid
    retrieval -- whichever supplies the initial candidate pool.
    """

    candidates = retrieve_fn(question, top_k=candidate_pool)

    if not candidates:
        return []

    reranker = get_reranker()

    if reranker is None:
        # Model failed to load: return the original top_k un-reranked
        # rather than crashing the whole pipeline.
        return candidates[:top_k]

    pairs = [[question, chunk["text"]] for chunk in candidates]
    cross_scores = reranker.predict(pairs)

    reranked = sorted(
        zip(candidates, cross_scores), key=lambda pair: pair[1], reverse=True
    )

    results = []
    for chunk, cross_score in reranked[:top_k]:
        new_chunk = dict(chunk)
        new_chunk["rerank_score"] = float(cross_score)
        results.append(new_chunk)

    return results


if __name__ == "__main__":
    question = input("Ask a question (reranked search): ").strip()

    if not question:
        print("Please enter a question.")
    else:
        results = retrieve_chunks_reranked(question)

        if not results:
            print("No matching chunks found.")
        else:
            for rank, result in enumerate(results, start=1):
                print(f"\n--- Result {rank} ---")
                print("Source:", result["source"])
                print("Page:", result["page"])
                print("Rerank score:", round(result.get("rerank_score", 0), 4))
                print("Text:", result["text"][:300])
