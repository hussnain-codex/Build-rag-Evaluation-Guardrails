"""Step 4 runner: re-run the cheap retrieval-hit-rate eval after each
retrieval change (semantic -> hybrid -> reranked) and save a comparison
table, so improvements are proven with numbers rather than assumed.
"""

import json

from config import RESULTS_FOLDER, DEFAULT_TOP_K
from evaluate_retrieval import evaluate_hit_rate
from retrieval import retrieve_chunks
from bm25_retrieval import retrieve_chunks_bm25
from hybrid_retrieval import retrieve_chunks_hybrid
from reranker import retrieve_chunks_reranked


def reranked_on_semantic(question, top_k=DEFAULT_TOP_K):
    return retrieve_chunks_reranked(question, top_k=top_k, retrieve_fn=retrieve_chunks)


def reranked_on_hybrid(question, top_k=DEFAULT_TOP_K):
    return retrieve_chunks_reranked(question, top_k=top_k, retrieve_fn=retrieve_chunks_hybrid)


STRATEGIES = [
    ("semantic", retrieve_chunks),
    ("bm25", retrieve_chunks_bm25),
    ("hybrid", retrieve_chunks_hybrid),
    ("hybrid_reranked", reranked_on_hybrid),
]


def run_comparison(top_k=DEFAULT_TOP_K):
    summaries = []

    for label, retrieve_fn in STRATEGIES:
        print(f"\n### Running retrieval strategy: {label} ###")
        summary = evaluate_hit_rate(
            top_k=top_k,
            save_results=True,
            verbose=False,
            retrieve_fn=retrieve_fn,
            label=label,
        )
        summaries.append(summary)
        print(
            f"{label:>16}: Hit Rate@{top_k} = {summary['hit_rate']:.2%} | "
            f"MRR = {summary['mrr']:.4f} | "
            f"Precision@{top_k} = {summary['precision_at_k']:.2%}"
        )

    RESULTS_FOLDER.mkdir(parents=True, exist_ok=True)
    out_path = RESULTS_FOLDER / "retrieval_comparison.json"
    with open(out_path, "w", encoding="utf-8") as file:
        json.dump(summaries, file, indent=2)

    print("\nSaved comparison table to:", out_path)
    return summaries


if __name__ == "__main__":
    run_comparison()
