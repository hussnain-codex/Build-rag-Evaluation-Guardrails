
import json

import pandas as pd

from config import EVAL_DATASET_CSV, RESULTS_FOLDER, DEFAULT_TOP_K
from retrieval import retrieve_chunks

TOP_K = DEFAULT_TOP_K


def load_eval_dataset():
    """Load eval/eval.csv. Unanswerable rows have blank file/page."""

    df = pd.read_csv(EVAL_DATASET_CSV, keep_default_na=False)
    df["page"] = pd.to_numeric(df["page"], errors="coerce")
    return df


def evaluate_hit_rate(
    top_k=TOP_K,
    save_results=True,
    verbose=True,
    retrieve_fn=retrieve_chunks,
    label="semantic",
):
    dataset = load_eval_dataset()

    # Only answerable questions have a real page to hit. Unanswerable
    # questions are reported separately (see Step 2d / evaluate_answers.py).
    answerable = dataset[dataset["answerable"].astype(str).str.lower() == "true"]

    total_questions = len(answerable)
    successful_hits = 0
    reciprocal_ranks = []
    precision_scores = []
    per_question_rows = []

    if verbose:
        print("\n========== RAG RETRIEVAL EVALUATION ==========\n")

    for _, item in answerable.iterrows():
        question = item["question"]
        expected_source = item["file"]
        expected_page = int(item["page"])

        results = retrieve_fn(question, top_k=top_k)
        num_results = len(results)

        rank = None
        for position, result in enumerate(results, start=1):
            if (
                result["source"] == expected_source
                and result["page"] == expected_page
            ):
                rank = position
                break

        hit = rank is not None

        if hit:
            successful_hits += 1
            reciprocal_ranks.append(1 / rank)
        else:
            reciprocal_ranks.append(0)

        relevant_count = sum(
            1 for result in results
            if result["source"] == expected_source
            and result["page"] == expected_page
        )
        # Precision denominator is the number of chunks actually returned,
        # not the requested top_k (handles empty/short result sets safely).
        precision = relevant_count / num_results if num_results else 0
        precision_scores.append(precision)

        per_question_rows.append({
            "question": question,
            "expected_source": expected_source,
            "expected_page": expected_page,
            "rank": rank,
            "hit": hit,
            "precision": precision,
            "num_results": num_results,
        })

        if verbose:
            print("Question:", question)
            print("Expected:", expected_source, "| Page:", expected_page)
            print("Retrieved:")
            for result in results:
                print(
                    f"  - {result['source']} | "
                    f"Page {result['page']} | "
                    f"Score: {result['score']:.4f}"
                )
            print("Result:", "HIT" if hit else "MISS")
            print("-" * 50)

    hit_rate = successful_hits / total_questions if total_questions else 0
    mrr = sum(reciprocal_ranks) / total_questions if total_questions else 0
    precision_at_k = sum(precision_scores) / total_questions if total_questions else 0

    summary = {
        "strategy": label,
        "top_k": top_k,
        "total_questions": total_questions,
        "successful_hits": successful_hits,
        "misses": total_questions - successful_hits,
        "hit_rate": hit_rate,
        "mrr": mrr,
        "precision_at_k": precision_at_k,
    }

    if verbose:
        print("\n========== FINAL RESULTS ==========")
        print("Total Questions:", total_questions)
        print("Successful Hits:", successful_hits)
        print("Misses:", total_questions - successful_hits)
        print(f"Hit Rate@{top_k}: {hit_rate:.2%}")
        print(f"MRR: {mrr:.4f}")
        print(f"Precision@{top_k}: {precision_at_k:.2%}")

    if save_results:
        RESULTS_FOLDER.mkdir(parents=True, exist_ok=True)
        out_path = RESULTS_FOLDER / f"retrieval_{label}_top{top_k}.json"
        with open(out_path, "w", encoding="utf-8") as file:
            json.dump({"summary": summary, "per_question": per_question_rows}, file, indent=2)
        if verbose:
            print("Saved results to:", out_path)

    return summary


if __name__ == "__main__":
    evaluate_hit_rate()
