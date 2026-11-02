"""Step 2b-2d + Step 3: full answer-quality evaluation.

For every question in eval/eval.csv:
  - generate an answer with the RAG pipeline
  - score answer correctness against the reference answer (Gemini judge)
  - score faithfulness of the answer against the retrieved context (Gemini judge)
  - for unanswerable questions, check refusal accuracy (did it say "I don't know"?)

Saves a full per-question CSV/JSON to results/ and logs each question to
Langfuse (retrieval + generation + scores) so it can be opened in the
dashboard.
"""

import json

import pandas as pd

from config import EVAL_DATASET_CSV, RESULTS_FOLDER, DEFAULT_TOP_K
from generate_answer import generate_answer, build_context, IDK_MESSAGE
from llm_judge import judge_answer_correctness, judge_faithfulness
from output_guardrails import apply_output_guardrails
from langfuse_tracing import trace_rag_query

REFUSAL_MARKERS = ("i don't know", "i'm not confident", "cannot answer", "no relevant")


def looks_like_refusal(answer_text):
    lowered = answer_text.lower()
    return any(marker in lowered for marker in REFUSAL_MARKERS)


def evaluate_answers(strategy="hybrid_reranked", top_k=DEFAULT_TOP_K, limit=None, trace=True):
    dataset = pd.read_csv(EVAL_DATASET_CSV, keep_default_na=False)
    if limit:
        dataset = dataset.head(limit)

    RESULTS_FOLDER.mkdir(parents=True, exist_ok=True)
    csv_path = RESULTS_FOLDER / f"answer_eval_{strategy}.csv"
    csv_path.unlink(missing_ok=True)  # start fresh; we append per-row below

    rows = []
    wrote_header = False

    for _, item in dataset.iterrows():
        question = item["question"]
        reference_answer = item["correct_answer"]
        is_answerable = str(item["answerable"]).lower() == "true"

        print(f"\n--- {question[:70]} ---")

        result = generate_answer(question, strategy=strategy, top_k=top_k)
        answer_text = result["answer"]
        chunks = result["chunks"]
        # Build the judge's context the same way generate_answer() built the
        # prompt's context (with Source/Page labels), so the faithfulness
        # judge can actually see the citations an answer refers to.
        context = build_context(chunks)

        row = {
            "question": question,
            "answerable": is_answerable,
            "generated_answer": answer_text,
            "strategy": strategy,
        }

        if is_answerable:
            correctness = judge_answer_correctness(question, answer_text, reference_answer)
            faithfulness = judge_faithfulness(question, answer_text, context)

            final_answer, guardrail_report = apply_output_guardrails(
                answer_text, chunks, faithfulness_result=faithfulness
            )

            row.update({
                "correctness_score": correctness["score"],
                "correctness_reason": correctness["reason"],
                "faithful": faithfulness["faithful"],
                "faithfulness_reason": faithfulness["reason"],
                "final_answer_after_guardrails": final_answer,
                "guardrail_modified": guardrail_report["modified"],
                "guardrail_reasons": ";".join(guardrail_report["reasons"]),
            })

            judge_scores = {
                "correctness": correctness["score"],
                "faithful": faithfulness["faithful"],
            }
            print(f"  correctness={correctness['score']:.2f}  faithful={faithfulness['faithful']}")
        else:
            refused = looks_like_refusal(answer_text)
            row["refusal_correct"] = refused
            judge_scores = {"correctness": 1.0 if refused else 0.0}
            print(f"  (unanswerable) refused_correctly={refused}")

        if trace:
            trace_url = trace_rag_query(question, result, judge_scores=judge_scores)
            row["langfuse_trace_url"] = trace_url

        rows.append(row)

        # Write incrementally so a crash/hang on a later question doesn't
        # lose the questions already completed.
        pd.DataFrame([row]).to_csv(
            csv_path, mode="a", index=False, header=not wrote_header
        )
        wrote_header = True

    results_df = pd.DataFrame(rows)

    answerable_rows = results_df[results_df["answerable"] == True]  # noqa: E712
    unanswerable_rows = results_df[results_df["answerable"] == False]  # noqa: E712

    summary = {
        "strategy": strategy,
        "total_questions": len(results_df),
        "answerable_questions": len(answerable_rows),
        "unanswerable_questions": len(unanswerable_rows),
        "mean_correctness": float(answerable_rows["correctness_score"].mean()) if len(answerable_rows) else None,
        "faithfulness_rate": float(answerable_rows["faithful"].mean()) if len(answerable_rows) else None,
        "refusal_accuracy": float(unanswerable_rows["refusal_correct"].mean()) if len(unanswerable_rows) else None,
    }

    summary_path = RESULTS_FOLDER / f"answer_eval_{strategy}_summary.json"
    with open(summary_path, "w", encoding="utf-8") as file:
        json.dump(summary, file, indent=2)

    print("\n========== ANSWER EVAL SUMMARY ==========")
    for key, value in summary.items():
        print(f"{key}: {value}")
    print("Saved per-question results to:", csv_path)
    print("Saved summary to:", summary_path)

    return summary, results_df


if __name__ == "__main__":
    evaluate_answers()
