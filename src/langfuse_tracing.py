"""Langfuse observability: trace each RAG query (retrieval + generation)
and attach evaluation scores, so any question can be opened in the
Langfuse dashboard to see exactly what was retrieved and generated.

Fails safe: if Langfuse credentials are missing or the API is unreachable,
tracing is silently skipped rather than crashing the pipeline.
"""

import os

from dotenv import load_dotenv

from config import ENV_FILE, GEMINI_MODEL_NAME

load_dotenv(ENV_FILE)

_client_cache = None
_tracing_available = None


def is_tracing_configured():
    return bool(
        os.getenv("LANGFUSE_SECRET_KEY")
        and os.getenv("LANGFUSE_PUBLIC_KEY")
    )


def get_tracing_client():
    """Return a Langfuse client, or None if tracing isn't usable."""

    global _client_cache, _tracing_available

    if _tracing_available is False:
        return None

    if _client_cache is not None:
        return _client_cache

    if not is_tracing_configured():
        print("Langfuse not configured (missing env vars) -- tracing skipped.")
        _tracing_available = False
        return None

    try:
        from langfuse import get_client

        client = get_client()
        if not client.auth_check():
            print("Langfuse auth_check() failed -- tracing skipped.")
            _tracing_available = False
            return None
        _client_cache = client
        _tracing_available = True
        return client
    except Exception as error:
        print(f"Langfuse unavailable ({error}) -- tracing skipped.")
        _tracing_available = False
        return None


def trace_rag_query(question, generation_result, judge_scores=None):
    """Trace one full RAG turn: retrieval span + generation span, with
    optional evaluation scores attached to the trace.

    generation_result is the dict returned by generate_answer():
    {"answer": str, "chunks": list, "strategy": str}.
    judge_scores, if given, is a dict like:
    {"correctness": 0.9, "faithful": True}.

    Returns the trace URL (str) if tracing succeeded, else None.
    """

    client = get_tracing_client()
    if client is None:
        return None

    chunks = generation_result.get("chunks", [])
    answer = generation_result.get("answer", "")
    strategy = generation_result.get("strategy", "unknown")

    trace_id = client.create_trace_id()

    try:
        with client.start_as_current_observation(
            trace_context={"trace_id": trace_id},
            name="rag_query",
            as_type="span",
            input=question,
            output=answer,
            metadata={"strategy": strategy},
        ):
            with client.start_as_current_observation(
                name="retrieval",
                as_type="retriever",
                input=question,
                output=[
                    {"source": c["source"], "page": c["page"], "score": c.get("score")}
                    for c in chunks
                ],
                metadata={"strategy": strategy, "num_chunks": len(chunks)},
            ):
                pass

            with client.start_as_current_observation(
                name="generation",
                as_type="generation",
                model=GEMINI_MODEL_NAME,
                input=question,
                output=answer,
            ):
                pass

            if judge_scores:
                if "correctness" in judge_scores:
                    client.score_current_trace(
                        name="answer_correctness",
                        value=float(judge_scores["correctness"]),
                        data_type="NUMERIC",
                    )
                if "faithful" in judge_scores:
                    client.score_current_trace(
                        name="faithfulness",
                        value=bool(judge_scores["faithful"]),
                        data_type="BOOLEAN",
                    )

        client.flush()
        return client.get_trace_url(trace_id=trace_id)
    except Exception as error:
        print(f"Langfuse tracing failed for this query ({error}).")
        return None


if __name__ == "__main__":
    print("Tracing configured:", is_tracing_configured())
    client = get_tracing_client()
    print("Client ready:", client is not None)

    if client is not None:
        fake_result = {
            "answer": "Pakistan Green Taxonomy was approved on 15th September 2025 [17_Climate_Change.pdf p.13].",
            "chunks": [{"source": "17_Climate_Change.pdf", "page": 13, "score": 0.81}],
            "strategy": "hybrid_reranked",
        }
        url = trace_rag_query(
            "What is Pakistan Green Taxonomy?",
            fake_result,
            judge_scores={"correctness": 0.95, "faithful": True},
        )
        print("Trace URL:", url)
