"""LLM-as-a-judge: use Gemini, independently from answer generation, to
score answer correctness and faithfulness with structured (Pydantic
validated) JSON output.
"""

import json
import os
import re

from dotenv import load_dotenv
from pydantic import BaseModel, Field, ValidationError
from langchain_google_genai import ChatGoogleGenerativeAI

from config import ENV_FILE, GEMINI_MODEL_NAME

load_dotenv(ENV_FILE)

_judge_llm_cache = None


def get_judge_llm():
    global _judge_llm_cache

    if _judge_llm_cache is not None:
        return _judge_llm_cache

    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key:
        raise ValueError("GOOGLE_API_KEY is missing from .env")

    _judge_llm_cache = ChatGoogleGenerativeAI(
        model=GEMINI_MODEL_NAME, google_api_key=api_key, temperature=0,
        timeout=30, max_retries=2,
    )
    return _judge_llm_cache


class CorrectnessVerdict(BaseModel):
    score: float = Field(ge=0, le=1)
    reason: str
    missing_facts: list[str] = Field(default_factory=list)


class FaithfulnessVerdict(BaseModel):
    faithful: bool
    unsupported_claims: list[str] = Field(default_factory=list)
    reason: str


def _extract_text(content):
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "\n".join(
            item["text"] for item in content
            if isinstance(item, dict) and item.get("type") == "text"
        )
    return str(content)


def _extract_json_block(text):
    """Pull the first {...} JSON object out of a model response, tolerating
    markdown code fences around it."""

    fenced = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL)
    if fenced:
        return fenced.group(1)

    brace_match = re.search(r"\{.*\}", text, re.DOTALL)
    if brace_match:
        return brace_match.group(0)

    return text


def _call_judge_json(prompt, retries=1):
    """Call Gemini and parse a JSON object out of the response.
    Returns (parsed_dict, raw_text) or (None, raw_text) on failure.
    """

    llm = get_judge_llm()

    last_error = None
    for _ in range(retries + 1):
        try:
            response = llm.invoke(prompt)
            raw_text = _extract_text(response.content)
        except Exception as error:
            last_error = error
            continue

        json_block = _extract_json_block(raw_text)
        try:
            return json.loads(json_block), raw_text
        except json.JSONDecodeError as error:
            last_error = error
            continue

    return None, f"JUDGE_CALL_FAILED: {last_error}"


def judge_answer_correctness(question, generated_answer, reference_answer):
    """Score 0-1 how well the generated answer matches the reference
    (ground truth) answer, with a one-line reason and any missing facts.
    """

    prompt = f"""You are grading an AI assistant's answer against a
reference (ground truth) answer. Be strict about factual accuracy but
tolerant of different wording.

Question: {question}

Reference answer (ground truth): {reference_answer}

Generated answer: {generated_answer}

Respond with ONLY a JSON object, no other text, in exactly this shape:
{{
  "score": <float between 0 and 1>,
  "reason": "<one-line reason for the score>",
  "missing_facts": ["<fact from the reference answer missing from the generated answer>", ...]
}}
"""

    parsed, raw_text = _call_judge_json(prompt)

    if parsed is None:
        return {
            "score": 0.0,
            "reason": f"Judge call/parse failed: {raw_text}",
            "missing_facts": [],
            "judge_error": True,
        }

    try:
        verdict = CorrectnessVerdict(**parsed)
        result = verdict.model_dump()
        result["judge_error"] = False
        return result
    except ValidationError as error:
        return {
            "score": 0.0,
            "reason": f"Judge returned malformed JSON: {error}",
            "missing_facts": [],
            "judge_error": True,
        }


def judge_faithfulness(question, generated_answer, retrieved_context):
    """Check whether every claim in the generated answer is supported by
    the retrieved context (detects hallucination / unsupported claims).
    """

    prompt = f"""You are checking whether an AI assistant's answer is
faithful to the context it was given -- i.e. every factual claim in the
answer must be directly supported by the retrieved context below. Outside
knowledge, even if true, counts as unsupported here.

Question: {question}

Retrieved context:
\"\"\"{retrieved_context}\"\"\"

Generated answer: {generated_answer}

Respond with ONLY a JSON object, no other text, in exactly this shape:
{{
  "faithful": <true or false>,
  "unsupported_claims": ["<claim in the answer not supported by the context>", ...],
  "reason": "<one-line explanation>"
}}
"""

    parsed, raw_text = _call_judge_json(prompt)

    if parsed is None:
        return {
            "faithful": False,
            "unsupported_claims": [],
            "reason": f"Judge call/parse failed: {raw_text}",
            "judge_error": True,
        }

    try:
        verdict = FaithfulnessVerdict(**parsed)
        result = verdict.model_dump()
        result["judge_error"] = False
        return result
    except ValidationError as error:
        return {
            "faithful": False,
            "unsupported_claims": [],
            "reason": f"Judge returned malformed JSON: {error}",
            "judge_error": True,
        }


if __name__ == "__main__":
    q = "What is Pakistan Green Taxonomy?"
    ref = "Pakistan Green Taxonomy was approved and notified on 15th September 2025."
    gen_good = "Pakistan Green Taxonomy was approved on 15th September 2025 [17_Climate_Change.pdf p.13]."
    gen_bad = "Pakistan Green Taxonomy was approved in 1999 by the United Nations."

    print("Correctness (good):", judge_answer_correctness(q, gen_good, ref))
    print("Correctness (bad):", judge_answer_correctness(q, gen_bad, ref))

    context = "Pakistan Green Taxonomy has been approved and notified on 15th September 2025."
    print("Faithfulness (good):", judge_faithfulness(q, gen_good, context))
    print("Faithfulness (bad):", judge_faithfulness(q, gen_bad, context))
