"""Output guardrails: check a generated answer before it is shown to the
user.

Checks:
  1. Citation validity -- every [file.pdf p.N] citation must point to a
     chunk that was actually retrieved for this answer.
  2. PII masking -- strip anything that looks like an email address or
     phone number out of the final answer (basic regex masking).
  3. Faithfulness gate -- if the faithfulness judge says the answer is not
     well supported by the retrieved context, replace the answer with an
     "I'm not confident" fallback instead of showing a possibly-hallucinated
     answer.
"""

import re

from citation_validator import validate_citations, check_citation_coverage

NOT_CONFIDENT_MESSAGE = (
    "I'm not confident in this answer based on the retrieved context. "
    "Please rephrase the question or check the source PDF directly."
)

EMAIL_PATTERN = re.compile(r"[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}")
PHONE_PATTERN = re.compile(
    r"(\+?\d{1,3}[\s.-]?)?(\(?\d{2,4}\)?[\s.-]?)\d{3,4}[\s.-]?\d{3,4}\b"
)


def mask_pii(text):
    """Replace anything that looks like an email or phone number."""

    masked = EMAIL_PATTERN.sub("[REDACTED EMAIL]", text)
    masked = PHONE_PATTERN.sub("[REDACTED PHONE]", masked)
    return masked


def apply_output_guardrails(answer_text, chunks, faithfulness_result=None):
    """Run every output check and return the (possibly modified) answer
    along with a report of what was checked.

    faithfulness_result, if provided, is the dict returned by
    llm_judge.judge_faithfulness(): {"faithful": bool, ...}. If it says the
    answer is not faithful, the answer is replaced with a safe fallback.
    """

    report = {
        "original_answer": answer_text,
        "modified": False,
        "reasons": [],
    }

    citation_report = validate_citations(answer_text, chunks)
    coverage_report = check_citation_coverage(answer_text)
    report["citation_report"] = citation_report
    report["coverage_report"] = coverage_report

    final_answer = answer_text

    if citation_report["invalid_citations"]:
        report["modified"] = True
        report["reasons"].append("invalid_citations")

    if faithfulness_result is not None and not faithfulness_result.get("faithful", True):
        final_answer = NOT_CONFIDENT_MESSAGE
        report["modified"] = True
        report["reasons"].append("failed_faithfulness_check")

    masked_answer = mask_pii(final_answer)
    if masked_answer != final_answer:
        report["modified"] = True
        report["reasons"].append("pii_masked")
    final_answer = masked_answer

    report["final_answer"] = final_answer
    return final_answer, report


if __name__ == "__main__":
    sample_answer = (
        "Pakistan Green Taxonomy was approved on 15th September 2025 "
        "[17_Climate_Change.pdf p.13]. Contact support@example.com or "
        "call 0300-1234567 for more info [17_Climate_Change.pdf p.99]."
    )
    sample_chunks = [{"source": "17_Climate_Change.pdf", "page": 13}]

    final, report = apply_output_guardrails(sample_answer, sample_chunks)
    print("Final answer:\n", final)
    print("\nReport:", report)
