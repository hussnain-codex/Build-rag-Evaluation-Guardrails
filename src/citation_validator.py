"""Validate [filename.pdf p.N] citations in a generated answer against the
chunks that were actually retrieved for that answer.
"""

import re

CITATION_PATTERN = re.compile(r"\[([^\[\]]+?\.pdf)\s+p\.\s*(\d+)\]", re.IGNORECASE)


def extract_citations(answer_text):
    """Return a list of (filename, page) tuples found in the answer."""

    citations = []
    for match in CITATION_PATTERN.finditer(answer_text):
        filename = match.group(1).strip()
        page = int(match.group(2))
        citations.append((filename, page))
    return citations


def _retrieved_sources(chunks):
    return {(chunk["source"], chunk["page"]) for chunk in chunks}


def validate_citations(answer_text, chunks):
    """Check every citation in the answer against the retrieved chunks.

    Returns a dict:
      - citations: all (file, page) pairs found in the text
      - valid_citations: ones that match a retrieved chunk
      - invalid_citations: ones that do NOT match any retrieved chunk
      - has_citations: whether the answer contains at least one citation
      - all_valid: True only if every citation found is valid
    """

    citations = extract_citations(answer_text)
    retrieved = _retrieved_sources(chunks)

    valid = [c for c in citations if c in retrieved]
    invalid = [c for c in citations if c not in retrieved]

    return {
        "citations": citations,
        "valid_citations": valid,
        "invalid_citations": invalid,
        "has_citations": len(citations) > 0,
        "all_valid": len(citations) > 0 and len(invalid) == 0,
    }


def sentence_has_factual_claim(sentence):
    """Very small heuristic: does this sentence look like it states a fact
    worth citing, as opposed to filler ("I don't know...", headers, etc.)?
    """

    stripped = sentence.strip()
    if not stripped:
        return False

    lowered = stripped.lower()
    skip_markers = ("i don't know", "i'm not confident", "answer:", "question:")
    if any(lowered.startswith(marker) for marker in skip_markers):
        return False

    # A sentence with no letters (e.g. just a citation or punctuation) isn't
    # itself a claim that needs its own citation.
    return bool(re.search(r"[a-zA-Z]{3,}", stripped))


def check_citation_coverage(answer_text):
    """Report which factual sentences are missing a citation entirely."""

    sentences = re.split(r"(?<=[.!?])\s+", answer_text.strip())
    uncited_sentences = []

    for sentence in sentences:
        if sentence_has_factual_claim(sentence) and not CITATION_PATTERN.search(sentence):
            uncited_sentences.append(sentence.strip())

    return {
        "total_sentences": len([s for s in sentences if s.strip()]),
        "uncited_sentences": uncited_sentences,
        "fully_cited": len(uncited_sentences) == 0,
    }


if __name__ == "__main__":
    sample_answer = (
        "Pakistan Green Taxonomy was approved on 15th September 2025 "
        "[17_Climate_Change.pdf p.13]. It has no relation to NDC 3.0 "
        "[17_Climate_Change.pdf p.99]."
    )
    sample_chunks = [{"source": "17_Climate_Change.pdf", "page": 13}]

    print(validate_citations(sample_answer, sample_chunks))
    print(check_citation_coverage(sample_answer))
