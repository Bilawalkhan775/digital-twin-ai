
import re


LIMITATION_PATTERNS = [
    r"\bcannot confirm\b",
    r"\bcan't confirm\b",
    r"\bnot confirmed\b",
    r"\bdoes not confirm\b",
    r"\bdo not confirm\b",
    r"\bdoesn't confirm\b",
    r"\bnot specified\b",
    r"\bdo not specify\b",
    r"\bdoes not specify\b",
    r"\bnot documented\b",
    r"\bno documented evidence\b",
    r"\bno evidence\b",
    r"\bnot established\b",
    r"\bcannot determine\b",
    r"\bunclear whether\b"
]

NEGATIVE_PATTERNS = [
    r"\bdid not\b",
    r"\bdoes not\b",
    r"\bdo not\b",
    r"\bnever\b",
    r"\bwas not\b",
    r"\bwere not\b"
]


def classify_claim_type(text):
    """
    Classify the wording of a claim.
    This is not a factual accuracy verdict.
    """

    for pattern in LIMITATION_PATTERNS:
        if re.search(pattern, text, re.IGNORECASE):
            return "EVIDENCE_LIMITATION"

    for pattern in NEGATIVE_PATTERNS:
        if re.search(pattern, text, re.IGNORECASE):
            return "NEGATIVE"

    return "AFFIRMATIVE"


def extract_claims(answer):
    """
    Extract candidate factual spans without
    rewriting the original text.

    Uses line and sentence boundaries.

    Limitation: a compound sentence may still
    contain more than one atomic factual claim.
    """

    if not isinstance(answer, str) or not answer.strip():
        raise ValueError("Answer cannot be empty.")

    claims = []

    # Split at existing line boundaries.
    lines = answer.splitlines()

    for line in lines:

        line = line.strip()

        if not line:
            continue

        # Ignore common Markdown headings.
        if line.startswith("#"):
            continue

        # Ignore Markdown table separators.
        if re.fullmatch(r"[\|\s:\-]+", line):
            continue

        # Ignore standalone source labels.
        if re.fullmatch(
            r"(?:Source|Sources)\s*:?\s*\d*",
            line,
            re.IGNORECASE
        ):
            continue

        # Split at sentence boundaries while
        # retaining the original characters.
        sentences = re.split(
            r"(?<=[.!?])\s+(?=[A-Z])",
            line
        )

        for sentence in sentences:

            span = sentence.strip()

            if not span:
                continue

            # Remove leading list markers by
            # selecting a substring of the line.
            span = re.sub(
                r"^(?:[-*+]|\d+[.)])\s+",
                "",
                span
            )

            if not span:
                continue

            # Validate that the text really
            # appears in the original answer.
            if span not in answer:
                raise ValueError(
                    "Claim extraction changed the answer."
                )

            claims.append({
                "id": len(claims) + 1,
                "claim": span,
                "answer_span": span,
                "claim_type": classify_claim_type(span)
            })

    if not claims:
        raise ValueError(
            "No candidate claims could be extracted."
        )

    return claims
