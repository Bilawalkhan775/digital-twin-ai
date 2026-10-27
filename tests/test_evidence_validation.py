
from rag.faithfulness import validate_evaluation


# ==========================================
# TEST REFERENCE
# ==========================================

SOURCES = [
    {
        "section": "Project — Alfalah Builders",
        "text": "Developed REST APIs using PHP.",
        "person": "Muhammad Bilawal Khan"
    }
]

ANSWER = "Bilawal developed REST APIs using PHP."


# ==========================================
# TEST 1: MISSING EVIDENCE QUOTE
# ==========================================

def test_missing_quote():

    evaluation = {
        "claims": [
            {
                "claim": ANSWER,
                "answer_span": ANSWER,
                "claim_type": "AFFIRMATIVE",
                "verdict": "SUPPORTED",
                "source_ids": [1],
                "evidence_quote": "",
                "reason": "Claim is supported."
            }
        ]
    }

    result = validate_evaluation(
        evaluation=evaluation,
        retrieved_results=SOURCES,
        answer=ANSWER
    )

    assert (
        result["claims"][0]["verdict"]
        == "UNCLEAR"
    )

    print("PASS: Missing evidence quotation rejected.")


# ==========================================
# TEST 2: VALID EVIDENCE QUOTE
# ==========================================

def test_valid_quote():

    evaluation = {
        "claims": [
            {
                "claim": ANSWER,
                "answer_span": ANSWER,
                "claim_type": "AFFIRMATIVE",
                "verdict": "SUPPORTED",
                "source_ids": [1],
                "evidence_quote": (
                    "Developed REST APIs using PHP."
                ),
                "reason": "Directly supported."
            }
        ]
    }

    result = validate_evaluation(
        evaluation=evaluation,
        retrieved_results=SOURCES,
        answer=ANSWER
    )

    assert (
        result["claims"][0]["verdict"]
        == "SUPPORTED"
    )

    print("PASS: Valid evidence quotation accepted.")


# ==========================================
# TEST 3: NEGATION MUST BE PRESERVED
# ==========================================

def test_negation_preservation():

    answer = (
        "I cannot confirm that the backend "
        "generates automated alerts."
    )

    evaluation = {
        "claims": [
            {
                "claim": (
                    "The backend generates "
                    "automated alerts."
                ),
                "answer_span": (
                    "The backend generates "
                    "automated alerts."
                ),
                "claim_type": "AFFIRMATIVE",
                "verdict": "SUPPORTED",
                "source_ids": [1],
                "evidence_quote": (
                    "Developed REST APIs using PHP."
                ),
                "reason": "Incorrect extraction."
            }
        ]
    }

    try:
        validate_evaluation(
            evaluation=evaluation,
            retrieved_results=SOURCES,
            answer=answer
        )

    except ValueError:
        print(
            "PASS: Negation-changing claim "
            "extraction rejected."
        )
        return

    raise AssertionError(
        "Validator incorrectly accepted "
        "a negation-changing extraction."
    )


# ==========================================
# RUN TESTS
# ==========================================

if __name__ == "__main__":

    test_missing_quote()
    test_valid_quote()
    test_negation_preservation()

    print("\nAll evidence validation tests passed.")
