
import chromadb

from rag.vector_store import DB_PATH
from rag.faithfulness import (
    validate_evaluation,
    evaluate_faithfulness
)


def load_evidence():

    client = chromadb.PersistentClient(
        path=str(DB_PATH)
    )

    collection = client.get_collection(
        name="digital_twin_knowledge"
    )

    stored = collection.get(
        include=["documents", "metadatas"]
    )

    wanted = {
        "Project — Alfalah Builders Web Portal",
        "Project — Eye Smart Helmet / Lamp Monitoring System",
        "Project — MV Sol Inc Dashboard Application"
    }

    sources = []

    for document, metadata in zip(
        stored["documents"],
        stored["metadatas"]
    ):
        if metadata["section"] in wanted:
            sources.append({
                "section": metadata["section"],
                "person": metadata.get("person", ""),
                "text": document
            })

    if {
        source["section"] for source in sources
    } != wanted:
        raise ValueError(
            "Some required project chunks are missing."
        )

    return sources


def test_span_validation(sources):

    answer = (
        "I cannot confirm that the backend "
        "generates automated alerts."
    )

    # Deliberately incorrect extraction:
    # negation has been removed.
    wrong = {
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
                "evidence_quote": "",
                "reason": "Incorrect extraction"
            }
        ]
    }

    try:
        validate_evaluation(
            wrong,
            sources,
            answer=answer
        )
    except ValueError:
        print(
            "PASS: Rewritten affirmative "
            "claim rejected."
        )
    else:
        raise AssertionError(
            "Negation-changing extraction was accepted."
        )


TEST_CASES = [
    {
        "name": "Documented PHP experience",
        "answer": (
            "Bilawal developed REST APIs using PHP "
            "for the Alfalah Builders Web Portal."
        ),
        "expected": "SUPPORTED"
    },
    {
        "name": "Documented Eye Smart Helmet APIs",
        "answer": (
            "Bilawal developed backend APIs for "
            "the Eye Smart Helmet Monitoring System."
        ),
        "expected": "SUPPORTED"
    },
    {
        "name": "Unsupported backend alert claim",
        "answer": (
            "The Eye Smart Helmet backend APIs "
            "generate automated alerts."
        ),
        "expected": "REVIEW"
    },
    {
        "name": "Uncertainty about backend alerts",
        "answer": (
            "I cannot confirm that automated "
            "alert generation was implemented "
            "in the Eye Smart Helmet backend."
        ),
        "expected": "SUPPORTED"
    },
    {
        "name": "API integration versus development",
        "answer": (
            "The MV Sol Inc Dashboard involved API "
            "integration, but the provided profile "
            "does not confirm development of its "
            "backend APIs."
        ),
        "expected": "SUPPORTED"
    }
]


def main():

    sources = load_evidence()

    print(
        f"Loaded {len(sources)} real project chunks."
    )

    test_span_validation(sources)

    passed = 0

    for index, test in enumerate(TEST_CASES, 1):

        print("\n" + "=" * 55)
        print(f"TEST {index}: {test['name']}")

        try:

            evaluation = evaluate_faithfulness(
                test["answer"],
                sources
            )

            claims = evaluation["claims"]

            actual = (
                "SUPPORTED"
                if claims and all(
                    claim["verdict"] == "SUPPORTED"
                    for claim in claims
                )
                else "REVIEW"
            )

            success = actual == test["expected"]

            passed += int(success)

            print("Expected:", test["expected"])
            print("Actual:", actual)
            print(
                "Result:",
                "PASS" if success else "FAIL"
            )

            for claim in claims:

                print("\nClaim:", claim["claim"])
                print(
                    "Type:",
                    claim.get("claim_type")
                )
                print(
                    "Verdict:",
                    claim["verdict"]
                )
                print(
                    "Evidence:",
                    claim.get("evidence_quote", "")
                )
                print(
                    "Reason:",
                    claim.get("reason", "")
                )

        except Exception as error:

            print("Result: FAIL")
            print("Error:", repr(error))

    print("\n" + "=" * 55)
    print(
        f"LAB 15 RESULTS: "
        f"{passed}/{len(TEST_CASES)} passed"
    )


if __name__ == "__main__":
    main()
