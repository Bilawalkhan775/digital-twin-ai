
import chromadb

from rag.vector_store import DB_PATH
from rag.claim_extractor import extract_claims
from rag.faithfulness import evaluate_faithfulness


def load_sources():

    client = chromadb.PersistentClient(
        path=str(DB_PATH)
    )

    collection = client.get_collection(
        name="digital_twin_knowledge"
    )

    data = collection.get(
        include=["documents", "metadatas"]
    )

    needed = {
        "Project — Alfalah Builders Web Portal",
        "Project — Eye Smart Helmet / Lamp Monitoring System",
        "Project — MV Sol Inc Dashboard Application"
    }

    sources = []

    for document, metadata in zip(
        data["documents"],
        data["metadatas"]
    ):

        if metadata.get("section") in needed:
            sources.append({
                "text": document,
                "section": metadata["section"],
                "person": metadata.get("person", "")
            })

    found = {
        source["section"] for source in sources
    }

    if found != needed:
        raise ValueError(
            f"Missing project chunks: {needed - found}"
        )

    return sources


def test_extraction():

    answer = (
        "I cannot confirm that backend APIs "
        "generate automated alerts."
    )

    claims = extract_claims(answer)

    assert len(claims) == 1
    assert claims[0]["answer_span"] == answer

    assert (
        claims[0]["claim_type"]
        == "EVIDENCE_LIMITATION"
    )

    print("PASS: Negation preserved exactly.")


TEST_CASES = [
    {
        "name": "PHP API development",
        "answer": (
            "Bilawal developed REST APIs using PHP "
            "for the Alfalah Builders Web Portal."
        ),
        "expected": "SUPPORTED"
    },
    {
        "name": "Eye Smart Helmet backend APIs",
        "answer": (
            "Bilawal developed backend APIs for "
            "the Eye Smart Helmet Monitoring System."
        ),
        "expected": "SUPPORTED"
    },
    {
        "name": "Unsupported backend alert ownership",
        "answer": (
            "The Eye Smart Helmet backend APIs "
            "generate automated alerts."
        ),
        "expected": "REVIEW"
    },
    {
        "name": "Accurate uncertainty",
        "answer": (
            "I cannot confirm that automated "
            "alert generation was implemented "
            "in the Eye Smart Helmet backend."
        ),
        "expected": "SUPPORTED"
    },
    {
        "name": "MV Sol Inc API integration",
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

    test_extraction()

    sources = load_sources()

    print(
        f"\nLoaded {len(sources)} real project chunks."
    )

    passed = 0

    for index, test in enumerate(TEST_CASES, 1):

        print("\n" + "=" * 55)
        print(f"TEST {index}: {test['name']}")

        try:

            result = evaluate_faithfulness(
                test["answer"],
                sources
            )

            claims = result["claims"]

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
                print("Type:", claim["claim_type"])
                print("Verdict:", claim["verdict"])
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
        f"TWO-STAGE RESULTS: "
        f"{passed}/{len(TEST_CASES)} passed"
    )


if __name__ == "__main__":
    main()
