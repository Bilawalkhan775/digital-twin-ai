
from pathlib import Path

import chromadb

from rag.faithfulness import evaluate_faithfulness


# ----------------------------------
# DATABASE CONFIGURATION
# ----------------------------------

PROJECT_DIR = Path(__file__).resolve().parent.parent
DB_PATH = PROJECT_DIR / "data" / "chroma"

COLLECTION_NAME = "digital_twin_knowledge"


# ----------------------------------
# LOAD REAL EVIDENCE
# ----------------------------------

def load_project_evidence():

    client = chromadb.PersistentClient(
        path=str(DB_PATH)
    )

    collection = client.get_collection(
        name=COLLECTION_NAME
    )

    stored = collection.get(
        include=["documents", "metadatas"]
    )

    required_sections = [
        "Project — Alfalah Builders Web Portal",
        "Project — Eye Smart Helmet / Lamp Monitoring System"
    ]

    sources = []

    for section in required_sections:

        matches = []

        for document, metadata in zip(
            stored["documents"],
            stored["metadatas"]
        ):
            if metadata.get("section") == section:
                matches.append({
                    "section": section,
                    "text": document,
                    "person": metadata.get("person", "")
                })

        if not matches:
            raise ValueError(
                f"Missing project section in ChromaDB: {section}"
            )

        sources.extend(matches)

    print(f"Loaded {len(sources)} real knowledge chunks.")

    for source in sources:
        print(" -", source["section"])

    return sources


# ----------------------------------
# REGRESSION TEST CASES
# ----------------------------------

TEST_CASES = [
    {
        "name": "PHP API Development",
        "answer": (
            "Bilawal developed REST APIs using PHP "
            "for the Alfalah Builders Web Portal."
        ),
        "expected": "SUPPORTED"
    },
    {
        "name": "Eye Smart Helmet Backend APIs",
        "answer": (
            "Bilawal developed backend APIs for "
            "the Eye Smart Helmet Monitoring System."
        ),
        "expected": "SUPPORTED"
    },
    {
        "name": "Unsupported Backend BLE",
        "answer": (
            "The Eye Smart Helmet backend APIs "
            "directly handle BLE communication."
        ),
        "expected": "REVIEW"
    },
    {
        "name": "Unsupported Backend Alerts",
        "answer": (
            "The Eye Smart Helmet backend APIs "
            "implement automated alert-generation logic."
        ),
        "expected": "REVIEW"
    }
]


# ----------------------------------
# RUN EVALUATION
# ----------------------------------

def main():

    sources = load_project_evidence()

    passed = 0

    for index, test in enumerate(TEST_CASES, 1):

        print("\n" + "=" * 60)
        print(f"TEST {index}: {test['name']}")

        evaluation = evaluate_faithfulness(
            answer=test["answer"],
            retrieved_results=sources
        )

        claims = evaluation["claims"]

        all_supported = (
            bool(claims)
            and all(
                claim["verdict"] == "SUPPORTED"
                for claim in claims
            )
        )

        actual = (
            "SUPPORTED"
            if all_supported
            else "REVIEW"
        )

        success = actual == test["expected"]

        if success:
            passed += 1

        print("Expected:", test["expected"])
        print("Actual:", actual)
        print("Result:", "PASS" if success else "FAIL")

        for claim in claims:

            print("\nClaim:", claim["claim"])
            print("Verdict:", claim["verdict"])
            print(
                "Evidence:",
                claim.get("evidence_quote", "")
            )
            print(
                "Reason:",
                claim.get("reason", "")
            )

    print("\n" + "=" * 60)
    print(
        f"REAL EVIDENCE RESULTS: "
        f"{passed}/{len(TEST_CASES)} passed"
    )


if __name__ == "__main__":
    main()
