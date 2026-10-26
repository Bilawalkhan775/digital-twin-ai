
from rag.retriever import HybridRetriever


TEST_CASES = [
    {
        "question": "Which projects demonstrate Bilawal's backend API development experience?",
        "expected": [
            "Project — Alfalah Builders Web Portal",
            "Project — Eye Smart Helmet / Lamp Monitoring System",
        ],
    },
    {
        "question": "What cybersecurity work has Bilawal done?",
        "expected": [
            "Work Experience — Cybersecurity Apprentice, PSEB / Ultrasoft Systems",
        ],
    },
    {
        "question": "What is Bilawal's educational background?",
        "expected": ["Education"],
    },
    {
        "question": "What machine learning research has Bilawal conducted?",
        "expected": [
            "Research — Undergraduate Thesis: Android Application Vulnerability Detection",
        ],
    },
]


def main():
    retriever = HybridRetriever()
    passed = 0

    for i, test in enumerate(TEST_CASES, start=1):
        results = retriever.search(
            test["question"],
            top_k=5,
            candidates=19
        )

        retrieved = {
            result["section"]
            for result in results
        }

        missing = [
            section
            for section in test["expected"]
            if section not in retrieved
        ]

        success = len(missing) == 0
        passed += int(success)

        print(f"\nTest {i}: {test['question']}")
        print("Result:", "PASS" if success else "FAIL")

        if missing:
            print("Missing:", missing)

    print(f"\nOverall: {passed}/{len(TEST_CASES)} tests passed")


if __name__ == "__main__":
    main()
