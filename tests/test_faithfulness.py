
from rag.faithfulness import evaluate_faithfulness


# Evidence taken from the current professional profile
SOURCES = [
    {
        "section": "Project — Alfalah Builders Web Portal",
        "text": (
            "Project — Alfalah Builders Web Portal\n"
            "Role: Backend API and web portal development.\n"
            "Developed REST APIs using PHP.\n"
            "Implemented session management.\n"
            "Developed a web portal for managing construction "
            "transactions, schemes, and contractor information."
        )
    },
    {
        "section": "Project — Eye Smart Helmet / Lamp Monitoring System",
        "text": (
            "Project — Eye Smart Helmet / Lamp Monitoring System\n"
            "Role: Mobile application, backend API, and "
            "web portal development.\n"
            "Developed backend APIs for the Eye Smart "
            "Helmet / Lamp Monitoring System.\n"
            "Developed a mobile application that communicates "
            "with smart devices using Bluetooth Low Energy (BLE).\n"
            "Worked on environmental monitoring and automated "
            "alert functionality for miners' safety."
        )
    }
]


TEST_CASES = [
    {
        "name": "Alfalah PHP API development",
        "claim": (
            "Bilawal developed REST APIs using PHP "
            "for the Alfalah Builders Web Portal."
        ),
        "expected": "SUPPORTED"
    },
    {
        "name": "Eye Smart Helmet backend API development",
        "claim": (
            "Bilawal developed backend APIs for "
            "the Eye Smart Helmet Monitoring System."
        ),
        "expected": "SUPPORTED"
    },
    {
        "name": "Backend BLE implementation",
        "claim": (
            "The Eye Smart Helmet backend APIs "
            "directly handle BLE communication."
        ),
        "expected": "REVIEW"
    },
    {
        "name": "Backend alert-generation implementation",
        "claim": (
            "The Eye Smart Helmet backend APIs "
            "implement automated alert-generation logic."
        ),
        "expected": "REVIEW"
    }
]


def main():
    passed = 0

    for test in TEST_CASES:
        evaluation = evaluate_faithfulness(
            test["claim"],
            SOURCES
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
        passed += success

        print(f"\nTest: {test['name']}")
        print("Expected:", test["expected"])
        print("Actual:", actual)
        print("Result:", "PASS" if success else "FAIL")

        for claim in claims:
            print("Verdict:", claim["verdict"])
            print("Evidence:", claim.get("evidence_quote", ""))
            print("Reason:", claim.get("reason", ""))

    print(f"\nTotal: {passed}/{len(TEST_CASES)} passed")


if __name__ == "__main__":
    main()
