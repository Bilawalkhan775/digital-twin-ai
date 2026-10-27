
from rag.faithfulness import evaluate_faithfulness


# ----------------------------------
# CONTROLLED EVIDENCE
# ----------------------------------

SOURCES = [
    {
        "section": "Project — Alfalah Builders Web Portal",
        "text": (
            "Muhammad Bilawal Khan's project:\n"
            "Project — Alfalah Builders Web Portal\n"
            "Role: Backend API and web portal development.\n"
            "Developed REST APIs using PHP.\n"
            "Implemented session management.\n"
            "Developed a web portal for managing "
            "construction transactions, schemes, "
            "and contractor information."
        )
    },
    {
        "section": "Project — Eye Smart Helmet / Lamp Monitoring System",
        "text": (
            "Muhammad Bilawal Khan's project:\n"
            "Project — Eye Smart Helmet / Lamp Monitoring System\n"
            "Role: Mobile application, backend API, "
            "and web portal development.\n"
            "Developed backend APIs for the Eye Smart "
            "Helmet / Lamp Monitoring System.\n"
            "Developed a mobile application using BLE.\n"
            "Worked on environmental monitoring and "
            "automated alert functionality."
        )
    }
]


# ----------------------------------
# TEST CASES
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


def main():

    passed = 0

    for index, test in enumerate(TEST_CASES, 1):

        print("\n" + "=" * 55)
        print(f"TEST {index}: {test['name']}")

        evaluation = evaluate_faithfulness(
            test["answer"],
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
        passed += int(success)

        print("Expected:", test["expected"])
        print("Actual:", actual)
        print("Result:", "PASS" if success else "FAIL")

        for claim in claims:
            print("Claim:", claim["claim"])
            print("Verdict:", claim["verdict"])
            print("Evidence:", claim.get("evidence_quote", ""))
            print("Reason:", claim.get("reason", ""))

    print("\n" + "=" * 55)
    print(f"TOTAL: {passed}/{len(TEST_CASES)} passed")


if __name__ == "__main__":
    main()
