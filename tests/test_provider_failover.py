
from unittest.mock import patch

from groq.resources.chat.completions import Completions

from main import generate_answer
from rag.context import RAGService

from memory import (
    create_conversation,
    get_history,
    save_message
)


# ==========================================
# FIVE TEST QUESTIONS
# ==========================================

QUESTIONS = [
    "Who are you?",

    "What backend API development experience does "
    "Bilawal have?",

    "Tell me more about the Eye Smart Helmet project "
    "you just mentioned.",

    "Did the backend APIs in that project implement "
    "automated alert generation?",

    "Apart from that project, which project demonstrates "
    "his PHP REST API development experience?"
]


# ==========================================
# SIMULATE GROQ FAILURE
# ==========================================

def simulated_groq_failure(*args, **kwargs):
    """
    Replace Groq SDK API calls with an error.

    Other providers remain available.
    """
    raise RuntimeError(
        "SIMULATED_GROQ_OUTAGE: Groq API unavailable."
    )


# ==========================================
# SEND QUESTION AND SAVE MEMORY
# ==========================================

def send_question(
    question,
    conversation_id,
    rag,
    question_number
):

    print("\n" + "=" * 70)
    print(f"QUESTION {question_number}/5")
    print("=" * 70)

    print("\nUser:", question)

    # Existing conversation history before
    # asking the current question.
    history_before = get_history(conversation_id)

    print(
        "\nHistory messages before question:",
        len(history_before)
    )

    # Use the real application pipeline.
    result = generate_answer(
        question=question,
        conversation_id=conversation_id,
        rag=rag
    )

    answer = result["answer"]
    provider = result["provider"]

    print("\nProvider:", provider)
    print("\nDigital Twin:", answer)

    # Store messages exactly as main.py does.
    save_message(
        conversation_id,
        "user",
        question
    )

    save_message(
        conversation_id,
        "assistant",
        answer
    )

    history_after = get_history(conversation_id)

    print(
        "\nHistory messages after question:",
        len(history_after)
    )

    print("Conversation ID:", conversation_id)

    # Confirm this question and answer
    # were added to Redis.
    assert len(history_after) == len(history_before) + 2, (
        "Conversation history was not updated correctly."
    )

    assert history_after[-2]["role"] == "user"
    assert history_after[-2]["content"] == question

    assert history_after[-1]["role"] == "assistant"
    assert history_after[-1]["content"] == answer

    return {
        "question": question,
        "answer": answer,
        "provider": provider
    }


# ==========================================
# MAIN FAILOVER TEST
# ==========================================

def main():

    print("\nInitializing Digital Twin AI...")

    rag = RAGService()

    conversation_id = create_conversation()

    print("\nTest conversation:", conversation_id)

    results = []

    # --------------------------------------
    # PHASE 1: NORMAL OPERATION
    # --------------------------------------

    print("\n" + "#" * 70)
    print("PHASE 1: PRIMARY PROVIDER")
    print("#" * 70)

    for index in range(2):

        result = send_question(
            question=QUESTIONS[index],
            conversation_id=conversation_id,
            rag=rag,
            question_number=index + 1
        )

        results.append(result)

    primary_provider = results[-1]["provider"]

    print("\nPrimary provider:", primary_provider)

    # This scenario specifically tests Groq
    # as the primary provider.
    if primary_provider.lower() != "groq":
        raise RuntimeError(
            "Expected Groq before injecting failure, "
            f"but the active provider was {primary_provider}. "
            "Check router priority and provider availability."
        )

    # --------------------------------------
    # PHASE 2: INJECT FAKE API FAILURE
    # --------------------------------------

    print("\n" + "#" * 70)
    print("PHASE 2: SIMULATED GROQ API FAILURE")
    print("#" * 70)

    print(
        "\n[FAULT INJECTION] Disabling Groq API calls "
        "for the remainder of this test."
    )

    # Patching the SDK makes Groq requests fail
    # without changing the router or credentials.
    #
    # The patch is local to this process.
    # It is restored automatically afterward.
    with patch.object(
        Completions,
        "create",
        side_effect=simulated_groq_failure
    ) as mocked_groq:

        for index in range(2, len(QUESTIONS)):

            result = send_question(
                question=QUESTIONS[index],
                conversation_id=conversation_id,
                rag=rag,
                question_number=index + 1
            )

            results.append(result)

        attempted_groq_calls = mocked_groq.call_count

    # --------------------------------------
    # PHASE 3: VALIDATE FAILOVER
    # --------------------------------------

    print("\n" + "#" * 70)
    print("PHASE 3: VALIDATION")
    print("#" * 70)

    fallback_providers = [
        result["provider"]
        for result in results[2:]
    ]

    assert attempted_groq_calls >= 1, (
        "Fault injection was never triggered. "
        "Check how llm_router.py calls Groq."
    )

    assert all(
        provider != "Groq"
        for provider in fallback_providers
    ), (
        "At least one response used Groq "
        "after the simulated outage."
    )

    # Same conversation should contain
    # all five user/assistant exchanges.
    final_history = get_history(conversation_id)

    assert len(final_history) == 10, (
        "Expected 10 conversation messages."
    )

    assert all(
        isinstance(result["answer"], str)
        and result["answer"].strip()
        for result in results
    ), "At least one answer was empty."

    # --------------------------------------
    # FINAL REPORT
    # --------------------------------------

    print("\n" + "=" * 70)
    print("FINAL PROVIDER FAILOVER REPORT")
    print("=" * 70)

    for index, result in enumerate(results, 1):

        print(
            f"Q{index} | "
            f"Provider: {result['provider']}"
        )

    print("\nGroq calls blocked:", attempted_groq_calls)

    print("Conversation ID:", conversation_id)

    print("Total memory messages:", len(final_history))

    print("\nPASS: Groq outage was injected.")
    print("PASS: Another provider continued responding.")
    print("PASS: Conversation ID remained unchanged.")
    print("PASS: All five exchanges were saved in Redis.")

    print(
        "\nMANUAL REVIEW REQUIRED: "
        "Check whether answers 3, 4, and 5 "
        "correctly understood earlier context."
    )


if __name__ == "__main__":
    main()
