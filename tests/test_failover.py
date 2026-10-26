
import os
from dotenv import load_dotenv

load_dotenv()

from memory import (
    create_conversation,
    save_message,
    get_history,
    delete_conversation,
    redis
)

import llm_router


# Save original Groq function
original_groq = llm_router.ask_groq


def simulated_groq_failure(messages):
    raise ConnectionError("TEST: Groq server is unavailable")


def send_message(conversation_id, question):
    history = get_history(conversation_id)

    messages = history + [
        {"role": "user", "content": question}
    ]

    answer, provider = llm_router.ask_ai(
        messages,
        conversation_id
    )

    if answer is None:
        raise RuntimeError("All LLM providers failed")

    save_message(conversation_id, "user", question)
    save_message(conversation_id, "assistant", answer)

    print(f"\nAI ({provider}): {answer}\n")

    return answer, provider


def main():
    conversation_id = create_conversation()
    provider_key = f"chat:{conversation_id}:provider"

    print("=" * 50)
    print("       DIGITAL TWIN AI - FAILOVER TEST")
    print("=" * 50)

    try:
        # TEST 1: Initial conversation with Groq
        print("\n[TEST 1] Starting conversation...")

        answer, provider = send_message(
            conversation_id,
            "My name is Bilawal. I have 2.5 years "
            "of software engineering experience."
        )

        if provider != "Groq":
            raise RuntimeError(
                f"Expected Groq, got {provider}. "
                "Check your Groq configuration."
            )

        # TEST 2: Check memory before switching
        print("\n[TEST 2] Checking memory with Groq...")

        send_message(
            conversation_id,
            "What is my name and how much "
            "experience do I have?"
        )

        # TEST 3: Simulate Groq outage
        print("\n[TEST 3] Simulating Groq failure...")

        llm_router.ask_groq = simulated_groq_failure

        # Update the provider registry used by ask_ai
        # by replacing the module's Groq function reference.
        # If ask_ai builds its provider dictionary at call time,
        # this change takes effect immediately.

        # TEST 4: Ask a question after failover
        print("\n[TEST 4] Checking conversation continuity...")

        answer, new_provider = send_message(
            conversation_id,
            "What is my name and how many years "
            "of experience did I tell you I have?"
        )

        if new_provider == "Groq":
            raise AssertionError(
                "Failover did not occur."
            )

        print(f"Failover successful: Groq -> {new_provider}")

        # Check whether the answer contains expected details
        answer_lower = answer.lower()

        name_found = "bilawal" in answer_lower
        experience_found = (
            "2.5" in answer_lower
            or "two and a half" in answer_lower
        )

        print("Name remembered:", name_found)
        print("Experience remembered:", experience_found)

        assert name_found, "Name was not remembered"
        assert experience_found, "Experience was not remembered"

        # TEST 5: Confirm sticky routing
        print("\n[TEST 5] Checking sticky provider...")

        _, next_provider = send_message(
            conversation_id,
            "Can you summarize my background?"
        )

        assert next_provider == new_provider, (
            "Provider changed unexpectedly."
        )

        selected = redis.get(provider_key)

        if isinstance(selected, bytes):
            selected = selected.decode()

        assert selected == new_provider

        print("\n" + "=" * 50)
        print("ALL FAILOVER TESTS PASSED!")
        print(f"Selected provider: {new_provider}")
        print("Conversation context preserved.")
        print("Redis sticky routing verified.")
        print("=" * 50)

    finally:
        # Restore Groq
        llm_router.ask_groq = original_groq

        # Delete temporary test conversation
        delete_conversation(conversation_id)
        redis.delete(provider_key)

        print("\nTest conversation cleaned up.")


if __name__ == "__main__":
    main()
