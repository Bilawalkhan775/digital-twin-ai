
from dotenv import load_dotenv

load_dotenv()

from memory import create_conversation, save_message, get_history
from llm_router import ask_ai


def main():
    conversation_id = create_conversation()

    print("Welcome to Digital Twin AI!")
    print("Type 'exit' to quit.\n")

    while True:
        try:
            question = input("You: ").strip()

            if question.lower() == "exit":
                print("Goodbye!")
                break

            if not question:
                continue

            history = get_history(conversation_id)

            messages = history + [
                {"role": "user", "content": question}
            ]

            answer, provider = ask_ai(
                messages,
                conversation_id
            )

            if answer is None:
                print("\nAI is temporarily unavailable.\n")
                continue

            save_message(conversation_id, "user", question)
            save_message(conversation_id, "assistant", answer)

            print(f"\nAI ({provider}): {answer}\n")

        except (KeyboardInterrupt, EOFError):
            print("\nGoodbye!")
            break

        except Exception:
            print("\nAn unexpected error occurred.\n")


if __name__ == "__main__":
    main()
