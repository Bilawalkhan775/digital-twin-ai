
from uuid import uuid4

from rag.context import RAGService
from rag.answer_guard import verify_and_revise


def main():
    rag = RAGService()

    question = (
        "Which projects demonstrate Bilawal's "
        "backend API development experience?"
    )

    retrieval = rag.retrieve(question, top_k=5)

    original_answer = """
Bilawal developed PHP REST APIs for the
Alfalah Builders Web Portal.

He also developed backend APIs for the
Eye Smart Helmet Monitoring System.

These backend APIs directly handle BLE sensor
communication and implement automated
alert-generation logic.
"""

    result = verify_and_revise(
        question=question,
        answer=original_answer,
        retrieved_results=retrieval["results"],
        conversation_id=str(uuid4())
    )

    print("\nORIGINAL ANSWER:")
    print(original_answer)

    print("\nFINAL ANSWER:")
    print(result["answer"])

    print("\nRevised:", result["revised"])
    print("Evaluator verified:", result["verified"])

    print("\nFINAL CLAIMS:")
    for claim in result["evaluation"]["claims"]:
        print(
            claim["verdict"],
            "|",
            claim["claim"]
        )


if __name__ == "__main__":
    main()
