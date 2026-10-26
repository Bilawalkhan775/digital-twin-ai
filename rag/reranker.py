
from sentence_transformers import CrossEncoder

from rag.hybrid_search import hybrid_search


# Load a pretrained reranking model
reranker = CrossEncoder(
    "cross-encoder/ms-marco-MiniLM-L6-v2"
)


def rerank(question, documents, top_k=3):

    # Pair every document with the question
    pairs = [
        (question, document)
        for document in documents
    ]

    # Predict relevance scores
    scores = reranker.predict(pairs)

    # Sort by descending relevance
    ranked = sorted(
        zip(documents, scores),
        key=lambda item: float(item[1]),
        reverse=True
    )

    print("\nReranked Results:")

    for document, score in ranked[:top_k]:
        print(
            f"Score: {float(score):.4f} | {document}"
        )

    return [
        document
        for document, _ in ranked[:top_k]
    ]


if __name__ == "__main__":

    question = (
        "Does Bilawal have backend development experience?"
    )

    # Retrieve candidates using Hybrid Search
    candidates = hybrid_search(question, top_k=5)

    # Rerank all candidates
    rerank(question, candidates, top_k=3)
