
import re
import numpy as np

from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer


# 1. Sample knowledge base
documents = [
    "2.5 years of software engineering experience.",
    "Backend development using PHP and REST APIs.",
    "Software project management.",
    "Mobile application development.",
    "Learning artificial intelligence and RAG."
]


# 2. Tokenization for BM25
def tokenize(text):
    return re.findall(r"\b\w+\b", text.lower())


tokenized_docs = [tokenize(doc) for doc in documents]

bm25 = BM25Okapi(tokenized_docs)


# 3. Load semantic embedding model
model = SentenceTransformer(
    "sentence-transformers/all-MiniLM-L6-v2"
)

document_vectors = model.encode(
    documents,
    normalize_embeddings=True
)


# 4. Reciprocal Rank Fusion
def reciprocal_rank_fusion(rankings, k=60):
    fused_scores = {}

    for ranking in rankings:
        for rank, doc_index in enumerate(ranking, start=1):
            fused_scores[doc_index] = (
                fused_scores.get(doc_index, 0)
                + 1 / (k + rank)
            )

    return sorted(
        fused_scores,
        key=fused_scores.get,
        reverse=True
    )


# 5. Hybrid search
def hybrid_search(question, top_k=3):

    # BM25 keyword retrieval
    bm25_scores = bm25.get_scores(tokenize(question))
    bm25_ranking = list(
        np.argsort(-bm25_scores)
    )

    # Semantic retrieval
    query_vector = model.encode(
        [question],
        normalize_embeddings=True
    )

    semantic_scores = (
        query_vector @ document_vectors.T
    )[0]

    semantic_ranking = list(
        np.argsort(-semantic_scores)
    )

    # Combine both rankings
    fused_ranking = reciprocal_rank_fusion(
        [bm25_ranking, semantic_ranking]
    )

    print("\nBM25 Ranking:")
    for index in bm25_ranking[:top_k]:
        print(documents[index])

    print("\nSemantic Ranking:")
    for index in semantic_ranking[:top_k]:
        print(documents[index])

    print("\nHybrid Ranking:")
    for index in fused_ranking[:top_k]:
        print(documents[index])

    return [
        documents[index]
        for index in fused_ranking[:top_k]
    ]


if __name__ == "__main__":
    hybrid_search(
        "Does Bilawal have backend development experience?"
    )
