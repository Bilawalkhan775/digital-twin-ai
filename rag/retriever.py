
import re

import chromadb
import numpy as np

from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer, CrossEncoder

from rag.vector_store import DB_PATH, MODEL_NAME


class HybridRetriever:

    def __init__(self):
        # Open existing Chroma database
        self.client = chromadb.PersistentClient(
            path=str(DB_PATH)
        )

        self.collection = self.client.get_collection(
            name="digital_twin_knowledge"
        )

        # Load documents stored in Chroma
        stored = self.collection.get(
            include=["documents", "metadatas"]
        )

        self.ids = stored["ids"]
        self.documents = stored["documents"]
        self.metadatas = stored["metadatas"]

        if not self.documents:
            raise ValueError(
                "Knowledge base is empty. Run rag.vector_store first."
            )

        # Build BM25 index using the same documents
        tokenized_docs = [
            self.tokenize(doc)
            for doc in self.documents
        ]

        self.bm25 = BM25Okapi(tokenized_docs)

        # Embedding model
        self.embedding_model = SentenceTransformer(
            MODEL_NAME
        )

        # Cross-Encoder reranker
        self.reranker = CrossEncoder(
            "cross-encoder/ms-marco-MiniLM-L6-v2"
        )

        print(
            f"Retriever ready: {len(self.documents)} chunks"
        )

    @staticmethod
    def tokenize(text):
        return re.findall(r"\b\w+\b", text.lower())

    @staticmethod
    def reciprocal_rank_fusion(rankings, k=60):
        scores = {}

        for ranking in rankings:
            for rank, doc_id in enumerate(ranking, 1):
                scores[doc_id] = (
                    scores.get(doc_id, 0)
                    + 1 / (k + rank)
                )

        return sorted(
            scores,
            key=scores.get,
            reverse=True
        )

    def search(self, question, top_k=3, candidates=19):

        if not question.strip():
            raise ValueError("Question cannot be empty.")

        count = len(self.documents)
        candidates = min(max(candidates, top_k), count)

        # 1. Keyword search
        bm25_scores = self.bm25.get_scores(
            self.tokenize(question)
        )

        bm25_order = np.argsort(-bm25_scores)

        bm25_ids = [
            self.ids[i]
            for i in bm25_order[:candidates]
            if bm25_scores[i] > 0
        ]

        # 2. Vector search
        question_embedding = self.embedding_model.encode(
            [question],
            normalize_embeddings=True
        ).tolist()

        vector_results = self.collection.query(
            query_embeddings=question_embedding,
            n_results=candidates
        )

        vector_ids = vector_results["ids"][0]

        # 3. Merge using RRF
        fused_ids = self.reciprocal_rank_fusion(
            [bm25_ids, vector_ids]
        )

        fused_ids = fused_ids[:candidates]

        # Match IDs to stored documents
        documents_by_id = {
            doc_id: (doc, metadata)
            for doc_id, doc, metadata in zip(
                self.ids,
                self.documents,
                self.metadatas
            )
        }

        # 4. Cross-Encoder reranking
        pairs = [
            (question, documents_by_id[doc_id][0])
            for doc_id in fused_ids
        ]

        scores = self.reranker.predict(pairs)

        ranked = sorted(
            zip(fused_ids, scores),
            key=lambda item: float(item[1]),
            reverse=True
        )

        # 5. Return structured results
        results = []

        for doc_id, score in ranked[:top_k]:
            document, metadata = documents_by_id[doc_id]

            results.append({
                "id": doc_id,
                "text": document,
                "source": metadata["source"],
                "section": metadata["section"],
                "person": metadata.get("person", ""),
                "score": float(score)
            })

        return results


if __name__ == "__main__":

    retriever = HybridRetriever()

    question = input("\nAsk about Bilawal: ")

    results = retriever.search(question)

    print("\nRetrieved Knowledge:")

    for i, result in enumerate(results, 1):
        print(f"\nResult {i}")
        print("Section:", result["section"])
        print("Source:", result["source"])
        print("Score:", round(result["score"], 4))
        print("Text:", result["text"])
