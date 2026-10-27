
from pathlib import Path

import chromadb
from sentence_transformers import SentenceTransformer

from rag.ingest import ingest_knowledge


PROJECT_DIR = Path(__file__).resolve().parent.parent
DB_PATH = PROJECT_DIR / "data" / "chroma"

MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"


def build_vector_store():

    # Load knowledge chunks
    chunks = ingest_knowledge()

    if not chunks:
        print("No knowledge chunks found.")
        return

    # Load model
    model = SentenceTransformer(MODEL_NAME)

    # Generate vector embeddings
    embeddings = model.encode(
        [chunk["text"] for chunk in chunks],
        normalize_embeddings=True
    ).tolist()

    # Open persistent database
    client = chromadb.PersistentClient(
        path=str(DB_PATH)
    )

    collection = client.get_or_create_collection(
        name="digital_twin_knowledge",
        metadata={"hnsw:space": "cosine"}
    )

    # Remove previous records for this full rebuild
    existing = collection.get(include=[])

    if existing["ids"]:
        collection.delete(ids=existing["ids"])

    # Save updated knowledge chunks
    collection.upsert(
        ids=[chunk["id"] for chunk in chunks],
        documents=[chunk["text"] for chunk in chunks],
        embeddings=embeddings,
        metadatas=[
            {
                "source": chunk["source"],
                "section": chunk["section"],
                "chunk_index": chunk["chunk_index"],
                "person": chunk["person"]
            }
            for chunk in chunks
        ]
    )

    print("\nVector store created successfully!")
    print("Stored chunks:", collection.count())
    print("Database:", DB_PATH)


if __name__ == "__main__":
    build_vector_store()
