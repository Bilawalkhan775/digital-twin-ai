
from sentence_transformers import SentenceTransformer
import numpy as np

model = SentenceTransformer(
    "sentence-transformers/all-MiniLM-L6-v2"
)

documents = [
    "2.5 years of software engineering experience.",
    "Backend development using PHP and REST APIs.",
    "Software project management.",
    "Mobile application development.",
    "Learning artificial intelligence and RAG."
]

question = "Does Bilawal have backend development experience?"

# Generate embeddings
document_vectors = model.encode(documents)
question_vector = model.encode([question])

# Calculate similarities
scores = model.similarity(
    question_vector,
    document_vectors
)[0].cpu().numpy()

# Retrieve top 3 results
top_k = 3

top_indices = np.argsort(scores)[::-1][:top_k]

print("Question:", question)
print("\nTop 3 Retrieved Documents:\n")

for rank, index in enumerate(top_indices, start=1):
    print(f"{rank}. {documents[index]}")
    print(f"   Similarity: {scores[index]:.4f}\n")
