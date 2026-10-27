
from pathlib import Path
import hashlib
import re

PROJECT_DIR = Path(__file__).resolve().parent.parent
KNOWLEDGE_DIR = PROJECT_DIR / "knowledge"


def load_documents():
    documents = []

    for path in sorted(KNOWLEDGE_DIR.rglob("*.md")):
        if not path.is_file():
            continue

        documents.append({
            "source": path.relative_to(
                KNOWLEDGE_DIR
            ).as_posix(),
            "content": path.read_text(encoding="utf-8")
        })

    return documents


def chunk_document(document, max_words=120):
    content = document["content"]

    # Split at Markdown H2 headings
    sections = re.split(
        r"(?=^##\s+)",
        content,
        flags=re.MULTILINE
    )

    chunks = []

    for section in sections:
        section = section.strip()

        # Only process actual H2 sections
        if not section.startswith("## "):
            continue

        lines = section.splitlines()
        heading = lines[0].removeprefix("## ").strip()
        body = "\n".join(lines[1:]).strip()

        # Exclude behavioral instructions from retrieved knowledge
        excluded_sections = {
            "Answering Guidance for the Digital Twin",
            "Contact and Public Links"
        }

        if heading in excluded_sections:
            continue

        if not body:
            continue

        # Split long sections into smaller pieces
        words = body.split()

        for start in range(0, len(words), max_words):
            part = " ".join(
                words[start:start + max_words]
            )

            text = f"{heading}\n{part}"
            index = len(chunks)

            unique_key = (
                f"{document['source']}:{heading}:"
                f"{start}:{text}"
            )

            chunk_id = hashlib.sha256(
                unique_key.encode("utf-8")
            ).hexdigest()[:16]

            chunks.append({
                "id": chunk_id,
                "source": document["source"],
                "section": heading,
                "chunk_index": index,
                "person": "Muhammad Bilawal Khan",
                "text": text
            })

    return chunks


def ingest_knowledge():
    all_chunks = []

    for document in load_documents():
        chunks = chunk_document(document)
        all_chunks.extend(chunks)

        print(
            f"Loaded {document['source']}: "
            f"{len(chunks)} chunks"
        )

    return all_chunks


if __name__ == "__main__":
    chunks = ingest_knowledge()

    print("\nTotal chunks:", len(chunks))

    for chunk in chunks:
        print("\n" + "-" * 50)
        print("Section:", chunk["section"])
        print("Source:", chunk["source"])
        print("Text:", chunk["text"])
