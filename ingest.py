"""
UNPLLM - Stage 2: Ingestion script
Reads .txt and .md files from ./documents, chunks them, embeds them,
and stores them in the local ChromaDB vector store.

Run whenever you add new source material (safe to re-run — it
overwrites matching chunk IDs instead of duplicating them):
    python3 ingest.py
"""

import os
import glob
from rag_utils import chunk_text, embed_texts, get_collection

DOCS_FOLDER = "./documents"


def load_documents():
    paths = glob.glob(os.path.join(DOCS_FOLDER, "*.txt")) + \
            glob.glob(os.path.join(DOCS_FOLDER, "*.md"))
    docs = []
    for path in paths:
        with open(path, "r", encoding="utf-8") as f:
            docs.append((os.path.basename(path), f.read()))
    return docs


def ingest():
    collection = get_collection()
    documents = load_documents()

    if not documents:
        print(f"No .txt or .md files found in {DOCS_FOLDER}/ — add some and re-run.")
        return

    all_chunks, all_ids, all_metadata = [], [], []
    for filename, text in documents:
        chunks = chunk_text(text)
        for i, chunk in enumerate(chunks):
            all_chunks.append(chunk)
            all_ids.append(f"{filename}-{i}")
            all_metadata.append({"source": filename, "chunk_index": i})

    print(f"Embedding {len(all_chunks)} chunks from {len(documents)} file(s)...")
    embeddings = embed_texts(all_chunks)

    collection.upsert(
        ids=all_ids,
        embeddings=embeddings,
        documents=all_chunks,
        metadatas=all_metadata,
    )
    print(f"Done. {len(all_chunks)} chunks stored in the vector database.")
    print(f"Total chunks in collection: {collection.count()}")


if __name__ == "__main__":
    ingest()
