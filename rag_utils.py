"""
UNPLLM - Stage 2: RAG utilities
Shared chunking, embedding, and retrieval functions used by both
ingest.py (populates the vector store) and main.py (queries it).
Keeping this in one place means Stage 3's sync engine can reuse the
exact same embedder later, instead of duplicating this logic.
"""

import os

# Must be set BEFORE sentence_transformers/huggingface_hub are imported.
# By default, loading a model still tries to phone home to check for
# updates even when it's already cached locally — which fails loudly
# if there's no connection. This forces it to rely purely on the local
# cache, which is exactly what an offline-first project needs. Safe
# here because the model has already been downloaded once (confirmed
# by your earlier successful runs) — this only blocks NEW downloads,
# not use of what's already cached.
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

from sentence_transformers import SentenceTransformer
import chromadb

EMBED_MODEL_NAME = "all-MiniLM-L6-v2"
CHROMA_PATH = "./chroma_db"
COLLECTION_NAME = "unpllm_knowledge"

# Loaded once, reused across calls — loading the model on every
# request would add several seconds of overhead each time.
_embedder = None
_chroma_client = None


def get_embedder():
    global _embedder
    if _embedder is None:
        _embedder = SentenceTransformer(EMBED_MODEL_NAME)
    return _embedder


def get_collection():
    global _chroma_client
    if _chroma_client is None:
        _chroma_client = chromadb.PersistentClient(path=CHROMA_PATH)
    return _chroma_client.get_or_create_collection(name=COLLECTION_NAME)


def chunk_text(text, chunk_words=400, overlap_words=40):
    """
    Splits text into overlapping word-based chunks.
    ~400 words is roughly 300-500 tokens for typical English text.
    """
    words = text.split()
    chunks = []
    start = 0
    while start < len(words):
        end = start + chunk_words
        chunk = " ".join(words[start:end])
        if chunk.strip():
            chunks.append(chunk)
        if end >= len(words):
            break
        start += chunk_words - overlap_words
    return chunks


def embed_texts(texts):
    """Returns a list of embedding vectors for a list of strings."""
    embedder = get_embedder()
    return embedder.encode(texts).tolist()


def retrieve_chunks(query, top_k=4):
    """
    Returns (chunks, query_embedding):
      - chunks: list of (chunk_text, distance) tuples, most to least
        relevant. Lower distance = more relevant — this is a distance
        metric, not a similarity score.
      - query_embedding: the query's own embedding vector, returned so
        callers (Stage 3's query logger) don't need to recompute it.
    """
    collection = get_collection()
    query_embedding = embed_texts([query])[0]

    if collection.count() == 0:
        return [], query_embedding

    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=min(top_k, collection.count()),
    )
    chunks = results["documents"][0] if results["documents"] else []
    distances = results["distances"][0] if results["distances"] else []
    return list(zip(chunks, distances)), query_embedding


def enforce_storage_cap(max_synced_chunks=100):
    """
    Keeps the number of SYNC-ADDED chunks under a cap by evicting the
    oldest ones first (by synced_at timestamp) — a simple FIFO policy,
    not true LRU-by-access, chosen because it needs no extra tracking
    on every retrieval call. Original documents from ingest.py are
    never touched; only content the sync engine added is subject to
    eviction. Returns the number of chunks evicted (0 if under cap).
    """
    collection = get_collection()
    result = collection.get(where={"is_synced": True}, include=["metadatas"])

    ids = result["ids"]
    metadatas = result["metadatas"]

    if len(ids) <= max_synced_chunks:
        return 0

    paired = sorted(zip(ids, metadatas), key=lambda pair: pair[1].get("synced_at", ""))
    num_to_evict = len(ids) - max_synced_chunks
    ids_to_evict = [pair[0] for pair in paired[:num_to_evict]]

    collection.delete(ids=ids_to_evict)
    return len(ids_to_evict)
