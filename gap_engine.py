"""
UNPLLM - Stage 3: Shared gap-clustering logic
Used by both analyze_gaps.py (reporting) and sync_watcher.py (deciding
what to fetch). Kept in one place so both always agree on what counts
as a "topic" — only reads UNRESOLVED gaps, so already-synced topics
don't keep reappearing.
"""

import sqlite3
import json
import numpy as np
from query_logger import DB_PATH

SIMILARITY_THRESHOLD = 0.85  # queries above this cosine similarity count as "the same topic"


def cosine_similarity(a, b):
    a, b = np.array(a), np.array(b)
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))


def load_unresolved_gaps():
    conn = sqlite3.connect(DB_PATH)
    rows = conn.execute(
        "SELECT query_text, query_embedding, top_distance FROM query_log "
        "WHERE is_gap = 1 AND resolved = 0"
    ).fetchall()
    conn.close()
    return [
        {"text": text, "embedding": json.loads(emb), "distance": dist}
        for text, emb, dist in rows
    ]


def cluster_gaps(gap_queries):
    """Groups near-duplicate gap queries together, most frequent first."""
    clusters = []

    for gq in gap_queries:
        placed = False
        for cluster in clusters:
            sim = cosine_similarity(gq["embedding"], cluster["queries"][0]["embedding"])
            if sim >= SIMILARITY_THRESHOLD:
                cluster["queries"].append(gq)
                placed = True
                break
        if not placed:
            clusters.append({"queries": [gq]})

    clusters.sort(key=lambda c: len(c["queries"]), reverse=True)
    return clusters
