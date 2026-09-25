"""
UNPLLM - Stage 3: Query logging
Every query sent to /generate_rag gets logged here: what was asked,
how well it matched the knowledge base, and whether it counts as a
"gap" (something the local knowledge base doesn't cover well).

This log is what the sync engine (Week 14+) will read to decide what
to fetch the next time the internet is available.
"""

import sqlite3
import json
from datetime import datetime, timezone

DB_PATH = "./unpllm_queries.db"

# Empirically calibrated from Stage 2 testing:
#   0.8978 = confirmed good match ("What is UNPLLM?")
#   2.0809 = confirmed bad match ("capital of France")
# Starting threshold sits closer to the good end on purpose: with a
# near-empty knowledge base, most queries genuinely ARE gaps right
# now, so it's correct to flag somewhat aggressively at this stage.
# Retune this once real usage data builds up.
GAP_THRESHOLD = 1.4


def init_db():
    conn = sqlite3.connect(DB_PATH)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS query_log (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TEXT NOT NULL,
            query_text TEXT NOT NULL,
            query_embedding TEXT NOT NULL,
            top_distance REAL,
            retrieved_chunk_preview TEXT,
            is_gap INTEGER NOT NULL,
            resolved INTEGER NOT NULL DEFAULT 0
        )
    """)
    # Migration safety net: your database file already exists from
    # before this column was added. CREATE TABLE IF NOT EXISTS is a
    # no-op on an existing table, so this adds the column by hand
    # instead of silently failing later with "no such column".
    existing_cols = [row[1] for row in conn.execute("PRAGMA table_info(query_log)")]
    if "resolved" not in existing_cols:
        conn.execute("ALTER TABLE query_log ADD COLUMN resolved INTEGER NOT NULL DEFAULT 0")
    conn.commit()
    conn.close()


def log_query(query_text, query_embedding, top_distance, retrieved_chunk_preview):
    """Logs one query and returns True if it was flagged as a knowledge gap."""
    is_gap = 1 if (top_distance is None or top_distance > GAP_THRESHOLD) else 0
    conn = sqlite3.connect(DB_PATH)
    conn.execute(
        """INSERT INTO query_log
           (timestamp, query_text, query_embedding, top_distance, retrieved_chunk_preview, is_gap)
           VALUES (?, ?, ?, ?, ?, ?)""",
        (
            datetime.now(timezone.utc).isoformat(),
            query_text,
            json.dumps(query_embedding),
            top_distance,
            retrieved_chunk_preview,
            is_gap,
        ),
    )
    conn.commit()
    conn.close()
    return bool(is_gap)


def mark_topic_resolved(query_texts):
    """Marks all logged rows matching these exact query texts as resolved,
    so the sync engine doesn't keep re-fetching the same topic every cycle."""
    conn = sqlite3.connect(DB_PATH)
    conn.executemany(
        "UPDATE query_log SET resolved = 1 WHERE query_text = ?",
        [(q,) for q in query_texts],
    )
    conn.commit()
    conn.close()


init_db()
