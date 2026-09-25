"""
UNPLLM - Stage 3: Connectivity watcher & opportunistic sync
This is the core innovation piece. Run it in its own terminal,
alongside main.py:

    python3 sync_watcher.py

It polls for internet connectivity. The moment it detects the machine
has come back online after being offline, it looks at the current
unresolved knowledge gaps (from query_logger's log), picks the most
frequently-asked ones, fetches relevant content for them from
Wikipedia, and adds it to the same vector store main.py already
retrieves from. No restart needed — the next /generate_rag query
benefits immediately.

Demo flow: disconnect Wi-Fi, ask something outside the current
knowledge base via /generate_rag (a couple of times, to make it a
real repeated gap, not a one-off), reconnect, watch this terminal
fetch and store new content, then ask the same question again and
see a properly grounded answer.
"""

import time
import requests
import ollama
from datetime import datetime, timezone

from gap_engine import load_unresolved_gaps, cluster_gaps, cosine_similarity
from query_logger import mark_topic_resolved
from rag_utils import chunk_text, embed_texts, get_collection, enforce_storage_cap

CHECK_INTERVAL_SECONDS = 15
CONNECTIVITY_CHECK_URL = "https://www.google.com"
TOP_N_TOPICS_PER_SYNC = 3
MAX_EXTRACT_WORDS = 2500  # caps how much of one Wikipedia article gets ingested per topic
MAX_SYNCED_CHUNKS = 100   # oldest synced chunks get evicted beyond this — see rag_utils.enforce_storage_cap
MODEL_NAME = "phi4-mini"

# Loose floor, not a precision filter — just meant to catch obviously
# wrong fetches (an unrelated article for an oddly-phrased question),
# not to second-guess borderline-relevant ones.
RELEVANCE_SANITY_FLOOR = 0.15

WIKI_API_URL = "https://en.wikipedia.org/w/api.php"
WIKI_HEADERS = {"User-Agent": "UNPLLM-FinalYearProject/1.0 (student project; contact: n/a)"}


def is_online():
    try:
        requests.head(CONNECTIVITY_CHECK_URL, timeout=3)
        return True
    except requests.RequestException:
        return False


def extract_search_terms(user_query):
    """Uses the local model to turn a natural-language question into
    concise Wikipedia search keywords. Wikipedia's search is keyword
    based, not semantic, so a raw question like "how do plants make
    their own food?" can return an unrelated top result. Falls back to
    the raw query if this fails for any reason."""
    try:
        result = ollama.chat(
            model=MODEL_NAME,
            messages=[
                {"role": "system", "content": (
                    "Extract exactly 3 to 5 concise search keywords that "
                    "would find the Wikipedia article answering the "
                    "user's question. Reply with ONLY the keywords "
                    "separated by spaces — no punctuation, no "
                    "explanation. Example: for 'how tall is the Eiffel "
                    "Tower', reply 'Eiffel Tower height'."
                )},
                {"role": "user", "content": user_query},
            ],
            options={"temperature": 0.0, "num_predict": 30},
        )
        terms = result["message"]["content"].strip().strip('"')
        # Small models don't reliably self-enforce a word-count limit —
        # enforce it in code rather than trusting the instruction alone.
        terms = " ".join(terms.split()[:5])
        return terms if terms else user_query
    except Exception:
        return user_query


def is_realtime_query(query):
    """Flags questions that need live/current information — weather,
    scores, prices, today's news — that NO static, cached source could
    ever correctly answer. Syncing should skip these entirely rather
    than store a wrong or coincidentally-topic-adjacent result."""
    try:
        result = ollama.chat(
            model=MODEL_NAME,
            messages=[
                {"role": "system", "content": (
                    "Does answering this question require real-time or "
                    "live information (current weather, live scores, "
                    "today's news, current prices) that a static, "
                    "cached knowledge source could never correctly "
                    "provide? Reply with only YES or NO."
                )},
                {"role": "user", "content": query},
            ],
            options={"temperature": 0.0, "num_predict": 5},
        )
        return result["message"]["content"].strip().upper().startswith("YES")
    except Exception:
        return False  # uncertain — don't block a possibly-valid sync attempt


def find_wikipedia_title(query):
    """Uses Wikipedia's search API to find the best-matching article title."""
    params = {
        "action": "query",
        "list": "search",
        "srsearch": query,
        "srlimit": 1,
        "format": "json",
    }
    resp = requests.get(WIKI_API_URL, params=params, headers=WIKI_HEADERS, timeout=10)
    resp.raise_for_status()
    results = resp.json().get("query", {}).get("search", [])
    return results[0]["title"] if results else None


def fetch_wikipedia_extract(title):
    """Fetches the full plain-text extract of a Wikipedia article (not just the intro)."""
    params = {
        "action": "query",
        "prop": "extracts",
        "explaintext": True,
        "titles": title,
        "format": "json",
    }
    resp = requests.get(WIKI_API_URL, params=params, headers=WIKI_HEADERS, timeout=15)
    resp.raise_for_status()
    pages = resp.json().get("query", {}).get("pages", {})
    for page in pages.values():
        return page.get("extract", "")
    return ""


def sync_topic(representative_query, query_embedding):
    """Looks up one gap topic on Wikipedia and adds it to the knowledge
    base, if the result actually looks relevant. Returns True if the
    topic should be marked resolved (either successfully synced, or
    correctly recognized as unsyncable) — False if it should be
    retried on a future cycle."""
    if is_realtime_query(representative_query):
        print(f"  Skipping (needs real-time info a static fetch can't "
              f"provide): {representative_query}")
        return True  # no future sync attempt would help either

    search_terms = extract_search_terms(representative_query)
    print(f"  Searching Wikipedia for: {representative_query}")
    print(f"    (search terms used: {search_terms})")

    title = find_wikipedia_title(search_terms)
    if not title:
        print(f"    No Wikipedia match found — skipping this topic.")
        return False

    print(f"    Found article: {title}")
    extract = fetch_wikipedia_extract(title)
    if not extract:
        print(f"    Article had no usable text — skipping.")
        return False

    # Sanity check: does the fetched article actually relate to the
    # original question? Catches cases like keyword search returning
    # a topically-adjacent but wrong-entity article. Always logged
    # (not just on failure) so the threshold can be calibrated with
    # real numbers over time.
    sample_embedding = embed_texts([extract[:1000]])[0]
    relevance = cosine_similarity(sample_embedding, query_embedding)
    print(f"    Relevance check: similarity {relevance:.3f} (floor: {RELEVANCE_SANITY_FLOOR})")
    if relevance < RELEVANCE_SANITY_FLOOR:
        print(f"    '{title}' doesn't look related enough — discarding, not stored.")
        return False

    words = extract.split()
    if len(words) > MAX_EXTRACT_WORDS:
        extract = " ".join(words[:MAX_EXTRACT_WORDS])

    chunks = chunk_text(extract)
    ids = [f"sync-{title}-{i}" for i in range(len(chunks))]
    metadatas = [
        {
            "source": f"wikipedia:{title}",
            "chunk_index": i,
            "synced_at": datetime.now(timezone.utc).isoformat(),
            "is_synced": True,
        }
        for i in range(len(chunks))
    ]
    embeddings = embed_texts(chunks)

    collection = get_collection()
    collection.upsert(ids=ids, embeddings=embeddings, documents=chunks, metadatas=metadatas)
    print(f"    Added {len(chunks)} new chunk(s) from '{title}' to the knowledge base.")
    return True


def run_sync_cycle():
    print(f"\n[{datetime.now().strftime('%H:%M:%S')}] Back online — checking for knowledge gaps...")
    gap_queries = load_unresolved_gaps()

    if not gap_queries:
        print("  No unresolved gaps right now. Nothing to sync.\n")
        return

    clusters = cluster_gaps(gap_queries)
    top_clusters = clusters[:TOP_N_TOPICS_PER_SYNC]
    print(f"  {len(clusters)} unresolved topic(s) found. Syncing top {len(top_clusters)}.")

    for cluster in top_clusters:
        representative = cluster["queries"][0]
        success = sync_topic(representative["text"], representative["embedding"])
        if success:
            all_texts = [q["text"] for q in cluster["queries"]]
            mark_topic_resolved(all_texts)

    print("Sync cycle complete.")

    evicted = enforce_storage_cap(max_synced_chunks=MAX_SYNCED_CHUNKS)
    if evicted:
        print(f"Storage cap reached — evicted {evicted} oldest synced chunk(s) to stay under {MAX_SYNCED_CHUNKS}.")

    print("Resuming offline monitoring.\n")


def main():
    print("UNPLLM sync watcher started.")
    print(f"Checking connectivity every {CHECK_INTERVAL_SECONDS}s. Ctrl+C to stop.\n")

    was_online = is_online()
    print(f"Initial status: {'ONLINE' if was_online else 'OFFLINE'}")

    while True:
        time.sleep(CHECK_INTERVAL_SECONDS)
        now_online = is_online()

        if now_online and not was_online:
            print(f"[{datetime.now().strftime('%H:%M:%S')}] Connectivity restored.")
            run_sync_cycle()
        elif (not now_online) and was_online:
            print(f"[{datetime.now().strftime('%H:%M:%S')}] Connection lost — offline mode.")

        was_online = now_online


if __name__ == "__main__":
    main()
