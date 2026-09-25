"""
UNPLLM - Stage 3: Gap analysis (reporting)
Prints the current state of unresolved knowledge gaps: which topics
are being asked about that the local knowledge base doesn't cover
well yet. Topics the sync engine has already fetched won't show up
here again (they're marked "resolved").

Run any time you want a status check:
    python3 analyze_gaps.py
"""

from gap_engine import load_unresolved_gaps, cluster_gaps


def main():
    gap_queries = load_unresolved_gaps()

    if not gap_queries:
        print("No unresolved knowledge gaps right now.")
        return

    clusters = cluster_gaps(gap_queries)

    print(f"Found {len(gap_queries)} gap queries across {len(clusters)} distinct topic(s).\n")
    print(f"{'Count':<6} {'Example query':<50} {'Avg distance'}")
    print("-" * 75)
    for cluster in clusters:
        count = len(cluster["queries"])
        example = cluster["queries"][0]["text"][:47] + "..."
        valid_distances = [q["distance"] for q in cluster["queries"] if q["distance"] is not None]
        avg_dist = sum(valid_distances) / len(valid_distances) if valid_distances else 0
        print(f"{count:<6} {example:<50} {avg_dist:.4f}")


if __name__ == "__main__":
    main()
