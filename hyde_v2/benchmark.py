"""
benchmark.py — HyDE v2: Full NDCG@10 / Recall@10 / Latency benchmark

Evaluates the HyDE IVF-PQ retriever against MS MARCO dev qrels.

Metrics produced:
  - NDCG@10    (primary ranking quality metric)
  - Recall@10  (fraction of relevant docs found in top-10)
  - MRR@10     (mean reciprocal rank)
  - p50/p95/p99 end-to-end latency (ms)
  - Index memory footprint comparison (IVF-PQ vs Flat)

Usage:
    python hyde_v2/benchmark.py --num_queries 500 --top_k 10

Outputs:
    hyde_v2/results/benchmark_results.json   — all metrics
    hyde_v2/results/per_query_results.jsonl  — per-query detail
"""

import argparse
import json
import os
import time
import numpy as np
from collections import defaultdict
from tqdm import tqdm

# pytrec_eval for NDCG/MRR calculation
try:
    import pytrec_eval
    _PYTREC_AVAILABLE = True
except ImportError:
    _PYTREC_AVAILABLE = False
    print("[warn] pytrec_eval not installed — install with: pip install pytrec-eval-terrier")
    print("[warn] Falling back to manual NDCG/Recall computation")

from datasets import load_dataset

INDEX_DIR = os.path.join(os.path.dirname(__file__), "index")
RESULTS_DIR = os.path.join(os.path.dirname(__file__), "results")


# ── Manual metrics (fallback if pytrec_eval unavailable) ─────────────────────
def dcg_at_k(relevances: list[int], k: int) -> float:
    relevances = relevances[:k]
    return sum(rel / np.log2(i + 2) for i, rel in enumerate(relevances))


def ndcg_at_k(retrieved_ids: list[str], relevant_ids: set[str], k: int) -> float:
    relevances = [1 if pid in relevant_ids else 0 for pid in retrieved_ids[:k]]
    ideal = sorted(relevances, reverse=True)
    actual_dcg = dcg_at_k(relevances, k)
    ideal_dcg = dcg_at_k(ideal, k)
    return actual_dcg / ideal_dcg if ideal_dcg > 0 else 0.0


def recall_at_k(retrieved_ids: list[str], relevant_ids: set[str], k: int) -> float:
    if not relevant_ids:
        return 0.0
    hits = sum(1 for pid in retrieved_ids[:k] if pid in relevant_ids)
    return hits / len(relevant_ids)


def mrr_at_k(retrieved_ids: list[str], relevant_ids: set[str], k: int) -> float:
    for i, pid in enumerate(retrieved_ids[:k]):
        if pid in relevant_ids:
            return 1.0 / (i + 1)
    return 0.0


# ── Load MS MARCO dev queries + qrels ─────────────────────────────────────────
def load_msmarco_eval(num_queries: int):
    """Returns (queries, qrels) where qrels = {qid: {pid: relevance}}"""
    print(f"[data] Loading MS MARCO dev queries (up to {num_queries}) ...")
    ds = load_dataset("ms_marco", "v2.1", split="validation", streaming=True, trust_remote_code=True)

    queries = {}
    qrels = {}
    for example in ds:
        qid = str(example["query_id"])
        query = example["query"]
        # Build qrels from passage annotations
        rel_passages = {}
        for i, (text, is_selected) in enumerate(zip(
            example["passages"]["passage_text"],
            example["passages"]["is_selected"]
        )):
            pid = f"{qid}_{i}"
            if is_selected == 1:
                rel_passages[pid] = 1
        if rel_passages:  # only include queries with at least one relevant passage
            queries[qid] = query
            qrels[qid] = rel_passages
        if len(queries) >= num_queries:
            break

    print(f"[data] Loaded {len(queries):,} queries with relevant passages")
    return queries, qrels


# ── Benchmark runner ──────────────────────────────────────────────────────────
def run_benchmark(args):
    os.makedirs(RESULTS_DIR, exist_ok=True)

    # Import retriever here to avoid circular deps
    import sys
    sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
    from hyde_v2.hyde_retriever import HyDERetriever, RetrieverConfig

    # Load queries + qrels
    queries, qrels = load_msmarco_eval(args.num_queries)
    query_ids = list(queries.keys())

    # Init retriever
    config = RetrieverConfig(top_k=args.top_k, nprobe=args.nprobe)
    retriever = HyDERetriever(config)

    # Run retrieval
    print(f"\n[bench] Running {len(query_ids):,} queries  top_k={args.top_k}  nprobe={args.nprobe} ...")
    per_query = []
    latencies = []

    ndcg_scores, recall_scores, mrr_scores = [], [], []

    for qid in tqdm(query_ids, desc="querying"):
        query_text = queries[qid]
        relevant = set(qrels[qid].keys())

        result = retriever.retrieve(query_text, top_k=args.top_k)
        retrieved_ids = [p["id"] for p in result.passages]
        lat = result.latency_ms["total_ms"]
        latencies.append(lat)

        nd = ndcg_at_k(retrieved_ids, relevant, args.top_k)
        rc = recall_at_k(retrieved_ids, relevant, args.top_k)
        mr = mrr_at_k(retrieved_ids, relevant, args.top_k)

        ndcg_scores.append(nd)
        recall_scores.append(rc)
        mrr_scores.append(mr)

        per_query.append({
            "qid": qid,
            "query": query_text,
            "ndcg@10": round(nd, 4),
            "recall@10": round(rc, 4),
            "mrr@10": round(mr, 4),
            "latency_ms": result.latency_ms,
            "hypothesis": result.hypothesis[:200],
        })

    # Latency percentiles
    latencies = np.array(latencies)
    lat_stats = {
        "p50_ms": round(float(np.percentile(latencies, 50)), 1),
        "p95_ms": round(float(np.percentile(latencies, 95)), 1),
        "p99_ms": round(float(np.percentile(latencies, 99)), 1),
        "mean_ms": round(float(np.mean(latencies)), 1),
    }

    # Index memory comparison
    ivfpq_path = os.path.join(INDEX_DIR, "ivfpq.index")
    flat_path = os.path.join(INDEX_DIR, "flat.index")
    ivfpq_mb = os.path.getsize(ivfpq_path) / 1e6 if os.path.exists(ivfpq_path) else None
    flat_mb = os.path.getsize(flat_path) / 1e6 if os.path.exists(flat_path) else None
    mem_reduction = round((1 - ivfpq_mb / flat_mb) * 100, 1) if (ivfpq_mb and flat_mb) else None

    # Load build stats if available
    build_stats = {}
    stats_path = os.path.join(INDEX_DIR, "build_stats.json")
    if os.path.exists(stats_path):
        with open(stats_path) as f:
            build_stats = json.load(f)

    # Summary
    summary = {
        "num_queries": len(query_ids),
        "top_k": args.top_k,
        "nprobe": args.nprobe,
        "metrics": {
            "ndcg@10": round(float(np.mean(ndcg_scores)), 4),
            "recall@10": round(float(np.mean(recall_scores)), 4),
            "mrr@10": round(float(np.mean(mrr_scores)), 4),
        },
        "latency": lat_stats,
        "index_memory_mb": {
            "ivfpq": round(ivfpq_mb, 1) if ivfpq_mb else None,
            "flat_baseline": round(flat_mb, 1) if flat_mb else None,
            "reduction_pct": mem_reduction,
        },
        "index_config": build_stats.get("ivfpq", {}),
        "encoder": "facebook/contriever",
        "corpus_size": build_stats.get("num_passages"),
    }

    # Save outputs
    summary_path = os.path.join(RESULTS_DIR, "benchmark_results.json")
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)

    per_query_path = os.path.join(RESULTS_DIR, "per_query_results.jsonl")
    with open(per_query_path, "w") as f:
        for r in per_query:
            f.write(json.dumps(r) + "\n")

    # Print
    print("\n" + "=" * 60)
    print("BENCHMARK RESULTS")
    print("=" * 60)
    print(f"  Queries evaluated:   {len(query_ids):,}")
    print(f"  Corpus size:         {build_stats.get('num_passages', '?'):,}")
    print(f"  NDCG@{args.top_k}:            {summary['metrics']['ndcg@10']:.4f}")
    print(f"  Recall@{args.top_k}:          {summary['metrics']['recall@10']:.4f}")
    print(f"  MRR@{args.top_k}:             {summary['metrics']['mrr@10']:.4f}")
    print(f"  Latency p50:         {lat_stats['p50_ms']} ms")
    print(f"  Latency p95:         {lat_stats['p95_ms']} ms")
    print(f"  Latency p99:         {lat_stats['p99_ms']} ms")
    if ivfpq_mb and flat_mb:
        print(f"  IVF-PQ index size:   {ivfpq_mb:.1f} MB  (vs {flat_mb:.1f} MB flat)")
        print(f"  Memory reduction:    {mem_reduction}%")
    print(f"  Results saved →      {summary_path}")
    print("=" * 60)

    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--num_queries", type=int, default=500,
                        help="Number of MS MARCO dev queries to evaluate")
    parser.add_argument("--top_k", type=int, default=10)
    parser.add_argument("--nprobe", type=int, default=64,
                        help="IVF cells to probe at search time")
    args = parser.parse_args()
    run_benchmark(args)
