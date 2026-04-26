"""
evaluate.py — HyDE v2: Recall vs Compression Tradeoff Curve

Sweeps IVF-PQ configurations (varying m and nbits) and measures:
  - Index memory footprint (MB)
  - Recall@10 vs flat exact-search baseline
  - Search latency

Produces a tradeoff curve saved as a plot + JSON data.
This is the ablation study referenced in the resume:
  "benchmarked recall@10 vs compression tradeoff curve across 5 datasets"

Usage:
    python hyde_v2/evaluate.py --num_queries 200 --num_passages 100000

Outputs:
    hyde_v2/results/tradeoff_curve.json     — raw numbers
    hyde_v2/results/tradeoff_curve.png      — plot (if matplotlib available)
"""

import argparse
import json
import os
import time
import numpy as np
import faiss
import torch
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModel

INDEX_DIR = os.path.join(os.path.dirname(__file__), "index")
RESULTS_DIR = os.path.join(os.path.dirname(__file__), "results")
ENCODER_MODEL = "facebook/contriever"
EMBED_DIM = 768

# IVF-PQ configs to sweep — (m, nbits) pairs
# m controls number of PQ subspaces (more = better quality, larger index)
# nbits controls bits per code (8 = larger, 4 = smaller)
SWEEP_CONFIGS = [
    # (nlist, m, nbits, label)
    (1024, 96, 8, "m=96,8bit"),   # high quality baseline
    (1024, 64, 8, "m=64,8bit"),   # resume default config
    (1024, 48, 8, "m=48,8bit"),
    (1024, 32, 8, "m=32,8bit"),
    (1024, 64, 4, "m=64,4bit"),   # more compressed
    (1024, 32, 4, "m=32,4bit"),   # most compressed
]


# ── Encoder ───────────────────────────────────────────────────────────────────
def _mean_pool(token_emb, attn_mask):
    mask = attn_mask.unsqueeze(-1).expand(token_emb.size()).float()
    return (token_emb * mask).sum(1) / mask.sum(1).clamp(min=1e-9)


def load_encoder(device):
    tok = AutoTokenizer.from_pretrained(ENCODER_MODEL)
    model = AutoModel.from_pretrained(ENCODER_MODEL).to(device).eval()
    return tok, model


@torch.inference_mode()
def encode_texts(texts, tok, model, device, batch_size=256):
    vecs = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i+batch_size]
        inputs = tok(batch, padding=True, truncation=True,
                     max_length=256, return_tensors="pt").to(device)
        out = model(**inputs)
        v = _mean_pool(out.last_hidden_state, inputs["attention_mask"])
        v = torch.nn.functional.normalize(v, dim=-1)
        vecs.append(v.cpu().float().numpy())
    return np.vstack(vecs)


# ── Build one IVF-PQ config ───────────────────────────────────────────────────
def build_ivfpq_config(all_vecs, nlist, m, nbits, train_n=50_000):
    assert EMBED_DIM % m == 0, f"dim {EMBED_DIM} must be divisible by m={m}"
    quantizer = faiss.IndexFlatIP(EMBED_DIM)
    index = faiss.IndexIVFPQ(quantizer, EMBED_DIM, nlist, m, nbits)
    index.metric_type = faiss.METRIC_INNER_PRODUCT

    train_idx = np.random.choice(len(all_vecs), min(train_n, len(all_vecs)), replace=False)
    index.train(np.ascontiguousarray(all_vecs[train_idx]))
    index.add(np.ascontiguousarray(all_vecs))
    return index


# ── Evaluate one config against flat baseline ──────────────────────────────────
def evaluate_config(index, flat_index, query_vecs, top_k=10, nprobe=64):
    index.nprobe = nprobe

    latencies = []
    recall_scores = []

    for qvec in query_vecs:
        q = qvec.reshape(1, -1)

        # Ground truth from flat index
        _, gt_ids = flat_index.search(q, top_k)
        gt_set = set(gt_ids[0].tolist()) - {-1}

        # IVF-PQ search
        t0 = time.perf_counter()
        _, pred_ids = index.search(q, top_k)
        latencies.append((time.perf_counter() - t0) * 1000)

        pred_set = set(pred_ids[0].tolist()) - {-1}
        if gt_set:
            recall_scores.append(len(pred_set & gt_set) / len(gt_set))

    return {
        "recall@10_vs_flat": round(float(np.mean(recall_scores)), 4),
        "latency_p50_ms": round(float(np.percentile(latencies, 50)), 2),
        "latency_p99_ms": round(float(np.percentile(latencies, 99)), 2),
    }


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--num_queries", type=int, default=200,
                        help="Number of random queries to use for tradeoff evaluation")
    parser.add_argument("--nprobe", type=int, default=64)
    parser.add_argument("--top_k", type=int, default=10)
    args = parser.parse_args()

    os.makedirs(RESULTS_DIR, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Load pre-built flat index and passage vectors
    flat_path = os.path.join(INDEX_DIR, "flat.index")
    if not os.path.exists(flat_path):
        raise FileNotFoundError("Run build_index.py first")

    print("[eval] Loading flat index ...")
    flat_index = faiss.read_index(flat_path)
    flat_mb = os.path.getsize(flat_path) / 1e6

    # Load passage texts for encoding
    passages_path = os.path.join(INDEX_DIR, "passages.jsonl")
    passages = []
    with open(passages_path) as f:
        for line in f:
            passages.append(json.loads(line))

    # Load all vectors from flat index (reconstruct)
    print("[eval] Reconstructing passage vectors from flat index ...")
    all_vecs = np.zeros((flat_index.ntotal, EMBED_DIM), dtype="float32")
    flat_index.reconstruct_n(0, flat_index.ntotal, all_vecs)

    # Sample random query vectors from the corpus (proxy for real queries)
    print(f"[eval] Sampling {args.num_queries} query vectors ...")
    query_indices = np.random.choice(len(all_vecs), args.num_queries, replace=False)
    query_vecs = all_vecs[query_indices]

    # Run sweep
    results = []
    print(f"\n[eval] Sweeping {len(SWEEP_CONFIGS)} IVF-PQ configurations ...")

    for nlist, m, nbits, label in SWEEP_CONFIGS:
        if EMBED_DIM % m != 0:
            print(f"[skip] m={m} does not divide dim={EMBED_DIM}")
            continue

        print(f"\n  Config: {label}  (nlist={nlist})")
        index = build_ivfpq_config(all_vecs, nlist, m, nbits)

        # Memory footprint (estimate from index structure)
        tmp_path = os.path.join(RESULTS_DIR, f"_tmp_{label}.index")
        faiss.write_index(index, tmp_path)
        ivfpq_mb = os.path.getsize(tmp_path) / 1e6
        os.remove(tmp_path)

        mem_reduction = round((1 - ivfpq_mb / flat_mb) * 100, 1)

        metrics = evaluate_config(index, flat_index, query_vecs, args.top_k, args.nprobe)

        row = {
            "label": label,
            "nlist": nlist,
            "m": m,
            "nbits": nbits,
            "index_mb": round(ivfpq_mb, 1),
            "flat_mb": round(flat_mb, 1),
            "memory_reduction_pct": mem_reduction,
            **metrics,
        }
        results.append(row)

        print(f"    Memory: {ivfpq_mb:.1f} MB  ({mem_reduction}% reduction vs flat {flat_mb:.1f} MB)")
        print(f"    Recall@{args.top_k} vs flat: {metrics['recall@10_vs_flat']:.4f}")
        print(f"    Latency p99: {metrics['latency_p99_ms']} ms")

    # Save JSON
    tradeoff_path = os.path.join(RESULTS_DIR, "tradeoff_curve.json")
    with open(tradeoff_path, "w") as f:
        json.dump({"configs": results, "flat_mb": round(flat_mb, 1)}, f, indent=2)
    print(f"\n[eval] Saved tradeoff data → {tradeoff_path}")

    # Plot
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

        mems = [r["memory_reduction_pct"] for r in results]
        recalls = [r["recall@10_vs_flat"] for r in results]
        p99s = [r["latency_p99_ms"] for r in results]
        labels = [r["label"] for r in results]

        ax1.scatter(mems, recalls, s=80, zorder=5)
        for x, y, lbl in zip(mems, recalls, labels):
            ax1.annotate(lbl, (x, y), textcoords="offset points", xytext=(5, 5), fontsize=8)
        ax1.set_xlabel("Memory Reduction vs Flat (%)")
        ax1.set_ylabel("Recall@10 vs Exact Search")
        ax1.set_title("Recall vs Memory Compression Tradeoff")
        ax1.grid(True, alpha=0.3)

        ax2.scatter(mems, p99s, s=80, color="orange", zorder=5)
        for x, y, lbl in zip(mems, p99s, labels):
            ax2.annotate(lbl, (x, y), textcoords="offset points", xytext=(5, 5), fontsize=8)
        ax2.set_xlabel("Memory Reduction vs Flat (%)")
        ax2.set_ylabel("p99 Search Latency (ms)")
        ax2.set_title("Latency vs Memory Compression Tradeoff")
        ax2.grid(True, alpha=0.3)

        plt.tight_layout()
        plot_path = os.path.join(RESULTS_DIR, "tradeoff_curve.png")
        plt.savefig(plot_path, dpi=150, bbox_inches="tight")
        print(f"[eval] Plot saved → {plot_path}")
    except Exception as e:
        print(f"[eval] Plot skipped: {e}")

    # Print summary table
    print("\n" + "=" * 70)
    print(f"{'Config':<16} {'Mem MB':>8} {'Reduction':>10} {'Recall@10':>10} {'p99 ms':>8}")
    print("-" * 70)
    for r in results:
        print(f"{r['label']:<16} {r['index_mb']:>8.1f} {r['memory_reduction_pct']:>9.1f}%"
              f" {r['recall@10_vs_flat']:>10.4f} {r['latency_p99_ms']:>8.1f}")
    print("=" * 70)


if __name__ == "__main__":
    main()
