"""
build_index.py — HyDE v2: Build IVF-PQ FAISS index over MS MARCO passages

Pipeline:
  1. Download MS MARCO passage corpus (100k subset via IR-Datasets)
  2. Encode with facebook/contriever (mean pooling, L2-norm)
  3. Train IVF-PQ index on a random training sample
  4. Add all passage vectors to the index
  5. Save index + passage store to disk

Usage:
    python hyde_v2/build_index.py --num_passages 100000 --nlist 1024 --m 64 --nbits 8

Outputs (written to hyde_v2/index/):
    passages.jsonl      — id, text for each passage
    ivfpq.index         — trained FAISS IVF-PQ index
    flat.index          — FlatIP baseline for memory/recall comparison
    build_stats.json    — timing, memory footprint, config
"""

import argparse
import json
import os
import time
import resource
import numpy as np
import faiss
import torch
from transformers import AutoTokenizer, AutoModel
from datasets import load_dataset
from tqdm import tqdm

# ── Config ────────────────────────────────────────────────────────────────────
ENCODER_MODEL = "facebook/contriever"
INDEX_DIR = os.path.join(os.path.dirname(__file__), "index")
BATCH_SIZE = 256
EMBED_DIM = 768
TRAIN_SAMPLE_SIZE = 50_000  # IVF-PQ training sample


# ── Encoder ───────────────────────────────────────────────────────────────────
def mean_pool(token_embeddings: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
    mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    return (token_embeddings * mask_expanded).sum(1) / mask_expanded.sum(1).clamp(min=1e-9)


class Contriever:
    def __init__(self, device: str = "cuda" if torch.cuda.is_available() else "cpu"):
        print(f"[encoder] Loading {ENCODER_MODEL} on {device} ...")
        self.device = device
        self.tokenizer = AutoTokenizer.from_pretrained(ENCODER_MODEL)
        self.model = AutoModel.from_pretrained(ENCODER_MODEL).to(device).eval()

    @torch.inference_mode()
    def encode(self, texts: list[str], normalize: bool = True) -> np.ndarray:
        inputs = self.tokenizer(
            texts, padding=True, truncation=True, max_length=256, return_tensors="pt"
        ).to(self.device)
        out = self.model(**inputs)
        vecs = mean_pool(out.last_hidden_state, inputs["attention_mask"])
        if normalize:
            vecs = torch.nn.functional.normalize(vecs, dim=-1)
        return vecs.cpu().float().numpy()


# ── Data ──────────────────────────────────────────────────────────────────────
def load_msmarco(num_passages: int) -> list[dict]:
    """Load MS MARCO passage corpus via HuggingFace datasets."""
    print(f"[data] Loading MS MARCO passages (first {num_passages:,}) ...")
    ds = load_dataset(
        "ms_marco", "v2.1", split="train", streaming=True, trust_remote_code=True
    )
    passages = []
    seen_ids = set()
    for example in ds:
        for i, passage in enumerate(example["passages"]["passage_text"]):
            pid = f"{example['query_id']}_{i}"
            if pid not in seen_ids:
                passages.append({"id": pid, "text": passage.strip()})
                seen_ids.add(pid)
            if len(passages) >= num_passages:
                break
        if len(passages) >= num_passages:
            break
    print(f"[data] Loaded {len(passages):,} passages")
    return passages


# ── Index builder ─────────────────────────────────────────────────────────────
def build_ivfpq(
    all_vecs: np.ndarray,
    nlist: int,
    m: int,
    nbits: int,
) -> faiss.Index:
    """Train and populate an IVF-PQ index."""
    print(f"[index] Building IVF-PQ  nlist={nlist}  m={m}  nbits={nbits} ...")
    quantizer = faiss.IndexFlatIP(EMBED_DIM)
    index = faiss.IndexIVFPQ(quantizer, EMBED_DIM, nlist, m, nbits)
    index.metric_type = faiss.METRIC_INNER_PRODUCT

    # Train on a random sample
    train_n = min(TRAIN_SAMPLE_SIZE, len(all_vecs))
    train_idx = np.random.choice(len(all_vecs), train_n, replace=False)
    train_vecs = np.ascontiguousarray(all_vecs[train_idx])
    print(f"[index] Training on {train_n:,} vectors ...")
    t0 = time.perf_counter()
    index.train(train_vecs)
    print(f"[index] Training done in {time.perf_counter()-t0:.1f}s")

    # Add all
    print(f"[index] Adding {len(all_vecs):,} vectors ...")
    t0 = time.perf_counter()
    index.add(np.ascontiguousarray(all_vecs))
    print(f"[index] Adding done in {time.perf_counter()-t0:.1f}s")
    return index


def build_flat(all_vecs: np.ndarray) -> faiss.Index:
    print("[index] Building FlatIP baseline ...")
    index = faiss.IndexFlatIP(EMBED_DIM)
    index.add(np.ascontiguousarray(all_vecs))
    return index


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--num_passages", type=int, default=100_000)
    parser.add_argument("--nlist", type=int, default=1024,
                        help="Number of IVF centroids")
    parser.add_argument("--m", type=int, default=64,
                        help="Number of PQ sub-quantizers (dim must be divisible by m)")
    parser.add_argument("--nbits", type=int, default=8,
                        help="Bits per sub-quantizer code (4 or 8)")
    args = parser.parse_args()

    assert EMBED_DIM % args.m == 0, f"dim {EMBED_DIM} must be divisible by m={args.m}"
    os.makedirs(INDEX_DIR, exist_ok=True)

    total_start = time.perf_counter()

    # 1. Load data
    passages = load_msmarco(args.num_passages)
    texts = [p["text"] for p in passages]

    # Save passage store
    passages_path = os.path.join(INDEX_DIR, "passages.jsonl")
    with open(passages_path, "w") as f:
        for p in passages:
            f.write(json.dumps(p) + "\n")
    print(f"[data] Saved {len(passages):,} passages → {passages_path}")

    # 2. Encode
    encoder = Contriever()
    all_vecs = []
    print(f"[encode] Encoding {len(texts):,} passages in batches of {BATCH_SIZE} ...")
    encode_start = time.perf_counter()
    for i in tqdm(range(0, len(texts), BATCH_SIZE), desc="encoding"):
        batch = texts[i : i + BATCH_SIZE]
        all_vecs.append(encoder.encode(batch))
    all_vecs = np.vstack(all_vecs).astype("float32")
    encode_time = time.perf_counter() - encode_start
    print(f"[encode] Done in {encode_time:.1f}s  shape={all_vecs.shape}")

    # 3. Build IVF-PQ
    ivfpq_index = build_ivfpq(all_vecs, args.nlist, args.m, args.nbits)
    ivfpq_path = os.path.join(INDEX_DIR, "ivfpq.index")
    faiss.write_index(ivfpq_index, ivfpq_path)

    # 4. Build flat baseline
    flat_index = build_flat(all_vecs)
    flat_path = os.path.join(INDEX_DIR, "flat.index")
    faiss.write_index(flat_index, flat_path)

    # 5. Memory comparison
    ivfpq_mb = os.path.getsize(ivfpq_path) / 1e6
    flat_mb = os.path.getsize(flat_path) / 1e6
    reduction_pct = (1 - ivfpq_mb / flat_mb) * 100

    total_time = time.perf_counter() - total_start

    stats = {
        "num_passages": len(passages),
        "embed_dim": EMBED_DIM,
        "encoder": ENCODER_MODEL,
        "ivfpq": {"nlist": args.nlist, "m": args.m, "nbits": args.nbits},
        "index_size_mb": {"ivfpq": round(ivfpq_mb, 1), "flat": round(flat_mb, 1)},
        "memory_reduction_pct": round(reduction_pct, 1),
        "encode_time_s": round(encode_time, 1),
        "total_time_s": round(total_time, 1),
    }

    stats_path = os.path.join(INDEX_DIR, "build_stats.json")
    with open(stats_path, "w") as f:
        json.dump(stats, f, indent=2)

    print("\n" + "=" * 60)
    print("BUILD COMPLETE")
    print("=" * 60)
    print(f"  Passages:          {len(passages):,}")
    print(f"  Flat index size:   {flat_mb:.1f} MB")
    print(f"  IVF-PQ index size: {ivfpq_mb:.1f} MB")
    print(f"  Memory reduction:  {reduction_pct:.1f}%")
    print(f"  Total time:        {total_time:.1f}s")
    print(f"  Stats saved →      {stats_path}")
    print("=" * 60)


if __name__ == "__main__":
    main()
