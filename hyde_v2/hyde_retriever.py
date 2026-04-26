"""
hyde_retriever.py — HyDE v2: Zero-Shot Dense Retrieval with IVF-PQ

Pipeline per query:
  1. Generate a hypothetical passage that would answer the query (via LLM)
  2. Encode the hypothetical passage with contriever (same encoder as corpus)
  3. Search IVF-PQ index for nearest neighbours
  4. Return ranked passage list with scores and latency breakdown

Usage:
    from hyde_v2.hyde_retriever import HyDERetriever
    retriever = HyDERetriever()
    results = retriever.retrieve("What is RLHF?", top_k=10)
"""

import json
import os
import time
from dataclasses import dataclass, field

import faiss
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModel

# Optional: OpenAI for hypothesis generation
try:
    from openai import OpenAI
    _OPENAI_AVAILABLE = True
except ImportError:
    _OPENAI_AVAILABLE = False

INDEX_DIR = os.path.join(os.path.dirname(__file__), "index")
ENCODER_MODEL = "facebook/contriever"
EMBED_DIM = 768


# ── Data classes ──────────────────────────────────────────────────────────────
@dataclass
class RetrievalResult:
    query: str
    hypothesis: str
    passages: list[dict]          # [{id, text, score}, ...]
    latency_ms: dict              # {hypothesis_ms, encode_ms, search_ms, total_ms}


@dataclass
class RetrieverConfig:
    top_k: int = 10
    nprobe: int = 64              # IVF cells to visit at search time (speed/recall tradeoff)
    openai_model: str = "gpt-3.5-turbo"
    hypothesis_max_tokens: int = 128
    use_flat_baseline: bool = False   # set True to compare with exact search


# ── Encoder ───────────────────────────────────────────────────────────────────
def _mean_pool(token_emb: torch.Tensor, attn_mask: torch.Tensor) -> torch.Tensor:
    mask = attn_mask.unsqueeze(-1).expand(token_emb.size()).float()
    return (token_emb * mask).sum(1) / mask.sum(1).clamp(min=1e-9)


class _Encoder:
    def __init__(self, device: str):
        self.device = device
        self.tokenizer = AutoTokenizer.from_pretrained(ENCODER_MODEL)
        self.model = AutoModel.from_pretrained(ENCODER_MODEL).to(device).eval()

    @torch.inference_mode()
    def encode(self, texts: list[str]) -> np.ndarray:
        inputs = self.tokenizer(
            texts, padding=True, truncation=True, max_length=256, return_tensors="pt"
        ).to(self.device)
        out = self.model(**inputs)
        vecs = _mean_pool(out.last_hidden_state, inputs["attention_mask"])
        vecs = torch.nn.functional.normalize(vecs, dim=-1)
        return vecs.cpu().float().numpy()


# ── Hypothesis generator ──────────────────────────────────────────────────────
class _HypothesisGenerator:
    SYSTEM = (
        "You are a retrieval assistant. Given a search query, write a single short "
        "passage (2-4 sentences) that would directly answer the query. "
        "Write only the passage, no preamble."
    )

    def __init__(self, openai_model: str):
        self.openai_model = openai_model
        if _OPENAI_AVAILABLE and os.environ.get("OPENAI_API_KEY"):
            self.client = OpenAI()
            self.use_openai = True
            print("[generator] Using OpenAI for hypothesis generation")
        else:
            self.client = None
            self.use_openai = False
            print("[generator] No OpenAI key found — using deterministic fallback")

    def generate(self, query: str, max_tokens: int = 128) -> str:
        if self.use_openai:
            resp = self.client.chat.completions.create(
                model=self.openai_model,
                messages=[
                    {"role": "system", "content": self.SYSTEM},
                    {"role": "user", "content": query},
                ],
                max_tokens=max_tokens,
                temperature=0.7,
            )
            return resp.choices[0].message.content.strip()
        else:
            # Deterministic fallback: expand query into a simple declarative passage
            return (
                f"This passage provides information about {query.lower().rstrip('?')}. "
                f"The topic involves key concepts and detailed explanations relevant to "
                f"the question of {query.lower().rstrip('?')}."
            )


# ── Main retriever ────────────────────────────────────────────────────────────
class HyDERetriever:
    def __init__(self, config: RetrieverConfig | None = None):
        self.config = config or RetrieverConfig()
        device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"[retriever] Initialising on {device} ...")

        # Load encoder
        self.encoder = _Encoder(device)

        # Load index
        index_name = "flat.index" if self.config.use_flat_baseline else "ivfpq.index"
        index_path = os.path.join(INDEX_DIR, index_name)
        if not os.path.exists(index_path):
            raise FileNotFoundError(
                f"Index not found: {index_path}\n"
                "Run `python hyde_v2/build_index.py` first."
            )
        self.index = faiss.read_index(index_path)
        if not self.config.use_flat_baseline:
            self.index.nprobe = self.config.nprobe
        print(f"[retriever] Loaded index ({index_name})  ntotal={self.index.ntotal:,}")

        # Load passage store
        passages_path = os.path.join(INDEX_DIR, "passages.jsonl")
        self.passages = []
        with open(passages_path) as f:
            for line in f:
                self.passages.append(json.loads(line))
        print(f"[retriever] Loaded {len(self.passages):,} passages")

        # Load hypothesis generator
        self.generator = _HypothesisGenerator(self.config.openai_model)

    def retrieve(self, query: str, top_k: int | None = None) -> RetrievalResult:
        k = top_k or self.config.top_k

        # 1. Generate hypothesis
        t0 = time.perf_counter()
        hypothesis = self.generator.generate(query, self.config.hypothesis_max_tokens)
        hyp_ms = (time.perf_counter() - t0) * 1000

        # 2. Encode hypothesis
        t1 = time.perf_counter()
        vec = self.encoder.encode([hypothesis])          # (1, 768)
        enc_ms = (time.perf_counter() - t1) * 1000

        # 3. Search
        t2 = time.perf_counter()
        scores, indices = self.index.search(vec, k)
        search_ms = (time.perf_counter() - t2) * 1000

        total_ms = hyp_ms + enc_ms + search_ms

        # 4. Collect results
        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx == -1:
                continue
            p = self.passages[idx]
            results.append({"id": p["id"], "text": p["text"], "score": float(score)})

        return RetrievalResult(
            query=query,
            hypothesis=hypothesis,
            passages=results,
            latency_ms={
                "hypothesis_ms": round(hyp_ms, 2),
                "encode_ms": round(enc_ms, 2),
                "search_ms": round(search_ms, 2),
                "total_ms": round(total_ms, 2),
            },
        )

    def retrieve_batch(self, queries: list[str], top_k: int | None = None) -> list[RetrievalResult]:
        return [self.retrieve(q, top_k) for q in queries]
