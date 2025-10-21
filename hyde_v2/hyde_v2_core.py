import json
import os
import re
import sys
import time
import urllib.parse
import uuid
from typing import List, Tuple, Dict, Any, Optional, NamedTuple
import numpy as np
import torch
from ddgs import DDGS
from transformers import AutoTokenizer, AutoModel
import faiss
import requests
from bs4 import BeautifulSoup
from openai import OpenAI
from sentence_transformers import SentenceTransformer, util as st_util

from hyde_v2.web_scraper import ScrapeWeb


class HyDEDemoV2:
    SEMANTIC_SEARCH_THRESHOLD: float = 0.70

    def __init__(self, corpus_path: str = "data/corpus.jsonl"):
        self.corpus_path = corpus_path
        self.device = "cpu"
        self.model_name = "facebook/contriever"

        # Initialize encoder
        print(f"Loading encoder: {self.model_name}")
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.model = AutoModel.from_pretrained(self.model_name)
        self.model.eval()

        # Initialize OpenAI client (optional)
        self.openai_client = None
        if os.getenv("OPENAI_API_KEY"):
            self.openai_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
            print("OpenAI API key found - will use LLM for hypothetical passages")
        else:
            print("No OpenAI API key found - using fallback generator")

        # Load corpus and build index
        self.corpus = self._load_corpus()
        self.embeddings = self._compute_embeddings()
        self.sentence_embedder = self._get_embedder()
        self.index = self._build_faiss_index()

        print(f"Loaded {len(self.corpus)} documents")
        print("FAISS index built successfully")
        print("Ready for queries! (Ctrl+C to exit)")
        print("-" * 50)

    def _load_corpus(self) -> List[Dict[str, Any]]:
        """Load corpus from JSONL file"""
        corpus = []
        with open(self.corpus_path, 'r', encoding='utf-8') as f:
            for line in f:
                corpus.append(json.loads(line.strip()))
        return corpus

    def _mean_pooling(self, token_embeddings: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        """Mean pooling over token embeddings using attention mask"""
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)

    def _embed_texts(self, texts: List[str]) -> np.ndarray:
        """Embed a batch of texts and return L2-normalized embeddings"""
        with torch.no_grad():
            # Tokenize
            encoded = self.tokenizer(texts, padding=True, truncation=True,
                                     max_length=512, return_tensors="pt")

            # Get embeddings
            model_output = self.model(**encoded)

            # Mean pooling
            embeddings = self._mean_pooling(model_output.last_hidden_state,
                                            encoded['attention_mask'])

            # Convert to numpy and L2 normalize
            embeddings = embeddings.numpy().astype(np.float32)
            embeddings = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)

            return embeddings

    def _compute_embeddings(self) -> np.ndarray:
        """Compute embeddings for all corpus texts"""
        texts = [doc["text"] for doc in self.corpus]
        return self._embed_texts(texts)

    @staticmethod
    def _get_embedder() -> SentenceTransformer:
        return SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

    def _build_faiss_index(self) -> faiss.Index:
        """Build FAISS index for cosine similarity search"""
        dimension = self.embeddings.shape[1]
        index = faiss.IndexFlatIP(dimension)  # Inner product for cosine similarity
        index.add(self.embeddings)
        return index

    # Web search
    @staticmethod
    def _scrape_web_for_recent_information(query: str, max_results: int = 3) -> List[str]:
        scrape_web: ScrapeWeb = ScrapeWeb(max_results=max_results)
        scraped_web_results: List[Dict[str, str]] = scrape_web.search(query=query)
        scraped_result_list: List[str] = []
        for each_result in scraped_web_results:
            if each_result and each_result["text"] is not None:
                scraped_result_list.append(each_result["text"][:1500])  # extracting only first 1500 characters

        return scraped_result_list

    def _filter_results(self, query: str, scraped_results: List[str]) -> List[str]:
        if not scraped_results:
            return []

        query_tokens = re.findall(r"[A-Za-z0-9\-]+", query.lower())
        query_keywords = [t for t in query_tokens if len(query_tokens) >= 1]

        if query_keywords:
            prefiltered_results = [result for result in scraped_results if
                                   any(k in result.lower() for k in query_keywords)]
            scraped_results = prefiltered_results or scraped_results

        embedder: SentenceTransformer = self.sentence_embedder
        query_embeddings = embedder.encode(query, convert_to_tensor=True, normalize_embeddings=True)
        result_embeddings = embedder.encode(scraped_results, convert_to_tensor=True, normalize_embeddings=True)
        similarity_score = st_util.cos_sim(query_embeddings, result_embeddings)[0]

        scored = [(float(s.item()), p) for s, p in zip(similarity_score, scraped_results)]
        keep = [sp for sp in scored if sp[0] >= self.SEMANTIC_SEARCH_THRESHOLD] or scored
        keep.sort(key=lambda x: x[0], reverse=True)
        return [p for _, p in keep[:4]]

    def _generate_hypothetical_passage(self, query: str) -> str:
        """Generate hypothetical passage using LLM or fallback"""
        print("🔍 Web augmentation: searching & scraping…")
        scraped_results = self._scrape_web_for_recent_information(query=query, max_results=3)
        if not scraped_results:
            print("⚠ No web data found — using fallback generator.")

        relevant = self._filter_results(query=query, scraped_results=scraped_results)

        context = " ".join(relevant)
        if len(context) > 2000:
            context = context[:2000]

        if self.openai_client:

            try:
                response = self.openai_client.chat.completions.create(
                    model="gpt-3.5-turbo",
                    messages=[
                        {
                            "role": "system",
                            "content":
                                "Write concise, factual passages that answer questions. "
                                "Your task is as follows:"
                                "- You can use the information in the context."
                                "- You can use this context and any information other available to you, if any and relevant. If you do, then you can summarize using your own information and the given context"
                                "- Synthesize in your own words; do not copy verbatim."
                                "- Keep it factual and neutral."
                        },
                        {
                            "role": "user",
                            "content": f"This is the user query: {query} and "
                                       f"context based on latest information: {context}."
                                       "Instructions: Your response should be in the following format:\n"
                                       "According to {source}, {response text}."
                                       "If you just use the given user context, the source will be \'GivenContext\' "
                                       "else the source will be \'All to the information provided\'"
                        }
                    ],
                    max_tokens=200,
                    temperature=0.5
                )
                return response.choices[0].message.content.strip()

            except Exception as e:
                print(f"LLM generation failed: {e}")

    def _calculate_hyde_passage_confidence_score(self, query: str, hypothetical_passage: str) -> float:
        # Embed query and hypothetical
        q_emb = self._embed_texts([query])[0]
        h_emb = self._embed_texts([hypothetical_passage])[0]

        confidence = float(np.dot(q_emb, h_emb))
        return confidence

    def calculate_hyde_passage_confidence_score(self, query: str, hypothetical_passage: str) -> float:
        return self._calculate_hyde_passage_confidence_score(query=query, hypothetical_passage=hypothetical_passage)

    def _add_passage_to_corpus(self, hypothetical_passage: str, confidence_score: float):
        print("✅ Confident hypothetical — adding to corpus & FAISS index")
        # 1) Append to corpus.jsonl
        record = {
            "id": f"hyde_{uuid.uuid4()}",
            "text": hypothetical_passage,
            "source": "hyde_web_augmented",
            "meta": {
                "confidence": confidence_score,
            },
        }
        try:
            with open(self.corpus_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(record, ensure_ascii=False) + "\n")
        except Exception as e:
            print(f"⚠ Failed to append to corpus file: {e}")
            return False

        # 2) Update in-memory corpus and FAISS
        self.corpus.append(record)
        h_emb = self._embed_texts([hypothetical_passage])[0]
        h_emb_v = h_emb.reshape(1, -1).astype(np.float32)
        self.index.add(h_emb_v)
        return True

    def add_passage_to_corpus(self, hypothetical_passage: str, confidence_score: float):
        print("Adding passage to corpus")
        return self._add_passage_to_corpus(hypothetical_passage=hypothetical_passage, confidence_score=confidence_score)

    def search(self, query: str, top_k: int = 5) -> Tuple[List[Tuple[int, float, str]], str]:
        hypothetical = self._generate_hypothetical_passage(query)

        hypothetical_embedding = self._embed_texts([hypothetical])

        scores, indices = self.index.search(hypothetical_embedding, top_k)

        results = []
        for i, (score, idx) in enumerate(zip(scores[0], indices[0])):
            if 0 <= idx < len(self.corpus) and score > 0.5:
                results.append((i + 1, float(score), self.corpus[idx]["text"]))

        return results, hypothetical

    def run_cli(self):
        """Main CLI loop"""
        try:
            while True:
                query = input("\nEnter your query (or Ctrl+C to exit): ").strip()
                if not query:
                    continue

                print(f"\nQuery: {query}")
                print("=" * 60)

                results, hypothetical = self.search(query)
                print("-" * 40)

                if results:
                    print(f"Hypothetical Passage:\n {hypothetical}\n")
                    for rank, score, text in results:
                        snippet = text[:150] + ("..." if len(text) > 150 else "")
                        print(f"{rank}. Score: {score:.4f}\n   {snippet}\n")
                else:
                    print(f"No valid documents found for user query={query}.")
                    confidence_score = self._calculate_hyde_passage_confidence_score(query=query,
                                                                                     hypothetical_passage=hypothetical)

                    print(f"But here is the information found from external sources: \n{hypothetical} \n"
                          f"with confidence_score={confidence_score:.3f}")

                    to_add = input("\n Do you want to add this information to your knowledge base?(Y/n): ").strip()
                    if to_add or to_add.lower() == "y":
                        added = self._add_passage_to_corpus(hypothetical_passage=hypothetical,
                                                            confidence_score=round(confidence_score, 3))
                        if added:
                            print("Added the new information to the knowledge base.")
                        else:
                            print("Failed to add new information to the knowledge base.")
                    else:
                        print("No input provided. The external information won't be added to the knowledge base")


        except KeyboardInterrupt:
            print("\n\nExiting...")
        except EOFError:
            print("\n\nExiting...")
