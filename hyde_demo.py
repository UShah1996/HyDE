#!/usr/bin/env python3
"""
Minimal HyDE (Zero-Shot Dense Retrieval) CLI Demo
Generate → Embed → Retrieve
"""

import json
import os
import sys
import time
from typing import List, Tuple, Dict, Any
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModel
import faiss
from openai import OpenAI


class HyDEDemo:
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

    def _build_faiss_index(self) -> faiss.Index:
        """Build FAISS index for cosine similarity search"""
        dimension = self.embeddings.shape[1]
        index = faiss.IndexFlatIP(dimension)  # Inner product for cosine similarity
        index.add(self.embeddings)
        return index

    def _generate_hypothetical_passage(self, query: str) -> str:
        """Generate hypothetical passage using LLM or fallback"""
        if self.openai_client:
            try:
                response = self.openai_client.chat.completions.create(
                    model="gpt-3.5-turbo",
                    messages=[
                        {"role": "system", "content": "Write concise, factual passages that answer questions."},
                        {"role": "user", "content": f"Write a concise passage (3-5 sentences) that answers the question.\n\nQuestion: {query}\n\nPassage:"}
                    ],
                    max_tokens=200,
                    temperature=0.5
                )
                return response.choices[0].message.content.strip()
            except Exception as e:
                print(f"LLM generation failed: {e}")
                print("Falling back to deterministic generator...")
        
        # Fallback: deterministic passage based on query
        query_lower = query.lower()
        if any(word in query_lower for word in ["covid", "coronavirus", "pandemic", "vaccine"]):
            return f"COVID-19 is a respiratory disease caused by the SARS-CoV-2 virus. The pandemic began in 2019 and affected millions worldwide. Vaccines were developed to prevent severe illness and reduce transmission. Public health measures like social distancing and mask-wearing helped control the spread."
        elif any(word in query_lower for word in ["machine learning", "ml", "ai", "neural", "model"]):
            return f"Machine learning is a branch of artificial intelligence that enables computers to learn from data. It uses algorithms to identify patterns and make predictions without explicit programming. Deep learning uses neural networks with multiple layers to process complex information. Applications include image recognition, natural language processing, and predictive analytics."
        elif any(word in query_lower for word in ["finance", "investment", "money", "stock", "retirement"]):
            return f"Personal finance involves managing money, investments, and financial planning. Key concepts include compound interest, diversification, and emergency funds. Retirement planning requires long-term savings through accounts like 401(k)s and IRAs. Credit scores affect loan terms and interest rates."
        else:
            return f"This topic relates to {query}. The subject involves various aspects and considerations that are important to understand. Research and analysis can provide valuable insights into this area. Different perspectives and approaches may offer unique solutions and understanding."

    def search(self, query: str, top_k: int = 5) -> List[Tuple[int, float, str]]:
        """HyDE search: generate hypothetical passage, embed it, search FAISS"""
        # Generate hypothetical passage
        start_time = time.time()
        hypothetical = self._generate_hypothetical_passage(query)
        gen_time = time.time() - start_time
        
        # Embed hypothetical passage
        start_time = time.time()
        hypo_embedding = self._embed_texts([hypothetical])
        embed_time = time.time() - start_time
        
        # Search FAISS
        start_time = time.time()
        scores, indices = self.index.search(hypo_embedding, top_k)
        search_time = time.time() - start_time
        
        # Format results
        results = []
        for i, (score, idx) in enumerate(zip(scores[0], indices[0])):
            if idx < len(self.corpus):
                results.append((i + 1, float(score), self.corpus[idx]["text"]))
        
        return results, hypothetical, gen_time, embed_time, search_time

    def run_cli(self):
        """Main CLI loop"""
        try:
            while True:
                query = input("\nEnter your query (or Ctrl+C to exit): ").strip()
                if not query:
                    continue
                
                print(f"\nQuery: {query}")
                print("=" * 60)
                
                results, hypothetical, gen_time, embed_time, search_time = self.search(query)
                
                print(f"Hypothetical Passage ({gen_time:.2f}s):")
                print(f"  {hypothetical}")
                print(f"\nTop Results (embed: {embed_time:.3f}s, search: {search_time:.3f}s):")
                print("-" * 40)
                
                for rank, score, text in results:
                    print(f"{rank}. Score: {score:.4f}")
                    print(f"   {text[:150]}{'...' if len(text) > 150 else ''}")
                    print()
                
        except KeyboardInterrupt:
            print("\n\nExiting...")
        except EOFError:
            print("\n\nExiting...")


def main():
    if len(sys.argv) > 1:
        corpus_path = sys.argv[1]
    else:
        corpus_path = "data/corpus.jsonl"
    
    if not os.path.exists(corpus_path):
        print(f"Error: Corpus file not found: {corpus_path}")
        sys.exit(1)
    
    demo = HyDEDemo(corpus_path)
    demo.run_cli()


if __name__ == "__main__":
    main()
