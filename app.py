#!/usr/bin/env python3
"""
Minimal HyDE (Zero-Shot Dense Retrieval) Streamlit UI
Generate → Embed → Retrieve
"""

import json
import os
import time
from typing import List, Tuple, Dict, Any
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModel
import faiss
import streamlit as st
from openai import OpenAI


@st.cache_resource
def load_hyde_system(corpus_path: str = "data/corpus.jsonl"):
    """Load and cache the HyDE system components"""
    
    class HyDESystem:
        def __init__(self, corpus_path: str):
            self.corpus_path = corpus_path
            self.device = "cpu"
            self.model_name = "facebook/contriever"
            
            # Initialize encoder
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            self.model = AutoModel.from_pretrained(self.model_name)
            self.model.eval()
            
            # Initialize OpenAI client (optional)
            self.openai_client = None
            if os.getenv("OPENAI_API_KEY"):
                self.openai_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
            
            # Load corpus and build index
            self.corpus = self._load_corpus()
            self.embeddings = self._compute_embeddings()
            self.index = self._build_faiss_index()

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
                    st.warning(f"LLM generation failed: {e}. Using fallback generator.")
            
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

        def search(self, query: str, top_k: int = 5) -> Tuple[List[Tuple[int, float, str]], str, float, float, float]:
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

    return HyDESystem(corpus_path)


def main():
    st.set_page_config(
        page_title="HyDE Demo",
        page_icon="🔍",
        layout="wide"
    )
    
    st.title("🔍 HyDE Demo: Zero-Shot Dense Retrieval")
    st.markdown("**Generate → Embed → Retrieve**")
    
    # Load system
    with st.spinner("Loading HyDE system..."):
        hyde_system = load_hyde_system()
    
    st.success(f"✅ Loaded {len(hyde_system.corpus)} documents")
    
    # Check for OpenAI API key
    has_openai = os.getenv("OPENAI_API_KEY") is not None
    if has_openai:
        st.info("🤖 OpenAI API key detected - using LLM for hypothetical passages")
    else:
        st.warning("⚠️ No OpenAI API key found - using fallback generator")
    
    st.markdown("---")
    
    # Query input
    col1, col2 = st.columns([3, 1])
    
    with col1:
        query = st.text_input(
            "Enter your query:",
            placeholder="e.g., What are the symptoms of COVID-19?",
            help="Ask any question about COVID-19, machine learning, or finance"
        )
    
    with col2:
        top_k = st.slider("Top-K results:", min_value=3, max_value=15, value=5)
    
    # Search button
    if st.button("🔍 Search", type="primary", disabled=not query.strip()):
        if not query.strip():
            st.error("Please enter a query")
            return
        
        # Perform search
        with st.spinner("Searching..."):
            results, hypothetical, gen_time, embed_time, search_time = hyde_system.search(query, top_k)
        
        # Display results
        st.markdown("### 📝 Hypothetical Passage")
        with st.expander("Show/Hide Hypothetical Passage", expanded=True):
            st.write(hypothetical)
            st.caption(f"Generation time: {gen_time:.2f}s")
        
        st.markdown("### 🎯 Search Results")
        
        # Results table
        for rank, score, text in results:
            with st.container():
                st.markdown(f"**{rank}. Score: {score:.4f}**")
                st.write(text)
                st.markdown("---")
        
        # Performance stats
        st.markdown("### ⚡ Performance")
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Generation", f"{gen_time:.2f}s")
        with col2:
            st.metric("Embedding", f"{embed_time:.3f}s")
        with col3:
            st.metric("Search", f"{search_time:.3f}s")
        
        total_time = gen_time + embed_time + search_time
        st.caption(f"Total time: {total_time:.2f}s")
    
    # Sidebar info
    with st.sidebar:
        st.markdown("### ℹ️ About HyDE")
        st.markdown("""
        **Hypothetical Document Embeddings (HyDE)** is a zero-shot dense retrieval method that:
        
        1. **Generates** a hypothetical passage answering the query
        2. **Embeds** it using the same encoder as the corpus
        3. **Retrieves** similar passages via cosine similarity
        
        **Key Features:**
        - Same encoder for query and corpus (facebook/contriever)
        - L2-normalized embeddings
        - No training or labels required
        - CPU-friendly implementation
        """)
        
        st.markdown("### 📊 Corpus Info")
        st.write(f"Documents: {len(hyde_system.corpus)}")
        st.write(f"Topics: COVID-19, Machine Learning, Finance")
        
        st.markdown("### 🔧 Technical Details")
        st.write(f"Encoder: {hyde_system.model_name}")
        st.write(f"Embedding dim: {hyde_system.embeddings.shape[1]}")
        st.write(f"Index type: FAISS IndexFlatIP")


if __name__ == "__main__":
    main()
