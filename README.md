# Minimal HyDE Demo: Zero-Shot Dense Retrieval

**Generate → Embed → Retrieve**

A tiny implementation of Hypothetical Document Embeddings (HyDE) for zero-shot dense retrieval. For any natural-language query, the system generates a hypothetical passage using an LLM, embeds it with the same unsupervised contrastive encoder as the corpus (facebook/contriever), searches a FAISS index, and returns top-k passages.

## What is HyDE?

HyDE is a zero-shot dense retrieval method that improves search by first generating a hypothetical document that answers the query, then using that hypothetical document to find similar passages in the corpus. This approach leverages the semantic understanding of large language models while maintaining the efficiency of dense retrieval.

**Key principles:**
- Same encoder model for corpus and hypothetical passage (facebook/contriever)
- L2-normalized embeddings for cosine similarity search
- No training or relevance labels required
- CPU-friendly implementation

## Setup & Run

### Prerequisites

```bash
pip install -r requirements.txt
```

### Optional: OpenAI API Key

For LLM-generated hypothetical passages, set your OpenAI API key:

```bash
export OPENAI_API_KEY="your-api-key-here"
```

If no API key is provided, the system uses a deterministic fallback generator.

### CLI Demo

```bash
python hyde_demo.py
```

Enter queries interactively. Example queries:
- "What are the symptoms of COVID-19?"
- "How does machine learning work?"
- "What is compound interest?"

### Streamlit UI

```bash
streamlit run app.py
```

Open your browser to the displayed URL (typically http://localhost:8501).

Features:
- Query input with top-k slider (3-15 results)
- Collapsible hypothetical passage display
- Ranked results with similarity scores
- Performance timing metrics
- Sidebar with technical details

## Notes & Limits

**Technical Implementation:**
- Uses facebook/contriever for consistent embedding space
- Mean pooling over token embeddings with attention mask
- FAISS IndexFlatIP for inner product (cosine similarity)
- L2 normalization ensures cosine ≈ inner product

**Corpus:**
- Small demo corpus (15 documents) across 3 topics: COVID-19, Machine Learning, Finance
- Each document is a short passage (1-3 sentences)
- JSONL format: `{"id": "doc_id", "text": "document text"}`

**Performance:**
- Designed for CPU execution
- Typical query processing: <10 seconds end-to-end
- Memory footprint: ~500MB (model + embeddings + index)

**Limitations:**
- No re-ranking or BM25 hybrid approaches
- No chunking pipeline (single document embeddings)
- Deterministic fallback when LLM unavailable
- Small corpus for demonstration purposes

**Fallback Behavior:**
When no OpenAI API key is available, the system generates topic-specific passages based on keyword matching in the query. This ensures the demo works offline while still demonstrating the HyDE retrieval pipeline.

## File Structure

```
├── requirements.txt          # Minimal dependencies
├── README.md                # This file
├── data/
│   └── corpus.jsonl        # Demo corpus (15 documents)
├── hyde_demo.py            # CLI entry point
└── app.py                  # Streamlit UI
```

## Success Criteria

✅ CLI and Streamlit both return plausible results for queries across different topics  
✅ README explains the method and run steps clearly  
✅ Works on typical laptop CPU with default settings  
✅ Same encoder used for corpus and hypothetical passages  
✅ Embeddings are L2-normalized for cosine similarity  
✅ Graceful fallback when LLM unavailable  
✅ Minimal memory footprint and fast iteration
