#!/usr/bin/env python3
"""
Minimal HyDE (Zero-Shot Dense Retrieval) Streamlit UI
Generate → Embed → Retrieve
"""

import os

import streamlit as st

from hyde_v2.hyde_v2_core import HyDEDemoV2


@st.cache_resource
def load_hyde_system(corpus_path: str = "data/corpus.jsonl"):
    return HyDEDemoV2(corpus_path=corpus_path)


def main():
    st.set_page_config(
        page_title="HyDE Demo V2",
        page_icon="🔍",
        layout="wide"
    )

    st.title("🔍 HyDE Demo V2: Zero-Shot Dense Retrieval with most recent relevant information")
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

    st.markdown("""
            <style>
            /* Fix the query input and search button at the bottom */
            .bottom-container {
                position: fixed;
                bottom: 2rem;
                left: 1rem;
                right: 1rem;
                background-color: #0e1117; /* Matches Streamlit dark mode */
                padding: 1rem;
                border-radius: 10px;
                box-shadow: 0px 0px 15px rgba(0,0,0,0.3);
                z-index: 1000;
            }

            /* Style text input */
            .bottom-container .stTextInput > div > div {
                width: 100%;
            }

            /* Remove top margin from bottom container */
            .stApp {
                padding-bottom: 100px;  /* Prevent overlap with bottom container */
            }
            </style>
            """, unsafe_allow_html=True)


    st.markdown('<div class="bottom-container">', unsafe_allow_html=True)
    # Query input
    col1, col2 = st.columns([4, 1])

    with col1:
        # Inject custom CSS to fix the text input at the bottom

        query = st.text_input(
            "Enter your query:",
            placeholder="e.g., What are the symptoms of COVID-19?",
            help="Ask any question about COVID-19, machine learning, finance or anything else"
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
            results, hypothetical = hyde_system.search(query, top_k)

        # Store in session state so it persists across reruns
        st.session_state.results = results
        st.session_state.hypothetical = hypothetical
        st.session_state.query = query
        st.session_state.confidence_score = hyde_system.calculate_hyde_passage_confidence_score(
            query=query,
            hypothetical_passage=hypothetical
        )
        st.session_state.corpus_action_taken = False


    # Results table
    if "results" in st.session_state:
        results = st.session_state.results
        hypothetical = st.session_state.hypothetical
        confidence_score = st.session_state.confidence_score

        # Display results
        if results:
            st.markdown("### 📝 Hypothetical Passage")
            with st.expander("Show/Hide Hypothetical Passage", expanded=True):
                st.write(hypothetical)

            st.markdown("### 🎯 Search Results")

            for rank, score, text in results:
                with st.container():
                    st.markdown(f"**{rank}. Score: {score:.4f}**")
                    st.write(text)
                    st.markdown("---")
        else:

            st.write(
                f"But here is the information found from external sources: \n\n"
                f"{hypothetical}\n\n"
                f"**Confidence Score:** {confidence_score:.3f}"
            )

            # --- Proper use of session state ---
            if "corpus_action_taken" not in st.session_state:
                st.session_state.corpus_action_taken = False
                st.session_state.corpus_action_result = None

            # Only show buttons if a choice hasn't been made yet
            if not st.session_state.corpus_action_taken:
                col1, col2, col3 = st.columns([1, 1, 5], gap=None)

                with col1:
                    yes_clicked = st.button("✅ Yes", key="yes_button")
                with col2:
                    no_clicked = st.button("❌ No", key="no_button")

                if yes_clicked:
                    added = hyde_system.add_passage_to_corpus(
                        hypothetical_passage=hypothetical,
                        confidence_score=round(confidence_score, 3)
                    )
                    st.session_state.corpus_action_taken = True
                    st.session_state.corpus_action_result = added

                elif no_clicked:
                    st.session_state.corpus_action_taken = True
                    st.session_state.corpus_action_result = False

            # After choice has been made, show result
            if st.session_state.corpus_action_taken:
                if st.session_state.corpus_action_result:
                    st.success("✅ Added to knowledge base.")
                else:
                    st.info("ℹ️ Not added to knowledge base.")

                # Clear only relevant session state keys (recommended)
                keys_to_clear = ["results", "hypothetical", "query", "confidence_score", "corpus_action_taken",
                                 "corpus_action_result"]
                for key in keys_to_clear:
                    if key in st.session_state:
                        del st.session_state[key]

                # Force app to rerun and reset interface
                st.rerun()

            # if to_add or to_add.lower() == "y":
            #     added = hyde_system.add_passage_to_corpus(hypothetical_passage=hypothetical,
            #                                               confidence_score=round(confidence_score, 3))
            #     if added:
            #         st.write("Added the new information to the knowledge base.")
            #     else:
            #         st.write("Failed to add new information to the knowledge base.")
            #
            # else:
            #     st.write("Skipped writing to corpus")



        # # Performance stats
        # st.markdown("### ⚡ Performance")
        # col1, col2, col3 = st.columns(3)
        # with col1:
        #     st.metric("Generation", f"{gen_time:.2f}s")
        # with col2:
        #     st.metric("Embedding", f"{embed_time:.3f}s")
        # with col3:
        #     st.metric("Search", f"{search_time:.3f}s")
        #
        # total_time = gen_time + embed_time + search_time
        # st.caption(f"Total time: {total_time:.2f}s")

    # Sidebar info
    with st.sidebar:
        st.markdown("### ℹ️ About HyDE V2")
        st.markdown("""
        **Hypothetical Document Embeddings (HyDE)** is a zero-shot dense retrieval method that:
        
        1. **Fetches** latest information from the web
        2. **Generates** a hypothetical passage answering the query
        3. **Embeds** it using the same encoder as the corpus
        4. **Retrieves** similar passages via cosine similarity
        5. **Capable** of saving the hypothetical passage to the corpus, if user finds it relevant
        
        **Key Features:**
        - NEW
            - Web Scraping before asking ChatGPT for hypothetical message
            - Keyword matching and sentence transformer
            - Confidence calculation for the hypothetical passage
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
