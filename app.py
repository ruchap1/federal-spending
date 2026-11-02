"""Streamlit UI for the Federal Spending & Budget Research Assistant."""

from __future__ import annotations

import streamlit as st

from src.embeddings import Embedder
from src.generator import LocalGenerator
from src.hybrid import HybridPipeline
from src.reranker import Reranker


st.set_page_config(page_title="Federal Spending Research Assistant", page_icon="💰")
st.title("💰 Federal Spending & Budget Research Assistant")
st.caption("Uses document retrieval for explanations and USAspending.gov for exact fiscal-year values.")


@st.cache_resource(show_spinner="Loading embedding model...")
def load_embedder() -> Embedder:
    return Embedder()


@st.cache_resource(show_spinner="Loading local answer model...")
def load_generator() -> LocalGenerator:
    return LocalGenerator()


@st.cache_resource(show_spinner="Loading reranker...")
def load_reranker() -> Reranker:
    return Reranker()


@st.cache_resource
def load_pipeline() -> HybridPipeline:
    return HybridPipeline(load_embedder(), load_generator(), reranker=load_reranker())


with st.sidebar:
    st.header("Structured spending lookup")
    st.caption("Fill these in for exact agency/fiscal-year spending questions.")
    agency_query = st.text_input("Agency name", placeholder="Department of Education")
    fiscal_year = st.number_input("Fiscal year", min_value=2000, max_value=2100, value=2024)

question = st.text_area(
    "Ask a federal spending question",
    placeholder="How much did the Department of Education obligate in fiscal year 2024?",
    height=100,
)

if st.button("Research", type="primary", disabled=not question.strip()):
    with st.spinner("Retrieving evidence and preparing a grounded answer..."):
        response = load_pipeline().answer(question, agency_query, int(fiscal_year))

    st.caption(f"Query mode: `{response.mode}`")
    st.subheader("Answer")
    st.markdown(response.answer)

    if response.results:
        st.subheader("Retrieved sources")
        for rank, result in enumerate(response.results, 1):
            st.markdown(
                f"**[{rank}] {result.metadata.get('source', 'unknown')}** — "
                f"{result.metadata.get('section', 'unknown')} "
                f"(score: {result.score:.4f})"
            )
        with st.expander("Show retrieved evidence"):
            for rank, result in enumerate(response.results, 1):
                st.markdown(f"#### [{rank}] {result.metadata.get('section', 'Unknown')}")
                st.write(result.text)

