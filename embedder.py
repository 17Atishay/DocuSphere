# embedder.py
# Converts text chunks into numerical vectors using HuggingFace sentence-transformers
# Model: all-MiniLM-L6-v2 (lightweight, fast, 384 dimensions)

from sentence_transformers import SentenceTransformer
from typing import List
import streamlit as st


# Load model once globally so it doesn't reload on every function call
# This model is free, runs locally, no API key needed
MODEL_NAME = "all-MiniLM-L6-v2"
@st.cache_resource   # Loads model ONCE and keeps it in memory
def load_model():
    return SentenceTransformer(MODEL_NAME)

model = load_model()

def get_embeddings(texts: List[str]) -> List[List[float]]:
    """
    Convert a list of text chunks into embeddings.
    Returns list of vectors, each with 384 dimensions.
    """
    embeddings = model.encode(texts, show_progress_bar=True)
    return embeddings.tolist()  # Convert numpy array to plain Python list


def get_single_embedding(text: str) -> List[float]:
    """
    Convert a single text (like a user query) into an embedding.
    Used at query time when user asks a question.
    """
    embedding = model.encode(text)
    return embedding.tolist()