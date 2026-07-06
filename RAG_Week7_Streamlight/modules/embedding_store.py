"""
embedding_store.py
-------------------
Handles embedding creation and the FAISS vector database.

Uses HuggingFace's all-MiniLM-L6-v2 (via sentence-transformers) to create
real semantic embeddings locally -- no API key needed for this step, but it
does need internet access the first time to download the model (~90MB) from
huggingface.co, then it's cached locally for all future runs.
"""

import hashlib
import os

# Force transformers/sentence-transformers to use PyTorch only. Without this,
# if TensorFlow (with Keras 3) is also installed, transformers tries to load
# the TF backend and fails with a Keras-3-not-supported error.
os.environ["USE_TF"] = "0"
os.environ["USE_TORCH"] = "1"
os.environ["TRANSFORMERS_NO_ADVISORY_WARNINGS"] = "1"

from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"
CACHE_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "faiss_cache")


def get_embedding_model() -> HuggingFaceEmbeddings:
    """Load the local sentence-transformers embedding model."""
    return HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL_NAME)


def _hash_chunks(chunks: list[str]) -> str:
    """Create a stable cache key from the chunk contents."""
    joined = "\n".join(chunks).encode("utf-8")
    return hashlib.sha256(joined).hexdigest()[:16]


def build_vector_store(chunks: list[str], embedding_model: HuggingFaceEmbeddings = None,
                        use_cache: bool = True) -> FAISS:
    """
    Embed each chunk and store the vectors in a FAISS index.
    If a cached index exists for these exact chunks, load it instead of
    re-embedding (saves time on repeated runs with the same document).
    """
    embedding_model = embedding_model or get_embedding_model()

    if use_cache:
        os.makedirs(CACHE_DIR, exist_ok=True)
        cache_key = _hash_chunks(chunks)
        cache_path = os.path.join(CACHE_DIR, cache_key)
        if os.path.exists(cache_path):
            return FAISS.load_local(cache_path, embedding_model, allow_dangerous_deserialization=True)

    vector_store = FAISS.from_texts(chunks, embedding_model)

    if use_cache:
        vector_store.save_local(cache_path)

    return vector_store


def retrieve_context(vector_store: FAISS, query: str, top_k: int = 4):
    """
    Convert the query to an embedding and retrieve the top_k most similar
    chunks, along with their similarity scores.
    """
    results = vector_store.similarity_search_with_score(query, k=top_k)
    return [(doc.page_content, float(score)) for doc, score in results]
