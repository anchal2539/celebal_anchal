"""
rag_pipeline.py
----------------
Combines context retrieval (FAISS) with answer generation (Google Gemini)
to produce a final, grounded answer to the user's question.
"""

import os

import google.generativeai as genai

from modules.embedding_store import retrieve_context

GEMINI_MODEL_NAME = "gemini-2.5-flash"

PROMPT_TEMPLATE = """You are a helpful assistant answering questions using ONLY the context provided below, which was retrieved from the user's document.

- If the answer is in the context, answer clearly and concisely.
- If the context does not contain enough information to answer, say so honestly instead of guessing.
- Do not use outside knowledge beyond what's in the context.

Context:
{context}

Question: {question}

Answer:"""


def configure_gemini(api_key: str = None):
    """Configure the Gemini client with an API key (env var or passed in)."""
    api_key = api_key or os.getenv("GOOGLE_API_KEY")
    if not api_key:
        raise ValueError(
            "No Gemini API key found. Set GOOGLE_API_KEY in your .env file, "
            "or pass it directly to configure_gemini()."
        )
    genai.configure(api_key=api_key)


def generate_answer(question: str, retrieved_chunks: list[tuple]) -> str:
    """
    Build a prompt from the retrieved context and call Gemini to generate
    the final natural-language answer.
    """
    if not retrieved_chunks:
        return "I couldn't find relevant information in the document to answer that question."

    context = "\n\n---\n\n".join(chunk for chunk, _ in retrieved_chunks)
    prompt = PROMPT_TEMPLATE.format(context=context, question=question)

    model = genai.GenerativeModel(GEMINI_MODEL_NAME)
    response = model.generate_content(prompt)
    return response.text.strip()


def answer_question(vector_store, question: str, top_k: int = 4) -> dict:
    """
    End-to-end: retrieve relevant chunks for the question, then generate
    a grounded answer using Gemini.
    """
    retrieved = retrieve_context(vector_store, question, top_k=top_k)
    answer = generate_answer(question, retrieved)
    return {
        "question": question,
        "answer": answer,
        "sources": retrieved,
    }