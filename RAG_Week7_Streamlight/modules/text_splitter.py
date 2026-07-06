"""
text_splitter.py
-----------------
Handles text chunking using LangChain's RecursiveCharacterTextSplitter,
which splits on paragraph/sentence/word boundaries (in that priority order)
so chunks stay semantically coherent instead of cutting words in half.
"""

from langchain.text_splitter import RecursiveCharacterTextSplitter


def split_text(text: str, chunk_size: int = 800, chunk_overlap: int = 150) -> list[str]:
    """
    Split raw document text into overlapping chunks.

    chunk_size: target number of characters per chunk
    chunk_overlap: how many characters of overlap between consecutive chunks
                   (preserves context across chunk boundaries)
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    chunks = splitter.split_text(text)
    return [c.strip() for c in chunks if c.strip()]
