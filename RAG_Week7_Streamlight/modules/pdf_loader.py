"""
pdf_loader.py
-------------
Handles document ingestion: extracts raw text from an uploaded PDF (or .txt
file) so it can be passed on to the text splitter.
"""

from pypdf import PdfReader


def load_pdf(file) -> str:
    """
    Extract raw text from a PDF.

    `file` can be a file path (str) or a file-like object (e.g. what
    Streamlit's `st.file_uploader` returns), so this works both in the CLI
    and in the web app.
    """
    reader = PdfReader(file)
    text_parts = []
    for page in reader.pages:
        page_text = page.extract_text() or ""
        text_parts.append(page_text)
    return "\n".join(text_parts)


def load_text_file(file) -> str:
    """Read a plain .txt file (path or file-like object)."""
    if hasattr(file, "read"):
        content = file.read()
        return content.decode("utf-8") if isinstance(content, bytes) else content
    with open(file, "r", encoding="utf-8") as f:
        return f.read()


def load_document(file, filename: str = None) -> str:
    """
    Dispatch to the right loader based on file extension.
    `filename` is required when `file` is a file-like object without a
    `.name` attribute (Streamlit uploads have `.name` already).
    """
    name = filename or getattr(file, "name", None) or (file if isinstance(file, str) else "")
    if name.lower().endswith(".pdf"):
        return load_pdf(file)
    return load_text_file(file)
