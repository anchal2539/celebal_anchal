# Document Question Answering System (RAG)

A Retrieval-Augmented Generation (RAG) app that answers questions based on
uploaded PDF/text documents. It retrieves the most relevant sections from
your document and uses **Google Gemini** to generate accurate, grounded
answers.

## Architecture

```
Upload PDF/TXT
      │
      ▼
Extract Text (pypdf)
      │
      ▼
Split into Chunks (LangChain RecursiveCharacterTextSplitter)
      │
      ▼
Generate Embeddings (HuggingFace all-MiniLM-L6-v2, local)
      │
      ▼
Store in FAISS Vector Database
      │
──────────────────────────────
      │
User asks a Question
      │
      ▼
Convert Question to Embedding
      │
      ▼
Retrieve Top-k Similar Chunks (FAISS)
      │
      ▼
Build Prompt with Context + Question
      │
      ▼
Send to Google Gemini API
      │
      ▼
Display Answer + Retrieved Context
```

## Project Structure

```
├── app.py                    # Main Streamlit application
├── requirements.txt          # Python dependencies
├── .env.example              # Template for your API key
├── data/                     # Sample doc + cached FAISS indexes
├── modules/
│   ├── pdf_loader.py         # PDF/text extraction
│   ├── text_splitter.py      # Chunking logic (LangChain)
│   ├── embedding_store.py    # Embeddings + FAISS vector store
│   └── rag_pipeline.py       # Retrieval + Gemini generation
└── utils/
    └── helper.py             # Timing, stats helpers
```

## Setup

### 1. Install dependencies
```bash
pip install -r requirements.txt
```

### 2. Get a free Gemini API key
Go to [aistudio.google.com/app/apikey](https://aistudio.google.com/app/apikey),
sign in, and create a free API key.

### 3. Add your key
```bash
cp .env.example .env
```
Then open `.env` and paste your key:
```
GOOGLE_API_KEY=your_actual_api_key_here
```

### 4. Run the app
```bash
streamlit run app.py
```
It opens at `http://localhost:8501`.

## How to Use

1. **Upload** a PDF or `.txt` file in the sidebar (a sample doc is in `data/sample_document.txt`).
2. Click **Process Document** — extracts text, chunks it, embeds it, and builds the FAISS index.
3. **Ask a question** in the chat box.
4. Expand **Retrieved Context** under the answer to see exactly which chunks were used.

## Notes on first run

- The first time you process a document, `sentence-transformers` downloads
  the `all-MiniLM-L6-v2` model (~90MB) from Hugging Face. This needs
  internet access once; after that it's cached locally.
- FAISS indexes are cached in `data/faiss_cache/` by document content hash,
  so re-processing the same file is instant on subsequent runs.
- Embeddings run **locally** (no API key needed for that step) — only the
  final answer generation calls the Gemini API.

## Technologies Used

| Technology | Purpose |
|---|---|
| Streamlit | Web interface |
| LangChain | Text splitting and retrieval pipeline |
| FAISS | Local vector database for similarity search |
| HuggingFace (`all-MiniLM-L6-v2`) | Local embedding generation |
| Google Gemini | LLM for answer generation |
| pypdf | PDF text extraction |

## Key Learnings (for your report)

- How retrieval and generation combine to ground LLM answers in real,
  private data instead of relying on the model's internal knowledge.
- Why chunk size/overlap matters: too small loses context, too large dilutes
  retrieval relevance.
- How embeddings capture semantic meaning, enabling similarity search beyond
  simple keyword matching.
- How a vector database (FAISS) enables fast retrieval even over large
  document collections.
- This same architecture (retrieve → augment → generate) powers production
  chatbots, enterprise search, and AI documentation assistants.

