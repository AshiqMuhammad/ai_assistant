# Simple Streamlit AI Document Assistant

A beginner-friendly document Q&A application built with **Streamlit + Sentence Transformers + FAISS + Groq**.

It supports:

- PDF
- DOCX
- TXT
- Markdown (`.md`)
- Local file uploads
- Public/shared Google Drive file or folder links
- Text extraction
- Overlapping text chunking
- Sentence Transformer embeddings
- FAISS semantic search
- Simple keyword search
- Hybrid semantic + keyword ranking
- Groq-powered question answering
- Retrieved source display
- Filename and page metadata where available
- Streamlit session-state reuse so embeddings are not recreated for every question

## 1. Project structure

```text
ai_document_assistant/
│
├── app.py
├── requirements.txt
└── readme.md
```

## 2. Install Python

Python 3.10 or newer is recommended.

## 3. Create a virtual environment

### Windows

```bash
python -m venv venv
venv\Scripts\activate
```

### macOS / Linux

```bash
python3 -m venv venv
source venv/bin/activate
```

## 4. Install dependencies

```bash
pip install -r requirements.txt
```

## 5. Add the Groq API key

Create this file:

```text
.streamlit/secrets.toml
```

Put your key in it:

```toml
GROQ_API_KEY = "your_groq_api_key_here"
```

The key is **not** hardcoded in `app.py`.

Do not commit `secrets.toml` to GitHub.

## 6. Run the app

```bash
streamlit run app.py
```

The application will open in your browser.

## 7. How the pipeline works

The application follows this pipeline:

```text
PDF / DOCX / TXT / MD
        ↓
Text extraction
        ↓
Overlapping chunks
        ↓
Sentence Transformer embeddings
        ↓
FAISS index
        ↓
User question
        ↓
Question embedding
        ↓
Semantic search
        +
Keyword search
        ↓
Hybrid ranking
        ↓
Top document chunks
        ↓
Groq
        ↓
Answer + retrieved sources
```

## 8. Why embeddings are not created for every question

When documents are loaded:

1. Text is extracted.
2. Text is split into chunks.
3. Every chunk gets one embedding.
4. The embeddings are stored in Streamlit session state.
5. A FAISS index is created.

When the user asks another question:

1. Only the question is embedded.
2. The existing document embeddings are reused.
3. The existing FAISS index is reused.
4. Keyword scores are calculated.
5. Semantic and keyword scores are combined.
6. The best chunks are sent to Groq.

So the application does **not** re-embed all documents for every question.

## 9. Chunking

The example uses:

```python
CHUNK_SIZE = 700
CHUNK_OVERLAP = 120
```

This means each chunk is approximately 700 characters and neighboring chunks overlap by approximately 120 characters.

Overlap helps preserve context when an important sentence falls near a chunk boundary.

## 10. Metadata

Each chunk keeps:

```python
{
    "filename": "...",
    "page": 1,
    "text": "..."
}
```

For PDF files, page numbers are available.

For DOCX, TXT and MD files, page numbers are normally unavailable from simple text extraction, so the application uses:

```text
Page not available
```

## 11. Hybrid search

The application combines two signals:

### Semantic search

Sentence Transformers convert text into vectors.

FAISS finds chunks that are semantically similar to the question.

### Keyword search

Important words from the question are extracted and matched against each chunk.

### Combined score

The example uses:

```text
Hybrid score =
    70% semantic score
  + 30% keyword score
```

This gives the search both meaning-based matching and exact-word matching.

## 12. Google Drive

The Google Drive input uses `gdown`.

Public/shared Drive files or folders can be loaded when they are accessible without private authentication.

Supported downloaded files are:

```text
.pdf
.docx
.txt
.md
```

Private Drive files may require an authenticated Google Drive API integration. This simple version intentionally keeps the Drive implementation lightweight.

## 13. Important note about persistence

`st.session_state` keeps processed documents, embeddings and the FAISS index during the current Streamlit session.

If the app/server restarts, the in-memory index is lost and documents must be processed again.

For a production application, the next step would be persistent storage such as:

- FAISS index saved to disk
- Metadata saved as JSON/SQLite
- Document hashing to detect unchanged files
- A persistent vector database

This simple version avoids that extra complexity so the code remains easy to understand.

## 14. Security

Never write the Groq API key directly inside `app.py`.

Use:

```toml
GROQ_API_KEY = "..."
```

inside:

```text
.streamlit/secrets.toml
```

Also add the following to `.gitignore`:

```text
.streamlit/secrets.toml
venv/
__pycache__/
```
