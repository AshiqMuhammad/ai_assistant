import os
import re
import tempfile
from pathlib import Path

import numpy as np
import requests
import streamlit as st
import faiss

from pypdf import PdfReader
from docx import Document
from sentence_transformers import SentenceTransformer
from groq import Groq


# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="AI Document Assistant",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ============================================================
# CUSTOM CSS
# ============================================================

st.markdown(
    """
    <style>

    .main {
        background-color: #f7f8fa;
    }

    .hero {
        padding: 28px;
        border-radius: 18px;
        background: linear-gradient(135deg, #111827, #1f2937);
        color: white;
        margin-bottom: 25px;
    }

    .hero h1 {
        margin: 0;
        font-size: 34px;
    }

    .hero p {
        margin-top: 8px;
        color: #d1d5db;
        font-size: 16px;
    }

    .card {
        background: white;
        padding: 20px;
        border-radius: 15px;
        border: 1px solid #e5e7eb;
        margin-bottom: 15px;
    }

    .source-card {
        background: white;
        padding: 16px;
        border-radius: 12px;
        border: 1px solid #e5e7eb;
        margin-bottom: 12px;
    }

    .source-title {
        font-size: 16px;
        font-weight: 700;
        margin-bottom: 6px;
    }

    .source-meta {
        font-size: 13px;
        color: #6b7280;
        margin-bottom: 8px;
    }

    .source-text {
        font-size: 14px;
        color: #374151;
        line-height: 1.6;
    }

    .answer-card {
        background: white;
        padding: 22px;
        border-radius: 15px;
        border: 1px solid #e5e7eb;
        line-height: 1.7;
    }

    .metric-card {
        background: white;
        padding: 18px;
        border-radius: 14px;
        border: 1px solid #e5e7eb;
        text-align: center;
    }

    .metric-number {
        font-size: 28px;
        font-weight: 700;
    }

    .metric-label {
        color: #6b7280;
        font-size: 13px;
    }

    </style>
    """,
    unsafe_allow_html=True,
)


# ============================================================
# SESSION STATE
# ============================================================

if "documents" not in st.session_state:
    st.session_state.documents = []

if "chunks" not in st.session_state:
    st.session_state.chunks = []

if "embeddings" not in st.session_state:
    st.session_state.embeddings = None

if "index" not in st.session_state:
    st.session_state.index = None

if "messages" not in st.session_state:
    st.session_state.messages = []

if "processed_names" not in st.session_state:
    st.session_state.processed_names = set()


# ============================================================
# CONSTANTS
# ============================================================

SUPPORTED_EXTENSIONS = {
    ".pdf",
    ".docx",
    ".txt",
    ".md",
}

CHUNK_SIZE = 700
CHUNK_OVERLAP = 120

EMBEDDING_MODEL = "all-MiniLM-L6-v2"

GROQ_MODEL = "openai/gpt-oss-120b"


# ============================================================
# LOAD EMBEDDING MODEL
# ============================================================

@st.cache_resource
def load_embedding_model():
    return SentenceTransformer(EMBEDDING_MODEL)


# ============================================================
# LOAD GROQ CLIENT
# ============================================================

@st.cache_resource
def load_groq_client(api_key):
    return Groq(api_key=api_key)


# ============================================================
# TEXT CLEANING
# ============================================================

def clean_text(text):
    if not text:
        return ""

    text = text.replace("\x00", " ")

    # Remove excessive spaces
    text = re.sub(r"[ \t]+", " ", text)

    # Remove excessive blank lines
    text = re.sub(r"\n\s*\n+", "\n\n", text)

    return text.strip()


# ============================================================
# PDF EXTRACTION
# ============================================================

def extract_pdf(file_path, filename):
    records = []

    try:
        reader = PdfReader(file_path)

        for page_number, page in enumerate(reader.pages, start=1):

            try:
                text = page.extract_text() or ""
            except Exception:
                text = ""

            text = clean_text(text)

            if text:
                records.append(
                    {
                        "text": text,
                        "filename": filename,
                        "page": page_number,
                        "source": "PDF",
                    }
                )

    except Exception as e:
        st.error(f"Could not read PDF: {e}")

    return records


# ============================================================
# DOCX EXTRACTION
# ============================================================

def extract_docx(file_path, filename):
    records = []

    try:
        doc = Document(file_path)

        paragraphs = []

        for paragraph in doc.paragraphs:
            text = paragraph.text.strip()

            if text:
                paragraphs.append(text)

        full_text = "\n".join(paragraphs)
        full_text = clean_text(full_text)

        if full_text:
            records.append(
                {
                    "text": full_text,
                    "filename": filename,
                    "page": None,
                    "source": "DOCX",
                }
            )

    except Exception as e:
        st.error(f"Could not read DOCX: {e}")

    return records


# ============================================================
# TXT EXTRACTION
# ============================================================

def extract_txt(file_path, filename):
    records = []

    try:
        with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
            text = f.read()

        text = clean_text(text)

        if text:
            records.append(
                {
                    "text": text,
                    "filename": filename,
                    "page": None,
                    "source": "TXT",
                }
            )

    except Exception as e:
        st.error(f"Could not read TXT: {e}")

    return records


# ============================================================
# MARKDOWN EXTRACTION
# ============================================================

def extract_md(file_path, filename):
    records = []

    try:
        with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
            text = f.read()

        text = clean_text(text)

        if text:
            records.append(
                {
                    "text": text,
                    "filename": filename,
                    "page": None,
                    "source": "MD",
                }
            )

    except Exception as e:
        st.error(f"Could not read Markdown: {e}")

    return records


# ============================================================
# FILE EXTRACTION ROUTER
# ============================================================

def extract_document(file_path, filename):

    extension = Path(filename).suffix.lower()

    if extension == ".pdf":
        return extract_pdf(file_path, filename)

    if extension == ".docx":
        return extract_docx(file_path, filename)

    if extension == ".txt":
        return extract_txt(file_path, filename)

    if extension == ".md":
        return extract_md(file_path, filename)

    return []


# ============================================================
# CHUNKING
# ============================================================

def create_chunks(records):

    chunks = []

    for record in records:

        text = record["text"]

        start = 0
        text_length = len(text)

        while start < text_length:

            end = min(start + CHUNK_SIZE, text_length)

            chunk_text = text[start:end].strip()

            if chunk_text:

                chunks.append(
                    {
                        "text": chunk_text,
                        "filename": record["filename"],
                        "page": record["page"],
                        "source": record["source"],
                    }
                )

            if end >= text_length:
                break

            start = end - CHUNK_OVERLAP

    return chunks


# ============================================================
# BUILD FAISS INDEX
# ============================================================

def rebuild_index():

    if not st.session_state.chunks:
        st.session_state.index = None
        st.session_state.embeddings = None
        return

    model = load_embedding_model()

    texts = [
        chunk["text"]
        for chunk in st.session_state.chunks
    ]

    embeddings = model.encode(
        texts,
        convert_to_numpy=True,
        normalize_embeddings=True,
        show_progress_bar=False,
    )

    embeddings = np.asarray(
        embeddings,
        dtype="float32",
    )

    index = faiss.IndexFlatIP(
        embeddings.shape[1]
    )

    index.add(embeddings)

    st.session_state.embeddings = embeddings
    st.session_state.index = index


# ============================================================
# PROCESS DOCUMENTS
# ============================================================

def process_records(records):

    if not records:
        return

    new_chunks = create_chunks(records)

    if not new_chunks:
        return

    st.session_state.chunks.extend(new_chunks)

    for record in records:
        if record["filename"] not in st.session_state.processed_names:
            st.session_state.documents.append(
                {
                    "filename": record["filename"],
                    "source": record["source"],
                }
            )

            st.session_state.processed_names.add(
                record["filename"]
            )

    rebuild_index()


# ============================================================
# SAVE UPLOADED FILE
# ============================================================

def save_uploaded_file(uploaded_file):

    suffix = Path(uploaded_file.name).suffix

    temp_file = tempfile.NamedTemporaryFile(
        delete=False,
        suffix=suffix,
    )

    temp_file.write(
        uploaded_file.getbuffer()
    )

    temp_file.close()

    return temp_file.name


# ============================================================
# GOOGLE DRIVE / GOOGLE DOCS ID
# ============================================================

def extract_google_id(url):

    patterns = [

        r"drive\.google\.com/file/d/([^/]+)",

        r"drive\.google\.com/open\?id=([^&]+)",

        r"drive\.google\.com/uc\?id=([^&]+)",

        r"drive\.google\.com/drive/folders/([^/?]+)",

        r"docs\.google\.com/document/d/([^/]+)",
    ]

    for pattern in patterns:

        match = re.search(
            pattern,
            url,
        )

        if match:
            return match.group(1)

    return None


# ============================================================
# GOOGLE DOC DOWNLOAD
# ============================================================

def download_google_doc(url):

    document_id = extract_google_id(url)

    if not document_id:
        raise ValueError(
            "Invalid Google Docs link."
        )

    export_url = (
        f"https://docs.google.com/document/d/"
        f"{document_id}/export?format=docx"
    )

    response = requests.get(
        export_url,
        timeout=30,
    )

    if response.status_code != 200:
        raise ValueError(
            "Could not download Google Doc. "
            "Make sure it is shared publicly."
        )

    temp_file = tempfile.NamedTemporaryFile(
        delete=False,
        suffix=".docx",
    )

    temp_file.write(response.content)
    temp_file.close()

    return temp_file.name


# ============================================================
# GOOGLE DRIVE FILE DOWNLOAD
# ============================================================

def download_drive_file(url):

    document_id = extract_google_id(url)

    if not document_id:
        raise ValueError(
            "Invalid Google Drive link."
        )

    download_url = (
        "https://drive.google.com/uc"
        f"?export=download&id={document_id}"
    )

    response = requests.get(
        download_url,
        timeout=30,
    )

    if response.status_code != 200:
        raise ValueError(
            "Could not download Google Drive file."
        )

    content_type = response.headers.get(
        "content-type",
        ""
    ).lower()

    # Most supported Drive files are handled as DOCX/PDF
    if "pdf" in content_type:
        suffix = ".pdf"
    else:
        suffix = ".docx"

    temp_file = tempfile.NamedTemporaryFile(
        delete=False,
        suffix=suffix,
    )

    temp_file.write(response.content)
    temp_file.close()

    return temp_file.name, suffix


# ============================================================
# SEMANTIC SEARCH
# ============================================================

def semantic_search(question, top_k=8):

    if (
        st.session_state.index is None
        or not st.session_state.chunks
    ):
        return []

    model = load_embedding_model()

    query_embedding = model.encode(
        [question],
        convert_to_numpy=True,
        normalize_embeddings=True,
    )

    query_embedding = np.asarray(
        query_embedding,
        dtype="float32",
    )

    scores, indices = (
        st.session_state.index.search(
            query_embedding,
            min(
                top_k,
                len(st.session_state.chunks),
            ),
        )
    )

    results = []

    for score, index in zip(
        scores[0],
        indices[0],
    ):

        if index < 0:
            continue

        chunk = dict(
            st.session_state.chunks[index]
        )

        chunk["semantic_score"] = float(score)

        results.append(chunk)

    return results


# ============================================================
# KEYWORD SCORE
# ============================================================

def keyword_score(question, text):

    question_words = set(
        re.findall(
            r"\b[a-zA-Z0-9]+\b",
            question.lower(),
        )
    )

    text_words = set(
        re.findall(
            r"\b[a-zA-Z0-9]+\b",
            text.lower(),
        )
    )

    if not question_words:
        return 0.0

    matches = question_words.intersection(
        text_words
    )

    return len(matches) / len(question_words)


# ============================================================
# HYBRID SEARCH
# ============================================================

def hybrid_search(question, top_k=5):

    candidates = semantic_search(
        question,
        top_k=12,
    )

    if not candidates:
        return []

    for item in candidates:

        item["keyword_score"] = keyword_score(
            question,
            item["text"],
        )

        item["score"] = (
            0.70 * item["semantic_score"]
            + 0.30 * item["keyword_score"]
        )

    candidates.sort(
        key=lambda x: x["score"],
        reverse=True,
    )

    return candidates[:top_k]


# ============================================================
# GROQ ANSWER
# ============================================================

def answer_question(question, results):

    if not results:
        return (
            "I could not find relevant information "
            "in the uploaded documents."
        )

    api_key = st.secrets.get(
        "GROQ_API_KEY",
        os.getenv("GROQ_API_KEY"),
    )

    if not api_key:
        return (
            "GROQ_API_KEY is not configured. "
            "Please add it to Streamlit Secrets."
        )

    context_parts = []

    for i, result in enumerate(
        results,
        start=1,
    ):

        page_text = ""

        if result["page"] is not None:
            page_text = (
                f"Page: {result['page']}"
            )

        context_parts.append(
            f"""
SOURCE {i}
Filename: {result['filename']}
{page_text}
Content:
{result['text']}
"""
        )

    context = "\n".join(
        context_parts
    )

    prompt = f"""
You are an AI document assistant.

Answer the user's question using ONLY
the information contained in the provided sources.

If the answer is not available in the sources,
clearly say:

"I could not find this information in the
uploaded documents."

Do not invent facts.
Do not use outside knowledge.

USER QUESTION:
{question}

DOCUMENT SOURCES:
{context}
"""

    try:

        client = load_groq_client(
            api_key
        )

        response = client.chat.completions.create(
            model=GROQ_MODEL,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You answer questions "
                        "strictly from supplied "
                        "document context."
                    ),
                },
                {
                    "role": "user",
                    "content": prompt,
                },
            ],
            temperature=0.1,
            max_tokens=1000,
        )

        return response.choices[0].message.content

    except Exception as e:

        return (
            "I could not generate the answer "
            f"because of a Groq/API error:\n\n{e}"
        )


# ============================================================
# HERO
# ============================================================

st.markdown(
    """
    <div class="hero">
        <h1>📚 AI Document Assistant</h1>
        <p>
            Upload your documents, search them intelligently,
            and ask questions using AI.
        </p>
    </div>
    """,
    unsafe_allow_html=True,
)


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:

    st.header("📂 Document Manager")

    uploaded_files = st.file_uploader(
        "Upload documents",
        type=[
            "pdf",
            "docx",
            "txt",
            "md",
        ],
        accept_multiple_files=True,
    )

    if uploaded_files:

        for uploaded_file in uploaded_files:

            if (
                uploaded_file.name
                in st.session_state.processed_names
            ):
                continue

            try:

                with st.spinner(
                    f"Processing {uploaded_file.name}..."
                ):

                    file_path = save_uploaded_file(
                        uploaded_file
                    )

                    records = extract_document(
                        file_path,
                        uploaded_file.name,
                    )

                    process_records(records)

                st.success(
                    f"Loaded: {uploaded_file.name}"
                )

            except Exception as e:

                st.error(
                    f"Could not process "
                    f"{uploaded_file.name}: {e}"
                )

    st.divider()

    st.subheader("🔗 Google Document")

    google_url = st.text_input(
        "Paste public Google Drive / Docs link",
        placeholder="https://drive.google.com/...",
    )

    if st.button(
        "Load Google Document",
        use_container_width=True,
    ):

        if not google_url.strip():

            st.warning(
                "Please paste a Google link."
            )

        else:

            try:

                if "docs.google.com/document" in google_url:

                    with st.spinner(
                        "Downloading Google Doc..."
                    ):

                        file_path = download_google_doc(
                            google_url
                        )

                        filename = (
                            "Google_Document.docx"
                        )

                        records = extract_document(
                            file_path,
                            filename,
                        )

                        process_records(
                            records
                        )

                    st.success(
                        "Google Doc loaded successfully."
                    )

                elif "drive.google.com" in google_url:

                    with st.spinner(
                        "Downloading Google Drive file..."
                    ):

                        file_path, suffix = (
                            download_drive_file(
                                google_url
                            )
                        )

                        filename = (
                            "Google_Drive_Document"
                            + suffix
                        )

                        records = extract_document(
                            file_path,
                            filename,
                        )

                        process_records(
                            records
                        )

                    st.success(
                        "Google Drive document loaded."
                    )

                else:

                    st.error(
                        "Please provide a valid "
                        "Google Drive or Google Docs link."
                    )

            except Exception as e:

                st.error(
                    f"Google document loading failed: {e}"
                )

    st.divider()

    if st.button(
        "🗑️ Clear All Documents",
        use_container_width=True,
    ):

        st.session_state.documents = []
        st.session_state.chunks = []
        st.session_state.embeddings = None
        st.session_state.index = None
        st.session_state.processed_names = set()

        st.success(
            "All documents cleared."
        )

        st.rerun()


# ============================================================
# DASHBOARD METRICS
# ============================================================

col1, col2, col3 = st.columns(3)

with col1:

    st.markdown(
        f"""
        <div class="metric-card">
            <div class="metric-number">
                {len(st.session_state.documents)}
            </div>
            <div class="metric-label">
                Documents
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

with col2:

    st.markdown(
        f"""
        <div class="metric-card">
            <div class="metric-number">
                {len(st.session_state.chunks)}
            </div>
            <div class="metric-label">
                Text Chunks
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

with col3:

    st.markdown(
        f"""
        <div class="metric-card">
            <div class="metric-number">
                {len(st.session_state.messages)}
            </div>
            <div class="metric-label">
                Questions
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


st.write("")


# ============================================================
# DOCUMENT LIBRARY
# ============================================================

st.subheader("📖 Document Library")

if st.session_state.documents:

    for document in st.session_state.documents:

        st.markdown(
            f"""
            <div class="card">
                <b>📄 {document['filename']}</b>
                <br>
                <span style="color:#6b7280;">
                    Source: {document['source']}
                </span>
            </div>
            """,
            unsafe_allow_html=True,
        )

else:

    st.info(
        "📂 No documents loaded. "
        "Upload a PDF, DOCX, TXT or MD file "
        "from the sidebar, or connect a public "
        "Google Drive / Google Docs document."
    )


# ============================================================
# CHAT SECTION
# ============================================================

st.subheader("💬 Ask Your Documents")

question = st.chat_input(
    "Ask a question about your documents..."
)


if question:

    if not st.session_state.chunks:

        st.warning(
            "Please upload a document first."
        )

    else:

        st.session_state.messages.append(
            {
                "role": "user",
                "content": question,
            }
        )

        with st.spinner(
            "Searching your documents..."
        ):

            results = hybrid_search(
                question,
                top_k=5,
            )

        with st.spinner(
            "Generating answer..."
        ):

            answer = answer_question(
                question,
                results,
            )

        st.session_state.messages.append(
            {
                "role": "assistant",
                "content": answer,
                "results": results,
            }
        )


# ============================================================
# DISPLAY CHAT
# ============================================================

for message in st.session_state.messages:

    if message["role"] == "user":

        with st.chat_message("user"):
            st.write(
                message["content"]
            )

    else:

        with st.chat_message("assistant"):

            st.markdown(
                message["content"]
            )

            results = message.get(
                "results",
                [],
            )

            if results:

                st.markdown(
                    "### 📚 Sources"
                )

                for i, result in enumerate(
                    results,
                    start=1,
                ):

                    filename = result[
                        "filename"
                    ]

                    page = result[
                        "page"
                    ]

                    score = result[
                        "score"
                    ]

                    text = result[
                        "text"
                    ]

                    if page is not None:

                        location = (
                            f"Page {page}"
                        )

                    else:

                        location = (
                            "Document"
                        )

                    # Limit displayed source text
                    preview = text[:500]

                    if len(text) > 500:
                        preview += "..."

                    st.markdown(
                        f"""
                        <div class="source-card">

                        <div class="source-title">
                            {i}. 📄 {filename}
                        </div>

                        <div class="source-meta">
                            {location}
                            &nbsp; • &nbsp;
                            Relevance: {score:.3f}
                        </div>

                        <div class="source-text">
                            {preview}
                        </div>

                        </div>
                        """,
                        unsafe_allow_html=True,
                    )


# ============================================================
# FOOTER
# ============================================================

st.divider()

st.caption(
    "AI Document Assistant • "
    "Semantic Search + Hybrid Ranking + Groq"
)
