import os
import re
import io
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
)


# ============================================================
# CSS
# ============================================================

st.markdown(
    """
    <style>

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
        color: #d1d5db;
        font-size: 16px;
    }

    .card {
        background: white;
        padding: 18px;
        border-radius: 14px;
        border: 1px solid #e5e7eb;
        margin-bottom: 12px;
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
    }

    .source-meta {
        color: #6b7280;
        font-size: 13px;
        margin-top: 5px;
        margin-bottom: 8px;
    }

    .source-text {
        font-size: 14px;
        line-height: 1.6;
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

if "index" not in st.session_state:
    st.session_state.index = None

if "embeddings" not in st.session_state:
    st.session_state.embeddings = None

if "messages" not in st.session_state:
    st.session_state.messages = []

if "processed_names" not in st.session_state:
    st.session_state.processed_names = set()


# ============================================================
# SETTINGS
# ============================================================

CHUNK_SIZE = 700
CHUNK_OVERLAP = 120

EMBEDDING_MODEL = "all-MiniLM-L6-v2"

GROQ_MODEL = "openai/gpt-oss-120b"


# ============================================================
# LOAD EMBEDDING MODEL
# ============================================================

@st.cache_resource
def get_embedding_model():

    return SentenceTransformer(
        EMBEDDING_MODEL
    )


# ============================================================
# GROQ CLIENT
# ============================================================

@st.cache_resource
def get_groq_client(api_key):

    return Groq(
        api_key=api_key
    )


# ============================================================
# CLEAN TEXT
# ============================================================

def clean_text(text):

    if not text:
        return ""

    text = text.replace(
        "\x00",
        " "
    )

    text = re.sub(
        r"[ \t]+",
        " ",
        text
    )

    text = re.sub(
        r"\n\s*\n+",
        "\n\n",
        text
    )

    return text.strip()


# ============================================================
# PDF EXTRACTION FROM BYTES
# ============================================================

def extract_pdf_bytes(
    file_bytes,
    filename
):

    records = []

    try:

        pdf_file = io.BytesIO(
            file_bytes
        )

        reader = PdfReader(
            pdf_file
        )

        for page_number, page in enumerate(
            reader.pages,
            start=1
        ):

            try:

                text = (
                    page.extract_text()
                    or ""
                )

            except Exception:

                text = ""

            text = clean_text(
                text
            )

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

        st.error(
            f"Could not read PDF: {e}"
        )

    return records


# ============================================================
# DOCX EXTRACTION FROM BYTES
# ============================================================

def extract_docx_bytes(
    file_bytes,
    filename
):

    records = []

    try:

        docx_file = io.BytesIO(
            file_bytes
        )

        document = Document(
            docx_file
        )

        paragraphs = []

        for paragraph in document.paragraphs:

            text = paragraph.text.strip()

            if text:

                paragraphs.append(
                    text
                )

        # Also read tables
        for table in document.tables:

            for row in table.rows:

                row_text = []

                for cell in row.cells:

                    cell_text = (
                        cell.text.strip()
                    )

                    if cell_text:

                        row_text.append(
                            cell_text
                        )

                if row_text:

                    paragraphs.append(
                        " | ".join(row_text)
                    )

        full_text = "\n".join(
            paragraphs
        )

        full_text = clean_text(
            full_text
        )

        if full_text:

            records.append(
                {
                    "text": full_text,
                    "filename": filename,
                    "page": None,
                    "source": "DOCX",
                }
            )

        else:

            st.warning(
                f"{filename} contains no readable text."
            )

    except Exception as e:

        st.error(
            f"Could not read DOCX: {e}"
        )

    return records


# ============================================================
# TXT EXTRACTION
# ============================================================

def extract_txt_bytes(
    file_bytes,
    filename
):

    records = []

    try:

        text = file_bytes.decode(
            "utf-8",
            errors="ignore"
        )

        text = clean_text(
            text
        )

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

        st.error(
            f"Could not read TXT: {e}"
        )

    return records


# ============================================================
# MARKDOWN EXTRACTION
# ============================================================

def extract_md_bytes(
    file_bytes,
    filename
):

    records = []

    try:

        text = file_bytes.decode(
            "utf-8",
            errors="ignore"
        )

        text = clean_text(
            text
        )

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

        st.error(
            f"Could not read MD: {e}"
        )

    return records


# ============================================================
# MAIN FILE READER
# ============================================================

def read_file_bytes(
    file_bytes,
    filename
):

    extension = (
        Path(filename)
        .suffix
        .lower()
    )

    if extension == ".pdf":

        return extract_pdf_bytes(
            file_bytes,
            filename
        )

    elif extension == ".docx":

        return extract_docx_bytes(
            file_bytes,
            filename
        )

    elif extension == ".txt":

        return extract_txt_bytes(
            file_bytes,
            filename
        )

    elif extension == ".md":

        return extract_md_bytes(
            file_bytes,
            filename
        )

    else:

        st.error(
            f"Unsupported file type: {extension}"
        )

        return []


# ============================================================
# CHUNKING
# ============================================================

def create_chunks(records):

    chunks = []

    for record in records:

        text = record["text"]

        start = 0

        while start < len(text):

            end = min(
                start + CHUNK_SIZE,
                len(text)
            )

            chunk_text = (
                text[start:end]
                .strip()
            )

            if chunk_text:

                chunks.append(
                    {
                        "text": chunk_text,
                        "filename": record["filename"],
                        "page": record["page"],
                        "source": record["source"],
                    }
                )

            if end >= len(text):
                break

            start = (
                end - CHUNK_OVERLAP
            )

    return chunks


# ============================================================
# BUILD FAISS
# ============================================================

def rebuild_index():

    if not st.session_state.chunks:

        st.session_state.index = None

        st.session_state.embeddings = None

        return

    model = get_embedding_model()

    texts = [
        item["text"]
        for item in st.session_state.chunks
    ]

    embeddings = model.encode(
        texts,
        convert_to_numpy=True,
        normalize_embeddings=True,
        show_progress_bar=False,
    )

    embeddings = np.asarray(
        embeddings,
        dtype="float32"
    )

    index = faiss.IndexFlatIP(
        embeddings.shape[1]
    )

    index.add(
        embeddings
    )

    st.session_state.embeddings = (
        embeddings
    )

    st.session_state.index = (
        index
    )


# ============================================================
# PROCESS RECORDS
# ============================================================

def process_records(records):

    if not records:
        return

    new_chunks = create_chunks(
        records
    )

    if not new_chunks:
        return

    st.session_state.chunks.extend(
        new_chunks
    )

    for record in records:

        filename = record[
            "filename"
        ]

        if (
            filename
            not in st.session_state.processed_names
        ):

            st.session_state.documents.append(
                {
                    "filename": filename,
                    "source": record[
                        "source"
                    ],
                }
            )

            st.session_state.processed_names.add(
                filename
            )

    rebuild_index()


# ============================================================
# GOOGLE ID
# ============================================================

def get_google_id(url):

    patterns = [

        r"drive\.google\.com/file/d/([^/]+)",

        r"drive\.google\.com/open\?id=([^&]+)",

        r"drive\.google\.com/uc\?id=([^&]+)",

        r"docs\.google\.com/document/d/([^/]+)",
    ]

    for pattern in patterns:

        match = re.search(
            pattern,
            url
        )

        if match:

            return match.group(1)

    return None


# ============================================================
# DOWNLOAD GOOGLE DOC
# ============================================================

def download_google_doc(url):

    document_id = get_google_id(
        url
    )

    if not document_id:

        raise ValueError(
            "Invalid Google Docs link."
        )

    export_url = (
        "https://docs.google.com/document/d/"
        f"{document_id}/export?format=docx"
    )

    response = requests.get(
        export_url,
        timeout=30
    )

    if response.status_code != 200:

        raise ValueError(
            "Google Doc could not be downloaded. "
            "Make sure sharing is set to "
            "'Anyone with the link'."
        )

    content = response.content

    # A real DOCX is a ZIP file and starts with PK
    if not content.startswith(
        b"PK"
    ):

        raise ValueError(
            "Google Docs did not return a valid DOCX file. "
            "Check that the document is publicly accessible."
        )

    return content


# ============================================================
# DOWNLOAD GOOGLE DRIVE FILE
# ============================================================

def download_google_drive_file(url):

    document_id = get_google_id(
        url
    )

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
        timeout=30
    )

    if response.status_code != 200:

        raise ValueError(
            "Could not download Google Drive file."
        )

    content = response.content

    content_type = (
        response.headers
        .get(
            "content-type",
            ""
        )
        .lower()
    )

    if content.startswith(
        b"%PDF"
    ):

        return content, ".pdf"

    if content.startswith(
        b"PK"
    ):

        return content, ".docx"

    # Google sometimes returns HTML
    if "text/html" in content_type:

        raise ValueError(
            "Google Drive returned a web page instead "
            "of the file. Make sure the file is shared "
            "as 'Anyone with the link → Viewer'."
        )

    raise ValueError(
        "Downloaded file format could not be recognized."
    )


# ============================================================
# SEMANTIC SEARCH
# ============================================================

def semantic_search(
    question,
    top_k=10
):

    if (
        st.session_state.index is None
        or not st.session_state.chunks
    ):

        return []

    model = get_embedding_model()

    query_embedding = model.encode(
        [question],
        convert_to_numpy=True,
        normalize_embeddings=True,
    )

    query_embedding = np.asarray(
        query_embedding,
        dtype="float32"
    )

    k = min(
        top_k,
        len(
            st.session_state.chunks
        )
    )

    scores, indices = (
        st.session_state.index.search(
            query_embedding,
            k
        )
    )

    results = []

    for score, index in zip(
        scores[0],
        indices[0]
    ):

        if index < 0:
            continue

        result = dict(
            st.session_state.chunks[
                index
            ]
        )

        result[
            "semantic_score"
        ] = float(score)

        results.append(
            result
        )

    return results


# ============================================================
# KEYWORD SCORE
# ============================================================

def keyword_score(
    question,
    text
):

    question_words = set(
        re.findall(
            r"\b[a-zA-Z0-9]+\b",
            question.lower()
        )
    )

    text_words = set(
        re.findall(
            r"\b[a-zA-Z0-9]+\b",
            text.lower()
        )
    )

    if not question_words:

        return 0.0

    matches = (
        question_words
        & text_words
    )

    return (
        len(matches)
        / len(question_words)
    )


# ============================================================
# HYBRID SEARCH
# ============================================================

def hybrid_search(
    question,
    top_k=5
):

    candidates = semantic_search(
        question,
        top_k=12
    )

    if not candidates:
        return []

    for item in candidates:

        item[
            "keyword_score"
        ] = keyword_score(
            question,
            item["text"]
        )

        item["score"] = (
            0.70
            * item["semantic_score"]
            +
            0.30
            * item["keyword_score"]
        )

    candidates.sort(
        key=lambda x: x["score"],
        reverse=True
    )

    return candidates[:top_k]


# ============================================================
# ANSWER WITH GROQ
# ============================================================

def answer_question(
    question,
    results
):

    if not results:

        return (
            "I could not find this information "
            "in the uploaded documents."
        )

    api_key = st.secrets.get(
        "GROQ_API_KEY",
        os.getenv(
            "GROQ_API_KEY"
        )
    )

    if not api_key:

        return (
            "GROQ_API_KEY is not configured. "
            "Please add it to Streamlit Secrets."
        )

    context = ""

    for i, result in enumerate(
        results,
        start=1
    ):

        page = ""

        if result["page"] is not None:

            page = (
                f"Page: {result['page']}"
            )

        context += f"""

SOURCE {i}

Filename:
{result['filename']}

{page}

Content:
{result['text']}

-------------------------
"""

    prompt = f"""
You are an AI Document Assistant.

Answer ONLY using the provided document context.

Do not use outside knowledge.

Do not invent information.

If the answer is not present in the
documents, say:

"I could not find this information
in the uploaded documents."

User question:

{question}

Document context:

{context}
"""

    try:

        client = get_groq_client(
            api_key
        )

        response = (
            client.chat.completions.create(
                model=GROQ_MODEL,
                messages=[
                    {
                        "role": "system",
                        "content": (
                            "Answer strictly "
                            "from document context."
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
        )

        return (
            response
            .choices[0]
            .message
            .content
        )

    except Exception as e:

        return (
            "Groq error:\n\n"
            + str(e)
        )


# ============================================================
# HERO
# ============================================================

st.markdown(
    """
    <div class="hero">

        <h1>
            📚 AI Document Assistant
        </h1>

        <p>
            Upload documents and ask questions
            using semantic search and AI.
        </p>

    </div>
    """,
    unsafe_allow_html=True
)


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:

    st.header(
        "📂 Document Manager"
    )

    uploaded_files = st.file_uploader(
        "Upload PDF, DOCX, TXT or MD",
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

            filename = uploaded_file.name

            if (
                filename
                in st.session_state.processed_names
            ):

                continue

            try:

                with st.spinner(
                    f"Processing {filename}..."
                ):

                    file_bytes = (
                        uploaded_file
                        .getvalue()
                    )

                    records = read_file_bytes(
                        file_bytes,
                        filename
                    )

                    if records:

                        process_records(
                            records
                        )

                        st.success(
                            f"Loaded: {filename}"
                        )

                    else:

                        st.warning(
                            f"No readable text found "
                            f"in {filename}."
                        )

            except Exception as e:

                st.error(
                    f"Could not process "
                    f"{filename}: {e}"
                )

    st.divider()

    st.subheader(
        "🔗 Google Document"
    )

    google_url = st.text_input(
        "Google Drive / Google Docs link",
        placeholder=(
            "Paste public link here"
        ),
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

                if (
                    "docs.google.com/document"
                    in google_url
                ):

                    with st.spinner(
                        "Downloading Google Doc..."
                    ):

                        file_bytes = (
                            download_google_doc(
                                google_url
                            )
                        )

                        records = (
                            extract_docx_bytes(
                                file_bytes,
                                "Google_Document.docx"
                            )
                        )

                        process_records(
                            records
                        )

                    st.success(
                        "Google Doc loaded successfully."
                    )

                elif (
                    "drive.google.com"
                    in google_url
                ):

                    with st.spinner(
                        "Downloading Google Drive file..."
                    ):

                        file_bytes, suffix = (
                            download_google_drive_file(
                                google_url
                            )
                        )

                        filename = (
                            "Google_Drive_Document"
                            + suffix
                        )

                        records = read_file_bytes(
                            file_bytes,
                            filename
                        )

                        process_records(
                            records
                        )

                    st.success(
                        "Google Drive file loaded successfully."
                    )

                else:

                    st.error(
                        "Invalid Google link."
                    )

            except Exception as e:

                st.error(
                    f"Google loading failed: {e}"
                )

    st.divider()

    if st.button(
        "🗑️ Clear Documents",
        use_container_width=True,
    ):

        st.session_state.documents = []

        st.session_state.chunks = []

        st.session_state.index = None

        st.session_state.embeddings = None

        st.session_state.processed_names = set()

        st.session_state.messages = []

        st.rerun()


# ============================================================
# METRICS
# ============================================================

c1, c2, c3 = st.columns(3)

with c1:

    st.metric(
        "Documents",
        len(
            st.session_state.documents
        )
    )

with c2:

    st.metric(
        "Chunks",
        len(
            st.session_state.chunks
        )
    )

with c3:

    st.metric(
        "Questions",
        len(
            st.session_state.messages
        )
        // 2
    )


# ============================================================
# DOCUMENT LIBRARY
# ============================================================

st.subheader(
    "📖 Document Library"
)

if st.session_state.documents:

    for document in (
        st.session_state.documents
    ):

        st.markdown(
            f"""
            <div class="card">

                <b>
                    📄 {document['filename']}
                </b>

                <br>

                <small>
                    Source: {document['source']}
                </small>

            </div>
            """,
            unsafe_allow_html=True
        )

else:

    st.info(
        "📂 No documents loaded. "
        "Upload a PDF, DOCX, TXT or MD "
        "from the sidebar."
    )


# ============================================================
# CHAT
# ============================================================

st.subheader(
    "💬 Ask Your Documents"
)

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
            "Searching documents..."
        ):

            results = hybrid_search(
                question
            )

        with st.spinner(
            "Generating answer..."
        ):

            answer = answer_question(
                question,
                results
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

for message in (
    st.session_state.messages
):

    if message["role"] == "user":

        with st.chat_message(
            "user"
        ):

            st.write(
                message["content"]
            )

    else:

        with st.chat_message(
            "assistant"
        ):

            st.markdown(
                message["content"]
            )

            results = message.get(
                "results",
                []
            )

            if results:

                st.markdown(
                    "### 📚 Sources"
                )

                for i, result in enumerate(
                    results,
                    start=1
                ):

                    filename = (
                        result["filename"]
                    )

                    page = (
                        result["page"]
                    )

                    score = (
                        result["score"]
                    )

                    text = (
                        result["text"]
                    )

                    if page is not None:

                        location = (
                            f"Page {page}"
                        )

                    else:

                        location = (
                            "Document"
                        )

                    preview = text[:500]

                    if len(text) > 500:

                        preview += "..."

                    # Escape HTML-sensitive text
                    # so document content cannot
                    # break the UI.
                    preview = (
                        preview
                        .replace(
                            "&",
                            "&amp;"
                        )
                        .replace(
                            "<",
                            "&lt;"
                        )
                        .replace(
                            ">",
                            "&gt;"
                        )
                    )

                    filename_display = (
                        filename
                        .replace(
                            "&",
                            "&amp;"
                        )
                        .replace(
                            "<",
                            "&lt;"
                        )
                        .replace(
                            ">",
                            "&gt;"
                        )
                    )

                    st.markdown(
                        f"""
                        <div class="source-card">

                            <div class="source-title">
                                {i}. 📄
                                {filename_display}
                            </div>

                            <div class="source-meta">
                                {location}
                                &nbsp; • &nbsp;
                                Relevance:
                                {score:.3f}
                            </div>

                            <div class="source-text">
                                {preview}
                            </div>

                        </div>
                        """,
                        unsafe_allow_html=True
                    )


# ============================================================
# FOOTER
# ============================================================

st.divider()

st.caption(
    "AI Document Assistant • "
    "FAISS + Sentence Transformers + Groq"
)
