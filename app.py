import os
import re
import tempfile
from pathlib import Path
from urllib.parse import urlparse, parse_qs

import faiss
import gdown
import numpy as np
import requests
import streamlit as st

from docx import Document
from groq import Groq
from pypdf import PdfReader
from sentence_transformers import SentenceTransformer


# ============================================================
# APP CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="DocuMind AI | Document Assistant",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="expanded",
)


SUPPORTED_EXTENSIONS = {
    ".pdf",
    ".docx",
    ".txt",
    ".md",
}

CHUNK_SIZE = 700
CHUNK_OVERLAP = 120
TOP_K = 5


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

if "processed_files" not in st.session_state:
    st.session_state.processed_files = set()

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []


# ============================================================
# CUSTOM CSS
# ============================================================

st.markdown(
    """
    <style>

    /* -----------------------------
       GLOBAL
    ----------------------------- */

    .stApp {
        background: #f6f7fb;
    }

    .block-container {
        max-width: 1450px;
        padding-top: 2rem;
        padding-bottom: 4rem;
    }


    /* -----------------------------
       SIDEBAR
    ----------------------------- */

    section[data-testid="stSidebar"] {
        background: #111827;
    }

    section[data-testid="stSidebar"] * {
        color: #f9fafb;
    }

    section[data-testid="stSidebar"] .stTextInput input {
        background: #1f2937;
        color: white;
        border: 1px solid #374151;
    }

    section[data-testid="stSidebar"] .stFileUploader {
        background: #1f2937;
        border-radius: 12px;
        padding: 8px;
    }


    /* -----------------------------
       HERO
    ----------------------------- */

    .hero {
        background: linear-gradient(
            135deg,
            #111827 0%,
            #1f2937 55%,
            #374151 100%
        );

        border-radius: 22px;
        padding: 34px 38px;
        margin-bottom: 25px;

        box-shadow:
            0 12px 30px rgba(0, 0, 0, 0.12);
    }

    .hero-title {
        font-size: 38px;
        font-weight: 800;
        color: white;
        letter-spacing: -1px;
        margin-bottom: 7px;
    }

    .hero-subtitle {
        font-size: 16px;
        color: #d1d5db;
        max-width: 760px;
        line-height: 1.6;
    }

    .online-badge {
        display: inline-block;
        margin-top: 16px;
        padding: 6px 13px;
        border-radius: 999px;

        background: #064e3b;
        color: #a7f3d0;

        font-size: 12px;
        font-weight: 700;
        letter-spacing: 0.5px;
    }


    /* -----------------------------
       SECTION HEADINGS
    ----------------------------- */

    .section-title {
        font-size: 22px;
        font-weight: 750;
        color: #111827;
        margin-top: 28px;
        margin-bottom: 14px;
    }


    /* -----------------------------
       CARDS
    ----------------------------- */

    .document-card {
        background: white;
        border: 1px solid #e5e7eb;
        border-radius: 16px;
        padding: 18px 20px;
        margin-bottom: 12px;

        box-shadow:
            0 3px 12px rgba(0, 0, 0, 0.035);
    }

    .document-name {
        font-size: 16px;
        font-weight: 700;
        color: #111827;
    }

    .document-meta {
        font-size: 13px;
        color: #6b7280;
        margin-top: 5px;
    }


    /* -----------------------------
       SOURCE CARDS
    ----------------------------- */

    .source-card {
        background: white;
        border: 1px solid #e5e7eb;
        border-radius: 15px;
        padding: 18px 20px;
        margin-bottom: 12px;

        box-shadow:
            0 3px 12px rgba(0, 0, 0, 0.035);
    }

    .source-title {
        font-size: 15px;
        font-weight: 750;
        color: #111827;
    }

    .source-meta {
        color: #6b7280;
        font-size: 12px;
        margin-top: 5px;
    }

    .source-text {
        color: #374151;
        font-size: 14px;
        line-height: 1.65;
        margin-top: 12px;
    }


    /* -----------------------------
       INFO BOX
    ----------------------------- */

    .info-card {
        background: white;
        border: 1px solid #e5e7eb;
        border-radius: 16px;
        padding: 20px;
    }


    /* -----------------------------
       FOOTER
    ----------------------------- */

    .footer {
        text-align: center;
        color: #9ca3af;
        font-size: 12px;
        padding-top: 35px;
    }


    /* -----------------------------
       STREAMLIT CLEANUP
    ----------------------------- */

    #MainMenu {
        visibility: hidden;
    }

    footer {
        visibility: hidden;
    }

    </style>
    """,
    unsafe_allow_html=True,
)


# ============================================================
# CACHED MODELS
# ============================================================

@st.cache_resource
def load_embedding_model():
    return SentenceTransformer("all-MiniLM-L6-v2")


@st.cache_resource
def load_groq_client():

    try:
        api_key = st.secrets.get(
            "GROQ_API_KEY",
            os.getenv("GROQ_API_KEY"),
        )
    except Exception:
        api_key = os.getenv("GROQ_API_KEY")

    if not api_key:
        return None

    return Groq(api_key=api_key)


# ============================================================
# DOCUMENT EXTRACTION
# ============================================================

def extract_pdf(file_path):
    """Extract one record per PDF page."""

    reader = PdfReader(file_path)

    records = []

    for page_number, page in enumerate(
        reader.pages,
        start=1,
    ):

        text = page.extract_text() or ""

        if text.strip():

            records.append(
                {
                    "filename": Path(file_path).name,
                    "page": page_number,
                    "text": text.strip(),
                }
            )

    return records


def extract_docx(file_path):
    """Extract text from DOCX."""

    document = Document(file_path)

    paragraphs = [
        paragraph.text.strip()
        for paragraph in document.paragraphs
        if paragraph.text.strip()
    ]

    text = "\n".join(paragraphs)

    if not text:
        return []

    return [
        {
            "filename": Path(file_path).name,
            "page": None,
            "text": text,
        }
    ]


def extract_txt(file_path):
    """Extract TXT text."""

    text = Path(file_path).read_text(
        encoding="utf-8",
        errors="ignore",
    ).strip()

    if not text:
        return []

    return [
        {
            "filename": Path(file_path).name,
            "page": None,
            "text": text,
        }
    ]


def extract_md(file_path):
    """Extract Markdown text."""

    text = Path(file_path).read_text(
        encoding="utf-8",
        errors="ignore",
    ).strip()

    if not text:
        return []

    return [
        {
            "filename": Path(file_path).name,
            "page": None,
            "text": text,
        }
    ]


def extract_document(file_path):
    """Choose extraction method according to file extension."""

    extension = Path(file_path).suffix.lower()

    if extension == ".pdf":
        return extract_pdf(file_path)

    if extension == ".docx":
        return extract_docx(file_path)

    if extension == ".txt":
        return extract_txt(file_path)

    if extension == ".md":
        return extract_md(file_path)

    return []


# ============================================================
# TEXT CHUNKING
# ============================================================

def split_text(
    text,
    chunk_size=CHUNK_SIZE,
    overlap=CHUNK_OVERLAP,
):
    """Create overlapping text chunks."""

    text = re.sub(
        r"\s+",
        " ",
        text,
    ).strip()

    if not text:
        return []

    chunks = []

    start = 0

    while start < len(text):

        end = min(
            start + chunk_size,
            len(text),
        )

        chunk = text[start:end].strip()

        if chunk:
            chunks.append(chunk)

        if end == len(text):
            break

        start = max(
            end - overlap,
            start + 1,
        )

    return chunks


def create_chunks(records):
    """Create chunks while preserving metadata."""

    all_chunks = []

    for record in records:

        pieces = split_text(
            record["text"]
        )

        for piece in pieces:

            all_chunks.append(
                {
                    "filename": record["filename"],
                    "page": record["page"],
                    "text": piece,
                }
            )

    return all_chunks


# ============================================================
# EMBEDDINGS + FAISS
# ============================================================

def create_embeddings(chunks):

    model = load_embedding_model()

    texts = [
        chunk["text"]
        for chunk in chunks
    ]

    embeddings = model.encode(
        texts,
        convert_to_numpy=True,
        normalize_embeddings=True,
        show_progress_bar=False,
    ).astype("float32")

    return embeddings


def build_faiss_index(embeddings):

    dimension = embeddings.shape[1]

    index = faiss.IndexFlatIP(
        dimension
    )

    index.add(embeddings)

    return index


def process_documents(records):

    chunks = create_chunks(records)

    if not chunks:
        return [], None, None

    embeddings = create_embeddings(
        chunks
    )

    index = build_faiss_index(
        embeddings
    )

    return (
        chunks,
        embeddings,
        index,
    )


# ============================================================
# KEYWORD SEARCH
# ============================================================

STOPWORDS = {
    "the",
    "a",
    "an",
    "and",
    "or",
    "is",
    "are",
    "was",
    "were",
    "to",
    "of",
    "in",
    "on",
    "for",
    "with",
    "what",
    "which",
    "who",
    "when",
    "where",
    "why",
    "how",
    "does",
    "do",
    "did",
    "this",
    "that",
    "these",
    "those",
    "from",
    "as",
    "by",
    "be",
    "can",
    "could",
    "would",
    "should",
    "about",
    "it",
    "its",
}


def important_words(question):

    words = re.findall(
        r"\b[a-zA-Z0-9]+\b",
        question.lower(),
    )

    return [
        word
        for word in words
        if word not in STOPWORDS
        and len(word) > 2
    ]


def keyword_scores(
    question,
    chunks,
):

    query_words = set(
        important_words(question)
    )

    scores = np.zeros(
        len(chunks),
        dtype="float32",
    )

    if not query_words:
        return scores

    for i, chunk in enumerate(chunks):

        chunk_words = set(
            re.findall(
                r"\b[a-zA-Z0-9]+\b",
                chunk["text"].lower(),
            )
        )

        matches = len(
            query_words.intersection(
                chunk_words
            )
        )

        scores[i] = (
            matches / len(query_words)
        )

    return scores


# ============================================================
# HYBRID SEARCH
# ============================================================

def hybrid_search(
    question,
    top_k=TOP_K,
):

    if (
        not st.session_state.chunks
        or st.session_state.index is None
    ):
        return []

    model = load_embedding_model()

    question_embedding = model.encode(
        [question],
        convert_to_numpy=True,
        normalize_embeddings=True,
    ).astype("float32")

    semantic_scores = (
        st.session_state.embeddings
        @ question_embedding[0]
    )

    keyword_score = keyword_scores(
        question,
        st.session_state.chunks,
    )

    semantic_score = (
        semantic_scores + 1.0
    ) / 2.0

    hybrid_score = (
        0.70 * semantic_score
        + 0.30 * keyword_score
    )

    ranked_indices = np.argsort(
        hybrid_score
    )[::-1][:top_k]

    results = []

    for index in ranked_indices:

        result = dict(
            st.session_state.chunks[
                index
            ]
        )

        result["semantic_score"] = float(
            semantic_score[index]
        )

        result["keyword_score"] = float(
            keyword_score[index]
        )

        result["hybrid_score"] = float(
            hybrid_score[index]
        )

        results.append(result)

    return results


# ============================================================
# GROQ ANSWER GENERATION
# ============================================================

def answer_question(
    question,
    results,
):

    client = load_groq_client()

    if client is None:

        return (
            "GROQ_API_KEY is not configured. "
            "Please add GROQ_API_KEY to "
            "Streamlit Secrets."
        )

    if not results:

        return (
            "I could not find relevant "
            "information in the provided documents."
        )

    context_parts = []

    for i, result in enumerate(
        results,
        start=1,
    ):

        page = (
            f", page {result['page']}"
            if result["page"]
            else ""
        )

        context_parts.append(
            f"""
[Source {i}: {result['filename']}{page}]

{result['text']}
"""
        )

    context = "\n".join(
        context_parts
    )

    prompt = f"""
You are an AI document question-answering assistant.

Answer the user's question using ONLY the
document context provided below.

Rules:

1. Do not use outside knowledge.
2. Do not invent information.
3. If the answer is not available in the documents,
   say exactly:

"I could not find that information in the provided documents."

4. Keep the answer clear and concise.
5. When possible, mention the relevant document name.

DOCUMENT CONTEXT:

{context}

USER QUESTION:

{question}
"""

    try:

        response = client.chat.completions.create(

            model="openai/gpt-oss-120b",

            messages=[
                {
                    "role": "system",
                    "content": (
                        "Answer only from the supplied "
                        "document context."
                    ),
                },
                {
                    "role": "user",
                    "content": prompt,
                },
            ],

            temperature=0,
        )

        return (
            response
            .choices[0]
            .message
            .content
        )

    except Exception as error:

        return (
            "The AI answer service could not "
            f"complete the request.\n\n"
            f"Error: {error}"
        )


# ============================================================
# GOOGLE DRIVE / GOOGLE DOCS
# ============================================================

def normalize_google_drive_link(
    drive_link
):

    link = drive_link.strip()

    # --------------------------------------------------------
    # Google Docs
    # --------------------------------------------------------

    docs_match = re.search(
        r"docs\.google\.com/document/d/"
        r"([a-zA-Z0-9_-]+)",
        link,
    )

    if docs_match:

        document_id = (
            docs_match.group(1)
        )

        return {
            "type": "google_doc",
            "id": document_id,
            "url": (
                "https://docs.google.com/document/d/"
                f"{document_id}/export?format=docx"
            ),
        }

    # --------------------------------------------------------
    # Google Drive Folder
    # --------------------------------------------------------

    folder_match = re.search(
        r"/folders/([a-zA-Z0-9_-]+)",
        link,
    )

    if folder_match:

        folder_id = (
            folder_match.group(1)
        )

        return {
            "type": "folder",
            "id": folder_id,
            "url": (
                "https://drive.google.com/drive/folders/"
                f"{folder_id}"
            ),
        }

    # --------------------------------------------------------
    # Google Drive File
    # --------------------------------------------------------

    file_match = re.search(
        r"/file/d/([a-zA-Z0-9_-]+)",
        link,
    )

    if file_match:

        file_id = (
            file_match.group(1)
        )

        return {
            "type": "file",
            "id": file_id,
            "url": (
                "https://drive.google.com/file/d/"
                f"{file_id}/view"
            ),
        }

    # --------------------------------------------------------
    # Google Drive open?id=...
    # --------------------------------------------------------

    parsed = urlparse(link)

    query = parse_qs(
        parsed.query
    )

    if "id" in query:

        file_id = query["id"][0]

        return {
            "type": "file",
            "id": file_id,
            "url": (
                "https://drive.google.com/uc?id="
                f"{file_id}"
            ),
        }

    raise ValueError(
        "Invalid Google Drive link. "
        "Use a Google Drive file/folder link "
        "or a public Google Docs link."
    )


def download_google_doc(
    document_id,
    temp_dir,
):

    export_url = (
        "https://docs.google.com/document/d/"
        f"{document_id}/export?format=docx"
    )

    response = requests.get(
        export_url,
        timeout=30,
        allow_redirects=True,
    )

    if response.status_code != 200:

        raise ValueError(
            "Could not access the Google Doc. "
            "Make sure General access is "
            "'Anyone with the link' and role is Viewer."
        )

    content_type = (
        response.headers
        .get("content-type", "")
        .lower()
    )

    if (
        "text/html" in content_type
        and len(response.content) < 10000
    ):

        raise ValueError(
            "Google Doc could not be downloaded. "
            "Make sure the document is publicly accessible."
        )

    output_path = (
        Path(temp_dir)
        / f"Google_Doc_{document_id}.docx"
    )

    output_path.write_bytes(
        response.content
    )

    return output_path


def download_drive_link(
    drive_link
):

    temp_dir = Path(
        tempfile.mkdtemp(
            prefix="drive_docs_"
        )
    )

    drive = normalize_google_drive_link(
        drive_link
    )

    # --------------------------------------------------------
    # Google Docs
    # --------------------------------------------------------

    if drive["type"] == "google_doc":

        output_path = download_google_doc(
            drive["id"],
            temp_dir,
        )

        return [output_path]

    # --------------------------------------------------------
    # Google Drive Folder
    # --------------------------------------------------------

    if drive["type"] == "folder":

        gdown.download_folder(
            url=drive["url"],
            output=str(temp_dir),
            quiet=True,
            use_cookies=False,
        )

    # --------------------------------------------------------
    # Google Drive File
    # --------------------------------------------------------

    else:

        output_path = (
            temp_dir
            / "drive_download"
        )

        downloaded = gdown.download(
            url=drive["url"],
            output=str(output_path),
            quiet=True,
            use_cookies=False,
        )

        if downloaded is None:

            raise ValueError(
                "Could not download the Drive file. "
                "Make sure it is shared as "
                "'Anyone with the link'."
            )

    supported_files = []

    for path in temp_dir.rglob("*"):

        if (
            path.is_file()
            and path.suffix.lower()
            in SUPPORTED_EXTENSIONS
        ):

            supported_files.append(path)

    return supported_files


# ============================================================
# LOCAL FILE UPLOAD
# ============================================================

def read_uploaded_file(
    uploaded_file
):

    suffix = Path(
        uploaded_file.name
    ).suffix.lower()

    if suffix not in SUPPORTED_EXTENSIONS:
        return None

    temp_dir = Path(
        tempfile.mkdtemp(
            prefix="local_docs_"
        )
    )

    file_path = (
        temp_dir
        / uploaded_file.name
    )

    file_path.write_bytes(
        uploaded_file.getvalue()
    )

    return file_path


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:

    st.markdown(
        """
        <div style="
            padding: 8px 0 22px 0;
        ">

            <div style="
                font-size: 25px;
                font-weight: 800;
            ">
                📚 DocuMind AI
            </div>

            <div style="
                font-size: 13px;
                color: #9ca3af;
                margin-top: 4px;
            ">
                Intelligent Document Assistant
            </div>

        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown(
        "### 📂 Add Documents"
    )

    uploaded_files = st.file_uploader(
        "Upload documents",
        type=[
            "pdf",
            "docx",
            "txt",
            "md",
        ],
        accept_multiple_files=True,
        help=(
            "Supported formats: "
            "PDF, DOCX, TXT and MD"
        ),
    )

    st.markdown("---")

    st.markdown(
        "### ☁️ Google Drive"
    )

    drive_link = st.text_input(
        "Drive or Google Docs link",
        placeholder=(
            "Paste Drive / Google Docs link..."
        ),
        label_visibility="collapsed",
    )

    load_drive = st.button(
        "☁️ Load Document",
        use_container_width=True,
    )

    st.markdown("---")

    st.markdown(
        "### 📊 Workspace"
    )

    st.metric(
        "Documents",
        len(
            st.session_state.documents
        ),
    )

    st.metric(
        "Text Chunks",
        len(
            st.session_state.chunks
        ),
    )

    st.markdown("---")

    if st.button(
        "🗑️ Clear Workspace",
        use_container_width=True,
    ):

        st.session_state.documents = []
        st.session_state.chunks = []
        st.session_state.embeddings = None
        st.session_state.index = None
        st.session_state.processed_files = set()
        st.session_state.chat_history = []

        st.rerun()

    st.markdown(
        """
        <div style="
            margin-top: 25px;
            padding: 13px;
            border-radius: 12px;
            background: #1f2937;
            color: #9ca3af;
            font-size: 12px;
            line-height: 1.5;
        ">
            🔒 Answers are generated from
            retrieved document context.
        </div>
        """,
        unsafe_allow_html=True,
    )


# ============================================================
# HERO
# ============================================================

st.markdown(
    """
    <div class="hero">

        <div class="hero-title">
            📚 AI Document Assistant
        </div>

        <div class="hero-subtitle">
            Upload documents, connect Google Drive,
            search intelligently and ask questions
            using AI-powered document retrieval.
        </div>

        <div class="online-badge">
            ● AI SYSTEM READY
        </div>

    </div>
    """,
    unsafe_allow_html=True,
)


# ============================================================
# LOCAL UPLOAD PROCESSING
# ============================================================

new_records = []


if uploaded_files:

    for uploaded_file in uploaded_files:

        file_id = (
            f"local:"
            f"{uploaded_file.name}:"
            f"{uploaded_file.size}"
        )

        if (
            file_id
            not in st.session_state.processed_files
        ):

            file_path = read_uploaded_file(
                uploaded_file
            )

            if file_path:

                records = extract_document(
                    file_path
                )

                if records:

                    new_records.extend(
                        records
                    )

                    st.session_state.documents.append(
                        {
                            "filename":
                                uploaded_file.name,

                            "source":
                                "Local Upload",

                            "characters":
                                sum(
                                    len(r["text"])
                                    for r in records
                                ),

                            "pages":
                                (
                                    len(records)
                                    if file_path.suffix.lower()
                                    == ".pdf"
                                    else None
                                ),
                        }
                    )

                    st.session_state.processed_files.add(
                        file_id
                    )


# ============================================================
# GOOGLE DRIVE PROCESSING
# ============================================================

if load_drive:

    if not drive_link.strip():

        st.warning(
            "Please paste a Google Drive "
            "or Google Docs link."
        )

    else:

        with st.spinner(
            "☁️ Loading document..."
        ):

            try:

                drive_files = (
                    download_drive_link(
                        drive_link
                    )
                )

                if not drive_files:

                    st.warning(
                        "No supported documents found. "
                        "Use PDF, DOCX, TXT or MD."
                    )

                else:

                    loaded_count = 0

                    for file_path in drive_files:

                        file_id = (
                            f"drive:"
                            f"{file_path.name}:"
                            f"{file_path.stat().st_size}"
                        )

                        if (
                            file_id
                            in st.session_state.processed_files
                        ):
                            continue

                        records = extract_document(
                            file_path
                        )

                        if not records:
                            continue

                        new_records.extend(
                            records
                        )

                        st.session_state.documents.append(
                            {
                                "filename":
                                    file_path.name,

                                "source":
                                    "Google Drive",

                                "characters":
                                    sum(
                                        len(r["text"])
                                        for r in records
                                    ),

                                "pages":
                                    (
                                        len(records)
                                        if file_path.suffix.lower()
                                        == ".pdf"
                                        else None
                                    ),
                            }
                        )

                        st.session_state.processed_files.add(
                            file_id
                        )

                        loaded_count += 1

                    if loaded_count:

                        st.success(
                            f"✓ Loaded "
                            f"{loaded_count} document(s)."
                        )

                    else:

                        st.info(
                            "These documents are "
                            "already loaded."
                        )

            except Exception as error:

                st.error(
                    f"Document loading failed: {error}"
                )


# ============================================================
# PROCESS NEW DOCUMENTS
# ============================================================

if new_records:

    with st.spinner(
        "🧠 Analyzing documents..."
    ):

        new_chunks, new_embeddings, _ = (
            process_documents(
                new_records
            )
        )

        if new_chunks:

            if (
                st.session_state.embeddings
                is None
            ):

                st.session_state.chunks = (
                    new_chunks
                )

                st.session_state.embeddings = (
                    new_embeddings
                )

            else:

                st.session_state.chunks = (
                    st.session_state.chunks
                    + new_chunks
                )

                st.session_state.embeddings = (
                    np.vstack(
                        [
                            st.session_state.embeddings,
                            new_embeddings,
                        ]
                    )
                )

            st.session_state.index = (
                build_faiss_index(
                    st.session_state.embeddings
                )
            )

    st.success(
        f"✓ Indexed "
        f"{len(new_chunks)} new text chunks."
    )


# ============================================================
# WORKSPACE OVERVIEW
# ============================================================

st.markdown(
    """
    <div class="section-title">
        Workspace Overview
    </div>
    """,
    unsafe_allow_html=True,
)


col1, col2, col3, col4 = st.columns(4)


with col1:

    st.metric(
        "📄 Documents",
        len(
            st.session_state.documents
        ),
    )


with col2:

    st.metric(
        "🧩 Text Chunks",
        len(
            st.session_state.chunks
        ),
    )


with col3:

    total_characters = sum(
        document["characters"]
        for document
        in st.session_state.documents
    )

    st.metric(
        "📝 Characters",
        f"{total_characters:,}",
    )


with col4:

    ai_status = (
        "Ready"
        if st.session_state.chunks
        else "Waiting"
    )

    st.metric(
        "⚡ AI Status",
        ai_status,
    )


# ============================================================
# DOCUMENT LIBRARY
# ============================================================

st.markdown(
    """
    <div class="section-title">
        📁 Document Library
    </div>
    """,
    unsafe_allow_html=True,
)


if st.session_state.documents:

    for document in (
        st.session_state.documents
    ):

        page_text = ""

        if document["pages"]:
            page_text = (
                f" • "
                f"{document['pages']} pages"
            )

        st.markdown(
            f"""
            <div class="document-card">

                <div class="document-name">
                    📄 {document["filename"]}
                </div>

                <div class="document-meta">
                    {document["source"]}
                    •
                    {document["characters"]:,}
                    characters
                    {page_text}
                </div>

            </div>
            """,
            unsafe_allow_html=True,
        )

else:

    st.markdown(
        """
        <div class="info-card">

            <b>📂 No documents loaded</b>

            <br><br>

            Upload a PDF, DOCX, TXT or MD file
            from the sidebar, or connect a
            public Google Drive document.

        </div>
        """,
        unsafe_allow_html=True,
    )


# ============================================================
# CHAT HISTORY
# ============================================================

for message in (
    st.session_state.chat_history
):

    with st.chat_message(
        message["role"]
    ):

        st.markdown(
            message["content"]
        )


# ============================================================
# ASK DOCUMENTS
# ============================================================

st.markdown(
    """
    <div class="section-title">
        💬 Ask Your Documents
    </div>
    """,
    unsafe_allow_html=True,
)


question = st.chat_input(
    "Ask anything about your documents..."
)


if question:

    if not st.session_state.chunks:

        st.warning(
            "Please upload or load a document first."
        )

    else:

        # ----------------------------------------------------
        # USER QUESTION
        # ----------------------------------------------------

        st.session_state.chat_history.append(
            {
                "role": "user",
                "content": question,
            }
        )

        with st.chat_message("user"):

            st.markdown(question)

        # ----------------------------------------------------
        # SEARCH
        # ----------------------------------------------------

        with st.spinner(
            "🔎 Searching your documents..."
        ):

            results = hybrid_search(
                question
            )

        # ----------------------------------------------------
        # AI ANSWER
        # ----------------------------------------------------

        with st.spinner(
            "🤖 Generating answer..."
        ):

            answer = answer_question(
                question,
                results,
            )

        st.session_state.chat_history.append(
            {
                "role": "assistant",
                "content": answer,
            }
        )

        # ----------------------------------------------------
        # ANSWER
        # ----------------------------------------------------

        with st.chat_message(
            "assistant"
        ):

            st.markdown(answer)

        # ----------------------------------------------------
        # SOURCES
        # ----------------------------------------------------

        st.markdown(
            """
            <div class="section-title">
                📚 Retrieved Sources
            </div>
            """,
            unsafe_allow_html=True,
        )

        if results:

            for i, result in enumerate(
                results,
                start=1,
            ):

                if result["page"]:

                    page_text = (
                        f"Page {result['page']}"
                    )

                else:

                    page_text = (
                        "Page unavailable"
                    )

                st.markdown(
                    f"""
                    <div class="source-card">

                        <div class="source-title">
                            {i}.
                            📄
                            {result["filename"]}
                        </div>

                        <div class="source-meta">
                            {page_text}
                            •
                            Relevance:
                            {result["hybrid_score"]:.3f}
                        </div>

                        <div class="source-text">
                            {result["text"]}
                        </div>

                    </div>
                    """,
                    unsafe_allow_html=True,
                )

        else:

            st.info(
                "No relevant document sources were found."
            )


# ============================================================
# FOOTER
# ============================================================

st.markdown(
    """
    <div class="footer">
        DocuMind AI • Intelligent Document Assistant
        <br>
        Semantic Search • Hybrid Retrieval • AI Answers
    </div>
    """,
    unsafe_allow_html=True,
)
