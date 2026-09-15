import os
import re
import tempfile
from pathlib import Path

import faiss
import gdown
import numpy as np
import streamlit as st
from docx import Document
from groq import Groq
from pypdf import PdfReader
from sentence_transformers import SentenceTransformer


# -----------------------------
# App settings
# -----------------------------
st.set_page_config(page_title="AI Document Assistant", page_icon="📄", layout="wide")

SUPPORTED_EXTENSIONS = {".pdf", ".docx", ".txt", ".md"}
CHUNK_SIZE = 700
CHUNK_OVERLAP = 120
TOP_K = 5


# -----------------------------
# Session state
# -----------------------------
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


# -----------------------------
# Cached models / clients
# -----------------------------
@st.cache_resource
def load_embedding_model():
    return SentenceTransformer("all-MiniLM-L6-v2")


@st.cache_resource
def load_groq_client():
    api_key = st.secrets.get("GROQ_API_KEY", os.getenv("GROQ_API_KEY"))
    if not api_key:
        return None
    return Groq(api_key=api_key)


# -----------------------------
# Document extraction
# -----------------------------
def extract_pdf(file_path):
    """Return one text record per PDF page."""
    reader = PdfReader(file_path)
    records = []

    for page_number, page in enumerate(reader.pages, start=1):
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
    """Return the DOCX text. DOCX paragraphs do not reliably contain page numbers."""
    document = Document(file_path)
    paragraphs = [p.text.strip() for p in document.paragraphs if p.text.strip()]
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
    """Return TXT text."""
    text = Path(file_path).read_text(encoding="utf-8", errors="ignore").strip()

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
    """Return Markdown text."""
    text = Path(file_path).read_text(encoding="utf-8", errors="ignore").strip()

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
    """Choose the correct extraction function based on file extension."""
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


# -----------------------------
# Text chunking
# -----------------------------
def split_text(text, chunk_size=CHUNK_SIZE, overlap=CHUNK_OVERLAP):
    """Split text into overlapping character-based chunks."""
    text = re.sub(r"\s+", " ", text).strip()

    if not text:
        return []

    chunks = []
    start = 0

    while start < len(text):
        end = min(start + chunk_size, len(text))
        chunks.append(text[start:end].strip())

        if end == len(text):
            break

        start = max(end - overlap, start + 1)

    return chunks


def create_chunks(records):
    """Create chunks while keeping filename and page metadata."""
    all_chunks = []

    for record in records:
        pieces = split_text(record["text"])

        for piece in pieces:
            all_chunks.append(
                {
                    "filename": record["filename"],
                    "page": record["page"],
                    "text": piece,
                }
            )

    return all_chunks


# -----------------------------
# Embeddings + FAISS
# -----------------------------
def create_embeddings(chunks):
    """Create embeddings once for all chunks."""
    model = load_embedding_model()
    texts = [chunk["text"] for chunk in chunks]

    embeddings = model.encode(
        texts,
        convert_to_numpy=True,
        normalize_embeddings=True,
        show_progress_bar=False,
    ).astype("float32")

    return embeddings


def build_faiss_index(embeddings):
    """Build an inner-product FAISS index for normalized embeddings."""
    dimension = embeddings.shape[1]
    index = faiss.IndexFlatIP(dimension)
    index.add(embeddings)
    return index


def process_documents(records):
    """Chunk, embed and index documents one time."""
    chunks = create_chunks(records)

    if not chunks:
        return [], None, None

    embeddings = create_embeddings(chunks)
    index = build_faiss_index(embeddings)

    return chunks, embeddings, index


# -----------------------------
# Keyword search
# -----------------------------
STOPWORDS = {
    "the", "a", "an", "and", "or", "is", "are", "was", "were",
    "to", "of", "in", "on", "for", "with", "what", "which",
    "who", "when", "where", "why", "how", "does", "do", "did",
    "this", "that", "these", "those", "from", "as", "by", "be",
    "can", "could", "would", "should", "about", "it", "its",
}


def important_words(question):
    words = re.findall(r"\b[a-zA-Z0-9]+\b", question.lower())
    return [word for word in words if word not in STOPWORDS and len(word) > 2]


def keyword_scores(question, chunks):
    """Score chunks by the fraction of important query words they contain."""
    query_words = set(important_words(question))
    scores = np.zeros(len(chunks), dtype="float32")

    if not query_words:
        return scores

    for i, chunk in enumerate(chunks):
        chunk_words = set(re.findall(r"\b[a-zA-Z0-9]+\b", chunk["text"].lower()))
        matches = len(query_words.intersection(chunk_words))
        scores[i] = matches / len(query_words)

    return scores


# -----------------------------
# Hybrid search
# -----------------------------
def hybrid_search(question, top_k=TOP_K):
    """Combine semantic similarity and keyword matching."""
    if not st.session_state.chunks or st.session_state.index is None:
        return []

    model = load_embedding_model()

    question_embedding = model.encode(
        [question],
        convert_to_numpy=True,
        normalize_embeddings=True,
    ).astype("float32")

    # Semantic scores for every chunk.
    semantic_scores = st.session_state.embeddings @ question_embedding[0]

    # Keyword scores for every chunk.
    keyword_score = keyword_scores(question, st.session_state.chunks)

    # Normalize semantic score from roughly [-1, 1] to [0, 1].
    semantic_score = (semantic_scores + 1.0) / 2.0

    # Hybrid ranking: semantic meaning + exact important words.
    hybrid_score = (0.70 * semantic_score) + (0.30 * keyword_score)

    ranked_indices = np.argsort(hybrid_score)[::-1][:top_k]

    results = []

    for index in ranked_indices:
        result = dict(st.session_state.chunks[index])
        result["semantic_score"] = float(semantic_score[index])
        result["keyword_score"] = float(keyword_score[index])
        result["hybrid_score"] = float(hybrid_score[index])
        results.append(result)

    return results


# -----------------------------
# Groq answer generation
# -----------------------------
def answer_question(question, results):
    client = load_groq_client()

    if client is None:
        return "GROQ_API_KEY is not configured. Add it to Streamlit secrets."

    context_parts = []

    for i, result in enumerate(results, start=1):
        page = f", page {result['page']}" if result["page"] else ""
        context_parts.append(
            f"[Source {i}: {result['filename']}{page}]\n{result['text']}"
        )

    context = "\n\n".join(context_parts)

    prompt = f"""
You are a document question-answering assistant.

Answer the user's question using ONLY the provided document context.
Do not use outside knowledge.
If the answer is not available in the context, clearly say:
"I could not find that information in the provided documents."

Keep the answer clear and concise.

DOCUMENT CONTEXT:
{context}

USER QUESTION:
{question}
"""

    response = client.chat.completions.create(
       model="openai/gpt-oss-120b",
        messages=[
            {
                "role": "system",
                "content": "Answer only from the supplied document context.",
            },
            {"role": "user", "content": prompt},
        ],
        temperature=0,
    )

    return response.choices[0].message.content


# -----------------------------
# Google Drive
# -----------------------------
def normalize_google_drive_link(drive_link):
    """Clean a Google Drive URL and detect whether it is a file or folder."""
    import re
    from urllib.parse import urlparse

    link = drive_link.strip()

    # Folder: https://drive.google.com/drive/folders/FOLDER_ID?usp=drive_link
    folder_match = re.search(r"/folders/([a-zA-Z0-9_-]+)", link)
    if folder_match:
        folder_id = folder_match.group(1)
        return {
            "type": "folder",
            "id": folder_id,
            "url": f"https://drive.google.com/drive/folders/{folder_id}",
        }

    # File: https://drive.google.com/file/d/FILE_ID/view?usp=sharing
    file_match = re.search(r"/file/d/([a-zA-Z0-9_-]+)", link)
    if file_match:
        file_id = file_match.group(1)
        return {
            "type": "file",
            "id": file_id,
            "url": f"https://drive.google.com/file/d/{file_id}/view",
        }

    # Also support https://drive.google.com/open?id=FILE_ID
    parsed = urlparse(link)
    query = dict(
        item.split("=", 1)
        for item in parsed.query.split("&")
        if "=" in item
    )

    if "id" in query:
        file_id = query["id"]
        return {
            "type": "file",
            "id": file_id,
            "url": f"https://drive.google.com/uc?id={file_id}",
        }

    raise ValueError(
        "Invalid Google Drive link. Paste a Drive file or folder sharing link."
    )


def download_drive_link(drive_link):
    """Download a public Google Drive file or folder."""
    temp_dir = Path(tempfile.mkdtemp(prefix="drive_docs_"))
    drive = normalize_google_drive_link(drive_link)

    if drive["type"] == "folder":
        # Pass the clean folder URL. This avoids treating ?usp=drive_link
        # or other query parameters as part of the folder ID.
        gdown.download_folder(
            url=drive["url"],
            output=str(temp_dir),
            quiet=True,
            use_cookies=False,
        )
    else:
        # Ask gdown for the real Drive filename first so the extension is kept.
        metadata = gdown.download(
            url=drive["url"],
            output=None,
            quiet=True,
            use_cookies=False,
            skip_download=True,
        )

        if metadata is None or not metadata.path:
            raise ValueError(
                "Could not read the Drive file information. "
                "Check that the file is shared as 'Anyone with the link'."
            )

        filename = Path(metadata.path).name
        output_path = temp_dir / filename
        gdown.download(
            url=drive["url"],
            output=str(output_path),
            quiet=True,
            use_cookies=False,
        )

    return [
        path
        for path in temp_dir.rglob("*")
        if path.is_file() and path.suffix.lower() in SUPPORTED_EXTENSIONS
    ]

def read_uploaded_file(uploaded_file):
    """Save a Streamlit UploadedFile to a temporary path."""
    suffix = Path(uploaded_file.name).suffix.lower()

    if suffix not in SUPPORTED_EXTENSIONS:
        return None

    temp_dir = tempfile.mkdtemp(prefix="local_docs_")
    file_path = Path(temp_dir) / uploaded_file.name
    file_path.write_bytes(uploaded_file.getvalue())

    return file_path


# -----------------------------
# Main UI
# -----------------------------
st.title("📄 Simple Streamlit AI Document Assistant")
st.write(
    "Upload documents or load public Google Drive files. "
    "The app extracts, chunks, embeds and indexes documents once, "
    "then uses hybrid search to answer questions."
)

with st.sidebar:
    st.header("Document Sources")

    uploaded_files = st.file_uploader(
        "Upload PDF, DOCX, TXT or MD files",
        type=["pdf", "docx", "txt", "md"],
        accept_multiple_files=True,
    )

    st.subheader("Google Drive")
    drive_link = st.text_input(
        "Paste a public/shared Drive file or folder link"
    )

    load_drive = st.button("Load Google Drive")

    if st.button("Clear all documents"):
        st.session_state.documents = []
        st.session_state.chunks = []
        st.session_state.embeddings = None
        st.session_state.index = None
        st.session_state.processed_files = set()
        st.rerun()


# -----------------------------
# Local uploads
# -----------------------------
new_records = []

if uploaded_files:
    for uploaded_file in uploaded_files:
        file_id = f"local:{uploaded_file.name}:{uploaded_file.size}"

        if file_id not in st.session_state.processed_files:
            file_path = read_uploaded_file(uploaded_file)

            if file_path:
                records = extract_document(file_path)
                new_records.extend(records)

                st.session_state.documents.append(
                    {
                        "filename": uploaded_file.name,
                        "source": "Local upload",
                        "characters": sum(len(r["text"]) for r in records),
                        "pages": len(records) if Path(file_path).suffix.lower() == ".pdf" else None,
                    }
                )

                st.session_state.processed_files.add(file_id)


# -----------------------------
# Google Drive loading
# -----------------------------
if load_drive:
    if not drive_link.strip():
        st.warning("Please paste a Google Drive file or folder link.")
    else:
        with st.spinner("Loading Google Drive files..."):
            try:
                drive_files = download_drive_link(drive_link)

                if not drive_files:
                    st.warning(
                        "No supported files were found. Make sure the Drive "
                        "file/folder is publicly accessible and contains PDF, "
                        "DOCX, TXT or MD files."
                    )
                else:
                    for file_path in drive_files:
                        file_id = f"drive:{file_path.name}:{file_path.stat().st_size}"

                        if file_id in st.session_state.processed_files:
                            continue

                        records = extract_document(file_path)
                        new_records.extend(records)

                        st.session_state.documents.append(
                            {
                                "filename": file_path.name,
                                "source": "Google Drive",
                                "characters": sum(len(r["text"]) for r in records),
                                "pages": len(records) if file_path.suffix.lower() == ".pdf" else None,
                            }
                        )

                        st.session_state.processed_files.add(file_id)

                    st.success(f"Loaded {len(drive_files)} supported file(s).")

            except Exception as error:
                st.error(f"Google Drive loading failed: {error}")


# -----------------------------
# Process only newly loaded text
# -----------------------------
if new_records:
    with st.spinner("Extracting, chunking and creating embeddings..."):
        new_chunks, new_embeddings, _ = process_documents(new_records)

        if new_chunks:
            # Keep one combined chunk list.
            old_chunks = st.session_state.chunks

            if st.session_state.embeddings is None:
                st.session_state.chunks = new_chunks
                st.session_state.embeddings = new_embeddings
            else:
                st.session_state.chunks = old_chunks + new_chunks
                st.session_state.embeddings = np.vstack(
                    [st.session_state.embeddings, new_embeddings]
                )

            st.session_state.index = build_faiss_index(
                st.session_state.embeddings
            )

    st.success(
        f"Processed {len(new_records)} extracted sections and created "
        f"{len(new_chunks)} new chunks."
    )


# -----------------------------
# Document information
# -----------------------------
st.subheader("Document Information")

if st.session_state.documents:
    for document in st.session_state.documents:
        pages = (
            f", {document['pages']} pages"
            if document["pages"] is not None
            else ""
        )

        st.write(
            f"**{document['filename']}** — "
            f"{document['source']} — "
            f"{document['characters']:,} characters{pages}"
        )

    st.info(f"Total chunks created: {len(st.session_state.chunks)}")

else:
    st.info("No documents loaded yet.")


# -----------------------------
# Ask questions
# -----------------------------
st.subheader("Ask a Question")

question = st.text_input(
    "Question",
    placeholder="What is this document about?",
)

if question:
    if not st.session_state.chunks:
        st.warning("Please upload or load a document first.")
    else:
        with st.spinner("Searching documents..."):
            results = hybrid_search(question)

        with st.spinner("Generating answer..."):
            answer = answer_question(question, results)

        st.markdown("### Answer")
        st.write(answer)

        st.markdown("### Retrieved Sources")

        for i, result in enumerate(results, start=1):
            page = (
                f"Page {result['page']}"
                if result["page"] is not None
                else "Page not available"
            )

            st.markdown(
                f"**{i}. {result['filename']} — {page}**  \n"
                f"Hybrid score: `{result['hybrid_score']:.3f}`"
            )
            st.caption(result["text"])
            st.divider()
