# document_processor.py
# Handles parsing and chunking of PDF and DOCX files
# Extracts clean text and splits into manageable chunks for embedding

import fitz  # PyMuPDF
from docx import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from typing import List, Dict
import os


# Initialize text splitter
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=150,
    separators=["\n\n", "\n", ".", " ", ""]
)


def extract_text_from_pdf(file_path: str, max_pages: int = 50) -> str:
    """Extract text from PDF. Max 50 pages to keep performance stable."""
    text = ""
    try:
        doc = fitz.open(file_path)
        total_pages = len(doc)

        if total_pages > max_pages:
            print(f"[DocProcessor] Document has {total_pages} pages. Processing first {max_pages} only.")

        pages_to_process = min(total_pages, max_pages)

        for page_num in range(pages_to_process):
            page = doc[page_num]
            page_text = page.get_text()
            if page_text.strip():
                text += f"\n[Page {page_num + 1}]\n{page_text}"

        doc.close()
        print(f"[DocProcessor] Extracted text from {pages_to_process}/{total_pages} pages.")
    except Exception as e:
        print(f"[DocProcessor] PDF extraction error: {e}")
    return text


def extract_text_from_docx(file_path: str) -> str:
    """Extract all text from a DOCX file using python-docx."""
    text = ""
    try:
        doc = Document(file_path)
        for para in doc.paragraphs:
            if para.text.strip():
                text += para.text + "\n"
        print(f"[DocProcessor] Extracted text from DOCX successfully.")
    except Exception as e:
        print(f"[DocProcessor] DOCX extraction error: {e}")
    return text


def chunk_text(text: str) -> List[str]:
    """
    Split extracted text into smaller overlapping chunks.
    Smaller chunks = more precise retrieval during Q&A.
    """
    chunks = text_splitter.split_text(text)
    print(f"[DocProcessor] Created {len(chunks)} chunks.")
    return chunks


def process_document(file_path: str) -> Dict:
    """
    Master function — takes a file path, detects type,
    extracts text, chunks it, and returns everything needed
    for embedding and storing in Endee.

    Returns:
        {
            "filename": "report.pdf",
            "chunks": ["chunk1 text...", "chunk2 text...", ...],
            "metadata": [{"text": "chunk1", "source": "report.pdf", "chunk_id": 0}, ...]
        }
    """
    filename = os.path.basename(file_path)
    extension = filename.split(".")[-1].lower()

    # Extract based on file type
    if extension == "pdf":
        raw_text = extract_text_from_pdf(file_path)
    elif extension == "docx":
        raw_text = extract_text_from_docx(file_path)
    else:
        print(f"[DocProcessor] Unsupported file type: {extension}")
        return {}

    if not raw_text.strip():
        print("[DocProcessor] No text extracted from document.")
        return {}

    # Chunk the text
    chunks = chunk_text(raw_text)

    # Safety cap — max 500 chunks to keep Endee fast
    # and embedding time under control
    if len(chunks) > 500:
        print(f"[DocProcessor] Capping chunks from {len(chunks)} to 500.")
        chunks = chunks[:500]

    # Build metadata for each chunk
    # Stored alongside vectors in Endee so we know the source on retrieval
    metadata = [
        {
            "text": chunk,
            "source": filename,
            "chunk_id": i
        }
        for i, chunk in enumerate(chunks)
    ]

    return {
        "filename": filename,
        "chunks": chunks,
        "metadata": metadata
    }