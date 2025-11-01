import os
from langchain_huggingface import HuggingFaceEmbeddings
from PyPDF2 import PdfReader

# === Paths ===
BASE_DIR = os.path.join(os.path.dirname(__file__), "..", "kb_docs")

# === Embedding model ===
embedding_model = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

# === Helpers ===
def extract_text_from_pdf(pdf_path):
    """Extract text from a PDF."""
    text = ""
    try:
        reader = PdfReader(pdf_path)
        for page in reader.pages:
            text += page.extract_text() or ""
    except Exception as e:
        print(f"❌ Error reading {pdf_path}: {e}")
    return text

def chunk_text(text, chunk_size=500, overlap=50):
    """Split text into chunks for embedding."""
    chunks = []
    start = 0
    while start < len(text):
        end = min(start + chunk_size, len(text))
        chunks.append(text[start:end])
        start += chunk_size - overlap
    return chunks

def pdf_to_chunks(pdf_path):
    """Extract text and split into chunks from a single PDF."""
    text = extract_text_from_pdf(pdf_path)
    if text:
        return chunk_text(text)
    return []
