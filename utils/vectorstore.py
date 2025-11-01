import os
import sys
from langchain_community.vectorstores import FAISS
from langchain.docstore.document import Document

# Robust import: prefer package-relative import, but when the file is executed
# directly (python utils/vectorstore.py) the package context is missing. In
# that case, add the project root to sys.path so `utils` can be imported.
try:
    # When run as part of the package
    from .embeddings import embedding_model, pdf_to_chunks
except Exception:
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    if project_root not in sys.path:
        sys.path.insert(0, project_root)
    try:
        # Try the package import which may work now that project root is on sys.path
        from utils.embeddings import embedding_model, pdf_to_chunks
    except Exception:
        # Fallback: provide minimal local implementations so the script can run
        # without depending on the possibly partially-initialized module.
        print("ℹ️ Using local fallback for embeddings and pdf chunking")
        try:
            from langchain_huggingface import HuggingFaceEmbeddings
        except Exception:
            HuggingFaceEmbeddings = None

        if HuggingFaceEmbeddings is not None:
            embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
        else:
            embedding_model = None

        # Minimal PDF helpers
        try:
            from PyPDF2 import PdfReader
        except Exception:
            PdfReader = None

        def extract_text_from_pdf(pdf_path):
            text = ""
            if PdfReader is None:
                print(f"⚠️ PdfReader unavailable, cannot extract from {pdf_path}")
                return text
            try:
                reader = PdfReader(pdf_path)
                for page in reader.pages:
                    text += page.extract_text() or ""
            except Exception as e:
                print(f"❌ Error reading {pdf_path}: {e}")
            return text

        def chunk_text(text, chunk_size=500, overlap=50):
            chunks = []
            start = 0
            while start < len(text):
                end = min(start + chunk_size, len(text))
                chunks.append(text[start:end])
                start += chunk_size - overlap
            return chunks

        def pdf_to_chunks(pdf_path):
            text = extract_text_from_pdf(pdf_path)
            if text:
                return chunk_text(text)
            return []

# === Paths ===
VECTOR_DB_PATH = os.path.join(os.path.dirname(__file__), "..", "vector_db")

def index_pdf(pdf_path, metadata=None):
    """Convert PDF to embeddings and store in FAISS."""
    chunks = pdf_to_chunks(pdf_path)
    docs = [Document(page_content=chunk, metadata=metadata or {"source": pdf_path}) for chunk in chunks]

    if not os.path.exists(VECTOR_DB_PATH):
        os.makedirs(VECTOR_DB_PATH, exist_ok=True)
        vectorstore = FAISS.from_documents(docs, embedding_model)
        vectorstore.save_local(VECTOR_DB_PATH)
        print(f"✅ Indexed and created new FAISS DB at {VECTOR_DB_PATH}")
    else:
        # Load existing vectorstore
        vectorstore = FAISS.load_local(VECTOR_DB_PATH, embedding_model, allow_dangerous_deserialization=True)
        vectorstore.add_documents(docs)
        vectorstore.save_local(VECTOR_DB_PATH)
        print(f"✅ Indexed PDF and updated FAISS DB at {VECTOR_DB_PATH}")

def index_all_pdfs(pdf_dir):
    """Index all PDFs in a directory."""
    for root, _, files in os.walk(pdf_dir):
        for f in files:
            if f.lower().endswith(".pdf"):
                pdf_path = os.path.join(root, f)
                index_pdf(pdf_path)

if __name__ == "__main__":
    # example: index all PDFs under kb_docs
    pdf_folder = os.path.join(os.path.dirname(__file__), "..", "kb_docs")
    index_all_pdfs(pdf_folder)
