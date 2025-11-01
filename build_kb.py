import os
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISSfrom utils.embeddings import embedding_model  # from utils/embeddings.py

KB_FOLDER = "./kb_docs"
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 100
VECTOR_DB_PATH = "./vectorstore/faiss_index"

# 1️⃣ Load & split PDFs
all_docs = []

for filename in os.listdir(KB_FOLDER):
    if filename.endswith(".pdf"):
        pdf_path = os.path.join(KB_FOLDER, filename)
        loader = PyPDFLoader(pdf_path)
        docs = loader.load()

        splitter = RecursiveCharacterTextSplitter(
            chunk_size=CHUNK_SIZE,
            chunk_overlap=CHUNK_OVERLAP
        )
        docs = splitter.split_documents(docs)

        # Add metadata
        for doc in docs:
            doc.metadata["source"] = filename

        all_docs.extend(docs)

print(f"[INFO] Loaded and split {len(all_docs)} document chunks.")

# 2️⃣ Build FAISS vector store
vectorstore = FAISS.from_documents(all_docs, embedding_model)

# 3️⃣ Save locally
os.makedirs(os.path.dirname(VECTOR_DB_PATH), exist_ok=True)
vectorstore.save_local(VECTOR_DB_PATH)

print(f"[INFO] FAISS vector store saved at {VECTOR_DB_PATH}")
