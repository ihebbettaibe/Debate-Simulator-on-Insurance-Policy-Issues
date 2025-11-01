from langchain_community.vectorstores import FAISS
from utils.embeddings import embedding_model

VECTOR_DB_PATH = "../vector_db"

def test_query(query):
    # load vectorstore
    vectorstore = FAISS.load_local(VECTOR_DB_PATH, embedding_model, allow_dangerous_deserialization=True)
    
    # search top 3 most relevant chunks
    results = vectorstore.similarity_search(query, k=3)
    
    print(f"\nTop results for: {query}\n")
    for i, r in enumerate(results, 1):
        print(f"--- Result {i} ---")
        print(f"Source: {r.metadata.get('source')}")
        print(f"Text Preview: {r.page_content[:300]}...\n")

if __name__ == "__main__":
    test_query("insurance technology trends 2025")
