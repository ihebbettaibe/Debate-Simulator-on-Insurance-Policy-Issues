"""
Hybrid Retriever combining semantic (FAISS) and keyword-based (BM25) search.
This provides more robust retrieval by leveraging both approaches.
"""
import os
import sys
from typing import List, Dict, Any
from langchain_community.vectorstores import FAISS
from langchain.docstore.document import Document
from rank_bm25 import BM25Okapi

# Add project root to path for imports
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from utils.embeddings import embedding_model

class HybridRetriever:
    """
    Combines FAISS (semantic) and BM25 (keyword) retrieval.
    Results are merged and re-ranked based on scores.
    """
    
    # Query expansion dictionary
    QUERY_EXPANSIONS = {
        'cyber insurance': ['cybersecurity insurance', 'data breach coverage', 'cyber risk insurance'],
        'climate risk': ['climate change', 'environmental risk', 'natural disasters', 'weather risk'],
        'insurtech': ['insurance technology', 'digital insurance', 'AI in insurance', 'fintech insurance'],
        'reinsurance': ['reinsurer', 're-insurance', 'risk transfer'],
        'parametric': ['parametric insurance', 'index-based insurance', 'parametric trigger'],
        'underwriting': ['risk assessment', 'policy pricing', 'risk evaluation'],
        'claims': ['claims processing', 'claims management', 'loss adjustment'],
        'actuarial': ['actuarial science', 'risk modeling', 'statistical analysis']
    }
    
    def __init__(self, vector_db_path: str, alpha: float = 0.5):
        """
        Initialize hybrid retriever.
        
        Args:
            vector_db_path: Path to FAISS index
            alpha: Weight for semantic search (0-1). BM25 weight = 1-alpha
        """
        self.vector_db_path = vector_db_path
        self.alpha = alpha
        self.vectorstore = None
        self.bm25 = None
        self.documents = []
        self.tokenized_corpus = []
        
        self._load_vectorstore()
        self._initialize_bm25()
    
    def expand_query(self, query: str) -> str:
        """
        Expand query with synonyms and related terms for better retrieval.
        
        Args:
            query: Original search query
        
        Returns:
            Expanded query string with related terms
        """
        expanded_terms = [query]
        query_lower = query.lower()
        
        # Check for matching terms and add expansions
        for key, synonyms in self.QUERY_EXPANSIONS.items():
            if key in query_lower:
                # Add up to 2 most relevant synonyms to avoid query bloat
                expanded_terms.extend(synonyms[:2])
        
        # Join with OR logic for better matching
        expanded_query = " ".join(expanded_terms)
        
        return expanded_query
    
    def _load_vectorstore(self):
        """Load FAISS vector store."""
        try:
            self.vectorstore = FAISS.load_local(
                self.vector_db_path, 
                embedding_model, 
                allow_dangerous_deserialization=True
            )
            print(f"✅ Loaded FAISS vectorstore from {self.vector_db_path}")
            
            # Extract documents for BM25 (use public accessors with fallbacks)
            self.documents = []
            # Prefer index_to_docstore_id mapping if available
            index_map = getattr(self.vectorstore, "index_to_docstore_id", None)
            docstore = getattr(self.vectorstore, "docstore", None)

            if index_map and docstore:
                # index_map can be a dict mapping index -> docstore_id or a list
                if isinstance(index_map, dict):
                    doc_ids = list(index_map.values())
                else:
                    doc_ids = list(index_map)

                for doc_id in doc_ids:
                    doc = None
                    # Try common public access patterns
                    if hasattr(docstore, "get"):
                        try:
                            doc = docstore.get(doc_id)
                        except Exception:
                            doc = None
                    if doc is None and hasattr(docstore, "__getitem__"):
                        try:
                            doc = docstore[doc_id]
                        except Exception:
                            doc = None
                    if doc is None and hasattr(docstore, "search"):
                        try:
                            doc = docstore.search(doc_id)
                        except Exception:
                            doc = None
                    # last-resort: read a public mapping attribute if present
                    if doc is None:
                        possible_dict = getattr(docstore, "docs", None) or getattr(docstore, "_dict", None)
                        if isinstance(possible_dict, dict):
                            doc = possible_dict.get(doc_id)
                    if doc:
                        self.documents.append(doc)
            else:
                # Generic fallback: try to find a public dict-like attribute on docstore
                docstore = getattr(self.vectorstore, "docstore", None)
                possible_dict = getattr(docstore, "docs", None) or getattr(docstore, "_dict", None)
                if isinstance(possible_dict, dict):
                    self.documents = list(possible_dict.values())
                elif docstore is not None:
                    try:
                        # Try to iterate over docstore if it supports it
                        self.documents = list(docstore)
                    except Exception:
                        self.documents = []
                else:
                    self.documents = []

            print(f"📚 Loaded {len(self.documents)} documents for hybrid search")
        except Exception as e:
            print(f"❌ Error loading vectorstore: {e}")
            raise
    
    def _initialize_bm25(self):
        """Initialize BM25 with tokenized corpus."""
        if not self.documents:
            print("⚠️ No documents available for BM25")
            return
        
        # Simple tokenization (can be enhanced with nltk/spacy)
        self.tokenized_corpus = [
            doc.page_content.lower().split() 
            for doc in self.documents
        ]
        self.bm25 = BM25Okapi(self.tokenized_corpus)
        print(f"✅ Initialized BM25 with {len(self.tokenized_corpus)} documents")
    
    def _semantic_search(self, query: str, k: int = 10) -> List[tuple]:
        """
        Perform semantic search using FAISS.
        
        Returns:
            List of (Document, score) tuples
        """
        if not self.vectorstore:
            print("⚠️ Vectorstore not available")
            return []
        
        results = self.vectorstore.similarity_search_with_score(query, k=k)
        # Convert to positive scores (FAISS uses distance, lower is better)
        # We'll normalize by converting to similarity
        return [(doc, 1 / (1 + score)) for doc, score in results]
    
    def _bm25_search(self, query: str, k: int = 10) -> List[tuple]:
        """
        Perform keyword search using BM25.
        
        Returns:
            List of (Document, score) tuples
        """
        if not self.bm25:
            print("⚠️ BM25 not available")
            return []
        
        tokenized_query = query.lower().split()
        scores = self.bm25.get_scores(tokenized_query)
        
        # Get top-k indices
        top_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:k]
        
        return [(self.documents[i], scores[i]) for i in top_indices]
    
    def hybrid_search(self, query: str, k: int = 5, semantic_k: int = 10, bm25_k: int = 10, use_expansion: bool = True) -> List[Document]:
        """
        Perform hybrid search combining semantic and keyword-based retrieval.
        
        Args:
            query: Search query
            k: Number of final results to return
            semantic_k: Number of results from semantic search
            bm25_k: Number of results from BM25 search
            use_expansion: Whether to use query expansion
        
        Returns:
            List of Documents ranked by hybrid score
        """
        # Expand query if enabled
        search_query = self.expand_query(query) if use_expansion else query
        
        # Get results from both methods
        semantic_results = self._semantic_search(search_query, k=semantic_k)
        bm25_results = self._bm25_search(search_query, k=bm25_k)
        
        # Normalize scores to 0-1 range for each method
        semantic_results = self._normalize_scores(semantic_results)
        bm25_results = self._normalize_scores(bm25_results)
        
        # Combine scores with weights
        combined_scores = {}
        
        # Add semantic scores
        for doc, score in semantic_results:
            doc_id = id(doc)
            combined_scores[doc_id] = {
                'doc': doc,
                'score': self.alpha * score
            }
        
        # Add BM25 scores
        for doc, score in bm25_results:
            doc_id = id(doc)
            if doc_id in combined_scores:
                combined_scores[doc_id]['score'] += (1 - self.alpha) * score
            else:
                combined_scores[doc_id] = {
                    'doc': doc,
                    'score': (1 - self.alpha) * score
                }
        
        # Sort by combined score
        ranked_results = sorted(
            combined_scores.values(), 
            key=lambda x: x['score'], 
            reverse=True
        )[:k]
        
        # Extract documents and add quality-based boosting
        results = []
        for item in ranked_results:
            doc = item['doc']
            score = item['score']
            
            # Boost score based on quality metadata
            quality_score = doc.metadata.get('quality_score', 0.5)
            boosted_score = score * (0.7 + 0.3 * quality_score)  # 70% original + 30% quality
            
            doc.metadata['hybrid_score'] = boosted_score
            results.append(doc)
        
        return results
    
    def _normalize_scores(self, results: List[tuple]) -> List[tuple]:
        """Normalize scores to 0-1 range."""
        if not results:
            return results
        
        scores = [score for _, score in results]
        min_score = min(scores)
        max_score = max(scores)
        
        if max_score == min_score:
            return [(doc, 1.0) for doc, _ in results]
        
        normalized = [
            (doc, (score - min_score) / (max_score - min_score))
            for doc, score in results
        ]
        return normalized
    
    def search(self, query: str, k: int = 5, method: str = "hybrid") -> List[Document]:
        """
        Unified search interface.
        
        Args:
            query: Search query
            k: Number of results
            method: "hybrid", "semantic", or "bm25"
        
        Returns:
            List of relevant Documents
        """
        if method == "semantic":
            results = self._semantic_search(query, k=k)
            return [doc for doc, _ in results]
        elif method == "bm25":
            results = self._bm25_search(query, k=k)
            return [doc for doc, _ in results]
        else:  # hybrid
            return self.hybrid_search(query, k=k)


# === Test Function ===
def test_hybrid_retriever():
    """Test the hybrid retriever."""
    VECTOR_DB_PATH = os.path.join(os.path.dirname(__file__), "..", "vectorstore", "faiss_index")
    
    print("\n" + "="*60)
    print("Testing Hybrid Retriever")
    print("="*60 + "\n")
    
    retriever = HybridRetriever(VECTOR_DB_PATH, alpha=0.5)
    
    query = "insurance technology trends 2025"
    
    print(f"\n🔍 Query: {query}\n")
    
    # Test all three methods
    for method in ["semantic", "bm25", "hybrid"]:
        print(f"\n--- {method.upper()} SEARCH ---")
        results = retriever.search(query, k=3, method=method)
        
        for i, doc in enumerate(results, 1):
            print(f"\n{i}. Source: {doc.metadata.get('source', 'Unknown')}")
            print(f"   Preview: {doc.page_content[:200]}...")


if __name__ == "__main__":
    test_hybrid_retriever()
