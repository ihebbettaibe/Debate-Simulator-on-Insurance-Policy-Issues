"""
Dynamic Knowledge Base Manager
Manages automatic updates, web scraping, and document indexing for the knowledge base.
"""
import os
import sys
import json
import hashlib
from datetime import datetime, timedelta
from typing import List, Dict, Optional, Set
from pathlib import Path
import time

from langchain.docstore.document import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import (
    PyPDFLoader, 
    TextLoader,
    DirectoryLoader
)
from langchain_community.vectorstores import FAISS
import numpy as np

# Add project root to path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from utils.embeddings import embedding_model
from retriever.dynamic_scraper import DynamicWebScraper


class DynamicKBManager:
    """
    Manages dynamic knowledge base with auto-updates and web scraping.
    """
    
    # Query expansion dictionary for semantic search improvements
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
    
    # Trusted sources for quality scoring
    TRUSTED_DOMAINS = [
        'swissre.com', 'munichre.com', 'lloyds.com', 'iii.org', 'naic.org',
        'insurancejournal.com', 'artemis.bm', 'reinsurancene.ws', 'am-best.com'
    ]
    
    def __init__(
        self,
        kb_folder: str = "./kb_docs",
        vector_db_path: str = "./vectorstore/faiss_index",
        metadata_path: str = "./vectorstore/kb_metadata.json",
        chunk_size: int = 1000,
        chunk_overlap: int = 100,
        auto_update_hours: int = 24
    ):
        """
        Initialize Dynamic KB Manager.
        
        Args:
            kb_folder: Folder containing source documents
            vector_db_path: Path to save FAISS index
            metadata_path: Path to save metadata about indexed docs
            chunk_size: Text chunk size
            chunk_overlap: Chunk overlap
            auto_update_hours: Hours between automatic updates
        """
        self.kb_folder = kb_folder
        self.vector_db_path = vector_db_path
        self.metadata_path = metadata_path
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.auto_update_hours = auto_update_hours
        
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap
        )
        self.web_scraper = DynamicWebScraper(max_results=5)
        
        # Ensure directories exist
        os.makedirs(os.path.dirname(self.vector_db_path), exist_ok=True)
        os.makedirs(self.kb_folder, exist_ok=True)
        
        # Load metadata
        self.metadata = self._load_metadata()
    
    def expand_query(self, query: str) -> List[str]:
        """
        Expand query with synonyms and related terms for better retrieval.
        
        Args:
            query: Original search query
        
        Returns:
            List of expanded queries including original
        """
        queries = [query]
        query_lower = query.lower()
        
        # Check for matching terms and add expansions
        for key, synonyms in self.QUERY_EXPANSIONS.items():
            if key in query_lower:
                # Add synonyms
                queries.extend(synonyms)
                print(f"🔍 Expanded '{key}' with {len(synonyms)} related terms")
        
        # Remove duplicates while preserving order
        seen = set()
        expanded = []
        for q in queries:
            if q.lower() not in seen:
                seen.add(q.lower())
                expanded.append(q)
        
        if len(expanded) > 1:
            print(f"✅ Query expanded: {len(expanded)} variations")
        
        return expanded
    
    def score_document_quality(self, doc: Document) -> float:
        """
        Score document quality based on source, length, metadata, and freshness.
        
        Args:
            doc: Document to score
        
        Returns:
            Quality score between 0.0 and 1.0
        """
        score = 0.5  # Base score
        
        # 1. Trusted sources get higher scores (up to +0.3)
        source = doc.metadata.get('source', '').lower()
        if any(domain in source for domain in self.TRUSTED_DOMAINS):
            score += 0.3
            doc.metadata['trusted_source'] = True
        else:
            doc.metadata['trusted_source'] = False
        
        # 2. Content length indicates thoroughness (up to +0.2)
        content_length = len(doc.page_content)
        if content_length > 1000:
            score += 0.2
        elif content_length > 500:
            score += 0.1
        elif content_length < 100:
            score -= 0.2  # Penalize very short content
        
        # 3. Recent content scores higher (up to +0.2)
        indexed_at = doc.metadata.get('indexed_at')
        if indexed_at:
            try:
                age_days = (datetime.now() - datetime.fromisoformat(indexed_at)).days
                if age_days < 30:
                    score += 0.2
                elif age_days < 90:
                    score += 0.1
                elif age_days > 365:
                    score -= 0.1  # Penalize very old content
            except:
                pass
        
        # 4. Static documents are generally more reliable than web scrapes (up to +0.1)
        if doc.metadata.get('type') == 'static_document':
            score += 0.1
        
        # Clamp score between 0 and 1
        final_score = max(0.0, min(1.0, score))
        doc.metadata['quality_score'] = final_score
        
        return final_score
    
    def deduplicate_documents(self, docs: List[Document], similarity_threshold: float = 0.95) -> List[Document]:
        """
        Remove duplicate documents based on content similarity.
        
        Args:
            docs: List of documents to deduplicate
            similarity_threshold: Cosine similarity threshold (0.0-1.0)
        
        Returns:
            List of unique documents
        """
        if len(docs) < 2:
            return docs
        
        print(f"🔍 Deduplicating {len(docs)} documents...")
        
        try:
            # Get embeddings for all documents (use truncated content for speed)
            embeddings = []
            for doc in docs:
                # Use first 500 chars for faster embedding
                content = doc.page_content[:500]
                emb = embedding_model.embed_query(content)
                embeddings.append(emb)
            
            embeddings_array = np.array(embeddings)
            
            # Calculate cosine similarity matrix
            from sklearn.metrics.pairwise import cosine_similarity
            similarity_matrix = cosine_similarity(embeddings_array)
            
            # Keep only unique documents
            unique_indices = []
            for i in range(len(docs)):
                is_unique = True
                # Check against previously selected documents
                for j in unique_indices:
                    if similarity_matrix[i][j] > similarity_threshold:
                        is_unique = False
                        print(f"  ⚠️ Duplicate found: docs {i} and {j} (similarity: {similarity_matrix[i][j]:.3f})")
                        break
                
                if is_unique:
                    unique_indices.append(i)
            
            unique_docs = [docs[i] for i in unique_indices]
            removed_count = len(docs) - len(unique_docs)
            
            if removed_count > 0:
                print(f"✅ Removed {removed_count} duplicate documents")
            else:
                print(f"✅ No duplicates found")
            
            return unique_docs
            
        except ImportError:
            print("⚠️ scikit-learn not installed. Skipping deduplication.")
            print("   Install with: pip install scikit-learn")
            return docs
        except Exception as e:
            print(f"⚠️ Error during deduplication: {e}")
            return docs
    
    def _load_metadata(self) -> Dict:
        """Load or initialize metadata."""
        if os.path.exists(self.metadata_path):
            try:
                with open(self.metadata_path, 'r', encoding='utf-8') as f:
                    return json.load(f)
            except Exception as e:
                print(f"⚠️ Error loading metadata: {e}")
        
        return {
            'indexed_files': {},
            'last_update': None,
            'total_documents': 0,
            'web_sources': [],
            'version': '1.0'
        }
    
    def _save_metadata(self):
        """Save metadata to disk."""
        try:
            with open(self.metadata_path, 'w', encoding='utf-8') as f:
                json.dump(self.metadata, f, indent=2)
            print(f"✅ Metadata saved to {self.metadata_path}")
        except Exception as e:
            print(f"❌ Error saving metadata: {e}")
    
    def _get_file_hash(self, filepath: str) -> str:
        """Get MD5 hash of a file."""
        hasher = hashlib.md5()
        try:
            with open(filepath, 'rb') as f:
                for chunk in iter(lambda: f.read(4096), b""):
                    hasher.update(chunk)
            return hasher.hexdigest()
        except Exception as e:
            print(f"⚠️ Error hashing {filepath}: {e}")
            return ""
    
    def scan_new_documents(self) -> List[str]:
        """
        Scan kb_folder for new or modified documents.
        
        Returns:
            List of file paths that need indexing
        """
        new_files = []
        
        for root, _, files in os.walk(self.kb_folder):
            for filename in files:
                if filename.endswith(('.pdf', '.txt', '.md')):
                    filepath = os.path.join(root, filename)
                    rel_path = os.path.relpath(filepath, self.kb_folder)
                    
                    # Check if file is new or modified
                    file_hash = self._get_file_hash(filepath)
                    
                    if rel_path not in self.metadata['indexed_files']:
                        print(f"🆕 New file: {rel_path}")
                        new_files.append(filepath)
                    elif self.metadata['indexed_files'][rel_path].get('hash') != file_hash:
                        print(f"🔄 Modified file: {rel_path}")
                        new_files.append(filepath)
        
        return new_files
    
    def load_documents(self, file_paths: Optional[List[str]] = None) -> List[Document]:
        """
        Load documents from files.
        
        Args:
            file_paths: Specific files to load (if None, loads all from kb_folder)
        
        Returns:
            List of loaded Documents
        """
        all_docs = []
        
        if file_paths is None:
            # Load all supported files
            file_paths = []
            for root, _, files in os.walk(self.kb_folder):
                for filename in files:
                    if filename.endswith(('.pdf', '.txt', '.md')):
                        file_paths.append(os.path.join(root, filename))
        
        for filepath in file_paths:
            try:
                # Load based on file type
                if filepath.endswith('.pdf'):
                    loader = PyPDFLoader(filepath)
                    docs = loader.load()
                elif filepath.endswith(('.txt', '.md')):
                    loader = TextLoader(filepath, encoding='utf-8')
                    docs = loader.load()
                else:
                    continue
                
                # Split documents
                docs = self.text_splitter.split_documents(docs)
                
                # Add metadata
                rel_path = os.path.relpath(filepath, self.kb_folder)
                for doc in docs:
                    doc.metadata.update({
                        'source': rel_path,
                        'file_path': filepath,
                        'indexed_at': datetime.now().isoformat(),
                        'type': 'static_document'
                    })
                    # Score document quality
                    self.score_document_quality(doc)
                
                all_docs.extend(docs)
                
                # Update metadata
                self.metadata['indexed_files'][rel_path] = {
                    'hash': self._get_file_hash(filepath),
                    'indexed_at': datetime.now().isoformat(),
                    'chunks': len(docs)
                }
                
                print(f"✅ Loaded {len(docs)} chunks from {rel_path}")
                
            except Exception as e:
                print(f"❌ Error loading {filepath}: {e}")
        
        return all_docs
    
    def scrape_web_sources(self, queries: List[str], sites: Optional[List[str]] = None) -> List[Document]:
        """
        Scrape web for fresh insurance content.
        
        Args:
            queries: Search queries
            sites: Optional list of specific sites to search
        
        Returns:
            List of scraped Documents
        """
        all_docs = []
        
        for query in queries:
            print(f"\n🌐 Scraping web for: {query}")
            
            if sites:
                docs = self.web_scraper.targeted_search(query, sites=sites)
            else:
                docs = self.web_scraper.scrape_to_documents(query, fetch_content=False)
            
            # Split long documents
            docs = self.text_splitter.split_documents(docs)
            
            # Add indexing metadata
            for doc in docs:
                doc.metadata['indexed_at'] = datetime.now().isoformat()
                doc.metadata['type'] = 'web_scraped'
                # Score document quality
                self.score_document_quality(doc)
            
            all_docs.extend(docs)
            
            # Track web sources
            source_info = {
                'query': query,
                'sites': sites,
                'scraped_at': datetime.now().isoformat(),
                'doc_count': len(docs)
            }
            self.metadata['web_sources'].append(source_info)
            
            print(f"✅ Scraped {len(docs)} documents")
        
        return all_docs
    
    def build_vectorstore(self, documents: List[Document], append: bool = False) -> Optional[FAISS]:
        """
        Build or update FAISS vectorstore.
        
        Args:
            documents: Documents to index
            append: If True, append to existing vectorstore
        
        Returns:
            FAISS vectorstore or None if no documents were provided
        """
        if not documents:
            print("⚠️ No documents to index")
            return None
        
        # Filter out low-quality documents
        original_count = len(documents)
        documents = [doc for doc in documents if len(doc.page_content) > 100]
        if len(documents) < original_count:
            print(f"📋 Filtered out {original_count - len(documents)} low-quality documents")
        
        # Deduplicate documents
        documents = self.deduplicate_documents(documents)
        
        print(f"📋 Final document count: {len(documents)} documents")
        
        print(f"\n🔨 Building vectorstore with {len(documents)} documents...")
        
        try:
            if append and os.path.exists(self.vector_db_path):
                # Load existing vectorstore and add new documents
                print("📚 Loading existing vectorstore...")
                vectorstore = FAISS.load_local(
                    self.vector_db_path,
                    embedding_model,
                    allow_dangerous_deserialization=True
                )
                
                print(f"➕ Adding {len(documents)} new documents...")
                vectorstore.add_documents(documents)
            else:
                # Create new vectorstore
                print("🆕 Creating new vectorstore...")
                vectorstore = FAISS.from_documents(documents, embedding_model)
            
            # Save vectorstore
            vectorstore.save_local(self.vector_db_path)
            
            # Update metadata
            self.metadata['total_documents'] = len(documents)
            self.metadata['last_update'] = datetime.now().isoformat()
            self._save_metadata()
            
            print(f"✅ Vectorstore saved to {self.vector_db_path}")
            return vectorstore
            
        except Exception as e:
            print(f"❌ Error building vectorstore: {e}")
            raise
    
    def full_rebuild(self, include_web: bool = False, web_queries: Optional[List[str]] = None) -> Optional[FAISS]:
        """
        Rebuild entire knowledge base from scratch.
        
        Args:
            include_web: Whether to include web-scraped content
            web_queries: Queries for web scraping
        
        Returns:
            Updated FAISS vectorstore or None if indexing failed
        """
        print("\n" + "="*60)
        print("🔄 FULL KNOWLEDGE BASE REBUILD")
        print("="*60 + "\n")
        
        all_docs = []
        
        # Load static documents
        print("📁 Loading static documents...")
        static_docs = self.load_documents()
        all_docs.extend(static_docs)
        print(f"✅ Loaded {len(static_docs)} static document chunks")
        
        # Scrape web if requested
        if include_web and web_queries:
            print("\n🌐 Scraping web sources...")
            web_docs = self.scrape_web_sources(web_queries)
            all_docs.extend(web_docs)
            print(f"✅ Scraped {len(web_docs)} web document chunks")
        
        # Build vectorstore
        print(f"\n📊 Total documents to index: {len(all_docs)}")
        vectorstore = self.build_vectorstore(all_docs, append=False)
        
        print("\n✅ FULL REBUILD COMPLETE!")
        return vectorstore
    
    def incremental_update(self, include_web: bool = False, web_queries: Optional[List[str]] = None) -> Optional[FAISS]:
        """
        Incrementally update knowledge base with new/modified documents.
        
        Args:
            include_web: Whether to include web-scraped content
            web_queries: Queries for web scraping
        
        Returns:
            Updated FAISS vectorstore or None if no updates needed
        """
        print("\n" + "="*60)
        print("🔄 INCREMENTAL KNOWLEDGE BASE UPDATE")
        print("="*60 + "\n")
        
        new_docs = []
        
        # Check for new/modified files
        print("🔍 Scanning for new or modified documents...")
        new_files = self.scan_new_documents()
        
        if new_files:
            print(f"📁 Found {len(new_files)} new/modified files")
            static_docs = self.load_documents(new_files)
            new_docs.extend(static_docs)
        else:
            print("✅ No new static documents found")
        
        # Scrape web if requested
        if include_web and web_queries:
            print("\n🌐 Scraping web sources...")
            web_docs = self.scrape_web_sources(web_queries)
            new_docs.extend(web_docs)
        
        if not new_docs:
            print("\n✅ Knowledge base is up to date!")
            return None
        
        # Append to existing vectorstore
        print(f"\n📊 Total new documents: {len(new_docs)}")
        vectorstore = self.build_vectorstore(new_docs, append=True)
        
        print("\n✅ INCREMENTAL UPDATE COMPLETE!")
        return vectorstore
    
    def needs_update(self) -> bool:
        """
        Check if knowledge base needs updating based on time.
        
        Returns:
            True if update is needed
        """
        last_update = self.metadata.get('last_update')
        
        if not last_update:
            return True
        
        last_update_time = datetime.fromisoformat(last_update)
        time_since_update = datetime.now() - last_update_time
        
        return time_since_update > timedelta(hours=self.auto_update_hours)
    
    def auto_update(self, web_queries: Optional[List[str]] = None) -> Optional[FAISS]:
        """
        Automatically update knowledge base if needed.
        
        Args:
            web_queries: Queries for web scraping
        
        Returns:
            Updated vectorstore or None if no update needed
        """
        if not self.needs_update():
            print("⏭️ Auto-update not needed yet")
            return None
        
        print("⏰ Auto-update triggered!")
        return self.incremental_update(
            include_web=bool(web_queries),
            web_queries=web_queries
        )
    
    def get_stats(self) -> Dict:
        """Get knowledge base statistics."""
        stats = {
            'total_files': len(self.metadata['indexed_files']),
            'total_documents': self.metadata.get('total_documents', 0),
            'last_update': self.metadata.get('last_update', 'Never'),
            'web_sources': len(self.metadata.get('web_sources', [])),
            'vectorstore_exists': os.path.exists(self.vector_db_path),
            'kb_folder': self.kb_folder,
            'needs_update': self.needs_update()
        }
        
        # File type breakdown
        file_types = {}
        for filepath in self.metadata['indexed_files'].keys():
            ext = os.path.splitext(filepath)[1]
            file_types[ext] = file_types.get(ext, 0) + 1
        stats['file_types'] = file_types
        
        # Calculate quality statistics
        try:
            if os.path.exists(self.vector_db_path):
                # Load vectorstore to access documents
                vectorstore = FAISS.load_local(
                    self.vector_db_path,
                    embedding_model,
                    allow_dangerous_deserialization=True
                )
                
                # Get documents with quality scores
                index_map = getattr(vectorstore, "index_to_docstore_id", None)
                docstore = getattr(vectorstore, "docstore", None)
                
                quality_scores = []
                trusted_count = 0
                
                if index_map and docstore:
                    doc_ids = list(index_map.values()) if isinstance(index_map, dict) else list(index_map)
                    
                    for doc_id in doc_ids[:100]:  # Sample first 100 for speed
                        try:
                            doc = None
                            if hasattr(docstore, "search"):
                                doc = docstore.search(doc_id)
                            elif hasattr(docstore, "_dict"):
                                doc = docstore._dict.get(doc_id)
                            
                            if doc and hasattr(doc, 'metadata'):
                                quality_score = doc.metadata.get('quality_score', 0.5)
                                quality_scores.append(quality_score)
                                if doc.metadata.get('trusted_source', False):
                                    trusted_count += 1
                        except:
                            continue
                
                if quality_scores:
                    stats['avg_quality_score'] = sum(quality_scores) / len(quality_scores)
                    stats['high_quality_docs'] = sum(1 for s in quality_scores if s >= 0.8)
                    stats['trusted_sources_count'] = trusted_count
        except Exception as e:
            print(f"⚠️ Could not calculate quality stats: {e}")
        
        return stats
    
    def clear_web_sources(self):
        """Clear web-scraped sources from metadata."""
        self.metadata['web_sources'] = []
        self._save_metadata()
        print("✅ Cleared web sources from metadata")
    
    def remove_file(self, rel_path: str):
        """
        Remove a file from metadata (requires full rebuild to update vectorstore).
        
        Args:
            rel_path: Relative path of file to remove
        """
        if rel_path in self.metadata['indexed_files']:
            del self.metadata['indexed_files'][rel_path]
            self._save_metadata()
            print(f"✅ Removed {rel_path} from metadata")
            print("⚠️ Run full_rebuild() to update vectorstore")
        else:
            print(f"⚠️ File {rel_path} not found in metadata")


# === Test Function ===
def test_dynamic_kb_manager():
    """Test the dynamic KB manager."""
    print("\n" + "="*60)
    print("Testing Dynamic KB Manager")
    print("="*60 + "\n")
    
    manager = DynamicKBManager(auto_update_hours=1)
    
    # Show stats
    print("\n--- Current Stats ---")
    stats = manager.get_stats()
    for key, value in stats.items():
        print(f"{key}: {value}")
    
    # Test incremental update
    print("\n--- Testing Incremental Update ---")
    manager.incremental_update(
        include_web=True,
        web_queries=["insurance technology 2025", "cyber insurance trends"]
    )
    
    # Show updated stats
    print("\n--- Updated Stats ---")
    stats = manager.get_stats()
    for key, value in stats.items():
        print(f"{key}: {value}")


if __name__ == "__main__":
    test_dynamic_kb_manager()
