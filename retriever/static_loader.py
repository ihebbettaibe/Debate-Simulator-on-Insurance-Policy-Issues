"""
Static document loader for loading and processing documents from kb_docs.
Supports PDFs and text files.
"""
import os
from typing import List, Dict
from langchain.document_loaders import PyPDFLoader, TextLoader, DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.docstore.document import Document
from typing import Optional


class StaticDocumentLoader:
    """Loads and processes static documents from the knowledge base."""
    
    def __init__(self, kb_folder: str = "./kb_docs", chunk_size: int = 1000, chunk_overlap: int = 100):
        """
        Initialize static document loader.
        
        Args:
            kb_folder: Path to knowledge base folder
            chunk_size: Size of text chunks
            chunk_overlap: Overlap between chunks
        """
        self.kb_folder = kb_folder
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            separators=["\n\n", "\n", " ", ""]
        )
    
    def load_pdf(self, pdf_path: str) -> List[Document]:
        """Load a single PDF file."""
        try:
            loader = PyPDFLoader(pdf_path)
            docs = loader.load()
            
            # Add metadata
            for doc in docs:
                doc.metadata["source"] = os.path.basename(pdf_path)
                doc.metadata["file_path"] = pdf_path
                doc.metadata["type"] = "pdf"
            
            print(f"✅ Loaded PDF: {os.path.basename(pdf_path)} ({len(docs)} pages)")
            return docs
        except Exception as e:
            print(f"❌ Error loading PDF {pdf_path}: {e}")
            return []
    
    def load_text(self, txt_path: str) -> List[Document]:
        """Load a single text file."""
        try:
            loader = TextLoader(txt_path, encoding="utf-8")
            docs = loader.load()
            
            # Add metadata
            for doc in docs:
                doc.metadata["source"] = os.path.basename(txt_path)
                doc.metadata["file_path"] = txt_path
                doc.metadata["type"] = "text"
            
            print(f"✅ Loaded text file: {os.path.basename(txt_path)}")
            return docs
        except Exception as e:
            print(f"❌ Error loading text file {txt_path}: {e}")
            return []
    
    def load_directory(self, directory: Optional[str] = None) -> List[Document]:
        """
        Load all documents from a directory recursively.
        
        Args:
            directory: Directory path (defaults to kb_folder)
        
        Returns:
            List of loaded documents
        """
        target_dir = directory or self.kb_folder
        all_docs = []
        
        print(f"\n📂 Loading documents from: {target_dir}")
        
        # Walk through directory
        for root, dirs, files in os.walk(target_dir):
            for file in files:
                file_path = os.path.join(root, file)
                
                if file.lower().endswith('.pdf'):
                    docs = self.load_pdf(file_path)
                    all_docs.extend(docs)
                elif file.lower().endswith('.txt'):
                    docs = self.load_text(file_path)
                    all_docs.extend(docs)
        
        print(f"\n📚 Total documents loaded: {len(all_docs)}")
        return all_docs
    
    def load_and_split(self, directory: Optional[str] = None) -> List[Document]:
        """
        Load documents and split them into chunks.
        
        Args:
            directory: Directory path (defaults to kb_folder)
        
        Returns:
            List of chunked documents
        """
        docs = self.load_directory(directory)
        
        if not docs:
            print("⚠️ No documents loaded")
            return []
        
        # Split documents
        print(f"\n✂️ Splitting documents (chunk_size={self.chunk_size}, overlap={self.chunk_overlap})...")
        split_docs = self.text_splitter.split_documents(docs)
        
        print(f"✅ Created {len(split_docs)} document chunks")
        return split_docs
    
    def load_by_topic(self, topic: str) -> List[Document]:
        """
        Load documents from a specific topic subfolder.
        
        Args:
            topic: Topic name (subfolder in kb_docs)
        
        Returns:
            List of documents from that topic
        """
        topic_dir = os.path.join(self.kb_folder, topic)
        
        if not os.path.exists(topic_dir):
            print(f"⚠️ Topic directory not found: {topic_dir}")
            return []
        
        return self.load_and_split(topic_dir)
    
    def get_available_topics(self) -> List[str]:
        """Get list of available topic folders."""
        if not os.path.exists(self.kb_folder):
            return []
        
        topics = [
            d for d in os.listdir(self.kb_folder)
            if os.path.isdir(os.path.join(self.kb_folder, d))
        ]
        return topics
    
    def get_document_stats(self) -> Dict:
        """Get statistics about documents in knowledge base."""
        stats = {
            "total_files": 0,
            "pdf_files": 0,
            "txt_files": 0,
            "topics": {},
            "total_size_mb": 0
        }
        
        if not os.path.exists(self.kb_folder):
            return stats
        
        for root, dirs, files in os.walk(self.kb_folder):
            for file in files:
                file_path = os.path.join(root, file)
                file_size = os.path.getsize(file_path) / (1024 * 1024)  # MB
                stats["total_size_mb"] += file_size
                
                # Get topic (parent folder)
                topic = os.path.basename(root)
                if topic not in stats["topics"]:
                    stats["topics"][topic] = {"files": 0, "size_mb": 0}
                
                stats["topics"][topic]["files"] += 1
                stats["topics"][topic]["size_mb"] += file_size
                
                if file.lower().endswith('.pdf'):
                    stats["pdf_files"] += 1
                    stats["total_files"] += 1
                elif file.lower().endswith('.txt'):
                    stats["txt_files"] += 1
                    stats["total_files"] += 1
        
        stats["total_size_mb"] = round(stats["total_size_mb"], 2)
        for topic in stats["topics"]:
            stats["topics"][topic]["size_mb"] = round(stats["topics"][topic]["size_mb"], 2)
        
        return stats


# === Test Function ===
def test_static_loader():
    """Test the static document loader."""
    loader = StaticDocumentLoader()
    
    print("\n" + "="*60)
    print("Testing Static Document Loader")
    print("="*60 + "\n")
    
    # Get stats
    stats = loader.get_document_stats()
    print("\n📊 Knowledge Base Statistics:")
    print(f"   Total files: {stats['total_files']}")
    print(f"   PDF files: {stats['pdf_files']}")
    print(f"   Text files: {stats['txt_files']}")
    print(f"   Total size: {stats['total_size_mb']} MB")
    print(f"\n   Topics:")
    for topic, info in stats['topics'].items():
        print(f"      - {topic}: {info['files']} files ({info['size_mb']} MB)")
    
    # Get available topics
    topics = loader.get_available_topics()
    print(f"\n📁 Available topics: {topics}")
    
    # Load and split documents
    docs = loader.load_and_split()
    
    if docs:
        print(f"\n📄 Sample document chunk:")
        print(f"   Source: {docs[0].metadata.get('source')}")
        print(f"   Type: {docs[0].metadata.get('type')}")
        print(f"   Content preview: {docs[0].page_content[:200]}...")


if __name__ == "__main__":
    test_static_loader()
