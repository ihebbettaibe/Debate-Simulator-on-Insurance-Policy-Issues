"""
Dynamic web scraper for real-time content retrieval during debates.
Fetches fresh insurance industry news and reports on-demand.
"""
import os
import sys
from typing import List, Dict, Optional
from datetime import datetime
import requests
from bs4 import BeautifulSoup
from langchain.docstore.document import Document

# Add project root to path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

try:
    from ddgs import DDGS
except ImportError:
    DDGS = None
    print("⚠️ duckduckgo-search not installed. Install with: pip install duckduckgo-search")


class DynamicWebScraper:
    """Scrapes web content dynamically based on queries."""
    
    def __init__(self, max_results: int = 5, timeout: int = 10):
        """
        Initialize dynamic scraper.
        
        Args:
            max_results: Maximum number of search results
            timeout: Request timeout in seconds
        """
        self.max_results = max_results
        self.timeout = timeout
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
        })
    
    def search_web(self, query: str, max_results: Optional[int] = None) -> List[Dict]:
        """
        Search the web using DuckDuckGo.
        
        Args:
            query: Search query
            max_results: Override default max_results
        
        Returns:
            List of search results with title, url, snippet
        """
        if DDGS is None:
            print("❌ DuckDuckGo search not available")
            return []
        
        results = []
        max_res = max_results or self.max_results
        
        try:
            with DDGS() as ddgs:
                for result in ddgs.text(query, max_results=max_res):
                    results.append({
                        'title': result.get('title', ''),
                        'url': result.get('href') or result.get('url', ''),
                        'snippet': result.get('body', ''),
                        'timestamp': datetime.now().isoformat()
                    })
            
            print(f"🔎 Found {len(results)} results for: {query}")
        except Exception as e:
            print(f"❌ Search error: {e}")
        
        return results
    
    def fetch_page_content(self, url: str) -> Optional[str]:
        """
        Fetch and extract main text content from a webpage.
        
        Args:
            url: URL to scrape
        
        Returns:
            Extracted text content or None
        """
        try:
            response = self.session.get(url, timeout=self.timeout)
            response.raise_for_status()
            
            soup = BeautifulSoup(response.content, 'html.parser')
            
            # Remove script and style elements
            for script in soup(["script", "style", "nav", "footer", "header"]):
                script.decompose()
            
            # Get text
            text = soup.get_text(separator='\n', strip=True)
            
            # Clean up whitespace
            lines = (line.strip() for line in text.splitlines())
            chunks = (phrase.strip() for line in lines for phrase in line.split("  "))
            text = '\n'.join(chunk for chunk in chunks if chunk)
            
            print(f"✅ Fetched content from: {url[:60]}...")
            return text
            
        except Exception as e:
            print(f"❌ Error fetching {url}: {e}")
            return None
    
    def scrape_to_documents(self, query: str, fetch_content: bool = True) -> List[Document]:
        """
        Search and convert results to LangChain Documents.
        
        Args:
            query: Search query
            fetch_content: Whether to fetch full page content
        
        Returns:
            List of Documents
        """
        search_results = self.search_web(query)
        documents = []
        
        for result in search_results:
            content = result['snippet']  # Start with snippet
            
            # Optionally fetch full content
            if fetch_content and result['url']:
                full_content = self.fetch_page_content(result['url'])
                if full_content:
                    content = full_content
            
            # Create document
            doc = Document(
                page_content=content,
                metadata={
                    'source': result['url'],
                    'title': result['title'],
                    'query': query,
                    'timestamp': result['timestamp'],
                    'type': 'web_scraped'
                }
            )
            documents.append(doc)
        
        print(f"📄 Created {len(documents)} documents from web search")
        return documents
    
    def search_insurance_news(self, topic: str = "insurance trends 2025") -> List[Document]:
        """
        Search for recent insurance industry news.
        
        Args:
            topic: Insurance topic to search
        
        Returns:
            List of Documents with news content
        """
        query = f"{topic} insurance industry news"
        return self.scrape_to_documents(query, fetch_content=False)
    
    def search_company_reports(self, company: str, year: int = 2025) -> List[Document]:
        """
        Search for company-specific insurance reports.
        
        Args:
            company: Insurance company name
            year: Report year
        
        Returns:
            List of Documents with report content
        """
        query = f"{company} insurance report {year} site:{company.lower()}.com"
        return self.scrape_to_documents(query, fetch_content=True)
    
    def search_regulatory_updates(self, region: str = "global") -> List[Document]:
        """
        Search for insurance regulatory updates.
        
        Args:
            region: Geographic region (global, US, EU, etc.)
        
        Returns:
            List of Documents with regulatory content
        """
        query = f"{region} insurance regulation updates 2025"
        return self.scrape_to_documents(query, fetch_content=False)
    
    def targeted_search(self, query: str, sites: Optional[List[str]] = None) -> List[Document]:
        """
        Search specific sites for information.
        
        Args:
            query: Search query
            sites: List of domains to search (e.g., ['swissre.com', 'munichre.com'])
        
        Returns:
            List of Documents
        """
        all_docs = []
        
        if sites:
            for site in sites:
                site_query = f"{query} site:{site}"
                docs = self.scrape_to_documents(site_query, fetch_content=True)
                all_docs.extend(docs)
        else:
            all_docs = self.scrape_to_documents(query, fetch_content=True)
        
        return all_docs


class RealtimeRetriever:
    """
    Combines static knowledge base with dynamic web scraping.
    """
    
    def __init__(self, static_retriever=None):
        """
        Initialize realtime retriever.
        
        Args:
            static_retriever: Optional hybrid retriever for static KB
        """
        self.static_retriever = static_retriever
        self.dynamic_scraper = DynamicWebScraper()
    
    def retrieve(self, query: str, use_static: bool = True, use_dynamic: bool = True, k: int = 5) -> List[Document]:
        """
        Retrieve documents from both static KB and live web.
        
        Args:
            query: Search query
            use_static: Whether to use static knowledge base
            use_dynamic: Whether to scrape web dynamically
            k: Number of results per source
        
        Returns:
            Combined list of Documents
        """
        all_docs = []
        
        # Get static results
        if use_static and self.static_retriever:
            print("\n🗃️ Retrieving from static knowledge base...")
            static_docs = self.static_retriever.search(query, k=k)
            all_docs.extend(static_docs)
            print(f"✅ Retrieved {len(static_docs)} static documents")
        
        # Get dynamic results
        if use_dynamic:
            print("\n🌐 Scraping web for latest information...")
            dynamic_docs = self.dynamic_scraper.scrape_to_documents(query, fetch_content=False)
            all_docs.extend(dynamic_docs[:k])
            print(f"✅ Retrieved {len(dynamic_docs[:k])} dynamic documents")
        
        return all_docs


# === Test Function ===
def test_dynamic_scraper():
    """Test the dynamic scraper."""
    print("\n" + "="*60)
    print("Testing Dynamic Web Scraper")
    print("="*60 + "\n")
    
    scraper = DynamicWebScraper(max_results=3)
    
    # Test 1: Search insurance news
    print("\n--- Test 1: Insurance News ---")
    docs = scraper.search_insurance_news("cyber insurance")
    
    if docs:
        print(f"\nSample document:")
        print(f"  Title: {docs[0].metadata['title']}")
        print(f"  URL: {docs[0].metadata['source']}")
        print(f"  Content: {docs[0].page_content[:200]}...")
    
    # Test 2: Targeted search
    print("\n--- Test 2: Targeted Search ---")
    docs = scraper.targeted_search(
        "climate risk insurance", 
        sites=['swissre.com', 'munichre.com']
    )
    
    print(f"\nFound {len(docs)} documents from targeted search")


if __name__ == "__main__":
    test_dynamic_scraper()
