"""
Retriever package for document retrieval and web scraping.
"""
from .hybrid_retriever import HybridRetriever
from .static_loader import StaticDocumentLoader
from .dynamic_scraper import DynamicWebScraper, RealtimeRetriever

__all__ = [
    'HybridRetriever',
    'StaticDocumentLoader',
    'DynamicWebScraper',
    'RealtimeRetriever'
]
