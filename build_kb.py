"""
Enhanced Knowledge Base Builder with Dynamic Updates
Supports static documents, web scraping, and incremental updates.
"""
import os
import sys
import argparse

# Add project root to path
project_root = os.path.abspath(os.path.dirname(__file__))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from retriever.dynamic_kb_manager import DynamicKBManager

# Configuration
KB_FOLDER = "./kb_docs"
VECTOR_DB_PATH = "./vectorstore/faiss_index"
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 100

# Default web queries for insurance knowledge
DEFAULT_WEB_QUERIES = [
    "insurance industry trends 2025",
    "cyber insurance market analysis",
    "climate risk insurance developments",
    "InsurTech innovations 2025",
    "reinsurance market outlook"
]

# Trusted insurance industry sites
TRUSTED_SITES = [
    'swissre.com',
    'munichre.com',
    'lloyds.com',
    'iii.org',  # Insurance Information Institute
    'naic.org'  # National Association of Insurance Commissioners
]


def main():
    """Main function to build/update knowledge base."""
    parser = argparse.ArgumentParser(
        description="Build or update the insurance knowledge base"
    )
    parser.add_argument(
        '--mode',
        choices=['full', 'incremental', 'stats'],
        default='full',
        help='Operation mode: full rebuild, incremental update, or show stats'
    )
    parser.add_argument(
        '--web',
        action='store_true',
        help='Include web scraping for latest content'
    )
    parser.add_argument(
        '--queries',
        nargs='+',
        help='Custom web search queries (overrides defaults)'
    )
    parser.add_argument(
        '--sites',
        nargs='+',
        help='Specific sites to scrape (e.g., swissre.com munichre.com)'
    )
    parser.add_argument(
        '--auto-update-hours',
        type=int,
        default=24,
        help='Hours between automatic updates (default: 24)'
    )
    
    args = parser.parse_args()
    
    # Initialize manager
    print("\n" + "="*70)
    print("🏗️  INSURANCE KNOWLEDGE BASE BUILDER")
    print("="*70 + "\n")
    
    manager = DynamicKBManager(
        kb_folder=KB_FOLDER,
        vector_db_path=VECTOR_DB_PATH,
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        auto_update_hours=args.auto_update_hours
    )
    
    # Determine web queries
    web_queries = None
    if args.web:
        if args.queries:
            web_queries = args.queries
        else:
            web_queries = DEFAULT_WEB_QUERIES
        
        # Apply site restrictions if specified
        if args.sites:
            print(f"🎯 Targeting specific sites: {', '.join(args.sites)}")
    
    # Execute based on mode
    if args.mode == 'stats':
        print("\n📊 KNOWLEDGE BASE STATISTICS")
        print("-" * 70)
        stats = manager.get_stats()
        for key, value in stats.items():
            if isinstance(value, dict):
                print(f"\n{key}:")
                for k, v in value.items():
                    print(f"  {k}: {v}")
            else:
                print(f"{key}: {value}")
        print("-" * 70)
    
    elif args.mode == 'full':
        print("\n🔄 Starting FULL REBUILD...")
        vectorstore = manager.full_rebuild(
            include_web=args.web,
            web_queries=web_queries
        )
        
        if vectorstore:
            print("\n" + "="*70)
            print("✅ KNOWLEDGE BASE SUCCESSFULLY BUILT!")
            print("="*70)
            print(f"\n📍 Vector DB Location: {VECTOR_DB_PATH}")
            print(f"📁 Source Documents: {KB_FOLDER}")
            
            stats = manager.get_stats()
            print(f"📊 Total Files: {stats['total_files']}")
            print(f"📄 Total Document Chunks: {stats['total_documents']}")
            if args.web:
                print(f"🌐 Web Sources: {stats['web_sources']}")
            print("\n🚀 Ready to use with debate system!")
            print("="*70 + "\n")
    
    elif args.mode == 'incremental':
        print("\n🔄 Starting INCREMENTAL UPDATE...")
        vectorstore = manager.incremental_update(
            include_web=args.web,
            web_queries=web_queries
        )
        
        if vectorstore:
            print("\n" + "="*70)
            print("✅ KNOWLEDGE BASE SUCCESSFULLY UPDATED!")
            print("="*70)
            stats = manager.get_stats()
            print(f"📊 Total Files: {stats['total_files']}")
            print(f"📄 Total Document Chunks: {stats['total_documents']}")
            print(f"🕐 Last Update: {stats['last_update']}")
            print("="*70 + "\n")
        else:
            print("\n✅ Knowledge base is already up to date!")


if __name__ == "__main__":
    # If run without arguments, show help and do basic build
    import sys
    if len(sys.argv) == 1:
        print("\n💡 TIP: Run with --help to see all options")
        print("💡 Running basic build without web scraping...\n")
        
        manager = DynamicKBManager(
            kb_folder=KB_FOLDER,
            vector_db_path=VECTOR_DB_PATH,
            chunk_size=CHUNK_SIZE,
            chunk_overlap=CHUNK_OVERLAP
        )
        
        vectorstore = manager.full_rebuild(include_web=False)
        
        if vectorstore:
            print("\n✅ Basic knowledge base built successfully!")
            print(f"📍 Location: {VECTOR_DB_PATH}")
            print("\n💡 To include web scraping: python build_kb.py --mode full --web")
    else:
        main()
