"""
Configuration file for the Insurance Debate System.
Modify these settings to customize system behavior.
"""
import os

# === Paths ===
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
KB_DOCS_PATH = os.path.join(PROJECT_ROOT, "kb_docs")
VECTOR_DB_PATH = os.path.join(PROJECT_ROOT, "vectorstore", "faiss_index")
VECTOR_DB_LEGACY_PATH = os.path.join(PROJECT_ROOT, "vector_db")

# === Embedding Configuration ===
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
# Alternative models (uncomment to use):
# EMBEDDING_MODEL = "sentence-transformers/all-mpnet-base-v2"  # Better quality, slower
# EMBEDDING_MODEL = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"  # Multilingual

# === Document Processing ===
CHUNK_SIZE = 1000           # Characters per chunk
CHUNK_OVERLAP = 100         # Overlap between chunks
CHUNK_SEPARATORS = ["\n\n", "\n", " ", ""]

# === Retrieval Configuration ===
HYBRID_ALPHA = 0.5          # Weight for semantic search (0-1). BM25 weight = 1-alpha
DEFAULT_K = 5               # Default number of results to retrieve
SEMANTIC_K = 10             # Number of results from semantic search
BM25_K = 10                 # Number of results from BM25 search

# === Web Scraping ===
MAX_SEARCH_RESULTS = 10     # Maximum search results per query
SCRAPE_TIMEOUT = 10         # Request timeout in seconds
USER_AGENT = 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'

# === Debate Configuration ===
DEFAULT_ROUNDS = 2          # Default number of debate rounds
DEFAULT_CONTEXT_K = 3       # Documents to retrieve per agent per round
ENABLE_RAG = True           # Whether to use RAG by default

# === Agent Configuration ===
AGENT_PERSONALITIES = {
    "analyst": "Data-driven, objective, focuses on market research and quantitative analysis",
    "advocate": "Optimistic, forward-thinking, emphasizes opportunities and benefits",
    "skeptic": "Critical thinker, risk-aware, questions assumptions and conventional wisdom",
    "regulator": "Compliance-focused, protective of consumers, emphasis on stability",
    "innovator": "Tech-savvy, futuristic, passionate about disruption and transformation",
    "consumer": "Practical, customer-focused, advocates for policyholder interests"
}

# === LLM Configuration ===
LLM_PROVIDER = None         # None, "openai", "anthropic", "ollama"
LLM_MODEL = "gpt-4"         # Model name
LLM_TEMPERATURE = 0.7       # Response creativity (0-1)
LLM_MAX_TOKENS = 500        # Maximum response length

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY", "")

# === Ollama Model Assignments per Agent Role ===
# Optimized for different reasoning styles and resource usage
AGENT_MODELS = {
    "Analyst": "mistral:7b",      # Data-driven: clean reasoning + low latency
    "Advocate": "qwen2:1.5b",     # Optimistic: faster, persuasive text
    "Skeptic": "llama3:8b",       # Critical: deeper reasoning, counter-arguments
    "Regulator": "mistral:7b",    # Compliance: balanced tone & fact structure
    "Innovator": "qwen2:1.5b",    # Tech-focused: creative & fluent phrasing
    "Consumer": "mistral:7b",     # Customer view: natural-sounding, concise
    "Moderator": "llama3:8b",     # Synthesis: can combine multiple arguments
    "PRO": "llama3:8b",           # Pro debates: strong logic & structure
    "CON": "llama3:8b",           # Con debates: strong counter-arguments
    "JUDGE": "mistral:7b",        # Evaluation: balanced, fact-based assessment
}

# Fallback model if specific model unavailable
DEFAULT_OLLAMA_MODEL = "llama3.2"

# === Output Configuration ===
VERBOSE = True              # Print detailed logs
SAVE_DEBATES = True         # Save debate transcripts
DEBATES_OUTPUT_DIR = os.path.join(PROJECT_ROOT, "debates_output")

# === Performance ===
BATCH_SIZE = 32             # Batch size for embeddings
USE_GPU = False             # Use GPU for embeddings (if available)

# === Debate Topics (Predefined) ===
SAMPLE_TOPICS = [
    "AI-powered underwriting: benefits vs risks",
    "Climate change impact on insurance pricing",
    "Parametric insurance vs traditional policies",
    "Cyber insurance: emerging market challenges",
    "InsurTech disruption in traditional insurance",
    "Blockchain for insurance claims processing",
    "Telematics-based auto insurance pricing",
    "Peer-to-peer insurance models",
    "Pandemic risk and business interruption coverage",
    "ESG considerations in insurance investment"
]

# === Insurance Industry Sources ===
TRUSTED_SOURCES = [
    "swissre.com",
    "munichre.com",
    "allianz.com",
    "lloyds.com",
    "iii.org",
    "naic.org",
    "oecd.org"
]

# === Feature Flags ===
ENABLE_DYNAMIC_SCRAPING = True      # Allow real-time web scraping
ENABLE_STATIC_KB = True             # Use static knowledge base
ENABLE_HYBRID_RETRIEVAL = True      # Use hybrid (semantic + keyword) search
ENABLE_CONSENSUS_REPORT = True      # Generate consensus reports
ENABLE_QA_ROUNDS = True             # Allow Q&A sessions


def get_config():
    """Get all configuration as a dictionary."""
    config = {
        "paths": {
            "project_root": PROJECT_ROOT,
            "kb_docs": KB_DOCS_PATH,
            "vector_db": VECTOR_DB_PATH
        },
        "embedding": {
            "model": EMBEDDING_MODEL,
            "batch_size": BATCH_SIZE,
            "use_gpu": USE_GPU
        },
        "chunking": {
            "chunk_size": CHUNK_SIZE,
            "chunk_overlap": CHUNK_OVERLAP,
            "separators": CHUNK_SEPARATORS
        },
        "retrieval": {
            "hybrid_alpha": HYBRID_ALPHA,
            "default_k": DEFAULT_K,
            "semantic_k": SEMANTIC_K,
            "bm25_k": BM25_K
        },
        "scraping": {
            "max_results": MAX_SEARCH_RESULTS,
            "timeout": SCRAPE_TIMEOUT,
            "user_agent": USER_AGENT
        },
        "debate": {
            "default_rounds": DEFAULT_ROUNDS,
            "context_k": DEFAULT_CONTEXT_K,
            "enable_rag": ENABLE_RAG
        },
        "llm": {
            "provider": LLM_PROVIDER,
            "model": LLM_MODEL,
            "temperature": LLM_TEMPERATURE,
            "max_tokens": LLM_MAX_TOKENS
        },
        "features": {
            "dynamic_scraping": ENABLE_DYNAMIC_SCRAPING,
            "static_kb": ENABLE_STATIC_KB,
            "hybrid_retrieval": ENABLE_HYBRID_RETRIEVAL,
            "consensus_report": ENABLE_CONSENSUS_REPORT,
            "qa_rounds": ENABLE_QA_ROUNDS
        }
    }
    return config


def print_config():
    """Print current configuration."""
    config = get_config()
    print("\n" + "="*60)
    print("SYSTEM CONFIGURATION")
    print("="*60 + "\n")
    
    for section, settings in config.items():
        print(f"\n[{section.upper()}]")
        for key, value in settings.items():
            print(f"  {key}: {value}")
    
    print("\n" + "="*60 + "\n")


if __name__ == "__main__":
    print_config()
