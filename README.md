# Insurance Policy Debate System 🏢💬

An advanced **multi-agent debate system** for analyzing insurance policies and industry trends using **Retrieval-Augmented Generation (RAG)** and diverse AI perspectives.

## 🎯 Overview

This system enables intelligent debates on insurance topics by:
- **6 specialized AI agents** with different roles (Analyst, Advocate, Skeptic, Regulator, Innovator, Consumer)
- **Hybrid retrieval** combining semantic search (FAISS) and keyword search (BM25)
- **Dynamic web scraping** for real-time insurance industry data
- **Structured debate orchestration** with multiple rounds and Q&A
- **Evidence-based argumentation** using RAG from knowledge base

## 🏗️ System Architecture

```
agentic_project/
├── agents/                    # Debate agent implementations
│   ├── debate_agents.py      # Agent classes with different perspectives
│   └── orchestrator.py       # Debate coordination and management
├── retriever/                 # Document retrieval systems
│   ├── hybrid_retriever.py   # FAISS + BM25 hybrid search
│   ├── static_loader.py      # PDF/text document loader
│   └── dynamic_scraper.py    # Real-time web scraping
├── utils/                     # Core utilities
│   ├── embeddings.py         # Text embedding with HuggingFace
│   ├── vectorstore.py        # FAISS vector database management
│   └── scraper.py            # Web scraping tools
├── kb_docs/                   # Knowledge base documents
├── vectorstore/              # FAISS indices
├── main.py                   # Main entry point
├── build_kb.py               # Knowledge base builder
└── test_retriever.py         # Retrieval testing
```

## 🤖 Debate Agents

### 1. **Dr. Sarah Chen** - Analyst
- Data-driven, objective analysis
- Market research and quantitative insights
- Statistical trends and forecasting

### 2. **Marcus Williams** - Advocate
- Optimistic, forward-thinking
- Highlights opportunities and benefits
- Promotes innovation and growth

### 3. **Dr. Elena Rodriguez** - Skeptic
- Critical evaluation of claims
- Risk identification and mitigation
- Questions assumptions

### 4. **Commissioner James Patterson** - Regulator
- Compliance and legal focus
- Consumer protection advocacy
- Market stability oversight

### 5. **Alex Kim** - Innovator
- Technology and digital transformation
- Emerging trends (AI, blockchain, IoT)
- Disruptive business models

### 6. **Jennifer Martinez** - Consumer Advocate
- Policyholder interests
- Affordability and accessibility
- Practical customer experience

## 🚀 Getting Started

### Installation

```bash
# Clone the repository
cd agentic_project

# Install dependencies
pip install -r requirements.txt
```

### Build Knowledge Base

```bash
# Scrape insurance documents
python utils/scraper.py

# Build vector database
python build_kb.py
```

### Run Debates

#### Sample Demo
```bash
python main.py --mode sample
```

#### Interactive Mode
```bash
python main.py --mode interactive
```

#### Custom Debate
```bash
python main.py --mode custom --topic "Cyber insurance market trends" --rounds 3
```

#### Without RAG (Faster)
```bash
python main.py --mode sample --no-retriever
```

## 📚 Key Features

### Hybrid Retrieval System

Combines two powerful search methods:

**Semantic Search (FAISS)**
- Vector embeddings using `sentence-transformers/all-MiniLM-L6-v2`
- Understands meaning and context
- Finds conceptually similar content

**Keyword Search (BM25)**
- Traditional information retrieval
- Exact term matching
- Effective for specific queries

**Hybrid Approach**
```python
from retriever.hybrid_retriever import HybridRetriever

retriever = HybridRetriever(
    vector_db_path="./vectorstore/faiss_index",
    alpha=0.5  # 50% semantic, 50% keyword
)

results = retriever.search("AI in insurance", k=5, method="hybrid")
```

### Dynamic Web Scraping

Real-time content retrieval during debates:

```python
from retriever.dynamic_scraper import DynamicWebScraper

scraper = DynamicWebScraper(max_results=5)

# Search latest news
docs = scraper.search_insurance_news("cyber insurance")

# Company-specific search
docs = scraper.search_company_reports("SwissRe", year=2025)

# Targeted site search
docs = scraper.targeted_search(
    "climate risk", 
    sites=['swissre.com', 'munichre.com']
)
```

### Structured Debates

Multi-round debates with evidence retrieval:

```python
from agents.orchestrator import quick_debate

debate_result = quick_debate(
    topic="Parametric insurance vs traditional policies",
    rounds=2
)
```

### Flexible Document Loading

```python
from retriever.static_loader import StaticDocumentLoader

loader = StaticDocumentLoader()

# Load all documents
docs = loader.load_and_split()

# Load by topic
docs = loader.load_by_topic("insurance trends 2025")

# Get statistics
stats = loader.get_document_stats()
```

## 🔧 Configuration

### Embedding Model
Change in `utils/embeddings.py`:
```python
embedding_model = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
    # Or: "sentence-transformers/all-mpnet-base-v2" for better quality
)
```

### Chunk Size
Adjust in `build_kb.py`:
```python
CHUNK_SIZE = 1000      # Characters per chunk
CHUNK_OVERLAP = 100    # Overlap between chunks
```

### Retrieval Parameters
Modify in retriever initialization:
```python
retriever = HybridRetriever(
    vector_db_path="./vectorstore/faiss_index",
    alpha=0.5  # Adjust semantic vs keyword weight (0-1)
)
```

## 📊 Example Usage

### Complete Workflow

```python
# 1. Set up system
from main import setup_system
agents, orchestrator, retriever = setup_system(use_retriever=True)

# 2. Conduct debate
debate = orchestrator.conduct_debate(
    topic="Impact of AI on insurance underwriting",
    rounds=2,
    retrieve_context=True,
    context_k=3
)

# 3. Q&A round
qa = orchestrator.facilitate_qa_round([
    "What are the privacy concerns?",
    "How can bias be mitigated?"
])

# 4. Generate report
report = orchestrator.generate_consensus_report()
print(report)
```

### Test Individual Components

```bash
# Test hybrid retriever
python retriever/hybrid_retriever.py

# Test static loader
python retriever/static_loader.py

# Test dynamic scraper
python retriever/dynamic_scraper.py

# Test debate agents
python agents/debate_agents.py

# Test orchestrator
python agents/orchestrator.py
```

## 📝 Sample Output

```
================================================================================
🎯 DEBATE TOPIC: AI-powered underwriting: benefits vs risks
================================================================================

📚 Retrieving relevant context...
✅ Retrieved 6 relevant documents

================================================================================
🔄 ROUND 1 of 2
================================================================================

🗣️  Dr. Sarah Chen (ANALYST)
--------------------------------------------------------------------------------
From an analytical perspective on 'AI-powered underwriting: benefits vs risks':

Based on the available data and market trends, I observe several key patterns...
[Evidence-based analysis with citations]

🗣️  Marcus Williams (ADVOCATE)
--------------------------------------------------------------------------------
I strongly support the developments in 'AI-powered underwriting: benefits vs risks'.

This represents a significant opportunity for the insurance industry...
[Optimistic perspective with benefits]

...

================================================================================
📊 DEBATE SUMMARY
================================================================================

👤 Dr. Sarah Chen (analyst)
   Arguments: 2
   Evidence used: 3
   Sources: MunichRe-Presentation-Monte-Carlo-2025.txt, ...

🔑 Key Themes Discussed:
   • Technology
   • Risk Management
   • Consumer Impact
   • Regulation
   • Market Trends
```

## 🧪 Testing

```bash
# Test retrieval
python test_retriever.py

# Run all component tests
python -m pytest tests/  # (if tests directory exists)
```

## 📦 Dependencies

Core packages:
- `langchain` - LLM framework
- `langchain-community` - Community integrations
- `langchain-huggingface` - HuggingFace embeddings
- `faiss-cpu` - Vector similarity search
- `rank-bm25` - Keyword search
- `sentence-transformers` - Text embeddings
- `beautifulsoup4` - Web scraping
- `duckduckgo-search` - Search engine
- `PyPDF2` - PDF processing
- `requests` - HTTP requests

## 🔮 Future Enhancements

- [ ] Integration with OpenAI GPT-4 for better responses
- [ ] LlamaIndex integration for advanced RAG
- [ ] AutoGen for autonomous agent orchestration
- [ ] Persistent debate history database
- [ ] Web UI for interactive debates
- [ ] Real-time streaming responses
- [ ] Multi-language support
- [ ] Export debates to PDF/JSON
- [ ] Agent learning from debate history
- [ ] Custom agent creation interface

## 🤝 Contributing

Contributions welcome! Areas for improvement:
- Additional agent roles
- Enhanced retrieval strategies
- Better context formatting
- LLM integration
- UI development

## 📄 License

MIT License - Feel free to use and modify for your projects.

## 🙏 Acknowledgments

- LangChain for RAG framework
- HuggingFace for embedding models
- FAISS for efficient vector search
- Insurance industry sources (SwissRe, MunichRe, etc.)

---

**Built with ❤️ for intelligent insurance policy analysis**
#   D e b a t e - S i m u l a t o r - o n - I n s u r a n c e - P o l i c y - I s s u e s  
 