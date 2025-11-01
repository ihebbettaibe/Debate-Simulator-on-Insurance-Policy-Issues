# Quick Start Guide 🚀

## 5-Minute Setup

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Build Knowledge Base (Optional but Recommended)
```bash
# Scrape insurance documents
python utils/scraper.py

# Build FAISS vector database
python build_kb.py
```

### 3. Run Your First Debate
```bash
python main.py --mode sample
```

## Quick Examples

### Example 1: Simple Debate (No RAG)
```python
from agents.orchestrator import quick_debate

# Fast debate without document retrieval
result = quick_debate(
    topic="Cyber insurance pricing strategies",
    retriever=None,
    rounds=1
)
```

### Example 2: RAG-Enhanced Debate
```python
from retriever.hybrid_retriever import HybridRetriever
from agents.orchestrator import quick_debate

# Load retriever
retriever = HybridRetriever("./vectorstore/faiss_index")

# Debate with evidence
result = quick_debate(
    topic="Climate risk in property insurance",
    retriever=retriever,
    rounds=2
)
```

### Example 3: Custom Agent Debate
```python
from agents import create_insurance_debate_agents, DebateOrchestrator
from retriever import HybridRetriever

# Set up
retriever = HybridRetriever("./vectorstore/faiss_index", alpha=0.6)
agents = create_insurance_debate_agents(retriever=retriever)
orchestrator = DebateOrchestrator(agents, retriever)

# Debate
debate = orchestrator.conduct_debate(
    topic="Parametric vs traditional insurance",
    rounds=2,
    retrieve_context=True
)

# Q&A
qa = orchestrator.facilitate_qa_round([
    "Which is more suitable for developing countries?",
    "What are the cost implications?"
])

# Report
print(orchestrator.generate_consensus_report())
```

### Example 4: Dynamic Web Search
```python
from retriever.dynamic_scraper import DynamicWebScraper

scraper = DynamicWebScraper()

# Latest news
news = scraper.search_insurance_news("InsurTech 2025")

# Company reports
reports = scraper.search_company_reports("Allianz", 2025)

# Targeted search
docs = scraper.targeted_search(
    "digital transformation",
    sites=['swissre.com', 'munichre.com']
)
```

### Example 5: Interactive Mode
```bash
python main.py --mode interactive
```
Then choose options:
1. Start new debate
2. Ask follow-up questions
3. View debate history
4. Generate consensus report
5. Exit

## Common Issues

### "Vector database not found"
**Solution:** Run `python build_kb.py` first

### "Import errors"
**Solution:** Install missing packages:
```bash
pip install rank-bm25 duckduckgo-search
```

### "No documents in kb_docs"
**Solution:** Run the scraper:
```bash
python utils/scraper.py
```

### Slow performance
**Solution:** Run without retriever for faster debates:
```bash
python main.py --mode sample --no-retriever
```

## Tips

1. **Start Simple**: Begin with `--no-retriever` to test agents
2. **Build KB**: Scrape docs → Build vector DB → Use full RAG
3. **Adjust Alpha**: In HybridRetriever, alpha=0.5 balances semantic/keyword
4. **Customize Agents**: Edit `agents/debate_agents.py` to add new roles
5. **More Rounds**: Use `--rounds 3` for deeper debates

## Next Steps

- Read full [README.md](README.md) for details
- Explore agent personalities in `agents/debate_agents.py`
- Customize retrieval in `retriever/hybrid_retriever.py`
- Add your own documents to `kb_docs/`

Happy debating! 🎭
