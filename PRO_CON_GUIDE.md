# Pro vs Con Debate System - Quick Start Guide

## 🎯 System Overview

The system has been refactored to use a **Pro vs Con** debate format with:
- **Pro Agent**: Argues in favor of the topic
- **Con Agent**: Argues against the topic
- **Judge Agent**: Evaluates both sides (optional, ready to integrate)

## 🚀 Quick Setup

### 1. Install Dependencies
```bash
pip install streamlit
```

Or install all requirements:
```bash
pip install -r requirements.txt
```

### 2. Build Knowledge Base (Optional but Recommended)
```bash
python build_kb.py
```

### 3. Run the System

#### Option A: Streamlit Web UI (Recommended)
```bash
streamlit run app.py
```

Then open your browser to `http://localhost:8501`

#### Option B: Command Line
```bash
# Quick test without RAG
python main.py --mode sample --no-retriever

# Full system with RAG
python main.py --mode sample

# Interactive CLI
python main.py --mode interactive
```

## 🌐 Using the Streamlit UI

The web interface provides:

### Features
1. **Simple Interface**: Easy-to-use web UI
2. **Pro vs Con Debates**: Two agents argue opposing positions
3. **Optional Judge**: Toggle to include judge evaluation
4. **RAG Toggle**: Enable/disable document retrieval
5. **Customizable**:
   - Number of debate rounds (1-5)
   - Documents per query (1-10)
6. **Debate History**: View past debates
7. **Live Updates**: See arguments as they're generated

### How to Use

1. **Configure Settings** (Left sidebar):
   - Toggle RAG (requires vector DB)
   - Toggle Judge agent
   - Set number of rounds
   - Set documents per query

2. **Enter Topic**:
   - Type your debate topic
   - Or use example topics provided
   - Topics should be statements (e.g., "AI should be used in underwriting")

3. **Start Debate**:
   - Click "Start Debate"
   - Watch Pro and Con agents argue
   - See Judge's verdict (if enabled)

4. **View History**:
   - Check past debates in History tab
   - See configuration of each debate

## 🤖 Agent Roles

### Pro Agent
```python
Role: Argues FOR the topic
Focus:
- Supporting arguments
- Benefits and opportunities
- Positive evidence
- Constructive solutions
```

### Con Agent
```python
Role: Argues AGAINST the topic
Focus:
- Opposing arguments
- Risks and challenges
- Critical analysis
- Counterevidence
```

### Judge Agent (Ready to Use)
```python
Role: Evaluates both sides
Focus:
- Objective assessment
- Evidence quality
- Logical reasoning
- Balanced verdict
```

## 📝 Example Topics

Good topics are clear statements that can be debated:

✅ **Good Topics:**
- "AI-powered underwriting should be mandatory in insurance"
- "Climate change insurance should be government-subsidized"
- "Parametric insurance is superior to traditional policies"
- "Insurance companies should be allowed to use genetic data"

❌ **Avoid:**
- Questions: "Should AI be used?" → Make it a statement
- Too broad: "Insurance technology" → Be specific
- Neutral: "AI in insurance" → Take a position

## 🔧 Integration with Judge Agent

The Judge agent is ready to use. To enable it:

### In Streamlit UI:
- Check "Include Judge Agent" in sidebar
- Judge will evaluate after all rounds complete

### In Code:
```python
from agents import create_debate_agents_with_judge

# Create agents with judge
agents = create_debate_agents_with_judge(retriever=retriever)

# Orchestrate debate
orchestrator = DebateOrchestrator(agents, retriever)
debate = orchestrator.conduct_debate(topic, rounds=2)
```

## 📊 System Architecture

```
Pro Agent  ←→  Evidence Retrieval (RAG)
    ↓
  Debate
    ↓
Con Agent  ←→  Evidence Retrieval (RAG)
    ↓
Judge Agent ←→  Evidence Retrieval (RAG)
    ↓
  Verdict
```

## 🎓 Code Examples

### Basic Pro/Con Debate
```python
from agents import create_debate_agents, DebateOrchestrator

# Create agents
agents = create_debate_agents()

# Create orchestrator
orchestrator = DebateOrchestrator(agents)

# Conduct debate
debate = orchestrator.conduct_debate(
    topic="AI underwriting should be mandatory",
    rounds=2,
    retrieve_context=False  # Without RAG
)
```

### With RAG
```python
from retriever import HybridRetriever
from agents import create_debate_agents, DebateOrchestrator

# Load retriever
retriever = HybridRetriever("./vectorstore/faiss_index")

# Create agents with retriever
agents = create_debate_agents(retriever=retriever)

# Create orchestrator
orchestrator = DebateOrchestrator(agents, retriever)

# Conduct debate with evidence
debate = orchestrator.conduct_debate(
    topic="Cyber insurance should cover ransomware",
    rounds=2,
    retrieve_context=True,
    context_k=3
)
```

### With Judge
```python
from retriever import HybridRetriever
from agents import create_debate_agents_with_judge, DebateOrchestrator

# Load retriever
retriever = HybridRetriever("./vectorstore/faiss_index")

# Create agents including judge
agents = create_debate_agents_with_judge(retriever=retriever)

# The last agent is the judge
pro_agent = agents[0]
con_agent = agents[1]
judge_agent = agents[2]

# Create orchestrator
orchestrator = DebateOrchestrator(agents, retriever)

# Conduct debate
debate = orchestrator.conduct_debate(
    topic="Parametric insurance is better than traditional",
    rounds=2,
    retrieve_context=True
)

# Judge evaluation happens in final round
```

## 🛠️ Customization

### Add More Agent Roles
Edit `agents/debate_agents.py`:

```python
class AgentRole(Enum):
    PRO = "pro"
    CON = "con"
    JUDGE = "judge"
    MODERATOR = "moderator"  # Add new role
```

### Modify Agent Behavior
Edit response templates in `_generate_simulated_response()`:

```python
AgentRole.PRO: f"""
**PRO POSITION on '{topic}':**
[Your custom template]
"""
```

### Change UI Appearance
Edit `app.py` to customize Streamlit interface:
- Colors and styling
- Layout structure
- Additional tabs/features

## 📈 Next Steps

1. **Test the basic system**: Run without RAG first
2. **Build knowledge base**: Add your documents
3. **Test with RAG**: Enable retrieval for evidence-based debates
4. **Customize agents**: Modify personalities and prompts
5. **Integrate LLM**: Replace simulated responses with GPT-4/Claude
6. **Extend UI**: Add more features to Streamlit app

## 🔍 Troubleshooting

### "Vector database not found"
- Run: `python build_kb.py`
- Or disable RAG in UI

### "Import streamlit could not be resolved"
- Run: `pip install streamlit`

### Agents give generic responses
- Enable RAG for evidence-based arguments
- Or integrate an LLM (see config.py)

### UI doesn't start
- Check port 8501 is available
- Try: `streamlit run app.py --server.port 8502`

## 📚 Documentation

- **README.md**: Full project documentation
- **IMPLEMENTATION.md**: Technical details
- **QUICKSTART.md**: Original quick start
- **This file**: Pro/Con specific guide

---

**Ready to debate!** 🎭

Run: `streamlit run app.py`
