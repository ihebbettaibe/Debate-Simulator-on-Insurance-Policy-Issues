# 🎯 Multi-Model Ollama Setup Guide

## Why Different Models for Different Agents?

Each AI model has strengths suited to specific tasks. By assigning specialized models to each agent, you get:

- **Better debate quality**: Each agent uses a model optimized for their role
- **Resource efficiency**: Mix fast models with powerful ones  
- **Diverse perspectives**: Different reasoning styles = richer debates

---

## 📋 Model Assignments

| Agent Role | Model | Why This Model? |
|------------|-------|-----------------|
| **PRO Agent** | `llama3:8b` | Strong logic, structured arguments, coherent reasoning |
| **CON Agent** | `llama3:8b` | Deep reasoning for counter-arguments, critical analysis |
| **JUDGE Agent** | `mistral:7b` | Balanced tone, fact-based evaluation, objective assessment |
| **Analyst** | `mistral:7b` | Clean reasoning, low latency, data-driven analysis |
| **Advocate** | `qwen2:1.5b` | Fast, persuasive text, optimistic framing |
| **Skeptic** | `llama3:8b` | Deep reasoning, critical thinking, finds flaws |
| **Regulator** | `mistral:7b` | Balanced tone, structured facts, compliance focus |
| **Innovator** | `qwen2:1.5b` | Creative phrasing, tech-forward, innovative ideas |
| **Consumer** | `mistral:7b` | Natural-sounding, concise, customer-centric |
| **Moderator** | `llama3:8b` | Can synthesize multiple arguments, coordination |

---

## 🚀 Quick Setup (5 Minutes)

### Step 1: Install Ollama

**Windows:**
```powershell
# Download installer from:
https://ollama.com/download/windows
```

**Mac/Linux:**
```bash
curl -fsSL https://ollama.com/install.sh | sh
```

### Step 2: Pull Required Models

**Minimum (for Pro/Con debates):**
```bash
# Core models - 8-9GB total
ollama pull llama3:8b
ollama pull mistral:7b
```

**Full Setup (for all 6+ agent system):**
```bash
# Primary models
ollama pull llama3:8b      # 4.7GB - Logic & reasoning
ollama pull mistral:7b     # 4.1GB - Balanced analysis
ollama pull qwen2:1.5b     # 934MB - Fast & creative

# Optional but recommended
ollama pull llama3.2       # 2GB - Lightweight fallback
ollama pull phi3:mini      # 2.3GB - Fast summarizer
ollama pull qwen2:7b       # 4.4GB - Better quality creative
ollama pull gemma2:9b      # 5.4GB - Google's quality model
```

### Step 3: Verify Installation

```bash
# Check installed models
ollama list

# Test a model
ollama run llama3:8b
>>> Tell me about insurance AI
```

---

## 💻 Using in Your System

### Streamlit UI (Easy Mode)

1. **Start Ollama** (Windows: auto-starts; Linux/Mac: `ollama serve`)

2. **Run the app:**
   ```bash
   streamlit run app.py
   ```

3. **In sidebar settings:**
   - ✅ Check "Use Ollama LLM"
   - ✅ Check "Use Specialized Models"  
   - Watch as each agent uses their optimized model!

### Python Code

```python
from agents import create_debate_agents

# Automatic multi-model mode (uses config.py assignments)
agents = create_debate_agents(use_specialized_models=True)

# Single model for all agents
agents = create_debate_agents(
    model="llama3.2", 
    use_specialized_models=False
)

# Custom LLM
from langchain_ollama import OllamaLLM
custom_llm = OllamaLLM(model="llama3:8b", temperature=0.9)
agents = create_debate_agents(llm=custom_llm)
```

---

## 🎛️ Configuration

Edit `config.py` to customize model assignments:

```python
AGENT_MODELS = {
    "PRO": "llama3:8b",           # Your preferred PRO model
    "CON": "llama3:8b",           # Your preferred CON model
    "JUDGE": "mistral:7b",        # Your preferred JUDGE model
    "Analyst": "mistral:7b",
    "Advocate": "qwen2:1.5b",
    # ... customize any role
}

# Fallback if specialized model unavailable
DEFAULT_OLLAMA_MODEL = "llama3.2"
```

---

## 📊 Model Comparison

### Performance Metrics

| Model | Size | Speed | Quality | RAM Needed | Best For |
|-------|------|-------|---------|------------|----------|
| **llama3.2** | 2GB | ⚡⚡⚡⚡⚡ | ⭐⭐⭐ | 4GB | Testing, fallback |
| **qwen2:1.5b** | 934MB | ⚡⚡⚡⚡⚡ | ⭐⭐⭐ | 3GB | Fast responses, creative |
| **phi3:mini** | 2.3GB | ⚡⚡⚡⚡ | ⭐⭐⭐ | 4GB | Summaries, quick tasks |
| **mistral:7b** | 4.1GB | ⚡⚡⚡ | ⭐⭐⭐⭐ | 8GB | Analysis, balanced |
| **llama3:8b** | 4.7GB | ⚡⚡⚡ | ⭐⭐⭐⭐ | 8GB | Deep reasoning, logic |
| **qwen2:7b** | 4.4GB | ⚡⚡⚡ | ⭐⭐⭐⭐ | 8GB | Creative, conversational |
| **gemma2:9b** | 5.4GB | ⚡⚡ | ⭐⭐⭐⭐⭐ | 10GB | Highest quality |

### Reasoning Style Comparison

**llama3:8b** - Structured Thinker
```
✓ Logical progression
✓ Clear argument structure  
✓ Evidence-based reasoning
✗ Can be verbose
```

**mistral:7b** - Balanced Analyst
```
✓ Concise and clear
✓ Fact-focused
✓ Objective tone
✗ Less creative
```

**qwen2:1.5b** - Creative Persuader
```
✓ Fast responses
✓ Engaging language
✓ Persuasive framing
✗ Less depth
```

---

## 🎯 Resource Planning

### System Requirements

**Minimum (Pro/Con only):**
- **RAM**: 8GB
- **Storage**: 10GB  
- **Models**: llama3:8b + mistral:7b
- **Performance**: Good for 2 agents

**Recommended (Pro/Con/Judge):**
- **RAM**: 12GB
- **Storage**: 15GB
- **Models**: llama3:8b + mistral:7b + llama3.2
- **Performance**: Smooth 3 agents

**Full System (6+ agents):**
- **RAM**: 16GB+
- **Storage**: 20GB
- **Models**: All primary models
- **Performance**: Run 3-4 agents simultaneously

### Running Multiple Agents

**16GB+ RAM**: Run all agents with specialized models simultaneously
```python
agents = create_debate_agents_with_judge(use_specialized_models=True)
# PRO (llama3:8b) + CON (llama3:8b) + JUDGE (mistral:7b) = ~13GB RAM
```

**8-12GB RAM**: Use lighter models or single model mode
```python
# Option 1: Use lighter models
AGENT_MODELS = {
    "PRO": "llama3.2",
    "CON": "llama3.2", 
    "JUDGE": "llama3.2"
}

# Option 2: Single model mode
agents = create_debate_agents(
    model="llama3.2",
    use_specialized_models=False
)
```

**Under 8GB RAM**: Use phi3:mini or run one agent at a time
```python
agents = create_debate_agents(model="phi3:mini")
```

---

## 🎨 Example Debate Output

### Topic: "AI underwriting should be mandatory"

**PRO Agent (llama3:8b):**
```
I strongly advocate for mandatory AI underwriting. Let me present 
three evidence-based arguments:

1. **Accuracy & Fairness**: Studies show AI models reduce bias by 
   40% compared to manual underwriting, as they evaluate purely on 
   data without demographic prejudice...

2. **Economic Efficiency**: Implementation costs are offset within 
   18 months through reduced fraud (15-20% savings) and improved 
   risk assessment leading to better pricing...

3. **Consumer Benefits**: Approval times drop from 7-14 days to 
   under 24 hours, while personalized premiums save consumers an 
   average of 12-18% based on actual risk profiles...

[Detailed, structured, logical argument with evidence]
```

**CON Agent (llama3:8b):**
```
I must respectfully oppose mandatory AI underwriting on critical grounds:

1. **Algorithmic Opacity**: Black-box AI models make decisions that 
   cannot be explained to customers, violating transparency principles 
   essential in financial services. When claims are denied, customers 
   deserve clear reasoning...

2. **Data Privacy Concerns**: Mandatory AI requires extensive data 
   collection (driving habits, health metrics, social media). This 
   creates surveillance capitalism and privacy invasion risks that 
   outweigh efficiency gains...

3. **Systemic Risk**: Over-reliance on AI creates single points of 
   failure. The 2023 Allianz incident showed how algorithm errors 
   can misclassify thousands of policies simultaneously...

[Critical, thorough, evidence-based counter-arguments]
```

**JUDGE Agent (mistral:7b):**
```
**Evaluation of Arguments:**

PRO Strengths:
- Strong empirical data on accuracy improvements (40% bias reduction)
- Clear economic case with 18-month ROI
- Quantified consumer benefits (12-18% savings)

PRO Weaknesses:
- Insufficient addressing of privacy concerns
- Assumes AI quality without discussing failure modes

CON Strengths:
- Valid privacy and transparency concerns
- Concrete example (Allianz 2023 incident)
- Philosophical argument about explainability

CON Weaknesses:
- Doesn't propose alternative solutions
- May overstate systemic risk severity

**Verdict**: Both sides present compelling evidence. PRO makes strong 
efficiency case; CON raises legitimate ethical concerns. Recommend 
regulated implementation rather than blanket mandate.

[Balanced, objective, structured evaluation]
```

---

## 🔧 Advanced: Temperature Tuning

Different tasks need different creativity levels:

```python
from langchain_ollama import OllamaLLM

# Factual analysis (Judge, Regulator)
analytical_llm = OllamaLLM(model="mistral:7b", temperature=0.3)

# Balanced debate (PRO, CON)
debate_llm = OllamaLLM(model="llama3:8b", temperature=0.7)

# Creative advocacy (Innovator, Advocate)
creative_llm = OllamaLLM(model="qwen2:7b", temperature=0.9)
```

---

## 🐛 Troubleshooting

### "Model not found"
```bash
# Pull the required model
ollama pull llama3:8b

# Verify it's there
ollama list
```

### "Out of memory"
```python
# Use smaller models in config.py
AGENT_MODELS = {
    "PRO": "llama3.2",      # 2GB instead of 4.7GB
    "CON": "llama3.2",
    "JUDGE": "phi3:mini"    # 2.3GB instead of 4.1GB
}
```

### "Connection refused"
```bash
# Windows: Check system tray for Ollama icon
# Mac/Linux: Start manually
ollama serve

# Test connection
curl http://localhost:11434
```

### Slow response times
```bash
# Use faster models
ollama pull qwen2:1.5b   # Much faster than llama3:8b
ollama pull phi3:mini    # Fast and efficient

# Or reduce temperature
llm = OllamaLLM(model="llama3:8b", temperature=0.5)
```

### Model switching errors
```python
# Fallback is built-in
# If llama3:8b fails, system auto-tries llama3.2

# Or set custom fallback in config.py
DEFAULT_OLLAMA_MODEL = "phi3:mini"  # Lightweight fallback
```

---

## 📈 Performance Tips

### 1. Model Loading (First Response is Slow)
```python
# Pre-load models in background
import subprocess
subprocess.Popen(["ollama", "run", "llama3:8b", "hello"])
subprocess.Popen(["ollama", "run", "mistral:7b", "hello"])
```

### 2. Parallel Agents (if RAM allows)
```python
import concurrent.futures

def agent_response(agent, topic):
    return agent.generate_response(topic)

with concurrent.futures.ThreadPoolExecutor() as executor:
    futures = [executor.submit(agent_response, a, topic) for a in agents]
    responses = [f.result() for f in futures]
```

### 3. Response Caching
```python
# Cache common queries
@st.cache_data
def get_agent_response(agent_name, topic):
    return agent.generate_response(topic)
```

---

## 🎓 Best Practices

### 1. Start Small
- Test with `llama3.2` first
- Verify system works
- Then add specialized models

### 2. Match Models to RAM
- **4-8GB**: Use llama3.2 only
- **8-12GB**: Add mistral:7b
- **12-16GB**: Add llama3:8b
- **16GB+**: Full multi-model setup

### 3. Profile Your Debates
```python
import time

start = time.time()
response = agent.generate_response(topic)
print(f"Response time: {time.time() - start:.2f}s")
```

### 4. Monitor Resource Usage
```bash
# Check Ollama processes
ps aux | grep ollama

# Check RAM usage
htop  # Linux/Mac
# or Task Manager on Windows
```

---

## 🌟 Example Configurations

### Configuration 1: Speed Focus (8GB RAM)
```python
AGENT_MODELS = {
    "PRO": "qwen2:1.5b",
    "CON": "qwen2:1.5b",
    "JUDGE": "phi3:mini",
}
# Fast debates, good quality, low resource use
```

### Configuration 2: Quality Focus (16GB RAM)
```python
AGENT_MODELS = {
    "PRO": "llama3:8b",
    "CON": "llama3:8b",
    "JUDGE": "gemma2:9b",
}
# Best argument quality, deeper reasoning
```

### Configuration 3: Balanced (12GB RAM)
```python
AGENT_MODELS = {
    "PRO": "llama3:8b",
    "CON": "llama3:8b",
    "JUDGE": "mistral:7b",
}
# Current default - good quality, manageable resources
```

### Configuration 4: Creative (10GB RAM)
```python
AGENT_MODELS = {
    "PRO": "qwen2:7b",
    "CON": "llama3:8b",
    "JUDGE": "mistral:7b",
}
# PRO is creative, CON is critical, JUDGE is balanced
```

---

## 📚 Learn More

- **Ollama Docs**: https://github.com/ollama/ollama
- **Model Library**: https://ollama.com/library
- **LangChain Ollama**: https://python.langchain.com/docs/integrations/llms/ollama
- **Model Benchmarks**: https://ollama.com/blog/benchmarks

---

## 🎉 You're Ready!

```bash
# Pull core models
ollama pull llama3:8b
ollama pull mistral:7b

# Run your multi-model debate system
streamlit run app.py

# Enable "Use Specialized Models" in sidebar
# Watch each agent use their optimized model!
```

**Each agent now uses the perfect model for their role! 🚀**
