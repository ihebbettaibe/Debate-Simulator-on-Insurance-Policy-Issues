# 🤖 Ollama LLM Integration Guide

## ✅ Now Using: Ollama (Local, Free LLM)

Your system now supports **Ollama** - a free, local LLM that runs on your machine!

---

## 🚀 Quick Setup (5 Minutes)

### 1. Install Ollama

**Windows:**
Download from: https://ollama.com/download/windows

**Mac:**
```bash
curl -fsSL https://ollama.com/install.sh | sh
```

**Linux:**
```bash
curl -fsSL https://ollama.com/install.sh | sh
```

### 2. Start Ollama Service

**Windows:**
- Ollama starts automatically after installation
- Check system tray for Ollama icon

**Mac/Linux:**
```bash
ollama serve
```

### 3. Pull a Model

```bash
# Recommended: Llama 3.2 (fastest, good quality)
ollama pull llama3.2

# Alternatives:
ollama pull llama3        # Larger, better quality
ollama pull mistral       # Fast, good for technical content
ollama pull phi3          # Smallest, fastest
ollama pull gemma2        # Google's model
```

### 4. Test It

```bash
ollama run llama3.2
>>> Hello, how are you?
```

---

## 🎯 Using Ollama in Your Debate System

### Streamlit UI (Easy)

1. Run the app:
   ```bash
   streamlit run app.py
   ```

2. In the sidebar:
   - ✅ Check "Use Ollama LLM"
   - Select your model (llama3.2, llama3, etc.)
   - Start debating!

### Python Code

```python
from agents import create_debate_agents

# Automatically uses Ollama if available
agents = create_debate_agents(use_ollama=True, model="llama3.2")

# Or disable LLM (use simulated responses)
agents = create_debate_agents(use_ollama=False)
```

---

## 📊 Model Comparison

| Model | Size | Speed | Quality | Best For |
|-------|------|-------|---------|----------|
| **llama3.2** | 2GB | ⚡⚡⚡ | ⭐⭐⭐ | Quick debates, testing |
| **llama3** | 4.7GB | ⚡⚡ | ⭐⭐⭐⭐ | Better arguments, depth |
| **mistral** | 4.1GB | ⚡⚡⚡ | ⭐⭐⭐ | Technical topics |
| **phi3** | 2.3GB | ⚡⚡⚡ | ⭐⭐ | Resource-constrained |
| **gemma2** | 5.4GB | ⚡⚡ | ⭐⭐⭐⭐ | Google's quality |

**Recommendation**: Start with `llama3.2` for speed, upgrade to `llama3` for quality.

---

## 🎓 How It Works

### Before (Simulated):
```python
response = "I argue in FAVOR of this topic. [Generic template]"
```

### Now (Ollama LLM):
```python
response = llm.invoke("""
You are a Pro agent in a debate about AI in insurance.
Topic: AI-powered underwriting should be mandatory

Evidence: [Retrieved documents from RAG]

Provide your PRO position with strong, evidence-based arguments.
""")
# → Real AI-generated response!
```

---

## 🔧 Configuration

### In Code (`agents/debate_agents.py`):

```python
# Default model
agents = create_debate_agents(model="llama3.2")

# Use different model
agents = create_debate_agents(model="llama3")

# Disable LLM
agents = create_debate_agents(use_ollama=False)

# Custom LLM
from langchain_ollama import OllamaLLM
custom_llm = OllamaLLM(model="llama3", temperature=0.9)
agents = create_debate_agents(llm=custom_llm)
```

### Temperature Control:

```python
from langchain_ollama import OllamaLLM

# More creative (0.7-1.0)
llm = OllamaLLM(model="llama3.2", temperature=0.9)

# More focused (0.3-0.7)
llm = OllamaLLM(model="llama3.2", temperature=0.5)

# Very deterministic (0.0-0.3)
llm = OllamaLLM(model="llama3.2", temperature=0.2)
```

---

## 🐛 Troubleshooting

### "Ollama not running"
```bash
# Start Ollama service
ollama serve

# Or on Windows, check system tray
```

### "Model not found"
```bash
# Pull the model first
ollama pull llama3.2

# List available models
ollama list
```

### "Connection refused"
```bash
# Check Ollama is running on default port
curl http://localhost:11434

# If using different port
export OLLAMA_HOST=http://localhost:11434
```

### Slow Response
```bash
# Use smaller model
ollama pull llama3.2  # Instead of llama3

# Or in app, select "llama3.2" or "phi3"
```

### Out of Memory
```bash
# Use smaller model
ollama pull phi3

# Or reduce context
# In app: Reduce "Documents per Query"
```

---

## 🎯 Example Debate Output

### Without LLM (Simulated):
```
**PRO POSITION on 'AI underwriting should be mandatory':**

I argue in FAVOR of this topic. Based on the evidence available,
there are compelling reasons to support this position.

**Key Arguments:**
1. The benefits clearly outweigh potential drawbacks
2. Data supports positive outcomes
3. This represents progress and opportunity
```

### With Ollama (Real AI):
```
**PRO POSITION on 'AI underwriting should be mandatory':**

I strongly advocate for mandatory AI-powered underwriting in the
insurance industry. The evidence overwhelmingly supports this position
through three key dimensions:

1. **Risk Assessment Accuracy**: Studies show AI models reduce
   underwriting errors by 40-60% compared to manual processes. The
   retrieved context indicates that machine learning algorithms can
   process 100x more data points than human underwriters...

2. **Economic Efficiency**: Implementation costs are offset within
   18-24 months through reduced claims fraud (estimated 15-20%
   reduction) and improved pricing accuracy...

3. **Consumer Benefit**: Faster approval times (from days to minutes)
   and more personalized premiums benefit consumers directly. The
   evidence from Swiss Re's 2024 report shows...

[Much more detailed, evidence-based argument]
```

---

## 📈 Next Steps

1. ✅ **Test basic Ollama**: `ollama run llama3.2`
2. ✅ **Run debate system**: `streamlit run app.py`
3. ✅ **Try different models**: Compare llama3.2 vs llama3
4. 📊 **Add RAG**: Build vector DB for evidence-based debates
5. ⚖️ **Enable Judge**: Get AI-powered verdicts
6. 🎨 **Customize**: Adjust temperature, prompts, models

---

## 🆚 Ollama vs Other LLMs

| Feature | Ollama | OpenAI GPT-4 | Claude |
|---------|--------|--------------|--------|
| **Cost** | FREE | $0.01-0.06/1K tokens | $0.008-0.024/1K tokens |
| **Privacy** | 100% Local | Cloud | Cloud |
| **Speed** | Fast (local) | Medium | Fast |
| **Quality** | Very Good | Excellent | Excellent |
| **Setup** | 5 minutes | API key needed | API key needed |
| **Internet** | Not required | Required | Required |

**Ollama is perfect for**:
- ✅ Development and testing
- ✅ Privacy-sensitive data
- ✅ Unlimited usage
- ✅ No API costs
- ✅ Offline capability

---

## 🎓 Advanced: Custom Models

### Fine-tune for Insurance:

```bash
# Create Modelfile
echo "FROM llama3.2
SYSTEM You are an insurance industry expert specialized in debates.
PARAMETER temperature 0.7" > Modelfile

# Create custom model
ollama create insurance-debater -f Modelfile

# Use in system
agents = create_debate_agents(model="insurance-debater")
```

---

## 📚 Learn More

- **Ollama Docs**: https://github.com/ollama/ollama
- **Model Library**: https://ollama.com/library
- **LangChain Ollama**: https://python.langchain.com/docs/integrations/llms/ollama

---

**🎉 You're now using real AI in your debates!**

Run: `streamlit run app.py` and toggle "Use Ollama LLM" ✅
