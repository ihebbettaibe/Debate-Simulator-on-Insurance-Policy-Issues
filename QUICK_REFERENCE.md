# 🎯 Multi-Model Quick Reference Card

## 📦 Required Models

```bash
# Minimum (Pro/Con only)
ollama pull llama3:8b      # 4.7GB - Main reasoning
ollama pull mistral:7b     # 4.1GB - Judge evaluation

# Fallback (lightweight testing)
ollama pull llama3.2       # 2GB - Quick testing

# Optional (enhanced)
ollama pull qwen2:1.5b     # 934MB - Fast & creative
ollama pull phi3:mini      # 2.3GB - Efficient processing
ollama pull qwen2:7b       # 4.4GB - Better creativity
ollama pull gemma2:9b      # 5.4GB - Highest quality
```

---

## 🎛️ Current Configuration

**File: `config.py`**

```python
AGENT_MODELS = {
    "PRO": "llama3:8b",       # ⚡ Strong logic
    "CON": "llama3:8b",       # 🧠 Deep reasoning
    "JUDGE": "mistral:7b",    # ⚖️ Balanced evaluation
}

DEFAULT_OLLAMA_MODEL = "llama3.2"  # 🔄 Fallback
```

---

## 🚀 Usage

### Streamlit UI
```bash
streamlit run app.py
```
- ✅ Check "Use Ollama LLM"
- ✅ Check "Use Specialized Models"

### Python Code
```python
# Multi-model (each agent optimized)
agents = create_debate_agents(use_specialized_models=True)

# Single model (all agents same)
agents = create_debate_agents(model="llama3.2", use_specialized_models=False)
```

---

## 💡 Model Selection Guide

| Role | Best Model | Alt Model | Fast Model |
|------|-----------|-----------|------------|
| PRO | llama3:8b | mistral:7b | llama3.2 |
| CON | llama3:8b | mistral:7b | llama3.2 |
| JUDGE | mistral:7b | llama3:8b | phi3:mini |

---

## 🎯 Resource Requirements

| Setup | RAM | Models | Agents |
|-------|-----|--------|--------|
| **Test** | 4GB | llama3.2 | 2 |
| **Standard** | 12GB | llama3:8b + mistral:7b | 3 |
| **Full** | 16GB | All models | 6+ |

---

## 🔧 Quick Customization

### Change Model for an Agent

**Edit `config.py`:**
```python
AGENT_MODELS = {
    "PRO": "gemma2:9b",    # Change to higher quality
    "CON": "qwen2:7b",     # Change to more creative
    "JUDGE": "llama3:8b",  # Change to deeper analysis
}
```

### Use Lightweight Setup

```python
AGENT_MODELS = {
    "PRO": "llama3.2",
    "CON": "llama3.2",
    "JUDGE": "phi3:mini",
}
```

### Maximum Quality Setup

```python
AGENT_MODELS = {
    "PRO": "gemma2:9b",
    "CON": "gemma2:9b",
    "JUDGE": "gemma2:9b",
}
```

---

## 🐛 Common Issues

| Issue | Solution |
|-------|----------|
| Model not found | `ollama pull llama3:8b` |
| Out of memory | Use `llama3.2` or `phi3:mini` |
| Slow responses | Use `qwen2:1.5b` for speed |
| Ollama not running | `ollama serve` (Mac/Linux) or check tray (Windows) |

---

## 📊 Model Characteristics

| Model | Size | Speed | Quality | Style |
|-------|------|-------|---------|-------|
| llama3.2 | 2GB | ⚡⚡⚡⚡⚡ | ⭐⭐⭐ | General |
| qwen2:1.5b | 934MB | ⚡⚡⚡⚡⚡ | ⭐⭐⭐ | Creative |
| phi3:mini | 2.3GB | ⚡⚡⚡⚡ | ⭐⭐⭐ | Concise |
| mistral:7b | 4.1GB | ⚡⚡⚡ | ⭐⭐⭐⭐ | Balanced |
| llama3:8b | 4.7GB | ⚡⚡⚡ | ⭐⭐⭐⭐ | Logical |
| qwen2:7b | 4.4GB | ⚡⚡⚡ | ⭐⭐⭐⭐ | Creative+ |
| gemma2:9b | 5.4GB | ⚡⚡ | ⭐⭐⭐⭐⭐ | Premium |

---

## 🎯 Test Commands

```bash
# Test single agent
python agents/debate_agents.py

# Run full UI
streamlit run app.py

# Check models installed
ollama list

# Test model directly
ollama run llama3:8b
>>> Tell me about insurance AI
```

---

## 📚 Documentation Files

- `IMPLEMENTATION_SUMMARY.md` - This implementation overview
- `MULTI_MODEL_GUIDE.md` - Detailed configuration guide
- `OLLAMA_GUIDE.md` - Ollama setup basics
- `PRO_CON_GUIDE.md` - Debate system usage

---

## 🎉 Quick Start (30 seconds)

```bash
# 1. Pull models
ollama pull llama3:8b

# 2. Run system
streamlit run app.py

# 3. In UI:
#    ✅ Use Ollama LLM
#    ✅ Use Specialized Models

# 4. Enter topic and start debate!
```

---

**🚀 You're ready to run multi-model debates!**

**Need help?** Check `MULTI_MODEL_GUIDE.md` for detailed info.
