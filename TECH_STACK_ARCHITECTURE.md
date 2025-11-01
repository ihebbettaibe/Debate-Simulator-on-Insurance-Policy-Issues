# 🏗️ Technical Architecture & Tech Stack

## Complete System Architecture Documentation

---

## 📊 System Overview

### High-Level Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                     Streamlit Web Interface                      │
│                        (User Interface)                          │
└───────────────────────────┬─────────────────────────────────────┘
                            │
                            ▼
┌─────────────────────────────────────────────────────────────────┐
│                    Debate Orchestrator                           │
│              (Coordinates agent interactions)                    │
└───────────────────────────┬─────────────────────────────────────┘
                            │
                ┌───────────┼───────────┐
                ▼           ▼           ▼
        ┌───────────┐ ┌───────────┐ ┌───────────┐
        │ Pro Agent │ │ Con Agent │ │Judge Agent│
        └─────┬─────┘ └─────┬─────┘ └─────┬─────┘
              │             │             │
              └─────────────┼─────────────┘
                            ▼
                ┌───────────────────────┐
                │  Hybrid RAG Retriever │
                └───────────┬───────────┘
                            │
                ┌───────────┼───────────┐
                ▼           ▼           ▼
        ┌──────────┐  ┌─────────┐  ┌─────────┐
        │  FAISS   │  │  BM25   │  │  Ollama │
        │ (Vector) │  │(Keyword)│  │  (LLM)  │
        └──────────┘  └─────────┘  └─────────┘
```

---

## 🤖 Agent Architecture

### Core Agent Components

Each `DebateAgent` instance consists of:

```python
class DebateAgent:
    # Identity & Role
    name: str                    # Agent identifier
    role: AgentRole              # PRO, CON, or JUDGE
    personality: str             # Role-specific behavior description
    
    # AI Components
    llm: OllamaLLM              # Language model (or None for simulated)
    retriever: HybridRetriever  # RAG system (optional)
    
    # Memory Systems
    conversation_history: List[Dict]  # Debate history
    evidence_used: List[Document]     # Retrieved documents
    
    # Performance Tracking
    metrics: Dict[str, Any]     # Performance metrics
```

---

## 🛠️ Tech Stack Breakdown

### **Frontend Layer**

| Component | Technology | Version | Purpose |
|-----------|-----------|---------|---------|
| **UI Framework** | Streamlit | 1.40.1 | Web interface |
| **Visualization** | Streamlit Charts | Built-in | Performance graphs |
| **Styling** | Custom HTML/CSS | - | UI enhancements |

### **Backend Layer**

| Component | Technology | Version | Purpose |
|-----------|-----------|---------|---------|
| **Language** | Python | 3.8+ | Core logic |
| **LLM Framework** | LangChain | 0.3.7 | Agent framework |
| **LLM Provider** | Ollama | 0.4.3 | Local LLM |
| **LLM Integration** | langchain-ollama | 0.2.0 | Ollama wrapper |

### **RAG (Retrieval) Layer**

| Component | Technology | Version | Purpose |
|-----------|-----------|---------|---------|
| **Vector DB** | FAISS | 1.9.0 | Semantic search |
| **Embeddings** | Sentence-Transformers | 3.3.1 | Text embeddings |
| **Keyword Search** | Rank-BM25 | 0.2.2 | BM25 algorithm |
| **Document Processing** | LangChain Community | 0.3.5 | Loaders & splitters |

### **Monitoring Layer**

| Component | Technology | Version | Purpose |
|-----------|-----------|---------|---------|
| **Process Monitor** | psutil | 6.1.0 | Memory tracking |
| **Time Tracking** | Python time | Built-in | Response times |
| **Metrics Storage** | Python dict | Built-in | In-memory metrics |

### **Utilities**

| Component | Technology | Version | Purpose |
|-----------|-----------|---------|---------|
| **Environment** | python-dotenv | - | Config management |
| **Data Processing** | pandas | - | Data manipulation |
| **Arrays** | numpy | - | Numerical operations |

---

## 🧠 Agent-Specific Tech Stack

### **1. PRO Agent**

#### Core Technologies
```python
Agent Configuration:
├── LLM: Ollama (llama3:8b - 4.7GB)
├── Temperature: 0.7
├── Context Window: 8192 tokens
└── Specialization: Logical reasoning, structured arguments
```

#### Tools & Capabilities
- **Text Generation**: OllamaLLM with llama3:8b
- **Document Retrieval**: HybridRetriever (FAISS + BM25)
- **Context Management**: 8K token window
- **Memory**: Conversation history list

#### Memory Structure
```python
{
    'conversation_history': [
        {
            'topic': 'AI underwriting',
            'response': 'I argue in favor...',
            'timestamp': datetime,
            'round': 1
        }
    ],
    'evidence_used': [
        Document(
            page_content='...',
            metadata={
                'source': 'file.txt',
                'relevance_score': 0.95,
                'confidence': 'High',
                'rank': 1
            }
        )
    ],
    'metrics': {
        'response_times': [1.2, 1.5, 1.3],
        'token_counts': [150, 180, 165],
        'memory_usage': [2.3, 2.5, 2.4],
        'model_used': 'llama3:8b'
    }
}
```

---

### **2. CON Agent**

#### Core Technologies
```python
Agent Configuration:
├── LLM: Ollama (llama3:8b - 4.7GB)
├── Temperature: 0.7
├── Context Window: 8192 tokens
└── Specialization: Critical analysis, counter-arguments
```

#### Tools & Capabilities
- **Text Generation**: OllamaLLM with llama3:8b
- **Document Retrieval**: HybridRetriever (FAISS + BM25)
- **Context Management**: 8K token window
- **Memory**: Conversation history + opponent arguments

#### Memory Structure
```python
{
    'conversation_history': [
        {
            'topic': 'AI underwriting',
            'response': 'I argue against...',
            'opponent_args': ['Pro argument 1', 'Pro argument 2'],
            'timestamp': datetime,
            'round': 1
        }
    ],
    'evidence_used': [Document(...)],
    'metrics': {...}
}
```

---

### **3. JUDGE Agent**

#### Core Technologies
```python
Agent Configuration:
├── LLM: Ollama (mistral:7b - 4.1GB)
├── Temperature: 0.7
├── Context Window: 8192 tokens
└── Specialization: Balanced evaluation, objective analysis
```

#### Tools & Capabilities
- **Text Generation**: OllamaLLM with mistral:7b
- **Document Retrieval**: HybridRetriever (FAISS + BM25)
- **Multi-Argument Analysis**: Processes all PRO/CON arguments
- **Verdict Generation**: Synthesizes comprehensive evaluation

#### Memory Structure
```python
{
    'conversation_history': [
        {
            'topic': 'AI underwriting',
            'all_arguments': [
                'Pro: argument 1',
                'Con: argument 1',
                'Pro: argument 2',
                'Con: argument 2'
            ],
            'verdict': 'Based on evidence...',
            'timestamp': datetime
        }
    ],
    'evidence_used': [Document(...)],
    'metrics': {...}
}
```

---

## 🔍 Retrieval-Augmented Generation (RAG) Stack

### **Hybrid Retriever Architecture**

```python
HybridRetriever:
├── Semantic Search (FAISS)
│   ├── Embeddings: sentence-transformers/all-MiniLM-L6-v2
│   ├── Vector Dimension: 384
│   ├── Index Type: Flat (L2 distance)
│   └── Similarity: Cosine similarity
│
├── Keyword Search (BM25)
│   ├── Algorithm: Okapi BM25
│   ├── Parameters: k1=1.5, b=0.75
│   └── Tokenization: Word-level
│
└── Fusion Strategy
    ├── Alpha: 0.5 (configurable)
    ├── Semantic Weight: 50%
    └── Keyword Weight: 50%
```

### **Document Processing Pipeline**

```
Raw Documents
    │
    ▼
┌─────────────────┐
│ Document Loader │ (PyPDF, TextLoader, DirectoryLoader)
└────────┬────────┘
         ▼
┌─────────────────┐
│  Text Splitter  │ (RecursiveCharacterTextSplitter)
│  Chunk: 1000    │
│  Overlap: 100   │
└────────┬────────┘
         ▼
┌─────────────────┐
│   Embeddings    │ (SentenceTransformer)
│   Model: L6-v2  │
└────────┬────────┘
         ▼
┌─────────────────┐
│  Vector Store   │ (FAISS Index)
└────────┬────────┘
         ▼
┌─────────────────┐
│  Hybrid Search  │ (FAISS + BM25)
└─────────────────┘
```

### **RAG Configuration**

```python
RAG Settings:
├── Chunk Size: 1000 characters
├── Chunk Overlap: 100 characters
├── Embedding Model: all-MiniLM-L6-v2 (384 dimensions)
├── Vector DB: FAISS (Flat index)
├── Keyword Search: BM25 (k1=1.5, b=0.75)
├── Hybrid Alpha: 0.5
├── Default K (results): 3-5 documents
├── Semantic K: 10 documents
└── BM25 K: 10 documents
```

---

## 💾 Memory Management

### **Agent Memory Types**

#### 1. **Short-Term Memory** (Conversation History)
```python
# Stored per debate round
conversation_history: List[Dict] = [
    {
        'topic': str,           # Current debate topic
        'response': str,        # Agent's response
        'round': int,           # Debate round number
        'timestamp': datetime,  # When generated
        'opponent_args': List[str]  # For context
    }
]
```

#### 2. **Evidence Memory** (Retrieved Documents)
```python
# Accumulated throughout debate
evidence_used: List[Document] = [
    Document(
        page_content: str,      # Document text
        metadata: {
            'source': str,      # File name
            'chunk_id': int,    # Chunk number
            'relevance_score': float,  # 0.0-1.0
            'confidence': str,  # High/Medium/Low
            'rank': int         # Position in results
        }
    )
]
```

#### 3. **Performance Memory** (Metrics)
```python
# Real-time performance tracking
metrics: Dict[str, Any] = {
    'response_times': List[float],     # Seconds per response
    'token_counts': List[int],         # Tokens per response
    'memory_usage': List[float],       # MB per response
    'total_responses': int,            # Total count
    'avg_response_time': float,        # Average seconds
    'total_tokens': int,               # Cumulative tokens
    'peak_memory_mb': float,           # Peak usage
    'model_used': str                  # Model identifier
}
```

### **Memory Persistence**

| Memory Type | Persistence | Scope | Cleanup |
|-------------|-------------|-------|---------|
| **Conversation** | Session | Per debate | Manual clear |
| **Evidence** | Session | Per debate | Manual clear |
| **Metrics** | Session | Per agent | On app restart |
| **Vector DB** | Disk | Global | Manual rebuild |
| **Session State** | Session | Global | On disconnect |

---

## 🎯 Context Management

### **Context Window Management**

Each agent manages context through multiple layers:

#### **Layer 1: System Prompt** (Static)
```python
system_prompt = f"""You are {agent.name}, an AI agent in insurance debate.

Role: {agent.role}
Personality: {agent.personality}

Responsibilities:
- Analyze insurance policies and trends
- Provide evidence-based arguments
- Cite sources when making claims
- Stay focused on insurance topics
"""
```

#### **Layer 2: Retrieved Context** (Dynamic)
```python
retrieved_context = """
[Source 1: insurance_trends_2025.txt] [Relevance: 1.0] [Confidence: High]
AI-powered underwriting systems have demonstrated...

[Source 2: market_analysis.txt] [Relevance: 0.9] [Confidence: High]
Industry adoption continues to accelerate...

[Source 3: risk_guidelines.pdf] [Relevance: 0.8] [Confidence: Medium]
Traditional methods face scaling challenges...
"""
```

#### **Layer 3: Opponent Arguments** (Dynamic)
```python
opponent_arguments = [
    "Pro: AI underwriting increases efficiency by 35%...",
    "Con: Privacy concerns remain unaddressed...",
    "Pro: Cost savings benefit consumers..."
]
```

#### **Layer 4: Current Topic** (Dynamic)
```python
topic = "AI-powered underwriting should be mandatory in insurance"
```

### **Complete Prompt Assembly**

```python
final_prompt = f"""
{system_prompt}

TOPIC: {topic}

RETRIEVED EVIDENCE:
{retrieved_context}

PREVIOUS ARGUMENTS:
{opponent_arguments}

YOUR TASK: Provide your {agent.role} perspective on this topic.
Use the evidence above to support your argument.
"""
```

### **Context Token Budget**

| Component | Typical Tokens | Max Tokens | Priority |
|-----------|---------------|------------|----------|
| **System Prompt** | ~200 | 500 | High |
| **Topic** | ~20 | 100 | High |
| **Retrieved Docs** | ~1500 | 3000 | Medium |
| **Opponent Args** | ~800 | 2000 | Medium |
| **Response Buffer** | ~500 | 2000 | High |
| **Safety Margin** | ~1000 | - | - |
| **Total Available** | ~4020 | 8192 | - |

---

## 🔧 Tool Integration

### **Available Tools per Agent**

#### **Core Tools (All Agents)**

1. **Text Generation**
   ```python
   Tool: OllamaLLM
   Input: Prompt (string)
   Output: Generated text (string)
   Model: llama3:8b / mistral:7b
   Temperature: 0.7
   Max Tokens: 2000
   ```

2. **Document Retrieval**
   ```python
   Tool: HybridRetriever
   Input: Query (string), K (int)
   Output: List[Document]
   Method: FAISS + BM25 hybrid search
   Default K: 3
   ```

3. **Context Formatting**
   ```python
   Tool: format_context()
   Input: List[Document]
   Output: Formatted string
   Features: Source citations, relevance scores
   ```

4. **Performance Tracking**
   ```python
   Tool: Built-in metrics system
   Tracks: Time, tokens, memory
   Storage: In-memory dictionary
   Access: get_metrics() method
   ```

#### **Specialized Tools**

**Judge Agent Only:**
- Multi-argument analysis
- Verdict synthesis
- Comparative evaluation

---

## 📊 Data Flow Architecture

### **Debate Execution Flow**

```
User Input (Topic)
    │
    ▼
┌─────────────────────┐
│ Debate Orchestrator │
└──────────┬──────────┘
           │
           ▼
    For each round:
           │
    ┌──────┴──────┐
    ▼             ▼
┌─────────┐  ┌─────────┐
│Pro Agent│  │Con Agent│
└────┬────┘  └────┬────┘
     │            │
     │ (1) Retrieve Context
     ▼            ▼
┌───────────────────────┐
│   Hybrid Retriever    │
│  (FAISS + BM25)       │
└───────────┬───────────┘
            │
            │ (2) Documents + Metadata
            ▼
     ┌──────┴──────┐
     ▼             ▼
┌─────────┐  ┌─────────┐
│Pro Agent│  │Con Agent│
└────┬────┘  └────┬────┘
     │            │
     │ (3) Format Context + Build Prompt
     ▼            ▼
┌─────────┐  ┌─────────┐
│ Ollama  │  │ Ollama  │
│ LLM     │  │ LLM     │
└────┬────┘  └────┬────┘
     │            │
     │ (4) Generated Response
     ▼            ▼
┌─────────┐  ┌─────────┐
│Pro Agent│  │Con Agent│
└────┬────┘  └────┬────┘
     │            │
     │ (5) Store in Memory + Track Metrics
     └──────┬─────┘
            ▼
     (Next round or Judge)
            │
            ▼
      ┌──────────┐
      │Judge (if │
      │enabled)  │
      └────┬─────┘
           │
           │ (6) Collect all arguments
           ▼
     ┌───────────┐
     │  Ollama   │
     │mistral:7b │
     └─────┬─────┘
           │
           │ (7) Verdict
           ▼
    ┌─────────────┐
    │   Display   │
    │   Results   │
    └─────────────┘
```

---

## 🗄️ Storage Architecture

### **Vector Database (FAISS)**

```
vectorstore/
└── faiss_index/
    ├── index.faiss          # Vector index (binary)
    ├── index.pkl            # Document metadata (pickle)
    └── docstore.pkl         # Document store (pickle)

Format: Binary (efficient storage)
Size: ~1-5MB per 1000 documents
Index Type: Flat L2 (brute force, accurate)
Persistence: Disk-based
Load Time: ~100-500ms
```

### **Session State (Streamlit)**

```python
st.session_state = {
    'debate_history': List[Dict],      # Past debates
    'orchestrator': DebateOrchestrator, # Current orchestrator
    'retriever': HybridRetriever,      # RAG system
    'current_agents': List[DebateAgent] # Active agents
}

Persistence: Browser session
Storage: In-memory (RAM)
Lifetime: Until page refresh/close
```

### **Configuration (Files)**

```
Project Root/
├── config.py              # System configuration
├── .env                   # Environment variables
├── requirements.txt       # Dependencies
└── kb_docs/              # Knowledge base
    └── (user documents)
```

---

## ⚡ Performance Specifications

### **Resource Requirements**

| Component | Min RAM | Recommended RAM | Disk Space |
|-----------|---------|-----------------|------------|
| **Streamlit App** | 200 MB | 500 MB | Minimal |
| **FAISS Index** | 50 MB | 200 MB | ~5 MB |
| **Ollama llama3:8b** | 6 GB | 8 GB | 4.7 GB |
| **Ollama mistral:7b** | 5 GB | 7 GB | 4.1 GB |
| **Python Runtime** | 100 MB | 300 MB | Minimal |
| **Total System** | 8 GB | 16 GB | 10 GB |

### **Performance Benchmarks**

| Operation | Time | Notes |
|-----------|------|-------|
| **App Startup** | 2-5s | Load dependencies |
| **Vector DB Load** | 0.1-0.5s | FAISS index |
| **Document Retrieval** | 0.05-0.2s | Hybrid search |
| **LLM Response (llama3:8b)** | 1-3s | 150 tokens |
| **LLM Response (mistral:7b)** | 0.8-2s | 150 tokens |
| **Full Debate (2 rounds)** | 8-15s | Without RAG |
| **Full Debate (2 rounds + RAG)** | 10-20s | With retrieval |

---

## 🔐 Security & Privacy

### **Data Flow Security**

| Layer | Data | Security Measure |
|-------|------|------------------|
| **User Input** | Debate topics | Client-side only |
| **Retrieved Docs** | KB documents | Local processing |
| **LLM Processing** | All prompts | Local Ollama (no external API) |
| **Responses** | Generated text | Session-only storage |
| **Metrics** | Performance data | In-memory only |

### **Privacy Features**

✅ **100% Local Processing** - No external API calls  
✅ **No Data Logging** - No persistent storage of debates  
✅ **Session-Based** - Data cleared on disconnect  
✅ **Offline Capable** - Works without internet  
✅ **User Control** - All data stays on local machine  

---

## 🔄 Scalability Considerations

### **Current Architecture**

| Aspect | Current | Scalable To | Method |
|--------|---------|-------------|--------|
| **Concurrent Debates** | 1 | 10+ | Threading/Async |
| **Agents per Debate** | 2-3 | 10+ | List scaling |
| **Documents in KB** | 100s | 10,000+ | FAISS handles well |
| **Debate History** | 10s | 1000+ | Database needed |
| **Concurrent Users** | 1 | 100+ | Multi-instance |

### **Scaling Strategies**

**Horizontal Scaling:**
- Deploy multiple Streamlit instances
- Load balance with nginx
- Shared vector database
- Redis for session state

**Vertical Scaling:**
- Upgrade to GPU for faster LLM
- Increase RAM for more agents
- SSD for faster vector DB access

---

## 📚 Summary: Complete Tech Stack

### **Core Technologies**

```
┌─────────────────────────────────────────┐
│           PRESENTATION LAYER            │
│  Streamlit 1.40.1 + HTML/CSS           │
└─────────────────┬───────────────────────┘
                  │
┌─────────────────▼───────────────────────┐
│          APPLICATION LAYER              │
│  Python 3.8+ + LangChain 0.3.7         │
└─────────────────┬───────────────────────┘
                  │
┌─────────────────▼───────────────────────┐
│             AI/ML LAYER                 │
│  Ollama 0.4.3 (llama3:8b, mistral:7b)  │
└─────────────────┬───────────────────────┘
                  │
┌─────────────────▼───────────────────────┐
│            RETRIEVAL LAYER              │
│  FAISS 1.9.0 + BM25 0.2.2              │
│  Sentence-Transformers 3.3.1            │
└─────────────────┬───────────────────────┘
                  │
┌─────────────────▼───────────────────────┐
│           MONITORING LAYER              │
│  psutil 6.1.0 + Python time            │
└─────────────────────────────────────────┘
```

### **Per-Agent Stack Summary**

**Each agent uses:**
- ✅ LangChain for orchestration
- ✅ Ollama for text generation
- ✅ FAISS + BM25 for retrieval
- ✅ Custom memory management
- ✅ Performance tracking with psutil
- ✅ Context management system
- ✅ Evidence scoring system

---

## 🎓 Key Takeaways

1. **Fully Local Stack** - No external dependencies
2. **Modular Architecture** - Easy to swap components
3. **Production-Ready** - Monitoring and metrics built-in
4. **Scalable Design** - Can handle growth
5. **Open Source** - All components are FOSS
6. **Resource Efficient** - Runs on consumer hardware
7. **Privacy-First** - All processing local

---

**This is a modern, professional AI debate system built with industry-standard tools!** 🚀
