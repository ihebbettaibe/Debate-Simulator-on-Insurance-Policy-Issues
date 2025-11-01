"""
Simple Streamlit UI for Insurance Debate System
Pro vs Con debate with optional Judge
"""
import streamlit as st
import sys
import os

# Add project root to path
project_root = os.path.dirname(os.path.abspath(__file__))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from agents import create_debate_agents, create_debate_agents_with_judge, DebateOrchestrator
from retriever import HybridRetriever
from config import AGENT_MODELS, DEFAULT_OLLAMA_MODEL

# Page config
st.set_page_config(
    page_title="Insurance Debate System - Multi-Model AI",
    page_icon="🏢",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Initialize session state
if 'debate_history' not in st.session_state:
    st.session_state.debate_history = []
if 'orchestrator' not in st.session_state:
    st.session_state.orchestrator = None
if 'retriever' not in st.session_state:
    st.session_state.retriever = None

# Title with custom styling
st.markdown("""
    <h1 style='text-align: center; color: #1f77b4;'>
        Insurance Policy Debate System
    </h1>
    <h3 style='text-align: center; color: #666;'>
        Multi-Model AI Debate Platform
    </h3>
""", unsafe_allow_html=True)

st.markdown("---")

# Sidebar configuration
st.sidebar.header("⚙️ Configuration")

# LLM Configuration
st.sidebar.markdown("### LLM Configuration")

use_llm = st.sidebar.checkbox(
    "Use Ollama LLM", 
    value=True,
    help="Enable local LLM via Ollama (requires Ollama to be running)"
)

if use_llm:
    st.sidebar.markdown("---")
    
    # Specialized models vs single model
    use_specialized = st.sidebar.checkbox(
        "Use Specialized Models", 
        value=True,
        help="Each agent uses an optimized model for their role"
    )
    
    if use_specialized:
        st.sidebar.success("Multi-Model Mode Active")
        
        # Show model assignments
        st.sidebar.markdown("**Model Assignments:**")
        st.sidebar.text(f"PRO:   {AGENT_MODELS.get('PRO', 'llama3:8b')}")
        st.sidebar.text(f"CON:   {AGENT_MODELS.get('CON', 'llama3:8b')}")
        st.sidebar.text(f"JUDGE: {AGENT_MODELS.get('JUDGE', 'mistral:7b')}")
        
        # Model info
        with st.sidebar.expander("Model Details"):
            st.markdown("""
            **llama3:8b** (PRO/CON)
            - Strong logical reasoning
            - Structured arguments
            
            **mistral:7b** (JUDGE)
            - Balanced evaluation
            - Fact-focused
            """)
    else:
        st.sidebar.info("Single Model Mode")
        ollama_model = st.sidebar.selectbox(
            "Select Model",
            ["llama3.2", "llama3:8b", "llama3", "mistral:7b", "mistral", 
             "qwen2:7b", "qwen2:1.5b", "phi3:mini", "phi3", "gemma2:9b", "gemma2"],
            index=0,
            help="All agents will use the same model"
        )
        
        st.sidebar.caption(f"Selected: {ollama_model}")
else:
    use_specialized = False
    ollama_model = None
    st.sidebar.warning("Using simulated responses (no LLM)")

st.sidebar.markdown("---")

# Debate Configuration
st.sidebar.markdown("### Debate Configuration")

# Toggle RAG
use_rag = st.sidebar.checkbox(
    "Enable RAG (Retrieval)", 
    value=True,
    help="Retrieve evidence from knowledge base to support arguments"
)

if use_rag:
    # Context documents
    context_k = st.sidebar.slider(
        "Documents per Query", 
        min_value=1, 
        max_value=10, 
        value=3,
        help="Number of relevant documents to retrieve per agent"
    )
else:
    context_k = 3

# Toggle Judge
include_judge = st.sidebar.checkbox(
    "Include Judge Agent", 
    value=False,
    help="Add a judge to evaluate both sides and provide a verdict"
)

# Number of rounds
rounds = st.sidebar.slider(
    "Debate Rounds", 
    min_value=1, 
    max_value=5, 
    value=2,
    help="Number of back-and-forth rounds between agents"
)

st.sidebar.markdown("---")

# System Info
st.sidebar.markdown("### About")

with st.sidebar.expander("Agent Roles", expanded=False):
    st.markdown("""
    **Pro Agent**
    - Argues in favor
    - Evidence-based support
    
    **Con Agent**
    - Argues against
    - Critical analysis
    
    **Judge Agent**
    - Evaluates both sides
    - Balanced verdict
    """)

with st.sidebar.expander("RAG System", expanded=False):
    st.markdown("""
    **Hybrid Retrieval:**
    - Semantic search (FAISS)
    - Keyword search (BM25)
    
    **Knowledge Base:**
    - Insurance reports
    - Industry research
    """)

st.sidebar.markdown("---")

# Setup Guide
st.sidebar.markdown("### � Setup Guide")

with st.sidebar.expander("📦 Install Ollama Models", expanded=False):
    st.markdown("""
    **Required (Multi-Model):**
    ```bash
    ollama pull llama3:8b
    ollama pull mistral:7b
    ```
    
    **Quick Test:**
    ```bash
    ollama pull llama3.2
    ```
    
    **Optional (Enhanced):**
    ```bash
    ollama pull qwen2:7b
    ollama pull phi3:mini
    ```
    """)

with st.sidebar.expander("🔧 Troubleshooting", expanded=False):
    st.markdown("""
    **Model not found:**
    ```bash
    ollama list
    ollama pull <model-name>
    ```
    
    **Ollama not running:**
    - Windows: Check system tray
    - Mac/Linux: `ollama serve`
    
    **Out of memory:**
    - Use llama3.2 (lighter)
    - Disable specialized models
    """)

# Initialize system
@st.cache_resource
def load_retriever():
    """Load the hybrid retriever."""
    try:
        vector_db_path = os.path.join(project_root, "vectorstore", "faiss_index")
        if os.path.exists(vector_db_path):
            retriever = HybridRetriever(vector_db_path, alpha=0.5)
            return retriever
        else:
            st.warning("⚠️ Vector database not found. Run `python build_kb.py` first.")
            return None
    except Exception as e:
        st.error(f"❌ Error loading retriever: {e}")
        return None

# Main interface
tab1, tab2, tab3, tab4, tab5 = st.tabs(["New Debate", "Performance Metrics", "Knowledge Base", "History", "Info"])

with tab1:
    st.markdown("## Start a New Debate")
    
    # Configuration summary
    col1, col2, col3 = st.columns(3)
    with col1:
        if use_llm and use_specialized:
            st.metric("Mode", "Multi-Model")
        elif use_llm:
            st.metric("Mode", "Single Model")
        else:
            st.metric("Mode", "Simulated")
    
    with col2:
        agent_count = 3 if include_judge else 2
        st.metric("Agents", agent_count)
    
    with col3:
        st.metric("RAG", "Enabled" if use_rag else "Disabled")
    
    st.markdown("---")
    
    # Topic input
    topic = st.text_input(
        "Debate Topic",
        value="AI-powered underwriting should be mandatory in insurance",
        help="Enter any insurance-related topic for debate",
        placeholder="Enter your topic here..."
    )
    
    # Example topics
    with st.expander("Example Topics"):
        st.markdown("""
        - AI-powered underwriting should be mandatory in insurance
        - Climate change insurance should be government-subsidized
        - Parametric insurance is better than traditional policies
        - Cyber insurance should cover ransomware payments
        - Insurance companies should use genetic data for pricing
        - Peer-to-peer insurance will replace traditional models
        """)
    
    # Start debate button
    if st.button("Start Debate", type="primary"):
        if not topic.strip():
            st.error("Please enter a debate topic!")
        else:
            with st.spinner("Initializing debate system..."):
                # Load retriever if RAG enabled
                retriever = None
                if use_rag:
                    retriever = load_retriever()
                
                # Create agents with specialized models
                model_to_use = None
                if use_llm and not use_specialized:
                    model_to_use = ollama_model
                
                if include_judge:
                    agents = create_debate_agents_with_judge(
                        retriever=retriever,
                        use_ollama=use_llm,
                        model=model_to_use,
                        use_specialized_models=use_specialized
                    )
                else:
                    agents = create_debate_agents(
                        retriever=retriever,
                        use_ollama=use_llm,
                        model=model_to_use,
                        use_specialized_models=use_specialized
                    )
                
                # Create orchestrator
                orchestrator = DebateOrchestrator(agents, retriever=retriever)
                
                # Show status
                st.success(f"{len(agents)} agents ready")
                
                if use_llm:
                    if use_specialized:
                        models_text = f"PRO: {AGENT_MODELS.get('PRO', 'llama3:8b')} | CON: {AGENT_MODELS.get('CON', 'llama3:8b')}"
                        if include_judge:
                            models_text += f" | JUDGE: {AGENT_MODELS.get('JUDGE', 'mistral:7b')}"
                        st.info(f"Multi-Model: {models_text}")
                    elif model_to_use:
                        st.info(f"Single Model: {model_to_use}")
                    else:
                        st.info(f"Default Model: {DEFAULT_OLLAMA_MODEL}")
                else:
                    st.info("Simulated responses")
            
            # Conduct debate
            st.markdown("---")
            
            # Topic header
            st.markdown(f"""
                <div style='background-color: #f0f2f6; padding: 15px; border-radius: 5px; margin: 15px 0;'>
                    <h3 style='color: #1f77b4; margin: 0;'>Topic: {topic}</h3>
                </div>
            """, unsafe_allow_html=True)
            
            # Progress bar
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            total_steps = rounds * len([a for a in agents if a.role.value != "judge" or include_judge])
            current_step = 0
            
            # Show debate rounds
            for round_num in range(1, rounds + 1):
                st.markdown(f"### Round {round_num} of {rounds}")
                
                for agent in agents:
                    # Skip judge in regular rounds
                    if agent.role.value == "judge" and round_num < rounds:
                        continue
                        
                    # Update progress
                    current_step += 1
                    progress_bar.progress(current_step / total_steps)
                    status_text.text(f"Round {round_num}/{rounds}: {agent.name} preparing argument...")
                    
                    # Retrieve context if RAG enabled
                    context = ""
                    retrieved_docs = []
                    if use_rag and retriever:
                        docs = agent.retrieve_context(f"{topic} {agent.role.value}", k=context_k)
                        retrieved_docs = docs[:2]
                        context = "\n".join([doc.page_content[:300] for doc in retrieved_docs])
                    
                    # Generate response
                    response = agent.generate_response(
                        topic=topic,
                        context=context,
                        opponent_arguments=[]
                    )
                        
                    # Display response
                    role_labels = {
                        "pro": "PRO",
                        "con": "CON",
                        "judge": "JUDGE"
                    }
                    role_label = role_labels.get(agent.role.value, agent.role.value.upper())
                    
                    with st.expander(f"**{agent.name}** ({role_label})", expanded=True):
                        # Show model being used
                        if use_llm:
                            if use_specialized and agent.role.value.upper() in AGENT_MODELS:
                                model_used = AGENT_MODELS[agent.role.value.upper()]
                                st.caption(f"Model: {model_used}")
                            elif model_to_use:
                                st.caption(f"Model: {model_to_use}")
                        
                        st.markdown(response)
                        
                        # Show evidence info with enhanced metadata
                        if use_rag and retrieved_docs:
                            with st.expander(f"Evidence ({len(retrieved_docs)} documents)", expanded=False):
                                for i, doc in enumerate(retrieved_docs, 1):
                                    # Enhanced source citations
                                    source = doc.metadata.get('source', 'Unknown')
                                    relevance = doc.metadata.get('relevance_score', 'N/A')
                                    confidence = doc.metadata.get('confidence', 'N/A')
                                    rank = doc.metadata.get('rank', i)
                                    
                                    # Color code confidence levels
                                    conf_color = {
                                        'High': '🟢',
                                        'Medium': '🟡',
                                        'Low': '🟠'
                                    }.get(confidence, '⚪')
                                    
                                    st.markdown(f"**Source {rank}:** {source}")
                                    st.markdown(f"{conf_color} Confidence: {confidence} | Relevance Score: {relevance}")
                                    st.caption(doc.page_content[:200] + "...")
                                    st.markdown("---")
                    
                    agent.conversation_history.append({
                        'topic': topic,
                        'response': response
                    })
                
                st.markdown("---")
                
                # Clear progress
                progress_bar.empty()
                status_text.empty()
                
                # Judge evaluation (if enabled)
                if include_judge:
                    st.markdown("---")
                    st.markdown("""
                        <div style='background-color: #fffbf0; padding: 15px; border-radius: 5px; border-left: 4px solid #ffc107;'>
                            <h3 style='color: #f57c00; margin: 0;'>Judge's Verdict</h3>
                        </div>
                    """, unsafe_allow_html=True)
                    st.markdown("")
                    
                    judge = agents[-1]  # Last agent is judge
                    
                    # Collect all arguments
                    all_arguments = []
                    for agent in agents[:-1]:  # Exclude judge
                        for entry in agent.conversation_history:
                            if entry['topic'] == topic:
                                all_arguments.append(entry['response'])
                    
                    # Get judge's context
                    judge_context = ""
                    if use_rag and retriever:
                        with st.spinner("Judge reviewing evidence..."):
                            docs = judge.retrieve_context(topic, k=context_k)
                            judge_context = "\n".join([doc.page_content[:300] for doc in docs[:3]])
                    
                    # Generate verdict
                    with st.spinner("Judge deliberating..."):
                        verdict = judge.generate_response(
                            topic=topic,
                            context=judge_context,
                            opponent_arguments=all_arguments
                        )
                    
                    # Display verdict
                    st.markdown("**Final Evaluation:**")
                    st.markdown(verdict)
                    
                    if use_llm and use_specialized:
                        st.caption(f"Judge Model: {AGENT_MODELS.get('JUDGE', 'mistral:7b')}")
                
                # Save to history
                st.session_state.debate_history.append({
                    'topic': topic,
                    'rounds': rounds,
                    'agents': len(agents),
                    'use_rag': use_rag,
                    'include_judge': include_judge,
                    'use_specialized': use_specialized if use_llm else False
                })
                
                # Completion message
                st.markdown("---")
                st.success("Debate completed")
                
                # Summary
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Rounds", rounds)
                with col2:
                    st.metric("Arguments", len(agents) * rounds)
                with col3:
                    if use_rag and retriever:
                        total_docs = sum(len(a.evidence_used) for a in agents)
                        st.metric("Evidence", total_docs)
                
                # Store agents in session for metrics tab
                st.session_state.current_agents = agents

with tab2:
    st.markdown("## Performance Metrics Dashboard")
    
    if 'current_agents' in st.session_state and st.session_state.current_agents:
        agents = st.session_state.current_agents
        
        st.markdown("### Real-time Performance Overview")
        
        # Overall metrics
        col1, col2, col3, col4 = st.columns(4)
        
        total_responses = sum(a.metrics['total_responses'] for a in agents)
        total_tokens = sum(a.metrics['total_tokens'] for a in agents)
        avg_response_time = sum(a.metrics['avg_response_time'] for a in agents if a.metrics['avg_response_time'] > 0) / max(1, len([a for a in agents if a.metrics['avg_response_time'] > 0]))
        total_evidence = sum(len(a.evidence_used) for a in agents)
        
        with col1:
            st.metric("Total Responses", total_responses)
        with col2:
            st.metric("Total Tokens", f"{total_tokens:,}")
        with col3:
            st.metric("Avg Response Time", f"{avg_response_time:.2f}s")
        with col4:
            st.metric("Evidence Retrieved", total_evidence)
        
        st.markdown("---")
        
        # Per-agent metrics
        st.markdown("### Agent Performance Breakdown")
        
        for agent in agents:
            metrics = agent.get_metrics()
            
            with st.expander(f"{agent.name} ({agent.role.value.upper()})", expanded=True):
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.markdown("**Response Metrics**")
                    st.metric("Responses Generated", metrics['total_responses'])
                    st.metric("Avg Response Time", f"{metrics['avg_response_time']:.2f}s")
                    st.metric("Peak Memory", f"{metrics['peak_memory_mb']:.2f} MB")
                
                with col2:
                    st.markdown("**Token Usage**")
                    st.metric("Total Tokens", f"{metrics['total_tokens']:,}")
                    st.metric("Avg Tokens/Response", f"{metrics['avg_tokens_per_response']:.0f}")
                    if use_llm and use_specialized:
                        model_name = AGENT_MODELS.get(agent.role.value.upper(), 'N/A')
                        st.metric("Model Used", model_name)
                
                with col3:
                    st.markdown("**Evidence Usage**")
                    st.metric("Documents Retrieved", metrics['evidence_documents_used'])
                    if metrics['evidence_documents_used'] > 0:
                        st.metric("Docs per Response", f"{metrics['evidence_documents_used'] / max(1, metrics['total_responses']):.1f}")
                
                # Response time chart
                if agent.metrics['response_times']:
                    st.markdown("**Response Time Trend**")
                    import pandas as pd
                    df = pd.DataFrame({
                        'Response #': range(1, len(agent.metrics['response_times']) + 1),
                        'Time (s)': agent.metrics['response_times']
                    })
                    st.line_chart(df.set_index('Response #'))
        
        st.markdown("---")
        
        # Knowledge Base Management
        st.markdown("### Knowledge Base Management")
        
        kb_path = os.path.join(project_root, "kb_docs")
        vector_db_path = os.path.join(project_root, "vectorstore", "faiss_index")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("**Current Knowledge Base**")
            if os.path.exists(kb_path):
                file_count = sum(len(files) for _, _, files in os.walk(kb_path))
                st.metric("Documents", file_count)
                
                # List document sources
                with st.expander("View Document Sources"):
                    for root, dirs, files in os.walk(kb_path):
                        for file in files:
                            if file.endswith(('.txt', '.pdf', '.md')):
                                rel_path = os.path.relpath(os.path.join(root, file), kb_path)
                                st.text(f"📄 {rel_path}")
            else:
                st.warning("Knowledge base folder not found")
        
        with col2:
            st.markdown("**Vector Database Status**")
            if os.path.exists(vector_db_path):
                st.success("Vector DB: Ready")
                
                # Show index info
                try:
                    import faiss
                    index_file = os.path.join(vector_db_path, "index.faiss")
                    if os.path.exists(index_file):
                        index = faiss.read_index(index_file)
                        st.metric("Vectors Indexed", index.ntotal)
                        st.metric("Dimension", index.d)
                except Exception as e:
                    st.caption(f"Could not read index details: {e}")
            else:
                st.warning("Vector DB not found")
                st.caption("Run: python build_kb.py")
        
    else:
        st.info("No active debate. Start a debate in the 'New Debate' tab to see performance metrics.")

with tab3:
    st.markdown("## 📚 Knowledge Base Management")
    
    # Import manager
    from retriever.dynamic_kb_manager import DynamicKBManager
    
    # Initialize manager
    @st.cache_resource
    def get_kb_manager():
        return DynamicKBManager(
            kb_folder="./kb_docs",
            vector_db_path="./vectorstore/faiss_index",
            auto_update_hours=24
        )
    
    manager = get_kb_manager()
    
    # Get stats
    stats = manager.get_stats()
    
    # Overview metrics
    st.markdown("### 📊 Overview")
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Total Files", stats['total_files'])
    with col2:
        st.metric("Document Chunks", stats['total_documents'])
    with col3:
        st.metric("Web Sources", stats['web_sources'])
    with col4:
        if stats['needs_update']:
            st.metric("Status", "⚠️ Update Needed")
        else:
            st.metric("Status", "✅ Up to Date")
    
    st.markdown("---")
    
    # Management actions
    st.markdown("### 🔧 Management Actions")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("#### 📁 Static Documents")
        
        # Show indexed files
        with st.expander("View Indexed Files", expanded=False):
            if stats['total_files'] > 0:
                for filepath, info in manager.metadata['indexed_files'].items():
                    st.text(f"📄 {filepath}")
                    st.caption(f"   Chunks: {info['chunks']} | Indexed: {info['indexed_at'][:10]}")
            else:
                st.info("No files indexed yet")
        
        # Scan for new files
        if st.button("🔍 Scan for New Documents"):
            with st.spinner("Scanning..."):
                new_files = manager.scan_new_documents()
                if new_files:
                    st.success(f"Found {len(new_files)} new/modified files!")
                    for file in new_files:
                        st.text(f"  • {os.path.basename(file)}")
                else:
                    st.info("No new files found")
        
        # Full rebuild
        if st.button("🔄 Full Rebuild (Static Only)"):
            with st.spinner("Rebuilding knowledge base..."):
                try:
                    manager.full_rebuild(include_web=False)
                    st.success("✅ Knowledge base rebuilt successfully!")
                    st.cache_resource.clear()
                    st.rerun()
                except Exception as e:
                    st.error(f"❌ Error: {e}")
    
    with col2:
        st.markdown("#### 🌐 Web Sources")
        
        # Show web sources
        with st.expander("View Web Sources", expanded=False):
            if stats['web_sources'] > 0:
                for source in manager.metadata['web_sources']:
                    st.text(f"🔍 {source['query']}")
                    st.caption(f"   Documents: {source['doc_count']} | Scraped: {source['scraped_at'][:10]}")
            else:
                st.info("No web sources scraped yet")
        
        # Web scraping options
        web_query = st.text_input("Search Query", placeholder="e.g., cyber insurance trends 2025")
        
        col2a, col2b = st.columns(2)
        with col2a:
            if st.button("🌐 Scrape & Add"):
                if web_query:
                    with st.spinner(f"Scraping web for: {web_query}"):
                        try:
                            manager.incremental_update(
                                include_web=True,
                                web_queries=[web_query]
                            )
                            st.success("✅ Web content added!")
                            st.cache_resource.clear()
                            st.rerun()
                        except Exception as e:
                            st.error(f"❌ Error: {e}")
                else:
                    st.warning("Please enter a search query")
        
        with col2b:
            if st.button("🗑️ Clear Web Sources"):
                manager.clear_web_sources()
                st.success("✅ Web sources cleared from metadata")
                st.info("💡 Run full rebuild to update vector store")
    
    st.markdown("---")
    
    # Advanced options
    with st.expander("⚙️ Advanced Options", expanded=False):
        st.markdown("#### Incremental Update")
        
        col1, col2 = st.columns(2)
        with col1:
            include_web_update = st.checkbox("Include web scraping", value=False)
        with col2:
            auto_queries = st.checkbox("Use default queries", value=True)
        
        if include_web_update and not auto_queries:
            custom_queries = st.text_area(
                "Custom Queries (one per line)",
                placeholder="insurance trends 2025\ncyber insurance\nclimate risk"
            )
        
        if st.button("🔄 Run Incremental Update"):
            with st.spinner("Updating knowledge base..."):
                try:
                    queries = None
                    if include_web_update:
                        if auto_queries:
                            queries = [
                                "insurance industry trends 2025",
                                "cyber insurance developments",
                                "InsurTech innovations"
                            ]
                        elif custom_queries:
                            queries = [q.strip() for q in custom_queries.split('\n') if q.strip()]
                    
                    result = manager.incremental_update(
                        include_web=include_web_update,
                        web_queries=queries
                    )
                    
                    if result:
                        st.success("✅ Knowledge base updated successfully!")
                        st.cache_resource.clear()
                        st.rerun()
                    else:
                        st.info("✅ Knowledge base is already up to date!")
                except Exception as e:
                    st.error(f"❌ Error: {e}")
        
        st.markdown("---")
        
        st.markdown("#### Database Information")
        if stats['vectorstore_exists']:
            st.success(f"📍 Location: {stats['kb_folder']}")
            st.info(f"🕐 Last Update: {stats['last_update']}")
            
            # File type breakdown
            if stats['file_types']:
                st.markdown("**File Types:**")
                for ext, count in stats['file_types'].items():
                    st.text(f"  {ext}: {count} files")
        else:
            st.warning("Vector store not initialized. Run a full rebuild.")
    
    st.markdown("---")
    
    # Document growth over time chart
    import pandas as pd
    import altair as alt

    # Create timeline from metadata
    updates = []
    for source_info in manager.metadata.get('web_sources', []):
        updates.append({
            'date': source_info['scraped_at'][:10],
            'docs': source_info['doc_count'],
            'type': 'web'
        })

    for file, info in manager.metadata.get('indexed_files', {}).items():
        updates.append({
            'date': info['indexed_at'][:10],
            'docs': info['chunks'],
            'type': 'static'
        })

    df = pd.DataFrame(updates)
    chart = alt.Chart(df).mark_line().encode(
        x='date:T',
        y='sum(docs):Q',
        color='type:N'
    ).properties(title='Knowledge Base Growth')

    st.altair_chart(chart, use_container_width=True)

with tab4:
    st.markdown("## Debate History")
    
    if st.session_state.debate_history:
        st.markdown(f"**Total:** {len(st.session_state.debate_history)} debates")
        st.markdown("---")
        
        for i, debate in enumerate(reversed(st.session_state.debate_history), 1):
            debate_num = len(st.session_state.debate_history) - i + 1
            
            with st.expander(f"Debate #{debate_num}: {debate['topic'][:60]}{'...' if len(debate['topic']) > 60 else ''}"):
                st.markdown(f"**Topic:** {debate['topic']}")
                st.markdown("")
                
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric("Rounds", debate['rounds'])
                with col2:
                    st.metric("Agents", debate['agents'])
                with col3:
                    st.metric("RAG", "Yes" if debate['use_rag'] else "No")
                with col4:
                    st.metric("Judge", "Yes" if debate['include_judge'] else "No")
        
        if st.button("Clear History"):
            st.session_state.debate_history = []
            st.rerun()
    else:
        st.info("No debates yet. Go to the New Debate tab to start.")

with tab5:
    st.markdown("## System Information")
    
    # System status
    st.markdown("### System Status")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        # Check vector DB
        vector_db_path = os.path.join(project_root, "vectorstore", "faiss_index")
        if os.path.exists(vector_db_path):
            st.success("Vector DB: Ready")
        else:
            st.warning("Vector DB: Not found")
    
    with col2:
        # Check knowledge base
        kb_path = os.path.join(project_root, "kb_docs")
        if os.path.exists(kb_path):
            file_count = sum(len(files) for _, _, files in os.walk(kb_path))
            st.info(f"Knowledge Base: {file_count} files")
        else:
            st.warning("Knowledge Base: Not found")
    
    with col3:
        # LLM status
        if use_llm:
            if use_specialized:
                st.success("LLM: Multi-Model")
            else:
                st.info("LLM: Single Model")
        else:
            st.warning("LLM: Simulated")
    
    st.markdown("---")
    
    # Detailed information
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### Model Configuration")
        
        st.markdown("**Current Assignments:**")
        
        models_df = {
            "Role": ["PRO", "CON", "JUDGE"],
            "Model": [
                AGENT_MODELS.get('PRO', 'llama3:8b'),
                AGENT_MODELS.get('CON', 'llama3:8b'),
                AGENT_MODELS.get('JUDGE', 'mistral:7b')
            ],
            "Purpose": [
                "Logic & structure",
                "Counter-arguments",
                "Evaluation"
            ]
        }
        
        st.table(models_df)
        
        st.markdown(f"Fallback: {DEFAULT_OLLAMA_MODEL}")
        
        st.markdown("### Agent Capabilities")
        st.markdown("""
        **Pro Agent:**
        - Argues in favor of the topic
        - Presents supporting evidence
        - Highlights benefits and opportunities
        
        **Con Agent:**
        - Argues against the topic
        - Presents opposing evidence
        - Identifies risks and challenges
        
        **Judge Agent (Optional):**
        - Evaluates both arguments
        - Assesses evidence quality
        - Provides balanced verdict
        """)
    
    with col2:
        st.markdown("### 🔍 Features")
        st.markdown("""
        **RAG (Retrieval-Augmented Generation):**
        - Retrieves relevant documents
        - Uses hybrid search (FAISS + BM25)
        - Evidence-based arguments
        
        **Debate Structure:**
        - Multi-round format
        - Turn-based arguments
        - Optional judge evaluation
        
        **Customization:**
        - Adjustable rounds
        - Context document count
        - Toggle RAG on/off
        """)
    
    st.markdown("---")
    
    # Setup instructions
    st.markdown("### � Getting Started")
    
    with st.expander("1️⃣ Install Ollama", expanded=False):
        st.markdown("""
        **Download & Install:**
        - Windows: https://ollama.com/download/windows
        - Mac: `brew install ollama`
        - Linux: `curl -fsSL https://ollama.com/install.sh | sh`
        """)
    
    with st.expander("Pull Required Models", expanded=False):
        st.code("""ollama pull llama3:8b
ollama pull mistral:7b
ollama pull llama3.2""", language="bash")
    
    with st.expander("Build Knowledge Base (Optional)", expanded=False):
        st.markdown("""
        To enable RAG:
        1. Add documents to kb_docs/ folder
        2. Run: python build_kb.py
        3. Enable RAG in sidebar
        """)

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center; padding: 15px;'>
    <p style='color: #666; font-size: 0.85em;'>
        Insurance Policy Debate System | Multi-Model AI Platform
    </p>
    <p style='color: #999; font-size: 0.75em;'>
        Powered by Ollama, Streamlit & LangChain
    </p>
</div>
""", unsafe_allow_html=True)
