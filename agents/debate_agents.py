"""
Insurance Policy Debate Agents with different perspectives and roles.
Each agent analyzes insurance policies and trends from a unique standpoint.
"""
import os
import sys
from typing import List, Dict, Any, Optional
from enum import Enum

# Add project root to path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# Import config
from config import AGENT_MODELS, DEFAULT_OLLAMA_MODEL

# LLM imports
try:
    from langchain_ollama import OllamaLLM
    OLLAMA_AVAILABLE = True
except ImportError:
    OLLAMA_AVAILABLE = False
    print("⚠️ Ollama not available. Install with: pip install ollama langchain-ollama")


class AgentRole(Enum):
    """Define different agent roles for debate."""
    PRO = "pro"
    CON = "con"
    JUDGE = "judge"


class DebateAgent:
    """Base class for insurance debate agents."""
    
    def __init__(
        self,
        name: str,
        role: AgentRole,
        personality: str,
        retriever=None,
        llm=None
    ):
        """
        Initialize debate agent.
        
        Args:
            name: Agent name
            role: Agent role (from AgentRole enum)
            personality: Agent's personality/perspective description
            retriever: Document retriever for RAG
            llm: Language model for generating responses
        """
        self.name = name
        self.role = role
        self.personality = personality
        self.retriever = retriever
        self.llm = llm
        self.conversation_history = []
        self.evidence_used = []
    
    def get_system_prompt(self) -> str:
        """Generate system prompt based on agent's role."""
        base_prompt = f"""You are {self.name}, an AI agent participating in an insurance policy debate.
        
Role: {self.role.value.upper()}
Personality: {self.personality}

Your responsibilities:
- Analyze insurance policies, trends, and industry developments
- Provide evidence-based arguments from your perspective
- Engage constructively with other agents' viewpoints
- Cite sources when making claims
- Stay focused on insurance industry topics

"""
        
        role_specific = {
            AgentRole.PRO: """As the PRO agent, you focus on:
- Supporting arguments and positive aspects of the topic
- Evidence-based reasoning for benefits and advantages
- Constructive solutions and opportunities
- Data and research supporting the affirmative position
- Addressing counterarguments with evidence""",
            
            AgentRole.CON: """As the CON agent, you focus on:
- Critical analysis and opposing arguments
- Evidence-based reasoning for risks and disadvantages
- Identifying potential problems and challenges
- Data and research supporting the negative position
- Addressing pro arguments with counterevidence""",
            
            AgentRole.JUDGE: """As the JUDGE agent, you focus on:
- Objective evaluation of both PRO and CON arguments
- Assessing the strength and quality of evidence presented
- Identifying logical fallacies and weak reasoning
- Weighing competing claims fairly
- Providing a balanced, reasoned verdict"""
        }
        
        return base_prompt + role_specific.get(self.role, "")
    
    def retrieve_context(self, query: str, k: int = 3) -> List[Any]:
        """
        Retrieve relevant documents for the query.
        
        Args:
            query: Search query
            k: Number of documents to retrieve
        
        Returns:
            List of relevant documents
        """
        if not self.retriever:
            print(f"⚠️ {self.name}: No retriever available")
            return []
        
        try:
            docs = self.retriever.search(query, k=k)
            self.evidence_used.extend(docs)
            return docs
        except Exception as e:
            print(f"❌ {self.name}: Retrieval error: {e}")
            return []
    
    def format_context(self, docs: List[Any]) -> str:
        """Format retrieved documents as context string."""
        if not docs:
            return "No relevant documents found."
        
        context_parts = []
        for i, doc in enumerate(docs, 1):
            source = doc.metadata.get('source', 'Unknown')
            content = doc.page_content[:500]  # Limit length
            context_parts.append(f"[Source {i}: {source}]\n{content}\n")
        
        return "\n---\n".join(context_parts)
    
    def generate_response(self, topic: str, context: str = "", opponent_arguments: Optional[List[str]] = None) -> str:
        """
        Generate agent's response to the debate topic.
        
        Args:
            topic: Debate topic
            context: Retrieved context/evidence
            opponent_arguments: Previous arguments from other agents
        
        Returns:
            Agent's response (simulated if no LLM)
        """
        # If LLM is available, use it
        if self.llm:
            return self._generate_with_llm(topic, context, opponent_arguments or [])
        
        # Otherwise, return simulated response based on role
        return self._generate_simulated_response(topic, context, opponent_arguments or [])
    
    def _generate_simulated_response(self, topic: str, context: str, opponent_arguments: List[str]) -> str:
        """Generate a simulated response without LLM."""
        role_templates = {
            AgentRole.PRO: f"""**PRO POSITION on '{topic}':**

I argue in FAVOR of this topic. Based on the evidence available, there are compelling reasons to support this position.

{f"**Evidence:** {context[:400]}..." if context else ""}

**Key Arguments:**
1. The benefits clearly outweigh potential drawbacks
2. Data supports positive outcomes
3. This represents progress and opportunity

{f"**Response to Opposition:** {opponent_arguments[0][:200] if opponent_arguments else 'I look forward to hearing counterarguments.'}" if opponent_arguments else ""}

The affirmative position is well-supported by evidence and reasoning.""",
            
            AgentRole.CON: f"""**CON POSITION on '{topic}':**

I argue AGAINST this topic. Critical analysis reveals significant concerns that must be addressed.

{f"**Evidence:** {context[:400]}..." if context else ""}

**Key Arguments:**
1. The risks and drawbacks are substantial
2. Evidence shows potential negative consequences
3. Alternative approaches may be more appropriate

{f"**Rebuttal:** {opponent_arguments[0][:200] if opponent_arguments else 'I will address pro arguments as they are presented.'}" if opponent_arguments else ""}

The negative position is warranted based on careful evaluation.""",
            
            AgentRole.JUDGE: f"""**JUDGE EVALUATION on '{topic}':**

After reviewing both PRO and CON arguments, I provide the following assessment:

**PRO Arguments Analysis:**
- Strengths: [Evaluation of supporting evidence]
- Weaknesses: [Identification of gaps or weak points]

**CON Arguments Analysis:**
- Strengths: [Evaluation of opposing evidence]
- Weaknesses: [Identification of gaps or weak points]

{f"**Context Considered:** {context[:300]}..." if context else ""}

**Preliminary Assessment:**
Both sides present valid points requiring careful weighing of evidence and logical consistency.

**Verdict:** [To be determined based on full debate]"""
        }
        
        response = role_templates.get(self.role, f"My perspective on {topic}: {context[:200] if context else 'Analyzing the situation...'}")
        
        return response
    
    def _generate_with_llm(self, topic: str, context: str, opponent_arguments: List[str]) -> str:
        """Generate response using LLM (Ollama)."""
        if not self.llm:
            print("⚠️ No LLM configured, using simulated response")
            return self._generate_simulated_response(topic, context, opponent_arguments or [])
        
        system_prompt = self.get_system_prompt()
        
        # Build user prompt
        user_prompt = f"""Topic: {topic}

Context/Evidence:
{context[:2000] if context else "No specific context provided."}

{"Previous opponent arguments: " + chr(10).join(opponent_arguments[:3]) if opponent_arguments else "This is the opening statement."}

Provide your {self.role.value} position on this topic. Be clear, evidence-based, and persuasive."""
        
        try:
            # Combine system and user prompts for Ollama
            full_prompt = f"{system_prompt}\n\n{user_prompt}"
            
            # Invoke LLM
            response = self.llm.invoke(full_prompt)
            
            return response
            
        except Exception as e:
            print(f"❌ LLM error: {e}")
            print("⚠️ Falling back to simulated response")
            return self._generate_simulated_response(topic, context, opponent_arguments or [])
    
    def summarize_position(self) -> Dict[str, Any]:
        """Summarize agent's position and evidence used."""
        return {
            'name': self.name,
            'role': self.role.value,
            'arguments_count': len(self.conversation_history),
            'evidence_count': len(self.evidence_used),
            'sources': list(set([doc.metadata.get('source', 'Unknown') for doc in self.evidence_used]))
        }


# === Predefined Agent Configurations ===

def create_llm(model: str = "llama3.2", role: Optional[str] = None):
    """
    Create Ollama LLM instance with role-specific model selection.
    
    Args:
        model: Ollama model name (llama3.2, llama3:8b, mistral:7b, etc.)
        role: Agent role name (e.g., "PRO", "CON", "JUDGE") - will use AGENT_MODELS if provided
    
    Returns:
        OllamaLLM instance or None if unavailable
    """
    if not OLLAMA_AVAILABLE:
        print("⚠️ Ollama not available. Install with: pip install ollama langchain-ollama")
        return None
    
    # Use role-specific model if role provided and found in config
    selected_model = model
    if role and role in AGENT_MODELS:
        selected_model = AGENT_MODELS[role]
        print(f"🎯 Using specialized model for {role}: {selected_model}")
    else:
        print(f"🤖 Using model: {selected_model}")
    
    try:
        llm = OllamaLLM(
            model=selected_model,
            temperature=0.7,
        )
        # Test if Ollama is running
        try:
            llm.invoke("test")
            print(f"✅ Ollama LLM loaded: {selected_model}")
            return llm
        except Exception as e:
            print(f"⚠️ Ollama not running. Start it with: ollama serve")
            print(f"⚠️ Or pull model: ollama pull {selected_model}")
            
            # Try fallback to default model
            if selected_model != DEFAULT_OLLAMA_MODEL:
                print(f"🔄 Trying fallback model: {DEFAULT_OLLAMA_MODEL}")
                try:
                    fallback_llm = OllamaLLM(model=DEFAULT_OLLAMA_MODEL, temperature=0.7)
                    fallback_llm.invoke("test")
                    print(f"✅ Using fallback model: {DEFAULT_OLLAMA_MODEL}")
                    return fallback_llm
                except:
                    pass
            
            print("⚠️ Continuing with simulated responses...")
            return None
    except Exception as e:
        print(f"❌ Error creating LLM: {e}")
        return None


def create_debate_agents(
    retriever=None, 
    llm=None, 
    use_ollama: bool = True, 
    model: Optional[str] = None,
    use_specialized_models: bool = True
) -> List[DebateAgent]:
    """
    Create PRO and CON debate agents with specialized models.
    
    Args:
        retriever: Document retriever for RAG
        llm: Language model for generation (overrides other LLM settings)
        use_ollama: Whether to attempt using Ollama (default: True)
        model: Override model name for all agents (e.g., "llama3.2") - if None, uses role-specific models
        use_specialized_models: Use role-specific models from AGENT_MODELS config (default: True)
    
    Returns:
        List of DebateAgent instances [PRO, CON]
    """
    # Create LLMs for each agent
    pro_llm = None
    con_llm = None
    
    if llm is not None:
        # Use provided LLM for all agents
        pro_llm = llm
        con_llm = llm
    elif use_ollama:
        if model:
            # Use same model for all agents
            pro_llm = create_llm(model)
            con_llm = create_llm(model)
        elif use_specialized_models:
            # Use role-specific models
            pro_llm = create_llm(role="PRO")
            con_llm = create_llm(role="CON")
        else:
            # Use default model
            default_llm = create_llm(DEFAULT_OLLAMA_MODEL)
            pro_llm = default_llm
            con_llm = default_llm
    
    agents = [
        DebateAgent(
            name="Pro Agent",
            role=AgentRole.PRO,
            personality="Supports the topic with evidence-based arguments, highlights benefits and opportunities",
            retriever=retriever,
            llm=pro_llm
        ),
        DebateAgent(
            name="Con Agent",
            role=AgentRole.CON,
            personality="Opposes the topic with critical analysis, identifies risks and challenges",
            retriever=retriever,
            llm=con_llm
        )
    ]
    
    return agents


def create_debate_agents_with_judge(
    retriever=None, 
    llm=None, 
    use_ollama: bool = True, 
    model: Optional[str] = None,
    use_specialized_models: bool = True
) -> List[DebateAgent]:
    """
    Create PRO, CON, and JUDGE debate agents with specialized models.
    
    Args:
        retriever: Document retriever for RAG
        llm: Language model for generation (overrides other LLM settings)
        use_ollama: Whether to attempt using Ollama (default: True)
        model: Override model name for all agents - if None, uses role-specific models
        use_specialized_models: Use role-specific models from AGENT_MODELS config (default: True)
    
    Returns:
        List of DebateAgent instances [PRO, CON, JUDGE]
    """
    # Create Pro and Con agents
    agents = create_debate_agents(retriever, llm, use_ollama, model, use_specialized_models)
    
    # Create Judge agent with specialized model
    judge_llm = None
    if llm is not None:
        judge_llm = llm
    elif use_ollama:
        if model:
            judge_llm = create_llm(model)
        elif use_specialized_models:
            judge_llm = create_llm(role="JUDGE")
        else:
            judge_llm = create_llm(DEFAULT_OLLAMA_MODEL)
    
    judge = DebateAgent(
        name="Judge Agent",
        role=AgentRole.JUDGE,
        personality="Objectively evaluates both sides, weighs evidence, provides balanced assessment",
        retriever=retriever,
        llm=judge_llm
    )
    
    agents.append(judge)
    return agents


# Backward compatibility alias
def create_insurance_debate_agents(retriever=None, llm=None) -> List[DebateAgent]:
    """Legacy function - creates PRO/CON agents."""
    return create_debate_agents(retriever, llm)


# === Test Function ===
def test_debate_agents():
    """Test debate agents without retriever."""
    print("\n" + "="*60)
    print("Testing Pro/Con Debate Agents")
    print("="*60 + "\n")
    
    agents = create_debate_agents()
    
    topic = "AI-powered underwriting should be mandatory in insurance"
    
    print(f"Debate Topic: {topic}\n")
    print("="*60 + "\n")
    
    for agent in agents:
        print(f"🗣️  {agent.name} ({agent.role.value.upper()})")
        print("-" * 60)
        
        response = agent.generate_response(topic)
        print(response)
        print("\n" + "="*60 + "\n")
        
        agent.conversation_history.append({
            'topic': topic,
            'response': response
        })


if __name__ == "__main__":
    test_debate_agents()
