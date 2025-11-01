"""
Debate Orchestrator - Manages multi-agent insurance policy debates.
Coordinates agent interactions, manages debate flow, and synthesizes conclusions.
"""
import os
import sys
from typing import List, Dict, Any, Optional
from datetime import datetime

# Add project root to path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from agents.debate_agents import DebateAgent, create_insurance_debate_agents


class DebateOrchestrator:
    """Orchestrates multi-agent debates on insurance topics."""
    
    def __init__(self, agents: List[DebateAgent], retriever=None):
        """
        Initialize debate orchestrator.
        
        Args:
            agents: List of debate agents
            retriever: Document retriever for RAG
        """
        self.agents = agents
        self.retriever = retriever
        self.debate_history = []
        self.current_topic = None
    
    def conduct_debate(
        self,
        topic: str,
        rounds: int = 2,
        retrieve_context: bool = True,
        context_k: int = 3
    ) -> Dict[str, Any]:
        """
        Conduct a structured debate on a topic.
        
        Args:
            topic: Debate topic
            rounds: Number of debate rounds
            retrieve_context: Whether to retrieve contextual evidence
            context_k: Number of documents to retrieve per agent
        
        Returns:
            Dictionary containing debate transcript and analysis
        """
        self.current_topic = topic
        debate_record = {
            'topic': topic,
            'timestamp': datetime.now().isoformat(),
            'rounds': rounds,
            'transcript': [],
            'participants': [agent.name for agent in self.agents],
            'summary': {}
        }
        
        print("\n" + "="*80)
        print(f"🎯 DEBATE TOPIC: {topic}")
        print("="*80 + "\n")
        
        # Retrieve shared context if enabled
        shared_context = ""
        if retrieve_context and self.retriever:
            print("📚 Retrieving relevant context...\n")
            context_docs = self.retriever.search(topic, k=context_k * 2)
            shared_context = self._format_context_docs(context_docs)
            print(f"✅ Retrieved {len(context_docs)} relevant documents\n")
        
        # Conduct debate rounds
        for round_num in range(1, rounds + 1):
            print(f"\n{'='*80}")
            print(f"🔄 ROUND {round_num} of {rounds}")
            print(f"{'='*80}\n")
            
            round_arguments = []
            
            # Each agent presents their argument
            for agent in self.agents:
                print(f"\n🗣️  {agent.name} ({agent.role.value.upper()})")
                print("-" * 80)
                
                # Get agent-specific context if needed
                agent_context = shared_context
                if retrieve_context and agent.retriever:
                    agent_docs = agent.retrieve_context(f"{topic} {agent.role.value}", k=context_k)
                    agent_context += "\n" + self._format_context_docs(agent_docs)
                
                # Generate response considering previous arguments
                previous_args = round_arguments if round_num > 1 else []
                response = agent.generate_response(
                    topic=topic,
                    context=agent_context,
                    opponent_arguments=previous_args
                )
                
                print(response)
                print("\n")
                
                # Record argument
                argument = {
                    'round': round_num,
                    'agent': agent.name,
                    'role': agent.role.value,
                    'response': response,
                    'timestamp': datetime.now().isoformat()
                }
                
                round_arguments.append(response)
                debate_record['transcript'].append(argument)
                agent.conversation_history.append(argument)
            
            print(f"\n{'='*80}")
            print(f"✅ Round {round_num} completed")
            print(f"{'='*80}\n")
        
        # Generate debate summary
        debate_record['summary'] = self._generate_summary(debate_record)
        self.debate_history.append(debate_record)
        
        return debate_record
    
    def _format_context_docs(self, docs: List[Any]) -> str:
        """Format documents into context string."""
        if not docs:
            return ""
        
        parts = []
        for doc in docs:
            source = doc.metadata.get('source', 'Unknown')
            content = doc.page_content[:400]
            parts.append(f"[{source}] {content}")
        
        return "\n\n".join(parts)
    
    def _generate_summary(self, debate_record: Dict[str, Any]) -> Dict[str, Any]:
        """
        Generate summary of the debate.
        
        Args:
            debate_record: Debate record dictionary
        
        Returns:
            Summary dictionary
        """
        print("\n" + "="*80)
        print("📊 DEBATE SUMMARY")
        print("="*80 + "\n")
        
        summary = {
            'topic': debate_record['topic'],
            'total_arguments': len(debate_record['transcript']),
            'participant_count': len(debate_record['participants']),
            'perspectives': {}
        }
        
        # Summarize each agent's position
        for agent in self.agents:
            position = agent.summarize_position()
            summary['perspectives'][agent.name] = position
            
            print(f"👤 {agent.name} ({agent.role.value})")
            print(f"   Arguments: {position['arguments_count']}")
            print(f"   Evidence used: {position['evidence_count']}")
            if position['sources']:
                print(f"   Sources: {', '.join(position['sources'][:3])}")
            print()
        
        # Key themes (simplified - could use NLP for better analysis)
        summary['key_themes'] = self._extract_themes(debate_record)
        
        print(f"\n🔑 Key Themes Discussed:")
        for theme in summary['key_themes']:
            print(f"   • {theme}")
        
        print("\n" + "="*80 + "\n")
        
        return summary
    
    def _extract_themes(self, debate_record: Dict[str, Any]) -> List[str]:
        """Extract key themes from debate (simplified version)."""
        # This is a simplified version - in production, would use NLP
        theme_keywords = {
            'Technology': ['AI', 'technology', 'digital', 'innovation', 'automation'],
            'Risk Management': ['risk', 'underwriting', 'assessment', 'evaluation'],
            'Consumer Impact': ['consumer', 'customer', 'policyholder', 'affordability'],
            'Regulation': ['regulation', 'compliance', 'regulatory', 'oversight'],
            'Market Trends': ['market', 'trend', 'industry', 'growth', 'competition'],
            'Data & Privacy': ['data', 'privacy', 'security', 'protection']
        }
        
        # Count keyword occurrences
        theme_scores = {theme: 0 for theme in theme_keywords}
        
        for arg in debate_record['transcript']:
            response_lower = arg['response'].lower()
            for theme, keywords in theme_keywords.items():
                for keyword in keywords:
                    if keyword.lower() in response_lower:
                        theme_scores[theme] += 1
        
        # Return themes with non-zero scores, sorted
        themes = [theme for theme, score in sorted(theme_scores.items(), key=lambda x: x[1], reverse=True) if score > 0]
        return themes[:5]  # Top 5 themes
    
    def facilitate_qa_round(self, questions: List[str]) -> Dict[str, Any]:
        """
        Facilitate Q&A round where agents answer specific questions.
        
        Args:
            questions: List of questions to ask agents
        
        Returns:
            Q&A record
        """
        if not self.current_topic:
            print("⚠️ No active debate topic. Start a debate first.")
            return {}
        
        print("\n" + "="*80)
        print("❓ Q&A ROUND")
        print("="*80 + "\n")
        
        qa_record = {
            'topic': self.current_topic,
            'timestamp': datetime.now().isoformat(),
            'qa_pairs': []
        }
        
        for question in questions:
            print(f"\n📌 Question: {question}\n")
            print("-" * 80 + "\n")
            
            responses = []
            for agent in self.agents:
                # Each agent answers the question
                context = ""
                if agent.retriever:
                    docs = agent.retrieve_context(question, k=2)
                    context = self._format_context_docs(docs)
                
                response = agent.generate_response(
                    topic=question,
                    context=context
                )
                
                print(f"🗣️  {agent.name}: {response[:200]}...\n")
                
                responses.append({
                    'agent': agent.name,
                    'role': agent.role.value,
                    'answer': response
                })
            
            qa_record['qa_pairs'].append({
                'question': question,
                'responses': responses
            })
        
        return qa_record
    
    def generate_consensus_report(self) -> str:
        """
        Generate a consensus report from the debate.
        
        Returns:
            Formatted consensus report
        """
        if not self.debate_history:
            return "No debates conducted yet."
        
        latest_debate = self.debate_history[-1]
        
        report = f"""
{'='*80}
INSURANCE POLICY DEBATE - CONSENSUS REPORT
{'='*80}

Topic: {latest_debate['topic']}
Date: {latest_debate['timestamp']}
Participants: {', '.join(latest_debate['participants'])}
Total Rounds: {latest_debate['rounds']}

{'='*80}
KEY THEMES
{'='*80}

"""
        for i, theme in enumerate(latest_debate['summary']['key_themes'], 1):
            report += f"{i}. {theme}\n"
        
        report += f"""
{'='*80}
PERSPECTIVES SUMMARY
{'='*80}

"""
        for agent in self.agents:
            position = latest_debate['summary']['perspectives'].get(agent.name, {})
            report += f"""
{agent.name} ({agent.role.value.upper()})
- Arguments presented: {position.get('arguments_count', 0)}
- Evidence cited: {position.get('evidence_count', 0)} sources
"""
        
        report += f"""
{'='*80}
CONCLUSION
{'='*80}

This debate has explored multiple perspectives on {latest_debate['topic']}.
Each viewpoint contributes valuable insights:

- Analytical rigor ensures data-driven decisions
- Advocacy highlights opportunities and benefits  
- Skepticism identifies risks and challenges
- Regulatory oversight protects consumers and market stability
- Innovation drives industry evolution
- Consumer focus ensures practical, user-centric solutions

A balanced approach considering all perspectives is recommended for 
sustainable insurance industry development.

{'='*80}
"""
        
        return report


# === Convenience Functions ===

def quick_debate(topic: str, retriever=None, rounds: int = 2) -> Dict[str, Any]:
    """
    Quickly set up and run a debate.
    
    Args:
        topic: Debate topic
        retriever: Optional retriever for RAG
        rounds: Number of debate rounds
    
    Returns:
        Debate record
    """
    agents = create_insurance_debate_agents(retriever=retriever)
    orchestrator = DebateOrchestrator(agents, retriever=retriever)
    return orchestrator.conduct_debate(topic, rounds=rounds, retrieve_context=(retriever is not None))


# === Test Function ===
def test_orchestrator():
    """Test the debate orchestrator."""
    print("\n" + "="*80)
    print("Testing Debate Orchestrator")
    print("="*80 + "\n")
    
    # Create debate without retriever for testing
    debate_result = quick_debate(
        topic="The impact of climate change on insurance underwriting",
        rounds=1
    )
    
    # Create orchestrator for additional tests
    agents = create_insurance_debate_agents()
    orchestrator = DebateOrchestrator(agents)
    
    # Q&A round
    orchestrator.current_topic = debate_result['topic']
    orchestrator.debate_history.append(debate_result)
    
    qa_result = orchestrator.facilitate_qa_round([
        "How should insurers balance profitability with climate risk coverage?"
    ])
    
    # Generate report
    print(orchestrator.generate_consensus_report())


if __name__ == "__main__":
    test_orchestrator()
