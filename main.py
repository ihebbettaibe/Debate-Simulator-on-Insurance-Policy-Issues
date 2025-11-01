"""
Main entry point for the Insurance Policy Debate System.
Demonstrates full system integration with RAG-enhanced multi-agent debates.
"""
import os
import sys

# Add project root to path
project_root = os.path.abspath(os.path.dirname(__file__))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from retriever.hybrid_retriever import HybridRetriever
from retriever.static_loader import StaticDocumentLoader
from retriever.dynamic_scraper import DynamicWebScraper, RealtimeRetriever
from agents.debate_agents import create_debate_agents, create_debate_agents_with_judge
from agents.orchestrator import DebateOrchestrator, quick_debate


def setup_system(use_retriever: bool = True):
    """
    Set up the complete debate system.
    
    Args:
        use_retriever: Whether to use RAG retrieval
    
    Returns:
        Tuple of (agents, orchestrator, retriever)
    """
    print("\n" + "="*80)
    print("🚀 Initializing Insurance Policy Debate System")
    print("="*80 + "\n")
    
    retriever = None
    
    if use_retriever:
        try:
            # Set up hybrid retriever
            vector_db_path = os.path.join(project_root, "vectorstore", "faiss_index")
            
            if os.path.exists(vector_db_path):
                print("📚 Loading hybrid retriever with FAISS + BM25...")
                retriever = HybridRetriever(vector_db_path, alpha=0.5)
                print("✅ Retriever loaded successfully\n")
            else:
                print(f"⚠️ Vector database not found at {vector_db_path}")
                print("💡 Run build_kb.py first to create the knowledge base\n")
        except Exception as e:
            print(f"❌ Error loading retriever: {e}")
            print("⚠️ Continuing without retriever\n")
    
    # Create debate agents
    print("👥 Creating debate agents...")
    agents = create_debate_agents(retriever=retriever)
    print(f"✅ Created {len(agents)} debate agents:\n")
    
    for agent in agents:
        print(f"   • {agent.name} - {agent.role.value.upper()}")
    
    # Create orchestrator
    print("\n🎭 Initializing debate orchestrator...")
    orchestrator = DebateOrchestrator(agents, retriever=retriever)
    print("✅ System ready!\n")
    
    return agents, orchestrator, retriever


def run_sample_debate():
    """Run a sample debate to demonstrate the system."""
    print("\n" + "="*80)
    print("🎯 Running Sample Debate")
    print("="*80 + "\n")
    
    # Set up system
    agents, orchestrator, retriever = setup_system(use_retriever=True)
    
    # Sample debate topics (Pro vs Con format)
    topics = [
        "AI-powered underwriting should be mandatory in insurance",
        "Climate change insurance should be government-subsidized",
        "Parametric insurance is superior to traditional policies",
        "Insurance companies should be allowed to use genetic data"
    ]
    
    print("Available debate topics:")
    for i, topic in enumerate(topics, 1):
        print(f"   {i}. {topic}")
    
    # Run debate on first topic
    selected_topic = topics[0]
    print(f"\n🎯 Selected topic: {selected_topic}\n")
    
    # Conduct debate
    debate_result = orchestrator.conduct_debate(
        topic=selected_topic,
        rounds=2,
        retrieve_context=(retriever is not None),
        context_k=3
    )
    
    # Q&A round
    print("\n" + "="*80)
    print("💬 Starting Q&A Round")
    print("="*80 + "\n")
    
    follow_up_questions = [
        "What are the key regulatory concerns?",
        "How will this affect insurance premiums?"
    ]
    
    qa_result = orchestrator.facilitate_qa_round(follow_up_questions)
    
    # Generate consensus report
    report = orchestrator.generate_consensus_report()
    print(report)
    
    return debate_result, qa_result


def interactive_mode():
    """Run interactive debate mode."""
    print("\n" + "="*80)
    print("🎮 Interactive Debate Mode")
    print("="*80 + "\n")
    
    agents, orchestrator, retriever = setup_system(use_retriever=True)
    
    while True:
        print("\n" + "="*80)
        print("Options:")
        print("  1. Start new debate")
        print("  2. Ask follow-up questions")
        print("  3. View debate history")
        print("  4. Generate consensus report")
        print("  5. Exit")
        print("="*80)
        
        try:
            choice = input("\nEnter your choice (1-5): ").strip()
            
            if choice == "1":
                topic = input("\nEnter debate topic: ").strip()
                if topic:
                    rounds = input("Number of rounds (default 2): ").strip() or "2"
                    try:
                        rounds = int(rounds)
                        orchestrator.conduct_debate(
                            topic=topic,
                            rounds=rounds,
                            retrieve_context=(retriever is not None)
                        )
                    except ValueError:
                        print("Invalid number of rounds")
            
            elif choice == "2":
                if not orchestrator.current_topic:
                    print("\n⚠️ No active debate. Start a debate first!")
                else:
                    question = input("\nEnter your question: ").strip()
                    if question:
                        orchestrator.facilitate_qa_round([question])
            
            elif choice == "3":
                print(f"\n📚 Debate History: {len(orchestrator.debate_history)} debates")
                for i, debate in enumerate(orchestrator.debate_history, 1):
                    print(f"   {i}. {debate['topic']} ({debate['timestamp']})")
            
            elif choice == "4":
                report = orchestrator.generate_consensus_report()
                print(report)
            
            elif choice == "5":
                print("\n👋 Goodbye!")
                break
            
            else:
                print("\n⚠️ Invalid choice")
        
        except KeyboardInterrupt:
            print("\n\n👋 Goodbye!")
            break
        except Exception as e:
            print(f"\n❌ Error: {e}")


def main():
    """Main entry point."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Insurance Policy Debate System")
    parser.add_argument(
        '--mode',
        choices=['sample', 'interactive', 'custom'],
        default='sample',
        help='Run mode: sample (demo), interactive (CLI), or custom'
    )
    parser.add_argument(
        '--topic',
        type=str,
        help='Debate topic (for custom mode)'
    )
    parser.add_argument(
        '--rounds',
        type=int,
        default=2,
        help='Number of debate rounds (for custom mode)'
    )
    parser.add_argument(
        '--no-retriever',
        action='store_true',
        help='Run without RAG retrieval'
    )
    
    args = parser.parse_args()
    
    if args.mode == 'sample':
        run_sample_debate()
    
    elif args.mode == 'interactive':
        interactive_mode()
    
    elif args.mode == 'custom':
        if not args.topic:
            print("❌ Error: --topic required for custom mode")
            return
        
        agents, orchestrator, retriever = setup_system(use_retriever=not args.no_retriever)
        
        debate_result = orchestrator.conduct_debate(
            topic=args.topic,
            rounds=args.rounds,
            retrieve_context=(retriever is not None)
        )
        
        report = orchestrator.generate_consensus_report()
        print(report)


if __name__ == "__main__":
    main()
