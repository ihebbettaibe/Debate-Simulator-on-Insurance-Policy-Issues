"""
Quick setup script for the Insurance Debate System
Installs required packages and checks system status
"""
import subprocess
import sys
import os

def run_command(cmd, description):
    """Run a command and print status."""
    print(f"\n{'='*60}")
    print(f"📦 {description}")
    print('='*60)
    try:
        result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
        if result.returncode == 0:
            print(f"✅ Success!")
            return True
        else:
            print(f"⚠️ Warning: {result.stderr[:200]}")
            return False
    except Exception as e:
        print(f"❌ Error: {e}")
        return False

def check_file_exists(path, name):
    """Check if a file/directory exists."""
    exists = os.path.exists(path)
    status = "✅" if exists else "❌"
    print(f"{status} {name}: {path}")
    return exists

def main():
    print("\n" + "="*60)
    print("🚀 Insurance Debate System - Setup")
    print("="*60)
    
    # Check Python version
    print(f"\n🐍 Python version: {sys.version}")
    
    # Install core requirements
    print("\n" + "="*60)
    print("Installing Core Requirements")
    print("="*60)
    
    core_packages = [
        "langchain",
        "langchain-community",
        "langchain-huggingface",
        "faiss-cpu",
        "rank-bm25",
        "sentence-transformers",
        "duckduckgo-search",
        "beautifulsoup4",
        "PyPDF2",
        "requests",
        "streamlit"
    ]
    
    print("\nInstalling packages:")
    for pkg in core_packages:
        print(f"  • {pkg}")
    
    install_cmd = f"pip install {' '.join(core_packages)}"
    run_command(install_cmd, "Installing packages")
    
    # Check system status
    print("\n" + "="*60)
    print("📊 System Status Check")
    print("="*60 + "\n")
    
    project_root = os.path.dirname(os.path.abspath(__file__))
    
    # Check critical files
    print("Critical Files:")
    check_file_exists(os.path.join(project_root, "agents", "debate_agents.py"), "Debate Agents")
    check_file_exists(os.path.join(project_root, "agents", "orchestrator.py"), "Orchestrator")
    check_file_exists(os.path.join(project_root, "retriever", "hybrid_retriever.py"), "Hybrid Retriever")
    check_file_exists(os.path.join(project_root, "app.py"), "Streamlit UI")
    check_file_exists(os.path.join(project_root, "main.py"), "Main Application")
    
    print("\nKnowledge Base:")
    kb_path = os.path.join(project_root, "kb_docs")
    kb_exists = check_file_exists(kb_path, "KB Documents Folder")
    
    if kb_exists:
        file_count = sum(len(files) for _, _, files in os.walk(kb_path))
        print(f"   📁 {file_count} files in knowledge base")
    
    print("\nVector Database:")
    vector_db_path = os.path.join(project_root, "vectorstore", "faiss_index")
    vdb_exists = check_file_exists(vector_db_path, "FAISS Index")
    
    if not vdb_exists:
        print("   ℹ️  Run 'python build_kb.py' to create vector database")
    
    # Next steps
    print("\n" + "="*60)
    print("✨ Setup Complete!")
    print("="*60)
    
    print("\n📝 Next Steps:\n")
    
    if not vdb_exists:
        print("1️⃣  Build Vector Database:")
        print("   python build_kb.py")
        print()
    
    print("2️⃣  Test the System:")
    print("   # Quick test without RAG")
    print("   python main.py --mode sample --no-retriever")
    print()
    
    print("3️⃣  Run Streamlit UI (Recommended):")
    print("   streamlit run app.py")
    print()
    
    print("4️⃣  Or use CLI:")
    print("   python main.py --mode interactive")
    print()
    
    print("📚 Documentation:")
    print("   • README.md - Full documentation")
    print("   • PRO_CON_GUIDE.md - Pro/Con system guide")
    print("   • REFACTORING_SUMMARY.md - What changed")
    
    print("\n" + "="*60)
    print("🎉 Ready to debate!")
    print("="*60 + "\n")

if __name__ == "__main__":
    main()
