"""
Test script for RAG system with LLM backend
============================================

This tests the system with a local LLM model to generate answers.
"""

from rag_flexible import FlexibleRAG

# Initialize with TinyLlama (smallest local model)
print("\n" + "="*70)
print("Testing RAG System with TinyLlama Model")
print("="*70)
print("\n⚠️  First run will download the model (~2.2GB)")
print("This may take 5-10 minutes depending on your internet speed.")
print("Subsequent runs will be much faster!\n")

input("Press Enter to continue...")

rag = FlexibleRAG(
    backend="tinyllama",  # Small local model
    data_dir="./data",
    chunk_size=500,
    chunk_overlap=50,
    k_retrieve=3
)

# Setup the system
print("\n🔧 Setting up RAG system...")
if rag.setup():
    print("\n✅ Setup successful!")

    # Test with a question
    test_questions = [
        "What is the purpose of student advisory services?",
    ]

    print("\n" + "="*70)
    print("Testing with LLM generation...")
    print("="*70)

    for question in test_questions:
        print(f"\n{'='*70}")
        print(f"❓ Question: {question}")
        print('='*70)

        result = rag.query(question)
        print()

    print("\n" + "="*70)
    print("✅ Test completed successfully!")
    print("="*70)
else:
    print("\n❌ Setup failed!")
