"""
Quick test script for the RAG system
"""

from rag_flexible import FlexibleRAG

# Initialize with retrieval-only mode (works everywhere!)
print("\n" + "="*70)
print("Testing RAG System with Retrieval-Only Mode")
print("="*70)

rag = FlexibleRAG(
    backend="retrieval_only",
    data_dir="./data",
    chunk_size=500,
    chunk_overlap=50,
    k_retrieve=3
)

# Setup the system
print("\nSetting up RAG system...")
if rag.setup():
    print("\n✅ Setup successful!")

    # Test with some questions about student counseling (English)
    test_questions = [
        "What is this document about?",
        "What is the purpose of student advisory services?",
        "What are the responsibilities of an academic advisor?",
    ]

    print("\n" + "="*70)
    print("Testing with sample questions...")
    print("="*70)

    for question in test_questions:
        print(f"\n{'='*70}")
        result = rag.query(question)
        print()

    print("\n" + "="*70)
    print("✅ Test completed successfully!")
    print("="*70)
else:
    print("\n❌ Setup failed!")
