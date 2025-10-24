"""
Quick RAG Test Script
=====================

A simple test script to verify the RAG system works with your PDF.
"""

from rag_simple import SimpleRAG

def main():
    print("=" * 70)
    print("RAG System Quick Test")
    print("=" * 70)
    print()

    # Initialize RAG with smaller, faster model for testing
    print("Initializing RAG system...")
    rag = SimpleRAG(
        data_dir="./data",
        model_name="microsoft/phi-2",  # Small but capable model
        chunk_size=500,
        chunk_overlap=50,
        k_retrieve=3
    )

    # Setup the system
    print()
    if not rag.setup():
        print("\n❌ Setup failed!")
        return

    print("\n" + "=" * 70)
    print("Testing with sample questions about CIU Student Advisory Services")
    print("=" * 70)
    print()

    # Test questions about the document
    test_questions = [
        "What is this document about?",
        "What are the main topics covered?",
        "Who is responsible for student advisory services?",
    ]

    for i, question in enumerate(test_questions, 1):
        print(f"\n{'='*70}")
        print(f"Question {i}: {question}")
        print('='*70)

        result = rag.query(question)
        print(f"\nAnswer: {result['answer']}")

        if result['sources']:
            print(f"\nBased on {len(result['sources'])} source(s)")

        print()

    print("=" * 70)
    print("Test complete!")
    print("=" * 70)
    print()
    print("Next steps:")
    print("  1. Try the web interface: python app.py")
    print("  2. Ask your own questions about the document")
    print("  3. Experiment with different parameters")
    print()

if __name__ == "__main__":
    main()
