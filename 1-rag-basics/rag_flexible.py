"""
Flexible RAG (Retrieval-Augmented Generation) System
====================================================

This module implements a RAG system with multiple backend options,
making it accessible for students with different hardware configurations.

Supports:
- Retrieval-only mode (no LLM needed)
- Local lightweight models (TinyLlama)
- Local standard models (Phi-2, Phi-3)
- API-based models (Groq, OpenAI, Anthropic)
"""

import os
import json
from typing import List, Dict, Any, Optional
from pathlib import Path

# Document loading and processing
from langchain_community.document_loaders import (
    TextLoader,
    PyPDFLoader,
)
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document

# Embeddings and vector store
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma

# Import configuration
from config import (
    DEFAULT_CONFIG,
    MODEL_BACKENDS,
    get_backend_config,
    check_api_key,
    print_available_backends
)


class FlexibleRAG:
    """
    A flexible RAG implementation supporting multiple backends.

    This class adapts to different hardware configurations by supporting
    various model backends from retrieval-only to full LLM generation.
    """

    def __init__(
        self,
        backend: str = "retrieval_only",
        data_dir: str = None,
        embedding_model: str = None,
        chunk_size: int = None,
        chunk_overlap: int = None,
        k_retrieve: int = None,
        **kwargs
    ):
        """
        Initialize the flexible RAG system.

        Args:
            backend: Backend to use (see config.py for options)
            data_dir: Directory containing documents to index
            embedding_model: Model for creating embeddings
            chunk_size: Size of text chunks
            chunk_overlap: Overlap between chunks
            k_retrieve: Number of chunks to retrieve
            **kwargs: Additional backend-specific arguments
        """
        # Load defaults
        self.data_dir = Path(data_dir or DEFAULT_CONFIG["data_dir"])
        self.embedding_model_name = embedding_model or DEFAULT_CONFIG["embedding_model"]
        self.chunk_size = chunk_size or DEFAULT_CONFIG["chunk_size"]
        self.chunk_overlap = chunk_overlap or DEFAULT_CONFIG["chunk_overlap"]
        self.k_retrieve = k_retrieve or DEFAULT_CONFIG["k_retrieve"]
        self.persist_directory = DEFAULT_CONFIG["persist_directory"]

        # Backend configuration
        self.backend = backend
        self.backend_config = get_backend_config(backend)

        # State
        self.vectorstore = None
        self.qa_chain = None
        self.embeddings = None
        self.llm = None

        # Check GPU availability
        try:
            import torch
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        except ImportError:
            self.device = "cpu"

        print(f"🚀 Initializing RAG System")
        print(f"   Backend: {self.backend_config.name}")
        print(f"   Device: {self.device}")
        print(f"   RAM Required: {self.backend_config.ram_required}")

        # Check API key if needed
        if self.backend_config.requires_api_key:
            available, message = check_api_key(backend)
            if not available:
                raise ValueError(f"API key check failed: {message}")
            print(f"   {message}")

    def load_prechunked_json(self, json_file: Path) -> List[Document]:
        """
        Load pre-chunked documents from JSON file.

        Expected format:
        [
          {
            "content": "chunk text...",
            "metadata": {"source": "...", "topic": "...", ...}
          },
          ...
        ]

        Args:
            json_file: Path to JSON file

        Returns:
            List of Document objects
        """
        with open(json_file, 'r', encoding='utf-8') as f:
            data = json.load(f)

        documents = []
        for idx, item in enumerate(data):
            content = item.get('content', '').strip()
            if not content:
                continue

            metadata = item.get('metadata', {})
            metadata['source'] = metadata.get('source', json_file.name)
            metadata['chunk_index'] = idx
            metadata['pre_chunked'] = True

            documents.append(Document(
                page_content=content,
                metadata=metadata
            ))

        return documents

    def load_prechunked_txt(self, txt_file: Path) -> List[Document]:
        """
        Load pre-chunked documents from text file.

        Format: Chunks separated by "---" on its own line
        Optional metadata after "META:" in format: key=value, key2=value2

        Args:
            txt_file: Path to text file

        Returns:
            List of Document objects
        """
        with open(txt_file, 'r', encoding='utf-8') as f:
            content = f.read()

        # Split by separator
        chunks = content.split('\n---\n')
        documents = []

        for idx, chunk in enumerate(chunks):
            chunk = chunk.strip()
            if not chunk or chunk.startswith('#'):
                continue

            # Extract metadata if present
            metadata = {
                'source': txt_file.name,
                'chunk_index': idx,
                'pre_chunked': True
            }

            if 'META:' in chunk:
                parts = chunk.split('META:', 1)
                chunk_text = parts[0].strip()
                meta_str = parts[1].strip()

                # Parse metadata
                for pair in meta_str.split(','):
                    pair = pair.strip()
                    if '=' in pair:
                        key, value = pair.split('=', 1)
                        metadata[key.strip()] = value.strip()
            else:
                chunk_text = chunk

            if chunk_text:
                documents.append(Document(
                    page_content=chunk_text,
                    metadata=metadata
                ))

        return documents

    def load_documents(self) -> List[Any]:
        """
        Load all documents from the data directory.

        Supports:
        - .pdf files (auto-chunked)
        - .txt files (auto-chunked)
        - *_chunks.json files (pre-chunked)
        - *_chunks.txt files (pre-chunked)

        Returns:
            List of loaded documents
        """
        print(f"\n📂 Loading documents from {self.data_dir}...")
        documents = []
        prechunked_count = 0

        if not self.data_dir.exists():
            raise FileNotFoundError(f"Data directory {self.data_dir} not found!")

        # Load pre-chunked JSON files
        json_chunk_files = list(self.data_dir.glob("*_chunks.json")) + list(self.data_dir.glob("*chunks.json"))
        for json_file in json_chunk_files:
            try:
                chunks = self.load_prechunked_json(json_file)
                documents.extend(chunks)
                prechunked_count += len(chunks)
                print(f"  ✓ Loaded pre-chunked: {json_file.name} ({len(chunks)} chunks)")
            except Exception as e:
                print(f"  ✗ Error loading {json_file.name}: {e}")

        # Load pre-chunked TXT files
        txt_chunk_files = list(self.data_dir.glob("*_chunks.txt")) + list(self.data_dir.glob("*chunks.txt"))
        for txt_file in txt_chunk_files:
            try:
                chunks = self.load_prechunked_txt(txt_file)
                documents.extend(chunks)
                prechunked_count += len(chunks)
                print(f"  ✓ Loaded pre-chunked: {txt_file.name} ({len(chunks)} chunks)")
            except Exception as e:
                print(f"  ✗ Error loading {txt_file.name}: {e}")

        # Track files to skip (already loaded as pre-chunked)
        prechunked_basenames = {f.stem.replace('_chunks', '').replace('chunks', '') for f in json_chunk_files + txt_chunk_files}

        # Load regular text files (excluding pre-chunked)
        txt_files = [f for f in self.data_dir.glob("*.txt")
                     if not f.name.endswith('_chunks.txt') and not f.name.endswith('chunks.txt')]
        for txt_file in txt_files:
            try:
                loader = TextLoader(str(txt_file), encoding='utf-8')
                documents.extend(loader.load())
                print(f"  ✓ Loaded: {txt_file.name}")
            except Exception as e:
                print(f"  ✗ Error loading {txt_file.name}: {e}")

        # Load PDF files
        pdf_files = list(self.data_dir.glob("*.pdf"))
        for pdf_file in pdf_files:
            try:
                loader = PyPDFLoader(str(pdf_file))
                documents.extend(loader.load())
                print(f"  ✓ Loaded: {pdf_file.name}")
            except Exception as e:
                print(f"  ✗ Error loading {pdf_file.name}: {e}")

        if not documents:
            print("\n⚠️  No documents found! Please add files to the data/ directory.")
            print("   Supported formats:")
            print("   - .pdf, .txt (will be auto-chunked)")
            print("   - *_chunks.json, *_chunks.txt (pre-chunked)")
            return []

        print(f"\n📚 Total documents loaded: {len(documents)}")
        if prechunked_count > 0:
            print(f"   ✓ {prechunked_count} pre-chunked")
            print(f"   ✓ {len(documents) - prechunked_count} will be auto-chunked")

        return documents

    def split_documents(self, documents: List[Any]) -> List[Any]:
        """
        Split documents into smaller chunks for better retrieval.
        Pre-chunked documents are kept as-is.

        Args:
            documents: List of documents to split

        Returns:
            List of document chunks
        """
        # Separate pre-chunked from auto-chunk documents
        prechunked = [doc for doc in documents if doc.metadata.get('pre_chunked', False)]
        to_chunk = [doc for doc in documents if not doc.metadata.get('pre_chunked', False)]

        all_chunks = list(prechunked)  # Start with pre-chunked

        if to_chunk:
            print(f"\n✂️  Splitting documents (size={self.chunk_size}, overlap={self.chunk_overlap})...")

            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=self.chunk_size,
                chunk_overlap=self.chunk_overlap,
                length_function=len,
                separators=["\n\n", "\n", ". ", " ", ""]
            )

            new_chunks = text_splitter.split_documents(to_chunk)
            all_chunks.extend(new_chunks)
            print(f"   Created {len(new_chunks)} auto-chunks")

        if prechunked:
            print(f"   Kept {len(prechunked)} pre-chunks as-is")

        print(f"\n📑 Total chunks: {len(all_chunks)}")
        return all_chunks

    def create_embeddings(self):
        """Initialize the embedding model."""
        print(f"\n🔢 Initializing embeddings with {self.embedding_model_name}...")

        self.embeddings = HuggingFaceEmbeddings(
            model_name=self.embedding_model_name,
            model_kwargs={'device': self.device},
            encode_kwargs={'normalize_embeddings': True}
        )

        print("   ✓ Embeddings initialized")

    def create_vectorstore(self, chunks: List[Any]):
        """
        Create a vector database from document chunks.

        Args:
            chunks: List of document chunks to embed
        """
        print("\n💾 Creating vector database...")

        if not chunks:
            raise ValueError("No chunks to create vectorstore from!")

        self.vectorstore = Chroma.from_documents(
            documents=chunks,
            embedding=self.embeddings,
            persist_directory=self.persist_directory
        )

        print(f"   ✓ Vector database created with {len(chunks)} chunks")

    def initialize_llm(self):
        """Initialize the language model based on backend."""
        if self.backend == "retrieval_only":
            print("\n💡 Using retrieval-only mode (no LLM)")
            self.llm = None
            return

        print(f"\n🤖 Initializing LLM: {self.backend_config.name}...")

        if self.backend in ["groq", "openai", "anthropic"]:
            self._initialize_api_llm()
        else:
            self._initialize_local_llm()

        print("   ✓ LLM initialized")

    def _initialize_local_llm(self):
        """Initialize a local Hugging Face model."""
        import torch
        from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
        from langchain_community.llms import HuggingFacePipeline

        print("   Downloading model (this may take a few minutes on first run)...")

        tokenizer = AutoTokenizer.from_pretrained(
            self.backend_config.model_id,
            trust_remote_code=True
        )

        model = AutoModelForCausalLM.from_pretrained(
            self.backend_config.model_id,
            torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
            device_map="auto" if self.device == "cuda" else None,
            trust_remote_code=True,
            low_cpu_mem_usage=True
        )

        # Create pipeline with backend-specific kwargs
        kwargs = self.backend_config.model_kwargs or {}
        pipe = pipeline(
            "text-generation",
            model=model,
            tokenizer=tokenizer,
            **kwargs
        )

        self.llm = HuggingFacePipeline(pipeline=pipe)

    def _initialize_api_llm(self):
        """Initialize an API-based model."""
        model_id = self.backend_config.model_id
        kwargs = self.backend_config.model_kwargs or {}

        if self.backend == "groq":
            from langchain_groq import ChatGroq
            self.llm = ChatGroq(
                model=model_id,
                **kwargs
            )
        elif self.backend == "openai":
            from langchain_openai import ChatOpenAI
            self.llm = ChatOpenAI(
                model=model_id,
                **kwargs
            )
        elif self.backend == "anthropic":
            from langchain_anthropic import ChatAnthropic
            self.llm = ChatAnthropic(
                model=model_id,
                **kwargs
            )

    def create_qa_chain(self):
        """Create the question-answering chain."""
        if self.backend == "retrieval_only":
            # No QA chain needed for retrieval-only mode
            self.qa_chain = None
            return

        print("\n⛓️  Creating QA chain...")

        from langchain_classic.chains import RetrievalQA
        from langchain_core.prompts import PromptTemplate

        template = """Use the following pieces of context to answer the question at the end.
If you don't know the answer based on the context, just say that you don't know, don't try to make up an answer.
Keep your answer concise and relevant to the question.

Context:
{context}

Question: {question}

Answer:"""

        prompt = PromptTemplate(
            template=template,
            input_variables=["context", "question"]
        )

        self.qa_chain = RetrievalQA.from_chain_type(
            llm=self.llm,
            chain_type="stuff",
            retriever=self.vectorstore.as_retriever(search_kwargs={"k": self.k_retrieve}),
            return_source_documents=True,
            chain_type_kwargs={"prompt": prompt}
        )

        print("   ✓ QA chain created")

    def setup(self):
        """Complete setup: load documents, create embeddings, initialize models."""
        print("\n" + "=" * 70)
        print("🔧 SETTING UP RAG SYSTEM")
        print("=" * 70)

        # Load and process documents
        documents = self.load_documents()
        if not documents:
            return False

        chunks = self.split_documents(documents)

        # Track chunk statistics for UI display
        prechunked_count = sum(1 for c in chunks if c.metadata.get('pre_chunked', False))
        autochunked_count = len(chunks) - prechunked_count
        self._chunk_stats = {
            'prechunked': prechunked_count,
            'autochunked': autochunked_count,
            'total': len(chunks)
        }

        # Create embeddings and vector store
        self.create_embeddings()
        self.create_vectorstore(chunks)

        # Initialize LLM and QA chain (if not retrieval-only)
        self.initialize_llm()
        self.create_qa_chain()

        print("\n" + "=" * 70)
        print("✅ RAG SYSTEM READY!")
        print("=" * 70)
        return True

    def query(self, question: str) -> Dict[str, Any]:
        """
        Ask a question and get an answer with sources.

        Args:
            question: The question to answer

        Returns:
            Dictionary containing answer and source documents
        """
        if self.vectorstore is None:
            raise RuntimeError("RAG system not initialized. Call setup() first!")

        print(f"\n❓ Query: {question}")
        print("-" * 70)

        # Get relevant chunks
        relevant_docs = self.vectorstore.similarity_search(question, k=self.k_retrieve)

        if self.backend == "retrieval_only":
            # Retrieval-only mode: just show the chunks
            print("\n📄 RETRIEVED CONTEXT (Retrieval-Only Mode):")
            print("=" * 70)

            for i, doc in enumerate(relevant_docs, 1):
                print(f"\n[Chunk {i}]")
                print(f"Source: {Path(doc.metadata.get('source', 'Unknown')).name}")
                if 'page' in doc.metadata:
                    print(f"Page: {doc.metadata['page']}")
                print(f"\nContent:\n{doc.page_content}")
                print("-" * 70)

            return {
                'answer': None,
                'chunks': [doc.page_content for doc in relevant_docs],
                'sources': relevant_docs,
                'question': question,
                'mode': 'retrieval_only'
            }
        else:
            # Full mode: use LLM to generate answer
            result = self.qa_chain({"query": question})

            answer = result['result']
            sources = result.get('source_documents', [])

            print(f"\n💬 Answer: {answer}\n")

            if sources:
                print(f"📚 Based on {len(sources)} source(s):")
                for i, doc in enumerate(sources, 1):
                    source = doc.metadata.get('source', 'Unknown')
                    print(f"  {i}. {Path(source).name}")

            return {
                'answer': answer,
                'sources': sources,
                'question': question,
                'mode': 'full'
            }

    def get_relevant_chunks(self, question: str, k: int = None) -> List[Dict[str, Any]]:
        """
        Retrieve relevant document chunks for a question.

        Args:
            question: The query
            k: Number of chunks to retrieve (default: self.k_retrieve)

        Returns:
            List of relevant chunks with metadata
        """
        if self.vectorstore is None:
            raise RuntimeError("Vector store not initialized. Call setup() first!")

        k = k or self.k_retrieve
        docs = self.vectorstore.similarity_search(question, k=k)

        results = []
        for doc in docs:
            results.append({
                'content': doc.page_content,
                'source': doc.metadata.get('source', 'Unknown'),
                'page': doc.metadata.get('page', 'N/A')
            })

        return results


def main():
    """Example usage with interactive backend selection."""
    print("\n" + "=" * 70)
    print("FLEXIBLE RAG SYSTEM - DEMO")
    print("=" * 70)

    # Show available backends
    print_available_backends()

    # Interactive selection (or use default)
    print("Choose a backend (or press Enter for 'retrieval_only'):")
    backend = input("> ").strip() or "retrieval_only"

    try:
        # Initialize RAG
        rag = FlexibleRAG(
            backend=backend,
            data_dir="./data",
            chunk_size=500,
            chunk_overlap=50,
            k_retrieve=3
        )

        # Setup the system
        if not rag.setup():
            print("\n❌ Setup failed! Please add documents to the data/ directory.")
            return

        # Example queries
        print("\n" + "=" * 70)
        print("RUNNING EXAMPLE QUERIES")
        print("=" * 70)

        example_questions = [
            "What is this document about?",
            "Can you summarize the main points?",
        ]

        for question in example_questions:
            result = rag.query(question)
            print()

        print("\n✅ Demo completed successfully!")

    except Exception as e:
        print(f"\n❌ Error: {e}")
        print("\nTip: Try 'retrieval_only' mode first to test the system!")


if __name__ == "__main__":
    main()
