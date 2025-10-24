"""
Simple RAG (Retrieval-Augmented Generation) Implementation
============================================================

This module implements a basic RAG system that:
1. Loads documents from the data/ directory
2. Chunks them into manageable pieces
3. Creates embeddings and stores in a vector database
4. Retrieves relevant chunks for a given query
5. Generates answers using an LLM with retrieved context
"""

import os
from typing import List, Dict, Any
from pathlib import Path

# Document loading and processing
from langchain_community.document_loaders import (
    TextLoader,
    PyPDFLoader,
    DirectoryLoader
)
from langchain_text_splitters import RecursiveCharacterTextSplitter

# Embeddings and vector store
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma

# LLM and chains
from langchain_community.llms import HuggingFacePipeline
from langchain_classic.chains import RetrievalQA
from langchain_core.prompts import PromptTemplate

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline


class SimpleRAG:
    """
    A simple RAG implementation using open-source models.

    This class handles document loading, embedding creation,
    and question-answering with retrieval augmentation.
    """

    def __init__(
        self,
        data_dir: str = "./data",
        model_name: str = "microsoft/phi-2",
        embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2",
        chunk_size: int = 500,
        chunk_overlap: int = 50,
        k_retrieve: int = 3
    ):
        """
        Initialize the RAG system.

        Args:
            data_dir: Directory containing documents to index
            model_name: HuggingFace model for text generation
            embedding_model: Model for creating embeddings
            chunk_size: Size of text chunks
            chunk_overlap: Overlap between chunks
            k_retrieve: Number of chunks to retrieve
        """
        self.data_dir = Path(data_dir)
        self.model_name = model_name
        self.embedding_model_name = embedding_model
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.k_retrieve = k_retrieve

        self.vectorstore = None
        self.qa_chain = None
        self.embeddings = None
        self.llm = None

        # Check for GPU
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"Using device: {self.device}")

    def load_documents(self) -> List[Any]:
        """
        Load all documents from the data directory.

        Supports: .txt, .pdf files

        Returns:
            List of loaded documents
        """
        print(f"Loading documents from {self.data_dir}...")
        documents = []

        if not self.data_dir.exists():
            raise FileNotFoundError(f"Data directory {self.data_dir} not found!")

        # Load text files
        txt_files = list(self.data_dir.glob("*.txt"))
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
            print("\n⚠️  No documents found! Please add .txt or .pdf files to the data/ directory.")
            return []

        print(f"\nTotal documents loaded: {len(documents)}")
        return documents

    def split_documents(self, documents: List[Any]) -> List[Any]:
        """
        Split documents into smaller chunks for better retrieval.

        Args:
            documents: List of documents to split

        Returns:
            List of document chunks
        """
        print(f"\nSplitting documents into chunks (size={self.chunk_size}, overlap={self.chunk_overlap})...")

        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
            length_function=len,
            separators=["\n\n", "\n", ". ", " ", ""]
        )

        chunks = text_splitter.split_documents(documents)
        print(f"Created {len(chunks)} chunks")
        return chunks

    def create_embeddings(self):
        """
        Initialize the embedding model.
        """
        print(f"\nInitializing embeddings with {self.embedding_model_name}...")

        self.embeddings = HuggingFaceEmbeddings(
            model_name=self.embedding_model_name,
            model_kwargs={'device': self.device},
            encode_kwargs={'normalize_embeddings': True}
        )

        print("✓ Embeddings initialized")

    def create_vectorstore(self, chunks: List[Any]):
        """
        Create a vector database from document chunks.

        Args:
            chunks: List of document chunks to embed
        """
        print("\nCreating vector database...")

        if not chunks:
            raise ValueError("No chunks to create vectorstore from!")

        self.vectorstore = Chroma.from_documents(
            documents=chunks,
            embedding=self.embeddings,
            persist_directory="./chroma_db"
        )

        print(f"✓ Vector database created with {len(chunks)} chunks")

    def initialize_llm(self):
        """
        Initialize the language model for generation.
        """
        print(f"\nInitializing LLM: {self.model_name}...")
        print("This may take a few minutes on first run...")

        # Load tokenizer and model
        tokenizer = AutoTokenizer.from_pretrained(self.model_name, trust_remote_code=True)
        model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
            device_map="auto" if self.device == "cuda" else None,
            trust_remote_code=True
        )

        # Create pipeline
        pipe = pipeline(
            "text-generation",
            model=model,
            tokenizer=tokenizer,
            max_new_tokens=512,
            temperature=0.7,
            top_p=0.95,
            repetition_penalty=1.15
        )

        self.llm = HuggingFacePipeline(pipeline=pipe)
        print("✓ LLM initialized")

    def create_qa_chain(self):
        """
        Create the question-answering chain with custom prompt.
        """
        print("\nCreating QA chain...")

        # Custom prompt template
        template = """Use the following pieces of context to answer the question at the end.
If you don't know the answer based on the context, just say that you don't know, don't try to make up an answer.

Context:
{context}

Question: {question}

Answer: Let me help you with that based on the provided context."""

        prompt = PromptTemplate(
            template=template,
            input_variables=["context", "question"]
        )

        # Create retrieval QA chain
        self.qa_chain = RetrievalQA.from_chain_type(
            llm=self.llm,
            chain_type="stuff",
            retriever=self.vectorstore.as_retriever(search_kwargs={"k": self.k_retrieve}),
            return_source_documents=True,
            chain_type_kwargs={"prompt": prompt}
        )

        print("✓ QA chain created")

    def setup(self):
        """
        Complete setup: load documents, create embeddings, initialize models.
        """
        print("=" * 60)
        print("Setting up RAG system...")
        print("=" * 60)

        # Load and process documents
        documents = self.load_documents()
        if not documents:
            return False

        chunks = self.split_documents(documents)

        # Create embeddings and vector store
        self.create_embeddings()
        self.create_vectorstore(chunks)

        # Initialize LLM and QA chain
        self.initialize_llm()
        self.create_qa_chain()

        print("\n" + "=" * 60)
        print("✓ RAG system ready!")
        print("=" * 60)
        return True

    def query(self, question: str) -> Dict[str, Any]:
        """
        Ask a question and get an answer with sources.

        Args:
            question: The question to answer

        Returns:
            Dictionary containing answer and source documents
        """
        if self.qa_chain is None:
            raise RuntimeError("RAG system not initialized. Call setup() first!")

        print(f"\nQuery: {question}")
        print("-" * 60)

        result = self.qa_chain({"query": question})

        answer = result['result']
        sources = result.get('source_documents', [])

        print(f"Answer: {answer}\n")

        if sources:
            print(f"Based on {len(sources)} source(s):")
            for i, doc in enumerate(sources, 1):
                source = doc.metadata.get('source', 'Unknown')
                print(f"  {i}. {Path(source).name}")

        return {
            'answer': answer,
            'sources': sources,
            'question': question
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
    """
    Example usage of the RAG system.
    """
    # Initialize RAG
    rag = SimpleRAG(
        data_dir="./data",
        model_name="microsoft/phi-2",  # Small but capable model
        chunk_size=500,
        chunk_overlap=50,
        k_retrieve=3
    )

    # Setup the system
    if not rag.setup():
        print("\n❌ Setup failed! Please add documents to the data/ directory.")
        return

    # Example queries
    example_questions = [
        "What is this document about?",
        "Can you summarize the main points?",
    ]

    print("\n" + "=" * 60)
    print("Running example queries...")
    print("=" * 60)

    for question in example_questions:
        result = rag.query(question)
        print()


if __name__ == "__main__":
    main()
