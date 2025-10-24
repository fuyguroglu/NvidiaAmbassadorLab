# RAG Basics: Simple Q&A System

This module implements a **Retrieval-Augmented Generation (RAG)** system - a powerful technique that combines document retrieval with LLM generation to answer questions based on your own documents.

## 🎯 What is RAG?

**RAG (Retrieval-Augmented Generation)** is a technique that enhances LLMs by:

1. **Retrieving** relevant information from a knowledge base
2. **Augmenting** the LLM prompt with this context
3. **Generating** answers grounded in your specific documents

Unlike traditional chatbots that only use their training data, RAG systems can answer questions about **YOUR** specific documents without requiring model fine-tuning!

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────┐
│                      Your Question                       │
└─────────────────────┬───────────────────────────────────┘
                      │
                      ▼
┌─────────────────────────────────────────────────────────┐
│              Embedding Model (Sentence BERT)             │
│         Converts question to vector embedding            │
└─────────────────────┬───────────────────────────────────┘
                      │
                      ▼
┌─────────────────────────────────────────────────────────┐
│          Vector Database (ChromaDB) Search               │
│      Find most similar document chunks (Top-K)           │
└─────────────────────┬───────────────────────────────────┘
                      │
                      ▼
┌─────────────────────────────────────────────────────────┐
│              Retrieved Context + Question                │
│                        ↓                                 │
│                  Language Model                          │
│              (Microsoft Phi-2 / Others)                  │
│                        ↓                                 │
│                  Generated Answer                        │
└─────────────────────────────────────────────────────────┘
```

## 📁 Files

- **`rag_simple.py`** - Core RAG implementation
  - Document loading (PDF, TXT)
  - Text chunking
  - Embedding creation
  - Vector database management
  - Question answering

- **`app.py`** - Web interface using Gradio
  - User-friendly chat interface
  - Document management
  - Context visualization

- **`data/`** - Your documents go here!
  - Supports: `.txt`, `.pdf`

## 🚀 Quick Start

### 1. Install Dependencies

Make sure you have the required packages:

```bash
pip install torch transformers langchain langchain-community chromadb sentence-transformers gradio pypdf
```

### 2. Add Documents

Place your documents in the `data/` directory:

```bash
cd 1-rag-basics/data
# Copy your .txt or .pdf files here
```

### 3. Run the Application

**Option A: Web Interface (Recommended)**

```bash
python app.py
```

Then open http://localhost:7860 in your browser!

**Option B: Command Line**

```bash
python rag_simple.py
```

## 🎮 Using the Web Interface

1. **Click "List Documents"** - Check that your files are loaded
2. **Click "Initialize RAG System"** - Wait for setup to complete (2-5 minutes first time)
3. **Ask Questions** - Type your questions and get answers!
4. **View Context** - Use the Advanced section to see retrieved chunks

## 🧪 How It Works

### Step 1: Document Processing

```python
# Load documents
documents = rag.load_documents()

# Split into chunks (default: 500 chars with 50 char overlap)
chunks = rag.split_documents(documents)
```

**Why chunking?**
- LLMs have context limits
- Smaller chunks = more precise retrieval
- Overlap ensures context continuity

### Step 2: Embedding Creation

```python
# Create embeddings using Sentence-BERT
embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)
```

**What are embeddings?**
- Vector representations of text
- Similar text = similar vectors
- Enables semantic search (meaning-based, not just keywords)

### Step 3: Vector Store

```python
# Store in ChromaDB for fast similarity search
vectorstore = Chroma.from_documents(
    documents=chunks,
    embedding=embeddings
)
```

**Why a vector database?**
- Efficient similarity search
- Scales to millions of documents
- Persistent storage

### Step 4: Retrieval & Generation

```python
# Retrieve top-K relevant chunks
retriever = vectorstore.as_retriever(k=3)

# Generate answer with context
answer = qa_chain.run(question)
```

## 🎛️ Configuration Options

### Model Selection

Edit `rag_simple.py` or `app.py`:

```python
rag = SimpleRAG(
    model_name="microsoft/phi-2",           # LLM for generation
    embedding_model="sentence-transformers/all-MiniLM-L6-v2",  # Embeddings
    chunk_size=500,                         # Chunk size in characters
    chunk_overlap=50,                       # Overlap between chunks
    k_retrieve=3                            # Number of chunks to retrieve
)
```

### Recommended Models

| Model | Size | VRAM | Speed | Quality |
|-------|------|------|-------|---------|
| microsoft/phi-2 | 2.7B | ~6GB | Fast | Good |
| meta-llama/Llama-2-7b-chat-hf | 7B | ~14GB | Medium | Better |
| mistralai/Mistral-7B-Instruct-v0.2 | 7B | ~14GB | Medium | Better |

### Embedding Models

| Model | Dimensions | Speed | Quality |
|-------|-----------|-------|---------|
| all-MiniLM-L6-v2 | 384 | Very Fast | Good |
| all-mpnet-base-v2 | 768 | Fast | Better |
| instructor-large | 768 | Slow | Best |

## 📊 Understanding Parameters

### Chunk Size & Overlap

```python
chunk_size = 500      # Characters per chunk
chunk_overlap = 50    # Overlap between chunks
```

**Trade-offs:**
- **Smaller chunks** (200-300): More precise, but may miss context
- **Larger chunks** (800-1000): More context, but less precise
- **More overlap**: Better context preservation, but more redundancy

### K (Number of Retrieved Chunks)

```python
k_retrieve = 3  # Retrieve top 3 most relevant chunks
```

**Trade-offs:**
- **Lower K** (1-2): Faster, more focused, but may miss information
- **Higher K** (5-10): More comprehensive, but slower and may include noise

## 🔬 Experiments to Try

### 1. Compare Chunk Sizes

```python
# Test different chunk sizes
for chunk_size in [200, 500, 1000]:
    rag = SimpleRAG(chunk_size=chunk_size)
    # Compare answer quality
```

### 2. Different Retrieval Strategies

- **Similarity search** (default): Vector similarity
- **MMR (Maximal Marginal Relevance)**: Diversity + relevance
- **Similarity with threshold**: Only above certain similarity score

### 3. Model Comparison

Try different LLMs and compare:
- Answer quality
- Speed
- VRAM usage
- Hallucination rate

## 🐛 Troubleshooting

### "CUDA out of memory"

**Solutions:**
1. Use smaller model (phi-2 instead of 7B)
2. Reduce batch size
3. Use CPU (slower): Set `device="cpu"`

### "No documents found"

**Check:**
1. Files are in `1-rag-basics/data/`
2. Supported formats: `.txt`, `.pdf`
3. Files are not empty or corrupted

### "Poor answer quality"

**Try:**
1. Add more relevant documents
2. Increase `k_retrieve` (retrieve more chunks)
3. Adjust `chunk_size` (smaller for precision, larger for context)
4. Use better embedding model
5. Use larger LLM

### "Too slow"

**Optimizations:**
1. Use smaller embedding model (all-MiniLM-L6-v2)
2. Use smaller LLM (phi-2)
3. Reduce `k_retrieve`
4. Use GPU if available

## 📈 Performance Benchmarking

Track these metrics:

```python
import time

start = time.time()
result = rag.query("Your question")
latency = time.time() - start

print(f"Latency: {latency:.2f}s")
print(f"Chunks retrieved: {len(result['sources'])}")
print(f"Answer length: {len(result['answer'])} chars")
```

## 🎓 Learning Objectives

By completing this module, you should understand:

- ✅ What RAG is and why it's useful
- ✅ How document chunking affects retrieval
- ✅ The role of embeddings in semantic search
- ✅ Trade-offs between retrieval parameters
- ✅ When to use RAG vs fine-tuning

## 🔜 Next Steps

1. **Experiment** with different chunk sizes and K values
2. **Add your own documents** and test domain-specific questions
3. **Try different models** and compare quality vs speed
4. **Move to Module 2** (Fine-tuning) to understand alternatives to RAG

## 📚 Additional Resources

- [LangChain Documentation](https://python.langchain.com/docs/get_started/introduction)
- [ChromaDB Guide](https://docs.trychroma.com/)
- [Sentence Transformers](https://www.sbert.net/)
- [RAG Paper](https://arxiv.org/abs/2005.11401)

## 💡 RAG vs Fine-tuning

| Aspect | RAG | Fine-tuning |
|--------|-----|-------------|
| **Setup Time** | Minutes | Hours/Days |
| **Data Required** | Any documents | Labeled examples |
| **Updates** | Add new docs instantly | Retrain model |
| **Cost** | Low (inference only) | High (training compute) |
| **Use Case** | Dynamic knowledge, Q&A | Behavior, style, format |
| **Hallucination** | Lower (grounded in docs) | Higher |

**When to use RAG:**
- Frequently changing information
- Large document collections
- Need source citations
- Quick prototyping

**When to use Fine-tuning:**
- Specific task/domain adaptation
- Style or tone requirements
- No access to documents at runtime
- Behavioral changes needed

---

**Ready to dive deeper?** Move on to `2-fine-tuning/` to learn about model adaptation!
