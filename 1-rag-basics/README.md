# RAG Basics: Flexible Q&A System

A **Retrieval-Augmented Generation (RAG)** system with multiple backend options, designed to work on any hardware - from basic laptops to powerful workstations!

## 🎯 What is RAG?

**RAG (Retrieval-Augmented Generation)** combines document retrieval with LLM generation to answer questions based on your own documents:

1. **Retrieve** relevant information from your documents using semantic search
2. **Augment** the LLM prompt with retrieved context
3. **Generate** answers grounded in your specific documents

Unlike traditional chatbots that rely only on training data, RAG systems answer questions about **YOUR** documents without requiring model fine-tuning!

## ✨ Key Features

- 🎛️ **Multiple Backend Options** - From retrieval-only (works everywhere!) to local LLMs and cloud APIs
- 🌐 **Beautiful Web Interface** - User-friendly Gradio UI, no command-line needed
- 💻 **Hardware Flexible** - Works on laptops with 4GB RAM to workstations with GPUs
- 📚 **Multi-format Support** - PDF and TXT documents
- 🚀 **Easy Setup** - Conda environment + pip install, ready in minutes
- 🔧 **Configurable** - Adjust chunk size, retrieval count, and backends

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────┐
│                      Your Question                       │
└─────────────────────┬───────────────────────────────────┘
                      │
                      ▼
┌─────────────────────────────────────────────────────────┐
│         Embedding Model (Sentence Transformers)          │
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
│           Retrieved Context + Question                   │
│                        ↓                                 │
│              Language Model (Optional)                   │
│      (TinyLlama / Phi / Groq / OpenAI / etc.)           │
│                        ↓                                 │
│                  Generated Answer                        │
└─────────────────────────────────────────────────────────┘
```

## 📁 Project Structure

```
1-rag-basics/
├── config.py                   # Backend configurations
├── rag_flexible.py             # Main RAG implementation
├── app_simple.py               # Gradio web interface
├── test_system.py              # Test with retrieval-only mode
├── test_with_llm.py            # Test with LLM backend
├── start_web_interface.sh      # Linux launcher script
├── start_web_interface.bat     # Windows launcher script
├── requirements.txt            # Python dependencies
├── SETUP_GUIDE.md             # Detailed setup instructions
├── CLAUDE.md                  # Development notes
└── data/                      # Your documents go here!
```

## 🚀 Quick Start

### 1. Setup Environment

See [SETUP_GUIDE.md](SETUP_GUIDE.md) for detailed instructions for Windows and Linux!

**Quick version:**

```bash
# Create conda environment
conda create -n nvidia_rag python=3.11 -y
conda activate nvidia_rag

# Install dependencies
pip install -r requirements.txt
```

### 2. Add Your Documents

```bash
# Place your PDF or TXT files in the data folder
cp your-document.pdf data/
```

### 3. Launch Web Interface

**Linux/WSL:**
```bash
./start_web_interface.sh
```

**Windows:**
```
start_web_interface.bat
```

**Or manually:**
```bash
python app_simple.py
```

Then open **http://localhost:7860** in your browser!

## 🎮 Using the Web Interface

1. **Select Backend** - Choose based on your hardware:
   - **Retrieval Only** - Works on ANY computer (< 1GB RAM)
   - **TinyLlama** - Small local model (3-4GB RAM)
   - **Phi-2/Phi-3** - Better local models (5-6GB RAM)
   - **Groq** - Fast cloud API (Free tier available!)
   - **OpenAI/Anthropic** - Premium cloud APIs

2. **Click "🚀 Initialize System"** - Wait for setup (downloads model on first run)

3. **Ask Questions** - Type your questions and get answers with source citations!

4. **View System Info** - Click "ℹ️ Show System Info" to see configuration

## 🎛️ Backend Options

### Retrieval-Only Mode (Recommended for Beginners)

- **RAM**: < 1GB
- **Setup**: None needed
- **Speed**: Very fast
- **Output**: Shows relevant document chunks without generation

Perfect for:
- Learning how RAG retrieval works
- Testing on any computer
- Understanding document chunking

### Local LLM Models

#### TinyLlama
- **RAM**: ~3-4GB
- **Download**: ~2.2GB (first time only)
- **Speed**: Fast on GPU, acceptable on CPU
- **Quality**: Basic but functional

#### Phi-2 / Phi-3 Mini
- **RAM**: ~5-6GB
- **Download**: ~5GB (first time only)
- **Speed**: Medium
- **Quality**: Good instruction following

### Cloud API Models

#### Groq (Recommended!)
- **RAM**: < 1GB (runs in cloud)
- **Cost**: Free tier available
- **Speed**: Very fast
- **Quality**: Excellent
- **Setup**: `export GROQ_API_KEY=your-key`
- **Sign up**: https://console.groq.com

#### OpenAI / Anthropic
- **RAM**: < 1GB (runs in cloud)
- **Cost**: Paid (pay per use)
- **Quality**: Excellent
- **Setup**: Set API key environment variable

## 🧪 Testing

### Test Retrieval-Only Mode

```bash
python test_system.py
```

Shows how the semantic search retrieves relevant chunks.

### Test with LLM

```bash
python test_with_llm.py
```

Downloads TinyLlama (first run) and generates answers.

## 🔧 Configuration

Edit `config.py` or pass parameters to `FlexibleRAG`:

```python
from rag_flexible import FlexibleRAG

rag = FlexibleRAG(
    backend="retrieval_only",     # or "tinyllama", "phi2", "groq", etc.
    data_dir="./data",
    chunk_size=500,                # Characters per chunk
    chunk_overlap=50,              # Overlap between chunks
    k_retrieve=3                   # Number of chunks to retrieve
)

rag.setup()
result = rag.query("Your question here")
```

## 📊 Understanding Parameters

### Chunk Size & Overlap

- **chunk_size=500**: Size of text chunks in characters
  - Smaller (200-300): More precise retrieval
  - Larger (800-1000): More context per chunk

- **chunk_overlap=50**: Overlap between consecutive chunks
  - Ensures context isn't lost at chunk boundaries

### K (Retrieval Count)

- **k_retrieve=3**: Number of chunks to retrieve
  - Lower (1-2): Faster, more focused
  - Higher (5-10): More comprehensive, may include noise

## 🐛 Common Issues & Solutions

### "Out of Memory" Error

**Solution**: Use retrieval-only mode or cloud API:
```python
backend = "retrieval_only"  # or "groq"
```

### "No documents found"

**Check**:
- Files are in `data/` folder
- Files are `.pdf` or `.txt` format
- Files are not empty

### API Key Not Found

**Set environment variable**:
```bash
# Linux/Mac/WSL
export GROQ_API_KEY="your-key-here"

# Windows (PowerShell)
$env:GROQ_API_KEY="your-key-here"
```

### Model Download is Slow

- This is normal! Models are 2-5GB
- Downloads only happen once
- Subsequent runs are fast (model is cached)

## 🎓 Learning Objectives

By completing this module, you will understand:

- ✅ What RAG is and when to use it
- ✅ How semantic search works with embeddings
- ✅ Document chunking strategies and trade-offs
- ✅ Retrieval quality vs quantity
- ✅ Local vs cloud LLM backends
- ✅ Real-world RAG challenges (retrieval accuracy, hallucination)

## 🔬 Experiments to Try

### 1. Compare Backends

Try the same question with different backends:
- Retrieval-only (see what chunks are found)
- TinyLlama (small model behavior)
- Groq (high-quality cloud model)

Compare answer quality, speed, and whether answers stay grounded in context!

### 2. Adjust Chunk Size

```python
# Try different chunk sizes
for size in [200, 500, 1000]:
    rag = FlexibleRAG(chunk_size=size)
    # Test the same question
```

### 3. Increase K (Retrieval Count)

```python
# Retrieve more chunks
rag = FlexibleRAG(k_retrieve=5)
```

Does more context help or hurt answer quality?

### 4. Test Retrieval Quality

Use retrieval-only mode to see what chunks are found. Are they relevant? If not, try:
- Different chunk sizes
- More overlap
- Better phrased questions

## 💡 RAG vs Fine-tuning

| Aspect | RAG | Fine-tuning |
|--------|-----|-------------|
| **Setup Time** | Minutes | Hours/Days |
| **Data Required** | Documents | Labeled examples |
| **Updates** | Add docs instantly | Retrain model |
| **Cost** | Low | High (training compute) |
| **Use Case** | Q&A, knowledge base | Task adaptation, style |
| **Hallucination** | Lower (grounded) | Higher |

**Use RAG when:**
- You have a knowledge base or documents
- Information changes frequently
- You need source citations
- Quick prototyping

**Use Fine-tuning when:**
- Adapting model behavior or style
- No documents available at runtime
- Specific task performance
- Consistent format requirements

## 📚 Additional Resources

- [SETUP_GUIDE.md](SETUP_GUIDE.md) - Detailed setup for Windows/Linux
- [LangChain Documentation](https://python.langchain.com/)
- [ChromaDB Guide](https://docs.trychroma.com/)
- [Sentence Transformers](https://www.sbert.net/)
- [RAG Paper](https://arxiv.org/abs/2005.11401)

## 🙏 Acknowledgments

Built as part of the NVIDIA Ambassador Lab program, designed to teach RAG fundamentals with accessibility in mind - works on any hardware!

---

**Ready to learn more?** Check out the next module on fine-tuning to see an alternative approach to customizing LLMs!
