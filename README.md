# LLM Training Lab: Adding New Knowledge to LLMs

A comprehensive training environment for learning modern LLM techniques including RAG, fine-tuning, and optimization strategies. Designed for NVIDIA Ambassador certification preparation with focus on practical, hands-on experience.

## 🎯 Project Goals

This project helps students gain practical experience with:
- **RAG (Retrieval-Augmented Generation)** - Easy-to-use interface for domain-specific Q&A
- **Fine-tuning** - Hands-on examples with parameter-efficient techniques
- **Model Optimization** - Pruning, distillation, and quantization
- **Evaluation** - ROUGE/BLEU, semantic similarity, and LLM-as-a-judge
- **CUDA Acceleration** - Understanding GPU optimization for LLMs

## 📁 Project Structure

```
.
├── 1-rag-basics/              # Simple RAG application (START HERE!)
│   ├── app.py                 # Easy-to-use web interface
│   ├── rag_simple.py          # Core RAG implementation
│   ├── data/                  # Your documents go here
│   └── README.md
│
├── 2-fine-tuning/             # Fine-tuning examples
│   ├── basic_finetuning.py    # Full fine-tuning example
│   ├── peft_lora.py           # Parameter-Efficient Fine-Tuning (LoRA)
│   ├── notebooks/             # Interactive Jupyter notebooks
│   └── README.md
│
├── 3-synthetic-data/          # Data generation strategies
│   ├── generate_qa_pairs.py   # Create Q&A from documents
│   ├── diverse_data.py        # Strategies for diverse datasets
│   └── README.md
│
├── 4-optimization/            # Model optimization techniques
│   ├── pruning.py             # Model pruning examples
│   ├── distillation.py        # Knowledge distillation
│   ├── quantization.py        # Model quantization
│   └── README.md
│
├── 5-decoding-strategies/     # LLM output generation
│   ├── sampling_methods.py    # Top-k, top-p, temperature
│   ├── beam_search.py         # Beam search implementation
│   └── README.md
│
├── 6-evaluation/              # Evaluation techniques
│   ├── metrics.py             # ROUGE, BLEU, semantic similarity
│   ├── llm_as_judge.py        # LLM-based evaluation
│   ├── benchmarking.py        # Performance benchmarking
│   └── README.md
│
├── cuda-optimization/         # CUDA-specific topics
│   ├── gpu_profiling.py       # Profile GPU usage
│   ├── memory_optimization.py # Memory management
│   └── README.md
│
├── datasets/                  # Your training materials
│   ├── documents/             # Add your PDFs, text files here
│   └── qa_pairs/              # Training Q&A pairs
│
├── requirements.txt           # Python dependencies
├── setup.sh                   # Easy setup script
└── docs/                      # Comprehensive documentation
    ├── CONCEPTS.md            # RAG vs Fine-tuning vs Alignment
    ├── NVIDIA_PREP.md         # Ambassador certification prep
    └── TROUBLESHOOTING.md
```

## 🚀 Quick Start (For Non-Technical Users)

### Prerequisites
- NVIDIA GPU (recommended: RTX 3060 or better)
- Python 3.10+
- 16GB+ RAM

### Installation

1. **Clone this repository**
```bash
git clone <repository-url>
cd "Nvidia Ambassador"
```

2. **Run the setup script**
```bash
bash setup.sh
```

3. **Add your documents to `datasets/documents/`**

4. **Start the RAG application**
```bash
cd 1-rag-basics
python app.py
```

5. **Open your browser to** `http://localhost:7860`

That's it! You can now ask questions about your documents!

## 📚 Learning Path

### Week 1: RAG Basics
- [ ] Understand what RAG is and why it's useful
- [ ] Run the RAG application with your documents
- [ ] Try different retrieval strategies
- [ ] Compare RAG vs traditional search

### Week 2: Fine-tuning Fundamentals
- [ ] Learn difference between RAG and fine-tuning
- [ ] Run basic fine-tuning on small model
- [ ] Understand training loss and evaluation metrics
- [ ] Experiment with hyperparameters

### Week 3: Parameter-Efficient Fine-Tuning (PEFT)
- [ ] Implement LoRA (Low-Rank Adaptation)
- [ ] Compare full fine-tuning vs PEFT
- [ ] Measure GPU memory usage differences
- [ ] Understand trade-offs

### Week 4: Synthetic Data Generation
- [ ] Generate Q&A pairs from your documents
- [ ] Create diverse training examples
- [ ] Evaluate data quality
- [ ] Fine-tune with synthetic data

### Week 5: Model Optimization
- [ ] Pruning: Reduce model size
- [ ] Distillation: Transfer knowledge to smaller models
- [ ] Quantization: Reduce precision
- [ ] Benchmark performance vs accuracy

### Week 6: Decoding & Evaluation
- [ ] Experiment with sampling strategies
- [ ] Implement beam search
- [ ] Calculate ROUGE/BLEU scores
- [ ] Set up LLM-as-a-judge evaluation

## 🎓 NVIDIA Ambassador Certification Prep

### Required Experience Documentation

As you work through this project, document:

1. **GPU Acceleration Benefits**
   - Measure training time: CPU vs GPU
   - Profile memory usage
   - Demonstrate speedup metrics

2. **Optimization Strategies**
   - Which PEFT methods you used
   - Memory optimization techniques
   - Batch size and gradient accumulation choices

3. **CUDA Challenges**
   - Out-of-memory errors and solutions
   - Multi-GPU training attempts
   - Performance bottlenecks identified

### Key Concepts to Master

- **RAG vs Fine-tuning vs Alignment**
  - RAG: Retrieve relevant docs at inference time
  - Fine-tuning: Update model weights with new data
  - Alignment: RLHF/DPO to align with human preferences

- **Synthetic Data Strategies**
  - Question generation from documents
  - Paraphrasing for diversity
  - Adversarial examples
  - Data augmentation techniques

- **PEFT Techniques**
  - LoRA (Low-Rank Adaptation)
  - QLoRA (Quantized LoRA)
  - Adapter layers
  - Prefix tuning

## 🔧 Technical Stack

- **Framework**: PyTorch + Hugging Face Transformers
- **RAG**: LangChain + ChromaDB
- **PEFT**: Hugging Face PEFT library
- **Evaluation**: NLTK, SacreBLEU, sentence-transformers
- **UI**: Gradio (simple web interface)
- **GPU**: CUDA 12.x with cuDNN

## 🤝 Contributing

Students are encouraged to:
- Share interesting use cases
- Improve the RAG interface
- Add new optimization techniques
- Share evaluation results
- Document CUDA optimization discoveries

## 📝 License

MIT License - Free for educational use

## 🆘 Getting Help

- Check `docs/TROUBLESHOOTING.md`
- Review `docs/CONCEPTS.md` for theory
- Open an issue on GitHub

---

**Note**: This project is designed for NVIDIA Ambassador certification preparation. Focus on understanding the concepts deeply and documenting your learning journey!
