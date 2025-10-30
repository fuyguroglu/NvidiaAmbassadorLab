# Pre-Chunked Document Format Guide

## Overview

The RAG system now supports **pre-chunked documents**, giving you complete control over how your content is divided and indexed. This is useful when:

- You want precise control over chunk boundaries
- Your content has natural divisions (sections, Q&A pairs, definitions)
- You've manually curated high-quality chunks
- You want to include custom metadata with each chunk

## How It Works

The system **automatically detects** chunk files and handles them differently:

1. **Pre-chunked files** (with `_chunks` in the name) → Used as-is, no auto-chunking
2. **Regular files** (.pdf, .txt) → Automatically chunked using RecursiveCharacterTextSplitter

You can **mix both approaches** in the same `data/` folder!

---

## Format 1: JSON (Recommended)

### File Naming
- Must end with `_chunks.json` or `chunks.json`
- Examples: `ml_basics_chunks.json`, `faq_chunks.json`, `chunks.json`

### Structure

```json
[
  {
    "content": "Your chunk text goes here...",
    "metadata": {
      "source": "Document Name",
      "topic": "Topic Name",
      "custom_field": "any value"
    }
  },
  {
    "content": "Another chunk...",
    "metadata": {
      "source": "Document Name",
      "topic": "Different Topic"
    }
  }
]
```

### Fields

- **`content`** (required): The text content of the chunk
- **`metadata`** (optional): Custom metadata as key-value pairs
  - `source`: Document source (defaults to filename)
  - `topic`: Topic or section name
  - Any custom fields you want!

### Example: FAQ Chunks

```json
[
  {
    "content": "Q: What is machine learning?\nA: Machine learning is a subset of AI that enables systems to learn from data without being explicitly programmed.",
    "metadata": {
      "source": "ML FAQ",
      "topic": "Definitions",
      "difficulty": "beginner",
      "keywords": "machine learning, AI, definition"
    }
  },
  {
    "content": "Q: What's the difference between supervised and unsupervised learning?\nA: Supervised learning uses labeled data, while unsupervised learning finds patterns in unlabeled data.",
    "metadata": {
      "source": "ML FAQ",
      "topic": "Learning Types",
      "difficulty": "beginner",
      "keywords": "supervised, unsupervised, learning types"
    }
  }
]
```

---

## Format 2: Plain Text

### File Naming
- Must end with `_chunks.txt` or `chunks.txt`
- Examples: `lecture_notes_chunks.txt`, `definitions_chunks.txt`

### Structure

Chunks are separated by `---` on its own line. Optional metadata follows `META:`.

```
---
First chunk content goes here.
Can span multiple lines.
META: source=Lecture 1, topic=Introduction

---
Second chunk content.
META: source=Lecture 1, topic=Background, difficulty=easy

---
Third chunk without metadata.
```

### Rules

1. **Chunk Separator**: `---` on its own line (newline before and after)
2. **Metadata** (optional): Add `META:` followed by comma-separated `key=value` pairs
3. **Comments**: Lines starting with `#` before first `---` are ignored
4. **Whitespace**: Leading/trailing whitespace is automatically trimmed

### Example: Definition Chunks

```
# Machine Learning Definitions
# Pre-chunked for RAG system

---
Machine Learning: A subset of artificial intelligence that enables systems to learn and improve from experience without being explicitly programmed.
META: source=ML Glossary, topic=Core Concepts, type=definition

---
Supervised Learning: A machine learning approach where the model is trained on labeled data, learning to map inputs to known outputs.
META: source=ML Glossary, topic=Learning Types, type=definition

---
Neural Network: A computing system inspired by biological neural networks, consisting of interconnected nodes organized in layers.
META: source=ML Glossary, topic=Architectures, type=definition
```

---

## Mixing Formats

You can have multiple types of files in `data/`:

```
data/
├── textbook.pdf                    # Auto-chunked
├── lecture_notes.txt               # Auto-chunked
├── faq_chunks.json                 # Pre-chunked (JSON)
├── definitions_chunks.txt          # Pre-chunked (text)
└── quiz_questions_chunks.json      # Pre-chunked (JSON)
```

The system will:
1. Load all pre-chunked files as-is
2. Auto-chunk PDF and regular TXT files
3. Combine everything into one vector database

---

## Advantages of Pre-Chunking

### 1. **Precise Control**
```json
{
  "content": "The capital of France is Paris.",
  "metadata": {"type": "fact", "verified": true}
}
```
vs automatic chunking that might split mid-sentence.

### 2. **Semantic Boundaries**
Group related content together:
```json
{
  "content": "Q: What is RAG?\nA: RAG combines retrieval with generation...\nExample: When you ask a question...",
  "metadata": {"type": "q_and_a"}
}
```

### 3. **Rich Metadata**
Add context for better filtering:
```json
{
  "content": "...",
  "metadata": {
    "author": "Dr. Smith",
    "date": "2024-01-15",
    "difficulty": "advanced",
    "prerequisites": ["linear algebra", "calculus"]
  }
}
```

### 4. **Quality Control**
Manually review and refine each chunk for accuracy.

---

## Best Practices

### Chunk Size
- **Aim for**: 100-500 words per chunk
- **Too small**: Lacks context
- **Too large**: Less precise retrieval

### Content Guidelines
1. **Self-contained**: Each chunk should make sense on its own
2. **Topic-focused**: One main idea per chunk
3. **Context**: Include enough background info
4. **No dangling references**: Avoid "as mentioned above" without context

### Metadata Tips
1. **Consistent naming**: Use the same metadata keys across chunks
2. **Meaningful values**: `topic=Introduction` not `topic=1`
3. **Searchable**: Include keywords that users might search for
4. **Hierarchical**: Use dot notation for structure: `section=2.1.3`

---

## Example Use Cases

### Use Case 1: Course Q&A
```json
[
  {
    "content": "Q: How do I submit homework?\nA: Go to the course portal...",
    "metadata": {"category": "submission", "course": "CS101"}
  },
  {
    "content": "Q: What's the late policy?\nA: Assignments can be submitted...",
    "metadata": {"category": "policy", "course": "CS101"}
  }
]
```

### Use Case 2: Code Documentation
```json
[
  {
    "content": "Function: calculate_accuracy(predictions, labels)\nPurpose: Computes classification accuracy\nParameters: predictions (array), labels (array)\nReturns: float accuracy score",
    "metadata": {"type": "function", "module": "metrics", "language": "python"}
  }
]
```

### Use Case 3: Product Documentation
```json
[
  {
    "content": "Installation Steps:\n1. Download installer\n2. Run setup.exe\n3. Follow wizard\n4. Restart computer",
    "metadata": {"section": "installation", "os": "windows", "version": "2.0"}
  }
]
```

---

## Testing Your Chunks

After creating your chunks file, test with:

```bash
conda activate nvidia_rag
cd 1-rag-basics

# Run the system
python rag_flexible.py

# Or use web interface
./start_web_interface.sh
```

The system will show:
```
📂 Loading documents from data...
  ✓ Loaded pre-chunked: your_chunks.json (10 chunks)
  ✓ Loaded: textbook.pdf

📚 Total documents loaded: 11
   ✓ 10 pre-chunked
   ✓ 1 will be auto-chunked

✂️  Splitting documents...
   Created 25 auto-chunks
   Kept 10 pre-chunks as-is

📑 Total chunks: 35
```

---

## Common Mistakes

### ❌ Wrong: Missing separator in TXT
```
First chunk
Second chunk immediately after
```

### ✅ Correct: Proper separator
```
---
First chunk
---
Second chunk
```

### ❌ Wrong: Invalid JSON
```json
[
  {
    content: "Missing quotes"
  }
]
```

### ✅ Correct: Valid JSON
```json
[
  {
    "content": "Proper JSON"
  }
]
```

### ❌ Wrong: Chunks too small
```json
{"content": "Paris."}
```

### ✅ Correct: Sufficient context
```json
{"content": "The capital of France is Paris, located in the north-central part of the country."}
```

---

## Converting Existing Documents

### Manual Conversion
1. Copy your content
2. Identify natural break points
3. Split into logical chunks
4. Add metadata
5. Save as `*_chunks.json` or `*_chunks.txt`

### Semi-Automated
```python
# Example: Split by sections
with open('document.txt') as f:
    content = f.read()

sections = content.split('\n\n## ')  # Split by headers
chunks = []

for i, section in enumerate(sections):
    chunks.append({
        "content": section.strip(),
        "metadata": {
            "source": "document.txt",
            "section_number": i
        }
    })

import json
with open('document_chunks.json', 'w') as f:
    json.dump(chunks, f, indent=2)
```

---

## Summary

- **File naming**: Must include `_chunks` or `chunks` before extension
- **Two formats**: JSON (structured) or TXT (simple)
- **Automatic detection**: System recognizes and handles appropriately
- **Mix and match**: Use pre-chunked + auto-chunked in same folder
- **Full control**: You decide chunk boundaries and metadata

Pre-chunking gives you **precision** where you need it while keeping the convenience of auto-chunking for other documents!

---

**Need Help?**
- Check `data/example_chunks.json` for a working example
- Check `data/example_chunks.txt` for text format example
- Run `python rag_flexible.py` to test your chunks
