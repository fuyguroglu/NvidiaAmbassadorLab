"""
Configuration for RAG System - Multiple Backend Support
========================================================

This file contains configuration options for different model backends,
allowing the RAG system to work on various hardware configurations.

Choose the backend that works best for your hardware!
"""

from dataclasses import dataclass
from typing import Optional, Dict, Any


@dataclass
class ModelConfig:
    """Configuration for a specific model backend."""
    name: str
    description: str
    model_id: Optional[str]
    ram_required: str
    requires_api_key: bool
    api_key_env: Optional[str] = None
    model_kwargs: Optional[Dict[str, Any]] = None


# Available Model Backends
MODEL_BACKENDS = {
    # ============================================================
    # RETRIEVAL ONLY - Works on ANY computer!
    # ============================================================
    "retrieval_only": ModelConfig(
        name="Retrieval Only (No LLM)",
        description="Shows retrieved document chunks without generation. Perfect for learning RAG retrieval!",
        model_id=None,
        ram_required="< 1GB",
        requires_api_key=False,
        model_kwargs=None
    ),

    # ============================================================
    # TINY LOCAL MODELS - Works on most laptops (4GB+ RAM)
    # ============================================================
    "tinyllama": ModelConfig(
        name="TinyLlama 1.1B",
        description="Smallest local model, fast and efficient. Good for basic Q&A.",
        model_id="TinyLlama/TinyLlama-1.1B-Chat-v1.0",
        ram_required="~3-4GB",
        requires_api_key=False,
        model_kwargs={
            "max_new_tokens": 256,
            "temperature": 0.7,
            "top_p": 0.95,
            "repetition_penalty": 1.1
        }
    ),

    # ============================================================
    # SMALL LOCAL MODELS - Requires 6-8GB RAM
    # ============================================================
    "phi2": ModelConfig(
        name="Microsoft Phi-2",
        description="Small but capable model. Good balance of size and performance.",
        model_id="microsoft/phi-2",
        ram_required="~5-6GB",
        requires_api_key=False,
        model_kwargs={
            "max_new_tokens": 512,
            "temperature": 0.7,
            "top_p": 0.95,
            "repetition_penalty": 1.15
        }
    ),

    "phi3_mini": ModelConfig(
        name="Microsoft Phi-3 Mini",
        description="Newer, more efficient than Phi-2. Better instruction following.",
        model_id="microsoft/Phi-3-mini-4k-instruct",
        ram_required="~5-6GB",
        requires_api_key=False,
        model_kwargs={
            "max_new_tokens": 512,
            "temperature": 0.7,
            "top_p": 0.95,
            "repetition_penalty": 1.15
        }
    ),

    # ============================================================
    # API-BASED MODELS - No local compute needed!
    # ============================================================
    "groq": ModelConfig(
        name="Groq (Llama 3.1 8B)",
        description="FASTEST option! Free tier available. Requires GROQ_API_KEY.",
        model_id="llama-3.1-8b-instant",
        ram_required="< 1GB (cloud-based)",
        requires_api_key=True,
        api_key_env="GROQ_API_KEY",
        model_kwargs={
            "temperature": 0.7,
            "max_tokens": 512,
        }
    ),

    "openai": ModelConfig(
        name="OpenAI GPT-3.5",
        description="Very reliable. Requires OPENAI_API_KEY (paid).",
        model_id="gpt-3.5-turbo",
        ram_required="< 1GB (cloud-based)",
        requires_api_key=True,
        api_key_env="OPENAI_API_KEY",
        model_kwargs={
            "temperature": 0.7,
            "max_tokens": 512,
        }
    ),

    "anthropic": ModelConfig(
        name="Anthropic Claude",
        description="High quality responses. Requires ANTHROPIC_API_KEY (paid).",
        model_id="claude-3-haiku-20240307",
        ram_required="< 1GB (cloud-based)",
        requires_api_key=True,
        api_key_env="ANTHROPIC_API_KEY",
        model_kwargs={
            "temperature": 0.7,
            "max_tokens": 512,
        }
    ),
}


# ============================================================
# Default Configuration
# ============================================================
DEFAULT_CONFIG = {
    # Model Backend Selection
    "backend": "retrieval_only",  # Start with the most compatible option

    # Embedding Model (works everywhere, very lightweight)
    "embedding_model": "sentence-transformers/all-MiniLM-L6-v2",

    # Document Processing
    "chunk_size": 500,
    "chunk_overlap": 50,

    # Retrieval
    "k_retrieve": 3,

    # Data Directory
    "data_dir": "./data",

    # Vector Database
    "persist_directory": "./chroma_db",
}


def get_backend_config(backend_name: str) -> ModelConfig:
    """
    Get configuration for a specific backend.

    Args:
        backend_name: Name of the backend (from MODEL_BACKENDS keys)

    Returns:
        ModelConfig object

    Raises:
        ValueError: If backend_name is not found
    """
    if backend_name not in MODEL_BACKENDS:
        available = ", ".join(MODEL_BACKENDS.keys())
        raise ValueError(
            f"Backend '{backend_name}' not found. "
            f"Available backends: {available}"
        )

    return MODEL_BACKENDS[backend_name]


def print_available_backends():
    """Print all available backends with their requirements."""
    print("\n" + "=" * 70)
    print("AVAILABLE MODEL BACKENDS")
    print("=" * 70)

    categories = {
        "No LLM Required": ["retrieval_only"],
        "Local Models (Lightweight)": ["tinyllama"],
        "Local Models (Standard)": ["phi2", "phi3_mini"],
        "API-Based (Cloud)": ["groq", "openai", "anthropic"],
    }

    for category, backends in categories.items():
        print(f"\n📁 {category}")
        print("-" * 70)

        for backend_key in backends:
            if backend_key in MODEL_BACKENDS:
                config = MODEL_BACKENDS[backend_key]
                print(f"\n  🔹 {backend_key.upper()}")
                print(f"     Name: {config.name}")
                print(f"     RAM Required: {config.ram_required}")
                if config.requires_api_key:
                    print(f"     API Key: {config.api_key_env} (required)")
                print(f"     {config.description}")

    print("\n" + "=" * 70)
    print("💡 TIP: Start with 'retrieval_only' to test the system!")
    print("=" * 70 + "\n")


def check_api_key(backend_name: str) -> tuple[bool, str]:
    """
    Check if required API key is available.

    Args:
        backend_name: Name of the backend

    Returns:
        Tuple of (is_available, message)
    """
    import os

    config = get_backend_config(backend_name)

    if not config.requires_api_key:
        return True, "No API key required"

    api_key = os.getenv(config.api_key_env)

    if api_key:
        return True, f"✓ {config.api_key_env} found"
    else:
        return False, (
            f"✗ {config.api_key_env} not found. "
            f"Please set it: export {config.api_key_env}=your_key_here"
        )


if __name__ == "__main__":
    # Print available backends when run directly
    print_available_backends()

    # Example usage
    print("\nExample: Checking API keys...")
    for backend in ["groq", "openai", "anthropic"]:
        available, message = check_api_key(backend)
        print(f"  {backend}: {message}")
