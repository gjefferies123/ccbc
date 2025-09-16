"""Configuration management for the RAG pipeline."""

import os
from typing import Optional
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()


class Config:
    """Configuration class that loads settings from environment variables."""
    
    # Pinecone Configuration (Required)
    PINECONE_API_KEY: str = os.getenv("PINECONE_API_KEY", "")
    PINECONE_INDEX: str = os.getenv("PINECONE_INDEX", "ragtube-hybrid")
    PINECONE_ENV: str = os.getenv("PINECONE_ENV", "us-east-1")
    
    # Optional API Keys
    COHERE_API_KEY: Optional[str] = os.getenv("COHERE_API_KEY")
    YOUTUBE_API_KEY: Optional[str] = os.getenv("YOUTUBE_API_KEY")
    
    # Model Configuration
    DENSE_MODEL: str = os.getenv("DENSE_MODEL", "sentence-transformers/multilingual-e5-large")
    COHERE_EMBED_MODEL: str = os.getenv("COHERE_EMBED_MODEL", "embed-english-v3.0")
    USE_COHERE_EMBEDDINGS: bool = os.getenv("USE_COHERE_EMBEDDINGS", "true").lower() == "true"
    RERANK_MODEL: str = os.getenv("RERANK_MODEL", "BAAI/bge-reranker-large")
    
    # Search Configuration
    DEFAULT_TOP_K: int = int(os.getenv("DEFAULT_TOP_K", "50"))
    DEFAULT_ALPHA: float = float(os.getenv("DEFAULT_ALPHA", "0.5"))
    DEFAULT_FINAL_K: int = int(os.getenv("DEFAULT_FINAL_K", "5"))
    MAX_CONTEXT_TOKENS: int = int(os.getenv("MAX_CONTEXT_TOKENS", "2500"))
    
    # Chunking Configuration
    PARENT_SEGMENT_DURATION: int = int(os.getenv("PARENT_SEGMENT_DURATION", "300"))
    CHILD_CHUNK_MIN_DURATION: int = int(os.getenv("CHILD_CHUNK_MIN_DURATION", "45"))
    CHILD_CHUNK_MAX_DURATION: int = int(os.getenv("CHILD_CHUNK_MAX_DURATION", "90"))
    CHILD_CHUNK_OVERLAP: int = int(os.getenv("CHILD_CHUNK_OVERLAP", "15"))
    
    # Query Enhancement
    DEFAULT_USE_MULTI_QUERY: bool = os.getenv("DEFAULT_USE_MULTI_QUERY", "true").lower() == "true"
    DEFAULT_USE_HYDE: bool = os.getenv("DEFAULT_USE_HYDE", "false").lower() == "true"
    MULTI_QUERY_COUNT: int = int(os.getenv("MULTI_QUERY_COUNT", "3"))
    
    @classmethod
    def validate(cls) -> None:
        """Validate required configuration."""
        if not cls.PINECONE_API_KEY:
            raise ValueError("PINECONE_API_KEY is required")
        
        if cls.CHILD_CHUNK_MIN_DURATION >= cls.CHILD_CHUNK_MAX_DURATION:
            raise ValueError("CHILD_CHUNK_MIN_DURATION must be less than CHILD_CHUNK_MAX_DURATION")
        
        if cls.CHILD_CHUNK_OVERLAP >= cls.CHILD_CHUNK_MIN_DURATION:
            raise ValueError("CHILD_CHUNK_OVERLAP must be less than CHILD_CHUNK_MIN_DURATION")


# Constants
PINECONE_DIMENSION = 1024  # multilingual-e5-large dimension
SENTENCE_TRANSFORMERS_CACHE = "./models/sentence-transformers"
BM25_MODEL_PATH = "./models/bm25_encoder.pkl"
MANIFEST_DIR = "./manifests"
