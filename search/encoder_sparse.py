"""Sparse encoder using BM25 for lexical matching."""

import logging
import pickle
from pathlib import Path
from typing import List, Dict, Any, Union
import numpy as np
from pinecone_text import BM25Encoder
try:
    from config import BM25_MODEL_PATH
except ImportError:
    import sys
    import os
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from config import BM25_MODEL_PATH

logger = logging.getLogger(__name__)


class SparseEncoder:
    """Sparse encoder using BM25 for lexical matching."""
    
    def __init__(self, model_path: str = None):
        """Initialize the sparse encoder.
        
        Args:
            model_path: Path to save/load the BM25 model
        """
        self.model_path = model_path or BM25_MODEL_PATH
        self.encoder = None
        self.is_fitted = False
        
        # Try to load existing model
        if Path(self.model_path).exists():
            self.load_model()
    
    def fit(self, corpus: List[str]) -> None:
        """Fit the BM25 encoder on a corpus of documents.
        
        Args:
            corpus: List of document texts to fit the encoder
        """
        logger.info(f"Fitting BM25 encoder on {len(corpus)} documents")
        
        self.encoder = BM25Encoder()
        self.encoder.fit(corpus)
        self.is_fitted = True
        
        # Save the fitted model
        self.save_model()
        logger.info("BM25 encoder fitted and saved successfully")
    
    def encode_queries(self, queries: Union[str, List[str]]) -> List[Dict[str, Any]]:
        """Encode queries for sparse search.
        
        Args:
            queries: Single query string or list of queries
            
        Returns:
            List of sparse vector dictionaries with 'indices' and 'values'
        """
        if not self.is_fitted:
            raise ValueError("BM25 encoder must be fitted before encoding")
        
        if isinstance(queries, str):
            queries = [queries]
        
        logger.debug(f"Encoding {len(queries)} queries with BM25")
        
        sparse_vectors = []
        for query in queries:
            sparse_vec = self.encoder.encode_queries([query])[0]
            sparse_vectors.append(sparse_vec)
        
        return sparse_vectors
    
    def encode_documents(self, documents: List[str]) -> List[Dict[str, Any]]:
        """Encode documents for sparse indexing.
        
        Args:
            documents: List of document texts
            
        Returns:
            List of sparse vector dictionaries with 'indices' and 'values'
        """
        if not self.is_fitted:
            raise ValueError("BM25 encoder must be fitted before encoding")
        
        logger.debug(f"Encoding {len(documents)} documents with BM25")
        
        # BM25Encoder processes documents in batches efficiently
        sparse_vectors = self.encoder.encode_documents(documents)
        
        return sparse_vectors
    
    def save_model(self) -> None:
        """Save the fitted BM25 encoder to disk."""
        if not self.is_fitted:
            raise ValueError("Cannot save unfitted encoder")
        
        # Create directory if it doesn't exist
        Path(self.model_path).parent.mkdir(parents=True, exist_ok=True)
        
        with open(self.model_path, 'wb') as f:
            pickle.dump(self.encoder, f)
        
        logger.info(f"BM25 encoder saved to {self.model_path}")
    
    def load_model(self) -> None:
        """Load a fitted BM25 encoder from disk."""
        if not Path(self.model_path).exists():
            raise FileNotFoundError(f"BM25 model not found at {self.model_path}")
        
        with open(self.model_path, 'rb') as f:
            self.encoder = pickle.load(f)
        
        self.is_fitted = True
        logger.info(f"BM25 encoder loaded from {self.model_path}")
    
    def get_vocab_size(self) -> int:
        """Get the vocabulary size of the fitted encoder."""
        if not self.is_fitted:
            raise ValueError("Encoder must be fitted to get vocab size")
        
        return len(self.encoder.get_vocabulary())
    
    def get_vocabulary(self) -> Dict[str, int]:
        """Get the vocabulary mapping of the fitted encoder."""
        if not self.is_fitted:
            raise ValueError("Encoder must be fitted to get vocabulary")
        
        return self.encoder.get_vocabulary()


# Global instance for reuse
_sparse_encoder_instance = None


def get_sparse_encoder() -> SparseEncoder:
    """Get or create a global sparse encoder instance."""
    global _sparse_encoder_instance
    if _sparse_encoder_instance is None:
        _sparse_encoder_instance = SparseEncoder()
    return _sparse_encoder_instance


def ensure_fitted_on_corpus(corpus: List[str]) -> SparseEncoder:
    """Ensure the sparse encoder is fitted on the given corpus.
    
    Args:
        corpus: List of document texts
        
    Returns:
        Fitted sparse encoder
    """
    encoder = get_sparse_encoder()
    
    if not encoder.is_fitted:
        logger.info("BM25 encoder not fitted, fitting on provided corpus")
        encoder.fit(corpus)
    else:
        logger.info("Using existing fitted BM25 encoder")
    
    return encoder
