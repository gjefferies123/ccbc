"""Dense encoder using multilingual-e5-large for semantic embeddings."""

import logging
from typing import List, Union
import numpy as np
import torch
from sentence_transformers import SentenceTransformer
try:
    from config import Config, SENTENCE_TRANSFORMERS_CACHE
except ImportError:
    import sys
    import os
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from config import Config, SENTENCE_TRANSFORMERS_CACHE

logger = logging.getLogger(__name__)


class DenseEncoder:
    """Dense encoder using multilingual-e5-large model."""
    
    def __init__(self, model_name: str = None, cache_folder: str = None):
        """Initialize the dense encoder.
        
        Args:
            model_name: Name of the sentence transformer model
            cache_folder: Local cache folder for the model
        """
        self.model_name = model_name or Config.DENSE_MODEL
        self.cache_folder = cache_folder or SENTENCE_TRANSFORMERS_CACHE
        self.model = None
        self._load_model()
    
    def _load_model(self) -> None:
        """Load the sentence transformer model."""
        try:
            logger.info(f"Loading dense encoder model: {self.model_name}")
            self.model = SentenceTransformer(
                self.model_name,
                cache_folder=self.cache_folder
            )
            # Set to evaluation mode for consistent results
            self.model.eval()
            logger.info(f"Dense encoder model loaded successfully")
        except Exception as e:
            logger.error(f"Failed to load dense encoder model: {e}")
            raise
    
    def encode_queries(self, queries: Union[str, List[str]], normalize: bool = True) -> np.ndarray:
        """Encode queries for search.
        
        Args:
            queries: Single query string or list of queries
            normalize: Whether to L2 normalize the embeddings
            
        Returns:
            Normalized embeddings array
        """
        if isinstance(queries, str):
            queries = [queries]
        
        # Add query prefix for e5 models
        prefixed_queries = [f"query: {query}" for query in queries]
        
        logger.debug(f"Encoding {len(queries)} queries")
        
        with torch.no_grad():
            embeddings = self.model.encode(
                prefixed_queries,
                convert_to_numpy=True,
                show_progress_bar=len(queries) > 10
            )
        
        if normalize:
            embeddings = self._normalize_embeddings(embeddings)
        
        return embeddings
    
    def encode_documents(self, documents: List[str], normalize: bool = True) -> np.ndarray:
        """Encode documents for indexing.
        
        Args:
            documents: List of document texts
            normalize: Whether to L2 normalize the embeddings
            
        Returns:
            Normalized embeddings array
        """
        # Add passage prefix for e5 models
        prefixed_docs = [f"passage: {doc}" for doc in documents]
        
        logger.debug(f"Encoding {len(documents)} documents")
        
        with torch.no_grad():
            embeddings = self.model.encode(
                prefixed_docs,
                convert_to_numpy=True,
                show_progress_bar=len(documents) > 10,
                batch_size=32  # Process in batches for memory efficiency
            )
        
        if normalize:
            embeddings = self._normalize_embeddings(embeddings)
        
        return embeddings
    
    def _normalize_embeddings(self, embeddings: np.ndarray) -> np.ndarray:
        """L2 normalize embeddings.
        
        Args:
            embeddings: Raw embeddings array
            
        Returns:
            L2 normalized embeddings
        """
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        norms[norms == 0] = 1  # Avoid division by zero
        return embeddings / norms
    
    def get_dimension(self) -> int:
        """Get the embedding dimension."""
        return self.model.get_sentence_embedding_dimension()
    
    @property
    def is_loaded(self) -> bool:
        """Check if the model is loaded."""
        return self.model is not None


# Global instance for reuse
_dense_encoder_instance = None


def get_dense_encoder() -> DenseEncoder:
    """Get or create a global dense encoder instance."""
    global _dense_encoder_instance
    if _dense_encoder_instance is None:
        _dense_encoder_instance = DenseEncoder()
    return _dense_encoder_instance
