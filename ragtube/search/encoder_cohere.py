"""Cohere embedding encoder for dense retrieval."""

import logging
from typing import List, Union, Optional
import numpy as np
import cohere
from config import Config

logger = logging.getLogger(__name__)


class CohereEncoder:
    """Dense encoder using Cohere's embedding models."""
    
    def __init__(self, 
                 api_key: Optional[str] = None, 
                 model: str = "embed-english-v3.0",
                 input_type_search: str = "search_query",
                 input_type_doc: str = "search_document"):
        """Initialize the Cohere encoder.
        
        Args:
            api_key: Cohere API key
            model: Cohere embedding model name
            input_type_search: Input type for search queries
            input_type_doc: Input type for documents
        """
        self.api_key = api_key or Config.COHERE_API_KEY
        self.model = model
        self.input_type_search = input_type_search
        self.input_type_doc = input_type_doc
        self.client = None
        
        if self.api_key:
            try:
                self.client = cohere.Client(self.api_key)
                logger.info(f"Cohere encoder initialized with model: {model}")
            except Exception as e:
                logger.warning(f"Failed to initialize Cohere client: {e}")
    
    def is_available(self) -> bool:
        """Check if Cohere encoder is available."""
        return self.client is not None
    
    def encode_queries(self, queries: Union[str, List[str]], normalize: bool = True) -> np.ndarray:
        """Encode queries for search.
        
        Args:
            queries: Single query string or list of queries
            normalize: Whether to L2 normalize the embeddings
            
        Returns:
            Normalized embeddings array
        """
        if not self.is_available():
            raise ValueError("Cohere client not available. Check COHERE_API_KEY.")
        
        if isinstance(queries, str):
            queries = [queries]
        
        logger.debug(f"Encoding {len(queries)} queries with Cohere")
        
        try:
            response = self.client.embed(
                texts=queries,
                model=self.model,
                input_type=self.input_type_search,
                embedding_types=["float"]
            )
            
            embeddings = np.array(response.embeddings.float)
            
            if normalize:
                embeddings = self._normalize_embeddings(embeddings)
            
            return embeddings
            
        except Exception as e:
            logger.error(f"Cohere query encoding failed: {e}")
            raise
    
    def encode_documents(self, documents: List[str], normalize: bool = True) -> np.ndarray:
        """Encode documents for indexing.
        
        Args:
            documents: List of document texts
            normalize: Whether to L2 normalize the embeddings
            
        Returns:
            Normalized embeddings array
        """
        if not self.is_available():
            raise ValueError("Cohere client not available. Check COHERE_API_KEY.")
        
        logger.debug(f"Encoding {len(documents)} documents with Cohere")
        
        try:
            # Process in batches to respect API limits
            batch_size = 96  # Cohere's batch limit
            all_embeddings = []
            
            for i in range(0, len(documents), batch_size):
                batch = documents[i:i + batch_size]
                
                response = self.client.embed(
                    texts=batch,
                    model=self.model,
                    input_type=self.input_type_doc,
                    embedding_types=["float"]
                )
                
                batch_embeddings = np.array(response.embeddings.float)
                all_embeddings.append(batch_embeddings)
                
                logger.debug(f"Processed batch {i//batch_size + 1}/{(len(documents) + batch_size - 1)//batch_size}")
            
            embeddings = np.vstack(all_embeddings)
            
            if normalize:
                embeddings = self._normalize_embeddings(embeddings)
            
            return embeddings
            
        except Exception as e:
            logger.error(f"Cohere document encoding failed: {e}")
            raise
    
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
        """Get the embedding dimension.
        
        Returns:
            Embedding dimension for the model
        """
        # Cohere v3 models have 1024 dimensions
        if "v3" in self.model:
            return 1024
        # Cohere v2 models have 4096 dimensions  
        elif "v2" in self.model:
            return 4096
        else:
            # Default assumption
            return 1024


class HybridDenseEncoder:
    """Hybrid encoder that can use either Sentence Transformers or Cohere."""
    
    def __init__(self, prefer_cohere: bool = True):
        """Initialize hybrid dense encoder.
        
        Args:
            prefer_cohere: Whether to prefer Cohere over Sentence Transformers
        """
        self.prefer_cohere = prefer_cohere
        self.cohere_encoder = None
        self.st_encoder = None
        self.active_encoder = None
        
        # Try to initialize Cohere first if preferred
        if prefer_cohere and Config.COHERE_API_KEY:
            try:
                self.cohere_encoder = CohereEncoder()
                if self.cohere_encoder.is_available():
                    self.active_encoder = self.cohere_encoder
                    logger.info("Using Cohere embeddings")
            except Exception as e:
                logger.warning(f"Failed to initialize Cohere encoder: {e}")
        
        # Fallback to Sentence Transformers
        if self.active_encoder is None:
            try:
                from search.encoder_dense import DenseEncoder
                self.st_encoder = DenseEncoder()
                self.active_encoder = self.st_encoder
                logger.info("Using Sentence Transformers embeddings")
            except Exception as e:
                logger.error(f"Failed to initialize any dense encoder: {e}")
                raise
    
    def encode_queries(self, queries: Union[str, List[str]], normalize: bool = True) -> np.ndarray:
        """Encode queries using the active encoder."""
        return self.active_encoder.encode_queries(queries, normalize)
    
    def encode_documents(self, documents: List[str], normalize: bool = True) -> np.ndarray:
        """Encode documents using the active encoder."""
        return self.active_encoder.encode_documents(documents, normalize)
    
    def get_dimension(self) -> int:
        """Get embedding dimension from active encoder."""
        return self.active_encoder.get_dimension()
    
    @property
    def is_loaded(self) -> bool:
        """Check if the active encoder is loaded."""
        return self.active_encoder is not None
    
    def get_encoder_info(self) -> dict:
        """Get information about the active encoder."""
        if isinstance(self.active_encoder, CohereEncoder):
            return {
                "type": "cohere",
                "model": self.active_encoder.model,
                "dimension": self.get_dimension()
            }
        else:
            return {
                "type": "sentence_transformers", 
                "model": getattr(self.active_encoder, 'model_name', 'unknown'),
                "dimension": self.get_dimension()
            }


# Global instance for reuse
_hybrid_dense_encoder_instance = None


def get_hybrid_dense_encoder() -> HybridDenseEncoder:
    """Get or create a global hybrid dense encoder instance."""
    global _hybrid_dense_encoder_instance
    if _hybrid_dense_encoder_instance is None:
        _hybrid_dense_encoder_instance = HybridDenseEncoder()
    return _hybrid_dense_encoder_instance
