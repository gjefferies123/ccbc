"""Cohere v2 embedding encoder for dense retrieval."""

import logging
from typing import List, Union, Optional
import numpy as np
import requests
import json
from config import Config

logger = logging.getLogger(__name__)


class CohereV2Encoder:
    """Dense encoder using Cohere's v2 embedding API."""
    
    def __init__(self, 
                 api_key: Optional[str] = None, 
                 model: str = "embed-english-v4.0",
                 input_type_search: str = "search_query",
                 input_type_doc: str = "search_document"):
        """Initialize the Cohere v2 encoder.
        
        Args:
            api_key: Cohere API key
            model: Cohere embedding model name (v4.0 as specified)
            input_type_search: Input type for search queries
            input_type_doc: Input type for documents
        """
        self.api_key = api_key or Config.COHERE_API_KEY
        self.model = model
        self.input_type_search = input_type_search
        self.input_type_doc = input_type_doc
        self.base_url = "https://api.cohere.com/v2/embed"
        
        if self.api_key:
            logger.info(f"Cohere v2 encoder initialized with model: {model}")
        else:
            logger.warning("No Cohere API key provided")
    
    def is_available(self) -> bool:
        """Check if Cohere v2 encoder is available."""
        return self.api_key is not None
    
    def _make_embed_request(self, texts: List[str], input_type: str) -> dict:
        """Make a v2 embed API request."""
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        
        payload = {
            "model": self.model,
            "texts": texts,
            "input_type": input_type,
            "embedding_types": ["float"]
        }
        
        response = requests.post(
            self.base_url, 
            headers=headers, 
            json=payload, 
            timeout=60
        )
        
        if response.status_code != 200:
            raise Exception(f"Cohere v2 embed API failed: {response.status_code} - {response.text}")
        
        return response.json()
    
    def encode_queries(self, queries: Union[str, List[str]], normalize: bool = True) -> np.ndarray:
        """Encode queries for search using v2 API.
        
        Args:
            queries: Single query string or list of queries
            normalize: Whether to L2 normalize the embeddings
            
        Returns:
            Normalized embeddings array
        """
        if not self.is_available():
            raise ValueError("Cohere API key not available. Check COHERE_API_KEY.")
        
        if isinstance(queries, str):
            queries = [queries]
        
        logger.debug(f"Encoding {len(queries)} queries with Cohere v2")
        
        try:
            response_data = self._make_embed_request(queries, self.input_type_search)
            
            # Extract embeddings from v2 API response
            if "embeddings" in response_data and "float" in response_data["embeddings"]:
                embeddings = np.array(response_data["embeddings"]["float"])
            else:
                raise Exception(f"Unexpected v2 API response structure: {response_data}")
            
            if normalize:
                embeddings = self._normalize_embeddings(embeddings)
            
            return embeddings
            
        except Exception as e:
            logger.error(f"Cohere v2 query encoding failed: {e}")
            raise
    
    def encode_documents(self, documents: List[str], normalize: bool = True) -> np.ndarray:
        """Encode documents for indexing using v2 API.
        
        Args:
            documents: List of document texts
            normalize: Whether to L2 normalize the embeddings
            
        Returns:
            Normalized embeddings array
        """
        if not self.is_available():
            raise ValueError("Cohere API key not available. Check COHERE_API_KEY.")
        
        logger.debug(f"Encoding {len(documents)} documents with Cohere v2")
        
        try:
            # Process in batches to respect API limits
            batch_size = 96  # Cohere's batch limit
            all_embeddings = []
            
            for i in range(0, len(documents), batch_size):
                batch = documents[i:i + batch_size]
                
                response_data = self._make_embed_request(batch, self.input_type_doc)
                
                # Extract embeddings from v2 API response
                if "embeddings" in response_data and "float" in response_data["embeddings"]:
                    batch_embeddings = np.array(response_data["embeddings"]["float"])
                else:
                    raise Exception(f"Unexpected v2 API response structure: {response_data}")
                
                all_embeddings.append(batch_embeddings)
                
                logger.debug(f"Processed batch {i//batch_size + 1}/{(len(documents) + batch_size - 1)//batch_size}")
            
            embeddings = np.vstack(all_embeddings)
            
            if normalize:
                embeddings = self._normalize_embeddings(embeddings)
            
            return embeddings
            
        except Exception as e:
            logger.error(f"Cohere v2 document encoding failed: {e}")
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
        # Cohere v4 models have different dimensions  
        if "v4" in self.model:
            return 1536  # v4.0 has 1536 dimensions
        elif "v3" in self.model:
            return 1024
        elif "v2" in self.model:
            return 4096
        else:
            # Default assumption
            return 1024


class HybridDenseEncoderV2:
    """Hybrid encoder that uses Cohere v2 API."""
    
    def __init__(self, prefer_cohere_v2: bool = True):
        """Initialize hybrid dense encoder with v2 API.
        
        Args:
            prefer_cohere_v2: Whether to prefer Cohere v2 over other encoders
        """
        self.prefer_cohere_v2 = prefer_cohere_v2
        self.cohere_v2_encoder = None
        self.fallback_encoder = None
        self.active_encoder = None
        
        # Try to initialize Cohere v2 first if preferred
        if prefer_cohere_v2 and Config.COHERE_API_KEY:
            try:
                self.cohere_v2_encoder = CohereV2Encoder(model="embed-english-v4.0")
                if self.cohere_v2_encoder.is_available():
                    self.active_encoder = self.cohere_v2_encoder
                    logger.info("Using Cohere v2 embeddings (embed-english-v4.0)")
            except Exception as e:
                logger.warning(f"Failed to initialize Cohere v2 encoder: {e}")
        
        # Fallback to original Cohere v1 or Sentence Transformers
        if self.active_encoder is None:
            try:
                from search.encoder_cohere import CohereEncoder
                self.fallback_encoder = CohereEncoder()
                if self.fallback_encoder.is_available():
                    self.active_encoder = self.fallback_encoder
                    logger.info("Falling back to Cohere v1 embeddings")
            except Exception as e:
                logger.warning(f"Cohere v1 fallback failed: {e}")
                
                # Final fallback to Sentence Transformers
                try:
                    from search.encoder_dense import DenseEncoder
                    self.fallback_encoder = DenseEncoder()
                    self.active_encoder = self.fallback_encoder
                    logger.info("Using Sentence Transformers embeddings as final fallback")
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
        if isinstance(self.active_encoder, CohereV2Encoder):
            return {
                "type": "cohere_v2",
                "model": self.active_encoder.model,
                "dimension": self.get_dimension(),
                "api_version": "v2"
            }
        elif hasattr(self.active_encoder, 'model'):
            return {
                "type": "cohere_v1", 
                "model": self.active_encoder.model,
                "dimension": self.get_dimension(),
                "api_version": "v1"
            }
        else:
            return {
                "type": "sentence_transformers", 
                "model": getattr(self.active_encoder, 'model_name', 'unknown'),
                "dimension": self.get_dimension(),
                "api_version": "local"
            }


# Global instance for reuse
_hybrid_dense_encoder_v2_instance = None


def get_hybrid_dense_encoder_v2() -> HybridDenseEncoderV2:
    """Get or create a global hybrid dense encoder v2 instance."""
    global _hybrid_dense_encoder_v2_instance
    if _hybrid_dense_encoder_v2_instance is None:
        _hybrid_dense_encoder_v2_instance = HybridDenseEncoderV2()
    return _hybrid_dense_encoder_v2_instance
