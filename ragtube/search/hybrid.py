"""Hybrid search combining dense and sparse embeddings with Pinecone."""

import logging
from typing import List, Dict, Any, Optional, Union
from dataclasses import dataclass
import numpy as np

from config import Config, PINECONE_DIMENSION
try:
    from search.encoder_cohere import get_hybrid_dense_encoder
    USE_HYBRID_ENCODER = True
except ImportError:
    from search.encoder_dense import get_dense_encoder
    USE_HYBRID_ENCODER = False
from search.encoder_sparse import get_sparse_encoder
from ingest.upsert import get_upserter

logger = logging.getLogger(__name__)


@dataclass
class SearchResult:
    """Search result from hybrid search."""
    id: str
    score: float
    metadata: Dict[str, Any]
    
    @property
    def video_id(self) -> str:
        return self.metadata.get('video_id', '')
    
    @property
    def text(self) -> str:
        return self.metadata.get('text', '')
    
    @property
    def start_sec(self) -> float:
        return self.metadata.get('start_sec', 0.0)
    
    @property
    def end_sec(self) -> float:
        return self.metadata.get('end_sec', 0.0)
    
    @property
    def source_url(self) -> str:
        return self.metadata.get('source_url', '')


class HybridSearcher:
    """Hybrid searcher combining dense and sparse retrieval."""
    
    def __init__(self, 
                 alpha: float = None,
                 top_k: int = None):
        """Initialize the hybrid searcher.
        
        Args:
            alpha: Weight for dense vs sparse (0.0 = pure sparse, 1.0 = pure dense)
            top_k: Number of results to return
        """
        self.alpha = alpha if alpha is not None else Config.DEFAULT_ALPHA
        self.top_k = top_k if top_k is not None else Config.DEFAULT_TOP_K
        
        if USE_HYBRID_ENCODER:
            self.dense_encoder = get_hybrid_dense_encoder()
        else:
            self.dense_encoder = get_dense_encoder()
        self.sparse_encoder = get_sparse_encoder()
        self.upserter = get_upserter()
        
        logger.info(f"Initialized hybrid searcher with alpha={self.alpha}, top_k={self.top_k}")
    
    def search(self, 
               query: str,
               top_k: Optional[int] = None,
               alpha: Optional[float] = None,
               filters: Optional[Dict[str, Any]] = None,
               include_metadata: bool = True) -> List[SearchResult]:
        """Perform hybrid search.
        
        Args:
            query: Search query
            top_k: Number of results to return
            alpha: Weight for dense vs sparse
            filters: Metadata filters
            include_metadata: Whether to include metadata in results
            
        Returns:
            List of search results
        """
        # Use provided values or defaults
        search_top_k = top_k or self.top_k
        search_alpha = alpha if alpha is not None else self.alpha
        
        logger.debug(f"Hybrid search: query='{query[:50]}...', top_k={search_top_k}, alpha={search_alpha}")
        
        # Generate embeddings
        dense_vector = self._get_dense_embedding(query)
        sparse_vector = self._get_sparse_embedding(query)
        
        # Perform Pinecone query
        try:
            query_response = self.upserter.index.query(
                vector=dense_vector.tolist(),
                sparse_vector=sparse_vector,
                top_k=search_top_k,
                filter=filters,
                include_metadata=include_metadata,
                alpha=search_alpha
            )
            
            # Convert to SearchResult objects
            results = []
            for match in query_response.matches:
                result = SearchResult(
                    id=match.id,
                    score=match.score,
                    metadata=match.metadata or {}
                )
                results.append(result)
            
            logger.debug(f"Retrieved {len(results)} results from hybrid search")
            return results
            
        except Exception as e:
            logger.error(f"Error performing hybrid search: {e}")
            return []
    
    def _get_dense_embedding(self, query: str) -> np.ndarray:
        """Get dense embedding for query.
        
        Args:
            query: Search query
            
        Returns:
            Dense embedding vector
        """
        embeddings = self.dense_encoder.encode_queries([query])
        return embeddings[0]
    
    def _get_sparse_embedding(self, query: str) -> Dict[str, Any]:
        """Get sparse embedding for query.
        
        Args:
            query: Search query
            
        Returns:
            Sparse embedding dictionary
        """
        if not self.sparse_encoder.is_fitted:
            logger.warning("Sparse encoder not fitted, returning empty sparse vector")
            return {'indices': [], 'values': []}
        
        sparse_vectors = self.sparse_encoder.encode_queries([query])
        return sparse_vectors[0]
    
    def search_by_video(self, 
                       query: str,
                       video_id: str,
                       top_k: Optional[int] = None,
                       alpha: Optional[float] = None) -> List[SearchResult]:
        """Search within a specific video.
        
        Args:
            query: Search query
            video_id: Video ID to search within
            top_k: Number of results to return
            alpha: Weight for dense vs sparse
            
        Returns:
            List of search results from the video
        """
        filters = {'video_id': video_id}
        return self.search(query, top_k, alpha, filters)
    
    def search_by_channel(self, 
                         query: str,
                         channel_title: str,
                         top_k: Optional[int] = None,
                         alpha: Optional[float] = None) -> List[SearchResult]:
        """Search within a specific channel.
        
        Args:
            query: Search query
            channel_title: Channel title to search within
            top_k: Number of results to return
            alpha: Weight for dense vs sparse
            
        Returns:
            List of search results from the channel
        """
        filters = {'channel_title': channel_title}
        return self.search(query, top_k, alpha, filters)
    
    def search_by_date_range(self, 
                           query: str,
                           start_date: str,
                           end_date: str,
                           top_k: Optional[int] = None,
                           alpha: Optional[float] = None) -> List[SearchResult]:
        """Search within a date range.
        
        Args:
            query: Search query
            start_date: Start date (ISO format)
            end_date: End date (ISO format)
            top_k: Number of results to return
            alpha: Weight for dense vs sparse
            
        Returns:
            List of search results from the date range
        """
        filters = {
            'published_at': {
                '$gte': start_date,
                '$lte': end_date
            }
        }
        return self.search(query, top_k, alpha, filters)
    
    def search_by_duration(self, 
                          query: str,
                          min_duration: float,
                          max_duration: float,
                          top_k: Optional[int] = None,
                          alpha: Optional[float] = None) -> List[SearchResult]:
        """Search for chunks within a duration range.
        
        Args:
            query: Search query
            min_duration: Minimum chunk duration in seconds
            max_duration: Maximum chunk duration in seconds
            top_k: Number of results to return
            alpha: Weight for dense vs sparse
            
        Returns:
            List of search results within duration range
        """
        filters = {
            'duration': {
                '$gte': min_duration,
                '$lte': max_duration
            }
        }
        return self.search(query, top_k, alpha, filters)
    
    def get_similar_chunks(self, 
                          chunk_id: str,
                          top_k: Optional[int] = None,
                          exclude_same_video: bool = False) -> List[SearchResult]:
        """Find similar chunks to a given chunk.
        
        Args:
            chunk_id: ID of the reference chunk
            top_k: Number of similar chunks to return
            exclude_same_video: Whether to exclude chunks from the same video
            
        Returns:
            List of similar chunks
        """
        search_top_k = top_k or self.top_k
        
        try:
            # Get the reference chunk
            fetch_response = self.upserter.index.fetch([chunk_id])
            
            if chunk_id not in fetch_response.vectors:
                logger.error(f"Chunk {chunk_id} not found")
                return []
            
            reference_vector = fetch_response.vectors[chunk_id]
            
            # Prepare filters
            filters = None
            if exclude_same_video:
                video_id = reference_vector.metadata.get('video_id')
                if video_id:
                    filters = {'video_id': {'$ne': video_id}}
            
            # Search for similar chunks
            query_response = self.upserter.index.query(
                vector=reference_vector.values,
                top_k=search_top_k + 1,  # +1 to account for the reference chunk itself
                filter=filters,
                include_metadata=True
            )
            
            # Convert to SearchResult objects and exclude the reference chunk
            results = []
            for match in query_response.matches:
                if match.id != chunk_id:  # Exclude the reference chunk
                    result = SearchResult(
                        id=match.id,
                        score=match.score,
                        metadata=match.metadata or {}
                    )
                    results.append(result)
            
            return results[:search_top_k]
            
        except Exception as e:
            logger.error(f"Error finding similar chunks: {e}")
            return []
    
    def search_with_context(self, 
                           query: str,
                           context_window: int = 2,
                           top_k: Optional[int] = None,
                           alpha: Optional[float] = None) -> List[SearchResult]:
        """Search and include neighboring chunks for context.
        
        Args:
            query: Search query
            context_window: Number of chunks before/after to include
            top_k: Number of primary results to return
            alpha: Weight for dense vs sparse
            
        Returns:
            List of search results with context
        """
        # Get primary results
        primary_results = self.search(query, top_k, alpha)
        
        if not primary_results:
            return []
        
        # Collect all chunk IDs including neighbors
        all_chunk_ids = set()
        
        for result in primary_results:
            all_chunk_ids.add(result.id)
            
            # Add previous chunks
            for i in range(1, context_window + 1):
                prev_id = result.metadata.get('prev_id')
                if prev_id:
                    all_chunk_ids.add(prev_id)
            
            # Add next chunks  
            for i in range(1, context_window + 1):
                next_id = result.metadata.get('next_id')
                if next_id:
                    all_chunk_ids.add(next_id)
        
        # Fetch all chunks
        try:
            fetch_response = self.upserter.index.fetch(list(all_chunk_ids))
            
            # Create results with context
            results_with_context = []
            for chunk_id, vector in fetch_response.vectors.items():
                result = SearchResult(
                    id=chunk_id,
                    score=1.0,  # Context chunks get default score
                    metadata=vector.metadata or {}
                )
                results_with_context.append(result)
            
            # Sort by video and timestamp
            results_with_context.sort(key=lambda x: (x.video_id, x.start_sec))
            
            return results_with_context
            
        except Exception as e:
            logger.error(f"Error fetching context chunks: {e}")
            return primary_results


class QueryEnhancer:
    """Enhances queries using multi-query and HyDE techniques."""
    
    def __init__(self):
        """Initialize query enhancer."""
        pass
    
    def generate_multi_queries(self, 
                             original_query: str,
                             num_queries: int = None) -> List[str]:
        """Generate multiple variations of the original query.
        
        Args:
            original_query: Original search query
            num_queries: Number of query variations to generate
            
        Returns:
            List of query variations including the original
        """
        num_queries = num_queries or Config.MULTI_QUERY_COUNT
        
        queries = [original_query]
        
        # Simple query variation techniques
        variations = []
        
        # Add question variations
        if not original_query.endswith('?'):
            variations.append(f"What is {original_query}?")
            variations.append(f"How does {original_query} work?")
            variations.append(f"Tell me about {original_query}")
        
        # Add keyword extraction
        keywords = original_query.split()
        if len(keywords) > 1:
            variations.append(' '.join(keywords[-2:]))  # Last two words
            variations.append(keywords[0])  # First word
        
        # Add synonyms (simple replacements)
        synonym_map = {
            'how': 'what way',
            'why': 'what reason',
            'when': 'what time',
            'where': 'what place',
            'explain': 'describe',
            'show': 'demonstrate',
            'teach': 'explain',
        }
        
        for word, synonym in synonym_map.items():
            if word in original_query.lower():
                variations.append(original_query.lower().replace(word, synonym))
        
        # Take the best variations
        queries.extend(variations[:num_queries - 1])
        
        return queries[:num_queries]
    
    def generate_hyde_query(self, original_query: str) -> str:
        """Generate a hypothetical document answer (HyDE).
        
        Args:
            original_query: Original search query
            
        Returns:
            Hypothetical answer text
        """
        # Simple HyDE implementation without LLM
        # In production, this would use an LLM to generate a hypothetical answer
        
        # For now, create a simple hypothetical document
        if '?' in original_query:
            # For questions, create a statement
            question = original_query.strip('?')
            return f"The answer to {question} is that it involves several key concepts and principles."
        else:
            # For statements, create an explanatory text
            return f"This topic about {original_query} covers important aspects and details that are relevant to understanding the subject matter."


# Global instance
_hybrid_searcher_instance = None


def get_hybrid_searcher() -> HybridSearcher:
    """Get or create a global hybrid searcher instance."""
    global _hybrid_searcher_instance
    if _hybrid_searcher_instance is None:
        _hybrid_searcher_instance = HybridSearcher()
    return _hybrid_searcher_instance
