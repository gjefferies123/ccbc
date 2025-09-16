"""Reranking module with Cohere API and local BGE fallback."""

import logging
from typing import List, Dict, Any, Optional, Tuple
import time
from dataclasses import dataclass
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import cohere

from config import Config
from search.hybrid import SearchResult

logger = logging.getLogger(__name__)


@dataclass
class RerankResult:
    """Reranked search result."""
    search_result: SearchResult
    rerank_score: float
    original_score: float
    rank_change: int  # Positive if moved up, negative if moved down


class CohereReranker:
    """Reranker using Cohere's Rerank API."""
    
    def __init__(self, api_key: Optional[str] = None, model: str = "rerank-3.5-turbo"):
        """Initialize Cohere reranker.
        
        Args:
            api_key: Cohere API key
            model: Cohere rerank model name
        """
        self.api_key = api_key or Config.COHERE_API_KEY
        self.model = model
        self.client = None
        
        if self.api_key:
            try:
                self.client = cohere.Client(self.api_key)
                logger.info(f"Cohere reranker initialized with model: {model}")
            except Exception as e:
                logger.warning(f"Failed to initialize Cohere client: {e}")
    
    def is_available(self) -> bool:
        """Check if Cohere reranker is available."""
        return self.client is not None
    
    def rerank(self, 
               query: str,
               documents: List[str],
               top_k: Optional[int] = None,
               max_chunks_per_doc: int = 512) -> List[Tuple[int, float]]:
        """Rerank documents using Cohere API.
        
        Args:
            query: Search query
            documents: List of document texts
            top_k: Number of top results to return
            max_chunks_per_doc: Maximum tokens per document
            
        Returns:
            List of (original_index, rerank_score) tuples
        """
        if not self.is_available():
            raise ValueError("Cohere reranker not available")
        
        if not documents:
            return []
        
        # Truncate documents if needed
        truncated_docs = []
        for doc in documents:
            if len(doc) > max_chunks_per_doc * 4:  # Rough token estimation
                truncated_docs.append(doc[:max_chunks_per_doc * 4])
            else:
                truncated_docs.append(doc)
        
        try:
            logger.debug(f"Reranking {len(documents)} documents with Cohere")
            
            response = self.client.rerank(
                model=self.model,
                query=query,
                documents=truncated_docs,
                top_k=top_k or len(documents),
                return_documents=False
            )
            
            # Extract results
            results = []
            for result in response.results:
                results.append((result.index, result.relevance_score))
            
            logger.debug(f"Cohere reranking completed, returned {len(results)} results")
            return results
            
        except Exception as e:
            logger.error(f"Cohere reranking failed: {e}")
            raise


class BGEReranker:
    """Local BGE reranker as fallback."""
    
    def __init__(self, model_name: str = None, device: str = None):
        """Initialize BGE reranker.
        
        Args:
            model_name: BGE model name
            device: Device to run on ('cpu', 'cuda', 'auto')
        """
        self.model_name = model_name or Config.RERANK_MODEL
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        
        self.tokenizer = None
        self.model = None
        self._load_model()
    
    def _load_model(self) -> None:
        """Load the BGE reranker model."""
        try:
            logger.info(f"Loading BGE reranker model: {self.model_name}")
            
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            self.model = AutoModelForSequenceClassification.from_pretrained(self.model_name)
            self.model.to(self.device)
            self.model.eval()
            
            logger.info(f"BGE reranker loaded on {self.device}")
            
        except Exception as e:
            logger.error(f"Failed to load BGE reranker: {e}")
            raise
    
    def is_available(self) -> bool:
        """Check if BGE reranker is available."""
        return self.model is not None and self.tokenizer is not None
    
    def rerank(self, 
               query: str,
               documents: List[str],
               top_k: Optional[int] = None,
               batch_size: int = 8,
               max_length: int = 512) -> List[Tuple[int, float]]:
        """Rerank documents using BGE cross-encoder.
        
        Args:
            query: Search query
            documents: List of document texts
            top_k: Number of top results to return
            batch_size: Batch size for processing
            max_length: Maximum sequence length
            
        Returns:
            List of (original_index, rerank_score) tuples
        """
        if not self.is_available():
            raise ValueError("BGE reranker not available")
        
        if not documents:
            return []
        
        logger.debug(f"Reranking {len(documents)} documents with BGE")
        
        scores = []
        
        # Process in batches
        for i in range(0, len(documents), batch_size):
            batch_docs = documents[i:i + batch_size]
            batch_pairs = [(query, doc) for doc in batch_docs]
            
            # Tokenize batch
            inputs = self.tokenizer(
                batch_pairs,
                padding=True,
                truncation=True,
                max_length=max_length,
                return_tensors='pt'
            ).to(self.device)
            
            # Get scores
            with torch.no_grad():
                logits = self.model(**inputs).logits
                batch_scores = torch.nn.functional.sigmoid(logits).squeeze(-1)
                
                if batch_scores.dim() == 0:  # Single score
                    batch_scores = batch_scores.unsqueeze(0)
                
                scores.extend(batch_scores.cpu().numpy().tolist())
        
        # Create results with original indices
        results = [(i, score) for i, score in enumerate(scores)]
        
        # Sort by score descending
        results.sort(key=lambda x: x[1], reverse=True)
        
        # Return top_k results
        if top_k:
            results = results[:top_k]
        
        logger.debug(f"BGE reranking completed, returned {len(results)} results")
        return results


class HybridReranker:
    """Hybrid reranker that uses Cohere with BGE fallback."""
    
    def __init__(self, 
                 cohere_api_key: Optional[str] = None,
                 prefer_cohere: bool = True,
                 fallback_enabled: bool = True):
        """Initialize hybrid reranker.
        
        Args:
            cohere_api_key: Cohere API key
            prefer_cohere: Whether to prefer Cohere over BGE
            fallback_enabled: Whether to enable BGE fallback
        """
        self.prefer_cohere = prefer_cohere
        self.fallback_enabled = fallback_enabled
        
        # Initialize rerankers
        self.cohere_reranker = None
        self.bge_reranker = None
        
        if cohere_api_key or Config.COHERE_API_KEY:
            try:
                self.cohere_reranker = CohereReranker(cohere_api_key)
            except Exception as e:
                logger.warning(f"Failed to initialize Cohere reranker: {e}")
        
        if fallback_enabled:
            try:
                self.bge_reranker = BGEReranker()
            except Exception as e:
                logger.warning(f"Failed to initialize BGE reranker: {e}")
        
        # Determine which reranker to use
        self.active_reranker = None
        if prefer_cohere and self.cohere_reranker and self.cohere_reranker.is_available():
            self.active_reranker = self.cohere_reranker
            logger.info("Using Cohere reranker")
        elif self.bge_reranker and self.bge_reranker.is_available():
            self.active_reranker = self.bge_reranker
            logger.info("Using BGE reranker")
        else:
            logger.warning("No reranker available")
    
    def is_available(self) -> bool:
        """Check if any reranker is available."""
        return self.active_reranker is not None
    
    def rerank_search_results(self, 
                            query: str,
                            search_results: List[SearchResult],
                            top_k: Optional[int] = None) -> List[RerankResult]:
        """Rerank search results.
        
        Args:
            query: Search query
            search_results: List of search results to rerank
            top_k: Number of top results to return
            
        Returns:
            List of reranked results
        """
        if not self.is_available():
            logger.warning("No reranker available, returning original results")
            return [
                RerankResult(
                    search_result=result,
                    rerank_score=result.score,
                    original_score=result.score,
                    rank_change=0
                )
                for result in search_results[:top_k] if top_k else search_results
            ]
        
        if not search_results:
            return []
        
        # Extract document texts
        documents = [result.text for result in search_results]
        
        # Perform reranking
        start_time = time.time()
        
        try:
            rerank_results = self.active_reranker.rerank(
                query=query,
                documents=documents,
                top_k=top_k
            )
            
            rerank_time = time.time() - start_time
            logger.debug(f"Reranking took {rerank_time:.2f} seconds")
            
            # Create rerank result objects
            reranked = []
            for new_rank, (original_idx, rerank_score) in enumerate(rerank_results):
                original_result = search_results[original_idx]
                rank_change = original_idx - new_rank  # Positive if moved up
                
                rerank_result = RerankResult(
                    search_result=original_result,
                    rerank_score=rerank_score,
                    original_score=original_result.score,
                    rank_change=rank_change
                )
                reranked.append(rerank_result)
            
            # Log reranking stats
            self._log_rerank_stats(search_results, reranked)
            
            return reranked
            
        except Exception as e:
            logger.error(f"Reranking failed: {e}")
            
            # Fallback to original results
            return [
                RerankResult(
                    search_result=result,
                    rerank_score=result.score,
                    original_score=result.score,
                    rank_change=0
                )
                for result in search_results[:top_k] if top_k else search_results
            ]
    
    def _log_rerank_stats(self, 
                         original_results: List[SearchResult],
                         reranked_results: List[RerankResult]) -> None:
        """Log reranking statistics.
        
        Args:
            original_results: Original search results
            reranked_results: Reranked results
        """
        if not reranked_results:
            return
        
        # Calculate rank changes
        significant_changes = sum(1 for r in reranked_results if abs(r.rank_change) >= 2)
        improved_ranks = sum(1 for r in reranked_results if r.rank_change > 0)
        
        # Top score comparison
        original_top_score = original_results[0].score if original_results else 0
        reranked_top_score = reranked_results[0].rerank_score
        
        logger.info(
            f"Rerank stats: {significant_changes}/{len(reranked_results)} significant changes, "
            f"{improved_ranks} improved ranks, "
            f"top score: {original_top_score:.3f} -> {reranked_top_score:.3f}"
        )
        
        # Log top-10 rerank scores for debugging
        top_scores = [r.rerank_score for r in reranked_results[:10]]
        logger.debug(f"Top-10 rerank scores: {top_scores}")
    
    def get_reranker_info(self) -> Dict[str, Any]:
        """Get information about the active reranker.
        
        Returns:
            Reranker information
        """
        if not self.active_reranker:
            return {'type': 'none', 'available': False}
        
        if isinstance(self.active_reranker, CohereReranker):
            return {
                'type': 'cohere',
                'model': self.active_reranker.model,
                'available': True
            }
        elif isinstance(self.active_reranker, BGEReranker):
            return {
                'type': 'bge',
                'model': self.active_reranker.model_name,
                'device': self.active_reranker.device,
                'available': True
            }
        else:
            return {'type': 'unknown', 'available': True}


# Global instance
_hybrid_reranker_instance = None


def get_hybrid_reranker() -> HybridReranker:
    """Get or create a global hybrid reranker instance."""
    global _hybrid_reranker_instance
    if _hybrid_reranker_instance is None:
        _hybrid_reranker_instance = HybridReranker()
    return _hybrid_reranker_instance
