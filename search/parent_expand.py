"""Parent segment expansion for providing fuller context."""

import logging
from typing import List, Dict, Any, Optional, Set
from dataclasses import dataclass

from search.hybrid import SearchResult
from search.rerank import RerankResult
from ingest.upsert import get_upserter

logger = logging.getLogger(__name__)


@dataclass
class ExpandedResult:
    """Search result with expanded parent context."""
    original_result: SearchResult
    parent_text: Optional[str] = None
    parent_metadata: Optional[Dict[str, Any]] = None
    neighbors: List[SearchResult] = None
    total_duration: float = 0.0
    token_count: int = 0
    
    def __post_init__(self):
        if self.neighbors is None:
            self.neighbors = []


class ParentExpander:
    """Expands child chunks with their parent segments and neighbors."""
    
    def __init__(self):
        """Initialize parent expander."""
        self.upserter = get_upserter()
    
    def expand_search_results(self, 
                            results: List[SearchResult],
                            include_neighbors: bool = True,
                            neighbor_window: int = 2) -> List[ExpandedResult]:
        """Expand search results with parent context.
        
        Args:
            results: Search results to expand
            include_neighbors: Whether to include neighboring chunks
            neighbor_window: Number of neighbors to include on each side
            
        Returns:
            List of expanded results
        """
        if not results:
            return []
        
        logger.debug(f"Expanding {len(results)} search results")
        
        # Collect all parent IDs and neighbor IDs
        parent_ids = set()
        neighbor_ids = set()
        
        for result in results:
            parent_id = result.metadata.get('parent_id')
            if parent_id:
                parent_ids.add(parent_id)
            
            if include_neighbors:
                # Collect neighbor IDs
                current_id = result.id
                for direction in ['prev_id', 'next_id']:
                    neighbor_id = result.metadata.get(direction)
                    
                    for _ in range(neighbor_window):
                        if neighbor_id:
                            neighbor_ids.add(neighbor_id)
                            # Try to get the next neighbor (we'll need to fetch this chunk first)
                            break
        
        # Fetch parent segments and neighbors
        parent_data = self._fetch_parent_segments(list(parent_ids))
        neighbor_data = self._fetch_neighbors(neighbor_ids, neighbor_window, results)
        
        # Expand each result
        expanded_results = []
        for result in results:
            expanded = self._expand_single_result(
                result, 
                parent_data, 
                neighbor_data,
                include_neighbors
            )
            expanded_results.append(expanded)
        
        logger.debug(f"Expanded {len(expanded_results)} results")
        return expanded_results
    
    def expand_rerank_results(self, 
                            rerank_results: List[RerankResult],
                            include_neighbors: bool = True,
                            neighbor_window: int = 2) -> List[ExpandedResult]:
        """Expand reranked results with parent context.
        
        Args:
            rerank_results: Reranked results to expand
            include_neighbors: Whether to include neighboring chunks
            neighbor_window: Number of neighbors to include on each side
            
        Returns:
            List of expanded results
        """
        # Extract search results
        search_results = [rr.search_result for rr in rerank_results]
        
        # Expand the search results
        expanded = self.expand_search_results(search_results, include_neighbors, neighbor_window)
        
        # Preserve rerank scores in metadata
        for i, expanded_result in enumerate(expanded):
            if i < len(rerank_results):
                expanded_result.original_result.metadata['rerank_score'] = rerank_results[i].rerank_score
                expanded_result.original_result.metadata['rank_change'] = rerank_results[i].rank_change
        
        return expanded
    
    def _fetch_parent_segments(self, parent_ids: List[str]) -> Dict[str, Dict[str, Any]]:
        """Fetch parent segment data.
        
        Args:
            parent_ids: List of parent segment IDs
            
        Returns:
            Dictionary mapping parent_id to parent data
        """
        if not parent_ids:
            return {}
        
        parent_data = {}
        
        try:
            # Since parent segments aren't directly stored in Pinecone,
            # we need to reconstruct them from child chunks
            # For now, we'll use a simpler approach: fetch the first chunk of each parent
            
            for parent_id in parent_ids:
                # Query for chunks with this parent_id, ordered by chunk_ix
                query_response = self.upserter.index.query(
                    vector=[0] * 1024,  # Dummy vector
                    filter={'parent_id': parent_id},
                    top_k=50,  # Get all chunks for this parent
                    include_metadata=True
                )
                
                if query_response.matches:
                    # Sort by chunk index to reconstruct parent text
                    chunks = sorted(
                        query_response.matches,
                        key=lambda x: x.metadata.get('chunk_ix', 0)
                    )
                    
                    # Combine text from all chunks to reconstruct parent
                    parent_text_parts = []
                    parent_metadata = chunks[0].metadata.copy()
                    
                    for chunk in chunks:
                        chunk_text = chunk.metadata.get('text', '')
                        if chunk_text:
                            parent_text_parts.append(chunk_text)
                    
                    parent_data[parent_id] = {
                        'text': ' '.join(parent_text_parts),
                        'metadata': parent_metadata,
                        'chunk_count': len(chunks)
                    }
                    
        except Exception as e:
            logger.error(f"Error fetching parent segments: {e}")
        
        return parent_data
    
    def _fetch_neighbors(self, 
                        neighbor_ids: Set[str], 
                        neighbor_window: int,
                        original_results: List[SearchResult]) -> Dict[str, SearchResult]:
        """Fetch neighboring chunks.
        
        Args:
            neighbor_ids: Set of neighbor chunk IDs
            neighbor_window: Number of neighbors to fetch
            original_results: Original search results to get neighbor chains
            
        Returns:
            Dictionary mapping chunk_id to SearchResult
        """
        if not neighbor_ids:
            return {}
        
        neighbor_data = {}
        
        try:
            # Fetch all neighbor chunks
            if neighbor_ids:
                fetch_response = self.upserter.index.fetch(list(neighbor_ids))
                
                for chunk_id, vector in fetch_response.vectors.items():
                    neighbor_result = SearchResult(
                        id=chunk_id,
                        score=0.0,  # Neighbors don't have search scores
                        metadata=vector.metadata or {}
                    )
                    neighbor_data[chunk_id] = neighbor_result
            
            # For each original result, fetch the full neighbor chain
            for result in original_results:
                self._fetch_neighbor_chain(result, neighbor_data, neighbor_window, 'prev_id')
                self._fetch_neighbor_chain(result, neighbor_data, neighbor_window, 'next_id')
                
        except Exception as e:
            logger.error(f"Error fetching neighbors: {e}")
        
        return neighbor_data
    
    def _fetch_neighbor_chain(self, 
                            result: SearchResult,
                            neighbor_data: Dict[str, SearchResult],
                            neighbor_window: int,
                            direction: str) -> None:
        """Fetch a chain of neighbors in one direction.
        
        Args:
            result: Starting result
            neighbor_data: Dictionary to store neighbor data
            neighbor_window: Number of neighbors to fetch
            direction: 'prev_id' or 'next_id'
        """
        current_id = result.metadata.get(direction)
        
        for _ in range(neighbor_window):
            if not current_id or current_id in neighbor_data:
                break
            
            try:
                fetch_response = self.upserter.index.fetch([current_id])
                
                if current_id in fetch_response.vectors:
                    vector = fetch_response.vectors[current_id]
                    neighbor_result = SearchResult(
                        id=current_id,
                        score=0.0,
                        metadata=vector.metadata or {}
                    )
                    neighbor_data[current_id] = neighbor_result
                    
                    # Get next neighbor in chain
                    current_id = vector.metadata.get(direction) if vector.metadata else None
                else:
                    break
                    
            except Exception as e:
                logger.debug(f"Error fetching neighbor {current_id}: {e}")
                break
    
    def _expand_single_result(self, 
                            result: SearchResult,
                            parent_data: Dict[str, Dict[str, Any]],
                            neighbor_data: Dict[str, SearchResult],
                            include_neighbors: bool) -> ExpandedResult:
        """Expand a single search result.
        
        Args:
            result: Search result to expand
            parent_data: Parent segment data
            neighbor_data: Neighbor chunk data
            include_neighbors: Whether to include neighbors
            
        Returns:
            Expanded result
        """
        # Get parent information
        parent_id = result.metadata.get('parent_id')
        parent_text = None
        parent_metadata = None
        
        if parent_id and parent_id in parent_data:
            parent_info = parent_data[parent_id]
            parent_text = parent_info['text']
            parent_metadata = parent_info['metadata']
        
        # Get neighbors
        neighbors = []
        total_duration = result.end_sec - result.start_sec
        
        if include_neighbors:
            # Get previous neighbors
            prev_id = result.metadata.get('prev_id')
            while prev_id and prev_id in neighbor_data:
                neighbor = neighbor_data[prev_id]
                neighbors.insert(0, neighbor)  # Insert at beginning
                total_duration += neighbor.end_sec - neighbor.start_sec
                prev_id = neighbor.metadata.get('prev_id')
            
            # Add the main result
            neighbors.append(result)
            
            # Get next neighbors
            next_id = result.metadata.get('next_id')
            while next_id and next_id in neighbor_data:
                neighbor = neighbor_data[next_id]
                neighbors.append(neighbor)
                total_duration += neighbor.end_sec - neighbor.start_sec
                next_id = neighbor.metadata.get('next_id')
        else:
            neighbors = [result]
        
        # Calculate token count (rough estimation)
        all_text = []
        if parent_text:
            all_text.append(parent_text)
        for neighbor in neighbors:
            all_text.append(neighbor.text)
        
        combined_text = ' '.join(all_text)
        token_count = len(combined_text.split()) * 1.3  # Rough token estimation
        
        return ExpandedResult(
            original_result=result,
            parent_text=parent_text,
            parent_metadata=parent_metadata,
            neighbors=neighbors,
            total_duration=total_duration,
            token_count=int(token_count)
        )
    
    def get_full_context_text(self, expanded_result: ExpandedResult) -> str:
        """Get the full context text for an expanded result.
        
        Args:
            expanded_result: Expanded result
            
        Returns:
            Combined context text
        """
        text_parts = []
        
        # Add parent context if available
        if expanded_result.parent_text:
            text_parts.append(f"[Context: {expanded_result.parent_text}]")
        
        # Add neighbor texts in order
        for neighbor in expanded_result.neighbors:
            text_parts.append(neighbor.text)
        
        return ' '.join(text_parts)
    
    def get_focused_context_text(self, 
                                expanded_result: ExpandedResult,
                                focus_on_original: bool = True) -> str:
        """Get focused context text highlighting the original result.
        
        Args:
            expanded_result: Expanded result
            focus_on_original: Whether to emphasize the original result
            
        Returns:
            Focused context text
        """
        if not expanded_result.neighbors:
            return expanded_result.original_result.text
        
        text_parts = []
        original_id = expanded_result.original_result.id
        
        for neighbor in expanded_result.neighbors:
            if neighbor.id == original_id and focus_on_original:
                # Highlight the original result
                text_parts.append(f"**{neighbor.text}**")
            else:
                text_parts.append(neighbor.text)
        
        return ' '.join(text_parts)


# Global instance
_parent_expander_instance = None


def get_parent_expander() -> ParentExpander:
    """Get or create a global parent expander instance."""
    global _parent_expander_instance
    if _parent_expander_instance is None:
        _parent_expander_instance = ParentExpander()
    return _parent_expander_instance
