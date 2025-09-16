"""Contextual compression to reduce token usage while preserving relevance."""

import logging
from typing import List, Dict, Any, Optional, Tuple
import re
from dataclasses import dataclass

from config import Config
from search.parent_expand import ExpandedResult
from utils.text import (
    extract_relevant_sentences, 
    truncate_to_token_limit, 
    count_tokens,
    split_into_sentences
)

logger = logging.getLogger(__name__)


@dataclass
class CompressedResult:
    """Compressed search result with reduced context."""
    original_result: ExpandedResult
    compressed_text: str
    original_token_count: int
    compressed_token_count: int
    compression_ratio: float
    relevance_score: float
    kept_sentences: int
    total_sentences: int


class ContextualCompressor:
    """Compresses search results while preserving query-relevant content."""
    
    def __init__(self, max_context_tokens: int = None):
        """Initialize contextual compressor.
        
        Args:
            max_context_tokens: Maximum tokens to keep after compression
        """
        self.max_context_tokens = max_context_tokens or Config.MAX_CONTEXT_TOKENS
    
    def compress_results(self, 
                        expanded_results: List[ExpandedResult],
                        query: str,
                        max_total_tokens: Optional[int] = None) -> List[CompressedResult]:
        """Compress expanded results to fit within token limits.
        
        Args:
            expanded_results: List of expanded results to compress
            query: Original search query for relevance filtering
            max_total_tokens: Maximum total tokens across all results
            
        Returns:
            List of compressed results
        """
        if not expanded_results:
            return []
        
        max_total = max_total_tokens or self.max_context_tokens
        
        logger.debug(f"Compressing {len(expanded_results)} results to max {max_total} tokens")
        
        # First pass: compress individual results
        individual_compressed = []
        for result in expanded_results:
            compressed = self._compress_single_result(result, query)
            individual_compressed.append(compressed)
        
        # Second pass: ensure total token budget
        final_compressed = self._fit_within_total_budget(
            individual_compressed, 
            max_total,
            query
        )
        
        # Log compression stats
        self._log_compression_stats(expanded_results, final_compressed)
        
        return final_compressed
    
    def _compress_single_result(self, 
                               expanded_result: ExpandedResult,
                               query: str) -> CompressedResult:
        """Compress a single expanded result.
        
        Args:
            expanded_result: Expanded result to compress
            query: Search query for relevance
            
        Returns:
            Compressed result
        """
        # Get full context text
        full_text = self._get_full_context_text(expanded_result)
        original_token_count = count_tokens(full_text)
        
        # If already within limits, return as-is
        target_tokens = self.max_context_tokens // 5  # Allocate 1/5 of budget per result initially
        
        if original_token_count <= target_tokens:
            return CompressedResult(
                original_result=expanded_result,
                compressed_text=full_text,
                original_token_count=original_token_count,
                compressed_token_count=original_token_count,
                compression_ratio=1.0,
                relevance_score=1.0,
                kept_sentences=len(split_into_sentences(full_text)),
                total_sentences=len(split_into_sentences(full_text))
            )
        
        # Apply compression strategies
        compressed_text = self._apply_compression_strategies(full_text, query, target_tokens)
        compressed_token_count = count_tokens(compressed_text)
        
        # Calculate metrics
        compression_ratio = compressed_token_count / original_token_count if original_token_count > 0 else 1.0
        relevance_score = self._calculate_relevance_score(compressed_text, query)
        
        original_sentences = split_into_sentences(full_text)
        compressed_sentences = split_into_sentences(compressed_text)
        
        return CompressedResult(
            original_result=expanded_result,
            compressed_text=compressed_text,
            original_token_count=original_token_count,
            compressed_token_count=compressed_token_count,
            compression_ratio=compression_ratio,
            relevance_score=relevance_score,
            kept_sentences=len(compressed_sentences),
            total_sentences=len(original_sentences)
        )
    
    def _get_full_context_text(self, expanded_result: ExpandedResult) -> str:
        """Get full context text from expanded result.
        
        Args:
            expanded_result: Expanded result
            
        Returns:
            Full context text
        """
        text_parts = []
        
        # Add parent context if available and not too long
        if expanded_result.parent_text:
            parent_preview = expanded_result.parent_text[:500]  # Limit parent context
            text_parts.append(f"[Context: {parent_preview}...]")
        
        # Add neighbor texts
        original_id = expanded_result.original_result.id
        
        for neighbor in expanded_result.neighbors:
            if neighbor.id == original_id:
                # Mark the main result for emphasis
                text_parts.append(f">>> {neighbor.text} <<<")
            else:
                text_parts.append(neighbor.text)
        
        return ' '.join(text_parts)
    
    def _apply_compression_strategies(self, 
                                    text: str, 
                                    query: str, 
                                    target_tokens: int) -> str:
        """Apply multiple compression strategies.
        
        Args:
            text: Text to compress
            query: Search query for relevance
            target_tokens: Target token count
            
        Returns:
            Compressed text
        """
        # Strategy 1: Extract relevant sentences
        relevant_text = extract_relevant_sentences(
            text, 
            query, 
            max_sentences=20,
            similarity_threshold=0.2
        )
        
        # Strategy 2: Remove redundant phrases
        deduplicated_text = self._remove_redundant_phrases(relevant_text)
        
        # Strategy 3: Truncate to token limit
        final_text = truncate_to_token_limit(deduplicated_text, target_tokens)
        
        # Strategy 4: Clean up formatting
        cleaned_text = self._clean_compressed_text(final_text)
        
        return cleaned_text
    
    def _remove_redundant_phrases(self, text: str) -> str:
        """Remove redundant phrases and filler words.
        
        Args:
            text: Input text
            
        Returns:
            Text with redundancy removed
        """
        # Remove common filler phrases
        filler_patterns = [
            r'\b(?:you know|I mean|basically|essentially|actually|obviously|clearly)\b',
            r'\b(?:um|uh|er|ah)\b',
            r'\[.*?\]',  # Remove bracketed content
            r'\(.*?\)',  # Remove parenthetical content
        ]
        
        cleaned = text
        for pattern in filler_patterns:
            cleaned = re.sub(pattern, '', cleaned, flags=re.IGNORECASE)
        
        # Normalize whitespace
        cleaned = re.sub(r'\s+', ' ', cleaned).strip()
        
        return cleaned
    
    def _clean_compressed_text(self, text: str) -> str:
        """Clean up compressed text formatting.
        
        Args:
            text: Compressed text
            
        Returns:
            Cleaned text
        """
        # Remove incomplete sentences at the end
        sentences = split_into_sentences(text)
        if sentences:
            # Remove last sentence if it doesn't end with proper punctuation
            last_sentence = sentences[-1].strip()
            if last_sentence and not last_sentence.endswith(('.', '!', '?')):
                sentences = sentences[:-1]
        
        # Join and clean
        cleaned = ' '.join(sentences)
        cleaned = re.sub(r'\s+', ' ', cleaned).strip()
        
        return cleaned
    
    def _calculate_relevance_score(self, text: str, query: str) -> float:
        """Calculate relevance score for compressed text.
        
        Args:
            text: Compressed text
            query: Search query
            
        Returns:
            Relevance score between 0 and 1
        """
        if not text or not query:
            return 0.0
        
        # Simple keyword-based relevance
        query_words = set(query.lower().split())
        text_words = set(text.lower().split())
        
        if not query_words:
            return 0.0
        
        # Calculate Jaccard similarity
        intersection = len(query_words & text_words)
        union = len(query_words | text_words)
        
        return intersection / union if union > 0 else 0.0
    
    def _fit_within_total_budget(self, 
                                compressed_results: List[CompressedResult],
                                max_total_tokens: int,
                                query: str) -> List[CompressedResult]:
        """Ensure all results fit within total token budget.
        
        Args:
            compressed_results: Individual compressed results
            max_total_tokens: Maximum total tokens
            query: Search query
            
        Returns:
            Final compressed results within budget
        """
        current_total = sum(r.compressed_token_count for r in compressed_results)
        
        if current_total <= max_total_tokens:
            return compressed_results
        
        logger.debug(f"Total tokens {current_total} exceeds budget {max_total_tokens}, further compression needed")
        
        # Sort by relevance score (keep most relevant)
        sorted_results = sorted(
            compressed_results, 
            key=lambda x: x.relevance_score, 
            reverse=True
        )
        
        # Iteratively add results until budget is exhausted
        final_results = []
        remaining_budget = max_total_tokens
        
        for result in sorted_results:
            if result.compressed_token_count <= remaining_budget:
                final_results.append(result)
                remaining_budget -= result.compressed_token_count
            else:
                # Try to compress this result further to fit
                if remaining_budget > 50:  # Minimum useful size
                    further_compressed = self._compress_to_exact_size(
                        result, 
                        remaining_budget,
                        query
                    )
                    if further_compressed:
                        final_results.append(further_compressed)
                        remaining_budget = 0
                break
        
        return final_results
    
    def _compress_to_exact_size(self, 
                               compressed_result: CompressedResult,
                               target_tokens: int,
                               query: str) -> Optional[CompressedResult]:
        """Compress a result to exact token size.
        
        Args:
            compressed_result: Result to compress further
            target_tokens: Exact target token count
            query: Search query
            
        Returns:
            Further compressed result or None if not possible
        """
        if target_tokens < 20:  # Too small to be useful
            return None
        
        # Extract most relevant sentences
        sentences = split_into_sentences(compressed_result.compressed_text)
        
        # Score sentences by relevance
        scored_sentences = []
        query_words = set(query.lower().split())
        
        for sentence in sentences:
            sentence_words = set(sentence.lower().split())
            intersection = len(query_words & sentence_words)
            union = len(query_words | sentence_words)
            score = intersection / union if union > 0 else 0.0
            scored_sentences.append((score, sentence))
        
        # Sort by relevance
        scored_sentences.sort(key=lambda x: x[0], reverse=True)
        
        # Add sentences until target is reached
        selected_text = ""
        for score, sentence in scored_sentences:
            candidate_text = selected_text + " " + sentence if selected_text else sentence
            if count_tokens(candidate_text) <= target_tokens:
                selected_text = candidate_text
            else:
                break
        
        if not selected_text:
            return None
        
        # Create new compressed result
        new_token_count = count_tokens(selected_text)
        new_compression_ratio = new_token_count / compressed_result.original_token_count
        
        return CompressedResult(
            original_result=compressed_result.original_result,
            compressed_text=selected_text,
            original_token_count=compressed_result.original_token_count,
            compressed_token_count=new_token_count,
            compression_ratio=new_compression_ratio,
            relevance_score=self._calculate_relevance_score(selected_text, query),
            kept_sentences=len(split_into_sentences(selected_text)),
            total_sentences=compressed_result.total_sentences
        )
    
    def _log_compression_stats(self, 
                              original_results: List[ExpandedResult],
                              compressed_results: List[CompressedResult]) -> None:
        """Log compression statistics.
        
        Args:
            original_results: Original expanded results
            compressed_results: Final compressed results
        """
        if not compressed_results:
            return
        
        # Calculate stats
        original_total_tokens = sum(r.token_count for r in original_results)
        compressed_total_tokens = sum(r.compressed_token_count for r in compressed_results)
        
        avg_compression_ratio = sum(r.compression_ratio for r in compressed_results) / len(compressed_results)
        avg_relevance_score = sum(r.relevance_score for r in compressed_results) / len(compressed_results)
        
        total_kept_sentences = sum(r.kept_sentences for r in compressed_results)
        total_original_sentences = sum(r.total_sentences for r in compressed_results)
        
        logger.info(
            f"Compression stats: {len(original_results)} -> {len(compressed_results)} results, "
            f"{original_total_tokens} -> {compressed_total_tokens} tokens "
            f"({avg_compression_ratio:.2f} ratio), "
            f"{total_kept_sentences}/{total_original_sentences} sentences kept, "
            f"avg relevance: {avg_relevance_score:.3f}"
        )


# Global instance
_contextual_compressor_instance = None


def get_contextual_compressor() -> ContextualCompressor:
    """Get or create a global contextual compressor instance."""
    global _contextual_compressor_instance
    if _contextual_compressor_instance is None:
        _contextual_compressor_instance = ContextualCompressor()
    return _contextual_compressor_instance
