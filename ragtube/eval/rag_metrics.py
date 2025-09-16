"""RAG evaluation metrics."""

import logging
from typing import List, Dict, Any, Set, Tuple, Optional
import re
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class EvaluationResult:
    """Result of evaluating a single query."""
    query: str
    predicted_sources: List[Dict[str, Any]]
    expected_phrases: List[str]
    expected_video_id: Optional[str]
    context_precision: float
    context_recall: float
    rerank_gain: float
    correct_video_linked: bool
    response_time: float
    token_count: int


class RAGMetrics:
    """Metrics calculator for RAG evaluation."""
    
    def __init__(self):
        """Initialize metrics calculator."""
        pass
    
    def evaluate_query(self, 
                      query: str,
                      predicted_sources: List[Dict[str, Any]],
                      expected_phrases: List[str],
                      expected_video_id: Optional[str] = None,
                      baseline_sources: Optional[List[Dict[str, Any]]] = None,
                      response_time: float = 0.0,
                      token_count: int = 0) -> EvaluationResult:
        """Evaluate a single query result.
        
        Args:
            query: The search query
            predicted_sources: List of predicted source dictionaries
            expected_phrases: List of phrases that should be found
            expected_video_id: Expected video ID (if any)
            baseline_sources: Baseline sources for rerank gain calculation
            response_time: Response time in seconds
            token_count: Total token count in response
            
        Returns:
            Evaluation result
        """
        # Calculate context precision and recall
        precision, recall = self.calculate_context_precision_recall(
            predicted_sources, expected_phrases
        )
        
        # Check if correct video is linked
        correct_video = self.check_correct_video_linked(
            predicted_sources, expected_video_id
        )
        
        # Calculate rerank gain
        rerank_gain = self.calculate_rerank_gain(
            predicted_sources, baseline_sources, expected_phrases
        )
        
        return EvaluationResult(
            query=query,
            predicted_sources=predicted_sources,
            expected_phrases=expected_phrases,
            expected_video_id=expected_video_id,
            context_precision=precision,
            context_recall=recall,
            rerank_gain=rerank_gain,
            correct_video_linked=correct_video,
            response_time=response_time,
            token_count=token_count
        )
    
    def calculate_context_precision_recall(self, 
                                         predicted_sources: List[Dict[str, Any]],
                                         expected_phrases: List[str]) -> Tuple[float, float]:
        """Calculate context precision and recall.
        
        Context Precision: fraction of retrieved contexts that are relevant
        Context Recall: fraction of relevant contexts that are retrieved
        
        Args:
            predicted_sources: List of predicted sources
            expected_phrases: List of expected phrases
            
        Returns:
            Tuple of (precision, recall)
        """
        if not expected_phrases:
            return 1.0, 1.0
        
        if not predicted_sources:
            return 0.0, 0.0
        
        # Extract all text from predicted sources
        predicted_texts = []
        for source in predicted_sources:
            # Get text from various possible fields
            text = source.get('text', '') or source.get('reason', '') or ''
            predicted_texts.append(text.lower())
        
        combined_predicted_text = ' '.join(predicted_texts)
        
        # Check which expected phrases are found
        phrases_found = 0
        relevant_sources = 0
        
        for phrase in expected_phrases:
            phrase_lower = phrase.lower()
            if phrase_lower in combined_predicted_text:
                phrases_found += 1
        
        # Count relevant sources (sources that contain at least one expected phrase)
        for text in predicted_texts:
            is_relevant = any(phrase.lower() in text for phrase in expected_phrases)
            if is_relevant:
                relevant_sources += 1
        
        # Calculate metrics
        precision = relevant_sources / len(predicted_sources) if predicted_sources else 0.0
        recall = phrases_found / len(expected_phrases) if expected_phrases else 1.0
        
        return precision, recall
    
    def check_correct_video_linked(self, 
                                  predicted_sources: List[Dict[str, Any]],
                                  expected_video_id: Optional[str]) -> bool:
        """Check if the correct video is linked in the sources.
        
        Args:
            predicted_sources: List of predicted sources
            expected_video_id: Expected video ID
            
        Returns:
            True if correct video is linked
        """
        if not expected_video_id or not predicted_sources:
            return True  # No expectation, so consider it correct
        
        for source in predicted_sources:
            url = source.get('url', '')
            # Extract video ID from YouTube URL
            video_id = self._extract_video_id_from_url(url)
            if video_id == expected_video_id:
                return True
        
        return False
    
    def calculate_rerank_gain(self, 
                            reranked_sources: List[Dict[str, Any]],
                            baseline_sources: Optional[List[Dict[str, Any]]],
                            expected_phrases: List[str]) -> float:
        """Calculate reranking gain (Rerank@5).
        
        Args:
            reranked_sources: Sources after reranking
            baseline_sources: Sources before reranking (baseline)
            expected_phrases: Expected phrases for relevance
            
        Returns:
            Rerank gain (positive means improvement)
        """
        if not baseline_sources or not expected_phrases:
            return 0.0
        
        # Calculate relevance scores for both rankings
        reranked_score = self._calculate_relevance_score(reranked_sources, expected_phrases)
        baseline_score = self._calculate_relevance_score(baseline_sources, expected_phrases)
        
        return reranked_score - baseline_score
    
    def _calculate_relevance_score(self, 
                                 sources: List[Dict[str, Any]],
                                 expected_phrases: List[str]) -> float:
        """Calculate relevance score for a list of sources.
        
        Args:
            sources: List of sources
            expected_phrases: Expected phrases
            
        Returns:
            Relevance score (higher is better)
        """
        if not sources or not expected_phrases:
            return 0.0
        
        total_score = 0.0
        
        for i, source in enumerate(sources[:5]):  # Top-5 evaluation
            # Get source text
            text = source.get('text', '') or source.get('reason', '') or ''
            text_lower = text.lower()
            
            # Count matching phrases
            phrase_matches = sum(1 for phrase in expected_phrases 
                               if phrase.lower() in text_lower)
            
            # Weight by position (higher weight for top results)
            position_weight = 1.0 / (i + 1)
            source_score = phrase_matches * position_weight
            
            total_score += source_score
        
        # Normalize by number of expected phrases
        return total_score / len(expected_phrases)
    
    def _extract_video_id_from_url(self, url: str) -> Optional[str]:
        """Extract video ID from YouTube URL.
        
        Args:
            url: YouTube URL
            
        Returns:
            Video ID or None
        """
        if not url:
            return None
        
        # Pattern for youtu.be URLs
        match = re.search(r'youtu\.be/([a-zA-Z0-9_-]+)', url)
        if match:
            return match.group(1)
        
        # Pattern for youtube.com URLs
        match = re.search(r'[?&]v=([a-zA-Z0-9_-]+)', url)
        if match:
            return match.group(1)
        
        return None
    
    def calculate_aggregate_metrics(self, 
                                  results: List[EvaluationResult]) -> Dict[str, float]:
        """Calculate aggregate metrics across all evaluation results.
        
        Args:
            results: List of evaluation results
            
        Returns:
            Dictionary of aggregate metrics
        """
        if not results:
            return {}
        
        # Calculate averages
        avg_precision = sum(r.context_precision for r in results) / len(results)
        avg_recall = sum(r.context_recall for r in results) / len(results)
        avg_rerank_gain = sum(r.rerank_gain for r in results) / len(results)
        avg_response_time = sum(r.response_time for r in results) / len(results)
        avg_token_count = sum(r.token_count for r in results) / len(results)
        
        # Calculate F1 score
        f1_score = 2 * avg_precision * avg_recall / (avg_precision + avg_recall) if (avg_precision + avg_recall) > 0 else 0
        
        # Calculate success rate (correct video linked)
        correct_video_count = sum(1 for r in results if r.correct_video_linked)
        video_success_rate = correct_video_count / len(results)
        
        # Calculate percentage of queries with positive rerank gain
        positive_rerank_count = sum(1 for r in results if r.rerank_gain > 0)
        positive_rerank_rate = positive_rerank_count / len(results)
        
        return {
            'context_precision': avg_precision,
            'context_recall': avg_recall,
            'f1_score': f1_score,
            'rerank_gain': avg_rerank_gain,
            'video_success_rate': video_success_rate,
            'positive_rerank_rate': positive_rerank_rate,
            'avg_response_time': avg_response_time,
            'avg_token_count': avg_token_count,
            'total_queries': len(results)
        }
    
    def print_detailed_results(self, results: List[EvaluationResult]) -> None:
        """Print detailed evaluation results.
        
        Args:
            results: List of evaluation results
        """
        print("\n" + "="*80)
        print("DETAILED EVALUATION RESULTS")
        print("="*80)
        
        for i, result in enumerate(results, 1):
            print(f"\nQuery {i}: {result.query}")
            print(f"Expected phrases: {result.expected_phrases}")
            print(f"Context Precision: {result.context_precision:.3f}")
            print(f"Context Recall: {result.context_recall:.3f}")
            print(f"Rerank Gain: {result.rerank_gain:+.3f}")
            print(f"Correct Video: {'✓' if result.correct_video_linked else '✗'}")
            print(f"Response Time: {result.response_time:.2f}s")
            print(f"Token Count: {result.token_count}")
            
            if result.predicted_sources:
                print("Sources found:")
                for j, source in enumerate(result.predicted_sources[:3], 1):
                    print(f"  {j}. {source.get('video_title', 'Unknown')} ({source.get('start', 0):.0f}s)")
            else:
                print("No sources found")
        
        # Print aggregate metrics
        agg_metrics = self.calculate_aggregate_metrics(results)
        
        print("\n" + "="*80)
        print("AGGREGATE METRICS")
        print("="*80)
        
        for metric, value in agg_metrics.items():
            if 'time' in metric:
                print(f"{metric.replace('_', ' ').title()}: {value:.2f}s")
            elif 'count' in metric:
                print(f"{metric.replace('_', ' ').title()}: {value:.0f}")
            else:
                print(f"{metric.replace('_', ' ').title()}: {value:.3f}")


def create_sample_evaluation_set() -> List[Dict[str, Any]]:
    """Create a sample evaluation set for testing.
    
    Returns:
        List of evaluation queries
    """
    return [
        {
            "q": "forgiveness",
            "must_include": ["forgive", "forgiveness", "mercy"],
            "should_link_video_id": None
        },
        {
            "q": "what is grace?",
            "must_include": ["grace", "unmerited", "favor"],
            "should_link_video_id": None
        },
        {
            "q": "how to pray effectively",
            "must_include": ["prayer", "pray", "communicate"],
            "should_link_video_id": None
        },
        {
            "q": "meaning of salvation",
            "must_include": ["salvation", "saved", "eternal"],
            "should_link_video_id": None
        },
        {
            "q": "biblical love",
            "must_include": ["love", "agape", "unconditional"],
            "should_link_video_id": None
        },
        {
            "q": "faith vs works",
            "must_include": ["faith", "works", "righteousness"],
            "should_link_video_id": None
        },
        {
            "q": "purpose of suffering",
            "must_include": ["suffering", "pain", "growth"],
            "should_link_video_id": None
        },
        {
            "q": "holy spirit role",
            "must_include": ["holy spirit", "spirit", "guidance"],
            "should_link_video_id": None
        },
        {
            "q": "christian discipleship",
            "must_include": ["disciple", "follow", "growth"],
            "should_link_video_id": None
        },
        {
            "q": "eternal life meaning",
            "must_include": ["eternal", "life", "heaven"],
            "should_link_video_id": None
        }
    ]
