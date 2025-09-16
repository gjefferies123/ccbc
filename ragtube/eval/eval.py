"""Evaluation harness for the RAG pipeline."""

import logging
import time
import yaml
import argparse
from pathlib import Path
from typing import List, Dict, Any, Optional
import sys
import os

# Add the parent directory to the path to import our modules
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import Config
from search.hybrid import get_hybrid_searcher
from search.rerank import get_hybrid_reranker
from search.parent_expand import get_parent_expander
from search.compress import get_contextual_compressor
from eval.rag_metrics import RAGMetrics, EvaluationResult, create_sample_evaluation_set

logger = logging.getLogger(__name__)


class RAGEvaluator:
    """Evaluator for the RAG pipeline."""
    
    def __init__(self):
        """Initialize the evaluator."""
        self.hybrid_searcher = get_hybrid_searcher()
        self.reranker = get_hybrid_reranker()
        self.parent_expander = get_parent_expander()
        self.compressor = get_contextual_compressor()
        self.metrics = RAGMetrics()
    
    def evaluate_query_set(self, 
                          query_set: List[Dict[str, Any]],
                          use_reranking: bool = True,
                          use_compression: bool = True,
                          alpha: float = 0.5,
                          top_k: int = 5) -> List[EvaluationResult]:
        """Evaluate a set of queries.
        
        Args:
            query_set: List of query dictionaries
            use_reranking: Whether to use reranking
            use_compression: Whether to use compression
            alpha: Alpha parameter for hybrid search
            top_k: Number of top results to return
            
        Returns:
            List of evaluation results
        """
        results = []
        
        logger.info(f"Evaluating {len(query_set)} queries")
        
        for i, query_data in enumerate(query_set, 1):
            logger.info(f"Evaluating query {i}/{len(query_set)}: {query_data['q']}")
            
            try:
                result = self.evaluate_single_query(
                    query_data,
                    use_reranking=use_reranking,
                    use_compression=use_compression,
                    alpha=alpha,
                    top_k=top_k
                )
                results.append(result)
                
            except Exception as e:
                logger.error(f"Failed to evaluate query {i}: {e}")
                # Create a failed result
                failed_result = EvaluationResult(
                    query=query_data['q'],
                    predicted_sources=[],
                    expected_phrases=query_data.get('must_include', []),
                    expected_video_id=query_data.get('should_link_video_id'),
                    context_precision=0.0,
                    context_recall=0.0,
                    rerank_gain=0.0,
                    correct_video_linked=False,
                    response_time=0.0,
                    token_count=0
                )
                results.append(failed_result)
        
        return results
    
    def evaluate_single_query(self, 
                             query_data: Dict[str, Any],
                             use_reranking: bool = True,
                             use_compression: bool = True,
                             alpha: float = 0.5,
                             top_k: int = 5) -> EvaluationResult:
        """Evaluate a single query.
        
        Args:
            query_data: Query data dictionary
            use_reranking: Whether to use reranking
            use_compression: Whether to use compression
            alpha: Alpha parameter for hybrid search
            top_k: Number of top results to return
            
        Returns:
            Evaluation result
        """
        query = query_data['q']
        expected_phrases = query_data.get('must_include', [])
        expected_video_id = query_data.get('should_link_video_id')
        
        start_time = time.time()
        
        # Step 1: Hybrid search
        search_results = self.hybrid_searcher.search(
            query=query,
            top_k=Config.DEFAULT_TOP_K,
            alpha=alpha
        )
        
        baseline_sources = self._convert_search_results_to_sources(search_results[:top_k])
        
        if not search_results:
            return EvaluationResult(
                query=query,
                predicted_sources=[],
                expected_phrases=expected_phrases,
                expected_video_id=expected_video_id,
                context_precision=0.0,
                context_recall=0.0,
                rerank_gain=0.0,
                correct_video_linked=False,
                response_time=time.time() - start_time,
                token_count=0
            )
        
        # Step 2: Reranking (optional)
        if use_reranking and self.reranker.is_available():
            reranked_results = self.reranker.rerank_search_results(
                query=query,
                search_results=search_results,
                top_k=top_k
            )
            
            # Convert to search results for consistency
            final_search_results = [rr.search_result for rr in reranked_results]
        else:
            final_search_results = search_results[:top_k]
        
        # Step 3: Parent expansion
        expanded_results = self.parent_expander.expand_search_results(final_search_results)
        
        # Step 4: Compression (optional)
        if use_compression:
            compressed_results = self.compressor.compress_results(
                expanded_results,
                query=query
            )
            total_tokens = sum(r.compressed_token_count for r in compressed_results)
        else:
            compressed_results = expanded_results
            total_tokens = sum(r.token_count for r in expanded_results)
        
        # Convert to source format
        predicted_sources = self._convert_compressed_results_to_sources(compressed_results)
        
        response_time = time.time() - start_time
        
        # Evaluate using metrics
        result = self.metrics.evaluate_query(
            query=query,
            predicted_sources=predicted_sources,
            expected_phrases=expected_phrases,
            expected_video_id=expected_video_id,
            baseline_sources=baseline_sources,
            response_time=response_time,
            token_count=total_tokens
        )
        
        return result
    
    def _convert_search_results_to_sources(self, search_results) -> List[Dict[str, Any]]:
        """Convert search results to source format.
        
        Args:
            search_results: List of SearchResult objects
            
        Returns:
            List of source dictionaries
        """
        sources = []
        
        for result in search_results:
            source = {
                'video_title': result.metadata.get('video_title', 'Unknown'),
                'url': result.metadata.get('source_url', ''),
                'start': result.start_sec,
                'end': result.end_sec,
                'text': result.text,
                'reason': f"Search score: {result.score:.3f}"
            }
            sources.append(source)
        
        return sources
    
    def _convert_compressed_results_to_sources(self, compressed_results) -> List[Dict[str, Any]]:
        """Convert compressed results to source format.
        
        Args:
            compressed_results: List of compressed results
            
        Returns:
            List of source dictionaries
        """
        sources = []
        
        for result in compressed_results:
            original_result = result.original_result.original_result
            metadata = original_result.metadata
            
            source = {
                'video_title': metadata.get('video_title', 'Unknown'),
                'url': metadata.get('source_url', ''),
                'start': original_result.start_sec,
                'end': original_result.end_sec,
                'text': result.compressed_text,
                'reason': f"Relevance: {result.relevance_score:.3f}, Compression: {result.compression_ratio:.2f}"
            }
            sources.append(source)
        
        return sources
    
    def run_comparative_evaluation(self, 
                                 query_set: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Run comparative evaluation with different configurations.
        
        Args:
            query_set: List of query dictionaries
            
        Returns:
            Comparative results
        """
        logger.info("Running comparative evaluation")
        
        configurations = [
            {"name": "Dense Only", "alpha": 1.0, "use_reranking": False},
            {"name": "Sparse Only", "alpha": 0.0, "use_reranking": False},
            {"name": "Hybrid (α=0.5)", "alpha": 0.5, "use_reranking": False},
            {"name": "Hybrid + Rerank", "alpha": 0.5, "use_reranking": True},
        ]
        
        results = {}
        
        for config in configurations:
            logger.info(f"Evaluating configuration: {config['name']}")
            
            config_results = self.evaluate_query_set(
                query_set,
                use_reranking=config["use_reranking"],
                alpha=config["alpha"]
            )
            
            aggregate_metrics = self.metrics.calculate_aggregate_metrics(config_results)
            
            results[config["name"]] = {
                "config": config,
                "results": config_results,
                "metrics": aggregate_metrics
            }
        
        return results
    
    def print_comparative_results(self, comparative_results: Dict[str, Any]) -> None:
        """Print comparative evaluation results.
        
        Args:
            comparative_results: Results from comparative evaluation
        """
        print("\n" + "="*100)
        print("COMPARATIVE EVALUATION RESULTS")
        print("="*100)
        
        # Print summary table
        print(f"\n{'Configuration':<20} {'Precision':<10} {'Recall':<10} {'F1':<10} {'Rerank Gain':<12} {'Video Success':<13} {'Avg Time':<10}")
        print("-" * 95)
        
        for config_name, data in comparative_results.items():
            metrics = data["metrics"]
            print(f"{config_name:<20} "
                  f"{metrics['context_precision']:<10.3f} "
                  f"{metrics['context_recall']:<10.3f} "
                  f"{metrics['f1_score']:<10.3f} "
                  f"{metrics['rerank_gain']:<12.3f} "
                  f"{metrics['video_success_rate']:<13.3f} "
                  f"{metrics['avg_response_time']:<10.2f}")
        
        # Print detailed results for best configuration
        best_config = max(comparative_results.items(), key=lambda x: x[1]["metrics"]["f1_score"])
        print(f"\n\nDETAILED RESULTS FOR BEST CONFIGURATION: {best_config[0]}")
        print("="*80)
        
        self.metrics.print_detailed_results(best_config[1]["results"])


def load_evaluation_set(file_path: str) -> List[Dict[str, Any]]:
    """Load evaluation set from YAML file.
    
    Args:
        file_path: Path to YAML file
        
    Returns:
        List of query dictionaries
    """
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            data = yaml.safe_load(f)
        
        if isinstance(data, dict) and 'queries' in data:
            return data['queries']
        elif isinstance(data, list):
            return data
        else:
            raise ValueError("Invalid evaluation file format")
            
    except Exception as e:
        logger.error(f"Failed to load evaluation set from {file_path}: {e}")
        raise


def create_sample_evalset_file(file_path: str) -> None:
    """Create a sample evaluation set file.
    
    Args:
        file_path: Path to create the file
    """
    sample_queries = create_sample_evaluation_set()
    
    evalset_data = {
        "description": "Sample evaluation set for RAGTube",
        "queries": sample_queries
    }
    
    with open(file_path, 'w', encoding='utf-8') as f:
        yaml.dump(evalset_data, f, default_flow_style=False, allow_unicode=True)
    
    logger.info(f"Created sample evaluation set at {file_path}")


def main():
    """Main evaluation function."""
    parser = argparse.ArgumentParser(description="Evaluate RAG pipeline")
    parser.add_argument("--evalset", type=str, help="Path to evaluation set YAML file")
    parser.add_argument("--create-sample", action="store_true", help="Create sample evaluation set")
    parser.add_argument("--comparative", action="store_true", help="Run comparative evaluation")
    parser.add_argument("--alpha", type=float, default=0.5, help="Alpha parameter for hybrid search")
    parser.add_argument("--top-k", type=int, default=5, help="Number of top results")
    parser.add_argument("--no-rerank", action="store_true", help="Disable reranking")
    parser.add_argument("--no-compress", action="store_true", help="Disable compression")
    
    args = parser.parse_args()
    
    # Set up logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Create sample evaluation set if requested
    if args.create_sample:
        sample_file = "evalset.yaml"
        create_sample_evalset_file(sample_file)
        print(f"Created sample evaluation set: {sample_file}")
        return
    
    # Load evaluation set
    if args.evalset:
        query_set = load_evaluation_set(args.evalset)
    else:
        logger.info("No evaluation set specified, using built-in sample")
        query_set = create_sample_evaluation_set()
    
    # Initialize evaluator
    try:
        Config.validate()
        evaluator = RAGEvaluator()
    except Exception as e:
        logger.error(f"Failed to initialize evaluator: {e}")
        return
    
    # Run evaluation
    if args.comparative:
        # Comparative evaluation
        comparative_results = evaluator.run_comparative_evaluation(query_set)
        evaluator.print_comparative_results(comparative_results)
    else:
        # Single configuration evaluation
        results = evaluator.evaluate_query_set(
            query_set,
            use_reranking=not args.no_rerank,
            use_compression=not args.no_compress,
            alpha=args.alpha,
            top_k=args.top_k
        )
        
        # Print results
        evaluator.metrics.print_detailed_results(results)


if __name__ == "__main__":
    main()
