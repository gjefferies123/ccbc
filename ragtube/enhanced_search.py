#!/usr/bin/env python3
"""Enhanced search with proper Cohere reranking."""

import os
from typing import List, Dict, Any, Optional
from dotenv import load_dotenv
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

load_dotenv()

class EnhancedChristChapelSearch:
    """Enhanced search with Cohere reranking."""
    
    def __init__(self):
        self.api_key = os.getenv("PINECONE_API_KEY")
        self.index_name = os.getenv("PINECONE_INDEX", "christ-chapel-sermons")
        self.cohere_api_key = os.getenv("COHERE_API_KEY")
        
        self.pc = None
        self.index = None
        self.cohere_client = None
        
        self._init_clients()
    
    def _init_clients(self):
        """Initialize clients."""
        try:
            # Cohere
            import cohere
            self.cohere_client = cohere.Client(self.cohere_api_key)
            
            # Pinecone
            from pinecone import Pinecone
            self.pc = Pinecone(api_key=self.api_key)
            self.index = self.pc.Index(self.index_name)
            
            logger.info("✅ Enhanced search clients ready")
            
        except Exception as e:
            raise RuntimeError(f"Failed to initialize: {e}")
    
    def search_with_reranking(self, 
                            query: str, 
                            top_k: int = 5,
                            retrieval_k: int = 20,
                            min_score: float = 0.25,
                            use_reranking: bool = True) -> Dict[str, Any]:
        """
        Enhanced search with Cohere reranking.
        
        Args:
            query: Search query
            top_k: Final number of results to return
            retrieval_k: Number of results to retrieve before reranking
            min_score: Minimum similarity score
            use_reranking: Whether to use Cohere reranking
        """
        
        logger.info(f"🔍 Enhanced search: '{query[:50]}...' (rerank: {use_reranking})")
        
        try:
            # Step 1: Create query embedding
            response = self.cohere_client.embed(
                texts=[query],
                model="embed-english-v3.0",
                input_type="search_query"
            )
            query_embedding = response.embeddings[0]
            
            # Step 2: Initial retrieval (get more results for reranking)
            search_k = retrieval_k if use_reranking else top_k
            results = self.index.query(
                vector=query_embedding,
                top_k=search_k,
                include_metadata=True
            )
            
            if not results.matches:
                return {
                    'query': query,
                    'results': [],
                    'total_results': 0,
                    'status': 'success',
                    'reranking_used': False
                }
            
            # Step 3: Prepare results for potential reranking
            search_results = []
            documents_for_rerank = []
            
            for match in results.matches:
                metadata = match.metadata
                text = metadata.get('text', '')
                
                result_item = {
                    'id': match.id,
                    'original_score': round(match.score, 3),
                    'rerank_score': None,
                    'video_id': metadata.get('video_id', 'unknown'),
                    'timestamp': metadata.get('timestamp', '0:00'),
                    'start_sec': int(metadata.get('start_sec', 0)),
                    'end_sec': int(metadata.get('end_sec', 0)),
                    'duration': int(metadata.get('duration', 0)),
                    'url': metadata.get('url', ''),
                    'text': text,
                    'preview': text[:300] + '...' if len(text) > 300 else text,
                    'video_title': metadata.get('video_title', 'Christ Chapel BC Sermon'),
                    'relevance': {
                        'contains_bible': metadata.get('contains_bible', False),
                        'contains_faith': metadata.get('contains_faith', False),
                        'contains_god': metadata.get('contains_god', False),
                        'contains_jesus': metadata.get('contains_jesus', False),
                        'contains_christ': metadata.get('contains_christ', False),
                        'contains_prayer': metadata.get('contains_prayer', False),
                    }
                }
                
                search_results.append(result_item)
                if text:  # Only add non-empty texts for reranking
                    documents_for_rerank.append(text)
            
            # Step 4: Apply reranking if requested and available
            reranking_used = False
            final_results = search_results
            
            if use_reranking and documents_for_rerank:
                try:
                    # Use correct parameter name: top_n instead of top_k
                    rerank_response = self.cohere_client.rerank(
                        model="rerank-english-v3.0",
                        query=query,
                        documents=documents_for_rerank,
                        top_n=min(top_k, len(documents_for_rerank))
                    )
                    
                    # Apply rerank scores and reorder
                    reranked_results = []
                    for rerank_result in rerank_response.results:
                        original_index = rerank_result.index
                        if original_index < len(search_results):
                            result = search_results[original_index].copy()
                            result['rerank_score'] = round(rerank_result.relevance_score, 3)
                            result['score'] = result['rerank_score']  # Use rerank score as primary
                            reranked_results.append(result)
                    
                    final_results = reranked_results
                    reranking_used = True
                    logger.info(f"✅ Reranking applied: {len(final_results)} results")
                    
                except Exception as e:
                    logger.warning(f"⚠️ Reranking failed: {e}, using original order")
                    final_results = search_results[:top_k]
            else:
                final_results = search_results[:top_k]
            
            # Step 5: Filter by minimum score
            filtered_results = []
            for result in final_results:
                score = result.get('rerank_score') or result.get('original_score', 0)
                if score >= min_score:
                    filtered_results.append(result)
            
            logger.info(f"✅ Found {len(filtered_results)} relevant results")
            
            return {
                'query': query,
                'results': filtered_results,
                'total_results': len(filtered_results),
                'status': 'success',
                'reranking_used': reranking_used,
                'search_params': {
                    'top_k': top_k,
                    'retrieval_k': retrieval_k,
                    'min_score': min_score,
                    'use_reranking': use_reranking
                }
            }
            
        except Exception as e:
            logger.error(f"❌ Enhanced search failed: {e}")
            return {
                'query': query,
                'results': [],
                'total_results': 0,
                'status': 'error',
                'error': str(e),
                'reranking_used': False
            }

def compare_search_methods(query: str):
    """Compare search with and without reranking."""
    
    print(f"🔍 COMPARING SEARCH METHODS")
    print(f"Query: '{query}'")
    print("=" * 60)
    
    try:
        search = EnhancedChristChapelSearch()
        
        # Search without reranking
        print("📊 WITHOUT RERANKING (Similarity Only):")
        no_rerank = search.search_with_reranking(
            query=query,
            top_k=5,
            use_reranking=False
        )
        
        if no_rerank['status'] == 'success' and no_rerank['results']:
            for i, result in enumerate(no_rerank['results'], 1):
                print(f"   {i}. Score: {result['original_score']} | {result['timestamp']}")
                print(f"      {result['preview']}")
                print()
        else:
            print("   No results found")
        
        # Search with reranking
        print("🎯 WITH COHERE RERANKING:")
        with_rerank = search.search_with_reranking(
            query=query,
            top_k=5,
            use_reranking=True
        )
        
        if with_rerank['status'] == 'success' and with_rerank['results']:
            for i, result in enumerate(with_rerank['results'], 1):
                rerank_score = result.get('rerank_score', 'N/A')
                orig_score = result['original_score']
                print(f"   {i}. Rerank: {rerank_score} | Original: {orig_score} | {result['timestamp']}")
                print(f"      {result['preview']}")
                print()
            
            print(f"✅ Reranking used: {with_rerank['reranking_used']}")
        else:
            print("   No results found")
            
        return True
        
    except Exception as e:
        print(f"❌ Comparison failed: {e}")
        return False

def main():
    """Test enhanced search."""
    
    print("🎯 ENHANCED CHRIST CHAPEL SEARCH")
    print("=" * 50)
    
    test_queries = [
        "What does the Bible say about faith and trusting God?",
        "How can I grow closer to Jesus in my daily walk?",
        "What is God's plan and purpose for my life?",
        "How do I pray when I'm struggling with doubt?"
    ]
    
    for query in test_queries:
        print()
        success = compare_search_methods(query)
        if not success:
            break
        print("-" * 60)
    
    print("\n🎉 Enhanced search testing complete!")
    print("💡 The web interface can now use enhanced search with reranking")

if __name__ == "__main__":
    main()
