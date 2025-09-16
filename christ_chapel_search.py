#!/usr/bin/env python3
"""Production-ready search interface for Christ Chapel BC sermons."""

import os
import json
from typing import List, Dict, Any, Optional
from dotenv import load_dotenv
import logging
import requests

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load environment
load_dotenv()

class ChristChapelSearch:
    """Production search interface for Christ Chapel BC sermons."""
    
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
            # Cohere (v2 used via HTTPS requests; client kept for future use if needed)
            self.cohere_client = None
            
            # Pinecone
            from pinecone import Pinecone
            self.pc = Pinecone(api_key=self.api_key)
            self.index = self.pc.Index(self.index_name)
            
            logger.info("✅ Search clients ready")
            
        except Exception as e:
            raise RuntimeError(f"Failed to initialize: {e}")
    
    def search(self, 
               query: str, 
               top_k: int = 5,
               min_score: float = 0.7,
               video_filter: Optional[str] = None,
               use_rerank: bool = True,
               candidate_k: int = 30) -> Dict[str, Any]:
        """
        Search Christ Chapel BC sermons.
        
        Args:
            query: Search query
            top_k: Number of results to return
            min_score: Minimum similarity score
            video_filter: Filter by specific video ID
        
        Returns:
            Dictionary with search results
        """
        
        logger.info(f"🔍 Searching: '{query[:50]}...'")
        
        try:
            # Create query embedding
            v2_url = "https://api.cohere.com/v2/embed"
            headers = {
                "Authorization": f"Bearer {self.cohere_api_key}",
                "Content-Type": "application/json"
            }
            payload = {
                "model": "embed-v4.0",
                "texts": [query],
                "input_type": "search_query",
                "embedding_types": ["float"]
            }
            resp = requests.post(v2_url, headers=headers, json=payload, timeout=30)
            if resp.status_code != 200:
                raise RuntimeError(f"Cohere v2 embed failed: {resp.status_code} {resp.text}")
            data = resp.json()
            query_embedding = data.get("embeddings", {}).get("float", [[]])[0]
            
            # Build filter if needed
            filter_dict = {}
            if video_filter:
                filter_dict['video_id'] = video_filter
            
            # Search (retrieve a larger candidate set for reranking)
            pc_top_k = max(candidate_k, top_k) if use_rerank else top_k
            results = self.index.query(
                vector=query_embedding,
                top_k=pc_top_k,
                include_metadata=True,
                filter=filter_dict if filter_dict else None
            )
            
            # Convert to interim list
            candidates: List[Dict[str, Any]] = []
            for match in results.matches:
                metadata = match.metadata or {}
                candidates.append({
                    'id': match.id,
                    'pinecone_score': float(getattr(match, 'score', 0.0) or 0.0),
                    'video_id': metadata.get('video_id', 'unknown'),
                    'timestamp': metadata.get('timestamp', '0:00'),
                    'start_sec': int(metadata.get('start_sec', 0)),
                    'end_sec': int(metadata.get('end_sec', 0)),
                    'duration': int(metadata.get('duration', 0)),
                    'url': metadata.get('url', ''),
                    'text': metadata.get('text', ''),
                    'video_title': metadata.get('video_title', 'Christ Chapel BC Sermon'),
                    'relevance': {
                        'contains_bible': metadata.get('contains_bible', False),
                        'contains_faith': metadata.get('contains_faith', False),
                        'contains_god': metadata.get('contains_god', False),
                        'contains_jesus': metadata.get('contains_jesus', False),
                        'contains_christ': metadata.get('contains_christ', False),
                        'contains_prayer': metadata.get('contains_prayer', False),
                    }
                })

            # Optional: Cohere v2 rerank
            if use_rerank and candidates:
                try:
                    rerank_url = "https://api.cohere.com/v2/rerank"
                    headers = {
                        "Authorization": f"Bearer {self.cohere_api_key}",
                        "Content-Type": "application/json"
                    }
                    documents = [c['text'] for c in candidates]
                    payload = {
                        "model": "rerank-v3.5",
                        "query": query,
                        "documents": documents,
                        "top_n": top_k
                    }
                    rr = requests.post(rerank_url, headers=headers, json=payload, timeout=30)
                    if rr.status_code == 200:
                        rr_data = rr.json()
                        # rr_data expected to contain 'results' with 'index' and 'relevance_score'
                        order: List[int] = []
                        scored: List[Dict[str, Any]] = []
                        for item in rr_data.get('results', []):
                            idx = int(item.get('index', 0))
                            score = float(item.get('relevance_score', 0.0))
                            if 0 <= idx < len(candidates):
                                entry = dict(candidates[idx])
                                entry['score'] = round(score, 3)
                                entry['rerank_model'] = 'rerank-v3.5'
                                scored.append(entry)
                        search_results = [r for r in scored if r.get('score', 0) >= min_score][:top_k]
                        logger.info(f"✅ Reranked with Cohere v2 (rerank-v3.5), kept {len(search_results)}")
                    else:
                        logger.warning(f"Rerank v2 failed {rr.status_code}: {rr.text}; falling back to Pinecone order")
                        # Fallback: Pinecone order
                        search_results = []
                        for c in candidates[:top_k]:
                            if c['pinecone_score'] >= min_score:
                                out = dict(c)
                                out['score'] = round(out.pop('pinecone_score', 0.0), 3)
                                out['preview'] = out['text'][:300] + '...'
                                search_results.append(out)
                except Exception as e:
                    logger.warning(f"Rerank error: {e}; falling back to Pinecone order")
                    search_results = []
                    for c in candidates[:top_k]:
                        if c['pinecone_score'] >= min_score:
                            out = dict(c)
                            out['score'] = round(out.pop('pinecone_score', 0.0), 3)
                            out['preview'] = out['text'][:300] + '...'
                            search_results.append(out)
            else:
                # No rerank; use Pinecone order
                search_results = []
                for c in candidates[:top_k]:
                    if c['pinecone_score'] >= min_score:
                        out = dict(c)
                        out['score'] = round(out.pop('pinecone_score', 0.0), 3)
                        out['preview'] = out['text'][:300] + '...'
                        search_results.append(out)
            
            logger.info(f"✅ Found {len(search_results)} relevant results")
            
            return {
                'query': query,
                'results': search_results,
                'total_results': len(search_results),
                'status': 'success',
                'search_params': {
                    'top_k': top_k,
                    'min_score': min_score,
                    'video_filter': video_filter
                }
            }
            
        except Exception as e:
            logger.error(f"❌ Search failed: {e}")
            return {
                'query': query,
                'results': [],
                'total_results': 0,
                'status': 'error',
                'error': str(e)
            }
    
    def get_video_summary(self, video_id: str) -> Dict[str, Any]:
        """Get summary information for a specific video."""
        try:
            # Search for all chunks from this video
            results = self.index.query(
                vector=[0.0] * 1024,  # Dummy vector
                top_k=100,
                include_metadata=True,
                filter={'video_id': video_id}
            )
            
            if not results.matches:
                return {'error': f'Video {video_id} not found'}
            
            # Extract metadata
            chunks = []
            total_duration = 0
            
            for match in results.matches:
                metadata = match.metadata
                chunks.append({
                    'start_sec': int(metadata.get('start_sec', 0)),
                    'end_sec': int(metadata.get('end_sec', 0)),
                    'timestamp': metadata.get('timestamp', '0:00'),
                    'text': metadata.get('text', ''),
                    'relevance': {
                        'contains_bible': metadata.get('contains_bible', False),
                        'contains_faith': metadata.get('contains_faith', False),
                        'contains_god': metadata.get('contains_god', False),
                        'contains_jesus': metadata.get('contains_jesus', False),
                        'contains_christ': metadata.get('contains_christ', False),
                        'contains_prayer': metadata.get('contains_prayer', False),
                    }
                })
                total_duration = max(total_duration, int(metadata.get('end_sec', 0)))
            
            # Sort by timestamp
            chunks.sort(key=lambda x: x['start_sec'])
            
            # Calculate content stats
            total_text = ' '.join([chunk['text'] for chunk in chunks])
            content_stats = {
                'total_chunks': len(chunks),
                'total_duration_sec': total_duration,
                'total_duration_min': round(total_duration / 60, 1),
                'total_words': len(total_text.split()),
                'contains_bible': sum(1 for c in chunks if c['relevance']['contains_bible']),
                'contains_faith': sum(1 for c in chunks if c['relevance']['contains_faith']),
                'contains_god': sum(1 for c in chunks if c['relevance']['contains_god']),
                'contains_jesus': sum(1 for c in chunks if c['relevance']['contains_jesus']),
                'contains_christ': sum(1 for c in chunks if c['relevance']['contains_christ']),
                'contains_prayer': sum(1 for c in chunks if c['relevance']['contains_prayer']),
            }
            
            return {
                'video_id': video_id,
                'url': f'https://youtu.be/{video_id}',
                'stats': content_stats,
                'chunks': chunks[:10],  # First 10 chunks as preview
                'status': 'success'
            }
            
        except Exception as e:
            return {'error': f'Failed to get video summary: {e}'}
    
    def get_index_stats(self) -> Dict[str, Any]:
        """Get overall index statistics."""
        try:
            stats = self.index.describe_index_stats()
            
            return {
                'total_vectors': stats.total_vector_count,
                'dimension': stats.dimension,
                'index_fullness': stats.index_fullness,
                'namespaces': dict(stats.namespaces),
                'status': 'success'
            }
        except Exception as e:
            return {'error': f'Failed to get stats: {e}'}


def run_demo_searches():
    """Run demonstration searches."""
    
    print("🏛️ Christ Chapel BC Sermon Search Demo")
    print("=" * 60)
    
    try:
        search = ChristChapelSearch()
        
        # Get index stats first
        stats = search.get_index_stats()
        if 'error' not in stats:
            print(f"📊 Index Stats: {stats['total_vectors']} vectors, {stats['dimension']}D")
            print()
        
        # Demo searches
        demo_queries = [
            ("What does the Bible say about faith?", 3),
            ("How can I grow spiritually?", 3),
            ("What is the gospel message?", 3),
            ("How do I pray effectively?", 2),
            ("What does it mean to follow Jesus?", 2),
            ("How can I find peace in difficult times?", 2),
        ]
        
        all_results = []
        
        for query, top_k in demo_queries:
            print(f"🔍 Query: '{query}'")
            
            result = search.search(query, top_k=top_k, min_score=0.3)
            
            if result['status'] == 'success' and result['results']:
                print(f"✅ Found {len(result['results'])} results:")
                
                for i, match in enumerate(result['results'], 1):
                    print(f"   {i}. Score: {match['score']} | {match['timestamp']} | {match['video_id']}")
                    print(f"      {match['preview']}")
                    print(f"      🔗 {match['url']}")
                    
                    # Show relevance
                    relevant = [k for k, v in match['relevance'].items() if v]
                    if relevant:
                        print(f"      📋 Contains: {', '.join(relevant)}")
                
                all_results.append({
                    'query': query,
                    'result_count': len(result['results']),
                    'top_score': result['results'][0]['score'] if result['results'] else 0,
                    'success': True
                })
            else:
                print(f"❌ No results found (min_score=0.3)")
                all_results.append({
                    'query': query,
                    'result_count': 0,
                    'top_score': 0,
                    'success': False
                })
            
            print()
        
        # Summary
        successful = sum(1 for r in all_results if r['success'])
        avg_score = sum(r['top_score'] for r in all_results if r['success']) / max(successful, 1)
        
        print("=" * 60)
        print(f"📊 DEMO SUMMARY:")
        print(f"   Successful queries: {successful}/{len(demo_queries)}")
        print(f"   Success rate: {successful/len(demo_queries)*100:.1f}%")
        print(f"   Average top score: {avg_score:.3f}")
        
        # Save demo results
        demo_summary = {
            'demo_results': all_results,
            'summary': {
                'total_queries': len(demo_queries),
                'successful_queries': successful,
                'success_rate': successful/len(demo_queries),
                'average_score': avg_score
            },
            'index_stats': stats
        }
        
        with open('christ_chapel_demo_results.json', 'w') as f:
            json.dump(demo_summary, f, indent=2)
        
        print(f"📄 Results saved: christ_chapel_demo_results.json")
        
        return True
        
    except Exception as e:
        print(f"❌ Demo failed: {e}")
        return False


def interactive_search():
    """Run interactive search mode."""
    
    print("\n🔍 Interactive Search Mode")
    print("Type your questions about Christ Chapel BC sermons")
    print("Type 'quit' to exit")
    print("=" * 50)
    
    try:
        search = ChristChapelSearch()
        
        while True:
            query = input("\n💬 Your question: ").strip()
            
            if query.lower() in ['quit', 'exit', 'q']:
                print("👋 Goodbye!")
                break
            
            if not query:
                continue
            
            print(f"\n🔍 Searching...")
            result = search.search(query, top_k=3, min_score=0.3)
            
            if result['status'] == 'success' and result['results']:
                print(f"✅ Found {len(result['results'])} results:\n")
                
                for i, match in enumerate(result['results'], 1):
                    print(f"{i}. 🎯 Score: {match['score']} | ⏰ {match['timestamp']}")
                    print(f"   📝 {match['preview']}")
                    print(f"   🔗 {match['url']}")
                    
                    relevant = [k.replace('contains_', '') for k, v in match['relevance'].items() if v]
                    if relevant:
                        print(f"   🏷️  {', '.join(relevant)}")
                    print()
            else:
                print("❌ No relevant results found. Try rephrasing your question.")
    
    except KeyboardInterrupt:
        print("\n👋 Goodbye!")
    except Exception as e:
        print(f"❌ Error: {e}")


def main():
    """Main function."""
    
    print("🏛️ CHRIST CHAPEL BC SERMON SEARCH")
    print("=" * 50)
    print("Choose an option:")
    print("1. Run demo searches")
    print("2. Interactive search")
    print("3. Both")
    
    choice = input("\nYour choice (1/2/3): ").strip()
    
    if choice in ['1', '3']:
        success = run_demo_searches()
        if not success:
            return
    
    if choice in ['2', '3']:
        interactive_search()


if __name__ == "__main__":
    main()
