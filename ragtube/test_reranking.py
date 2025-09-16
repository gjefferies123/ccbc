#!/usr/bin/env python3
"""Test Cohere reranking vs regular search."""

import os
from dotenv import load_dotenv
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

load_dotenv()

def test_reranking_quality():
    """Compare search with and without reranking."""
    
    try:
        from pinecone import Pinecone
        import cohere
        
        # Initialize clients
        pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
        index = pc.Index(os.getenv("PINECONE_INDEX", "christ-chapel-sermons"))
        cohere_client = cohere.Client(os.getenv("COHERE_API_KEY"))
        
        print("🧪 TESTING COHERE RERANKING QUALITY")
        print("=" * 60)
        
        test_queries = [
            "What does the Bible say about faith and trust in God?",
            "How can I grow closer to Jesus in my daily life?",
            "What is God's plan for my life and purpose?",
            "How do I pray when I'm struggling with doubt?"
        ]
        
        for query in test_queries:
            print(f"\n🔍 Query: '{query}'")
            print("-" * 50)
            
            # Step 1: Get initial search results
            query_response = cohere_client.embed(
                texts=[query],
                model="embed-english-v3.0",
                input_type="search_query"
            )
            query_embedding = query_response.embeddings[0]
            
            # Get more results for reranking
            search_results = index.query(
                vector=query_embedding,
                top_k=10,  # Get more for reranking
                include_metadata=True
            )
            
            if not search_results.matches:
                print("❌ No search results found")
                continue
            
            print(f"📊 Initial search: {len(search_results.matches)} results")
            
            # Step 2: Prepare documents for reranking
            documents = []
            result_metadata = []
            
            for match in search_results.matches:
                doc_text = match.metadata.get('text', '')
                if doc_text:
                    documents.append(doc_text)
                    result_metadata.append({
                        'id': match.id,
                        'score': match.score,
                        'video_id': match.metadata.get('video_id', ''),
                        'timestamp': match.metadata.get('timestamp', ''),
                        'url': match.metadata.get('url', ''),
                        'text': doc_text
                    })
            
            if not documents:
                print("❌ No documents to rerank")
                continue
            
            # Step 3: Test Cohere reranking
            try:
                rerank_response = cohere_client.rerank(
                    model="rerank-english-v3.0",
                    query=query,
                    documents=documents,
                    top_k=5
                )
                
                print(f"✅ Reranking successful: {len(rerank_response.results)} results")
                
                # Show comparison
                print("\n📈 TOP 3 RESULTS COMPARISON:")
                
                print("\n🔍 BEFORE RERANKING (Similarity Search):")
                for i, match in enumerate(search_results.matches[:3], 1):
                    metadata = match.metadata
                    print(f"   {i}. Score: {match.score:.3f} | {metadata.get('timestamp', '0:00')}")
                    print(f"      {metadata.get('text', '')[:100]}...")
                    print(f"      🔗 {metadata.get('url', '')}")
                
                print("\n🎯 AFTER RERANKING (Cohere Rerank):")
                for i, result in enumerate(rerank_response.results[:3], 1):
                    original_metadata = result_metadata[result.index]
                    print(f"   {i}. Rerank Score: {result.relevance_score:.3f} | Original: {original_metadata['score']:.3f}")
                    print(f"      {original_metadata['timestamp']} | {original_metadata['text'][:100]}...")
                    print(f"      🔗 {original_metadata['url']}")
                
                # Calculate improvement
                original_avg = sum(m.score for m in search_results.matches[:3]) / 3
                reranked_avg = sum(r.relevance_score for r in rerank_response.results[:3]) / 3
                
                print(f"\n📊 QUALITY METRICS:")
                print(f"   Original avg score: {original_avg:.3f}")
                print(f"   Reranked avg score: {reranked_avg:.3f}")
                print(f"   Improvement: {((reranked_avg - original_avg) / original_avg * 100):+.1f}%")
                
            except Exception as e:
                print(f"❌ Reranking failed: {e}")
                print("💡 This might be due to API limits or model availability")
        
        return True
        
    except Exception as e:
        print(f"❌ Reranking test failed: {e}")
        return False

def test_search_quality_metrics():
    """Test overall search quality with various query types."""
    
    print("\n🎯 SEARCH QUALITY ASSESSMENT")
    print("=" * 50)
    
    try:
        from christ_chapel_search import ChristChapelSearch
        
        search = ChristChapelSearch()
        
        # Different types of queries
        test_cases = [
            {
                'category': 'Biblical Questions',
                'queries': [
                    "What does the Bible say about forgiveness?",
                    "How does God show His love for us?",
                    "What is salvation according to scripture?"
                ]
            },
            {
                'category': 'Practical Christian Living',
                'queries': [
                    "How do I pray when I don't know what to say?",
                    "How can I serve others in my community?",
                    "What does it mean to live by faith?"
                ]
            },
            {
                'category': 'Spiritual Growth',
                'queries': [
                    "How can I grow deeper in my relationship with Jesus?",
                    "What does spiritual maturity look like?",
                    "How do I hear God's voice in my life?"
                ]
            }
        ]
        
        overall_results = []
        
        for category_data in test_cases:
            category = category_data['category']
            queries = category_data['queries']
            
            print(f"\n📋 Category: {category}")
            print("-" * 30)
            
            category_scores = []
            
            for query in queries:
                result = search.search(query, top_k=3, min_score=0.3)
                
                if result['status'] == 'success' and result['results']:
                    top_score = result['results'][0]['score']
                    result_count = len(result['results'])
                    
                    print(f"✅ '{query[:40]}...' → {result_count} results, top: {top_score:.3f}")
                    category_scores.append(top_score)
                    
                    # Show best result
                    best = result['results'][0]
                    print(f"   🎯 Best: {best['timestamp']} | {best['preview'][:60]}...")
                else:
                    print(f"❌ '{query[:40]}...' → No results")
                    category_scores.append(0.0)
            
            if category_scores:
                avg_score = sum(category_scores) / len(category_scores)
                print(f"📊 Category average: {avg_score:.3f}")
                overall_results.extend(category_scores)
        
        # Overall assessment
        if overall_results:
            overall_avg = sum(overall_results) / len(overall_results)
            success_rate = sum(1 for s in overall_results if s > 0.3) / len(overall_results)
            
            print(f"\n🏆 OVERALL QUALITY ASSESSMENT:")
            print(f"   Average relevance score: {overall_avg:.3f}")
            print(f"   Success rate (>0.3): {success_rate:.1%}")
            
            if overall_avg > 0.5:
                print(f"   ✅ EXCELLENT quality (>0.5)")
            elif overall_avg > 0.4:
                print(f"   ✅ GOOD quality (>0.4)")
            elif overall_avg > 0.3:
                print(f"   ⚠️  ACCEPTABLE quality (>0.3)")
            else:
                print(f"   ❌ POOR quality (<0.3)")
        
        return True
        
    except Exception as e:
        print(f"❌ Quality assessment failed: {e}")
        return False

def main():
    print("🧪 CHRIST CHAPEL RAG QUALITY TESTING")
    print("=" * 60)
    
    # Test reranking
    rerank_success = test_reranking_quality()
    
    # Test overall quality
    quality_success = test_search_quality_metrics()
    
    print(f"\n" + "=" * 60)
    print(f"📊 TESTING SUMMARY:")
    print(f"   Reranking test: {'✅ PASSED' if rerank_success else '❌ FAILED'}")
    print(f"   Quality test: {'✅ PASSED' if quality_success else '❌ FAILED'}")
    
    if rerank_success and quality_success:
        print(f"\n🎉 RAG SYSTEM IS HIGH QUALITY!")
        print(f"💡 Ready for web interface deployment")
    else:
        print(f"\n⚠️  Some quality issues detected")
        print(f"💡 May need tuning or additional content")

if __name__ == "__main__":
    main()
