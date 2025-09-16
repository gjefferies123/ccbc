#!/usr/bin/env python3
"""Diagnose Pinecone index issues."""

import os
from dotenv import load_dotenv
import logging
import time

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

load_dotenv()

def diagnose_index():
    """Diagnose the Pinecone index."""
    
    try:
        from pinecone import Pinecone
        
        pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
        index_name = os.getenv("PINECONE_INDEX", "christ-chapel-sermons")
        
        index = pc.Index(index_name)
        
        print("🔍 PINECONE INDEX DIAGNOSIS")
        print("=" * 50)
        
        # Get stats
        stats = index.describe_index_stats()
        print(f"📊 Index Stats:")
        print(f"   Total vectors: {stats.total_vector_count}")
        print(f"   Dimension: {stats.dimension}")
        print(f"   Index fullness: {stats.index_fullness}")
        print(f"   Namespaces: {stats.namespaces}")
        
        # Try a simple upsert test
        print("\n🧪 Testing simple upsert...")
        test_vector = {
            'id': 'test-vector-1',
            'values': [0.1] * 1024,  # 1024-dim test vector
            'metadata': {'test': 'true', 'content': 'This is a test vector'}
        }
        
        index.upsert(vectors=[test_vector])
        print("✅ Test upsert successful")
        
        # Wait a moment for indexing
        print("⏳ Waiting for indexing...")
        time.sleep(5)
        
        # Check stats again
        stats2 = index.describe_index_stats()
        print(f"\n📊 Updated Stats:")
        print(f"   Total vectors: {stats2.total_vector_count}")
        
        # Try to fetch the test vector
        print("\n🔍 Testing fetch...")
        try:
            fetch_result = index.fetch(ids=['test-vector-1'])
            if 'test-vector-1' in fetch_result.vectors:
                print("✅ Test vector found!")
                vector_data = fetch_result.vectors['test-vector-1']
                print(f"   Metadata: {vector_data.metadata}")
            else:
                print("❌ Test vector not found in fetch")
        except Exception as e:
            print(f"❌ Fetch failed: {e}")
        
        # Try a simple query
        print("\n🔍 Testing query...")
        try:
            query_result = index.query(
                vector=[0.1] * 1024,
                top_k=1,
                include_metadata=True
            )
            
            if query_result.matches:
                print(f"✅ Query found {len(query_result.matches)} results")
                for match in query_result.matches:
                    print(f"   ID: {match.id}, Score: {match.score}")
                    print(f"   Metadata: {match.metadata}")
            else:
                print("❌ Query returned no results")
                
        except Exception as e:
            print(f"❌ Query failed: {e}")
        
        # Clean up test vector
        print("\n🗑️ Cleaning up test vector...")
        try:
            index.delete(ids=['test-vector-1'])
            print("✅ Test vector deleted")
        except Exception as e:
            print(f"⚠️ Delete failed: {e}")
        
        return stats2.total_vector_count > 0
        
    except Exception as e:
        logger.error(f"❌ Diagnosis failed: {e}")
        return False

def check_christ_chapel_vectors():
    """Check if Christ Chapel vectors are in the index."""
    
    try:
        from pinecone import Pinecone
        
        pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
        index_name = os.getenv("PINECONE_INDEX", "christ-chapel-sermons")
        
        index = pc.Index(index_name)
        
        print("\n🏛️ CHECKING CHRIST CHAPEL VECTORS")
        print("=" * 50)
        
        # Try to fetch some specific Christ Chapel vector IDs
        test_ids = [
            'Jz1Zb57NUMg_chunk_0',
            'Jz1Zb57NUMg_chunk_1', 
            '6_HgIPUXpVM_chunk_0'
        ]
        
        for vector_id in test_ids:
            try:
                fetch_result = index.fetch(ids=[vector_id])
                if vector_id in fetch_result.vectors:
                    print(f"✅ Found: {vector_id}")
                    vector_data = fetch_result.vectors[vector_id]
                    metadata = vector_data.metadata
                    print(f"   Video: {metadata.get('video_id', 'unknown')}")
                    print(f"   Text: {metadata.get('text', '')[:100]}...")
                else:
                    print(f"❌ Not found: {vector_id}")
            except Exception as e:
                print(f"❌ Error checking {vector_id}: {e}")
        
        # Try a semantic query with Christ Chapel content
        print(f"\n🔍 Testing Christ Chapel query...")
        
        try:
            import cohere
            cohere_client = cohere.Client(os.getenv("COHERE_API_KEY"))
            
            query_text = "Christ Chapel good morning"
            query_response = cohere_client.embed(
                texts=[query_text],
                model="embed-english-v3.0",
                input_type="search_query"
            )
            
            query_embedding = query_response.embeddings[0]
            
            results = index.query(
                vector=query_embedding,
                top_k=5,
                include_metadata=True
            )
            
            if results.matches:
                print(f"✅ Query found {len(results.matches)} results")
                for i, match in enumerate(results.matches, 1):
                    print(f"   {i}. Score: {match.score:.3f} | ID: {match.id}")
                    print(f"      Text: {match.metadata.get('text', '')[:100]}...")
            else:
                print("❌ No results for Christ Chapel query")
                
        except Exception as e:
            print(f"❌ Christ Chapel query failed: {e}")
            
    except Exception as e:
        print(f"❌ Christ Chapel check failed: {e}")

def main():
    print("🔍 Pinecone Diagnostic Tool")
    print("=" * 30)
    
    # Basic diagnosis
    basic_working = diagnose_index()
    
    # Check Christ Chapel data
    check_christ_chapel_vectors()
    
    print("\n" + "=" * 50)
    if basic_working:
        print("✅ Pinecone index is working")
        print("💡 If searches aren't working, it may be a timing or API issue")
    else:
        print("❌ Pinecone index has issues")
        print("💡 May need to recreate index or check API keys")

if __name__ == "__main__":
    main()
