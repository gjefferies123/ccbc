#!/usr/bin/env python3
"""Clear Pinecone index - working version."""

import os
from dotenv import load_dotenv

# Load environment
load_dotenv()

def clear_pinecone_index():
    """Clear the Pinecone index."""
    api_key = os.getenv("PINECONE_API_KEY")
    index_name = os.getenv("PINECONE_INDEX", "christ-chapel-sermons")
    
    if not api_key:
        print("❌ PINECONE_API_KEY not found")
        return False
    
    print(f"🔧 Clearing Pinecone index: {index_name}")
    print(f"🔑 Using API key: {api_key[:8]}...")
    
    try:
        # Try the new Pinecone client
        try:
            from pinecone import Pinecone
            pc = Pinecone(api_key=api_key)
            print("✅ Using new Pinecone client")
        except ImportError:
            print("❌ New Pinecone client not available, trying alternatives...")
            return False
        
        # List existing indexes
        try:
            indexes = pc.list_indexes()
            index_names = [idx.name for idx in indexes]
            print(f"📋 Existing indexes: {index_names}")
        except Exception as e:
            print(f"⚠️  Could not list indexes: {e}")
            index_names = [index_name]  # Assume it exists
        
        if index_name not in index_names:
            print(f"ℹ️  Index '{index_name}' doesn't exist yet - nothing to clear")
            return True
        
        # Connect to index
        index = pc.Index(index_name)
        
        # Get current stats
        try:
            stats = index.describe_index_stats()
            print(f"📊 Current index stats:")
            print(f"   - Total vectors: {stats.total_vector_count}")
            print(f"   - Dimension: {stats.dimension}")
            print(f"   - Index fullness: {stats.index_fullness}")
            
            if stats.total_vector_count == 0:
                print("✅ Index is already empty!")
                return True
                
        except Exception as e:
            print(f"⚠️  Could not get stats: {e}")
        
        # Clear the index
        print(f"🗑️  Clearing all vectors from index...")
        
        try:
            # Try the simple delete_all method
            index.delete(delete_all=True)
            print("✅ Index cleared successfully!")
            
            # Verify
            final_stats = index.describe_index_stats()
            print(f"📊 Final stats: {final_stats.total_vector_count} vectors remaining")
            
        except Exception as e:
            print(f"⚠️  delete_all failed: {e}")
            print("🔄 Trying alternative method...")
            
            # Alternative: Query and delete in batches
            try:
                # Get all vector IDs (limited approach)
                dummy_vector = [0.0] * 1024  # Assume 1024 dimensions
                query_response = index.query(
                    vector=dummy_vector,
                    top_k=10000,
                    include_metadata=False
                )
                
                if query_response.matches:
                    vector_ids = [match.id for match in query_response.matches]
                    print(f"🔍 Found {len(vector_ids)} vectors to delete")
                    
                    # Delete in batches
                    batch_size = 1000
                    for i in range(0, len(vector_ids), batch_size):
                        batch = vector_ids[i:i + batch_size]
                        index.delete(ids=batch)
                        print(f"🗑️  Deleted batch {i//batch_size + 1}")
                    
                    print("✅ Index cleared via batch deletion!")
                else:
                    print("ℹ️  No vectors found to delete")
                    
            except Exception as e2:
                print(f"❌ Alternative method also failed: {e2}")
                return False
        
        return True
        
    except Exception as e:
        print(f"❌ Error clearing Pinecone index: {e}")
        return False

if __name__ == "__main__":
    print("🚀 Clearing Pinecone Index")
    print("=" * 40)
    
    success = clear_pinecone_index()
    
    if success:
        print("✅ Index clearing completed!")
    else:
        print("❌ Index clearing failed!")
        exit(1)
