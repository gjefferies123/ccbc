#!/usr/bin/env python3
"""Script to clear the Pinecone index."""

import os
from dotenv import load_dotenv
from pinecone import Pinecone

# Load environment variables
load_dotenv()

def clear_pinecone_index():
    """Clear all data from the Pinecone index."""
    api_key = os.getenv("PINECONE_API_KEY")
    index_name = os.getenv("PINECONE_INDEX", "christ-chapel-sermons")
    
    if not api_key:
        print("❌ PINECONE_API_KEY not found in environment")
        return False
    
    try:
        # Initialize Pinecone
        pc = Pinecone(api_key=api_key)
        
        # Check if index exists
        existing_indexes = [index.name for index in pc.list_indexes()]
        print(f"📋 Existing indexes: {existing_indexes}")
        
        if index_name not in existing_indexes:
            print(f"ℹ️  Index '{index_name}' does not exist yet. Nothing to clear.")
            return True
        
        # Get index stats before clearing
        index = pc.Index(index_name)
        stats = index.describe_index_stats()
        
        print(f"📊 Current index stats:")
        print(f"   - Total vectors: {stats.total_vector_count}")
        print(f"   - Dimension: {stats.dimension}")
        print(f"   - Index fullness: {stats.index_fullness}")
        
        if stats.total_vector_count == 0:
            print("✅ Index is already empty!")
            return True
        
        # Clear the index by deleting all vectors
        print(f"🗑️  Clearing all vectors from index '{index_name}'...")
        
        # Delete all vectors (delete_all is available in newer versions)
        try:
            index.delete(delete_all=True)
            print("✅ Index cleared successfully using delete_all!")
        except Exception as e:
            print(f"⚠️  delete_all failed: {e}")
            print("🔄 Trying alternative method...")
            
            # Alternative: query for all IDs and delete in batches
            # This is a workaround for older Pinecone versions
            try:
                # Query to get all vector IDs
                query_response = index.query(
                    vector=[0.0] * stats.dimension,
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
                        print(f"🗑️  Deleted batch {i//batch_size + 1}/{(len(vector_ids) + batch_size - 1)//batch_size}")
                    
                    print("✅ Index cleared successfully!")
                else:
                    print("✅ No vectors found to delete!")
                    
            except Exception as e2:
                print(f"❌ Failed to clear index: {e2}")
                return False
        
        # Verify the index is cleared
        final_stats = index.describe_index_stats()
        print(f"📊 Final index stats:")
        print(f"   - Total vectors: {final_stats.total_vector_count}")
        
        return True
        
    except Exception as e:
        print(f"❌ Error accessing Pinecone: {e}")
        return False

if __name__ == "__main__":
    print("🚀 Starting Pinecone index cleanup...")
    success = clear_pinecone_index()
    
    if success:
        print("✅ Index cleanup completed!")
    else:
        print("❌ Index cleanup failed!")
        exit(1)
