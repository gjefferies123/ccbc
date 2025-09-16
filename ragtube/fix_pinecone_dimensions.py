#!/usr/bin/env python3
"""Fix Pinecone index dimensions for Cohere embeddings."""

import os
from dotenv import load_dotenv
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

load_dotenv()

def fix_pinecone_index():
    """Delete and recreate index with correct dimensions."""
    
    api_key = os.getenv("PINECONE_API_KEY")
    index_name = os.getenv("PINECONE_INDEX", "christ-chapel-sermons")
    
    if not api_key:
        logger.error("❌ PINECONE_API_KEY not found")
        return False
    
    try:
        from pinecone import Pinecone, ServerlessSpec
        
        pc = Pinecone(api_key=api_key)
        logger.info("✅ Pinecone client connected")
        
        # Check existing indexes
        existing_indexes = [idx.name for idx in pc.list_indexes()]
        logger.info(f"📋 Existing indexes: {existing_indexes}")
        
        # Delete existing index if it exists
        if index_name in existing_indexes:
            logger.info(f"🗑️ Deleting existing index: {index_name}")
            pc.delete_index(index_name)
            logger.info(f"✅ Index {index_name} deleted")
            
            # Wait for deletion to complete
            import time
            logger.info("⏳ Waiting for deletion to complete...")
            time.sleep(10)
        
        # Create new index with correct dimensions
        logger.info(f"🔧 Creating new index: {index_name}")
        logger.info(f"📏 Dimensions: 1024 (Cohere embed-english-v3.0)")
        
        pc.create_index(
            name=index_name,
            dimension=1024,  # Cohere embed-english-v3.0 dimension
            metric="cosine",
            spec=ServerlessSpec(
                cloud="aws",
                region="us-west-2"
            )
        )
        
        logger.info(f"✅ Index {index_name} created successfully!")
        
        # Verify the new index
        index = pc.Index(index_name)
        stats = index.describe_index_stats()
        logger.info(f"📊 New index stats: {stats}")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Failed to fix index: {e}")
        return False

def main():
    print("🔧 Fixing Pinecone Index Dimensions")
    print("=" * 50)
    print("Problem: Existing index has 1536 dimensions (OpenAI)")
    print("Solution: Create new index with 1024 dimensions (Cohere)")
    print("=" * 50)
    
    success = fix_pinecone_index()
    
    if success:
        print("\n✅ INDEX FIXED!")
        print("🎯 Ready to run production indexer")
        print("💡 Run: py production_indexer.py")
    else:
        print("\n❌ Index fix failed!")

if __name__ == "__main__":
    main()
