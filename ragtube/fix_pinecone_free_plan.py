#!/usr/bin/env python3
"""Fix Pinecone index for free plan."""

import os
from dotenv import load_dotenv
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

load_dotenv()

def fix_pinecone_free_plan():
    """Create index compatible with free plan."""
    
    api_key = os.getenv("PINECONE_API_KEY")
    index_name = os.getenv("PINECONE_INDEX", "christ-chapel-sermons")
    
    try:
        from pinecone import Pinecone, ServerlessSpec
        
        pc = Pinecone(api_key=api_key)
        logger.info("✅ Pinecone client connected")
        
        # Check existing indexes
        existing = [idx.name for idx in pc.list_indexes()]
        logger.info(f"📋 Existing: {existing}")
        
        # Delete if exists
        if index_name in existing:
            logger.info(f"🗑️ Deleting: {index_name}")
            pc.delete_index(index_name)
            import time
            time.sleep(15)  # Wait longer for deletion
            logger.info("✅ Deleted")
        
        # Create with free plan compatible settings
        logger.info(f"🔧 Creating index with free plan settings...")
        
        pc.create_index(
            name=index_name,
            dimension=1024,
            metric="cosine",
            spec=ServerlessSpec(
                cloud="aws",
                region="us-east-1"  # Free plan supported region
            )
        )
        
        logger.info(f"✅ Index created successfully!")
        
        # Test connection
        index = pc.Index(index_name)
        stats = index.describe_index_stats()
        logger.info(f"📊 Stats: {stats}")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Failed: {e}")
        
        # Try alternative approach for free plan
        try:
            logger.info("🔄 Trying pod-based index for free plan...")
            
            pc.create_index(
                name=index_name,
                dimension=1024,
                metric="cosine"
                # Default pod configuration for free plan
            )
            
            logger.info("✅ Pod-based index created!")
            return True
            
        except Exception as e2:
            logger.error(f"❌ Alternative also failed: {e2}")
            return False

def main():
    print("🔧 Fixing Pinecone for Free Plan")
    print("=" * 40)
    
    success = fix_pinecone_free_plan()
    
    if success:
        print("\n✅ PINECONE FIXED!")
        print("🎯 Index ready for Cohere embeddings (1024D)")
        print("💡 Run: py production_indexer.py")
    else:
        print("\n❌ Still having issues with Pinecone")
        print("💡 May need to check free plan limitations")

if __name__ == "__main__":
    main()
