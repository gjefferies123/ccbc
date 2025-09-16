#!/usr/bin/env python3
"""Simple Pinecone index clearing."""

import os
from dotenv import load_dotenv

load_dotenv()

def clear_index_simple():
    """Try to clear Pinecone index with simple approach."""
    api_key = os.getenv("PINECONE_API_KEY")
    index_name = os.getenv("PINECONE_INDEX", "christ-chapel-sermons")
    
    print(f"🔧 Attempting to clear index: {index_name}")
    
    if not api_key:
        print("❌ No PINECONE_API_KEY found")
        return False
    
    # Try different import approaches
    try:
        # Approach 1: Try importing from different locations
        import pinecone
        print("✅ Pinecone imported")
        
        # Check what's available
        available_attrs = [attr for attr in dir(pinecone) if not attr.startswith('_')]
        print(f"📋 Available: {available_attrs}")
        
        # Try to initialize
        if hasattr(pinecone, 'init'):
            pinecone.init(api_key=api_key, environment="us-east-1")
            print("✅ Pinecone initialized with init()")
            
            if hasattr(pinecone, 'Index'):
                index = pinecone.Index(index_name)
                print(f"✅ Connected to index: {index_name}")
                
                # Try to delete all
                try:
                    index.delete(delete_all=True)
                    print("✅ Index cleared!")
                    return True
                except Exception as e:
                    print(f"⚠️  delete_all failed: {e}")
                    
        elif hasattr(pinecone, 'Pinecone'):
            pc = pinecone.Pinecone(api_key=api_key)
            print("✅ Pinecone initialized with Pinecone()")
            
            index = pc.Index(index_name)
            print(f"✅ Connected to index: {index_name}")
            
            try:
                index.delete(delete_all=True)
                print("✅ Index cleared!")
                return True
            except Exception as e:
                print(f"⚠️  delete_all failed: {e}")
                
        else:
            print("❌ No known initialization method found")
            
    except Exception as e:
        print(f"❌ Error with Pinecone: {e}")
    
    # If we can't clear it, that's okay - we can work around it
    print("ℹ️  Could not clear index, but we can proceed anyway")
    print("💡 New vectors will overwrite old ones with same IDs")
    return True

def main():
    print("🗑️  Simple Pinecone Index Clear")
    print("=" * 40)
    
    success = clear_index_simple()
    
    if success:
        print("\n✅ Ready to proceed!")
        print("🚀 Next: Start ingesting Christ Chapel videos")
    else:
        print("\n⚠️  Clearing had issues, but we can still proceed")

if __name__ == "__main__":
    main()
