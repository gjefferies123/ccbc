#!/usr/bin/env python3
"""
Debug YouTube link generation and title matching issues.
"""

import os
import sys
from dotenv import load_dotenv

# Add the parent directory to the path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ragtube.enhanced_wisdom_app import EnhancedWisdomApp

# Load environment variables
load_dotenv()

def debug_youtube_links():
    """Debug YouTube link generation and title matching."""
    print("🔍 Debugging YouTube link generation...")
    
    try:
        # Initialize the app
        app = EnhancedWisdomApp()
        print("✅ Enhanced Wisdom App initialized")
        
        # Test query
        test_query = "What does the Bible say about humility?"
        print(f"\n🔍 Test Query: {test_query}")
        print("=" * 60)
        
        # Get search results
        search_results = app.search_engine.search(test_query, top_k=5)
        
        if search_results['status'] == 'success' and search_results['results']:
            print(f"✅ Found {len(search_results['results'])} results")
            
            for i, result in enumerate(search_results['results'], 1):
                print(f"\n📹 Result {i}:")
                print(f"   🆔 Video ID: {result.get('video_id', 'MISSING')}")
                print(f"   📝 Title: {result.get('video_title', 'MISSING')}")
                print(f"   ⏰ Timestamp: {result.get('timestamp', 'MISSING')}")
                print(f"   🔗 URL: {result.get('url', 'MISSING')}")
                print(f"   📊 Score: {result.get('score', 'MISSING')}")
                print(f"   📝 Text Preview: {result.get('text', 'MISSING')[:100]}...")
                
                # Check if URL construction is correct
                video_id = result.get('video_id')
                if video_id:
                    expected_url = f"https://www.youtube.com/watch?v={video_id}"
                    actual_url = result.get('url', '')
                    if expected_url in actual_url:
                        print(f"   ✅ URL construction looks correct")
                    else:
                        print(f"   ❌ URL construction issue - Expected: {expected_url}")
                        print(f"   ❌ Actual: {actual_url}")
        
        # Test parent expansion to see all chunks for a video
        print(f"\n🔍 Testing parent expansion for video chunks...")
        expanded_context = app._expand_to_full_videos(search_results['results'])
        
        for video_id, video_info in expanded_context['videos'].items():
            print(f"\n📹 Video: {video_id}")
            print(f"   📝 Title: {video_info['video_title']}")
            print(f"   📊 Total chunks: {video_info['total_chunks']}")
            
            # Check first few chunks
            for i, chunk in enumerate(video_info['chunks'][:3], 1):
                print(f"   📝 Chunk {i}:")
                print(f"      🆔 ID: {chunk.get('id', 'MISSING')}")
                print(f"      📝 Title: {chunk.get('video_title', 'MISSING')}")
                print(f"      ⏰ Timestamp: {chunk.get('timestamp', 'MISSING')}")
                print(f"      🔗 URL: {chunk.get('url', 'MISSING')}")
                print(f"      📊 Index: {chunk.get('chunk_index', 'MISSING')}")
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    debug_youtube_links()
