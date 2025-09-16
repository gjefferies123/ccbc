#!/usr/bin/env python3
"""
Test YouTube links and titles across multiple videos.
"""

import os
import sys
from dotenv import load_dotenv

# Add the parent directory to the path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ragtube.enhanced_wisdom_app import EnhancedWisdomApp

# Load environment variables
load_dotenv()

def test_multiple_videos():
    """Test YouTube links and titles across multiple videos."""
    print("🔍 Testing YouTube links and titles across multiple videos...")
    
    try:
        # Initialize the app
        app = EnhancedWisdomApp()
        print("✅ Enhanced Wisdom App initialized")
        
        # Test queries that should hit different videos
        test_queries = [
            "What does the Bible say about humility?",
            "How does God use wilderness experiences?",
            "What is the significance of the Passover?",
            "How do I walk by faith?",
            "What does it mean to prepare my heart?"
        ]
        
        for query in test_queries:
            print(f"\n🔍 Query: {query}")
            print("=" * 60)
            
            # Get search results
            search_results = app.search_engine.search(query, top_k=3)
            
            if search_results['status'] == 'success' and search_results['results']:
                print(f"✅ Found {len(search_results['results'])} results")
                
                for i, result in enumerate(search_results['results'], 1):
                    video_id = result.get('video_id', 'MISSING')
                    title = result.get('video_title', 'MISSING')
                    url = result.get('url', 'MISSING')
                    timestamp = result.get('timestamp', 'MISSING')
                    
                    print(f"\n📹 Result {i}:")
                    print(f"   🆔 Video ID: {video_id}")
                    print(f"   📝 Title: {title}")
                    print(f"   ⏰ Timestamp: {timestamp}")
                    print(f"   🔗 URL: {url}")
                    
                    # Verify URL construction
                    if video_id != 'MISSING' and url != 'MISSING':
                        expected_base = f"https://www.youtube.com/watch?v={video_id}"
                        if expected_base in url:
                            print(f"   ✅ URL construction correct")
                        else:
                            print(f"   ❌ URL construction issue")
                            print(f"      Expected: {expected_base}")
                            print(f"      Actual: {url}")
                    
                    # Check if title looks reasonable
                    if title != 'MISSING' and len(title) > 10:
                        print(f"   ✅ Title looks reasonable")
                    else:
                        print(f"   ❌ Title issue: {title}")
            else:
                print(f"❌ No results found")
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_multiple_videos()
