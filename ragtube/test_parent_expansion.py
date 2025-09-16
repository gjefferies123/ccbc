#!/usr/bin/env python3
"""
Test parent expansion feature with the new indexing.
"""

import os
import sys
from dotenv import load_dotenv

# Add the parent directory to the path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ragtube.enhanced_wisdom_app import EnhancedWisdomApp

# Load environment variables
load_dotenv()

def test_parent_expansion():
    """Test parent expansion feature."""
    print("🧪 Testing parent expansion feature...")
    
    try:
        # Initialize the app
        app = EnhancedWisdomApp()
        print("✅ Enhanced Wisdom App initialized")
        
        # Test query that should trigger parent expansion
        test_query = "What does the Bible say about humility?"
        print(f"\n🔍 Test Query: {test_query}")
        print("=" * 60)
        
        # Get biblical guidance (this should trigger parent expansion)
        result = app.get_biblical_guidance(test_query)
        
        if result['status'] == 'success':
            print(f"✅ Guidance generated successfully!")
            print(f"📊 Context Info:")
            print(f"   📹 Original chunks found: {result['context_info']['original_chunks']}")
            print(f"   📹 Videos expanded: {result['context_info']['expanded_videos']}")
            print(f"   📝 Total chunks used: {result['context_info']['total_chunks_used']}")
            
            print(f"\n📖 Guidance Preview:")
            print(f"{result['guidance'][:300]}...")
            
            print(f"\n📹 Sources:")
            for i, source in enumerate(result['sources'][:3], 1):
                print(f"   {i}. {source['video_title']} - {source['timestamp']}")
                print(f"      {source.get('text', 'No text available')[:100]}...")
                print(f"      🔗 {source.get('url', 'No URL available')}")
        else:
            print(f"❌ No results found")
        
        # Test direct parent expansion
        print(f"\n🔍 Testing direct parent expansion...")
        search_results = app.search_engine.search(test_query, top_k=3)
        
        if search_results['status'] == 'success' and search_results['results']:
            print(f"✅ Found {len(search_results['results'])} initial chunks")
            
            # Test expansion
            expanded_context = app._expand_to_full_videos(search_results['results'])
            
            print(f"📊 Expansion Results:")
            for video_id, video_info in expanded_context['videos'].items():
                print(f"   📹 {video_id}: {video_info['video_title']}")
                print(f"      🎯 Trigger chunks: {len(video_info['trigger_chunks'])}")
                print(f"      📝 Total chunks: {video_info['total_chunks']}")
                print(f"      📈 Expansion ratio: {video_info['total_chunks'] / len(video_info['trigger_chunks']):.1f}x")
        else:
            print(f"❌ No search results found")
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_parent_expansion()
