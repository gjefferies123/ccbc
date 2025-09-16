#!/usr/bin/env python3
"""
Test search quality with the expanded knowledge base (20 videos, 913 chunks).
"""

import os
import sys
from dotenv import load_dotenv

# Add the parent directory to the path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ragtube.enhanced_wisdom_app import EnhancedWisdomApp

# Load environment variables
load_dotenv()

def test_search_quality():
    """Test search quality with various queries."""
    print("🧪 Testing search quality with expanded knowledge base...")
    
    try:
        # Initialize the app
        app = EnhancedWisdomApp()
        print("✅ Enhanced Wisdom App initialized")
        
        # Test queries
        test_queries = [
            "How can I find peace in difficult times?",
            "What does the Bible say about humility?",
            "How should I handle spiritual battles?",
            "What is the importance of faith?",
            "How does God use wilderness experiences?",
            "What does it mean to walk by faith?",
            "How can I prepare my heart to hear God's word?",
            "What is the significance of the Passover?",
            "How do I avoid spiritual plagues?",
            "What does it mean to have unshakeable truth?"
        ]
        
        for i, query in enumerate(test_queries, 1):
            print(f"\n🔍 Test Query {i}: {query}")
            print("=" * 60)
            
            try:
                # Get search results
                search_results = app.search_engine.search(query, top_k=5)
                results = search_results.get('results', [])
                
                if results:
                    print(f"✅ Found {len(results)} results")
                    for j, result in enumerate(results, 1):
                        print(f"\n📹 Result {j}:")
                        print(f"   🎥 Video: {result['video_title']}")
                        print(f"   ⏰ Time: {result['timestamp']}")
                        print(f"   📝 Text: {result['text'][:200]}...")
                        print(f"   🔗 URL: {result['url']}")
                else:
                    print("❌ No results found")
                    
            except Exception as e:
                print(f"❌ Error with query: {e}")
        
        print(f"\n🎉 Search quality testing complete!")
        
    except Exception as e:
        print(f"❌ Failed to initialize app: {e}")

if __name__ == "__main__":
    test_search_quality()
