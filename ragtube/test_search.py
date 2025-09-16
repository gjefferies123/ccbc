#!/usr/bin/env python3
"""Test the search functionality."""

from christ_chapel_search import ChristChapelSearch

def test_search():
    print("🔍 Testing Christ Chapel Search...")
    
    search = ChristChapelSearch()
    result = search.search('How do I handle conflict with my spouse biblically?', top_k=3, min_score=0.3)
    
    print(f'Status: {result["status"]}')
    
    if result['status'] == 'success':
        print(f'✅ Found {len(result["results"])} results')
        for i, r in enumerate(result['results'][:2], 1):
            print(f'{i}. Score: {r["score"]} | {r["timestamp"]} | {r["text"][:100]}...')
        return True
    else:
        print(f'❌ Error: {result.get("error", "unknown")}')
        return False

if __name__ == "__main__":
    test_search()
