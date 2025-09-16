#!/usr/bin/env python3
"""Test v4.0 search functionality with multiple queries."""

from christ_chapel_search import ChristChapelSearch

def test_v4_search():
    search = ChristChapelSearch()

    # Test different types of queries
    queries = [
        'How do I handle conflict with my spouse biblically?',
        'What does the Bible say about faith?',
        'How to pray effectively?',
        'What is walking by faith?',
        'How does God use wilderness experiences?'
    ]

    for query in queries:
        print(f'\n🔍 Query: {query}')
        result = search.search(query, top_k=2, min_score=0.2)
        if result['status'] == 'success':
            print(f'✅ Found {len(result["results"])} results')
            for i, r in enumerate(result['results'][:2]):
                print(f'  {i+1}. Score: {r["score"]} | {r["timestamp"]} | {r["text"][:80]}...')
        else:
            print(f'❌ Search failed: {result.get("error", "Unknown error")}')

if __name__ == "__main__":
    test_v4_search()
