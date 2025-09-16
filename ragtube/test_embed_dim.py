#!/usr/bin/env python3
"""Test Cohere v2 embed dimension."""

import requests
import os
from dotenv import load_dotenv

load_dotenv()

def test_embed_dimension():
    url = 'https://api.cohere.com/v2/embed'
    headers = {
        'Authorization': f'Bearer {os.getenv("COHERE_API_KEY")}',
        'Content-Type': 'application/json'
    }
    payload = {
        'model': 'embed-english-v3.0',
        'texts': ['test'],
        'input_type': 'search_query',
        'embedding_types': ['float']
    }
    
    resp = requests.post(url, headers=headers, json=payload, timeout=30)
    if resp.status_code == 200:
        data = resp.json()
        emb = data.get('embeddings', {}).get('float', [[]])[0]
        print(f'✅ embed-v4.0 dimension: {len(emb)}')
        return len(emb)
    else:
        print(f'❌ Error: {resp.status_code} {resp.text}')
        return None

if __name__ == "__main__":
    test_embed_dimension()
