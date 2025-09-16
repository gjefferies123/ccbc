#!/usr/bin/env python3
"""Test Cohere v4.0 embedding API response structure in detail."""

import requests
import os
from dotenv import load_dotenv

load_dotenv()

def test_embed_structure():
    url = 'https://api.cohere.com/v2/embed'
    headers = {
        'Authorization': f'Bearer {os.getenv("COHERE_API_KEY")}',
        'Content-Type': 'application/json'
    }
    payload = {
        'model': 'embed-v4.0',
        'texts': ['test'],
        'input_type': 'search_document',
        'embedding_types': ['float']
    }

    try:
        response = requests.post(url, headers=headers, json=payload, timeout=30)
        print('Status:', response.status_code)
        data = response.json()
        print('Response structure:')
        print(f"  id: {data['id']}")
        print(f"  texts: {data['texts']}")
        print(f"  embeddings type: {type(data['embeddings'])}")
        print(f"  embeddings keys: {list(data['embeddings'].keys())}")
        print(f"  meta: {data['meta']}")
        print(f"  response_type: {data['response_type']}")
        
        # Check the actual embedding structure
        if 'float' in data['embeddings']:
            print(f"  float embeddings type: {type(data['embeddings']['float'])}")
            print(f"  float embeddings length: {len(data['embeddings']['float'])}")
            print(f"  first float embedding: {data['embeddings']['float'][0][:5]}...")
        
    except Exception as e:
        print('Error:', e)

if __name__ == "__main__":
    test_embed_structure()
