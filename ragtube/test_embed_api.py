#!/usr/bin/env python3
"""Test Cohere v4.0 embedding API response structure."""

import requests
import os
from dotenv import load_dotenv

load_dotenv()

def test_embed_api():
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
        print('Response keys:', list(response.json().keys()))
        print('Embeddings type:', type(response.json()['embeddings']))
        print('First embedding type:', type(response.json()['embeddings'][0]))
        print('First embedding keys:', list(response.json()['embeddings'][0].keys()))
    except Exception as e:
        print('Error:', e)

if __name__ == "__main__":
    test_embed_api()
