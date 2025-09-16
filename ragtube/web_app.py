#!/usr/bin/env python3
"""Simple web interface for Christ Chapel BC sermon search."""

from flask import Flask, render_template, request, jsonify, send_from_directory
import os
from dotenv import load_dotenv
import logging
from christ_chapel_search import ChristChapelSearch

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load environment
load_dotenv()

app = Flask(__name__)

# Initialize search (do this once at startup)
try:
    search_engine = ChristChapelSearch()
    logger.info("✅ Search engine initialized")
except Exception as e:
    logger.error(f"❌ Failed to initialize search: {e}")
    search_engine = None

@app.route('/')
def index():
    """Main search page."""
    return render_template('index.html')

@app.route('/search', methods=['POST'])
def search():
    """Handle search requests."""
    try:
        data = request.get_json()
        query = data.get('query', '').strip()
        
        if not query:
            return jsonify({'error': 'Please enter a question'}), 400
        
        if not search_engine:
            return jsonify({'error': 'Search engine not available'}), 500
        
        # Perform search
        result = search_engine.search(
            query=query,
            top_k=5,
            min_score=0.25  # Lower threshold for web interface
        )
        
        if result['status'] == 'success':
            return jsonify({
                'success': True,
                'query': query,
                'results': result['results'],
                'total': result['total_results']
            })
        else:
            return jsonify({
                'success': False,
                'error': result.get('error', 'Search failed')
            }), 500
            
    except Exception as e:
        logger.error(f"Search error: {e}")
        return jsonify({'error': 'Internal server error'}), 500

@app.route('/stats')
def stats():
    """Get index statistics."""
    try:
        if not search_engine:
            return jsonify({'error': 'Search engine not available'}), 500
        
        stats = search_engine.get_index_stats()
        return jsonify(stats)
        
    except Exception as e:
        logger.error(f"Stats error: {e}")
        return jsonify({'error': 'Failed to get stats'}), 500

@app.route('/video/<video_id>')
def video_info(video_id):
    """Get information about a specific video."""
    try:
        if not search_engine:
            return jsonify({'error': 'Search engine not available'}), 500
        
        video_data = search_engine.get_video_summary(video_id)
        return jsonify(video_data)
        
    except Exception as e:
        logger.error(f"Video info error: {e}")
        return jsonify({'error': 'Failed to get video info'}), 500

if __name__ == '__main__':
    print("🏛️ Starting Christ Chapel BC Search Web App")
    print("=" * 50)
    
    if search_engine:
        print("✅ Search engine ready")
        
        # Get some stats
        try:
            stats = search_engine.get_index_stats()
            print(f"📊 Index: {stats.get('total_vectors', 'unknown')} vectors")
        except:
            pass
        
        print("🌐 Web interface starting...")
        print("📍 URL: http://localhost:5000")
        print("💡 Press Ctrl+C to stop")
        
        app.run(debug=True, host='0.0.0.0', port=5000)
    else:
        print("❌ Cannot start - search engine initialization failed")
        print("💡 Check your API keys and Pinecone connection")
