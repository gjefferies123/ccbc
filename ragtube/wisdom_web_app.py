#!/usr/bin/env python3
"""Web interface for the Biblical Wisdom App."""

from flask import Flask, render_template, request, jsonify
import os
from dotenv import load_dotenv
import logging
from biblical_wisdom_app import BiblicalWisdomBot

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load environment
load_dotenv()

app = Flask(__name__)

# Initialize wisdom bot (do this once at startup)
try:
    wisdom_bot = BiblicalWisdomBot()
    logger.info("✅ Biblical Wisdom App initialized")
except Exception as e:
    logger.error(f"❌ Failed to initialize Biblical Wisdom App: {e}")
    wisdom_bot = None

@app.route('/')
def index():
    """Main wisdom app page."""
    return render_template('wisdom_app.html')

@app.route('/ask', methods=['POST'])
def ask_question():
    """Handle wisdom questions."""
    try:
        data = request.get_json()
        question = data.get('question', '').strip()
        
        if not question:
            return jsonify({'error': 'Please ask a question'}), 400
        
        if not wisdom_bot:
            return jsonify({'error': 'Biblical Wisdom App not available'}), 500
        
        # Get biblical guidance
        guidance_data = wisdom_bot.get_biblical_guidance(question)
        
        if guidance_data['status'] in ['success', 'success_general']:
            return jsonify({
                'success': True,
                'question': question,
                'guidance': guidance_data['guidance'],
                'sources': guidance_data.get('sources', []),
                'grounded': len(guidance_data.get('sources', [])) > 0
            })
        else:
            return jsonify({
                'success': False,
                'error': guidance_data.get('error', 'Guidance failed')
            }), 500
            
    except Exception as e:
        logger.error(f"Wisdom error: {e}")
        return jsonify({'error': 'Internal server error'}), 500

@app.route('/health')
def health():
    """Health check endpoint."""
    return jsonify({
        'status': 'healthy',
        'wisdom_app_ready': wisdom_bot is not None
    })

if __name__ == '__main__':
    print("📖 Starting Biblical Wisdom App")
    print("=" * 50)
    
    if wisdom_bot:
        print("✅ Biblical Wisdom App ready")
        print("🌐 Web interface starting...")
        print("📍 URL: http://localhost:5000")
        print("💡 Ask biblical questions and get practical guidance!")
        print("💡 Press Ctrl+C to stop")
        
        app.run(debug=True, host='0.0.0.0', port=5000)
    else:
        print("❌ Cannot start - Biblical Wisdom App initialization failed")
        print("💡 Check your API keys and search system")
