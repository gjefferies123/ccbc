#!/usr/bin/env python3
"""Web interface for the Pastor Chatbot."""

from flask import Flask, render_template, request, jsonify
import os
from dotenv import load_dotenv
import logging
from pastor_chatbot import PastorChatbot

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load environment
load_dotenv()

app = Flask(__name__)

# Initialize chatbot (do this once at startup)
try:
    pastor_bot = PastorChatbot()
    logger.info("✅ Pastor chatbot initialized")
except Exception as e:
    logger.error(f"❌ Failed to initialize pastor chatbot: {e}")
    pastor_bot = None

@app.route('/')
def index():
    """Main chat page."""
    return render_template('pastor_chat.html')

@app.route('/chat', methods=['POST'])
def chat():
    """Handle chat requests."""
    try:
        data = request.get_json()
        question = data.get('question', '').strip()
        
        if not question:
            return jsonify({'error': 'Please ask a question'}), 400
        
        if not pastor_bot:
            return jsonify({'error': 'Pastor chatbot not available'}), 500
        
        # Get pastoral response
        response_data = pastor_bot.get_pastoral_response(question)
        
        if response_data['status'] in ['success', 'success_general']:
            return jsonify({
                'success': True,
                'question': question,
                'response': response_data['response'],
                'sources': response_data.get('sources', []),
                'grounded': len(response_data.get('sources', [])) > 0
            })
        else:
            return jsonify({
                'success': False,
                'error': response_data.get('error', 'Chat failed')
            }), 500
            
    except Exception as e:
        logger.error(f"Chat error: {e}")
        return jsonify({'error': 'Internal server error'}), 500

@app.route('/health')
def health():
    """Health check endpoint."""
    return jsonify({
        'status': 'healthy',
        'chatbot_ready': pastor_bot is not None
    })

if __name__ == '__main__':
    print("🙏 Starting Christ Chapel BC Pastor Chatbot")
    print("=" * 50)
    
    if pastor_bot:
        print("✅ Pastor chatbot ready")
        print("🌐 Web interface starting...")
        print("📍 URL: http://localhost:5000")
        print("💡 Ask spiritual questions and get pastoral responses!")
        print("💡 Press Ctrl+C to stop")
        
        app.run(debug=True, host='0.0.0.0', port=5000)
    else:
        print("❌ Cannot start - pastor chatbot initialization failed")
        print("💡 Check your API keys and search system")
