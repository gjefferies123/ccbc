#!/usr/bin/env python3
"""Final Biblical Wisdom Web App."""

from flask import Flask, render_template, request, jsonify
import logging
from enhanced_wisdom_app import EnhancedWisdomApp

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)

# Initialize the Enhanced Biblical Wisdom App
try:
    wisdom_bot = EnhancedWisdomApp()
    logger.info("✅ Enhanced Biblical Wisdom App initialized successfully")
except Exception as e:
    logger.error(f"❌ Failed to initialize Enhanced Biblical Wisdom App: {e}")
    wisdom_bot = None

@app.route('/')
def home():
    """Serve the Biblical Wisdom App homepage."""
    return render_template('wisdom_app.html')

@app.route('/ask', methods=['POST'])
def ask_question():
    """Handle biblical guidance requests."""
    
    if not wisdom_bot:
        return jsonify({
            'error': 'Biblical Wisdom Bot is not available',
            'guidance': 'Sorry, the service is temporarily unavailable. Please try again later.',
            'sources': []
        }), 500
    
    try:
        data = request.get_json()
        question = data.get('question', '').strip()
        
        if not question:
            return jsonify({
                'error': 'No question provided',
                'guidance': 'Please ask a question to receive biblical guidance.',
                'sources': []
            }), 400
        
        logger.info(f"📖 Received question: {question[:50]}...")
        
        # Get biblical guidance
        guidance_data = wisdom_bot.get_biblical_guidance(question)
        
        # Format sources for the web interface
        formatted_sources = []
        if guidance_data.get('sources'):
            for source in guidance_data['sources']:
                formatted_sources.append({
                    'timestamp': source.get('timestamp', 'Unknown time'),
                    'preview': source.get('preview', source.get('text', ''))[:100] + '...',
                    'video_title': source.get('video_title', 'Christ Chapel BC Sermon'),
                    'url': source.get('url', '#')
                })
        
        return jsonify({
            'guidance': guidance_data['guidance'],
            'sources': formatted_sources,
            'status': guidance_data.get('status', 'success')
        })
        
    except Exception as e:
        logger.error(f"❌ Error processing question: {e}")
        return jsonify({
            'error': 'Internal server error',
            'guidance': 'Sorry, there was an error processing your question. Please try again.',
            'sources': []
        }), 500

@app.route('/health')
@app.route('/healthz')
def health_check():
    """Health check endpoint."""
    return jsonify({
        'status': 'healthy',
        'wisdom_bot_available': wisdom_bot is not None
    })

if __name__ == '__main__':
    import os
    port = int(os.environ.get('PORT', 5000))
    debug = os.environ.get('FLASK_DEBUG', 'False').lower() == 'true'
    
    print("🌐 Starting Biblical Wisdom Web App...")
    print(f"📖 Running on port: {port}")
    print("💡 Get practical biblical guidance based on Christ Chapel BC sermons!")
    
    app.run(debug=debug, host='0.0.0.0', port=port)
