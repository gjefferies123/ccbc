#!/usr/bin/env python3
"""Complete Biblical Wisdom App using Cohere v2 API for both chat and embeddings."""

import os
from typing import List, Dict, Any, Optional
from dotenv import load_dotenv
import logging
import requests
import json

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

load_dotenv()

class CompleteV2WisdomBot:
    """Biblical guidance bot using Cohere v2 API for everything."""
    
    def __init__(self):
        self.cohere_api_key = os.getenv("COHERE_API_KEY")
        
        # Initialize search engine with v2 encoder
        self.search_engine = None
        
        self._init_clients()
    
    def _init_clients(self):
        """Initialize search with v2 encoder."""
        try:
            # Import and monkey-patch the search engine to use v2
            from christ_chapel_search import ChristChapelSearch
            
            # Initialize the search engine
            self.search_engine = ChristChapelSearch()
            
            # Replace the dense encoder with our v2 version
            from search.encoder_cohere_v2 import get_hybrid_dense_encoder_v2
            self.search_engine.dense_encoder = get_hybrid_dense_encoder_v2()
            
            logger.info("✅ Complete v2 Biblical Wisdom App ready")
            logger.info(f"Encoder info: {self.search_engine.dense_encoder.get_encoder_info()}")
            
        except Exception as e:
            raise RuntimeError(f"Failed to initialize: {e}")
    
    def get_biblical_guidance(self, 
                            user_question: str,
                            max_context_length: int = 2000) -> Dict[str, Any]:
        """
        Get practical biblical guidance for life questions.
        """
        
        logger.info(f"📖 Biblical question: '{user_question[:50]}...'")
        
        try:
            # Step 1: Search for relevant sermon content using v2 embeddings
            search_results = self.search_engine.search(
                query=user_question,
                top_k=5,
                min_score=0.25,
                use_rerank=True,
                candidate_k=20
            )
            
            if search_results['status'] != 'success' or not search_results['results']:
                return self._generate_general_biblical_response(user_question)
            
            # Step 2: Prepare sermon context
            context = self._prepare_sermon_context(search_results['results'], max_context_length)
            
            # Step 3: Generate biblical guidance using Cohere v2 chat API
            guidance = self._generate_biblical_response_v2_chat(user_question, context)
            
            # Step 4: Format response with sources
            return {
                'guidance': guidance,
                'sources': search_results['results'][:3],
                'context_used': len(context),
                'status': 'success'
            }
            
        except Exception as e:
            logger.error(f"❌ Biblical guidance failed: {e}")
            return {
                'guidance': self._fallback_biblical_response(user_question),
                'sources': search_results['results'][:3] if search_results.get('results') else [],
                'status': 'fallback',
                'error': str(e)
            }
    
    def _prepare_sermon_context(self, search_results: List[Dict], max_length: int) -> str:
        """Prepare sermon context for generation."""
        
        context_parts = []
        total_length = 0
        
        for i, result in enumerate(search_results, 1):
            sermon_text = result['text']
            timestamp = result['timestamp']
            
            context_part = f"[Teaching {i} - {timestamp}]: {sermon_text}"
            
            if total_length + len(context_part) > max_length:
                break
            
            context_parts.append(context_part)
            total_length += len(context_part)
        
        return "\n\n".join(context_parts)
    
    def _generate_biblical_response_v2_chat(self, question: str, sermon_context: str) -> str:
        """Generate biblical guidance using Cohere v2 chat API."""
        
        url = "https://api.cohere.com/v2/chat"
        
        headers = {
            "Authorization": f"Bearer {self.cohere_api_key}",
            "Content-Type": "application/json"
        }
        
        # Try the correct model name
        models_to_try = ["command-a-03-2025"]
        
        for model in models_to_try:
            try:
                logger.info(f"Trying v2 chat model: {model}")
                
                payload = {
                    "model": model,
                    "messages": [
                        {
                            "role": "system",
                            "content": "You are a helpful Christian guidance app that provides practical biblical wisdom. IMPORTANT: You must ONLY use information from the provided sermon content below. Do not add any external knowledge, general biblical knowledge, or information not found in the provided sermons. Base your response entirely on the sermon teachings provided. Give clear, practical advice based on the sermon content, address real-world situations using the sermon teachings, be encouraging but honest about challenges, suggest concrete next steps based on the sermon content, and reference specific teachings from the provided sermons."
                        },
                        {
                            "role": "user", 
                            "content": f"""Question: "{question}"

Relevant teachings from Christ Chapel BC sermons:
{sermon_context}

Please provide practical biblical guidance for this question."""
                        }
                    ],
                    "max_tokens": 400,
                    "temperature": 0.6
                }
                
                response = requests.post(url, headers=headers, json=payload, timeout=30)
                
                if response.status_code == 200:
                    data = response.json()
                    
                    # Extract the response text from v2 API structure
                    if "message" in data and "content" in data["message"]:
                        if isinstance(data["message"]["content"], list):
                            logger.info(f"✅ Success with v2 chat model: {model}")
                            return data["message"]["content"][0]["text"]
                        else:
                            logger.info(f"✅ Success with v2 chat model: {model}")
                            return data["message"]["content"]
                    elif "text" in data:
                        logger.info(f"✅ Success with v2 chat model: {model}")
                        return data["text"]
                
                logger.warning(f"Model {model} failed with status {response.status_code}: {response.text}")
                
            except Exception as e:
                logger.warning(f"Model {model} error: {e}")
                continue
        
        logger.warning("All v2 chat models failed, using fallback")
        return self._fallback_biblical_response(question)
    
    def _generate_general_biblical_response(self, question: str) -> Dict[str, Any]:
        """Generate general biblical guidance when no sermon content is found."""
        
        url = "https://api.cohere.com/v2/chat"
        headers = {
            "Authorization": f"Bearer {self.cohere_api_key}",
            "Content-Type": "application/json"
        }
        
        models_to_try = ["command-a-03-2025"]
        
        for model in models_to_try:
            try:
                payload = {
                    "model": model,
                    "messages": [
                        {
                            "role": "system",
                            "content": "You are a helpful Christian guidance app. IMPORTANT: You must ONLY use information from the provided sermon content. Do not add external knowledge or general biblical information not found in the provided sermons. Base your response entirely on the sermon teachings provided."
                        },
                        {
                            "role": "user", 
                            "content": f'Provide practical biblical guidance for: "{question}"\n\nGive clear, practical biblical guidance with relevant scripture and actionable advice.'
                        }
                    ],
                    "max_tokens": 350,
                    "temperature": 0.6
                }
                
                response = requests.post(url, headers=headers, json=payload, timeout=30)
                
                if response.status_code == 200:
                    data = response.json()
                    
                    if "message" in data and "content" in data["message"]:
                        if isinstance(data["message"]["content"], list):
                            guidance_text = data["message"]["content"][0]["text"]
                        else:
                            guidance_text = data["message"]["content"]
                    elif "text" in data:
                        guidance_text = data["text"]
                    else:
                        guidance_text = self._fallback_biblical_response(question)
                    
                    return {
                        'guidance': guidance_text,
                        'sources': [],
                        'context_used': 0,
                        'status': 'success_general'
                    }
                
            except Exception as e:
                logger.warning(f"General response with {model} failed: {e}")
                continue
        
        return {
            'guidance': self._fallback_biblical_response(question),
            'sources': [],
            'status': 'fallback'
        }
    
    def _fallback_biblical_response(self, question: str) -> str:
        """Fallback biblical responses for different topics."""
        
        question_lower = question.lower()
        
        # Categorize and provide relevant biblical guidance
        if any(word in question_lower for word in ['relationship', 'marriage', 'dating', 'love', 'spouse', 'conflict']):
            return """**Biblical Guidance for Relationships:**

• **Love as Christ loved** (Ephesians 5:25) - selfless, sacrificial love
• **Seek wisdom in decisions** (Proverbs 27:17) - iron sharpens iron  
• **Communication and forgiveness** (Ephesians 4:32) - be kind and forgiving
• **Shared faith foundation** (2 Corinthians 6:14) - be equally yoked
• **Prayer together** - bring God into your relationship
• **Gentle answers** (Proverbs 15:1) - a gentle answer turns away wrath

**Practical steps**: When in conflict, listen first, speak with love, seek to understand before being understood, and pray together for God's wisdom."""

        elif any(word in question_lower for word in ['work', 'job', 'career', 'money', 'finances']):
            return """**Biblical Principles for Work & Finance:**

• **Work as unto the Lord** (Colossians 3:23) - excellence in all we do
• **Contentment and trust** (Philippians 4:19) - God will provide  
• **Generosity and stewardship** (Luke 6:38) - give and it will be given
• **Wisdom in decisions** (Proverbs 21:5) - good planning leads to profit
• **Integrity in business** (Proverbs 11:1) - honest dealings please God

**Practical steps**: Seek God's guidance through prayer, make decisions based on biblical principles, budget wisely, give generously, and trust Him with the outcome."""

        elif any(word in question_lower for word in ['anxiety', 'worry', 'stress', 'fear', 'peace']):
            return """**God's Peace for Anxious Times:**

• **Cast your anxiety on Him** (1 Peter 5:7) - He cares for you
• **Don't worry about tomorrow** (Matthew 6:34) - each day has enough trouble
• **Peace that surpasses understanding** (Philippians 4:6-7) - pray about everything  
• **God is in control** (Romans 8:28) - He works all things for good
• **Focus on truth** (Philippians 4:8) - think on whatever is pure and lovely

**Practical steps**: Practice daily prayer and thanksgiving, memorize encouraging scriptures, share your burdens with trusted believers, and focus on what you can control."""

        elif any(word in question_lower for word in ['purpose', 'calling', 'direction', 'will', 'plan']):
            return """**Finding God's Will for Your Life:**

• **Seek first His kingdom** (Matthew 6:33) - prioritize God's purposes
• **Trust in the Lord** (Proverbs 3:5-6) - lean not on your own understanding
• **Use your gifts** (1 Peter 4:10) - serve others with what God has given you
• **Walk in obedience** - God reveals His will as we obey what we already know  
• **Community discernment** - seek counsel from mature believers

**Practical steps**: Spend time in prayer and Scripture, serve where you see needs, seek godly counsel, and trust God to direct your steps as you walk faithfully."""

        else:
            return """**Foundational Biblical Principles:**

• **Love God and love others** (Matthew 22:37-39) - the greatest commandments
• **Trust in God's goodness** (Psalm 34:8) - taste and see that the Lord is good
• **Seek wisdom** (James 1:5) - God gives wisdom generously to those who ask
• **Walk in community** (Hebrews 10:25) - don't neglect gathering together  
• **Rest in God's grace** (Ephesians 2:8-9) - salvation is by faith, not works

**Practical steps**: Spend time in God's Word daily, pray about your specific situation, seek godly counsel from your church community, and trust in God's faithfulness."""


def main():
    """Test the complete v2 biblical wisdom app."""
    print("📖 COMPLETE COHERE V2 BIBLICAL WISDOM APP")
    print("=" * 60)
    
    try:
        wisdom_bot = CompleteV2WisdomBot()
        
        # Test one question
        test_question = "How do I handle conflict with my spouse biblically?"
        print(f"\n❓ Test Question: {test_question}")
        print("-" * 40)
        
        guidance_data = wisdom_bot.get_biblical_guidance(test_question)
        
        print(f"📖 Biblical Guidance:")
        print(guidance_data['guidance'])
        
        if guidance_data['sources']:
            print(f"\n🎯 Based on sermon teachings:")
            for source in guidance_data['sources']:
                print(f"   • {source['timestamp']} - {source['preview'][:60]}...")
        
        print(f"\n✅ Complete v2 Biblical Wisdom App is working!")
        return True
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        return False


if __name__ == "__main__":
    main()
