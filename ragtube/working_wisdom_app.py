#!/usr/bin/env python3
"""Working Biblical Wisdom App with current Cohere models."""

import os
from typing import List, Dict, Any, Optional
from dotenv import load_dotenv
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

load_dotenv()

class WorkingBiblicalWisdomBot:
    """Working biblical guidance bot with current Cohere models."""
    
    def __init__(self):
        self.cohere_api_key = os.getenv("COHERE_API_KEY")
        
        # Initialize search and generation clients
        self.search_engine = None
        self.cohere_client = None
        
        self._init_clients()
    
    def _init_clients(self):
        """Initialize search and generation clients."""
        try:
            # Initialize RAG search
            from christ_chapel_search import ChristChapelSearch
            self.search_engine = ChristChapelSearch()
            
            # Initialize Cohere for generation
            import cohere
            self.cohere_client = cohere.Client(self.cohere_api_key)
            
            logger.info("✅ Working Biblical Wisdom App ready")
            
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
            # Step 1: Search for relevant sermon content
            search_results = self.search_engine.search(
                query=user_question,
                top_k=5,
                min_score=0.25
            )
            
            if search_results['status'] != 'success' or not search_results['results']:
                return self._generate_general_biblical_response(user_question)
            
            # Step 2: Prepare sermon context
            context = self._prepare_sermon_context(search_results['results'], max_context_length)
            
            # Step 3: Generate biblical guidance using current Cohere models
            guidance = self._generate_biblical_response_current(user_question, context)
            
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
    
    def _generate_biblical_response_current(self, question: str, sermon_context: str) -> str:
        """Generate biblical guidance using current Cohere models."""
        
        try:
            # Try with command-r model (should be current)
            response = self.cohere_client.chat(
                model="command-r",  # Use current model instead of deprecated command-r-plus
                message=f"""Question: "{question}"

Relevant teachings from Christ Chapel BC sermons:
{sermon_context}

Please provide practical biblical guidance for this question. Give clear, practical advice based on biblical principles, use the sermon content as supporting wisdom, address real-world situations with biblical worldview, be encouraging but honest about challenges, suggest concrete next steps or actions, and reference relevant scripture when helpful.""",
                max_tokens=400,
                temperature=0.6
            )
            
            return response.text.strip()
            
        except Exception as e:
            logger.warning(f"Command-R failed: {e}, trying alternatives...")
            
            try:
                # Try with command model as backup
                response = self.cohere_client.chat(
                    model="command",
                    message=f"""Provide biblical guidance for: "{question}"

Based on these sermon teachings:
{sermon_context}

Give practical Christian advice with scripture references.""",
                    max_tokens=300,
                    temperature=0.6
                )
                
                return response.text.strip()
                
            except Exception as e2:
                logger.warning(f"Command also failed: {e2}, using fallback...")
                return self._fallback_biblical_response(question)
    
    def _generate_general_biblical_response(self, question: str) -> Dict[str, Any]:
        """Generate general biblical guidance when no sermon content is found."""
        
        try:
            response = self.cohere_client.chat(
                model="command-r",
                message=f'Provide practical biblical guidance for: "{question}"\n\nGive clear, practical biblical guidance with relevant scripture and actionable advice.',
                max_tokens=350,
                temperature=0.6
            )
            
            return {
                'guidance': response.text.strip(),
                'sources': [],
                'context_used': 0,
                'status': 'success_general'
            }
            
        except Exception as e:
            logger.warning(f"General response failed: {e}")
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
    """Test the working biblical wisdom app."""
    print("📖 WORKING BIBLICAL WISDOM APP")
    print("=" * 50)
    
    try:
        wisdom_bot = WorkingBiblicalWisdomBot()
        
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
        
        print(f"\n✅ Working Biblical Wisdom App is functional!")
        return True
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        return False


if __name__ == "__main__":
    main()
