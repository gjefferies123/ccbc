#!/usr/bin/env python3
"""Pastor chatbot powered by Christ Chapel BC RAG system."""

import os
from typing import List, Dict, Any, Optional
from dotenv import load_dotenv
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

load_dotenv()

class PastorChatbot:
    """A conversational pastor chatbot using RAG retrieval."""
    
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
            
            logger.info("✅ Pastor chatbot ready")
            
        except Exception as e:
            raise RuntimeError(f"Failed to initialize chatbot: {e}")
    
    def get_pastoral_response(self, 
                            user_question: str,
                            max_context_length: int = 2000,
                            temperature: float = 0.7) -> Dict[str, Any]:
        """
        Generate a pastoral response to a user's spiritual question.
        
        Args:
            user_question: The user's spiritual question
            max_context_length: Maximum length of sermon context to include
            temperature: Generation temperature (0.0-1.0, higher = more creative)
        
        Returns:
            Dictionary with pastoral response and metadata
        """
        
        logger.info(f"🙏 Pastoral question: '{user_question[:50]}...'")
        
        try:
            # Step 1: Retrieve relevant sermon content
            search_results = self.search_engine.search(
                query=user_question,
                top_k=5,
                min_score=0.3
            )
            
            if search_results['status'] != 'success' or not search_results['results']:
                return self._generate_general_pastoral_response(user_question)
            
            # Step 2: Prepare context from sermon content
            context = self._prepare_sermon_context(search_results['results'], max_context_length)
            
            # Step 3: Generate pastoral response
            pastoral_response = self._generate_response_with_context(
                user_question, 
                context,
                temperature
            )
            
            # Step 4: Format response with sources
            return {
                'response': pastoral_response,
                'sources': search_results['results'][:3],  # Top 3 sources
                'context_used': len(context),
                'status': 'success'
            }
            
        except Exception as e:
            logger.error(f"❌ Pastoral response failed: {e}")
            return {
                'response': "I apologize, but I'm having trouble accessing the sermon content right now. However, I encourage you to continue seeking God's wisdom through prayer and His Word. Is there a specific Bible verse or topic you'd like to explore together?",
                'sources': [],
                'status': 'error',
                'error': str(e)
            }
    
    def _prepare_sermon_context(self, search_results: List[Dict], max_length: int) -> str:
        """Prepare sermon context for generation."""
        
        context_parts = []
        total_length = 0
        
        for i, result in enumerate(search_results, 1):
            # Format each piece of sermon content
            sermon_text = result['text']
            timestamp = result['timestamp']
            video_id = result['video_id']
            
            context_part = f"[Sermon Context {i} - {timestamp}]: {sermon_text}"
            
            if total_length + len(context_part) > max_length:
                break
            
            context_parts.append(context_part)
            total_length += len(context_part)
        
        return "\n\n".join(context_parts)
    
    def _generate_response_with_context(self, 
                                      question: str, 
                                      sermon_context: str,
                                      temperature: float) -> str:
        """Generate pastoral response using sermon context."""
        
        # Create pastoral prompt
        pastoral_prompt = f"""You are a caring, wise pastor from Christ Chapel BC responding to a member's spiritual question. You should:

1. Speak warmly and personally, as a pastor would in conversation
2. Use the provided sermon content as inspiration and grounding for your response
3. Apply biblical wisdom practically to their situation
4. Encourage spiritual growth and deeper relationship with God
5. Sound conversational and caring, not academic or formal
6. Include relevant scripture when appropriate
7. Offer practical next steps or encouragement

Question from church member: "{question}"

Relevant content from recent Christ Chapel BC sermons:
{sermon_context}

Respond as Pastor would, drawing wisdom from these sermons but speaking naturally and personally:"""

        try:
            response = self.cohere_client.generate(
                model="command-r-plus",  # Best Cohere model for conversation
                prompt=pastoral_prompt,
                max_tokens=400,
                temperature=temperature,
                stop_sequences=["Question:", "Sermon Context"]
            )
            
            return response.generations[0].text.strip()
            
        except Exception as e:
            logger.error(f"Generation failed: {e}")
            return self._fallback_pastoral_response(question)
    
    def _generate_general_pastoral_response(self, question: str) -> Dict[str, Any]:
        """Generate a general pastoral response when no sermon content is found."""
        
        general_prompt = f"""You are a caring pastor responding to a spiritual question. Even though you don't have specific sermon content to reference, provide a warm, biblical, and encouraging response.

Question: "{question}"

Respond as a pastor would, with biblical wisdom and practical encouragement:"""
        
        try:
            response = self.cohere_client.generate(
                model="command-r-plus",
                prompt=general_prompt,
                max_tokens=300,
                temperature=0.7
            )
            
            return {
                'response': response.generations[0].text.strip(),
                'sources': [],
                'context_used': 0,
                'status': 'success_general'
            }
            
        except Exception as e:
            return {
                'response': self._fallback_pastoral_response(question),
                'sources': [],
                'status': 'fallback'
            }
    
    def _fallback_pastoral_response(self, question: str) -> str:
        """Fallback response when generation fails."""
        
        fallback_responses = {
            'faith': "Thank you for your question about faith. Faith is such a central part of our walk with God. I encourage you to spend time in God's Word, especially in books like Romans and Hebrews, which speak beautifully about faith. Remember, faith grows through relationship with Jesus and community with other believers. How can I pray for you in your faith journey?",
            
            'prayer': "Prayer is one of the most precious gifts God has given us - direct access to Him! Don't worry about having the 'right' words. God knows your heart. Start simple: thank Him, confess your needs, and listen. The Holy Spirit helps us when we don't know what to pray. I'd encourage you to set aside a few minutes each day to just be with God. What's on your heart that you'd like to bring to Him?",
            
            'purpose': "What a meaningful question about God's purpose for your life! Remember, God created you uniquely and has good plans for you (Jeremiah 29:11). Often His purpose unfolds as we walk faithfully with Him, serve others, and use the gifts He's given us. Spend time in prayer and His Word, and look for ways to love and serve others. God often reveals His purpose through faithful obedience in the small things. How can I help you discern your next steps?",
            
            'salvation': "This is the most important question anyone can ask! Salvation is God's free gift to us through Jesus Christ. Romans 10:9 tells us that if we confess Jesus as Lord and believe God raised Him from the dead, we will be saved. It's not about being good enough - it's about God's grace and love for you. Jesus died for your sins and rose again to give you new life. Would you like to talk more about what it means to follow Jesus?",
            
            'default': "Thank you for your heart-felt question. While I don't have specific sermon content to reference right now, I want you to know that God sees you and cares deeply about what you're going through. I encourage you to spend time in prayer and in God's Word. Consider reaching out to our pastoral team or joining a small group where you can find community and support. Remember, God's love for you is unchanging, and He wants to walk with you through every season of life. How can I pray for you today?"
        }
        
        # Simple keyword matching for fallback
        question_lower = question.lower()
        
        if any(word in question_lower for word in ['faith', 'believe', 'trust']):
            return fallback_responses['faith']
        elif any(word in question_lower for word in ['pray', 'prayer', 'praying']):
            return fallback_responses['prayer']
        elif any(word in question_lower for word in ['purpose', 'plan', 'calling', 'will']):
            return fallback_responses['purpose']
        elif any(word in question_lower for word in ['salvation', 'saved', 'forgiveness', 'sin']):
            return fallback_responses['salvation']
        else:
            return fallback_responses['default']


def test_pastor_chatbot():
    """Test the pastor chatbot with sample questions."""
    
    print("🙏 CHRIST CHAPEL BC PASTOR CHATBOT")
    print("=" * 60)
    
    try:
        chatbot = PastorChatbot()
        
        test_questions = [
            "I'm struggling with doubt in my faith. How can I trust God more?",
            "What does it mean to have a personal relationship with Jesus?",
            "How do I know God's will for my life?",
            "I feel distant from God lately. What should I do?",
            "How can I pray more effectively?",
            "What does the Bible say about forgiveness?"
        ]
        
        for i, question in enumerate(test_questions, 1):
            print(f"\n👤 Question {i}: {question}")
            print("-" * 50)
            
            response_data = chatbot.get_pastoral_response(question)
            
            print(f"🙏 Pastor's Response:")
            print(response_data['response'])
            
            if response_data['sources']:
                print(f"\n📖 Grounded in sermons:")
                for source in response_data['sources']:
                    print(f"   • {source['timestamp']} - {source['preview'][:60]}...")
            
            print("\n" + "=" * 60)
        
        return True
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        return False


def interactive_pastor_chat():
    """Interactive chat with the pastor chatbot."""
    
    print("\n💬 INTERACTIVE PASTOR CHAT")
    print("Ask any spiritual question - I'm here to help!")
    print("Type 'quit' to exit")
    print("=" * 50)
    
    try:
        chatbot = PastorChatbot()
        
        while True:
            question = input("\n🙏 Your question: ").strip()
            
            if question.lower() in ['quit', 'exit', 'bye']:
                print("\n✨ May God bless you on your journey! Feel free to come back anytime.")
                break
            
            if not question:
                continue
            
            print("\n🤔 Let me pray and reflect on that...")
            
            response_data = chatbot.get_pastoral_response(question)
            
            print(f"\n🙏 Pastor's Response:")
            print(response_data['response'])
            
            if response_data['sources']:
                print(f"\n📖 This wisdom comes from our recent sermons:")
                for source in response_data['sources']:
                    print(f"   • Watch at {source['timestamp']}: {source['url']}")
    
    except KeyboardInterrupt:
        print("\n✨ Blessings to you! Come back anytime.")
    except Exception as e:
        print(f"❌ I'm having some technical difficulties: {e}")
        print("But remember, God is always with you!")


def main():
    """Main function."""
    
    print("🏛️ CHRIST CHAPEL BC PASTOR CHATBOT")
    print("=" * 50)
    print("Choose an option:")
    print("1. Test with sample questions")
    print("2. Interactive chat")
    print("3. Both")
    
    choice = input("\nYour choice (1/2/3): ").strip()
    
    if choice in ['1', '3']:
        test_success = test_pastor_chatbot()
        if not test_success:
            return
    
    if choice in ['2', '3']:
        interactive_pastor_chat()


if __name__ == "__main__":
    main()
