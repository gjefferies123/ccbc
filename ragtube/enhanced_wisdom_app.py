#!/usr/bin/env python3
"""Enhanced Biblical Wisdom App with Parent Expansion."""

import os
import logging
import requests
from typing import List, Dict, Any, Optional
from dotenv import load_dotenv
from christ_chapel_search import ChristChapelSearch

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

class EnhancedWisdomApp:
    """Enhanced Biblical Wisdom App with parent expansion for full video context."""
    
    def __init__(self):
        self.cohere_api_key = os.getenv("COHERE_API_KEY")
        self.search_engine = ChristChapelSearch()
        self.pinecone_index = self.search_engine.index
        
        if not self.cohere_api_key:
            raise ValueError("COHERE_API_KEY not found in environment variables")
        
        logger.info("✅ Enhanced Biblical Wisdom App ready")
    
    def get_biblical_guidance(self, user_question: str) -> Dict[str, Any]:
        """Get biblical guidance with full video context expansion."""
        
        logger.info(f"📖 Enhanced biblical question: '{user_question[:50]}...'")
        
        # Step 1: Initial search to find relevant chunks
        search_results = self.search_engine.search(
            query=user_question,
            top_k=5,
            min_score=0.25,
            use_rerank=True,
            candidate_k=20
        )
        
        if search_results['status'] != 'success' or not search_results['results']:
            return {
                'guidance': "I'm sorry, I couldn't find relevant teachings for your question. Please try rephrasing it or asking about a different topic.",
                'sources': [],
                'status': 'no_results'
            }
        
        # Step 2: Expand to get full video context
        expanded_context = self._expand_to_full_videos(search_results['results'])
        
        # Step 3: Generate response with full context
        guidance = self._generate_enhanced_response(user_question, expanded_context)
        
        # Step 4: Prepare sources (show original chunks that triggered the search)
        sources = self._prepare_sources(search_results['results'])
        
        return {
            'guidance': guidance,
            'sources': sources,
            'status': 'success',
            'context_info': {
                'original_chunks': len(search_results['results']),
                'expanded_videos': len(expanded_context['videos']),
                'total_chunks_used': sum(len(video['chunks']) for video in expanded_context['videos'].values())
            }
        }
    
    def _expand_to_full_videos(self, search_results: List[Dict]) -> Dict[str, Any]:
        """Expand search results to include all chunks from the same videos."""
        
        logger.info("🔍 Expanding to full video context...")
        
        # Group results by video_id
        video_groups = {}
        for result in search_results:
            video_id = result['video_id']
            if video_id not in video_groups:
                video_groups[video_id] = {
                    'video_title': result['video_title'],
                    'chunks': [],
                    'trigger_chunks': []  # Original chunks that triggered the search
                }
            video_groups[video_id]['trigger_chunks'].append(result)
        
        # For each video, fetch ALL chunks
        expanded_videos = {}
        for video_id, video_info in video_groups.items():
            logger.info(f"📹 Expanding video: {video_id} - {video_info['video_title']}")
            
            # Query Pinecone for ALL chunks from this video
            all_chunks = self._get_all_video_chunks(video_id)
            
            if all_chunks:
                # Sort chunks by chunk_index to maintain order
                all_chunks.sort(key=lambda x: x.get('chunk_index', 0))
                
                expanded_videos[video_id] = {
                    'video_title': video_info['video_title'],
                    'chunks': all_chunks,
                    'trigger_chunks': video_info['trigger_chunks'],
                    'total_chunks': len(all_chunks)
                }
                
                logger.info(f"✅ Expanded to {len(all_chunks)} chunks for video {video_id}")
            else:
                logger.warning(f"⚠️ No chunks found for video {video_id}")
        
        return {
            'videos': expanded_videos,
            'total_videos': len(expanded_videos)
        }
    
    def _get_all_video_chunks(self, video_id: str) -> List[Dict]:
        """Get all chunks for a specific video from Pinecone."""
        
        try:
            # Query Pinecone for all chunks with this video_id
            query_response = self.pinecone_index.query(
                vector=[0.0] * 1536,  # Dummy vector since we're filtering by metadata
                filter={"video_id": {"$eq": video_id}},
                top_k=1000,  # Get all chunks for this video
                include_metadata=True
            )
            
            chunks = []
            for match in query_response.matches:
                metadata = match.metadata
                chunk = {
                    'id': match.id,
                    'video_id': metadata.get('video_id', video_id),
                    'video_title': metadata.get('video_title', 'Unknown'),
                    'text': metadata.get('text', ''),
                    'timestamp': metadata.get('timestamp', '0:00-0:00'),
                    'start_sec': metadata.get('start_sec', 0),
                    'end_sec': metadata.get('end_sec', 0),
                    'chunk_index': metadata.get('chunk_ix', 0),
                    'url': metadata.get('url', ''),
                    'score': match.score
                }
                chunks.append(chunk)
            
            return chunks
            
        except Exception as e:
            logger.error(f"❌ Failed to get chunks for video {video_id}: {e}")
            return []
    
    def _generate_enhanced_response(self, question: str, expanded_context: Dict) -> str:
        """Generate response using full video context."""
        
        # Prepare comprehensive context from all video chunks
        context_parts = []
        
        for video_id, video_info in expanded_context['videos'].items():
            video_title = video_info['video_title']
            chunks = video_info['chunks']
            
            # Create a comprehensive transcript for this video
            video_transcript = f"=== {video_title} ===\n"
            
            for chunk in chunks:
                timestamp = chunk['timestamp']
                text = chunk['text']
                video_transcript += f"[{timestamp}] {text}\n"
            
            context_parts.append(video_transcript)
        
        full_context = "\n\n".join(context_parts)
        
        # Generate response using Cohere v2 chat
        return self._generate_biblical_response_v2_chat(question, full_context)
    
    def _generate_biblical_response_v2_chat(self, question: str, sermon_context: str) -> str:
        """Generate biblical guidance using Cohere v2 chat API with full context."""
        
        url = "https://api.cohere.com/v2/chat"
        headers = {
            "Authorization": f"Bearer {self.cohere_api_key}",
            "Content-Type": "application/json"
        }
        
        # Enhanced prompt for full video context
        system_prompt = """You are a helpful Christian guidance app that provides practical biblical wisdom. 

IMPORTANT: You have access to COMPLETE sermon transcripts from Christ Chapel BC. Use the full context of these sermons to provide comprehensive, well-informed guidance. 

Instructions:
1. ONLY use information from the provided sermon content
2. Draw from the COMPLETE sermon context, not just isolated quotes
3. Reference specific teachings, stories, and examples from the sermons
4. Provide practical, actionable advice based on the sermon teachings
5. Be encouraging but honest about challenges
6. Suggest concrete next steps based on the sermon content
7. Reference specific sermon moments when relevant

The sermon content below contains complete transcripts with timestamps. Use this full context to give the best possible guidance."""
        
        payload = {
            "model": "command-a-03-2025",
            "messages": [
                {
                    "role": "system",
                    "content": system_prompt
                },
                {
                    "role": "user",
                    "content": f"""Question: "{question}"

Complete Sermon Transcripts from Christ Chapel BC:
{sermon_context}

Please provide comprehensive biblical guidance for this question using the full context of the sermons above."""
                }
            ],
            "max_tokens": 1000,
            "temperature": 0.6
        }
        
        try:
            response = requests.post(url, headers=headers, json=payload, timeout=30)
            response.raise_for_status()
            
            data = response.json()
            
            # Extract response text from v2 API structure
            response_text = ""
            if "message" in data and "content" in data["message"]:
                if isinstance(data["message"]["content"], list):
                    response_text = data["message"]["content"][0]["text"]
                else:
                    response_text = data["message"]["content"]
            elif "text" in data:
                response_text = data["text"]
            else:
                logger.error(f"Unexpected response structure: {data}")
                return "I'm sorry, I had trouble processing your question. Please try again."
            
            # Check if response was cut off (common indicators)
            if response_text and not response_text.strip().endswith(('.', '!', '?', ':', ';')):
                logger.warning("Response may have been cut off - attempting to continue")
                # Try to get a continuation
                continuation = self._get_response_continuation(question, sermon_context, response_text)
                if continuation:
                    response_text += "\n\n" + continuation
                else:
                    response_text += "\n\n*[Response may be incomplete due to length limits. Please ask a more specific question for a complete answer.]"
            
            return response_text
                
        except Exception as e:
            logger.error(f"❌ Failed to generate response: {e}")
            return "I'm sorry, I'm having trouble providing guidance right now. Please try again later."
    
    def _get_response_continuation(self, question: str, sermon_context: str, partial_response: str) -> Optional[str]:
        """Get continuation of a cut-off response."""
        
        url = "https://api.cohere.com/v2/chat"
        headers = {
            "Authorization": f"Bearer {self.cohere_api_key}",
            "Content-Type": "application/json"
        }
        
        payload = {
            "model": "command-a-03-2025",
            "messages": [
                {
                    "role": "system",
                    "content": "You are continuing a biblical guidance response. Complete the response naturally where it was cut off. Keep the same tone and style. Do not repeat what was already said."
                },
                {
                    "role": "user",
                    "content": f"""Question: "{question}"

Sermon Context: {sermon_context[:2000]}...

Partial Response (complete this): {partial_response}

Please continue the response naturally from where it was cut off."""
                }
            ],
            "max_tokens": 500,
            "temperature": 0.6
        }
        
        try:
            response = requests.post(url, headers=headers, json=payload, timeout=30)
            response.raise_for_status()
            
            data = response.json()
            
            if "message" in data and "content" in data["message"]:
                if isinstance(data["message"]["content"], list):
                    return data["message"]["content"][0]["text"]
                else:
                    return data["message"]["content"]
            elif "text" in data:
                return data["text"]
            
            return None
            
        except Exception as e:
            logger.error(f"❌ Failed to get continuation: {e}")
            return None
    
    def _prepare_sources(self, search_results: List[Dict]) -> List[Dict]:
        """Prepare source information for display."""
        
        sources = []
        for result in search_results:
            source = {
                'video_id': result['video_id'],
                'video_title': result['video_title'],
                'timestamp': result['timestamp'],
                'url': result['url'],
                'text': result['text'],  # Add the full text field
                'preview': result['text'][:200] + '...' if len(result['text']) > 200 else result['text'],
                'score': result['score']
            }
            sources.append(source)
        
        return sources

def main():
    """Test the enhanced wisdom app."""
    try:
        app = EnhancedWisdomApp()
        
        # Test question
        question = "How do I handle conflict with my spouse biblically?"
        
        print(f"🔍 Question: {question}")
        print("=" * 50)
        
        result = app.get_biblical_guidance(question)
        
        print(f"Status: {result['status']}")
        print(f"Context Info: {result.get('context_info', {})}")
        print(f"\nGuidance:\n{result['guidance']}")
        print(f"\nSources: {len(result['sources'])}")
        
        for i, source in enumerate(result['sources'][:3]):
            print(f"  {i+1}. {source['timestamp']} - {source['video_title']}")
        
    except Exception as e:
        logger.error(f"❌ Test failed: {e}")

if __name__ == "__main__":
    main()
