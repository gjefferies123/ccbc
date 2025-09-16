#!/usr/bin/env python3
"""Upgrade to Cohere v4.0 system with 1536-dimensional embeddings."""

import os
import json
from typing import List, Dict, Any
from dotenv import load_dotenv
import logging
import requests
import time

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

load_dotenv()

class V4Upgrader:
    """Upgrade system to Cohere v4.0 with 1536-dimensional embeddings."""
    
    def __init__(self):
        self.pinecone_api_key = os.getenv("PINECONE_API_KEY")
        self.cohere_api_key = os.getenv("COHERE_API_KEY")
        self.index_name = os.getenv("PINECONE_INDEX", "christ-chapel-sermons")
        
        # Video IDs to process
        self.video_ids = [
            # Existing videos
            "Jz1Zb57NUMg", "6_HgIPUXpVM", "iHypAArzphY", "vM_XX9P66RU", "OO6l1lkK3yM",
            # New videos
            "JurVL5nt34U", "n7DIs9sg_NQ", "rsr3l0zNp6Q", "8MUivfHHM4w", "_YE2Yd2GD2U",
            "zJiA-UHWcHw", "eCweRDNtCfQ", "3JWoLag6xg4", "UIkn-t1khDE", "bqb_G4zzbOQ",
            "2IgkYGzT5fo", "Zncv266jSN", "9d-5h2-DzCE", "qllastULszc", "1RYR9vuhbYQ"
        ]
        
        self.pc = None
        self.index = None
    
    def _init_pinecone(self):
        """Initialize Pinecone client."""
        try:
            from pinecone import Pinecone
            self.pc = Pinecone(api_key=self.pinecone_api_key)
            logger.info("✅ Pinecone client initialized")
        except Exception as e:
            raise RuntimeError(f"Failed to initialize Pinecone: {e}")
    
    def delete_old_index(self):
        """Delete the existing 1024-dimensional index."""
        try:
            logger.info("🗑️ Deleting old index...")
            self.pc.delete_index(self.index_name)
            logger.info("✅ Old index deleted")
            
            # Wait for deletion to complete
            logger.info("⏳ Waiting for deletion to complete...")
            time.sleep(10)
            
        except Exception as e:
            logger.warning(f"Index deletion failed (may not exist): {e}")
    
    def create_v4_index(self):
        """Create new index with 1536 dimensions for v4.0."""
        try:
            logger.info("🏗️ Creating new v4.0 index with 1536 dimensions...")
            
            self.pc.create_index(
                name=self.index_name,
                dimension=1536,  # v4.0 embedding dimension
                metric="cosine",
                spec={
                    "serverless": {
                        "cloud": "aws",
                        "region": "us-east-1"  # Free tier region
                    }
                }
            )
            
            logger.info("✅ New v4.0 index created")
            
            # Wait for index to be ready
            logger.info("⏳ Waiting for index to be ready...")
            time.sleep(30)
            
            # Connect to the new index
            self.index = self.pc.Index(self.index_name)
            logger.info("✅ Connected to new v4.0 index")
            
        except Exception as e:
            raise RuntimeError(f"Failed to create v4.0 index: {e}")
    
    def test_v4_embedding(self):
        """Test v4.0 embedding to verify dimension."""
        try:
            logger.info("🧪 Testing v4.0 embedding dimension...")
            
            url = "https://api.cohere.com/v2/embed"
            headers = {
                "Authorization": f"Bearer {self.cohere_api_key}",
                "Content-Type": "application/json"
            }
            payload = {
                "model": "embed-v4.0",
                "texts": ["test embedding"],
                "input_type": "search_query",
                "embedding_types": ["float"]
            }
            
            resp = requests.post(url, headers=headers, json=payload, timeout=30)
            if resp.status_code == 200:
                data = resp.json()
                emb = data.get("embeddings", {}).get("float", [[]])[0]
                logger.info(f"✅ v4.0 embedding dimension: {len(emb)}")
                return len(emb) == 1536
            else:
                logger.error(f"❌ v4.0 embedding failed: {resp.status_code} {resp.text}")
                return False
                
        except Exception as e:
            logger.error(f"❌ v4.0 embedding test failed: {e}")
            return False
    
    def get_video_transcript(self, video_id: str) -> Dict[str, Any]:
        """Get transcript for a video."""
        try:
            from youtube_transcript_api import YouTubeTranscriptApi
            
            # Get available transcripts
            api = YouTubeTranscriptApi()
            transcript_list = api.list(video_id)
            
            # Try to get English transcript (auto-generated)
            transcript = None
            try:
                # Try to get English auto-generated transcript
                transcript = transcript_list.find_generated_transcript(['en'])
            except:
                try:
                    # Try any available auto-generated transcript
                    transcript = transcript_list.find_generated_transcript()
                except:
                    # Try any available transcript
                    try:
                        transcript = next(iter(transcript_list))
                    except:
                        raise Exception("No transcript available")
            
            # Fetch the transcript
            transcript_items = transcript.fetch()
            
            # Convert to text
            text = " ".join([entry['text'] for entry in transcript_items])
            
            return {
                'video_id': video_id,
                'text': text,
                'entries': transcript_items,
                'duration': transcript_items[-1]['start'] + transcript_items[-1]['duration'] if transcript_items else 0
            }
            
        except Exception as e:
            logger.warning(f"Failed to get transcript for {video_id}: {e}")
            return None
    
    def segment_transcript(self, transcript_data: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Segment transcript into hierarchical chunks."""
        video_id = transcript_data['video_id']
        text = transcript_data['text']
        entries = transcript_data['entries']
        duration = transcript_data['duration']
        
        chunks = []
        
        # Simple time-based segmentation (45-90s chunks with 10-15s overlap)
        chunk_duration = 75  # seconds
        overlap = 12  # seconds
        
        current_time = 0
        chunk_id = 0
        
        while current_time < duration:
            end_time = min(current_time + chunk_duration, duration)
            
            # Find text for this time range
            chunk_text = ""
            for entry in entries:
                if current_time <= entry['start'] < end_time:
                    chunk_text += entry['text'] + " "
            
            if chunk_text.strip():
                chunk = {
                    'id': f"{video_id}_chunk_{chunk_id}",
                    'video_id': video_id,
                    'text': chunk_text.strip(),
                    'start_sec': int(current_time),
                    'end_sec': int(end_time),
                    'duration': int(end_time - current_time),
                    'timestamp': f"{int(current_time//60)}:{int(current_time%60):02d}",
                    'chunk_ix': chunk_id,
                    'parent_id': f"{video_id}_parent_{chunk_id//3}",  # Group every 3 chunks
                    'video_title': f"Christ Chapel BC Sermon {video_id}",
                    'channel_title': "Christ Chapel BC",
                    'published_at': "2024-01-01T00:00:00Z",
                    'chapter_title': f"Chapter {chunk_id//3 + 1}",
                    'language': "en",
                    'source_url': f"https://youtu.be/{video_id}?t={int(current_time)}",
                    'url': f"https://youtu.be/{video_id}?t={int(current_time)}"
                }
                chunks.append(chunk)
                chunk_id += 1
            
            current_time = end_time - overlap
        
        return chunks
    
    def embed_chunks_v4(self, chunks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Embed chunks using Cohere v4.0."""
        try:
            logger.info(f"🔢 Embedding {len(chunks)} chunks with v4.0...")
            
            url = "https://api.cohere.com/v2/embed"
            headers = {
                "Authorization": f"Bearer {self.cohere_api_key}",
                "Content-Type": "application/json"
            }
            
            # Process in batches
            batch_size = 96
            all_vectors = []
            
            for i in range(0, len(chunks), batch_size):
                batch = chunks[i:i + batch_size]
                texts = [chunk['text'] for chunk in batch]
                
                payload = {
                    "model": "embed-v4.0",
                    "texts": texts,
                    "input_type": "search_document",
                    "embedding_types": ["float"]
                }
                
                resp = requests.post(url, headers=headers, json=payload, timeout=60)
                if resp.status_code == 200:
                    data = resp.json()
                    vectors = data.get("embeddings", {}).get("float", [])
                    all_vectors.extend(vectors)
                    logger.info(f"✅ Processed batch {i//batch_size + 1}/{(len(chunks) + batch_size - 1)//batch_size}")
                else:
                    logger.error(f"❌ Embedding batch failed: {resp.status_code} {resp.text}")
                    return []
            
            # Add vectors to chunks
            for i, chunk in enumerate(chunks):
                if i < len(all_vectors):
                    chunk['vector'] = all_vectors[i]
            
            return chunks
            
        except Exception as e:
            logger.error(f"❌ v4.0 embedding failed: {e}")
            return []
    
    def upsert_to_pinecone(self, chunks: List[Dict[str, Any]]):
        """Upsert chunks to Pinecone."""
        try:
            logger.info(f"📤 Upserting {len(chunks)} chunks to Pinecone...")
            
            # Prepare vectors for upsert
            vectors = []
            for chunk in chunks:
                if 'vector' in chunk:
                    vectors.append({
                        'id': chunk['id'],
                        'values': chunk['vector'],
                        'metadata': {
                            'video_id': chunk['video_id'],
                            'video_title': chunk['video_title'],
                            'channel_title': chunk['channel_title'],
                            'published_at': chunk['published_at'],
                            'start_sec': chunk['start_sec'],
                            'end_sec': chunk['end_sec'],
                            'duration': chunk['duration'],
                            'timestamp': chunk['timestamp'],
                            'parent_id': chunk['parent_id'],
                            'chapter_title': chunk['chapter_title'],
                            'chunk_ix': chunk['chunk_ix'],
                            'language': chunk['language'],
                            'source_url': chunk['source_url'],
                            'url': chunk['url'],
                            'text': chunk['text']
                        }
                    })
            
            # Upsert in batches
            batch_size = 100
            for i in range(0, len(vectors), batch_size):
                batch = vectors[i:i + batch_size]
                self.index.upsert(vectors=batch)
                logger.info(f"✅ Upserted batch {i//batch_size + 1}/{(len(vectors) + batch_size - 1)//batch_size}")
            
            logger.info("✅ All chunks upserted to Pinecone")
            
        except Exception as e:
            logger.error(f"❌ Pinecone upsert failed: {e}")
    
    def process_video(self, video_id: str) -> bool:
        """Process a single video."""
        try:
            logger.info(f"🎥 Processing video: {video_id}")
            
            # Get transcript
            transcript_data = self.get_video_transcript(video_id)
            if not transcript_data:
                logger.warning(f"❌ No transcript for {video_id}")
                return False
            
            # Segment transcript
            chunks = self.segment_transcript(transcript_data)
            logger.info(f"📝 Created {len(chunks)} chunks")
            
            # Embed chunks
            embedded_chunks = self.embed_chunks_v4(chunks)
            if not embedded_chunks:
                logger.error(f"❌ Embedding failed for {video_id}")
                return False
            
            # Upsert to Pinecone
            self.upsert_to_pinecone(embedded_chunks)
            
            logger.info(f"✅ Successfully processed {video_id}")
            return True
            
        except Exception as e:
            logger.error(f"❌ Failed to process {video_id}: {e}")
            return False
    
    def upgrade_system(self):
        """Complete system upgrade to v4.0."""
        try:
            logger.info("🚀 Starting v4.0 system upgrade...")
            
            # Test v4.0 embedding
            if not self.test_v4_embedding():
                raise RuntimeError("v4.0 embedding test failed")
            
            # Initialize Pinecone
            self._init_pinecone()
            
            # Delete old index
            self.delete_old_index()
            
            # Create new v4.0 index
            self.create_v4_index()
            
            # Process all videos
            successful = 0
            failed = 0
            
            for video_id in self.video_ids:
                if self.process_video(video_id):
                    successful += 1
                else:
                    failed += 1
                
                # Small delay between videos
                time.sleep(2)
            
            logger.info(f"🎉 Upgrade complete! Success: {successful}, Failed: {failed}")
            
            # Test the new system
            self.test_search()
            
        except Exception as e:
            logger.error(f"❌ Upgrade failed: {e}")
            raise
    
    def test_search(self):
        """Test search functionality."""
        try:
            logger.info("🔍 Testing search functionality...")
            
            # Create a simple test query
            test_query = "How do I handle conflict with my spouse biblically?"
            
            # Get embedding
            url = "https://api.cohere.com/v2/embed"
            headers = {
                "Authorization": f"Bearer {self.cohere_api_key}",
                "Content-Type": "application/json"
            }
            payload = {
                "model": "embed-v4.0",
                "texts": [test_query],
                "input_type": "search_query",
                "embedding_types": ["float"]
            }
            
            resp = requests.post(url, headers=headers, json=payload, timeout=30)
            if resp.status_code != 200:
                logger.error(f"❌ Test embedding failed: {resp.status_code}")
                return
            
            data = resp.json()
            query_vector = data.get("embeddings", {}).get("float", [[]])[0]
            
            # Search Pinecone
            results = self.index.query(
                vector=query_vector,
                top_k=5,
                include_metadata=True
            )
            
            logger.info(f"✅ Search test successful! Found {len(results.matches)} results")
            for i, match in enumerate(results.matches[:3], 1):
                logger.info(f"  {i}. Score: {match.score:.3f} | {match.metadata.get('timestamp', 'N/A')} | {match.metadata.get('text', '')[:100]}...")
            
        except Exception as e:
            logger.error(f"❌ Search test failed: {e}")


def main():
    """Run the v4.0 upgrade."""
    print("🚀 COHERE V4.0 SYSTEM UPGRADE")
    print("=" * 50)
    
    upgrader = V4Upgrader()
    upgrader.upgrade_system()


if __name__ == "__main__":
    main()
