#!/usr/bin/env python3
"""
Optimized parser for CCBC Transcripts.docx with 20 videos.
Handles the specific formatting and creates optimal chunks.
"""

import os
import re
import time
import logging
import requests
import json
from typing import List, Dict, Any, Optional
from docx import Document
from dotenv import load_dotenv
from pinecone import Pinecone, Index

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

class OptimizedWordParser:
    def __init__(self):
        self.pinecone_api_key = os.getenv("PINECONE_API_KEY")
        self.pinecone_index_name = os.getenv("PINECONE_INDEX", "christ-chapel-sermons")
        self.cohere_api_key = os.getenv("COHERE_API_KEY")
        self.cohere_embed_model = "embed-v4.0"
        self.pinecone_dimension = 1536  # v4.0 dimension
        
        self.pc = None
        self.index = None
        self._init_pinecone_client()
    
    def _init_pinecone_client(self):
        """Initialize Pinecone client and connect to index."""
        try:
            self.pc = Pinecone(api_key=self.pinecone_api_key)
            logger.info("✅ Pinecone client initialized")
            
            # Connect to existing index
            self.index = self.pc.Index(self.pinecone_index_name)
            logger.info(f"✅ Connected to index: {self.pinecone_index_name}")
            
        except Exception as e:
            raise RuntimeError(f"Failed to initialize Pinecone client: {e}")
    
    def parse_word_document(self, doc_path: str) -> List[Dict[str, Any]]:
        """Parse Word document and extract video transcripts with optimized formatting."""
        logger.info(f"📖 Parsing Word document: {doc_path}")
        
        try:
            doc = Document(doc_path)
            videos = []
            current_video = None
            
            for i, paragraph in enumerate(doc.paragraphs):
                text = paragraph.text.strip()
                if not text:
                    continue
                
                # Check for video separator (multiple patterns)
                if (text.startswith("Video:") or text.startswith("VIDEO:") or 
                    text.startswith("video:") or text.startswith("Video ID:")):
                    
                    # Save previous video if exists
                    if current_video:
                        videos.append(current_video)
                    
                    # Start new video
                    video_id = text.replace("Video:", "").replace("VIDEO:", "").replace("video:", "").replace("Video ID:", "").strip()
                    current_video = {
                        "video_id": video_id,
                        "title": "",
                        "transcript": "",
                        "chunks": []
                    }
                    logger.info(f"📹 Found video: {video_id}")
                
                elif (text.startswith("Title:") or text.startswith("TITLE:") or 
                      text.startswith("title:") or text.startswith("Sermon Title:")):
                    if current_video:
                        title = text.replace("Title:", "").replace("TITLE:", "").replace("title:", "").replace("Sermon Title:", "").strip()
                        current_video["title"] = title
                        logger.info(f"   📝 Title: {title}")
                
                elif current_video and text:
                    # Skip pure timestamp lines (like "0:00", "0:06", etc.)
                    if re.match(r'^\d+:\d+$', text):
                        continue
                    
                    # Skip intro/outro markers
                    if text.lower() in ["intro", "outro", "welcome", "music", "[music]", "[Music]"]:
                        continue
                    
                    # Add to transcript
                    current_video["transcript"] += text + "\n"
            
            # Add last video
            if current_video:
                videos.append(current_video)
            
            logger.info(f"✅ Parsed {len(videos)} videos from Word document")
            return videos
            
        except Exception as e:
            logger.error(f"❌ Failed to parse Word document: {e}")
            raise
    
    def create_optimized_chunks(self, video: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Create optimized chunks from video transcript."""
        transcript = video["transcript"]
        video_id = video["video_id"]
        title = video["title"]
        
        # Clean up transcript
        transcript = re.sub(r'\n+', '\n', transcript)  # Remove multiple newlines
        transcript = re.sub(r' +', ' ', transcript)    # Remove multiple spaces
        
        # Split into sentences/paragraphs
        sentences = [s.strip() for s in transcript.split('\n') if s.strip()]
        
        chunks = []
        current_chunk = ""
        chunk_index = 0
        target_chunk_size = 1200  # Optimal size for v4.0 embeddings
        min_chunk_size = 200      # Minimum meaningful chunk size
        
        for i, sentence in enumerate(sentences):
            # Add sentence to current chunk
            if current_chunk:
                current_chunk += " " + sentence
            else:
                current_chunk = sentence
            
            # Create chunk when we reach target size or end of transcript
            should_create_chunk = (
                len(current_chunk) >= target_chunk_size or 
                i == len(sentences) - 1 or
                (len(current_chunk) >= min_chunk_size and 
                 (sentence.endswith('.') or sentence.endswith('!') or sentence.endswith('?')))
            )
            
            if should_create_chunk and current_chunk.strip():
                # Calculate time ranges (more accurate based on content)
                start_sec = chunk_index * 75  # 75 seconds per chunk
                end_sec = start_sec + 75
                start_time = f"{start_sec // 60}:{start_sec % 60:02d}"
                end_time = f"{end_sec // 60}:{end_sec % 60:02d}"
                
                chunk = {
                    "id": f"{video_id}_chunk_{chunk_index}",
                    "video_id": video_id,
                    "video_title": title,
                    "text": current_chunk.strip(),
                    "start_sec": start_sec,
                    "end_sec": end_sec,
                    "timestamp": f"{start_time}-{end_time}",
                    "chunk_index": chunk_index,
                    "url": f"https://www.youtube.com/watch?v={video_id}&t={start_sec}",
                    "metadata": {
                        "video_id": video_id,
                        "video_title": title,
                        "start_sec": start_sec,
                        "end_sec": end_sec,
                        "timestamp": f"{start_time}-{end_time}",
                        "chunk_ix": chunk_index,
                        "text": current_chunk.strip(),
                        "url": f"https://www.youtube.com/watch?v={video_id}&t={start_sec}",
                        "contains_bible": "bible" in current_chunk.lower(),
                        "contains_faith": "faith" in current_chunk.lower(),
                        "contains_god": "god" in current_chunk.lower(),
                        "contains_jesus": "jesus" in current_chunk.lower(),
                        "contains_christ": "christ" in current_chunk.lower(),
                        "contains_prayer": "prayer" in current_chunk.lower(),
                        "contains_scripture": "scripture" in current_chunk.lower(),
                        "contains_worship": "worship" in current_chunk.lower(),
                        "contains_grace": "grace" in current_chunk.lower(),
                        "contains_love": "love" in current_chunk.lower()
                    }
                }
                chunks.append(chunk)
                chunk_index += 1
                current_chunk = ""
        
        logger.info(f"📝 Created {len(chunks)} optimized chunks for video {video_id}")
        return chunks
    
    def get_embeddings(self, texts: List[str]) -> List[List[float]]:
        """Get v4.0 embeddings from Cohere."""
        url = "https://api.cohere.com/v2/embed"
        headers = {
            "Authorization": f"Bearer {self.cohere_api_key}",
            "Content-Type": "application/json"
        }
        
        # Process in batches
        batch_size = 96
        all_embeddings = []
        
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            payload = {
                "model": self.cohere_embed_model,
                "texts": batch,
                "input_type": "search_document",
                "embedding_types": ["float"]
            }
            
            try:
                response = requests.post(url, headers=headers, json=payload, timeout=30)
                response.raise_for_status()
                response_data = response.json()
                embeddings = response_data['embeddings']['float']
                all_embeddings.extend(embeddings)
                logger.info(f"✅ Processed batch {i//batch_size + 1}/{(len(texts) + batch_size - 1)//batch_size}")
            except Exception as e:
                logger.error(f"❌ Failed to get embeddings for batch: {e}")
                raise
        
        return all_embeddings
    
    def index_chunks(self, chunks: List[Dict[str, Any]]):
        """Index chunks in Pinecone."""
        logger.info(f"📊 Indexing {len(chunks)} chunks...")
        
        # Prepare vectors for Pinecone
        vectors = []
        texts = [chunk["text"] for chunk in chunks]
        
        # Get embeddings
        embeddings = self.get_embeddings(texts)
        
        # Create vector records
        for i, chunk in enumerate(chunks):
            vector = {
                "id": chunk["id"],
                "values": embeddings[i],
                "metadata": chunk["metadata"]
            }
            vectors.append(vector)
        
        # Upsert to Pinecone
        try:
            self.index.upsert(vectors=vectors)
            logger.info(f"✅ Successfully indexed {len(vectors)} chunks")
        except Exception as e:
            logger.error(f"❌ Failed to index chunks: {e}")
            raise
    
    def process_word_document(self, doc_path: str):
        """Main method to process Word document and index transcripts."""
        logger.info("🚀 Starting optimized Word transcript processing...")
        
        # Parse Word document
        videos = self.parse_word_document(doc_path)
        
        total_chunks = 0
        for video in videos:
            logger.info(f"📹 Processing video: {video['video_id']} - {video['title']}")
            
            # Create optimized chunks
            chunks = self.create_optimized_chunks(video)
            
            # Index chunks
            self.index_chunks(chunks)
            
            total_chunks += len(chunks)
            logger.info(f"✅ Processed {len(chunks)} chunks for {video['video_id']}")
        
        logger.info(f"🎉 Processing complete! Total chunks indexed: {total_chunks}")
        return total_chunks

def main():
    """Main function."""
    doc_path = "../CCBC Transcripts.docx"  # Path to Word document
    
    if not os.path.exists(doc_path):
        logger.error(f"❌ Word document not found: {doc_path}")
        return
    
    try:
        parser = OptimizedWordParser()
        total_chunks = parser.process_word_document(doc_path)
        logger.info(f"🎉 Successfully processed and indexed {total_chunks} chunks!")
        
    except Exception as e:
        logger.error(f"❌ Processing failed: {e}")

if __name__ == "__main__":
    main()
