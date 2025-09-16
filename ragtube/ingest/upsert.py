"""Pinecone indexing and upsertion for hybrid search."""

import logging
from typing import List, Dict, Any, Optional, Tuple
import time
from pathlib import Path
import pinecone
from pinecone import Pinecone, ServerlessSpec
import numpy as np

from config import Config, PINECONE_DIMENSION
from search.encoder_dense import get_dense_encoder
from search.encoder_sparse import get_sparse_encoder, ensure_fitted_on_corpus
from ingest.segmenter import ChildChunk, ParentSegment
from ingest.fetch_youtube import create_source_url

logger = logging.getLogger(__name__)


class PineconeUpserter:
    """Handles upserting embeddings and metadata to Pinecone."""
    
    def __init__(self, 
                 api_key: str = None,
                 index_name: str = None,
                 environment: str = None):
        """Initialize the Pinecone upserter.
        
        Args:
            api_key: Pinecone API key
            index_name: Pinecone index name
            environment: Pinecone environment
        """
        self.api_key = api_key or Config.PINECONE_API_KEY
        self.index_name = index_name or Config.PINECONE_INDEX
        self.environment = environment or Config.PINECONE_ENV
        
        if not self.api_key:
            raise ValueError("Pinecone API key is required")
        
        # Initialize Pinecone
        self.pc = Pinecone(api_key=self.api_key)
        self.index = None
        
        self._initialize_index()
    
    def _initialize_index(self) -> None:
        """Initialize or create the Pinecone index."""
        try:
            # Check if index exists
            existing_indexes = [index.name for index in self.pc.list_indexes()]
            
            if self.index_name not in existing_indexes:
                logger.info(f"Creating Pinecone index: {self.index_name}")
                
                # Create serverless index in us-east-1
                self.pc.create_index(
                    name=self.index_name,
                    dimension=PINECONE_DIMENSION,
                    metric="cosine",
                    spec=ServerlessSpec(
                        cloud="aws",
                        region=self.environment
                    )
                )
                
                # Wait for index to be ready
                while not self.pc.describe_index(self.index_name).status['ready']:
                    logger.info("Waiting for index to be ready...")
                    time.sleep(5)
            
            # Connect to index
            self.index = self.pc.Index(self.index_name)
            logger.info(f"Connected to Pinecone index: {self.index_name}")
            
        except Exception as e:
            logger.error(f"Failed to initialize Pinecone index: {e}")
            raise
    
    def upsert_video_chunks(self, 
                           child_chunks: List[ChildChunk],
                           parent_segments: List[ParentSegment],
                           video_info: Dict[str, Any],
                           batch_size: int = 100) -> Dict[str, Any]:
        """Upsert child chunks to Pinecone with hybrid embeddings.
        
        Args:
            child_chunks: List of child chunks to upsert
            parent_segments: List of parent segments for metadata
            video_info: Video metadata
            batch_size: Batch size for upserting
            
        Returns:
            Upsert statistics
        """
        if not child_chunks:
            logger.warning("No child chunks to upsert")
            return {'upserted': 0, 'errors': 0}
        
        logger.info(f"Upserting {len(child_chunks)} chunks for video {video_info.get('video_id')}")
        
        # Prepare corpus for BM25 fitting if needed
        chunk_texts = [chunk.text for chunk in child_chunks]
        sparse_encoder = ensure_fitted_on_corpus(chunk_texts)
        dense_encoder = get_dense_encoder()
        
        # Create parent lookup
        parent_lookup = {p.id: p for p in parent_segments}
        
        # Generate embeddings
        logger.info("Generating dense embeddings...")
        dense_embeddings = dense_encoder.encode_documents(chunk_texts)
        
        logger.info("Generating sparse embeddings...")
        sparse_embeddings = sparse_encoder.encode_documents(chunk_texts)
        
        # Prepare vectors for upserting
        vectors = []
        for i, chunk in enumerate(child_chunks):
            # Get parent segment
            parent = parent_lookup.get(chunk.parent_id)
            
            # Prepare metadata
            metadata = self._prepare_chunk_metadata(chunk, parent, video_info)
            
            # Prepare vector with dense and sparse values
            vector = {
                'id': chunk.id,
                'values': dense_embeddings[i].tolist(),
                'sparse_values': sparse_embeddings[i],
                'metadata': metadata
            }
            
            vectors.append(vector)
        
        # Upsert in batches
        stats = {'upserted': 0, 'errors': 0}
        
        for i in range(0, len(vectors), batch_size):
            batch = vectors[i:i + batch_size]
            
            try:
                logger.debug(f"Upserting batch {i//batch_size + 1}/{(len(vectors) + batch_size - 1)//batch_size}")
                self.index.upsert(vectors=batch)
                stats['upserted'] += len(batch)
                
                # Small delay to avoid rate limits
                time.sleep(0.1)
                
            except Exception as e:
                logger.error(f"Error upserting batch {i//batch_size + 1}: {e}")
                stats['errors'] += len(batch)
        
        logger.info(f"Upsert completed: {stats['upserted']} successful, {stats['errors']} errors")
        return stats
    
    def _prepare_chunk_metadata(self, 
                              chunk: ChildChunk,
                              parent: Optional[ParentSegment],
                              video_info: Dict[str, Any]) -> Dict[str, Any]:
        """Prepare metadata for a child chunk.
        
        Args:
            chunk: Child chunk
            parent: Parent segment (if available)
            video_info: Video information
            
        Returns:
            Metadata dictionary
        """
        metadata = {
            # Required fields
            'video_id': chunk.video_id,
            'video_title': video_info.get('title', ''),
            'channel_title': video_info.get('channel_title', ''),
            'published_at': video_info.get('published_at', ''),
            'start_sec': float(chunk.start_sec),
            'end_sec': float(chunk.end_sec),
            'parent_id': chunk.parent_id or '',
            'chunk_ix': chunk.chunk_ix,
            'language': video_info.get('language', 'en'),
            'source_url': create_source_url(chunk.video_id, chunk.start_sec),
            
            # Optional fields
            'chapter_title': parent.chapter_title if parent and parent.chapter_title else '',
            'prev_id': chunk.prev_id or '',
            'next_id': chunk.next_id or '',
            'token_count': chunk.token_count,
            'duration': float(chunk.end_sec - chunk.start_sec),
            
            # Additional metadata
            'view_count': video_info.get('view_count', 0),
            'like_count': video_info.get('like_count', 0),
            'is_generated_transcript': video_info.get('is_generated_transcript', True),
            
            # Text content (for retrieval)
            'text': chunk.text[:1000]  # Limit text length for metadata
        }
        
        # Ensure all values are JSON serializable
        for key, value in metadata.items():
            if isinstance(value, np.ndarray):
                metadata[key] = value.tolist()
            elif isinstance(value, (np.int64, np.int32)):
                metadata[key] = int(value)
            elif isinstance(value, (np.float64, np.float32)):
                metadata[key] = float(value)
        
        return metadata
    
    def delete_video_chunks(self, video_id: str) -> Dict[str, Any]:
        """Delete all chunks for a specific video.
        
        Args:
            video_id: Video ID to delete
            
        Returns:
            Deletion statistics
        """
        try:
            logger.info(f"Deleting chunks for video {video_id}")
            
            # Query for all chunks of this video
            query_response = self.index.query(
                vector=[0] * PINECONE_DIMENSION,  # Dummy vector
                filter={'video_id': video_id},
                top_k=10000,  # Large number to get all matches
                include_metadata=False
            )
            
            if not query_response.matches:
                logger.info(f"No chunks found for video {video_id}")
                return {'deleted': 0}
            
            # Extract IDs
            chunk_ids = [match.id for match in query_response.matches]
            
            # Delete in batches
            batch_size = 1000
            deleted_count = 0
            
            for i in range(0, len(chunk_ids), batch_size):
                batch_ids = chunk_ids[i:i + batch_size]
                self.index.delete(ids=batch_ids)
                deleted_count += len(batch_ids)
                
                logger.debug(f"Deleted batch {i//batch_size + 1}/{(len(chunk_ids) + batch_size - 1)//batch_size}")
            
            logger.info(f"Deleted {deleted_count} chunks for video {video_id}")
            return {'deleted': deleted_count}
            
        except Exception as e:
            logger.error(f"Error deleting video chunks: {e}")
            return {'deleted': 0, 'error': str(e)}
    
    def get_index_stats(self) -> Dict[str, Any]:
        """Get index statistics.
        
        Returns:
            Index statistics
        """
        try:
            stats = self.index.describe_index_stats()
            return {
                'total_vectors': stats.total_vector_count,
                'dimension': stats.dimension,
                'index_fullness': stats.index_fullness,
                'namespaces': stats.namespaces
            }
        except Exception as e:
            logger.error(f"Error getting index stats: {e}")
            return {}
    
    def check_video_exists(self, video_id: str) -> bool:
        """Check if a video already exists in the index.
        
        Args:
            video_id: Video ID to check
            
        Returns:
            True if video exists in index
        """
        try:
            query_response = self.index.query(
                vector=[0] * PINECONE_DIMENSION,
                filter={'video_id': video_id},
                top_k=1,
                include_metadata=False
            )
            
            return len(query_response.matches) > 0
            
        except Exception as e:
            logger.error(f"Error checking if video exists: {e}")
            return False
    
    def list_indexed_videos(self, limit: int = 100) -> List[Dict[str, Any]]:
        """List videos currently in the index.
        
        Args:
            limit: Maximum number of videos to return
            
        Returns:
            List of video metadata
        """
        try:
            # This is a workaround since Pinecone doesn't have a direct way to list all unique videos
            # We'll query with a dummy vector and collect unique video_ids
            query_response = self.index.query(
                vector=[0] * PINECONE_DIMENSION,
                top_k=min(limit * 10, 10000),  # Get more than needed to find unique videos
                include_metadata=True
            )
            
            videos = {}
            for match in query_response.matches:
                video_id = match.metadata.get('video_id')
                if video_id and video_id not in videos:
                    videos[video_id] = {
                        'video_id': video_id,
                        'video_title': match.metadata.get('video_title', ''),
                        'channel_title': match.metadata.get('channel_title', ''),
                        'published_at': match.metadata.get('published_at', ''),
                        'language': match.metadata.get('language', 'en')
                    }
                
                if len(videos) >= limit:
                    break
            
            return list(videos.values())
            
        except Exception as e:
            logger.error(f"Error listing indexed videos: {e}")
            return []


# Global instance
_upserter_instance = None


def get_upserter() -> PineconeUpserter:
    """Get or create a global Pinecone upserter instance."""
    global _upserter_instance
    if _upserter_instance is None:
        _upserter_instance = PineconeUpserter()
    return _upserter_instance
