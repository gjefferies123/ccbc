"""FastAPI application for the hybrid RAG pipeline."""

import logging
from typing import List, Dict, Any, Optional, Union
import time
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from config import Config
from ingest.fetch_youtube import YouTubeFetcher, VideoInfo, TranscriptInfo
from ingest.segmenter import TranscriptSegmenter, SegmentMetadata
from ingest.upsert import get_upserter
from search.hybrid import get_hybrid_searcher, QueryEnhancer
from search.rerank import get_hybrid_reranker
from search.parent_expand import get_parent_expander
from search.compress import get_contextual_compressor
from ingest.fetch_youtube import create_source_url

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


# Pydantic models for API
class IngestVideoRequest(BaseModel):
    """Request to ingest videos by ID."""
    video_ids: List[str] = Field(..., description="List of YouTube video IDs")
    force_update: bool = Field(False, description="Force update even if video exists")


class IngestChannelRequest(BaseModel):
    """Request to ingest videos from a channel."""
    channel_id: str = Field(..., description="YouTube channel ID")
    max_videos: int = Field(50, description="Maximum number of videos to ingest")
    force_update: bool = Field(False, description="Force update even if videos exist")


class QueryRequest(BaseModel):
    """Request to query the RAG system."""
    q: str = Field(..., description="Search query")
    k: int = Field(5, description="Number of final results to return")
    alpha: float = Field(0.5, description="Dense/sparse weight (0.0=pure sparse, 1.0=pure dense)")
    use_hyde: bool = Field(False, description="Use HyDE query enhancement")
    use_multi_query: bool = Field(True, description="Use multi-query enhancement")
    include_neighbors: bool = Field(True, description="Include neighboring chunks for context")
    filters: Optional[Dict[str, Any]] = Field(None, description="Metadata filters")


class SourceInfo(BaseModel):
    """Source information for a result."""
    video_title: str
    url: str
    start: float
    end: float
    reason: str


class QueryResponse(BaseModel):
    """Response from the RAG system."""
    answer: str
    sources: List[SourceInfo]
    metadata: Dict[str, Any] = Field(default_factory=dict)


class IngestResponse(BaseModel):
    """Response from ingestion."""
    success: bool
    message: str
    videos_processed: int
    chunks_created: int
    errors: List[str] = Field(default_factory=list)


class HealthResponse(BaseModel):
    """Health check response."""
    status: str
    timestamp: float
    components: Dict[str, Any]


# Global components
youtube_fetcher = None
segmenter = None
upserter = None
hybrid_searcher = None
reranker = None
parent_expander = None
compressor = None
query_enhancer = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan manager."""
    # Startup
    logger.info("Starting RAG pipeline application...")
    
    try:
        # Validate configuration
        Config.validate()
        
        # Initialize global components
        global youtube_fetcher, segmenter, upserter, hybrid_searcher
        global reranker, parent_expander, compressor, query_enhancer
        
        youtube_fetcher = YouTubeFetcher()
        segmenter = TranscriptSegmenter()
        upserter = get_upserter()
        hybrid_searcher = get_hybrid_searcher()
        reranker = get_hybrid_reranker()
        parent_expander = get_parent_expander()
        compressor = get_contextual_compressor()
        query_enhancer = QueryEnhancer()
        
        logger.info("All components initialized successfully")
        
    except Exception as e:
        logger.error(f"Failed to initialize application: {e}")
        raise
    
    yield
    
    # Shutdown
    logger.info("Shutting down RAG pipeline application...")


# Create FastAPI app
app = FastAPI(
    title="RAGTube - Hybrid RAG for YouTube Transcripts",
    description="Production-ready hybrid RAG pipeline for YouTube transcripts with Pinecone and reranking",
    version="1.0.0",
    lifespan=lifespan
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/healthz", response_model=HealthResponse)
async def health_check():
    """Health check endpoint."""
    timestamp = time.time()
    
    # Check component status
    components = {
        "pinecone": {"available": upserter is not None},
        "dense_encoder": {"available": hybrid_searcher.dense_encoder.is_loaded},
        "sparse_encoder": {"available": hybrid_searcher.sparse_encoder.is_fitted},
        "reranker": reranker.get_reranker_info() if reranker else {"available": False},
        "youtube_api": {"available": youtube_fetcher.youtube is not None if youtube_fetcher else False}
    }
    
    # Overall status
    critical_components = ["pinecone", "dense_encoder"]
    status = "healthy" if all(
        components[comp]["available"] for comp in critical_components
    ) else "degraded"
    
    return HealthResponse(
        status=status,
        timestamp=timestamp,
        components=components
    )


@app.post("/ingest", response_model=IngestResponse)
async def ingest_videos(request: IngestVideoRequest, background_tasks: BackgroundTasks):
    """Ingest videos by their IDs."""
    logger.info(f"Ingest request received for {len(request.video_ids)} videos")
    
    try:
        # Process videos
        result = await _process_videos(
            video_ids=request.video_ids,
            force_update=request.force_update
        )
        
        return result
        
    except Exception as e:
        logger.error(f"Ingestion failed: {e}")
        raise HTTPException(status_code=500, detail=f"Ingestion failed: {str(e)}")


@app.post("/ingest/channel", response_model=IngestResponse)
async def ingest_channel(request: IngestChannelRequest, background_tasks: BackgroundTasks):
    """Ingest videos from a YouTube channel."""
    logger.info(f"Channel ingest request received for channel {request.channel_id}")
    
    try:
        # Get video IDs from channel
        if not youtube_fetcher.youtube:
            raise HTTPException(
                status_code=400, 
                detail="YouTube API key required for channel ingestion"
            )
        
        video_ids = youtube_fetcher.get_channel_videos(
            request.channel_id, 
            request.max_videos
        )
        
        if not video_ids:
            return IngestResponse(
                success=False,
                message=f"No videos found for channel {request.channel_id}",
                videos_processed=0,
                chunks_created=0
            )
        
        # Process videos
        result = await _process_videos(
            video_ids=video_ids,
            force_update=request.force_update
        )
        
        return result
        
    except Exception as e:
        logger.error(f"Channel ingestion failed: {e}")
        raise HTTPException(status_code=500, detail=f"Channel ingestion failed: {str(e)}")


@app.post("/query", response_model=QueryResponse)
async def query_rag(request: QueryRequest):
    """Query the RAG system."""
    logger.info(f"Query request received: '{request.q[:50]}...'")
    
    start_time = time.time()
    
    try:
        # Prepare queries
        queries = [request.q]
        
        if request.use_multi_query:
            queries = query_enhancer.generate_multi_queries(request.q)
            logger.debug(f"Generated {len(queries)} query variations")
        
        if request.use_hyde:
            hyde_query = query_enhancer.generate_hyde_query(request.q)
            queries.append(hyde_query)
            logger.debug("Added HyDE query")
        
        # Perform hybrid search for all queries
        all_results = []
        for query in queries:
            results = hybrid_searcher.search(
                query=query,
                top_k=Config.DEFAULT_TOP_K,
                alpha=request.alpha,
                filters=request.filters
            )
            all_results.extend(results)
        
        # Deduplicate by ID and take top results
        seen_ids = set()
        deduplicated_results = []
        for result in all_results:
            if result.id not in seen_ids:
                deduplicated_results.append(result)
                seen_ids.add(result.id)
        
        # Take top results
        top_results = deduplicated_results[:Config.DEFAULT_TOP_K]
        
        if not top_results:
            return QueryResponse(
                answer="No relevant content found for your query.",
                sources=[],
                metadata={
                    "query_time": time.time() - start_time,
                    "results_found": 0
                }
            )
        
        # Rerank results
        rerank_start = time.time()
        reranked_results = reranker.rerank_search_results(
            query=request.q,
            search_results=top_results,
            top_k=request.k
        )
        rerank_time = time.time() - rerank_start
        
        # Expand with parent context
        expanded_results = parent_expander.expand_rerank_results(
            reranked_results,
            include_neighbors=request.include_neighbors
        )
        
        # Compress context
        compressed_results = compressor.compress_results(
            expanded_results,
            query=request.q
        )
        
        # Generate answer and sources
        answer = _generate_answer(compressed_results)
        sources = _generate_sources(compressed_results)
        
        total_time = time.time() - start_time
        
        # Prepare metadata
        metadata = {
            "query_time": total_time,
            "rerank_time": rerank_time,
            "results_found": len(top_results),
            "final_results": len(compressed_results),
            "reranker_used": reranker.get_reranker_info()["type"],
            "total_tokens": sum(r.compressed_token_count for r in compressed_results),
            "compression_ratio": (
                sum(r.compression_ratio for r in compressed_results) / len(compressed_results)
                if compressed_results else 0
            )
        }
        
        logger.info(f"Query completed in {total_time:.2f}s, returned {len(sources)} sources")
        
        return QueryResponse(
            answer=answer,
            sources=sources,
            metadata=metadata
        )
        
    except Exception as e:
        logger.error(f"Query failed: {e}")
        raise HTTPException(status_code=500, detail=f"Query failed: {str(e)}")


async def _process_videos(video_ids: List[str], force_update: bool = False) -> IngestResponse:
    """Process a list of videos for ingestion."""
    errors = []
    videos_processed = 0
    total_chunks = 0
    
    # Get video information
    video_infos = youtube_fetcher.get_video_info(video_ids)
    video_info_map = {vi.video_id: vi for vi in video_infos}
    
    for video_id in video_ids:
        try:
            # Check if video already exists
            if not force_update and upserter.check_video_exists(video_id):
                logger.info(f"Video {video_id} already exists, skipping")
                continue
            
            # Get video info
            video_info = video_info_map.get(video_id)
            if not video_info:
                errors.append(f"Could not fetch info for video {video_id}")
                continue
            
            # Get transcript
            transcript_info = youtube_fetcher.get_transcript(video_id)
            if not transcript_info:
                errors.append(f"No transcript available for video {video_id}")
                continue
            
            # Extract chapters from description
            chapters = youtube_fetcher.extract_chapters_from_description(video_info.description)
            
            # Create segment metadata
            metadata = SegmentMetadata(
                video_id=video_info.video_id,
                video_title=video_info.title,
                channel_title=video_info.channel_title,
                published_at=video_info.published_at,
                language=transcript_info.language,
                source_url=f"https://youtu.be/{video_id}"
            )
            
            # Segment transcript
            parent_segments, child_chunks = segmenter.segment_transcript(
                transcript_info.transcript_items,
                metadata,
                chapters
            )
            
            if not child_chunks:
                errors.append(f"No valid chunks created for video {video_id}")
                continue
            
            # Delete existing chunks if updating
            if force_update:
                upserter.delete_video_chunks(video_id)
            
            # Upsert to Pinecone
            video_info_dict = {
                'video_id': video_info.video_id,
                'title': video_info.title,
                'channel_title': video_info.channel_title,
                'published_at': video_info.published_at,
                'language': transcript_info.language,
                'view_count': video_info.view_count,
                'like_count': video_info.like_count,
                'is_generated_transcript': transcript_info.is_generated
            }
            
            upsert_stats = upserter.upsert_video_chunks(
                child_chunks,
                parent_segments,
                video_info_dict
            )
            
            # Save manifest
            youtube_fetcher.save_manifest(
                video_info,
                transcript_info,
                chapters,
                parent_segments,
                child_chunks
            )
            
            videos_processed += 1
            total_chunks += upsert_stats['upserted']
            
            logger.info(f"Successfully processed video {video_id}: {len(child_chunks)} chunks")
            
        except Exception as e:
            error_msg = f"Failed to process video {video_id}: {str(e)}"
            logger.error(error_msg)
            errors.append(error_msg)
    
    success = videos_processed > 0
    message = f"Processed {videos_processed}/{len(video_ids)} videos successfully"
    
    if errors:
        message += f" with {len(errors)} errors"
    
    return IngestResponse(
        success=success,
        message=message,
        videos_processed=videos_processed,
        chunks_created=total_chunks,
        errors=errors
    )


def _generate_answer(compressed_results: List) -> str:
    """Generate answer from compressed results.
    
    For now, this creates a simple concatenated answer.
    In production, this would use an LLM to generate a coherent answer.
    """
    if not compressed_results:
        return "No relevant information found."
    
    # Simple approach: concatenate the most relevant text snippets
    answer_parts = []
    
    for i, result in enumerate(compressed_results[:3]):  # Use top 3 results
        text = result.compressed_text
        
        # Clean up the text
        if text.startswith(">>>") and text.endswith("<<<"):
            text = text[3:-3].strip()
        
        # Remove context markers
        text = text.replace("[Context:", "").replace("]", "")
        
        answer_parts.append(text)
    
    answer = " ".join(answer_parts)
    
    # Add a note about the sources
    if len(compressed_results) > 1:
        answer += f"\n\n(Based on {len(compressed_results)} video segments)"
    
    return answer


def _generate_sources(compressed_results: List) -> List[SourceInfo]:
    """Generate source information from compressed results."""
    sources = []
    
    for result in compressed_results:
        original_result = result.original_result.original_result
        metadata = original_result.metadata
        
        # Create reason based on relevance
        reason = f"Discusses the topic with relevance score {result.relevance_score:.2f}"
        
        # Add chapter context if available
        chapter_title = metadata.get('chapter_title', '')
        if chapter_title:
            reason += f" in chapter '{chapter_title}'"
        
        source = SourceInfo(
            video_title=metadata.get('video_title', 'Unknown Video'),
            url=create_source_url(metadata.get('video_id', ''), original_result.start_sec),
            start=original_result.start_sec,
            end=original_result.end_sec,
            reason=reason
        )
        
        sources.append(source)
    
    return sources


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
