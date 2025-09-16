"""Transcript segmentation into parent/child hierarchy for optimal RAG performance."""

import logging
from typing import List, Dict, Any, Tuple, Optional
from dataclasses import dataclass
from config import Config
from utils.text import merge_transcript_items, clean_transcript_text, count_tokens

logger = logging.getLogger(__name__)


@dataclass
class SegmentMetadata:
    """Metadata for a segment."""
    video_id: str
    video_title: str
    channel_title: str
    published_at: str
    language: str
    source_url: str


@dataclass
class ParentSegment:
    """Parent segment representing a larger section of the video."""
    id: str
    video_id: str
    start_sec: float
    end_sec: float
    text: str
    chapter_title: Optional[str] = None
    token_count: int = 0


@dataclass
class ChildChunk:
    """Child chunk for embedding and search."""
    id: str
    video_id: str
    parent_id: str
    start_sec: float
    end_sec: float
    text: str
    chunk_ix: int
    prev_id: Optional[str] = None
    next_id: Optional[str] = None
    token_count: int = 0


class TranscriptSegmenter:
    """Segments transcripts into parent/child hierarchy for hybrid search."""
    
    def __init__(self, 
                 parent_duration: int = None,
                 child_min_duration: int = None,
                 child_max_duration: int = None,
                 child_overlap: int = None):
        """Initialize the segmenter with configuration.
        
        Args:
            parent_duration: Target duration for parent segments (seconds)
            child_min_duration: Minimum duration for child chunks (seconds)
            child_max_duration: Maximum duration for child chunks (seconds)
            child_overlap: Overlap between child chunks (seconds)
        """
        self.parent_duration = parent_duration or Config.PARENT_SEGMENT_DURATION
        self.child_min_duration = child_min_duration or Config.CHILD_CHUNK_MIN_DURATION
        self.child_max_duration = child_max_duration or Config.CHILD_CHUNK_MAX_DURATION
        self.child_overlap = child_overlap or Config.CHILD_CHUNK_OVERLAP
    
    def segment_transcript(self, 
                          transcript_items: List[Dict[str, Any]],
                          metadata: SegmentMetadata,
                          chapters: Optional[List[Dict[str, Any]]] = None) -> Tuple[List[ParentSegment], List[ChildChunk]]:
        """Segment transcript into parent segments and child chunks.
        
        Args:
            transcript_items: Raw transcript items with 'text', 'start', 'duration'
            metadata: Video metadata
            chapters: Optional chapter information with 'title', 'start_time'
            
        Returns:
            Tuple of (parent_segments, child_chunks)
        """
        if not transcript_items:
            return [], []
        
        logger.info(f"Segmenting transcript for video {metadata.video_id} with {len(transcript_items)} items")
        
        # Step 1: Merge transcript items into sentences
        sentences = merge_transcript_items(transcript_items)
        
        if not sentences:
            return [], []
        
        # Step 2: Create parent segments
        parent_segments = self._create_parent_segments(sentences, metadata, chapters)
        
        # Step 3: Create child chunks from sentences
        child_chunks = self._create_child_chunks(sentences, metadata, parent_segments)
        
        # Step 4: Set up adjacency links for child chunks
        self._setup_adjacency_links(child_chunks)
        
        logger.info(f"Created {len(parent_segments)} parent segments and {len(child_chunks)} child chunks")
        
        return parent_segments, child_chunks
    
    def _create_parent_segments(self, 
                              sentences: List[Dict[str, Any]],
                              metadata: SegmentMetadata,
                              chapters: Optional[List[Dict[str, Any]]]) -> List[ParentSegment]:
        """Create parent segments from sentences.
        
        Args:
            sentences: Merged sentence items
            metadata: Video metadata
            chapters: Optional chapter information
            
        Returns:
            List of parent segments
        """
        parent_segments = []
        
        if chapters:
            # Use chapter boundaries for parent segments
            parent_segments = self._create_chapter_based_parents(sentences, metadata, chapters)
        else:
            # Create time-based parent segments
            parent_segments = self._create_time_based_parents(sentences, metadata)
        
        # Calculate token counts for each parent
        for parent in parent_segments:
            parent.token_count = count_tokens(parent.text)
        
        return parent_segments
    
    def _create_chapter_based_parents(self, 
                                    sentences: List[Dict[str, Any]],
                                    metadata: SegmentMetadata,
                                    chapters: List[Dict[str, Any]]) -> List[ParentSegment]:
        """Create parent segments based on chapter boundaries.
        
        Args:
            sentences: Merged sentence items
            metadata: Video metadata
            chapters: Chapter information
            
        Returns:
            List of chapter-based parent segments
        """
        parent_segments = []
        
        # Sort chapters by start time
        chapters = sorted(chapters, key=lambda x: x.get('start_time', 0))
        
        for i, chapter in enumerate(chapters):
            chapter_start = chapter.get('start_time', 0)
            chapter_end = chapters[i + 1].get('start_time') if i + 1 < len(chapters) else float('inf')
            
            # Find sentences within this chapter
            chapter_sentences = [
                s for s in sentences 
                if s['start'] >= chapter_start and s['start'] < chapter_end
            ]
            
            if not chapter_sentences:
                continue
            
            # Combine sentences into chapter text
            chapter_text = ' '.join([s['text'] for s in chapter_sentences])
            chapter_text = clean_transcript_text(chapter_text)
            
            if not chapter_text.strip():
                continue
            
            parent_id = f"{metadata.video_id}:{int(chapter_start)}"
            
            parent_segment = ParentSegment(
                id=parent_id,
                video_id=metadata.video_id,
                start_sec=chapter_sentences[0]['start'],
                end_sec=chapter_sentences[-1]['end'],
                text=chapter_text,
                chapter_title=chapter.get('title', f"Chapter {i + 1}")
            )
            
            parent_segments.append(parent_segment)
        
        return parent_segments
    
    def _create_time_based_parents(self, 
                                 sentences: List[Dict[str, Any]],
                                 metadata: SegmentMetadata) -> List[ParentSegment]:
        """Create parent segments based on time duration.
        
        Args:
            sentences: Merged sentence items
            metadata: Video metadata
            
        Returns:
            List of time-based parent segments
        """
        parent_segments = []
        
        if not sentences:
            return parent_segments
        
        current_segment_sentences = []
        segment_start_time = sentences[0]['start']
        
        for sentence in sentences:
            # Check if we should start a new segment
            if (sentence['start'] - segment_start_time >= self.parent_duration and 
                current_segment_sentences):
                
                # Create parent segment from accumulated sentences
                parent_segment = self._create_parent_from_sentences(
                    current_segment_sentences, metadata, segment_start_time
                )
                parent_segments.append(parent_segment)
                
                # Start new segment
                current_segment_sentences = [sentence]
                segment_start_time = sentence['start']
            else:
                current_segment_sentences.append(sentence)
        
        # Create final segment
        if current_segment_sentences:
            parent_segment = self._create_parent_from_sentences(
                current_segment_sentences, metadata, segment_start_time
            )
            parent_segments.append(parent_segment)
        
        return parent_segments
    
    def _create_parent_from_sentences(self, 
                                    sentences: List[Dict[str, Any]],
                                    metadata: SegmentMetadata,
                                    start_time: float) -> ParentSegment:
        """Create a parent segment from a list of sentences.
        
        Args:
            sentences: List of sentence items
            metadata: Video metadata
            start_time: Start time for the segment
            
        Returns:
            Parent segment
        """
        combined_text = ' '.join([s['text'] for s in sentences])
        combined_text = clean_transcript_text(combined_text)
        
        parent_id = f"{metadata.video_id}:{int(start_time)}"
        
        return ParentSegment(
            id=parent_id,
            video_id=metadata.video_id,
            start_sec=sentences[0]['start'],
            end_sec=sentences[-1]['end'],
            text=combined_text
        )
    
    def _create_child_chunks(self, 
                           sentences: List[Dict[str, Any]],
                           metadata: SegmentMetadata,
                           parent_segments: List[ParentSegment]) -> List[ChildChunk]:
        """Create child chunks with sliding window approach.
        
        Args:
            sentences: Merged sentence items
            metadata: Video metadata
            parent_segments: Parent segments for mapping
            
        Returns:
            List of child chunks
        """
        child_chunks = []
        
        if not sentences:
            return child_chunks
        
        # Create mapping from time to parent segment
        parent_map = {}
        for parent in parent_segments:
            for t in range(int(parent.start_sec), int(parent.end_sec) + 1):
                parent_map[t] = parent.id
        
        chunk_ix = 0
        i = 0
        
        while i < len(sentences):
            # Start building a chunk
            chunk_sentences = []
            chunk_start = sentences[i]['start']
            chunk_end = chunk_start
            
            # Add sentences until we reach the minimum duration
            j = i
            while (j < len(sentences) and 
                   chunk_end - chunk_start < self.child_min_duration):
                chunk_sentences.append(sentences[j])
                chunk_end = sentences[j]['end']
                j += 1
            
            # Continue adding sentences until we exceed max duration or hit sentence boundary
            while (j < len(sentences) and 
                   sentences[j]['end'] - chunk_start <= self.child_max_duration):
                chunk_sentences.append(sentences[j])
                chunk_end = sentences[j]['end']
                j += 1
            
            if not chunk_sentences:
                i += 1
                continue
            
            # Create child chunk
            chunk_text = ' '.join([s['text'] for s in chunk_sentences])
            chunk_text = clean_transcript_text(chunk_text)
            
            if not chunk_text.strip():
                i += 1
                continue
            
            # Find parent segment for this chunk
            chunk_mid_time = int((chunk_start + chunk_end) / 2)
            parent_id = parent_map.get(chunk_mid_time)
            
            if parent_id is None and parent_segments:
                # Fallback: find closest parent
                parent_id = min(parent_segments, 
                              key=lambda p: abs(p.start_sec - chunk_start)).id
            
            chunk_id = f"{metadata.video_id}:chunk:{chunk_ix}"
            
            child_chunk = ChildChunk(
                id=chunk_id,
                video_id=metadata.video_id,
                parent_id=parent_id,
                start_sec=chunk_start,
                end_sec=chunk_end,
                text=chunk_text,
                chunk_ix=chunk_ix,
                token_count=count_tokens(chunk_text)
            )
            
            child_chunks.append(child_chunk)
            chunk_ix += 1
            
            # Move to next chunk with overlap
            # Find the sentence that starts after (chunk_start + chunk_duration - overlap)
            next_start_time = chunk_start + (chunk_end - chunk_start) - self.child_overlap
            next_i = i
            
            for k in range(i, len(sentences)):
                if sentences[k]['start'] >= next_start_time:
                    next_i = k
                    break
            
            # Ensure we make progress
            i = max(i + 1, next_i)
        
        return child_chunks
    
    def _setup_adjacency_links(self, child_chunks: List[ChildChunk]) -> None:
        """Set up prev_id and next_id links for child chunks.
        
        Args:
            child_chunks: List of child chunks to link
        """
        for i, chunk in enumerate(child_chunks):
            if i > 0:
                chunk.prev_id = child_chunks[i - 1].id
            if i < len(child_chunks) - 1:
                chunk.next_id = child_chunks[i + 1].id
    
    def validate_segmentation(self, 
                            parent_segments: List[ParentSegment],
                            child_chunks: List[ChildChunk]) -> Dict[str, Any]:
        """Validate the segmentation results.
        
        Args:
            parent_segments: Parent segments to validate
            child_chunks: Child chunks to validate
            
        Returns:
            Validation report
        """
        report = {
            'valid': True,
            'errors': [],
            'warnings': [],
            'stats': {
                'parent_count': len(parent_segments),
                'child_count': len(child_chunks),
                'avg_parent_duration': 0,
                'avg_child_duration': 0,
                'child_duration_violations': 0
            }
        }
        
        # Validate parent segments
        if parent_segments:
            durations = [p.end_sec - p.start_sec for p in parent_segments]
            report['stats']['avg_parent_duration'] = sum(durations) / len(durations)
        
        # Validate child chunks
        if child_chunks:
            durations = [c.end_sec - c.start_sec for c in child_chunks]
            report['stats']['avg_child_duration'] = sum(durations) / len(durations)
            
            # Check duration constraints
            for chunk in child_chunks:
                duration = chunk.end_sec - chunk.start_sec
                if duration < self.child_min_duration or duration > self.child_max_duration:
                    report['stats']['child_duration_violations'] += 1
                    
                    if duration < self.child_min_duration:
                        report['warnings'].append(
                            f"Chunk {chunk.id} duration {duration:.1f}s below minimum {self.child_min_duration}s"
                        )
        
        # Check for gaps or overlaps (warnings only)
        for i in range(len(child_chunks) - 1):
            current = child_chunks[i]
            next_chunk = child_chunks[i + 1]
            
            gap = next_chunk.start_sec - current.end_sec
            if gap > self.child_overlap * 2:  # Significant gap
                report['warnings'].append(
                    f"Large gap {gap:.1f}s between chunks {current.id} and {next_chunk.id}"
                )
        
        return report
