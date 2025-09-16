"""Tests for transcript segmentation functionality."""

import unittest
from typing import List, Dict, Any

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ingest.segmenter import TranscriptSegmenter, SegmentMetadata


class TestTranscriptSegmenter(unittest.TestCase):
    """Test cases for TranscriptSegmenter."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.segmenter = TranscriptSegmenter(
            parent_duration=300,  # 5 minutes
            child_min_duration=45,
            child_max_duration=90,
            child_overlap=15
        )
        
        self.metadata = SegmentMetadata(
            video_id="test_video_123",
            video_title="Test Video",
            channel_title="Test Channel",
            published_at="2023-01-01T00:00:00Z",
            language="en",
            source_url="https://youtu.be/test_video_123"
        )
    
    def create_sample_transcript(self) -> List[Dict[str, Any]]:
        """Create a sample transcript for testing."""
        return [
            {"text": "Hello everyone", "start": 0.0, "duration": 3.0},
            {"text": "and welcome to this video", "start": 3.0, "duration": 4.0},
            {"text": "Today we're going to talk about", "start": 7.0, "duration": 5.0},
            {"text": "the importance of testing", "start": 12.0, "duration": 4.0},
            {"text": "in software development.", "start": 16.0, "duration": 4.0},
            {"text": "Testing helps us ensure", "start": 20.0, "duration": 4.0},
            {"text": "that our code works correctly", "start": 24.0, "duration": 5.0},
            {"text": "and prevents bugs", "start": 29.0, "duration": 3.0},
            {"text": "from reaching production.", "start": 32.0, "duration": 4.0},
            {"text": "Now let's dive deeper", "start": 60.0, "duration": 4.0},
            {"text": "into different types of testing", "start": 64.0, "duration": 5.0},
            {"text": "including unit tests", "start": 69.0, "duration": 4.0},
            {"text": "integration tests", "start": 73.0, "duration": 3.0},
            {"text": "and end-to-end tests.", "start": 76.0, "duration": 4.0},
            {"text": "Each type serves", "start": 120.0, "duration": 3.0},
            {"text": "a different purpose", "start": 123.0, "duration": 4.0},
            {"text": "in ensuring code quality.", "start": 127.0, "duration": 5.0},
        ]
    
    def test_basic_segmentation(self):
        """Test basic segmentation functionality."""
        transcript = self.create_sample_transcript()
        
        parent_segments, child_chunks = self.segmenter.segment_transcript(
            transcript, self.metadata
        )
        
        # Should create at least one parent and multiple children
        self.assertGreater(len(parent_segments), 0)
        self.assertGreater(len(child_chunks), 0)
        
        # Child chunks should be more numerous than parents
        self.assertGreater(len(child_chunks), len(parent_segments))
    
    def test_child_chunk_durations(self):
        """Test that child chunks meet duration constraints."""
        transcript = self.create_sample_transcript()
        
        _, child_chunks = self.segmenter.segment_transcript(
            transcript, self.metadata
        )
        
        for chunk in child_chunks:
            duration = chunk.end_sec - chunk.start_sec
            
            # Most chunks should be within the target range
            # Allow some flexibility for edge cases
            if len(child_chunks) > 1:  # Skip if only one chunk
                self.assertGreaterEqual(duration, self.segmenter.child_min_duration * 0.8)
                self.assertLessEqual(duration, self.segmenter.child_max_duration * 1.2)
    
    def test_sentence_boundaries(self):
        """Test that chunks respect sentence boundaries."""
        transcript = self.create_sample_transcript()
        
        _, child_chunks = self.segmenter.segment_transcript(
            transcript, self.metadata
        )
        
        for chunk in child_chunks:
            # Text should not be empty
            self.assertGreater(len(chunk.text.strip()), 0)
            
            # Should not break words (basic check)
            self.assertFalse(chunk.text.endswith('-'))
            self.assertFalse(chunk.text.startswith('-'))
    
    def test_chunk_ordering(self):
        """Test that chunks are in correct temporal order."""
        transcript = self.create_sample_transcript()
        
        _, child_chunks = self.segmenter.segment_transcript(
            transcript, self.metadata
        )
        
        for i in range(len(child_chunks) - 1):
            current = child_chunks[i]
            next_chunk = child_chunks[i + 1]
            
            # Next chunk should start after current chunk starts
            self.assertGreaterEqual(next_chunk.start_sec, current.start_sec)
            
            # Chunk indices should be sequential
            self.assertEqual(next_chunk.chunk_ix, current.chunk_ix + 1)
    
    def test_parent_coverage(self):
        """Test that parent segments cover the entire video."""
        transcript = self.create_sample_transcript()
        
        parent_segments, _ = self.segmenter.segment_transcript(
            transcript, self.metadata
        )
        
        if len(parent_segments) > 1:
            # Sort parents by start time
            sorted_parents = sorted(parent_segments, key=lambda p: p.start_sec)
            
            # Check coverage
            total_start = min(item['start'] for item in transcript)
            total_end = max(item['start'] + item['duration'] for item in transcript)
            
            # First parent should start near the beginning
            self.assertLessEqual(abs(sorted_parents[0].start_sec - total_start), 10.0)
            
            # Last parent should end near the end
            self.assertLessEqual(abs(sorted_parents[-1].end_sec - total_end), 10.0)
    
    def test_chapter_based_segmentation(self):
        """Test segmentation with chapter information."""
        transcript = self.create_sample_transcript()
        
        chapters = [
            {"title": "Introduction", "start_time": 0},
            {"title": "Types of Testing", "start_time": 60},
            {"title": "Best Practices", "start_time": 120}
        ]
        
        parent_segments, child_chunks = self.segmenter.segment_transcript(
            transcript, self.metadata, chapters
        )
        
        # Should create one parent per chapter
        self.assertEqual(len(parent_segments), len(chapters))
        
        # Check chapter titles are preserved
        chapter_titles = [p.chapter_title for p in parent_segments]
        expected_titles = [c["title"] for c in chapters]
        
        for title in expected_titles:
            self.assertIn(title, chapter_titles)
    
    def test_adjacency_links(self):
        """Test that adjacency links are set up correctly."""
        transcript = self.create_sample_transcript()
        
        _, child_chunks = self.segmenter.segment_transcript(
            transcript, self.metadata
        )
        
        if len(child_chunks) > 1:
            # First chunk should have no prev_id
            self.assertIsNone(child_chunks[0].prev_id)
            
            # Last chunk should have no next_id
            self.assertIsNone(child_chunks[-1].next_id)
            
            # Middle chunks should have both links
            for i in range(1, len(child_chunks) - 1):
                chunk = child_chunks[i]
                self.assertIsNotNone(chunk.prev_id)
                self.assertIsNotNone(chunk.next_id)
                
                # Links should point to correct chunks
                self.assertEqual(chunk.prev_id, child_chunks[i - 1].id)
                self.assertEqual(chunk.next_id, child_chunks[i + 1].id)
    
    def test_empty_transcript(self):
        """Test handling of empty transcript."""
        parent_segments, child_chunks = self.segmenter.segment_transcript(
            [], self.metadata
        )
        
        self.assertEqual(len(parent_segments), 0)
        self.assertEqual(len(child_chunks), 0)
    
    def test_single_item_transcript(self):
        """Test handling of single-item transcript."""
        transcript = [{"text": "Short video", "start": 0.0, "duration": 5.0}]
        
        parent_segments, child_chunks = self.segmenter.segment_transcript(
            transcript, self.metadata
        )
        
        # Should create at least one chunk even if short
        self.assertGreaterEqual(len(child_chunks), 1)
        self.assertGreaterEqual(len(parent_segments), 1)
    
    def test_validation(self):
        """Test segmentation validation."""
        transcript = self.create_sample_transcript()
        
        parent_segments, child_chunks = self.segmenter.segment_transcript(
            transcript, self.metadata
        )
        
        validation_report = self.segmenter.validate_segmentation(
            parent_segments, child_chunks
        )
        
        # Should be valid
        self.assertTrue(validation_report['valid'])
        
        # Should have reasonable stats
        stats = validation_report['stats']
        self.assertGreater(stats['parent_count'], 0)
        self.assertGreater(stats['child_count'], 0)
        self.assertGreater(stats['avg_child_duration'], 0)
    
    def test_long_transcript(self):
        """Test with a longer transcript to ensure scalability."""
        # Create a longer transcript (10 minutes)
        transcript = []
        current_time = 0.0
        
        for i in range(200):  # 200 items over 10 minutes
            transcript.append({
                "text": f"This is sentence number {i + 1}.",
                "start": current_time,
                "duration": 3.0
            })
            current_time += 3.0
        
        parent_segments, child_chunks = self.segmenter.segment_transcript(
            transcript, self.metadata
        )
        
        # Should create multiple parents (more than 1 for 10 minute video)
        self.assertGreater(len(parent_segments), 1)
        
        # Should create many child chunks
        self.assertGreater(len(child_chunks), 10)
        
        # Validate the result
        validation_report = self.segmenter.validate_segmentation(
            parent_segments, child_chunks
        )
        self.assertTrue(validation_report['valid'])


if __name__ == '__main__':
    unittest.main()
