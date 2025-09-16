"""Tests for hybrid search functionality."""

import unittest
from unittest.mock import Mock, patch, MagicMock
import numpy as np

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from search.hybrid import HybridSearcher, SearchResult


class TestHybridSearcher(unittest.TestCase):
    """Test cases for HybridSearcher."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.searcher = HybridSearcher(alpha=0.5, top_k=10)
        
        # Mock the dependencies
        self.mock_dense_encoder = Mock()
        self.mock_sparse_encoder = Mock()
        self.mock_upserter = Mock()
        
        self.searcher.dense_encoder = self.mock_dense_encoder
        self.searcher.sparse_encoder = self.mock_sparse_encoder
        self.searcher.upserter = self.mock_upserter
    
    def test_alpha_mixing(self):
        """Test alpha parameter effects on search behavior."""
        # Test with pure dense (alpha=1.0)
        dense_searcher = HybridSearcher(alpha=1.0)
        self.assertEqual(dense_searcher.alpha, 1.0)
        
        # Test with pure sparse (alpha=0.0)
        sparse_searcher = HybridSearcher(alpha=0.0)
        self.assertEqual(sparse_searcher.alpha, 0.0)
        
        # Test with hybrid (alpha=0.5)
        hybrid_searcher = HybridSearcher(alpha=0.5)
        self.assertEqual(hybrid_searcher.alpha, 0.5)
    
    def test_dense_embedding_generation(self):
        """Test dense embedding generation."""
        query = "test query"
        mock_embedding = np.array([0.1, 0.2, 0.3])
        
        self.mock_dense_encoder.encode_queries.return_value = np.array([mock_embedding])
        
        result = self.searcher._get_dense_embedding(query)
        
        self.mock_dense_encoder.encode_queries.assert_called_once_with([query])
        np.testing.assert_array_equal(result, mock_embedding)
    
    def test_sparse_embedding_generation(self):
        """Test sparse embedding generation."""
        query = "test query"
        mock_sparse_vector = {"indices": [1, 5, 10], "values": [0.8, 0.6, 0.4]}
        
        self.mock_sparse_encoder.is_fitted = True
        self.mock_sparse_encoder.encode_queries.return_value = [mock_sparse_vector]
        
        result = self.searcher._get_sparse_embedding(query)
        
        self.mock_sparse_encoder.encode_queries.assert_called_once_with([query])
        self.assertEqual(result, mock_sparse_vector)
    
    def test_sparse_encoder_not_fitted(self):
        """Test handling when sparse encoder is not fitted."""
        query = "test query"
        
        self.mock_sparse_encoder.is_fitted = False
        
        result = self.searcher._get_sparse_embedding(query)
        
        expected = {'indices': [], 'values': []}
        self.assertEqual(result, expected)
    
    def test_search_with_results(self):
        """Test search with mock results."""
        query = "test query"
        
        # Mock embeddings
        mock_dense = np.array([0.1, 0.2, 0.3])
        mock_sparse = {"indices": [1, 5], "values": [0.8, 0.6]}
        
        self.mock_dense_encoder.encode_queries.return_value = np.array([mock_dense])
        self.mock_sparse_encoder.is_fitted = True
        self.mock_sparse_encoder.encode_queries.return_value = [mock_sparse]
        
        # Mock Pinecone response
        mock_match = Mock()
        mock_match.id = "chunk_1"
        mock_match.score = 0.85
        mock_match.metadata = {
            "video_id": "test_video",
            "text": "This is a test chunk",
            "start_sec": 10.0,
            "end_sec": 20.0,
            "source_url": "https://youtu.be/test_video?t=10"
        }
        
        mock_response = Mock()
        mock_response.matches = [mock_match]
        
        self.mock_upserter.index.query.return_value = mock_response
        
        # Perform search
        results = self.searcher.search(query, top_k=5, alpha=0.7)
        
        # Verify results
        self.assertEqual(len(results), 1)
        self.assertIsInstance(results[0], SearchResult)
        self.assertEqual(results[0].id, "chunk_1")
        self.assertEqual(results[0].score, 0.85)
        self.assertEqual(results[0].video_id, "test_video")
        
        # Verify Pinecone was called correctly
        self.mock_upserter.index.query.assert_called_once_with(
            vector=mock_dense.tolist(),
            sparse_vector=mock_sparse,
            top_k=5,
            filter=None,
            include_metadata=True,
            alpha=0.7
        )
    
    def test_search_with_filters(self):
        """Test search with metadata filters."""
        query = "test query"
        filters = {"video_id": "specific_video"}
        
        # Mock embeddings
        self.mock_dense_encoder.encode_queries.return_value = np.array([[0.1, 0.2, 0.3]])
        self.mock_sparse_encoder.is_fitted = True
        self.mock_sparse_encoder.encode_queries.return_value = [{"indices": [], "values": []}]
        
        # Mock empty response
        mock_response = Mock()
        mock_response.matches = []
        self.mock_upserter.index.query.return_value = mock_response
        
        # Perform search
        results = self.searcher.search(query, filters=filters)
        
        # Verify filters were passed
        call_args = self.mock_upserter.index.query.call_args
        self.assertEqual(call_args[1]['filter'], filters)
    
    def test_search_by_video(self):
        """Test searching within a specific video."""
        query = "test query"
        video_id = "specific_video"
        
        with patch.object(self.searcher, 'search') as mock_search:
            mock_search.return_value = []
            
            self.searcher.search_by_video(query, video_id, top_k=3, alpha=0.8)
            
            # Verify search was called with correct filters
            mock_search.assert_called_once_with(
                query, 3, 0.8, {'video_id': video_id}
            )
    
    def test_search_by_channel(self):
        """Test searching within a specific channel."""
        query = "test query"
        channel_title = "Test Channel"
        
        with patch.object(self.searcher, 'search') as mock_search:
            mock_search.return_value = []
            
            self.searcher.search_by_channel(query, channel_title, top_k=5)
            
            # Verify search was called with correct filters
            mock_search.assert_called_once_with(
                query, 5, None, {'channel_title': channel_title}
            )
    
    def test_search_by_date_range(self):
        """Test searching within a date range."""
        query = "test query"
        start_date = "2023-01-01"
        end_date = "2023-12-31"
        
        with patch.object(self.searcher, 'search') as mock_search:
            mock_search.return_value = []
            
            self.searcher.search_by_date_range(query, start_date, end_date)
            
            # Verify search was called with correct filters
            expected_filters = {
                'published_at': {
                    '$gte': start_date,
                    '$lte': end_date
                }
            }
            mock_search.assert_called_once_with(
                query, None, None, expected_filters
            )
    
    def test_search_by_duration(self):
        """Test searching by duration range."""
        query = "test query"
        min_duration = 30.0
        max_duration = 120.0
        
        with patch.object(self.searcher, 'search') as mock_search:
            mock_search.return_value = []
            
            self.searcher.search_by_duration(query, min_duration, max_duration)
            
            # Verify search was called with correct filters
            expected_filters = {
                'duration': {
                    '$gte': min_duration,
                    '$lte': max_duration
                }
            }
            mock_search.assert_called_once_with(
                query, None, None, expected_filters
            )
    
    def test_get_similar_chunks(self):
        """Test finding similar chunks."""
        chunk_id = "test_chunk_1"
        
        # Mock fetch response
        mock_vector = Mock()
        mock_vector.values = [0.1, 0.2, 0.3]
        mock_vector.metadata = {"video_id": "test_video"}
        
        mock_fetch_response = Mock()
        mock_fetch_response.vectors = {chunk_id: mock_vector}
        self.mock_upserter.index.fetch.return_value = mock_fetch_response
        
        # Mock query response
        mock_match = Mock()
        mock_match.id = "similar_chunk_1"
        mock_match.score = 0.9
        mock_match.metadata = {"video_id": "other_video", "text": "Similar content"}
        
        mock_query_response = Mock()
        mock_query_response.matches = [mock_match]
        self.mock_upserter.index.query.return_value = mock_query_response
        
        # Find similar chunks
        results = self.searcher.get_similar_chunks(chunk_id, top_k=5)
        
        # Verify fetch was called
        self.mock_upserter.index.fetch.assert_called_once_with([chunk_id])
        
        # Verify query was called with the vector
        self.mock_upserter.index.query.assert_called_once_with(
            vector=mock_vector.values,
            top_k=6,  # +1 to account for reference chunk
            filter=None,
            include_metadata=True
        )
        
        # Verify results
        self.assertEqual(len(results), 1)
        self.assertEqual(results[0].id, "similar_chunk_1")
    
    def test_get_similar_chunks_exclude_same_video(self):
        """Test finding similar chunks excluding same video."""
        chunk_id = "test_chunk_1"
        
        # Mock fetch response
        mock_vector = Mock()
        mock_vector.values = [0.1, 0.2, 0.3]
        mock_vector.metadata = {"video_id": "test_video"}
        
        mock_fetch_response = Mock()
        mock_fetch_response.vectors = {chunk_id: mock_vector}
        self.mock_upserter.index.fetch.return_value = mock_fetch_response
        
        # Mock empty query response
        mock_query_response = Mock()
        mock_query_response.matches = []
        self.mock_upserter.index.query.return_value = mock_query_response
        
        # Find similar chunks excluding same video
        results = self.searcher.get_similar_chunks(
            chunk_id, exclude_same_video=True
        )
        
        # Verify query was called with video exclusion filter
        call_args = self.mock_upserter.index.query.call_args
        expected_filter = {'video_id': {'$ne': 'test_video'}}
        self.assertEqual(call_args[1]['filter'], expected_filter)
    
    def test_search_error_handling(self):
        """Test error handling in search."""
        query = "test query"
        
        # Mock embeddings
        self.mock_dense_encoder.encode_queries.return_value = np.array([[0.1, 0.2, 0.3]])
        self.mock_sparse_encoder.is_fitted = True
        self.mock_sparse_encoder.encode_queries.return_value = [{"indices": [], "values": []}]
        
        # Mock Pinecone error
        self.mock_upserter.index.query.side_effect = Exception("Pinecone error")
        
        # Search should return empty results on error
        results = self.searcher.search(query)
        self.assertEqual(len(results), 0)


class TestSearchResult(unittest.TestCase):
    """Test cases for SearchResult."""
    
    def test_search_result_properties(self):
        """Test SearchResult property access."""
        metadata = {
            "video_id": "test_video",
            "text": "Test content",
            "start_sec": 10.5,
            "end_sec": 25.3,
            "source_url": "https://youtu.be/test_video?t=10"
        }
        
        result = SearchResult(
            id="test_chunk",
            score=0.85,
            metadata=metadata
        )
        
        self.assertEqual(result.video_id, "test_video")
        self.assertEqual(result.text, "Test content")
        self.assertEqual(result.start_sec, 10.5)
        self.assertEqual(result.end_sec, 25.3)
        self.assertEqual(result.source_url, "https://youtu.be/test_video?t=10")
    
    def test_search_result_missing_metadata(self):
        """Test SearchResult with missing metadata."""
        result = SearchResult(
            id="test_chunk",
            score=0.85,
            metadata={}
        )
        
        self.assertEqual(result.video_id, "")
        self.assertEqual(result.text, "")
        self.assertEqual(result.start_sec, 0.0)
        self.assertEqual(result.end_sec, 0.0)
        self.assertEqual(result.source_url, "")


if __name__ == '__main__':
    unittest.main()
