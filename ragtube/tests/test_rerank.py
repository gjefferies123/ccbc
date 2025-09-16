"""Tests for reranking functionality."""

import unittest
from unittest.mock import Mock, patch, MagicMock
import torch
import numpy as np

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from search.rerank import HybridReranker, BGEReranker, CohereReranker, RerankResult
from search.hybrid import SearchResult


class TestHybridReranker(unittest.TestCase):
    """Test cases for HybridReranker."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.search_results = [
            SearchResult(
                id="chunk_1",
                score=0.8,
                metadata={"text": "First result about testing", "video_title": "Test Video 1"}
            ),
            SearchResult(
                id="chunk_2", 
                score=0.7,
                metadata={"text": "Second result about development", "video_title": "Test Video 2"}
            ),
            SearchResult(
                id="chunk_3",
                score=0.6,
                metadata={"text": "Third result about software", "video_title": "Test Video 3"}
            )
        ]
    
    @patch('search.rerank.CohereReranker')
    @patch('search.rerank.BGEReranker')
    def test_initialization_with_cohere(self, mock_bge, mock_cohere):
        """Test initialization when Cohere is available."""
        # Mock Cohere as available
        mock_cohere_instance = Mock()
        mock_cohere_instance.is_available.return_value = True
        mock_cohere.return_value = mock_cohere_instance
        
        # Mock BGE as available
        mock_bge_instance = Mock()
        mock_bge_instance.is_available.return_value = True
        mock_bge.return_value = mock_bge_instance
        
        reranker = HybridReranker(cohere_api_key="test_key", prefer_cohere=True)
        
        # Should use Cohere when available and preferred
        self.assertEqual(reranker.active_reranker, mock_cohere_instance)
    
    @patch('search.rerank.CohereReranker')
    @patch('search.rerank.BGEReranker')
    def test_initialization_fallback_to_bge(self, mock_bge, mock_cohere):
        """Test fallback to BGE when Cohere is not available."""
        # Mock Cohere as not available
        mock_cohere.side_effect = Exception("Cohere not available")
        
        # Mock BGE as available
        mock_bge_instance = Mock()
        mock_bge_instance.is_available.return_value = True
        mock_bge.return_value = mock_bge_instance
        
        reranker = HybridReranker(cohere_api_key="test_key", prefer_cohere=True)
        
        # Should fallback to BGE
        self.assertEqual(reranker.active_reranker, mock_bge_instance)
    
    @patch('search.rerank.CohereReranker')
    @patch('search.rerank.BGEReranker')
    def test_no_reranker_available(self, mock_bge, mock_cohere):
        """Test when no reranker is available."""
        # Mock both as failing
        mock_cohere.side_effect = Exception("Cohere not available")
        mock_bge.side_effect = Exception("BGE not available")
        
        reranker = HybridReranker(cohere_api_key="test_key")
        
        # Should have no active reranker
        self.assertIsNone(reranker.active_reranker)
        self.assertFalse(reranker.is_available())
    
    def test_rerank_with_no_reranker(self):
        """Test reranking when no reranker is available."""
        reranker = HybridReranker()
        reranker.active_reranker = None
        
        query = "test query"
        results = reranker.rerank_search_results(query, self.search_results, top_k=2)
        
        # Should return original results wrapped in RerankResult
        self.assertEqual(len(results), 2)
        self.assertIsInstance(results[0], RerankResult)
        self.assertEqual(results[0].search_result.id, "chunk_1")
        self.assertEqual(results[0].rerank_score, 0.8)  # Original score
        self.assertEqual(results[0].rank_change, 0)
    
    def test_rerank_with_mock_reranker(self):
        """Test reranking with a mock reranker."""
        # Create mock reranker
        mock_reranker = Mock()
        mock_reranker.rerank.return_value = [
            (2, 0.95),  # Third result gets highest score
            (0, 0.85),  # First result gets second score
            (1, 0.75)   # Second result gets lowest score
        ]
        
        reranker = HybridReranker()
        reranker.active_reranker = mock_reranker
        
        query = "test query"
        results = reranker.rerank_search_results(query, self.search_results, top_k=3)
        
        # Verify reranker was called correctly
        mock_reranker.rerank.assert_called_once()
        call_args = mock_reranker.rerank.call_args
        self.assertEqual(call_args[1]['query'], query)
        self.assertEqual(len(call_args[1]['documents']), 3)
        self.assertEqual(call_args[1]['top_k'], 3)
        
        # Verify results are reordered
        self.assertEqual(len(results), 3)
        self.assertEqual(results[0].search_result.id, "chunk_3")  # Third result now first
        self.assertEqual(results[0].rerank_score, 0.95)
        self.assertEqual(results[0].rank_change, 2)  # Moved up 2 positions
        
        self.assertEqual(results[1].search_result.id, "chunk_1")  # First result now second
        self.assertEqual(results[1].rerank_score, 0.85)
        self.assertEqual(results[1].rank_change, -1)  # Moved down 1 position
    
    def test_rerank_error_handling(self):
        """Test error handling during reranking."""
        # Create mock reranker that fails
        mock_reranker = Mock()
        mock_reranker.rerank.side_effect = Exception("Reranking failed")
        
        reranker = HybridReranker()
        reranker.active_reranker = mock_reranker
        
        query = "test query"
        results = reranker.rerank_search_results(query, self.search_results)
        
        # Should fallback to original results
        self.assertEqual(len(results), len(self.search_results))
        for i, result in enumerate(results):
            self.assertEqual(result.search_result.id, self.search_results[i].id)
            self.assertEqual(result.rank_change, 0)
    
    def test_get_reranker_info(self):
        """Test getting reranker information."""
        # Test with no reranker
        reranker = HybridReranker()
        reranker.active_reranker = None
        
        info = reranker.get_reranker_info()
        self.assertEqual(info['type'], 'none')
        self.assertFalse(info['available'])
        
        # Test with mock Cohere reranker
        mock_cohere = Mock(spec=CohereReranker)
        mock_cohere.model = "rerank-english-v3.0"
        reranker.active_reranker = mock_cohere
        
        info = reranker.get_reranker_info()
        self.assertEqual(info['type'], 'cohere')
        self.assertEqual(info['model'], "rerank-english-v3.0")
        self.assertTrue(info['available'])


class TestBGEReranker(unittest.TestCase):
    """Test cases for BGEReranker."""
    
    @patch('search.rerank.AutoTokenizer')
    @patch('search.rerank.AutoModelForSequenceClassification')
    def test_initialization(self, mock_model_class, mock_tokenizer_class):
        """Test BGE reranker initialization."""
        # Mock tokenizer and model
        mock_tokenizer = Mock()
        mock_model = Mock()
        mock_model.to.return_value = mock_model
        
        mock_tokenizer_class.from_pretrained.return_value = mock_tokenizer
        mock_model_class.from_pretrained.return_value = mock_model
        
        reranker = BGEReranker(model_name="test-model", device="cpu")
        
        # Verify initialization
        mock_tokenizer_class.from_pretrained.assert_called_once_with("test-model")
        mock_model_class.from_pretrained.assert_called_once_with("test-model")
        mock_model.to.assert_called_once_with("cpu")
        mock_model.eval.assert_called_once()
        
        self.assertTrue(reranker.is_available())
    
    @patch('search.rerank.AutoTokenizer')
    @patch('search.rerank.AutoModelForSequenceClassification')
    @patch('torch.cuda.is_available')
    def test_device_selection(self, mock_cuda, mock_model_class, mock_tokenizer_class):
        """Test automatic device selection."""
        # Mock CUDA as available
        mock_cuda.return_value = True
        
        mock_tokenizer_class.from_pretrained.return_value = Mock()
        mock_model = Mock()
        mock_model.to.return_value = mock_model
        mock_model_class.from_pretrained.return_value = mock_model
        
        reranker = BGEReranker()
        
        # Should use CUDA when available
        self.assertEqual(reranker.device, "cuda")
    
    @patch('search.rerank.AutoTokenizer')
    @patch('search.rerank.AutoModelForSequenceClassification')
    def test_rerank_functionality(self, mock_model_class, mock_tokenizer_class):
        """Test BGE reranking functionality."""
        # Mock tokenizer
        mock_tokenizer = Mock()
        mock_inputs = {
            'input_ids': torch.tensor([[1, 2, 3]]),
            'attention_mask': torch.tensor([[1, 1, 1]])
        }
        mock_inputs_to_device = Mock()
        mock_inputs_to_device.to.return_value = mock_inputs
        mock_tokenizer.return_value = mock_inputs_to_device
        
        # Mock model
        mock_model = Mock()
        mock_model.to.return_value = mock_model
        
        # Mock model output
        mock_logits = torch.tensor([[2.0], [1.5], [0.5]])
        mock_output = Mock()
        mock_output.logits = mock_logits
        mock_model.return_value = mock_output
        
        mock_tokenizer_class.from_pretrained.return_value = mock_tokenizer
        mock_model_class.from_pretrained.return_value = mock_model
        
        reranker = BGEReranker(device="cpu")
        
        query = "test query"
        documents = ["doc 1", "doc 2", "doc 3"]
        
        results = reranker.rerank(query, documents, top_k=2)
        
        # Verify tokenizer was called correctly
        mock_tokenizer.assert_called_once()
        call_args = mock_tokenizer.call_args[0][0]
        expected_pairs = [(query, doc) for doc in documents]
        self.assertEqual(call_args, expected_pairs)
        
        # Verify results are sorted by score
        self.assertEqual(len(results), 2)
        self.assertEqual(results[0][0], 0)  # First document (highest score)
        self.assertEqual(results[1][0], 1)  # Second document
        
        # Scores should be in descending order
        self.assertGreater(results[0][1], results[1][1])
    
    def test_rerank_empty_documents(self):
        """Test reranking with empty document list."""
        reranker = BGEReranker()
        reranker.model = Mock()  # Set to indicate it's loaded
        reranker.tokenizer = Mock()
        
        results = reranker.rerank("test query", [])
        self.assertEqual(len(results), 0)
    
    def test_rerank_not_available(self):
        """Test reranking when model is not loaded."""
        reranker = BGEReranker()
        reranker.model = None
        reranker.tokenizer = None
        
        with self.assertRaises(ValueError):
            reranker.rerank("test query", ["doc 1"])


class TestCohereReranker(unittest.TestCase):
    """Test cases for CohereReranker."""
    
    @patch('search.rerank.cohere.Client')
    def test_initialization(self, mock_cohere_client):
        """Test Cohere reranker initialization."""
        mock_client_instance = Mock()
        mock_cohere_client.return_value = mock_client_instance
        
        reranker = CohereReranker(api_key="test_key", model="test-model")
        
        # Verify initialization
        mock_cohere_client.assert_called_once_with("test_key")
        self.assertEqual(reranker.model, "test-model")
        self.assertTrue(reranker.is_available())
    
    @patch('search.rerank.cohere.Client')
    def test_initialization_failure(self, mock_cohere_client):
        """Test Cohere reranker initialization failure."""
        mock_cohere_client.side_effect = Exception("API key invalid")
        
        reranker = CohereReranker(api_key="invalid_key")
        
        self.assertFalse(reranker.is_available())
    
    @patch('search.rerank.cohere.Client')
    def test_rerank_functionality(self, mock_cohere_client):
        """Test Cohere reranking functionality."""
        # Mock client and response
        mock_client = Mock()
        mock_result_1 = Mock()
        mock_result_1.index = 2
        mock_result_1.relevance_score = 0.95
        
        mock_result_2 = Mock()
        mock_result_2.index = 0
        mock_result_2.relevance_score = 0.85
        
        mock_response = Mock()
        mock_response.results = [mock_result_1, mock_result_2]
        
        mock_client.rerank.return_value = mock_response
        mock_cohere_client.return_value = mock_client
        
        reranker = CohereReranker(api_key="test_key")
        
        query = "test query"
        documents = ["doc 1", "doc 2", "doc 3"]
        
        results = reranker.rerank(query, documents, top_k=2)
        
        # Verify client was called correctly
        mock_client.rerank.assert_called_once_with(
            model=reranker.model,
            query=query,
            documents=documents,
            top_k=2,
            return_documents=False
        )
        
        # Verify results
        self.assertEqual(len(results), 2)
        self.assertEqual(results[0], (2, 0.95))
        self.assertEqual(results[1], (0, 0.85))
    
    def test_rerank_not_available(self):
        """Test reranking when client is not available."""
        reranker = CohereReranker()
        reranker.client = None
        
        with self.assertRaises(ValueError):
            reranker.rerank("test query", ["doc 1"])
    
    @patch('search.rerank.cohere.Client')
    def test_rerank_api_error(self, mock_cohere_client):
        """Test handling of Cohere API errors."""
        mock_client = Mock()
        mock_client.rerank.side_effect = Exception("API error")
        mock_cohere_client.return_value = mock_client
        
        reranker = CohereReranker(api_key="test_key")
        
        with self.assertRaises(Exception):
            reranker.rerank("test query", ["doc 1"])


if __name__ == '__main__':
    unittest.main()
