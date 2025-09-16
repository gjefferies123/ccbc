"""Text processing utilities for transcript segmentation and token counting."""

import re
import logging
from typing import List, Tuple, Dict, Any
import tiktoken
import nltk
from nltk.tokenize import sent_tokenize

logger = logging.getLogger(__name__)

# Download required NLTK data
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt', quiet=True)


def count_tokens(text: str, model: str = "gpt-3.5-turbo") -> int:
    """Count tokens in text using tiktoken.
    
    Args:
        text: Input text
        model: Model name for tokenizer
        
    Returns:
        Number of tokens
    """
    try:
        encoding = tiktoken.encoding_for_model(model)
        return len(encoding.encode(text))
    except Exception:
        # Fallback to simple word count approximation
        return len(text.split()) * 1.3  # Rough approximation


def merge_transcript_items(transcript_items: List[Dict[str, Any]], 
                         max_gap_seconds: float = 2.0) -> List[Dict[str, Any]]:
    """Merge transcript items into sentences based on punctuation and time gaps.
    
    Args:
        transcript_items: List of transcript items with 'text', 'start', 'duration'
        max_gap_seconds: Maximum gap between items to merge
        
    Returns:
        List of merged sentence items
    """
    if not transcript_items:
        return []
    
    sentences = []
    current_sentence = {
        'text': '',
        'start': transcript_items[0]['start'],
        'end': transcript_items[0]['start'] + transcript_items[0]['duration'],
        'items': []
    }
    
    for i, item in enumerate(transcript_items):
        item_start = item['start']
        item_end = item['start'] + item['duration']
        item_text = item['text'].strip()
        
        if not item_text:
            continue
        
        # Check if we should start a new sentence
        should_split = False
        
        if current_sentence['text']:
            # Check time gap
            time_gap = item_start - current_sentence['end']
            if time_gap > max_gap_seconds:
                should_split = True
            
            # Check punctuation at end of current sentence
            if current_sentence['text'].rstrip().endswith(('.', '!', '?')):
                should_split = True
        
        if should_split and current_sentence['text'].strip():
            # Finalize current sentence
            current_sentence['text'] = current_sentence['text'].strip()
            sentences.append(current_sentence)
            
            # Start new sentence
            current_sentence = {
                'text': '',
                'start': item_start,
                'end': item_end,
                'items': []
            }
        
        # Add item to current sentence
        if current_sentence['text']:
            current_sentence['text'] += ' '
        current_sentence['text'] += item_text
        current_sentence['end'] = item_end
        current_sentence['items'].append(item)
    
    # Add final sentence
    if current_sentence['text'].strip():
        current_sentence['text'] = current_sentence['text'].strip()
        sentences.append(current_sentence)
    
    return sentences


def split_into_sentences(text: str) -> List[str]:
    """Split text into sentences using NLTK.
    
    Args:
        text: Input text
        
    Returns:
        List of sentences
    """
    try:
        return sent_tokenize(text)
    except Exception as e:
        logger.warning(f"NLTK sentence tokenization failed: {e}, using fallback")
        # Fallback to simple regex-based splitting
        sentences = re.split(r'[.!?]+', text)
        return [s.strip() for s in sentences if s.strip()]


def extract_relevant_sentences(text: str, query: str, 
                             max_sentences: int = 10,
                             similarity_threshold: float = 0.3) -> str:
    """Extract sentences from text that are relevant to the query.
    
    Args:
        text: Input text
        query: Search query
        max_sentences: Maximum number of sentences to return
        similarity_threshold: Minimum similarity threshold
        
    Returns:
        Filtered text with relevant sentences
    """
    sentences = split_into_sentences(text)
    
    if len(sentences) <= max_sentences:
        return text
    
    # Simple keyword-based relevance scoring
    query_words = set(query.lower().split())
    scored_sentences = []
    
    for i, sentence in enumerate(sentences):
        sentence_words = set(sentence.lower().split())
        
        # Calculate Jaccard similarity
        intersection = len(query_words & sentence_words)
        union = len(query_words | sentence_words)
        similarity = intersection / union if union > 0 else 0
        
        scored_sentences.append((similarity, i, sentence))
    
    # Sort by similarity and keep top sentences in original order
    scored_sentences.sort(key=lambda x: x[0], reverse=True)
    relevant_sentences = [
        (idx, sent) for score, idx, sent in scored_sentences[:max_sentences]
        if score >= similarity_threshold
    ]
    
    # Sort by original order
    relevant_sentences.sort(key=lambda x: x[0])
    
    if not relevant_sentences:
        # If no sentences meet threshold, return first few sentences
        return ' '.join(sentences[:max_sentences])
    
    return ' '.join([sent for _, sent in relevant_sentences])


def clean_transcript_text(text: str) -> str:
    """Clean transcript text by removing artifacts and normalizing.
    
    Args:
        text: Raw transcript text
        
    Returns:
        Cleaned text
    """
    # Remove common transcript artifacts
    text = re.sub(r'\[.*?\]', '', text)  # Remove bracketed content
    text = re.sub(r'\(.*?\)', '', text)  # Remove parenthetical content
    text = re.sub(r'<.*?>', '', text)    # Remove HTML-like tags
    
    # Normalize whitespace
    text = re.sub(r'\s+', ' ', text)
    text = text.strip()
    
    # Remove repeated punctuation
    text = re.sub(r'[.]{2,}', '.', text)
    text = re.sub(r'[!]{2,}', '!', text)
    text = re.sub(r'[?]{2,}', '?', text)
    
    return text


def truncate_to_token_limit(text: str, max_tokens: int, model: str = "gpt-3.5-turbo") -> str:
    """Truncate text to fit within token limit.
    
    Args:
        text: Input text
        max_tokens: Maximum number of tokens
        model: Model name for tokenizer
        
    Returns:
        Truncated text
    """
    current_tokens = count_tokens(text, model)
    
    if current_tokens <= max_tokens:
        return text
    
    # Binary search for the right length
    sentences = split_into_sentences(text)
    left, right = 0, len(sentences)
    best_text = ""
    
    while left <= right:
        mid = (left + right) // 2
        candidate_text = ' '.join(sentences[:mid])
        candidate_tokens = count_tokens(candidate_text, model)
        
        if candidate_tokens <= max_tokens:
            best_text = candidate_text
            left = mid + 1
        else:
            right = mid - 1
    
    return best_text


def format_duration(seconds: float) -> str:
    """Format duration in seconds to human-readable format.
    
    Args:
        seconds: Duration in seconds
        
    Returns:
        Formatted duration string (e.g., "1:23:45")
    """
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    
    if hours > 0:
        return f"{hours}:{minutes:02d}:{secs:02d}"
    else:
        return f"{minutes}:{secs:02d}"


def extract_keywords(text: str, max_keywords: int = 10) -> List[str]:
    """Extract keywords from text using simple frequency analysis.
    
    Args:
        text: Input text
        max_keywords: Maximum number of keywords to return
        
    Returns:
        List of keywords
    """
    # Simple keyword extraction based on word frequency
    words = re.findall(r'\b\w{3,}\b', text.lower())
    
    # Remove common stop words
    stop_words = {
        'the', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with',
        'by', 'from', 'up', 'about', 'into', 'through', 'during', 'before',
        'after', 'above', 'below', 'between', 'among', 'this', 'that', 'these',
        'those', 'his', 'her', 'its', 'their', 'what', 'which', 'who', 'when',
        'where', 'why', 'how', 'all', 'any', 'both', 'each', 'few', 'more',
        'most', 'other', 'some', 'such', 'only', 'own', 'same', 'so', 'than',
        'too', 'very', 'can', 'will', 'just', 'should', 'now', 'said', 'like',
        'time', 'way', 'many', 'may', 'use', 'make', 'get', 'go', 'come', 'know'
    }
    
    filtered_words = [word for word in words if word not in stop_words]
    
    # Count frequencies
    word_freq = {}
    for word in filtered_words:
        word_freq[word] = word_freq.get(word, 0) + 1
    
    # Sort by frequency and return top keywords
    sorted_words = sorted(word_freq.items(), key=lambda x: x[1], reverse=True)
    return [word for word, freq in sorted_words[:max_keywords]]
