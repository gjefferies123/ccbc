"""YouTube chapter parsing utilities."""

import re
import logging
from typing import List, Dict, Any, Optional

logger = logging.getLogger(__name__)


def parse_chapters_from_description(description: str) -> List[Dict[str, Any]]:
    """Parse chapter timestamps and titles from video description.
    
    Args:
        description: Video description text
        
    Returns:
        List of chapters with 'title' and 'start_time' (seconds)
    """
    if not description:
        return []
    
    chapters = []
    lines = description.split('\n')
    
    # Common timestamp patterns for chapters
    timestamp_patterns = [
        # HH:MM:SS format
        r'^(\d{1,2}):(\d{2}):(\d{2})\s*[-–—]?\s*(.+)$',
        # MM:SS format
        r'^(\d{1,2}):(\d{2})\s*[-–—]?\s*(.+)$',
        # With leading timestamp indicator
        r'^\d+[.\)]\s*(\d{1,2}):(\d{2}):(\d{2})\s*[-–—]?\s*(.+)$',
        r'^\d+[.\)]\s*(\d{1,2}):(\d{2})\s*[-–—]?\s*(.+)$',
        # With chapter word
        r'^(?:chapter\s*\d+\s*[:\-]?\s*)?(\d{1,2}):(\d{2}):(\d{2})\s*[-–—]?\s*(.+)$',
        r'^(?:chapter\s*\d+\s*[:\-]?\s*)?(\d{1,2}):(\d{2})\s*[-–—]?\s*(.+)$',
    ]
    
    for line in lines:
        line = line.strip()
        if not line:
            continue
        
        chapter = _parse_timestamp_line(line, timestamp_patterns)
        if chapter:
            chapters.append(chapter)
    
    # Sort chapters by start time
    chapters.sort(key=lambda x: x['start_time'])
    
    # Validate and clean chapters
    chapters = _validate_chapters(chapters)
    
    if chapters:
        logger.info(f"Parsed {len(chapters)} chapters from description")
        for i, chapter in enumerate(chapters):
            logger.debug(f"Chapter {i+1}: {chapter['start_time']}s - {chapter['title']}")
    
    return chapters


def _parse_timestamp_line(line: str, patterns: List[str]) -> Optional[Dict[str, Any]]:
    """Parse a single line for timestamp and chapter title.
    
    Args:
        line: Line of text to parse
        patterns: List of regex patterns to try
        
    Returns:
        Chapter dict or None if no match
    """
    line_lower = line.lower()
    
    for pattern in patterns:
        match = re.match(pattern, line, re.IGNORECASE)
        if match:
            groups = match.groups()
            
            if len(groups) == 3:  # MM:SS format
                minutes, seconds, title = groups
                start_time = int(minutes) * 60 + int(seconds)
            elif len(groups) == 4:  # HH:MM:SS format
                hours, minutes, seconds, title = groups
                start_time = int(hours) * 3600 + int(minutes) * 60 + int(seconds)
            else:
                continue
            
            # Clean up the title
            title = title.strip()
            title = re.sub(r'^[-–—]\s*', '', title)  # Remove leading dashes
            title = re.sub(r'\s*[-–—]\s*$', '', title)  # Remove trailing dashes
            
            # Skip if title is too short or just numbers/symbols
            if len(title) < 3 or not re.search(r'[a-zA-Z]', title):
                continue
            
            return {
                'title': title,
                'start_time': start_time
            }
    
    return None


def _validate_chapters(chapters: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Validate and clean the parsed chapters.
    
    Args:
        chapters: Raw parsed chapters
        
    Returns:
        Validated and cleaned chapters
    """
    if not chapters:
        return []
    
    validated = []
    prev_time = -1
    
    for chapter in chapters:
        start_time = chapter['start_time']
        title = chapter['title']
        
        # Skip if timestamp is not increasing
        if start_time <= prev_time:
            logger.debug(f"Skipping chapter with non-increasing timestamp: {start_time}s")
            continue
        
        # Skip if title is too generic or empty
        if _is_generic_title(title):
            logger.debug(f"Skipping chapter with generic title: '{title}'")
            continue
        
        validated.append(chapter)
        prev_time = start_time
    
    return validated


def _is_generic_title(title: str) -> bool:
    """Check if a chapter title is too generic.
    
    Args:
        title: Chapter title
        
    Returns:
        True if title is too generic
    """
    title_lower = title.lower().strip()
    
    # Too short
    if len(title_lower) < 3:
        return True
    
    # Just numbers or basic punctuation
    if re.match(r'^[0-9\s\-_.()]+$', title_lower):
        return True
    
    # Common generic titles
    generic_titles = {
        'intro', 'introduction', 'outro', 'conclusion', 'end', 'start',
        'beginning', 'chapter', 'section', 'part', 'segment', 'clip',
        'music', 'song', 'ad', 'advertisement', 'sponsor', 'break'
    }
    
    return title_lower in generic_titles


def extract_chapters_from_transcript(transcript_items: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Extract potential chapter breaks from transcript content.
    
    This is a fallback method when no description chapters are available.
    
    Args:
        transcript_items: Transcript items with 'text', 'start', 'duration'
        
    Returns:
        List of inferred chapters
    """
    if not transcript_items:
        return []
    
    chapters = []
    
    # Look for natural topic transitions in the transcript
    chapter_indicators = [
        r'\b(?:now|next|moving on|let\'s talk about|today we\'re|in this section)\b',
        r'\b(?:chapter|part|section) \d+\b',
        r'\b(?:first|second|third|fourth|fifth|finally|lastly)\b',
        r'\b(?:outro|conclusion|in summary|to wrap up|that\'s all)\b'
    ]
    
    current_chapter_start = transcript_items[0]['start']
    current_chapter_text = []
    chapter_count = 1
    
    for i, item in enumerate(transcript_items):
        text = item['text'].lower()
        current_chapter_text.append(text)
        
        # Check for chapter indicators
        has_indicator = any(re.search(pattern, text, re.IGNORECASE) 
                          for pattern in chapter_indicators)
        
        # Check for long pause (potential chapter break)
        long_pause = False
        if i < len(transcript_items) - 1:
            next_item = transcript_items[i + 1]
            gap = next_item['start'] - (item['start'] + item['duration'])
            long_pause = gap > 5.0  # 5 second gap
        
        # Check if we should create a new chapter
        should_create_chapter = (
            has_indicator or
            long_pause or
            (len(current_chapter_text) > 100 and  # Accumulated enough content
             item['start'] - current_chapter_start > 300)  # At least 5 minutes
        )
        
        if should_create_chapter and len(current_chapter_text) > 20:
            # Try to extract a meaningful title
            chapter_title = _extract_chapter_title_from_text(' '.join(current_chapter_text))
            
            if not chapter_title:
                chapter_title = f"Section {chapter_count}"
            
            chapters.append({
                'title': chapter_title,
                'start_time': current_chapter_start
            })
            
            # Start new chapter
            current_chapter_start = item['start']
            current_chapter_text = []
            chapter_count += 1
    
    # Add final chapter if needed
    if len(current_chapter_text) > 20:
        chapter_title = _extract_chapter_title_from_text(' '.join(current_chapter_text))
        if not chapter_title:
            chapter_title = f"Section {chapter_count}"
        
        chapters.append({
            'title': chapter_title,
            'start_time': current_chapter_start
        })
    
    if chapters:
        logger.info(f"Inferred {len(chapters)} chapters from transcript content")
    
    return chapters


def _extract_chapter_title_from_text(text: str) -> Optional[str]:
    """Extract a potential chapter title from text content.
    
    Args:
        text: Text content to analyze
        
    Returns:
        Extracted title or None
    """
    # Look for common patterns that might indicate topics
    title_patterns = [
        r'(?:about|discuss|talk about|cover|explain)\s+([^.!?]{10,50})',
        r'(?:how to|ways to|steps to)\s+([^.!?]{10,50})',
        r'(?:the|understanding|basics of)\s+([^.!?]{10,50})',
    ]
    
    for pattern in title_patterns:
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            title = match.group(1).strip()
            # Clean up the title
            title = re.sub(r'\s+', ' ', title)
            if 10 <= len(title) <= 50 and not _is_generic_title(title):
                return title.title()
    
    return None


def merge_overlapping_chapters(chapters: List[Dict[str, Any]], 
                             min_duration: int = 60) -> List[Dict[str, Any]]:
    """Merge chapters that are too short or overlapping.
    
    Args:
        chapters: List of chapters
        min_duration: Minimum chapter duration in seconds
        
    Returns:
        Merged chapters
    """
    if len(chapters) <= 1:
        return chapters
    
    merged = []
    current_chapter = chapters[0].copy()
    
    for i in range(1, len(chapters)):
        next_chapter = chapters[i]
        
        # Calculate duration of current chapter
        duration = next_chapter['start_time'] - current_chapter['start_time']
        
        if duration < min_duration:
            # Merge with next chapter
            current_chapter['title'] += f" / {next_chapter['title']}"
        else:
            # Keep current chapter and move to next
            merged.append(current_chapter)
            current_chapter = next_chapter.copy()
    
    # Add the last chapter
    merged.append(current_chapter)
    
    return merged
