"""YouTube data fetching using YouTube Data API and transcript API."""

import logging
import json
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass, asdict
from pathlib import Path
import re
from googleapiclient.discovery import build
from youtube_transcript_api import YouTubeTranscriptApi
from youtube_transcript_api.formatters import TextFormatter
from config import Config, MANIFEST_DIR

logger = logging.getLogger(__name__)


@dataclass
class VideoInfo:
    """Video information from YouTube API."""
    video_id: str
    title: str
    channel_title: str
    published_at: str
    duration: str
    description: str
    view_count: int
    like_count: int
    language: str = "en"


@dataclass
class TranscriptInfo:
    """Transcript information."""
    video_id: str
    language: str
    is_generated: bool
    transcript_items: List[Dict[str, Any]]


class YouTubeFetcher:
    """Fetches video metadata and transcripts from YouTube."""
    
    def __init__(self, api_key: Optional[str] = None):
        """Initialize the YouTube fetcher.
        
        Args:
            api_key: YouTube Data API key
        """
        self.api_key = api_key or Config.YOUTUBE_API_KEY
        self.youtube = None
        
        if self.api_key:
            try:
                self.youtube = build('youtube', 'v3', developerKey=self.api_key)
                logger.info("YouTube Data API client initialized")
            except Exception as e:
                logger.warning(f"Failed to initialize YouTube API client: {e}")
    
    def get_channel_videos(self, channel_id: str, max_results: int = 50) -> List[str]:
        """Get video IDs from a YouTube channel.
        
        Args:
            channel_id: YouTube channel ID
            max_results: Maximum number of videos to retrieve
            
        Returns:
            List of video IDs
        """
        if not self.youtube:
            raise ValueError("YouTube API client not available. Set YOUTUBE_API_KEY.")
        
        video_ids = []
        next_page_token = None
        
        logger.info(f"Fetching videos from channel {channel_id}")
        
        while len(video_ids) < max_results:
            try:
                # Get channel's uploads playlist
                channels_response = self.youtube.channels().list(
                    part='contentDetails',
                    id=channel_id
                ).execute()
                
                if not channels_response['items']:
                    logger.error(f"Channel {channel_id} not found")
                    break
                
                uploads_playlist_id = channels_response['items'][0]['contentDetails']['relatedPlaylists']['uploads']
                
                # Get videos from uploads playlist
                playlist_response = self.youtube.playlistItems().list(
                    part='snippet',
                    playlistId=uploads_playlist_id,
                    maxResults=min(50, max_results - len(video_ids)),
                    pageToken=next_page_token
                ).execute()
                
                for item in playlist_response['items']:
                    video_id = item['snippet']['resourceId']['videoId']
                    video_ids.append(video_id)
                
                next_page_token = playlist_response.get('nextPageToken')
                if not next_page_token:
                    break
                    
            except Exception as e:
                logger.error(f"Error fetching channel videos: {e}")
                break
        
        logger.info(f"Found {len(video_ids)} videos from channel {channel_id}")
        return video_ids[:max_results]
    
    def get_video_info(self, video_ids: List[str]) -> List[VideoInfo]:
        """Get detailed information for videos.
        
        Args:
            video_ids: List of YouTube video IDs
            
        Returns:
            List of VideoInfo objects
        """
        if not self.youtube:
            # Return minimal info without API
            return [
                VideoInfo(
                    video_id=vid,
                    title=f"Video {vid}",
                    channel_title="Unknown Channel",
                    published_at="1970-01-01T00:00:00Z",
                    duration="PT0S",
                    description="",
                    view_count=0,
                    like_count=0
                )
                for vid in video_ids
            ]
        
        video_infos = []
        
        # Process videos in batches of 50 (API limit)
        for i in range(0, len(video_ids), 50):
            batch_ids = video_ids[i:i + 50]
            
            try:
                response = self.youtube.videos().list(
                    part='snippet,statistics,contentDetails',
                    id=','.join(batch_ids)
                ).execute()
                
                for item in response['items']:
                    video_info = VideoInfo(
                        video_id=item['id'],
                        title=item['snippet']['title'],
                        channel_title=item['snippet']['channelTitle'],
                        published_at=item['snippet']['publishedAt'],
                        duration=item['contentDetails']['duration'],
                        description=item['snippet'].get('description', ''),
                        view_count=int(item['statistics'].get('viewCount', 0)),
                        like_count=int(item['statistics'].get('likeCount', 0)),
                        language=item['snippet'].get('defaultLanguage', 'en')
                    )
                    video_infos.append(video_info)
                    
            except Exception as e:
                logger.error(f"Error fetching video info for batch: {e}")
                continue
        
        logger.info(f"Retrieved info for {len(video_infos)} videos")
        return video_infos
    
    def get_transcript(self, video_id: str, 
                      languages: List[str] = None) -> Optional[TranscriptInfo]:
        """Get transcript for a video.
        
        Args:
            video_id: YouTube video ID
            languages: Preferred languages (defaults to ['en'])
            
        Returns:
            TranscriptInfo object or None if no transcript available
        """
        if languages is None:
            languages = ['en']
        
        try:
            # Get available transcripts
            transcript_list = YouTubeTranscriptApi.list_transcripts(video_id)
            
            # Try to get manually created transcript first
            transcript = None
            is_generated = True
            selected_language = 'en'
            
            for lang in languages:
                try:
                    # Try manually created first
                    transcript = transcript_list.find_manually_created_transcript([lang])
                    is_generated = False
                    selected_language = lang
                    break
                except:
                    try:
                        # Fall back to generated
                        transcript = transcript_list.find_generated_transcript([lang])
                        is_generated = True
                        selected_language = lang
                        break
                    except:
                        continue
            
            if transcript is None:
                # Try any available transcript
                try:
                    transcript = next(iter(transcript_list))
                    selected_language = transcript.language_code
                    is_generated = transcript.is_generated
                except:
                    logger.warning(f"No transcript available for video {video_id}")
                    return None
            
            # Fetch the transcript
            transcript_items = transcript.fetch()
            
            logger.debug(f"Retrieved {len(transcript_items)} transcript items for video {video_id}")
            
            return TranscriptInfo(
                video_id=video_id,
                language=selected_language,
                is_generated=is_generated,
                transcript_items=transcript_items
            )
            
        except Exception as e:
            logger.warning(f"Failed to get transcript for video {video_id}: {e}")
            return None
    
    def extract_chapters_from_description(self, description: str) -> List[Dict[str, Any]]:
        """Extract chapter information from video description.
        
        Args:
            description: Video description text
            
        Returns:
            List of chapters with 'title' and 'start_time'
        """
        chapters = []
        
        # Common chapter patterns
        patterns = [
            r'(\d{1,2}):(\d{2})\s+(.+)',  # MM:SS Title
            r'(\d{1,2}):(\d{2}):(\d{2})\s+(.+)',  # HH:MM:SS Title
            r'(\d{1,2})\.(\d{2})\s+(.+)',  # MM.SS Title
        ]
        
        lines = description.split('\n')
        
        for line in lines:
            line = line.strip()
            if not line:
                continue
            
            for pattern in patterns:
                match = re.match(pattern, line)
                if match:
                    if len(match.groups()) == 3:  # MM:SS format
                        minutes, seconds, title = match.groups()
                        start_time = int(minutes) * 60 + int(seconds)
                    else:  # HH:MM:SS format
                        hours, minutes, seconds, title = match.groups()
                        start_time = int(hours) * 3600 + int(minutes) * 60 + int(seconds)
                    
                    chapters.append({
                        'title': title.strip(),
                        'start_time': start_time
                    })
                    break
        
        # Sort chapters by start time
        chapters.sort(key=lambda x: x['start_time'])
        
        if chapters:
            logger.debug(f"Extracted {len(chapters)} chapters from description")
        
        return chapters
    
    def save_manifest(self, video_info: VideoInfo, 
                     transcript_info: Optional[TranscriptInfo],
                     chapters: List[Dict[str, Any]],
                     parent_segments: List[Any],
                     child_chunks: List[Any]) -> None:
        """Save video manifest to disk.
        
        Args:
            video_info: Video information
            transcript_info: Transcript information
            chapters: Chapter information
            parent_segments: Parent segments
            child_chunks: Child chunks
        """
        manifest_path = Path(MANIFEST_DIR)
        manifest_path.mkdir(parents=True, exist_ok=True)
        
        manifest = {
            'video_info': asdict(video_info),
            'transcript_info': asdict(transcript_info) if transcript_info else None,
            'chapters': chapters,
            'parent_segments': [asdict(p) for p in parent_segments],
            'child_chunks': [asdict(c) for c in child_chunks],
            'stats': {
                'parent_count': len(parent_segments),
                'child_count': len(child_chunks),
                'total_duration': max([c.end_sec for c in child_chunks]) if child_chunks else 0,
                'has_chapters': len(chapters) > 0,
                'has_transcript': transcript_info is not None
            }
        }
        
        manifest_file = manifest_path / f"{video_info.video_id}.json"
        
        with open(manifest_file, 'w', encoding='utf-8') as f:
            json.dump(manifest, f, indent=2, ensure_ascii=False)
        
        logger.info(f"Saved manifest for video {video_info.video_id} to {manifest_file}")
    
    def load_manifest(self, video_id: str) -> Optional[Dict[str, Any]]:
        """Load video manifest from disk.
        
        Args:
            video_id: YouTube video ID
            
        Returns:
            Manifest dictionary or None if not found
        """
        manifest_file = Path(MANIFEST_DIR) / f"{video_id}.json"
        
        if not manifest_file.exists():
            return None
        
        try:
            with open(manifest_file, 'r', encoding='utf-8') as f:
                return json.load(f)
        except Exception as e:
            logger.error(f"Failed to load manifest for video {video_id}: {e}")
            return None


def parse_duration(duration_str: str) -> int:
    """Parse ISO 8601 duration string to seconds.
    
    Args:
        duration_str: ISO 8601 duration (e.g., 'PT4M13S')
        
    Returns:
        Duration in seconds
    """
    # Simple parser for YouTube duration format
    match = re.match(r'PT(?:(\d+)H)?(?:(\d+)M)?(?:(\d+)S)?', duration_str)
    if not match:
        return 0
    
    hours = int(match.group(1) or 0)
    minutes = int(match.group(2) or 0)
    seconds = int(match.group(3) or 0)
    
    return hours * 3600 + minutes * 60 + seconds


def create_source_url(video_id: str, start_sec: float) -> str:
    """Create timestamped YouTube URL.
    
    Args:
        video_id: YouTube video ID
        start_sec: Start timestamp in seconds
        
    Returns:
        Timestamped YouTube URL
    """
    return f"https://youtu.be/{video_id}?t={int(start_sec)}"
