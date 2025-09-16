#!/usr/bin/env python3
"""Test YouTube Transcript API."""

from youtube_transcript_api import YouTubeTranscriptApi

def test_transcript():
    try:
        from youtube_transcript_api import YouTubeTranscriptApi
        
        # Get available transcripts
        transcript_list = YouTubeTranscriptApi.list('Jz1Zb57NUMg')
        
        # Try to get English auto-generated transcript
        transcript = None
        try:
            transcript = transcript_list.find_generated_transcript(['en'])
        except:
            try:
                transcript = transcript_list.find_generated_transcript()
            except:
                try:
                    transcript = next(iter(transcript_list))
                except:
                    raise Exception("No transcript available")
        
        # Fetch the transcript
        transcript_items = transcript.fetch()
        print(f'Success: {len(transcript_items)} entries')
        if transcript_items:
            print(f'First entry: {transcript_items[0]}')
        return True
    except Exception as e:
        print(f'Error: {e}')
        return False

if __name__ == "__main__":
    test_transcript()
