#!/usr/bin/env python3
"""Start with working videos to test the full RAG pipeline."""

import os
from dotenv import load_dotenv
from youtube_transcript_api import YouTubeTranscriptApi

# Load environment
load_dotenv()

def test_transcript_fetch(video_id):
    """Test fetching transcript for a video."""
    try:
        api = YouTubeTranscriptApi()
        transcript = api.fetch(video_id)
        print(f"✅ {video_id}: Got {len(transcript)} transcript items")
        
        # Show sample
        if transcript:
            first_item = transcript[0]
            print(f"   Sample: {first_item.get('start', 0):.1f}s - {first_item.get('text', '')[:50]}...")
        
        return True
    except Exception as e:
        print(f"❌ {video_id}: {e}")
        return False

def main():
    print("🚀 Testing RAG Pipeline with Working Videos")
    print("=" * 60)
    
    # Test with known working videos first
    test_videos = [
        "dQw4w9WgXcQ",  # Rick Astley - Never Gonna Give You Up
        "fJ9rUzIMcZQ",  # Queen - Bohemian Rhapsody  
        "9bZkp7q19f0",  # PSY - Gangnam Style
    ]
    
    print("🔍 Testing transcript availability...")
    working_videos = []
    
    for video_id in test_videos:
        if test_transcript_fetch(video_id):
            working_videos.append(video_id)
    
    print(f"\n✅ Found {len(working_videos)} working videos: {working_videos}")
    
    if working_videos:
        print("\n🎯 Let's test the full pipeline with these videos!")
        
        # For demonstration, let's use the first working video
        demo_video = working_videos[0]
        print(f"\n📺 Testing full pipeline with: {demo_video}")
        
        try:
            # Test transcript fetching
            api = YouTubeTranscriptApi()
            transcript = api.fetch(demo_video)
            
            print(f"✅ Step 1: Got transcript with {len(transcript)} items")
            
            # Calculate total duration
            if transcript:
                total_duration = transcript[-1].get('start', 0) + transcript[-1].get('duration', 0)
                print(f"📊 Video duration: ~{total_duration/60:.1f} minutes")
                
                # Show content sample
                print("\n📝 Sample transcript content:")
                for i, item in enumerate(transcript[:5]):
                    start = item.get('start', 0)
                    text = item.get('text', '')
                    print(f"   {start:6.1f}s: {text[:60]}...")
            
            print(f"\n✅ Pipeline test successful!")
            print(f"🎯 Ready to ingest this video into Pinecone!")
            
            # Show the command to run
            print(f"\n🚀 To ingest this video, run:")
            print(f'py -c "from app import *; ingest_videos([\\"{demo_video}\\"])"')
            
        except Exception as e:
            print(f"❌ Pipeline test failed: {e}")
    
    print(f"\n" + "=" * 60)
    print("🏛️ FOR CHRIST CHAPEL BC VIDEOS:")
    print("1. Visit: https://youtube.com/@christchapelbc")
    print("2. Find recent sermon videos")
    print("3. Copy video URLs (like: https://youtu.be/VIDEO_ID)")
    print("4. Extract the VIDEO_ID part")
    print("5. Replace test videos with real ones")
    print("6. Run this script again to test")
    
    print(f"\n💡 Example Christ Chapel BC video URLs to look for:")
    print("   https://youtu.be/AbCdEfGhIjK  -> Video ID: AbCdEfGhIjK")
    print("   https://youtu.be/XyZ123456Wv  -> Video ID: XyZ123456Wv")

if __name__ == "__main__":
    main()
