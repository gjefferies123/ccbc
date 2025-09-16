#!/usr/bin/env python3
"""Working transcript test with correct API usage."""

from youtube_transcript_api import YouTubeTranscriptApi

def test_transcript_api():
    """Test the correct way to use YouTube Transcript API."""
    test_video = "dQw4w9WgXcQ"  # Rick Astley - known to have transcripts
    
    print(f"🔍 Testing transcript for video: {test_video}")
    
    try:
        # Method 1: Try to list available transcripts
        transcript_list = YouTubeTranscriptApi.list(test_video)
        print(f"✅ Found {len(transcript_list)} transcript options")
        
        # Get the first available transcript
        if transcript_list:
            transcript = transcript_list[0]
            print(f"📝 Using transcript: {transcript}")
            
            # Fetch the actual transcript content
            content = transcript.fetch()
            print(f"✅ Got transcript content with {len(content)} items")
            
            # Show sample content
            print("\n📋 Sample transcript items:")
            for i, item in enumerate(content[:3]):
                start_time = item.get('start', 0)
                text = item.get('text', '')
                print(f"   {i+1}. {start_time:.1f}s: {text}")
            
            return True
    except Exception as e:
        print(f"❌ list() method failed: {e}")
    
    try:
        # Method 2: Try direct fetch
        content = YouTubeTranscriptApi.fetch(test_video)
        print(f"✅ Direct fetch worked! Got {len(content)} items")
        return True
    except Exception as e:
        print(f"❌ fetch() method failed: {e}")
    
    return False

def get_christchapelbc_videos():
    """Get some actual Christ Chapel BC video IDs for testing."""
    # Since we don't have YouTube API, I'll manually provide some video IDs
    # These would need to be manually collected from the channel
    
    print("\n🏛️ Christ Chapel BC Video Collection")
    print("=" * 50)
    
    # Example URLs that we'd collect manually from https://youtube.com/@christchapelbc
    sample_urls = [
        "https://youtu.be/EXAMPLE1",  # Replace with real URLs
        "https://youtu.be/EXAMPLE2",  # Replace with real URLs  
        "https://youtu.be/EXAMPLE3",  # Replace with real URLs
    ]
    
    import re
    
    video_ids = []
    for url in sample_urls:
        match = re.search(r'(?:youtube\.com/watch\?v=|youtu\.be/)([a-zA-Z0-9_-]+)', url)
        if match:
            video_ids.append(match.group(1))
    
    print(f"📋 Extracted video IDs: {video_ids}")
    
    # For now, let's test with some known videos that have transcripts
    known_working_videos = [
        "dQw4w9WgXcQ",  # Rick Astley
        "fJ9rUzIMcZQ",  # Queen
        "9bZkp7q19f0",  # PSY - Gangnam Style (usually has transcripts)
    ]
    
    print(f"\n🧪 Testing with known working videos first:")
    working_videos = []
    
    for video_id in known_working_videos:
        try:
            transcript_list = YouTubeTranscriptApi.list(video_id)
            print(f"✅ {video_id}: Has transcripts ({len(transcript_list)} options)")
            working_videos.append(video_id)
        except Exception as e:
            print(f"❌ {video_id}: No transcripts - {e}")
    
    return working_videos

if __name__ == "__main__":
    print("🚀 YouTube Transcript API Testing")
    print("=" * 50)
    
    # Test the API
    api_working = test_transcript_api()
    
    if api_working:
        print("\n✅ API is working! Ready to test Christ Chapel BC videos.")
        working_videos = get_christchapelbc_videos()
        
        print(f"\n🎯 Ready to use these video IDs for testing: {working_videos}")
        print("\n📝 Next steps:")
        print("1. Visit https://youtube.com/@christchapelbc")
        print("2. Find recent sermon videos")
        print("3. Copy their URLs and extract video IDs")
        print("4. Replace the EXAMPLE IDs with real ones")
        print("5. Test transcript availability")
        
    else:
        print("\n❌ API not working properly. Need to debug further.")

    print("\n💡 For now, we can test the full pipeline with working video IDs!")
