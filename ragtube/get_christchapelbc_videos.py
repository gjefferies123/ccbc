#!/usr/bin/env python3
"""Get Christ Chapel BC videos and test transcript availability."""

from youtube_transcript_api import YouTubeTranscriptApi

# Christ Chapel BC YouTube channel recent videos
# I'll manually add some video IDs from @christchapelbc channel
christchapelbc_videos = [
    # These are example IDs - we'll need to find real ones from the channel
    "TBD1",  # Will replace with real IDs
    "TBD2",  # Will replace with real IDs
    "TBD3",  # Will replace with real IDs
]

print("🏛️ Christ Chapel BC Video Testing")
print("=" * 50)

# First, let's test the transcript API with a known working video
test_video = "dQw4w9WgXcQ"  # Rick Astley - known to have transcripts
print(f"🔍 Testing transcript API with known video: {test_video}")

try:
    # Try to get transcript directly
    transcript = YouTubeTranscriptApi.fetch(test_video)
    print(f"✅ Success! Got {len(transcript)} transcript items")
    print("📝 Sample transcript item:")
    if transcript:
        item = transcript[0]
        print(f"   Time: {item.get('start', 0):.1f}s")
        print(f"   Text: {item.get('text', '')}")
except Exception as e:
    print(f"❌ fetch() failed: {e}")
    
    # Try alternative method
    try:
        transcript_list = YouTubeTranscriptApi.list(test_video)
        print(f"✅ list() worked! Found {len(transcript_list)} transcript options")
        
        # Try to get the first available transcript
        if transcript_list:
            transcript = transcript_list[0].fetch()
            print(f"✅ Got transcript with {len(transcript)} items")
    except Exception as e2:
        print(f"❌ list() also failed: {e2}")

print("\n" + "=" * 50)
print("💡 Next steps:")
print("1. Go to https://youtube.com/@christchapelbc")
print("2. Find recent videos with transcripts enabled")
print("3. Get the video IDs from the URLs")
print("4. Add them to our script")

# Let me create a helper to extract video ID from YouTube URL
def extract_video_id(url):
    """Extract video ID from YouTube URL."""
    import re
    patterns = [
        r'(?:youtube\.com/watch\?v=|youtu\.be/)([a-zA-Z0-9_-]+)',
        r'youtube\.com/embed/([a-zA-Z0-9_-]+)',
    ]
    
    for pattern in patterns:
        match = re.search(pattern, url)
        if match:
            return match.group(1)
    return None

print("\n📋 URL Helper:")
print("If you have YouTube URLs, this script can extract video IDs:")

sample_urls = [
    "https://youtube.com/watch?v=dQw4w9WgXcQ",
    "https://youtu.be/dQw4w9WgXcQ",
]

for url in sample_urls:
    video_id = extract_video_id(url)
    print(f"   {url} -> {video_id}")

print("\n🎯 Please provide some Christ Chapel BC video URLs or IDs to test!")
