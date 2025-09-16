#!/usr/bin/env python3
"""Test transcript functionality with known videos."""

from youtube_transcript_api import YouTubeTranscriptApi

# Test with a well-known video that definitely has transcripts
test_video_id = "dQw4w9WgXcQ"  # Rick Astley - Never Gonna Give You Up

print(f"🔍 Testing transcript API with video: {test_video_id}")

try:
    # List available transcripts
    transcript_list = YouTubeTranscriptApi.list(test_video_id)
    print(f"✅ Transcript list method works!")
    print(f"📋 Available transcripts: {len(transcript_list)}")
    
    # Get the transcript
    transcript = YouTubeTranscriptApi.get_transcript(test_video_id)
    print(f"✅ Got transcript with {len(transcript)} items")
    print("📝 First few items:")
    for i, item in enumerate(transcript[:3]):
        print(f"   {i+1}. {item['start']:.1f}s: {item['text']}")
        
except Exception as e:
    print(f"❌ Error: {e}")

# Now let's try some actual Christ Chapel videos
# I'll search for real Christ Chapel video IDs
christ_chapel_test_videos = [
    "r0n-1RJwZHQ",  # Christ Chapel example (needs real IDs)
    "123456789XX",  # This will definitely fail
]

print(f"\n🏛️ Testing Christ Chapel videos...")
for video_id in christ_chapel_test_videos:
    try:
        transcript = YouTubeTranscriptApi.get_transcript(video_id)
        print(f"✅ {video_id}: Got transcript with {len(transcript)} items")
    except Exception as e:
        print(f"❌ {video_id}: {e}")

print("\n💡 Note: We need real Christ Chapel video IDs that have transcripts enabled.")
print("Let me search for the actual Christ Chapel YouTube channel...")

# Let's try some educational/religious content that typically has transcripts
sample_videos_with_transcripts = [
    "dQw4w9WgXcQ",  # Rick Astley (definitely has transcripts)
    "XqZsoesa55w",  # Example educational video
    "OiQ1YhQ6mfI",  # Another example
]

print(f"\n📺 Testing sample videos with known transcripts...")
working_videos = []

for video_id in sample_videos_with_transcripts:
    try:
        transcript = YouTubeTranscriptApi.get_transcript(video_id)
        print(f"✅ {video_id}: {len(transcript)} transcript items")
        working_videos.append(video_id)
    except Exception as e:
        print(f"❌ {video_id}: {e}")

print(f"\n🎯 Working videos for testing: {working_videos}")
