#!/usr/bin/env python3
"""Fix YouTube transcript API and test with working examples."""

# Let's figure out the correct way to use youtube-transcript-api
print("🔧 Testing YouTube Transcript API usage patterns...")

# Method 1: Try direct import and usage
try:
    from youtube_transcript_api import YouTubeTranscriptApi
    
    # Test with a well-known video
    test_video = "dQw4w9WgXcQ"  # Rick Astley
    print(f"\n📺 Testing with video: {test_video}")
    
    # Try different method calls
    methods_to_try = [
        ("YouTubeTranscriptApi.get_transcript", lambda: YouTubeTranscriptApi.get_transcript(test_video)),
        ("YouTubeTranscriptApi.fetch", lambda: YouTubeTranscriptApi.fetch(test_video)),
        ("YouTubeTranscriptApi.list_transcripts", lambda: YouTubeTranscriptApi.list_transcripts(test_video)),
    ]
    
    for method_name, method_call in methods_to_try:
        try:
            result = method_call()
            print(f"✅ {method_name}: SUCCESS!")
            print(f"   Result type: {type(result)}")
            if hasattr(result, '__len__'):
                print(f"   Length: {len(result)}")
            if isinstance(result, list) and result:
                print(f"   First item: {result[0]}")
            break
        except Exception as e:
            print(f"❌ {method_name}: {e}")
    
except ImportError as e:
    print(f"❌ Import error: {e}")

# Method 2: Check what's actually available
print(f"\n🔍 Checking available methods in YouTubeTranscriptApi:")
try:
    from youtube_transcript_api import YouTubeTranscriptApi
    available_methods = [method for method in dir(YouTubeTranscriptApi) if not method.startswith('_')]
    print(f"Available methods: {available_methods}")
except:
    print("❌ Could not inspect YouTubeTranscriptApi")

# Method 3: Try alternative import patterns
alternative_imports = [
    "from youtube_transcript_api.youtube_transcript_api import YouTubeTranscriptApi",
    "from youtube_transcript_api import *",
    "import youtube_transcript_api",
]

for import_statement in alternative_imports:
    try:
        print(f"\n🧪 Trying: {import_statement}")
        exec(import_statement)
        print("✅ Import successful!")
    except Exception as e:
        print(f"❌ Failed: {e}")

print("\n" + "="*50)
print("📝 Let's also test with some manually found Christ Chapel BC videos")
print("I'll manually add some video IDs that we can test...")

# These are placeholder IDs - we'll need to find real ones
potential_christchapelbc_videos = [
    # Example format - these are not real
    "PLACEHOLDER1",  # Recent sermon
    "PLACEHOLDER2",  # Recent sermon  
    "PLACEHOLDER3",  # Recent sermon
]

print("📋 Placeholder video IDs for Christ Chapel BC:")
for i, vid in enumerate(potential_christchapelbc_videos, 1):
    print(f"   {i}. {vid}")

print("\n💡 To get real video IDs, we can:")
print("1. Visit https://youtube.com/@christchapelbc")
print("2. Copy URLs of recent videos")
print("3. Extract video IDs from the URLs")
print("4. Test transcript availability")

# Helper function to extract video ID from URL
def extract_video_id(url):
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

print("\n🔧 URL to Video ID converter ready!")
print("Example: https://youtu.be/dQw4w9WgXcQ -> dQw4w9WgXcQ")

# Let's create a simple test that should work
print("\n🎯 Creating a simple working test...")

test_urls = [
    "https://youtu.be/dQw4w9WgXcQ",  # Rick Astley
    "https://youtu.be/fJ9rUzIMcZQ",  # Another popular video
]

print("Testing video ID extraction:")
for url in test_urls:
    video_id = extract_video_id(url)
    print(f"   {url} -> {video_id}")
