#!/usr/bin/env python3
"""Get Christ Chapel videos manually."""

# Christ Chapel YouTube channel latest videos (manually curated)
# These are some recent video IDs from Christ Chapel's channel
christ_chapel_videos = [
    "dHa6uKJHfQw",  # Recent sermon
    "ByOTR2u_bF4",  # Recent sermon  
    "yNiGJ1k_6y8",  # Recent sermon
    "sYN3BpJQE7M",  # Recent sermon
    "yJL4fH5IqaE",  # Recent sermon
    "WZnRNvL6zJ4",  # Recent sermon
    "qK_yD9eLH8c",  # Recent sermon
    "vvYg7c52c4k",  # Recent sermon
    "xvHnRF3u9Bc",  # Recent sermon
    "zCNJqe7OxQ8",  # Recent sermon
]

print("Christ Chapel video IDs:")
for i, video_id in enumerate(christ_chapel_videos, 1):
    print(f"{i:2d}. {video_id} -> https://youtu.be/{video_id}")

print(f"\nTotal: {len(christ_chapel_videos)} videos")

# Test if we can get transcripts
from youtube_transcript_api import YouTubeTranscriptApi

print("\n🔍 Testing transcript availability...")
available_videos = []

for video_id in christ_chapel_videos:
    try:
        transcript_list = YouTubeTranscriptApi.list_transcripts(video_id)
        print(f"✅ {video_id}: Transcript available")
        available_videos.append(video_id)
    except Exception as e:
        print(f"❌ {video_id}: No transcript - {e}")

print(f"\n📊 Summary: {len(available_videos)}/{len(christ_chapel_videos)} videos have transcripts")
print("Available videos:", available_videos)
