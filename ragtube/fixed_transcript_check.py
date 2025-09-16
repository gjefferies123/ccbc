#!/usr/bin/env python3
"""Fixed transcript checking for Christ Chapel BC videos."""

from youtube_transcript_api import YouTubeTranscriptApi

def check_video_transcripts(video_id):
    """Check if a video has transcripts and return details."""
    try:
        api = YouTubeTranscriptApi()
        transcript_list = api.list(video_id)
        
        print(f"✅ {video_id}: HAS TRANSCRIPTS!")
        
        # Try to find manually created transcript first
        try:
            manual_transcript = transcript_list.find_manually_created_transcript(['en'])
            content = manual_transcript.fetch()
            print(f"   ✅ Manual transcript: {len(content)} items")
            return True, len(content), 'manual'
        except:
            pass
        
        # Try to find auto-generated transcript
        try:
            auto_transcript = transcript_list.find_generated_transcript(['en'])
            content = auto_transcript.fetch()
            print(f"   ✅ Auto-generated transcript: {len(content)} items")
            return True, len(content), 'auto'
        except:
            pass
        
        # Try to find any transcript
        try:
            any_transcript = transcript_list.find_transcript(['en', 'en-US', 'en-GB'])
            content = any_transcript.fetch()
            print(f"   ✅ Found transcript: {len(content)} items")
            return True, len(content), 'found'
        except:
            pass
        
        print(f"   ⚠️  Has transcript list but couldn't fetch content")
        return False, 0, 'unavailable'
        
    except Exception as e:
        print(f"❌ {video_id}: NO TRANSCRIPTS - {e}")
        return False, 0, 'none'

def main():
    print("🏛️ Christ Chapel BC Transcript Check (Fixed)")
    print("=" * 60)
    
    # Test with a known working video first
    print("🧪 Testing with known working video...")
    test_success, test_count, test_type = check_video_transcripts("dQw4w9WgXcQ")
    
    if test_success:
        print(f"✅ Test successful! API is working.\n")
    else:
        print(f"❌ Test failed! API may have issues.\n")
    
    # Christ Chapel BC videos
    christ_chapel_videos = [
        "Jz1Zb57NUMg",
        "6_HgIPUXpVM", 
        "iHypAArzphY",
        "vM_XX9P66RU",
        "OO6l1lkK3yM"
    ]
    
    print(f"🔍 Checking {len(christ_chapel_videos)} Christ Chapel BC videos...")
    print()
    
    available_videos = []
    
    for i, video_id in enumerate(christ_chapel_videos, 1):
        print(f"📺 Video {i}: https://youtu.be/{video_id}")
        
        has_transcript, item_count, transcript_type = check_video_transcripts(video_id)
        
        if has_transcript:
            available_videos.append({
                'id': video_id,
                'items': item_count,
                'type': transcript_type
            })
        
        print()  # Empty line
    
    # Results
    print("=" * 60)
    print(f"📊 RESULTS:")
    print(f"   Videos checked: {len(christ_chapel_videos)}")
    print(f"   Available: {len(available_videos)}")
    print(f"   Success rate: {len(available_videos)/len(christ_chapel_videos)*100:.1f}%")
    
    if available_videos:
        print(f"\n✅ VIDEOS WITH TRANSCRIPTS:")
        total_items = 0
        for video in available_videos:
            print(f"   {video['id']}: {video['items']} items ({video['type']})")
            total_items += video['items']
        
        print(f"\n📊 Total transcript items: {total_items}")
        
        # Ready for ingestion
        ready_ids = [v['id'] for v in available_videos]
        print(f"\n🚀 READY FOR INGESTION:")
        print(f"video_ids = {ready_ids}")
        
    else:
        print(f"\n❌ NO TRANSCRIPTS FOUND!")
        print(f"\n💡 SUGGESTIONS:")
        print(f"   1. Check if Christ Chapel BC has enabled captions")
        print(f"   2. Try newer videos (transcripts may be processing)")
        print(f"   3. Look for videos with [CC] in the title")
        print(f"   4. Some channels disable auto-captions")
        
        print(f"\n🔍 MANUAL CHECK:")
        print(f"   Visit each video and look for the CC button:")
        for video_id in christ_chapel_videos:
            print(f"   https://youtu.be/{video_id}")

if __name__ == "__main__":
    main()
