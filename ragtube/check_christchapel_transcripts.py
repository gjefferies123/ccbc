#!/usr/bin/env python3
"""Check Christ Chapel BC videos for transcript availability."""

from youtube_transcript_api import YouTubeTranscriptApi

def check_transcript_availability(video_id):
    """Check if a video has transcripts available."""
    try:
        api = YouTubeTranscriptApi()
        transcript_list = api.list(video_id)
        
        print(f"✅ {video_id}: HAS TRANSCRIPTS ({len(transcript_list)} options)")
        
        # Show available transcript types
        for i, transcript in enumerate(transcript_list):
            lang = getattr(transcript, 'language_code', 'unknown')
            is_generated = getattr(transcript, 'is_generated', 'unknown')
            print(f"   Option {i+1}: {lang} (Generated: {is_generated})")
        
        # Test fetching the first transcript
        try:
            first_transcript = transcript_list[0]
            content = first_transcript.fetch()
            print(f"   ✅ Successfully fetched: {len(content)} items")
            
            # Show sample content
            if content:
                sample = content[0]
                print(f"   📝 Sample: {sample.start:.1f}s - {sample.text[:50]}...")
            
            return True, len(content)
            
        except Exception as e:
            print(f"   ⚠️  Could not fetch content: {e}")
            return True, 0
            
    except Exception as e:
        print(f"❌ {video_id}: NO TRANSCRIPTS - {e}")
        return False, 0

def main():
    print("🏛️ Christ Chapel BC Transcript Availability Check")
    print("=" * 60)
    
    # Video IDs provided by user
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
    total_transcript_items = 0
    
    for i, video_id in enumerate(christ_chapel_videos, 1):
        print(f"📺 Video {i}: {video_id}")
        print(f"   URL: https://youtu.be/{video_id}")
        
        has_transcript, item_count = check_transcript_availability(video_id)
        
        if has_transcript:
            available_videos.append({
                'id': video_id,
                'transcript_items': item_count
            })
            total_transcript_items += item_count
        
        print()  # Empty line for readability
    
    # Summary
    print("=" * 60)
    print(f"📊 SUMMARY:")
    print(f"   Total videos checked: {len(christ_chapel_videos)}")
    print(f"   Videos with transcripts: {len(available_videos)}")
    print(f"   Success rate: {len(available_videos)/len(christ_chapel_videos)*100:.1f}%")
    
    if available_videos:
        print(f"   Total transcript items: {total_transcript_items}")
        print(f"\n✅ AVAILABLE VIDEOS:")
        for video in available_videos:
            print(f"      {video['id']} ({video['transcript_items']} items)")
            
        print(f"\n🚀 READY FOR INGESTION:")
        video_ids = [v['id'] for v in available_videos]
        print(f"      {video_ids}")
        
        # Create a ready-to-use list
        print(f"\n📋 Copy this for ingestion:")
        print(f'video_ids = {video_ids}')
        
    else:
        print(f"\n❌ No videos with transcripts found!")
        print(f"💡 Suggestions:")
        print(f"   - Check if videos have closed captions enabled")
        print(f"   - Try different/newer videos from the channel")
        print(f"   - Some videos may have auto-generated transcripts disabled")

if __name__ == "__main__":
    main()
