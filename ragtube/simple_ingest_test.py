#!/usr/bin/env python3
"""Simple ingestion test for any YouTube videos."""

import os
import json
from dotenv import load_dotenv
from youtube_transcript_api import YouTubeTranscriptApi

# Load environment
load_dotenv()

def fetch_transcript(video_id):
    """Fetch transcript for a video."""
    try:
        api = YouTubeTranscriptApi()
        transcript_items = api.fetch(video_id)
        
        # Convert to dictionary format
        transcript = []
        for item in transcript_items:
            transcript.append({
                'text': item.text,
                'start': item.start,
                'duration': item.duration
            })
        
        return transcript
    except Exception as e:
        print(f"❌ Error fetching transcript for {video_id}: {e}")
        return None

def test_simple_pipeline(video_id, video_title="Test Video"):
    """Test the basic pipeline steps."""
    print(f"\n🚀 Testing pipeline with video: {video_id}")
    print(f"📺 Title: {video_title}")
    print("-" * 50)
    
    # Step 1: Fetch transcript
    print("🔍 Step 1: Fetching transcript...")
    transcript = fetch_transcript(video_id)
    
    if not transcript:
        print("❌ No transcript available. Cannot proceed.")
        return False
    
    print(f"✅ Got transcript with {len(transcript)} items")
    
    # Step 2: Show transcript info
    total_duration = transcript[-1]['start'] + transcript[-1]['duration']
    print(f"📊 Video duration: {total_duration/60:.1f} minutes")
    
    # Step 3: Show sample content
    print(f"\n📝 Sample transcript content:")
    for i, item in enumerate(transcript[:5]):
        print(f"   {item['start']:6.1f}s: {item['text'][:60]}...")
    
    # Step 4: Basic segmentation test
    print(f"\n🔧 Step 2: Basic segmentation test...")
    
    # Simple chunking - group items into ~60 second chunks
    chunks = []
    current_chunk = []
    chunk_start = 0
    
    for item in transcript:
        if not current_chunk:
            chunk_start = item['start']
        
        current_chunk.append(item)
        
        # If chunk is getting long (~60 seconds), finalize it
        if item['start'] - chunk_start >= 60:
            chunk_text = ' '.join([x['text'] for x in current_chunk])
            chunks.append({
                'start_sec': chunk_start,
                'end_sec': item['start'] + item['duration'],
                'text': chunk_text,
                'item_count': len(current_chunk)
            })
            current_chunk = []
    
    # Add final chunk
    if current_chunk:
        chunk_text = ' '.join([x['text'] for x in current_chunk])
        chunks.append({
            'start_sec': chunk_start,
            'end_sec': current_chunk[-1]['start'] + current_chunk[-1]['duration'],
            'text': chunk_text,
            'item_count': len(current_chunk)
        })
    
    print(f"✅ Created {len(chunks)} chunks")
    
    # Show sample chunks
    print(f"\n📦 Sample chunks:")
    for i, chunk in enumerate(chunks[:3]):
        duration = chunk['end_sec'] - chunk['start_sec']
        print(f"   Chunk {i+1}: {chunk['start_sec']:.1f}s-{chunk['end_sec']:.1f}s ({duration:.1f}s)")
        print(f"      Text: {chunk['text'][:80]}...")
        print(f"      Items: {chunk['item_count']}")
    
    # Step 5: Save sample data
    print(f"\n💾 Step 3: Saving sample data...")
    
    sample_data = {
        'video_id': video_id,
        'title': video_title,
        'transcript_items': len(transcript),
        'total_duration': total_duration,
        'chunks': len(chunks),
        'sample_chunks': chunks[:2],  # Save first 2 chunks as examples
        'sample_transcript': transcript[:10]  # Save first 10 items as examples
    }
    
    output_file = f"sample_data_{video_id}.json"
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(sample_data, f, indent=2, ensure_ascii=False)
    
    print(f"✅ Saved sample data to: {output_file}")
    
    print(f"\n✅ Pipeline test successful!")
    print(f"🎯 This video is ready for full ingestion!")
    
    return True

def main():
    print("🎯 Simple YouTube Video Ingestion Test")
    print("=" * 60)
    
    # Test videos - replace these with Christ Chapel BC video IDs
    test_videos = [
        {
            'id': 'dQw4w9WgXcQ',
            'title': 'Rick Astley - Never Gonna Give You Up'
        },
        # Add Christ Chapel BC videos here:
        # {
        #     'id': 'CHRIST_CHAPEL_VIDEO_ID',
        #     'title': 'Christ Chapel BC - Sermon Title'
        # },
    ]
    
    print("📋 Videos to test:")
    for i, video in enumerate(test_videos, 1):
        print(f"   {i}. {video['id']} - {video['title']}")
    
    # Test each video
    successful_tests = 0
    for video in test_videos:
        success = test_simple_pipeline(video['id'], video['title'])
        if success:
            successful_tests += 1
    
    print(f"\n🏆 Results: {successful_tests}/{len(test_videos)} videos processed successfully")
    
    if successful_tests > 0:
        print(f"\n🚀 Ready for next steps:")
        print(f"1. ✅ Transcript fetching works")
        print(f"2. ✅ Basic segmentation works") 
        print(f"3. 🔄 Next: Set up Pinecone indexing")
        print(f"4. 🔄 Next: Test Cohere embeddings")
        print(f"5. 🔄 Next: Test full RAG pipeline")
    
    print(f"\n🏛️ TO ADD CHRIST CHAPEL BC VIDEOS:")
    print(f"1. Visit: https://youtube.com/@christchapelbc")
    print(f"2. Find recent videos with transcripts")
    print(f"3. Copy video IDs and add them to the test_videos list above")
    print(f"4. Run this script again")

if __name__ == "__main__":
    main()
