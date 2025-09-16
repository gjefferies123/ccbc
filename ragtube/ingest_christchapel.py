#!/usr/bin/env python3
"""Ingest Christ Chapel BC videos into the RAG system."""

import os
import json
from dotenv import load_dotenv
from youtube_transcript_api import YouTubeTranscriptApi

# Load environment
load_dotenv()

# Christ Chapel BC video IDs (confirmed to have transcripts)
CHRIST_CHAPEL_VIDEOS = [
    'Jz1Zb57NUMg',  # 876 items
    '6_HgIPUXpVM',  # 1003 items
    'iHypAArzphY',  # 1038 items
    'vM_XX9P66RU',  # 945 items
    'OO6l1lkK3yM'   # 943 items
]

def fetch_transcript(video_id):
    """Fetch transcript for a video."""
    try:
        api = YouTubeTranscriptApi()
        transcript_list = api.list(video_id)
        
        # Try auto-generated first (what we know works)
        transcript = transcript_list.find_generated_transcript(['en'])
        content = transcript.fetch()
        
        # Convert to our format
        result = []
        for item in content:
            result.append({
                'text': item.text,
                'start': item.start,
                'duration': item.duration
            })
        
        return result
        
    except Exception as e:
        print(f"❌ Error fetching transcript for {video_id}: {e}")
        return None

def segment_transcript(transcript_items, video_id):
    """Create basic segments from transcript."""
    if not transcript_items:
        return []
    
    chunks = []
    current_chunk = []
    chunk_start = 0
    chunk_id = 0
    
    for item in transcript_items:
        if not current_chunk:
            chunk_start = item['start']
        
        current_chunk.append(item)
        
        # Create chunk when we reach ~60 seconds or have ~15 items
        chunk_duration = item['start'] + item['duration'] - chunk_start
        if chunk_duration >= 60 or len(current_chunk) >= 15:
            
            # Create chunk
            chunk_text = ' '.join([x['text'] for x in current_chunk])
            chunk_end = item['start'] + item['duration']
            
            chunk = {
                'id': f"{video_id}_chunk_{chunk_id}",
                'video_id': video_id,
                'start_sec': chunk_start,
                'end_sec': chunk_end,
                'duration': chunk_end - chunk_start,
                'text': chunk_text,
                'item_count': len(current_chunk),
                'url': f"https://youtu.be/{video_id}?t={int(chunk_start)}"
            }
            
            chunks.append(chunk)
            current_chunk = []
            chunk_id += 1
    
    # Add final chunk if needed
    if current_chunk:
        chunk_text = ' '.join([x['text'] for x in current_chunk])
        chunk_end = current_chunk[-1]['start'] + current_chunk[-1]['duration']
        
        chunk = {
            'id': f"{video_id}_chunk_{chunk_id}",
            'video_id': video_id,
            'start_sec': chunk_start,
            'end_sec': chunk_end,
            'duration': chunk_end - chunk_start,
            'text': chunk_text,
            'item_count': len(current_chunk),
            'url': f"https://youtu.be/{video_id}?t={int(chunk_start)}"
        }
        chunks.append(chunk)
    
    return chunks

def test_cohere_embeddings(sample_texts):
    """Test Cohere embeddings with sample text."""
    try:
        from search.encoder_cohere import CohereEncoder
        
        encoder = CohereEncoder()
        if not encoder.is_available():
            print("⚠️  Cohere encoder not available, skipping embedding test")
            return False
        
        print("🧪 Testing Cohere embeddings...")
        embeddings = encoder.encode_documents(sample_texts[:3])  # Test with first 3
        
        print(f"✅ Cohere embeddings work! Shape: {embeddings.shape}")
        print(f"   Dimension: {embeddings.shape[1]}")
        print(f"   Sample embedding norm: {(embeddings[0]**2).sum()**0.5:.3f}")
        
        return True
        
    except Exception as e:
        print(f"❌ Cohere embeddings failed: {e}")
        return False

def process_video(video_id):
    """Process a single video."""
    print(f"\n📺 Processing video: {video_id}")
    print(f"🔗 URL: https://youtu.be/{video_id}")
    
    # Step 1: Fetch transcript
    print("🔍 Fetching transcript...")
    transcript = fetch_transcript(video_id)
    
    if not transcript:
        return None
    
    print(f"✅ Got {len(transcript)} transcript items")
    
    # Step 2: Segment
    print("🔧 Creating segments...")
    chunks = segment_transcript(transcript, video_id)
    
    print(f"✅ Created {len(chunks)} chunks")
    
    # Step 3: Calculate stats
    total_duration = transcript[-1]['start'] + transcript[-1]['duration']
    avg_chunk_duration = sum(c['duration'] for c in chunks) / len(chunks)
    
    print(f"📊 Stats:")
    print(f"   Total duration: {total_duration/60:.1f} minutes")
    print(f"   Average chunk: {avg_chunk_duration:.1f} seconds")
    print(f"   Text sample: {chunks[0]['text'][:80]}...")
    
    # Save processed data
    output_data = {
        'video_id': video_id,
        'url': f"https://youtu.be/{video_id}",
        'stats': {
            'transcript_items': len(transcript),
            'total_duration': total_duration,
            'chunk_count': len(chunks),
            'avg_chunk_duration': avg_chunk_duration
        },
        'chunks': chunks,
        'sample_transcript': transcript[:5]  # First 5 items for reference
    }
    
    filename = f"processed_{video_id}.json"
    with open(filename, 'w', encoding='utf-8') as f:
        json.dump(output_data, f, indent=2, ensure_ascii=False)
    
    print(f"💾 Saved to: {filename}")
    
    return output_data

def main():
    print("🏛️ Christ Chapel BC Video Ingestion")
    print("=" * 60)
    
    # Process all videos
    all_data = []
    all_chunks = []
    
    for i, video_id in enumerate(CHRIST_CHAPEL_VIDEOS, 1):
        print(f"\n🎯 Processing video {i}/{len(CHRIST_CHAPEL_VIDEOS)}")
        
        data = process_video(video_id)
        if data:
            all_data.append(data)
            all_chunks.extend(data['chunks'])
    
    # Summary
    print(f"\n" + "=" * 60)
    print(f"📊 PROCESSING SUMMARY:")
    print(f"   Videos processed: {len(all_data)}/{len(CHRIST_CHAPEL_VIDEOS)}")
    print(f"   Total chunks created: {len(all_chunks)}")
    
    if all_chunks:
        total_duration = sum(d['stats']['total_duration'] for d in all_data)
        print(f"   Total content: {total_duration/60:.1f} minutes")
        
        # Test embeddings with sample chunks
        print(f"\n🧪 Testing embeddings with sample chunks...")
        sample_texts = [chunk['text'] for chunk in all_chunks[:5]]
        
        embeddings_work = test_cohere_embeddings(sample_texts)
        
        # Create final summary
        summary = {
            'videos_processed': len(all_data),
            'total_chunks': len(all_chunks),
            'total_duration_minutes': total_duration/60,
            'embeddings_tested': embeddings_work,
            'video_ids': [d['video_id'] for d in all_data],
            'sample_chunks': all_chunks[:3]  # First 3 for reference
        }
        
        with open('christchapel_ingestion_summary.json', 'w', encoding='utf-8') as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)
        
        print(f"\n✅ INGESTION COMPLETE!")
        print(f"📄 Summary saved to: christchapel_ingestion_summary.json")
        
        if embeddings_work:
            print(f"\n🚀 READY FOR NEXT STEPS:")
            print(f"   1. ✅ Transcripts fetched and segmented")
            print(f"   2. ✅ Cohere embeddings working")
            print(f"   3. 🔄 Next: Set up Pinecone indexing")
            print(f"   4. 🔄 Next: Test full search pipeline")
        else:
            print(f"\n⚠️  EMBEDDINGS NOT WORKING:")
            print(f"   Check Cohere API key and connection")
    
    else:
        print(f"\n❌ No videos processed successfully!")

if __name__ == "__main__":
    main()
