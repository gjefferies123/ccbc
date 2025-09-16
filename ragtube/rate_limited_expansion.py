#!/usr/bin/env python3
"""Rate-limited expansion to handle YouTube API rate limits."""

import os
import json
import time
import random
from youtube_transcript_api import YouTubeTranscriptApi, NoTranscriptFound, TranscriptsDisabled
from dotenv import load_dotenv
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

load_dotenv()

# New video IDs provided by user
NEW_VIDEO_IDS = [
    "JurVL5nt34U",
    "n7DIs9sg_NQ", 
    "rsr3l0zNp6Q",
    "8MUivfHHM4w",
    "_YE2Yd2GD2U",
    "zJiA-UHWcHw",
    "eCweRDNtCfQ",
    "3JWoLag6xg4",
    "UIkn-t1khDE",
    "bqb_G4zzbOQ",
    "2IgkYGzT5fo",
    # "Zncv266jSN",  # This video seems unavailable
    "9d-5h2-DzCE",
    "qllastULszc",
    "1RYR9vuhbYQ"
]

def fetch_transcript_with_retry(video_id, max_retries=3, base_delay=5):
    """Fetch transcript with retry logic and rate limiting."""
    
    for attempt in range(max_retries):
        try:
            print(f"   🔄 Attempt {attempt + 1}/{max_retries}")
            
            # Add random delay to avoid rate limiting
            if attempt > 0:
                delay = base_delay * (2 ** attempt) + random.uniform(1, 3)
                print(f"   ⏳ Waiting {delay:.1f} seconds...")
                time.sleep(delay)
            
            api = YouTubeTranscriptApi()
            transcript_list = api.list(video_id)
            
            # Try auto-generated transcript (we know they exist)
            transcript = transcript_list.find_generated_transcript(['en'])
            content = transcript.fetch()
            
            return content, 'auto'
            
        except Exception as e:
            error_str = str(e)
            
            if "IP has been blocked" in error_str or "too many requests" in error_str:
                print(f"   ⚠️ Rate limited, attempt {attempt + 1}")
                if attempt < max_retries - 1:
                    delay = base_delay * (2 ** attempt) + random.uniform(5, 10)
                    print(f"   ⏳ Backing off for {delay:.1f} seconds...")
                    time.sleep(delay)
                    continue
            else:
                print(f"   ❌ Other error: {e}")
                break
    
    return None, None

def process_videos_with_rate_limiting():
    """Process videos with proper rate limiting."""
    
    print("🔄 PROCESSING WITH RATE LIMITING")
    print("=" * 60)
    print("⚠️ YouTube is rate limiting - using delays and retries")
    print()
    
    successful_videos = []
    failed_videos = []
    
    for i, video_id in enumerate(NEW_VIDEO_IDS, 1):
        print(f"📺 Video {i}/{len(NEW_VIDEO_IDS)}: {video_id}")
        print(f"   🔗 https://youtu.be/{video_id}")
        
        # Add delay between videos to avoid rate limiting
        if i > 1:
            delay = random.uniform(3, 7)
            print(f"   ⏳ Rate limiting delay: {delay:.1f} seconds...")
            time.sleep(delay)
        
        # Try to fetch transcript
        content, transcript_type = fetch_transcript_with_retry(video_id)
        
        if content:
            print(f"   ✅ Success: {len(content)} transcript items ({transcript_type})")
            
            # Process immediately while we have the data
            processed_data = process_video_immediately(video_id, content, transcript_type)
            
            if processed_data:
                successful_videos.append({
                    'id': video_id,
                    'transcript_items': len(content),
                    'chunks': len(processed_data['chunks']),
                    'type': transcript_type
                })
                print(f"   ✅ Processed: {len(processed_data['chunks'])} chunks")
            else:
                failed_videos.append(video_id)
                print(f"   ❌ Processing failed")
        else:
            failed_videos.append(video_id)
            print(f"   ❌ Could not fetch transcript")
        
        print()  # Blank line for readability
    
    return successful_videos, failed_videos

def process_video_immediately(video_id, transcript_content, transcript_type):
    """Process video immediately while we have the transcript data."""
    
    try:
        # Convert to our format
        transcript_items = []
        for item in transcript_content:
            transcript_items.append({
                'text': item.text,
                'start': item.start,
                'duration': item.duration
            })
        
        # Create chunks
        chunks = create_chunks_optimized(video_id, transcript_items)
        
        # Calculate stats
        if transcript_items:
            total_duration = transcript_items[-1]['start'] + transcript_items[-1]['duration']
            avg_chunk_duration = sum(c['duration'] for c in chunks) / len(chunks) if chunks else 0
        else:
            total_duration = 0
            avg_chunk_duration = 0
        
        # Save processed data
        output_data = {
            'video_id': video_id,
            'url': f"https://youtu.be/{video_id}",
            'transcript_type': transcript_type,
            'stats': {
                'transcript_items': len(transcript_items),
                'total_duration': total_duration,
                'chunk_count': len(chunks),
                'avg_chunk_duration': avg_chunk_duration
            },
            'chunks': chunks,
            'sample_transcript': transcript_items[:5] if transcript_items else []
        }
        
        filename = f"processed_{video_id}.json"
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(output_data, f, indent=2, ensure_ascii=False)
        
        return output_data
        
    except Exception as e:
        print(f"   ❌ Processing error: {e}")
        return None

def create_chunks_optimized(video_id, transcript_items):
    """Create chunks using optimized approach."""
    
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

def index_processed_videos():
    """Index all successfully processed videos."""
    
    print(f"\n🔧 INDEXING PROCESSED VIDEOS")
    print("=" * 50)
    
    try:
        from production_indexer import ProductionIndexer
        
        # Find all processed files
        new_chunks = []
        processed_files = []
        
        for video_id in NEW_VIDEO_IDS:
            filename = f"processed_{video_id}.json"
            if os.path.exists(filename):
                processed_files.append(filename)
                with open(filename, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    chunks = data.get('chunks', [])
                    new_chunks.extend(chunks)
                    print(f"✅ Loaded {len(chunks)} chunks from {filename}")
        
        if not new_chunks:
            print("❌ No processed videos found to index!")
            return False
        
        print(f"📊 Total new chunks: {len(new_chunks)}")
        
        # Initialize indexer and add to existing index
        indexer = ProductionIndexer()
        
        print("🔧 Creating embeddings...")
        vectors = indexer.prepare_vectors(new_chunks)
        
        print("🚀 Adding to Pinecone index...")
        success = indexer.upsert_vectors(vectors)
        
        if success:
            try:
                stats = indexer.index.describe_index_stats()
                total_vectors = stats.total_vector_count
                print(f"✅ Index now has {total_vectors} total vectors!")
            except:
                print(f"✅ Indexing successful!")
            
            return True, len(new_chunks)
        else:
            print(f"❌ Indexing failed!")
            return False, 0
            
    except Exception as e:
        print(f"❌ Indexing error: {e}")
        return False, 0

def main():
    """Main rate-limited expansion pipeline."""
    
    print("🏛️ CHRIST CHAPEL BC INDEX EXPANSION (RATE LIMITED)")
    print("=" * 70)
    print(f"📺 Processing {len(NEW_VIDEO_IDS)} videos with rate limiting")
    print("⚠️ This will take longer but should work around YouTube's IP blocking")
    print()
    
    # Process videos with rate limiting
    successful_videos, failed_videos = process_videos_with_rate_limiting()
    
    print("=" * 60)
    print(f"📊 PROCESSING SUMMARY:")
    print(f"   Successful: {len(successful_videos)}/{len(NEW_VIDEO_IDS)}")
    print(f"   Failed: {len(failed_videos)}")
    
    if successful_videos:
        total_chunks = sum(v['chunks'] for v in successful_videos)
        total_items = sum(v['transcript_items'] for v in successful_videos)
        
        print(f"   Total chunks created: {total_chunks}")
        print(f"   Total transcript items: {total_items:,}")
        
        print(f"\n✅ SUCCESSFUL VIDEOS:")
        for video in successful_videos:
            print(f"      {video['id']}: {video['chunks']} chunks ({video['transcript_items']} items)")
        
        # Index the successful ones
        print(f"\n🔧 INDEXING...")
        success, chunks_indexed = index_processed_videos()
        
        if success:
            print(f"\n🎉 EXPANSION COMPLETE!")
            print(f"✅ Added {chunks_indexed} new sermon chunks to the index")
            print(f"🔍 Your pastor chatbot now has even more content!")
            
            # Save expansion summary
            expansion_summary = {
                'videos_processed': len(successful_videos),
                'chunks_added': chunks_indexed,
                'successful_video_ids': [v['id'] for v in successful_videos],
                'failed_video_ids': failed_videos,
                'rate_limited': True,
                'success': True
            }
            
            with open('rate_limited_expansion_summary.json', 'w') as f:
                json.dump(expansion_summary, f, indent=2)
            
            print(f"📄 Summary saved: rate_limited_expansion_summary.json")
        else:
            print(f"\n❌ Indexing failed, but videos are processed and saved")
    
    if failed_videos:
        print(f"\n❌ FAILED VIDEOS:")
        for video_id in failed_videos:
            print(f"      {video_id}")
        
        print(f"\n💡 RETRY SUGGESTIONS:")
        print(f"   - Try again later when rate limits reset")
        print(f"   - Process failed videos individually")
        print(f"   - Use a different IP/VPN if available")

if __name__ == "__main__":
    main()
