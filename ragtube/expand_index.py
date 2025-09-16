#!/usr/bin/env python3
"""Expand the index with additional Christ Chapel BC videos."""

import os
import json
from youtube_transcript_api import YouTubeTranscriptApi
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
    "Zncv266jSN",
    "9d-5h2-DzCE",
    "qllastULszc",
    "1RYR9vuhbYQ"
]

def check_transcript_availability(video_ids):
    """Check which videos have transcripts available."""
    
    print("🔍 CHECKING TRANSCRIPT AVAILABILITY")
    print("=" * 60)
    
    available_videos = []
    failed_videos = []
    
    for i, video_id in enumerate(video_ids, 1):
        print(f"📺 Video {i}/{len(video_ids)}: {video_id}")
        print(f"   🔗 https://youtu.be/{video_id}")
        
        try:
            api = YouTubeTranscriptApi()
            transcript_list = api.list(video_id)
            
            # Try to get transcript (prefer auto-generated English)
            try:
                transcript = transcript_list.find_generated_transcript(['en'])
                content = transcript.fetch()
                
                print(f"   ✅ Auto-generated transcript: {len(content)} items")
                available_videos.append({
                    'id': video_id,
                    'transcript_items': len(content),
                    'type': 'auto'
                })
                
            except:
                # Try manual transcript
                try:
                    transcript = transcript_list.find_manually_created_transcript(['en'])
                    content = transcript.fetch()
                    
                    print(f"   ✅ Manual transcript: {len(content)} items")
                    available_videos.append({
                        'id': video_id,
                        'transcript_items': len(content),
                        'type': 'manual'
                    })
                    
                except:
                    print(f"   ❌ No usable transcript found")
                    failed_videos.append(video_id)
        
        except Exception as e:
            print(f"   ❌ Error: {e}")
            failed_videos.append(video_id)
    
    print(f"\n📊 SUMMARY:")
    print(f"   Available: {len(available_videos)}/{len(video_ids)}")
    print(f"   Success rate: {len(available_videos)/len(video_ids)*100:.1f}%")
    
    if available_videos:
        total_items = sum(v['transcript_items'] for v in available_videos)
        print(f"   Total transcript items: {total_items:,}")
        
        print(f"\n✅ VIDEOS WITH TRANSCRIPTS:")
        for video in available_videos:
            print(f"      {video['id']}: {video['transcript_items']} items ({video['type']})")
    
    if failed_videos:
        print(f"\n❌ VIDEOS WITHOUT TRANSCRIPTS:")
        for video_id in failed_videos:
            print(f"      {video_id}")
    
    return available_videos, failed_videos

def process_video(video_id):
    """Process a single video into chunks."""
    
    print(f"\n🎬 PROCESSING: {video_id}")
    print(f"🔗 https://youtu.be/{video_id}")
    
    try:
        # Fetch transcript
        api = YouTubeTranscriptApi()
        transcript_list = api.list(video_id)
        
        # Get transcript content
        try:
            transcript = transcript_list.find_generated_transcript(['en'])
            raw_transcript = transcript.fetch()
            transcript_type = 'auto'
        except:
            transcript = transcript_list.find_manually_created_transcript(['en'])
            raw_transcript = raw_transcript.fetch()
            transcript_type = 'manual'
        
        print(f"📝 Transcript: {len(raw_transcript)} items ({transcript_type})")
        
        # Convert to our format
        transcript_items = []
        for item in raw_transcript:
            transcript_items.append({
                'text': item.text,
                'start': item.start,
                'duration': item.duration
            })
        
        # Create chunks using simplified segmentation
        chunks = create_chunks(video_id, transcript_items)
        
        print(f"📦 Created: {len(chunks)} chunks")
        
        # Calculate stats
        total_duration = transcript_items[-1]['start'] + transcript_items[-1]['duration']
        avg_chunk_duration = sum(c['duration'] for c in chunks) / len(chunks) if chunks else 0
        
        print(f"📊 Duration: {total_duration/60:.1f} minutes")
        print(f"📊 Avg chunk: {avg_chunk_duration:.1f} seconds")
        
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
            'sample_transcript': transcript_items[:3]
        }
        
        filename = f"processed_{video_id}.json"
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(output_data, f, indent=2, ensure_ascii=False)
        
        print(f"💾 Saved: {filename}")
        
        return output_data
        
    except Exception as e:
        print(f"❌ Processing failed: {e}")
        return None

def create_chunks(video_id, transcript_items):
    """Create chunks from transcript items."""
    
    chunks = []
    current_chunk = []
    chunk_start = 0
    chunk_id = 0
    
    # Target chunk duration: 45-90 seconds
    MIN_CHUNK_DURATION = 45
    MAX_CHUNK_DURATION = 90
    
    for item in transcript_items:
        if not current_chunk:
            chunk_start = item['start']
        
        current_chunk.append(item)
        
        # Calculate current chunk duration
        chunk_end = item['start'] + item['duration']
        chunk_duration = chunk_end - chunk_start
        
        # Create chunk if we've reached good duration or have enough items
        should_create_chunk = (
            chunk_duration >= MIN_CHUNK_DURATION and (
                chunk_duration >= MAX_CHUNK_DURATION or 
                len(current_chunk) >= 15
            )
        )
        
        if should_create_chunk:
            # Create chunk
            chunk_text = ' '.join([x['text'] for x in current_chunk])
            
            chunk = {
                'id': f"{video_id}_chunk_{chunk_id}",
                'video_id': video_id,
                'start_sec': chunk_start,
                'end_sec': chunk_end,
                'duration': chunk_duration,
                'text': chunk_text,
                'item_count': len(current_chunk),
                'url': f"https://youtu.be/{video_id}?t={int(chunk_start)}"
            }
            
            chunks.append(chunk)
            current_chunk = []
            chunk_id += 1
    
    # Add final chunk if needed
    if current_chunk:
        chunk_end = current_chunk[-1]['start'] + current_chunk[-1]['duration']
        chunk_text = ' '.join([x['text'] for x in current_chunk])
        
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

def index_new_chunks():
    """Index all new chunks to Pinecone."""
    
    print(f"\n🔧 INDEXING NEW CHUNKS TO PINECONE")
    print("=" * 50)
    
    try:
        from production_indexer import ProductionIndexer
        
        # Load all new processed files
        new_chunks = []
        video_files = []
        
        for video_id in NEW_VIDEO_IDS:
            filename = f"processed_{video_id}.json"
            if os.path.exists(filename):
                video_files.append(filename)
                with open(filename, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    chunks = data.get('chunks', [])
                    new_chunks.extend(chunks)
                    print(f"✅ Loaded {len(chunks)} chunks from {filename}")
        
        if not new_chunks:
            print("❌ No new chunks to index!")
            return False
        
        print(f"📊 Total new chunks: {len(new_chunks)}")
        
        # Initialize indexer
        indexer = ProductionIndexer()
        
        # Prepare vectors
        print("🔧 Creating embeddings...")
        vectors = indexer.prepare_vectors(new_chunks)
        
        # Add to existing index (don't clear)
        print("🚀 Adding to Pinecone index...")
        success = indexer.upsert_vectors(vectors)
        
        if success:
            # Get final stats
            try:
                stats = indexer.index.describe_index_stats()
                total_vectors = stats.total_vector_count
                print(f"✅ Index now has {total_vectors} total vectors!")
                
                return True
            except:
                print(f"✅ Indexing successful!")
                return True
        else:
            print(f"❌ Indexing failed!")
            return False
            
    except Exception as e:
        print(f"❌ Indexing error: {e}")
        return False

def main():
    """Main expansion pipeline."""
    
    print("🏛️ CHRIST CHAPEL BC INDEX EXPANSION")
    print("=" * 70)
    print(f"📺 Adding {len(NEW_VIDEO_IDS)} new videos to the search index")
    print()
    
    # Step 1: Check transcript availability
    available_videos, failed_videos = check_transcript_availability(NEW_VIDEO_IDS)
    
    if not available_videos:
        print("❌ No videos with transcripts found!")
        return
    
    # Step 2: Process available videos
    print(f"\n🎬 PROCESSING {len(available_videos)} VIDEOS")
    print("=" * 50)
    
    processed_count = 0
    all_chunks = []
    
    for video_info in available_videos:
        video_id = video_info['id']
        processed_data = process_video(video_id)
        
        if processed_data:
            processed_count += 1
            all_chunks.extend(processed_data['chunks'])
    
    if processed_count == 0:
        print("❌ No videos processed successfully!")
        return
    
    print(f"\n📊 PROCESSING SUMMARY:")
    print(f"   Videos processed: {processed_count}/{len(available_videos)}")
    print(f"   Total new chunks: {len(all_chunks)}")
    
    # Step 3: Index to Pinecone
    success = index_new_chunks()
    
    if success:
        print(f"\n🎉 INDEX EXPANSION COMPLETE!")
        print(f"✅ Added {len(all_chunks)} new sermon chunks")
        print(f"🔍 Your search system now has even more content!")
        
        # Create expansion summary
        expansion_summary = {
            'videos_added': processed_count,
            'new_chunks': len(all_chunks),
            'video_ids': [v['id'] for v in available_videos if f"processed_{v['id']}.json" in [f"processed_{vid}.json" for vid in [d['video_id'] for d in [json.load(open(f"processed_{v['id']}.json")) for v in available_videos if os.path.exists(f"processed_{v['id']}.json")]]]],
            'failed_videos': failed_videos,
            'success': True
        }
        
        with open('index_expansion_summary.json', 'w') as f:
            json.dump(expansion_summary, f, indent=2)
        
        print(f"📄 Summary saved: index_expansion_summary.json")
        
    else:
        print(f"\n❌ Index expansion failed!")

if __name__ == "__main__":
    main()
