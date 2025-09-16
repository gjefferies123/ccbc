#!/usr/bin/env python3
"""Fixed expansion script that properly detects auto-generated transcripts."""

import os
import json
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
    "Zncv266jSN",
    "9d-5h2-DzCE",
    "qllastULszc",
    "1RYR9vuhbYQ"
]

def check_transcript_availability_fixed(video_ids):
    """Fixed transcript detection - same approach as first videos."""
    
    print("🔍 CHECKING TRANSCRIPT AVAILABILITY (FIXED)")
    print("=" * 60)
    
    available_videos = []
    failed_videos = []
    
    for i, video_id in enumerate(video_ids, 1):
        print(f"📺 Video {i}/{len(video_ids)}: {video_id}")
        print(f"   🔗 https://youtu.be/{video_id}")
        
        try:
            api = YouTubeTranscriptApi()
            transcript_list = api.list(video_id)
            
            # Try auto-generated first (what worked for original videos)
            try:
                transcript = transcript_list.find_generated_transcript(['en'])
                content = transcript.fetch()
                
                print(f"   ✅ Auto-generated transcript: {len(content)} items")
                available_videos.append({
                    'id': video_id,
                    'transcript_items': len(content),
                    'type': 'auto'
                })
                continue
                
            except Exception as e:
                print(f"   ⚠️ Auto-generated failed: {e}")
            
            # Try manual transcript as fallback
            try:
                transcript = transcript_list.find_manually_created_transcript(['en'])
                content = transcript.fetch()
                
                print(f"   ✅ Manual transcript: {len(content)} items")
                available_videos.append({
                    'id': video_id,
                    'transcript_items': len(content),
                    'type': 'manual'
                })
                continue
                
            except Exception as e:
                print(f"   ⚠️ Manual failed: {e}")
            
            # Try any available transcript
            try:
                transcript = transcript_list.find_transcript(['en', 'a.en'])
                content = transcript.fetch()
                
                print(f"   ✅ Found transcript: {len(content)} items")
                available_videos.append({
                    'id': video_id,
                    'transcript_items': len(content),
                    'type': 'found'
                })
                continue
                
            except Exception as e:
                print(f"   ⚠️ General search failed: {e}")
            
            print(f"   ❌ No usable transcript found")
            failed_videos.append(video_id)
            
        except NoTranscriptFound:
            print(f"   ❌ No transcript found")
            failed_videos.append(video_id)
        except TranscriptsDisabled:
            print(f"   ❌ Transcripts disabled")
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

def process_video_fixed(video_id):
    """Process a single video into chunks - same as original approach."""
    
    print(f"\n🎬 PROCESSING: {video_id}")
    print(f"🔗 https://youtu.be/{video_id}")
    
    try:
        # Fetch transcript using same approach as original videos
        api = YouTubeTranscriptApi()
        transcript_list = api.list(video_id)
        
        # Try auto-generated first (what worked originally)
        transcript_data = None
        transcript_type = None
        
        try:
            transcript = transcript_list.find_generated_transcript(['en'])
            transcript_data = transcript.fetch()
            transcript_type = 'auto'
            print(f"📝 Auto-generated transcript: {len(transcript_data)} items")
        except:
            try:
                transcript = transcript_list.find_manually_created_transcript(['en'])
                transcript_data = transcript.fetch()
                transcript_type = 'manual'
                print(f"📝 Manual transcript: {len(transcript_data)} items")
            except:
                transcript = transcript_list.find_transcript(['en', 'a.en'])
                transcript_data = transcript.fetch()
                transcript_type = 'found'
                print(f"📝 Found transcript: {len(transcript_data)} items")
        
        # Convert to our format (same as original)
        transcript_items = []
        for item in transcript_data:
            transcript_items.append({
                'text': item.text,
                'start': item.start,
                'duration': item.duration
            })
        
        # Create chunks using same approach as original
        chunks = create_chunks_same_as_original(video_id, transcript_items)
        
        print(f"📦 Created: {len(chunks)} chunks")
        
        # Calculate stats
        if transcript_items:
            total_duration = transcript_items[-1]['start'] + transcript_items[-1]['duration']
            avg_chunk_duration = sum(c['duration'] for c in chunks) / len(chunks) if chunks else 0
            
            print(f"📊 Duration: {total_duration/60:.1f} minutes")
            print(f"📊 Avg chunk: {avg_chunk_duration:.1f} seconds")
        else:
            total_duration = 0
            avg_chunk_duration = 0
        
        # Save processed data (same format as original)
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
        
        print(f"💾 Saved: {filename}")
        
        return output_data
        
    except Exception as e:
        print(f"❌ Processing failed: {e}")
        return None

def create_chunks_same_as_original(video_id, transcript_items):
    """Create chunks using the same approach as the original videos."""
    
    chunks = []
    current_chunk = []
    chunk_start = 0
    chunk_id = 0
    
    for item in transcript_items:
        if not current_chunk:
            chunk_start = item['start']
        
        current_chunk.append(item)
        
        # Create chunk when we reach ~60 seconds or have ~15 items (same as original)
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

def index_new_chunks_to_pinecone():
    """Index all new chunks to Pinecone - same as before."""
    
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
    """Main expansion pipeline with fixed transcript detection."""
    
    print("🏛️ CHRIST CHAPEL BC INDEX EXPANSION (FIXED)")
    print("=" * 70)
    print(f"📺 Adding {len(NEW_VIDEO_IDS)} new videos to the search index")
    print("🔧 Using the same transcript detection as the original working videos")
    print()
    
    # Step 1: Check transcript availability with fixed detection
    available_videos, failed_videos = check_transcript_availability_fixed(NEW_VIDEO_IDS)
    
    if not available_videos:
        print("❌ No videos with transcripts found!")
        print("💡 This is strange - let me try individual debugging...")
        
        # Debug a few videos individually
        debug_videos = NEW_VIDEO_IDS[:3]
        print(f"\n🔍 DEBUGGING FIRST 3 VIDEOS:")
        
        for video_id in debug_videos:
            try:
                api = YouTubeTranscriptApi()
                transcript_list = api.list(video_id)
                print(f"   {video_id}: Has transcript list object")
                
                # Check what transcripts are available
                available_transcripts = []
                for transcript in transcript_list:
                    lang = getattr(transcript, 'language_code', 'unknown')
                    is_generated = getattr(transcript, 'is_generated', 'unknown')
                    available_transcripts.append(f"{lang}(generated={is_generated})")
                
                print(f"      Available: {available_transcripts}")
                
            except Exception as e:
                print(f"   {video_id}: Error - {e}")
        
        return
    
    # Step 2: Process available videos
    print(f"\n🎬 PROCESSING {len(available_videos)} VIDEOS")
    print("=" * 50)
    
    processed_count = 0
    all_chunks = []
    
    for video_info in available_videos:
        video_id = video_info['id']
        processed_data = process_video_fixed(video_id)
        
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
    success = index_new_chunks_to_pinecone()
    
    if success:
        print(f"\n🎉 INDEX EXPANSION COMPLETE!")
        print(f"✅ Added {len(all_chunks)} new sermon chunks")
        print(f"🔍 Your search system now has even more content!")
        
        # Create expansion summary
        expansion_summary = {
            'videos_added': processed_count,
            'new_chunks': len(all_chunks),
            'video_ids': [v['id'] for v in available_videos if os.path.exists(f"processed_{v['id']}.json")],
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
