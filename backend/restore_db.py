"""
Restore Qdrant collections from snapshots.

Instructions:
1. Make sure Qdrant is running (Docker or standalone)
2. Place this script in the same directory as your qdrant_export folder
3. Update the Qdrant connection settings below if needed
4. Run: python restore_db.py
"""

from qdrant_client import QdrantClient
from pathlib import Path
import sys

# ============================================================
# CONFIGURATION - Update these if needed
# ============================================================
QDRANT_HOST = "localhost"
QDRANT_PORT = 6333
QDRANT_API_KEY = None  # Set if using Qdrant Cloud
EXPORT_DIR = Path("../qdrant_export")  # Path to your exported snapshots

# ============================================================
# RESTORE SCRIPT
# ============================================================

def main():
    print("="*60)
    print("Qdrant Database Restore")
    print("="*60)
    
    # Check if export directory exists
    if not EXPORT_DIR.exists():
        print(f"✗ Export directory not found: {EXPORT_DIR.absolute()}")
        print(f"\nMake sure the qdrant_export folder is in the correct location.")
        print(f"Current script location: {Path(__file__).parent.absolute()}")
        sys.exit(1)
    
    # Find all snapshots
    snapshots = []
    for collection_dir in EXPORT_DIR.iterdir():
        if collection_dir.is_dir():
            for snapshot_file in collection_dir.glob("*.snapshot"):
                snapshots.append({
                    'collection': collection_dir.name,
                    'path': snapshot_file,
                    'filename': snapshot_file.name
                })
    
    if not snapshots:
        print(f"✗ No snapshots found in {EXPORT_DIR.absolute()}")
        sys.exit(1)
    
    print(f"\nFound {len(snapshots)} snapshot(s) to restore:")
    for s in snapshots:
        print(f"  - {s['collection']}: {s['filename']}")
    
    # Connect to Qdrant
    print(f"\n{'='*60}")
    print(f"Connecting to Qdrant at {QDRANT_HOST}:{QDRANT_PORT}")
    print(f"{'='*60}")
    
    try:
        client = QdrantClient(
            host=QDRANT_HOST,
            port=QDRANT_PORT,
            api_key=QDRANT_API_KEY,
            timeout=300
        )
        # Test connection
        collections = client.get_collections()
        print(f"✓ Connected successfully")
        print(f"Existing collections: {[c.name for c in collections.collections]}")
    except Exception as e:
        print(f"✗ Failed to connect to Qdrant: {e}")
        print(f"\nMake sure Qdrant is running:")
        print(f"  Docker: docker run -p 6333:6333 qdrant/qdrant")
        print(f"  Or check your connection settings in this script")
        sys.exit(1)
    
    # Restore each snapshot
    print(f"\n{'='*60}")
    print(f"Restoring collections")
    print(f"{'='*60}")
    
    for snapshot in snapshots:
        collection_name = snapshot['collection']
        snapshot_path = snapshot['path']
        
        print(f"\nRestoring: {collection_name}")
        
        try:
            # Check if collection already exists
            existing_collections = [c.name for c in client.get_collections().collections]
            if collection_name in existing_collections:
                response = input(f"  ⚠ Collection '{collection_name}' already exists. Overwrite? (y/N): ")
                if response.lower() != 'y':
                    print(f"  ⏭ Skipped {collection_name}")
                    continue
                
                # Delete existing collection
                client.delete_collection(collection_name)
                print(f"  Deleted existing collection")
            
            # Recover from snapshot
            print(f"  Restoring from {snapshot['filename']}...")
            client.recover_snapshot(
                collection_name=collection_name,
                snapshot_path=str(snapshot_path.absolute())
            )
            
            # Verify restoration
            info = client.get_collection(collection_name)
            print(f"  ✓ Restored successfully!")
            print(f"    - Vectors: {info.points_count}")
            print(f"    - Vector size: {info.config.params.vectors.size}")
            
        except Exception as e:
            print(f"  ✗ Failed to restore {collection_name}: {e}")
            continue
    
    print(f"\n{'='*60}")
    print(f"Restore complete!")
    print(f"{'='*60}")
    
    # Show final status
    collections = client.get_collections()
    print(f"\nFinal collections in database:")
    for collection in collections.collections:
        info = client.get_collection(collection.name)
        print(f"  - {collection.name}: {info.points_count} vectors")
    
    print(f"\n✓ All done! Your Qdrant database is ready to use.")


if __name__ == "__main__":
    main()
