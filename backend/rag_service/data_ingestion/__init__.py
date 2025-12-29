"""
Data Ingestion Module - Load fatwas and hadiths into vector database.

Handles:
- Loading JSON/JSONL fatwa files
- Extracting and indexing hadiths
- Creating embeddings
- Uploading to Qdrant
"""

from .fatwa_loader import FatwaLoader
from .hadith_loader import HadithLoader
from .indexer import VectorIndexer
from .reference_extractor import ReferenceExtractor

__all__ = [
    "FatwaLoader",
    "HadithLoader", 
    "VectorIndexer",
    "ReferenceExtractor",
]
