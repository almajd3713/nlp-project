"""
Linked Retrieval - Retrieve fatwas and linked hadiths.

Implements two-stage retrieval:
1. Primary: Retrieve relevant fatwas
2. Secondary: Retrieve hadiths mentioned in fatwas
"""

from typing import Optional
from dataclasses import dataclass
from loguru import logger

from ..core.retriever import Retriever, RetrievalResult
from ..data_ingestion.reference_extractor import ReferenceExtractor


@dataclass
class LinkedRetrievalResult:
    """Result with primary and linked documents."""
    
    primary_docs: list[RetrievalResult]  # Fatwas
    linked_hadiths: list[RetrievalResult]  # Referenced hadiths
    total_documents: int
    
    @property
    def all_documents(self) -> list[RetrievalResult]:
        """Get all documents (fatwas + hadiths) for context."""
        return self.primary_docs + self.linked_hadiths


class LinkedRetriever:
    """
    Retriever that follows references to fetch linked documents.
    
    Usage:
    ```python
    retriever = LinkedRetriever()
    result = retriever.retrieve_with_links("ما حكم الصلاة؟")
    
    # Use in RAG
    all_docs = result.all_documents  # Fatwas + Hadiths
    ```
    """
    
    def __init__(
        self,
        fatwa_collection: str = "fatwas",
        hadith_collection: str = "hadiths",
        fatwa_retriever: Optional[Retriever] = None,
        hadith_retriever: Optional[Retriever] = None,
    ):
        """
        Initialize linked retriever.
        
        Args:
            fatwa_collection: Fatwa collection name
            hadith_collection: Hadith collection name  
            fatwa_retriever: Pre-configured fatwa retriever
            hadith_retriever: Pre-configured hadith retriever
        """
        self.fatwa_collection = fatwa_collection
        self.hadith_collection = hadith_collection
        
        # Create retrievers
        self.fatwa_retriever = fatwa_retriever or Retriever(
            collection_name=fatwa_collection
        )
        
        self.hadith_retriever = hadith_retriever or Retriever(
            collection_name=hadith_collection
        )
        
        # If both retrievers exist, share models from fatwa to hadith retriever
        if fatwa_retriever and hadith_retriever:
            self._share_models_if_needed()
        
        self.ref_extractor = ReferenceExtractor()
    
    def _share_models_if_needed(self):
        """Share embedding model and reranker between retrievers to avoid double loading."""
        if self.fatwa_retriever._initialized and not self.hadith_retriever._initialized:
            logger.info("Sharing embedding model and reranker with hadith retriever")
            self.hadith_retriever._embedding_model = self.fatwa_retriever._embedding_model
            self.hadith_retriever._qdrant_client = self.fatwa_retriever._qdrant_client
            self.hadith_retriever._reranker = self.fatwa_retriever._reranker
            self.hadith_retriever._initialized = True
    
    def retrieve_with_links(
        self,
        query: str,
        top_k_fatwas: int = 5,
        top_k_hadiths_per_fatwa: int = 2,
        retrieve_hadiths: bool = True,
        use_semantic_fallback: bool = True,
    ) -> LinkedRetrievalResult:
        """
        Retrieve fatwas and their linked hadiths.
        
        Args:
            query: User query
            top_k_fatwas: Number of fatwas to retrieve
            top_k_hadiths_per_fatwa: Max hadiths per fatwa
            retrieve_hadiths: Whether to retrieve linked hadiths
            use_semantic_fallback: Use semantic search if pattern extraction fails
            
        Returns:
            LinkedRetrievalResult with fatwas and hadiths
        """
        # Ensure hadith retriever shares models if not yet initialized
        if self.fatwa_retriever._initialized and not self.hadith_retriever._initialized:
            self._share_models_if_needed()
        
        # Step 1: Retrieve primary fatwas
        fatwas = self.fatwa_retriever.retrieve(
            query=query,
            top_k=top_k_fatwas,
        )
        
        logger.info(f"Retrieved {len(fatwas)} fatwas for query")
        
        # Step 2: Extract hadith references and retrieve
        linked_hadiths = []
        
        if retrieve_hadiths and fatwas:
            # Ensure models are shared before ANY hadith retrieval
            if self.fatwa_retriever._initialized and not self.hadith_retriever._initialized:
                self._share_models_if_needed()
            
            # Try pattern-based extraction first
            hadith_queries = self._extract_hadith_queries(fatwas)
            
            if hadith_queries:
                logger.info(f"Found {len(hadith_queries)} hadith references via pattern extraction")
                
                # Retrieve hadiths using extracted patterns
                for hadith_query in hadith_queries[:top_k_hadiths_per_fatwa * len(fatwas)]:
                    try:
                        results = self.hadith_retriever.retrieve(
                            query=hadith_query,
                            top_k=1,
                            similarity_threshold=0.15,  # Match semantic fallback threshold
                        )
                        linked_hadiths.extend(results)
                    except Exception as e:
                        logger.warning(f"Failed to retrieve hadith for query '{hadith_query[:50]}': {e}")
                
                # Deduplicate by ID
                seen_ids = set()
                unique_hadiths = []
                for h in linked_hadiths:
                    if h.id not in seen_ids:
                        seen_ids.add(h.id)
                        unique_hadiths.append(h)
                
                linked_hadiths = unique_hadiths[:top_k_fatwas * top_k_hadiths_per_fatwa]
                logger.info(f"Retrieved {len(linked_hadiths)} unique hadiths via patterns")
            
            # Semantic fallback if no patterns found
            elif use_semantic_fallback:
                logger.info("No hadith references found via patterns, using semantic search fallback")
                linked_hadiths = self._retrieve_hadiths_by_topic(
                    query=query,
                    fatwas=fatwas,
                    top_k=min(3, top_k_hadiths_per_fatwa),
                )
                logger.info(f"Retrieved {len(linked_hadiths)} hadiths via semantic search")
            else:
                logger.info("No hadith references found and semantic fallback disabled")
        
        return LinkedRetrievalResult(
            primary_docs=fatwas,
            linked_hadiths=linked_hadiths,
            total_documents=len(fatwas) + len(linked_hadiths),
        )
    
    def _extract_hadith_queries(self, fatwas: list[RetrievalResult]) -> list[str]:
        """Extract hadith search queries from fatwa content."""
        queries = []
        
        for fatwa in fatwas:
            # Extract references
            refs = self.ref_extractor.extract_hadiths(fatwa.content)
            
            for ref in refs:
                query = ref.get('query')
                if query and len(query) > 10:  # Meaningful query
                    queries.append(query)
        
        return queries
    
    def _retrieve_hadiths_by_topic(
        self,
        query: str,
        fatwas: list[RetrievalResult],
        top_k: int = 3,
    ) -> list[RetrievalResult]:
        """
        Retrieve topically relevant hadiths using semantic search.
        
        Fallback when pattern extraction doesn't find explicit references.
        Uses the original user query to find hadiths about the same topic.
        
        Args:
            query: Original user query
            fatwas: Retrieved fatwas (for context, not currently used)
            top_k: Number of hadiths to retrieve
            
        Returns:
            List of topically relevant hadiths
        """
        try:
            # Search hadiths using the original query
            # Lower threshold since we're doing topical matching
            hadiths = self.hadith_retriever.retrieve(
                query=query,
                top_k=top_k,
                similarity_threshold=0.15,  # Lower threshold for semantic relevance
            )
            
            return hadiths
            
        except Exception as e:
            logger.error(f"Semantic hadith retrieval failed: {e}")
            return []
    
    def health_check(self) -> dict:
        """Check health of both retrievers."""
        return {
            'fatwa_retriever': self.fatwa_retriever.health_check(),
            'hadith_retriever': self.hadith_retriever.health_check(),
        }
