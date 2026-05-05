"""
Hybrid search retriever combining full-text and vector-based search.

This module provides the HybridRetriever class that:
    - Integrates keyword/phrase/prefix-based FTS results.
    - Integrates semantic vector search results (Chroma DB).
    - Computes hybrid scores for ranking results across both search types.
    - Returns deduplicated and ranked SearchResult objects.
"""

# Local imports
from pydantic_models import SearchResult
from utils import setup_logger

logger = setup_logger("hybrid_search")
       

class HybridRetriever:
    """
    Combines FTS and vector search results into a hybrid ranking.

    Attributes:
        fts (FTSStore): Instance of FTSStore for keyword/phrase/prefix search.
        vector (Any): Vector store instance (Chroma DB).
        fts_weight (float): Weight of FTS scores in hybrid scoring.
        vector_similarity_weight (float): Weight for vector similarity scores.
        vector_mmr_weight (float): Weight for vector MMR scores.
        fts_max_score (float): Maximum expected FTS score (for normalization).
    """

    def __init__(
            self,
            fts_store,
            vector_store,
            fts_weight: float,
            vector_similarity_weight: float,
            vector_mmr_weight: float,
    ):
        """
        Wire up the retriever with the two backends and their fusion weights.

        Backend weights are kept as plain attributes so the dashboard can
        mutate them at runtime (the agent picks up the new values on the
        next search call without rebuilding the retriever).

        Args:
            fts_store: FTS5-backed keyword store (see FTSStore).
            vector_store: Embedding-backed vector store (Chroma).
            fts_weight (float): Multiplier applied to FTS scores.
            vector_similarity_weight (float): Multiplier for Chroma similarity scores.
            vector_mmr_weight (float): Multiplier for synthetic MMR rank scores.
        """

        self.fts = fts_store
        self.vector = vector_store
        self.fts_weight = fts_weight
        self.vector_similarity_weight = vector_similarity_weight
        self.vector_mmr_weight = vector_mmr_weight


    def _hybrid_score(self, doc: SearchResult) -> float:
        """
        Map a single backend-specific raw score onto a weighted [0, 1] scale.

        Each retrieval backend produces scores on a different scale:
            - fts               — negative BM25 (smaller = better),
                                       normalised by fts_max_score.
            - vector_similarity — Chroma cosine relevance in [0, 1].
            - vector_mmr        — synthetic rank-based score in [0, 1].

        Each is capped, normalised to [0, 1], and multiplied by the
        corresponding backend weight.  Unknown backends are logged and
        scored 0.0 so label drift is visible rather than silent.

        Args:
            doc (SearchResult): A single retrieved chunk.

        Returns:
            float: Weighted hybrid score used for ranking.
        """

        # Ensure we have a numeric score to work with, defaulting to 0 if None
        raw_score = doc.score or 0

        if doc.backend == "fts":
            return raw_score * self.fts_weight
        
        # Chroma DB similarity search with relevance scores: 1 best, 0 worst.
        elif doc.backend == "vector_similarity":
            norm_score = max(raw_score, 0)
            return norm_score * self.vector_similarity_weight

        # Synthetic score based on rank: Best rank (i=0) gets highest score (score = (1 - (i / k))).
        elif doc.backend == "vector_mmr":
            norm_score = max(raw_score, 0)
            return norm_score * self.vector_mmr_weight

        # Fallback for unknown backend — log so silent label drift is visible.
        logger.warning(
            f"_hybrid_score: unknown backend {doc.backend!r} (chunk_id={doc.chunk_id}); "
            "scoring as 0.0."
        )
        return 0.0


    def search(
            self,
            query: str,
            k: int = 5, 
            vector_search_method: str = "similarity", 
            use_phrase: bool = False,
            use_prefix: bool = False,
            multi_fts: bool = False,
            fts_multi_weights: dict = None,
            **metadata_filters
    ): 
        """
        Perform a hybrid search combining FTS and vector results.

        Supports:
            - Single-mode FTS (keyword, phrase, or prefix)
            - Multi-mode FTS (keyword + phrase + prefix with weighted score fusion)
            - Vector similarity search or max marginal relevance (MMR) search
            - Deduplication and hybrid scoring

        Args:
            query (str): Search query string.
            k (int): Maximum number of results to return.
            vector_search_method (str): "similarity" or "mmr" for vector search.
            use_phrase (bool): Use exact phrase matching in FTS.
            use_prefix (bool): Use prefix search in FTS.
            multi_fts (bool): If True, perform multi-mode FTS search with score fusion.
            fts_multi_weights (dict, optional): Weights per FTS mode, e.g.,
                {"phrase": 1.0, "keyword": 1.0, "prefix": 1.0}.
            **metadata_filters: Optional key-value filters for FTS or vector results.

        Returns:
            list[SearchResult]: Ranked and deduplicated hybrid search results.
        """

        # --- FTS search ---
        if multi_fts:
            fts_multi_weights = fts_multi_weights or {"phrase": 1.0, "keyword": 1.0, "prefix": 1.0}

            fts_results = self.fts.search_multi(
                query_keywords=[query],
                query_phrases=[query] if use_phrase else [],
                query_prefixes=[query] if use_prefix else [],
                k=k,
                metadata_filters=metadata_filters,
                fts_multi_weights=fts_multi_weights
            )
        
        else:
            fts_results = self.fts.search_single(
                query, 
                k=k, 
                use_phrase=use_phrase,
                use_prefix=use_prefix,
                metadata_filters=metadata_filters
                )

        # --- Vector search ---
        if vector_search_method == "similarity":
            vector_result_objects = self.vector.similarity_search_with_relevance_scores(
                query, k=k, 
                filter=metadata_filters if metadata_filters else None
            )
            backend_label = "vector_similarity"
       
        elif vector_search_method == "mmr":
            vector_result_mmr = self.vector.max_marginal_relevance_search(
                query, 
                k=k, 
                filter=metadata_filters if metadata_filters else None
            )

            # Assumes the first result in the list is the most important and the last the least.
            # Since "mmr" search gives no raw score, assign a synthetic score based on its rank.
            vector_result_objects = []
            for i, result in enumerate(vector_result_mmr):
                # Synthetic score based on rank: Best rank (i=0) gets highest score.
                score = (1 - (i / k)) 
                vector_result_objects.append((result, score))

            backend_label = "vector_mmr"
        
        else:
            raise ValueError(f"Unknown vector search method: {vector_search_method}")

        # Convert vector results to SearchResult format.  Every typed
        # ChunkMetadata field is pulled from obj.metadata so vector hits
        # carry the same typed-field surface as FTS hits.
        vector_results = []
        for obj, score in vector_result_objects:
            vector_results.append(
                SearchResult(
                    page_content = obj.page_content,
                    source = obj.metadata.get("source"),
                    doc_hash_id = obj.metadata.get("doc_hash_id"),
                    filename = obj.metadata.get("filename"),
                    file_type = obj.metadata.get("file_type"),
                    folder = obj.metadata.get("folder"),
                    doc_word_count = obj.metadata.get("doc_word_count"),
                    doc_char_count = obj.metadata.get("doc_char_count"),
                    ingested_at = obj.metadata.get("ingested_at"),
                    category = obj.metadata.get("category"),
                    language = obj.metadata.get("language"),
                    chunk_id = obj.metadata.get("chunk_id"),
                    chunk_hash_id = obj.metadata.get("chunk_hash_id"),
                    chunk_start_char = obj.metadata.get("chunk_start_char"),
                    chunk_end_char = obj.metadata.get("chunk_end_char"),
                    metadata = obj.metadata,
                    score = score,
                    backend = backend_label
                )
            )

         # --- Combine and deduplicate results ---
        combined_results = fts_results + vector_results
        unique_dict = {}   
        
        for item in combined_results:
            item_key = item.chunk_hash_id
        
            if item_key not in unique_dict:
                unique_dict[item_key] = item  
            else:
                existing = unique_dict[item_key] 

                if self._hybrid_score(item) > self._hybrid_score(existing):    
                    unique_dict[item_key] = item    

        # Rank higher scores first
        unique_list = list(unique_dict.values()) 
        ranked = sorted(unique_list, key=self._hybrid_score, reverse=True) 
        
        # --- Final ranking ---
        return ranked[:k]
