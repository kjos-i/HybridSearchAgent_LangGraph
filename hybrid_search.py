"""
Hybrid search retriever combining full-text and vector-based search.

This module provides the `HybridRetriever` class that:
    - Integrates keyword/phrase/prefix-based FTS results.
    - Integrates semantic vector search results (Chroma DB).
    - Computes hybrid scores for ranking results across both search types.
    - Returns deduplicated and ranked `SearchResult` objects.
"""

# Local imports
from pydantic_models import SearchResult
       

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
            fts_max_score: float
     ):
        """
        Initialize the HybridRetriever.
        """

        self.fts = fts_store
        self.vector = vector_store
        self.fts_weight = fts_weight
        self.vector_similarity_weight = vector_similarity_weight
        self.vector_mmr_weight = vector_mmr_weight
        self.fts_max_score = fts_max_score


    def _hybrid_score(self, doc: SearchResult) -> float:
        """
        Compute a unified hybrid score for ranking a single document.
        """
        
        # Ensure we have a numeric score to work with, defaulting to 0 if None
        raw_score = doc.score or 0  

        # Scores from FTS5 BM25 are negative, smaller is better
        if doc.backend == "fts":
            # Cap at 0 to avoid positive scores.
            norm_score = min(raw_score, 0)  
            # Normalize to [0, 1] range, 1 being best score.
            norm_score = abs(norm_score) / self.fts_max_score  
            return norm_score * self.fts_weight
        
        # Chroma DB similarity search with relevance scores: 1 best, 0 worst.
        elif doc.backend == "vector_similarity":
            norm_score = max(raw_score, 0)
            return norm_score * self.vector_similarity_weight

        # # Synthetic score based on rank: Best rank (i=0) gets highest score (score = (1 - (i / k))).
        elif doc.backend == "vector_mmr":
            norm_score = max(raw_score, 0)
            return norm_score * self.vector_mmr_weight

        # fallback for unknown backend
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

        # Convert vector results to SearchResult format
        vector_results = []
        for obj, score in vector_result_objects:       
            vector_results.append(
                SearchResult(
                    page_content = obj.page_content,
                    source = obj.metadata.get("source"),
                    category = obj.metadata.get("category"),
                    chunk_id = obj.metadata.get("chunk_id"),
                    start_char = obj.metadata.get("start_char"),
                    end_char = obj.metadata.get("end_char"),
                    metadata = obj.metadata,
                    score = score,
                    backend = backend_label
                )
            )

         # --- Combine and deduplicate results ---
        combined_results = fts_results + vector_results
        unique_dict = {}   
        
        for item in combined_results:
            item_key = item.page_content 
        
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
