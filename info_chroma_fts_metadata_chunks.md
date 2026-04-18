# Document Indexing Pipeline — `chroma_fts_metadata_chunks.ipynb`

This notebook builds the two retrieval indexes used by the HybridSearchAgent:

- **Chroma vector store** — stores OpenAI-embedded document chunks for semantic (vector) search.
- **SQLite FTS5 index** — stores raw chunk text for full-text search (keyword, phrase, and prefix) via BM25.

Both indexes must be populated before the agent or dashboard can retrieve results. Run the notebook cells top-to-bottom on first use; re-run individual sections to refresh either index after document changes.

## Pipeline Overview

```
documents/                   (.txt, .pdf, .docx source files)
    ↓
1. Load documents            (DirectoryLoader — TextLoader, PyPDFLoader, Docx2txtLoader)
    ↓
2. Build DataFrame           (Pandas — inspect content, enrich metadata)
    ↓
3. Save CSV snapshot         (documents_with_metadata_df.csv — optional, for reproducibility)
    ↓
4. Convert back to Documents (LangChain Document objects with enriched metadata)
    ↓
5. Split into chunks         (RecursiveCharacterTextSplitter — configurable size and overlap)
    ↓
6. Add chunk metadata        (chunk_id, chunk_hash_id, start_char, end_char)
    ↓
7. Standardize metadata      (Pydantic validation via ChunkMetadata model)
    ↓
8. Ingest into Chroma        (OpenAI embeddings → chroma_db/)
    ↓
9. Ingest into SQLite FTS    (raw text + metadata → fts.db)
    ↓
10. Verify and visualize     (inspection cells, PCA scatter plot)
```

## Detailed Walkthrough

### 1. Load Documents

Uses LangChain's `DirectoryLoader` with format-specific loaders to read all files from the `documents/` folder:

| Format | Loader | Extension |
|--------|--------|-----------|
| Plain text | `TextLoader` | `.txt` |
| PDF | `PyPDFLoader` | `.pdf` |
| Word | `Docx2txtLoader` | `.docx` |

All loaded documents are merged into a single list. Each document has a `page_content` (the text) and a `metadata` dict (initially just the `source` path).

### 2. Build Pandas DataFrame and Enrich Metadata

The raw documents are converted into a Pandas DataFrame for inspection and metadata enrichment. The notebook includes two approaches for trimming the DataFrame columns:

- **Drop approach** — remove specific columns you don't need.
- **Keep approach** — keep only `text` and `source`, then add fields selectively.

Either way, `text` and `source` must be retained.

The notebook provides cells for adding the following fields:

| Field | How it is derived | Required? |
|-------|-------------------|-----------|
| `doc_hash_id` | MD5 hash of the document text — deterministic, content-based ID | Yes |
| `filename` | Extracted from the `source` path (`os.path.basename`) | Optional |
| `file_type` | File extension extracted from `filename` | Optional |
| `folder` | Directory extracted from the `source` path | Optional |
| `doc_char_count` | Character length of the document text | Optional |
| `doc_word_count` | Word count (whitespace split) | Optional |
| `ingested_at` | UTC Unix timestamp at ingestion time | Yes |
| `category` | Rule-based classification (currently checks for "norway" in text) | Optional |
| `language` | Detected via `langdetect` | Optional |
| `summary` | Snippet-based summary (placeholder — LLM summarization is a future option) | Optional |
| `content_type` | Heuristic classification into "table", "header", or "text" | Optional |

### 3. Save CSV Snapshot (optional)

The enriched DataFrame can be saved to `documents_with_metadata_df.csv` for reproducibility and offline inspection. This is optional, the pipeline does not read from the CSV, it continues from the in-memory DataFrame.

### 4. Convert Back to LangChain Documents

The DataFrame rows are converted back into LangChain `Document` objects, with all enriched metadata fields carried over. The `text` column becomes `page_content` and all other columns become metadata.

A post-conversion cell is provided for any final metadata tweaks (e.g., overriding `category` for all documents).

### 5. Split into Chunks

Documents are split into overlapping chunks using `RecursiveCharacterTextSplitter`:

| Parameter | Purpose |
|-----------|---------|
| `chunk_size` | Maximum characters per chunk |
| `chunk_overlap` | Overlap between adjacent chunks for context continuity |

The splitter preserves all document-level metadata on each chunk. Chunk boundaries are chosen at natural break points (paragraphs, sentences, words) using the recursive splitting strategy.

The current values are tuned for the short, factual Norway-themed documents in this project. These are not universal defaults. Optimal chunk size and overlap depend on the dataset. Shorter chunks improve precision for factual lookups but lose surrounding context; longer chunks preserve context but may dilute relevance scores. Common starting points are 500–1000 characters with 50–200 overlap, adjusted based on retrieval evaluation results.

### 6. Add Chunk-Level Metadata

Three positional metadata fields are added to each chunk after splitting:

| Field | Value | Purpose |
|-------|-------|---------|
| `chunk_id` | Sequential integer (0, 1, 2, ...) | Position in the chunk list |
| `chunk_hash_id` | MD5 of `{doc_hash_id}_{chunk_id}` | Deterministic unique ID per chunk — used as the Chroma document ID for idempotent upserts |
| `chunk_start_char` / `chunk_end_char` | Character offsets | Position of the chunk within the concatenated document text |

### 7. Standardize Metadata with Pydantic

Each chunk's metadata is validated and type-coerced through the `ChunkMetadata` Pydantic model (defined in `pydantic_models.py`). This ensures consistent types across all chunks. For example, a `chunk_id` stored as the string `"3"` is converted to the integer `3`.

Any metadata fields not defined in the Pydantic schema are preserved as extra fields and merged back into the chunk metadata.

### 8. Ingest into Chroma (Vector Store)

The notebook provides two ingestion approaches:

**Simple insert**: For first-time setup when no Chroma database exists:
```python
vector_store.add_documents(clean_chunks)
```

**Upsert with source-level refresh**: For updating an existing database after document changes:
1. Identifies all unique `source` values in the current batch.
2. If `REFRESH_EXISTING_SOURCES = True`: Deletes all existing chunks for those sources first (prevents "ghost chunks" if the new file version has fewer chunks than the old one).
3. Upserts the new chunks using `chunk_hash_id` as the document ID — this ensures deterministic, idempotent inserts.

Embeddings are generated via `OpenAIEmbeddings` and stored alongside the chunk text and metadata in `chroma_db/`.

### 9. Ingest into SQLite FTS (Full-Text Search)

The notebook provides two ingestion approaches, mirroring the Chroma logic:

**Simple insert**: For first-time setup:
```python
fts_store = FTSStore()
fts_store.add_documents(clean_chunks)
```

**Update with source-level refresh**: For updating an existing index:
- If `REFRESH_EXISTING_SOURCES = True`: Deletes all existing rows for each source, then re-indexes the new chunks.
- If `REFRESH_EXISTING_SOURCES = False`: Append-only, checks which sources are already indexed and only inserts chunks from new files.

The FTS index is stored in `fts.db` and queried by the `FTSStore` class in `fts_search.py`.

### 10. Verify and Visualize

Several inspection cells are provided for verifying the indexed data:

- **Chroma inspection** — print all stored documents, metadata, and embedding vectors.
- **Chunk metadata review** — iterate over chunks and print metadata fields.
- **Test retrieval** — run a similarity search query against Chroma to verify results.
- **PCA visualization** — reduce the embedding space to 2 dimensions and plot a scatter chart to see how document chunks cluster.

## Adapting for a Different Corpus

To use this pipeline with your own documents:

1. **Replace the files** in `documents/` with your own `.txt`, `.pdf`, or `.docx` files.
2. **Update the category classifier** — the `classify_doc()` function currently checks for "norway" in the text. Replace this with logic appropriate to your domain, or remove it entirely.
3. **Review the metadata fields** — add or remove enrichment cells based on what metadata is relevant for your use case. If you add new fields, update the `ChunkMetadata` model in `pydantic_models.py` and the FTS schema in `fts_search.py` to keep everything in sync.
4. **Adjust chunking parameters** — adjust `chunk_size` and `chunk_overlap` to fit your dataset.
5. **Re-run the notebook** top-to-bottom to rebuild both indexes.
6. **Delete old indexes** if you want a clean start — remove `fts.db` and the `chroma_db/` directory before re-running.

## Dependencies

The notebook uses packages from all three dependency groups in `requirements.txt`:

- **Ingestion**: `langchain-community`, `langchain-text-splitters`, `langdetect`, `docx2txt`, `pypdf`
- **Embeddings**: `langchain-openai`, `langchain-chroma`, `chromadb`
- **Visualization**: `numpy`, `matplotlib`, `scikit-learn` (PCA), `pandas`

The `OPENAI_API_KEY` environment variable is required for the embedding step (Chroma ingestion).
