"""Pydantic data models that validate the evaluation dataset.

EvalCase is one row in eval_cases.json — a question plus the
ground-truth signals the harness scores against (expected sources,
chunk snippets, keyword lists, metadata filters).  RetrievalSettings
captures the per-case search hyperparameters that the retriever expects.
Validation runs at load time in eval_utils.load_cases so a malformed
dataset fails fast with a Pydantic error instead of crashing mid-run.
"""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field


class RetrievalSettings(BaseModel):
    """Search hyperparameters for a single eval case."""

    k: int = Field(5, ge=1, le=10)
    vector_search_method: Literal["similarity", "mmr"] = "similarity"
    use_phrase: bool = False
    use_prefix: bool = False
    multi_fts: bool = True


class EvalCase(BaseModel):
    """A single evaluation test case with its question, expected outputs, and retrieval config."""

    id: str
    question: str
    expected_answer_points: list[str] = Field(default_factory=list)
    expected_sources: list[str] = Field(default_factory=list)
    expected_chunks: list[str] = Field(
        default_factory=list,
        description=(
            "Representative text snippets for chunk-level retrieval metrics. "
            "A retrieved chunk is considered relevant when any of these snippets "
            "appears as a substring of the chunk's page_content (after "
            "normalization). Snippets are preferred over chunk IDs because "
            "chunk IDs change whenever the chunking strategy is tuned, while "
            "short representative text remains stable."
        ),
    )
    required_keywords: list[str] = Field(default_factory=list)
    disallowed_keywords: list[str] = Field(default_factory=list)
    metadata_filters: dict[str, Any] = Field(default_factory=dict)
    answer_style: str = "Answer concisely and include a short Evidence section."
    retrieval: RetrievalSettings = Field(default_factory=RetrievalSettings)
    category: str = ""
    notes: str = ""
