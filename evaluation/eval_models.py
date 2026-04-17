"""Data models for the DeepEval harness."""

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
    required_keywords: list[str] = Field(default_factory=list)
    disallowed_keywords: list[str] = Field(default_factory=list)
    metadata_filters: dict[str, Any] = Field(default_factory=dict)
    answer_style: str = "Answer concisely and include a short Evidence section."
    retrieval: RetrievalSettings = Field(default_factory=RetrievalSettings)
    category: str = ""
    notes: str = ""
