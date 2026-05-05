"""Unit tests for eval_utils pure helpers.

Covers the non-metric plumbing used elsewhere in the harness: the
shared normalize_text text comparator, safe_mean math helper,
LangChain message/content extractors, make_prompt constraint
appending, the three DeepEval context builders
(build_expected_output, build_gold_context,
build_retrieval_context), and the JSON-report retrieval preview
formatter.  These helpers have no LLM side effects, so the suite is
fast and deterministic.
"""

from __future__ import annotations

import json

from langchain_core.messages import HumanMessage, ToolMessage

from eval_models import EvalCase
from eval_utils import (
    build_expected_output,
    build_gold_context,
    build_retrieval_context,
    extract_agent_retrieval_results,
    extract_message_text,
    make_prompt,
    normalize_text,
    preview_results,
    safe_mean,
)


def _case(**overrides) -> EvalCase:
    """Build an EvalCase with sensible defaults; override any field with kwargs."""
    defaults = dict(id="t", question="What is the policy?")
    defaults.update(overrides)
    return EvalCase(**defaults)


# ---------------------------------------------------------------------------
# normalize_text
# ---------------------------------------------------------------------------

class TestNormalizeText:
    def test_lowercases(self):
        assert normalize_text("Hello WORLD") == "hello world"

    def test_strips_accents(self):
        assert normalize_text("café") == "cafe"
        assert normalize_text("naïve") == "naive"

    def test_collapses_punctuation_to_space(self):
        assert normalize_text("foo, bar; baz!") == "foo bar baz"

    def test_collapses_whitespace(self):
        assert normalize_text("foo   bar\n\nbaz") == "foo bar baz"

    def test_preserves_digits_and_percent(self):
        assert normalize_text("Score: 7.5%") == "score 7.5%"

    def test_empty_input(self):
        assert normalize_text("") == ""


# ---------------------------------------------------------------------------
# safe_mean
# ---------------------------------------------------------------------------

class TestSafeMean:
    def test_basic_mean(self):
        assert safe_mean([1.0, 2.0, 3.0]) == 2.0

    def test_skips_none(self):
        assert safe_mean([1.0, None, 3.0]) == 2.0

    def test_all_none_returns_zero(self):
        # The HSA implementation returns 0.0 for empty/all-None — distinct
        # from the CI agent which returns None. This test pins the contract.
        assert safe_mean([None, None]) == 0.0

    def test_empty_list_returns_zero(self):
        assert safe_mean([]) == 0.0

    def test_decimals_argument(self):
        # 6/7 = 0.857142… — decimals controls the rounding.
        assert safe_mean([1, 1, 1, 1, 1, 1, 0], decimals=2) == 0.86
        assert safe_mean([1, 1, 1, 1, 1, 1, 0], decimals=4) == 0.8571

    def test_handles_int_and_float_mix(self):
        assert safe_mean([1, 2.0, 3]) == 2.0


# ---------------------------------------------------------------------------
# extract_message_text
# ---------------------------------------------------------------------------

class TestExtractMessageText:
    def test_string_content(self):
        assert extract_message_text("hello") == "hello"

    def test_strips_whitespace(self):
        assert extract_message_text("  hello  ") == "hello"

    def test_block_list_with_text(self):
        content = [{"text": "first"}, {"text": "second"}]
        assert extract_message_text(content) == "first\nsecond"

    def test_block_list_with_content_key(self):
        # Some providers use "content" instead of "text".
        assert extract_message_text([{"content": "hi"}]) == "hi"

    def test_block_list_with_strings(self):
        assert extract_message_text(["a", "b"]) == "a\nb"

    def test_mixed_blocks_skip_unparseable(self):
        # Blocks without text/content keys are silently dropped.
        content = [{"text": "kept"}, {"image": "ignored"}]
        assert extract_message_text(content) == "kept"

    def test_empty_string(self):
        assert extract_message_text("") == ""

    def test_empty_list(self):
        assert extract_message_text([]) == ""

    def test_falls_back_to_str(self):
        assert extract_message_text(42) == "42"


# ---------------------------------------------------------------------------
# extract_agent_retrieval_results
# ---------------------------------------------------------------------------

class TestExtractAgentRetrievalResults:
    def test_collects_results_from_tool_messages(self):
        payload = {"results": [{"source": "a.pdf", "page_content": "x"}]}
        messages = [
            HumanMessage(content="ignored"),
            ToolMessage(content=json.dumps(payload), tool_call_id="t1"),
        ]
        out = extract_agent_retrieval_results(messages)
        assert out == [{"source": "a.pdf", "page_content": "x"}]

    def test_concatenates_across_multiple_tool_messages(self):
        messages = [
            ToolMessage(content=json.dumps({"results": [{"source": "a.pdf"}]}), tool_call_id="t1"),
            ToolMessage(content=json.dumps({"results": [{"source": "b.pdf"}]}), tool_call_id="t2"),
        ]
        out = extract_agent_retrieval_results(messages)
        assert out == [{"source": "a.pdf"}, {"source": "b.pdf"}]

    def test_skips_non_tool_messages(self):
        # Only ToolMessage instances should contribute results.
        messages = [HumanMessage(content=json.dumps({"results": [{"source": "x.pdf"}]}))]
        assert extract_agent_retrieval_results(messages) == []

    def test_skips_invalid_json(self):
        # Malformed payloads must not crash the extractor.
        messages = [ToolMessage(content="not json at all", tool_call_id="t1")]
        assert extract_agent_retrieval_results(messages) == []

    def test_skips_payload_without_results_key(self):
        messages = [ToolMessage(content=json.dumps({"other": "stuff"}), tool_call_id="t1")]
        assert extract_agent_retrieval_results(messages) == []

    def test_skips_empty_content(self):
        messages = [ToolMessage(content="", tool_call_id="t1")]
        assert extract_agent_retrieval_results(messages) == []


# ---------------------------------------------------------------------------
# make_prompt
# ---------------------------------------------------------------------------

class TestMakePrompt:
    def test_question_only_when_no_extras(self):
        case = _case(question="What is the policy?", answer_style="", metadata_filters={})
        assert make_prompt(case) == "What is the policy?"

    def test_appends_answer_style(self):
        case = _case(answer_style="Be concise.", metadata_filters={})
        prompt = make_prompt(case)
        assert "What is the policy?" in prompt
        assert "Be concise." in prompt
        assert "Evaluation constraints:" in prompt

    def test_appends_metadata_filters(self):
        case = _case(answer_style="", metadata_filters={"category": "policy"})
        prompt = make_prompt(case)
        assert "category=policy" in prompt
        assert "Evaluation constraints:" in prompt

    def test_combines_style_and_filters(self):
        case = _case(answer_style="Be brief.", metadata_filters={"category": "policy"})
        prompt = make_prompt(case)
        assert "Be brief." in prompt
        assert "category=policy" in prompt

    def test_strips_question_whitespace(self):
        case = _case(question="  spaced  ", answer_style="", metadata_filters={})
        assert make_prompt(case) == "spaced"


# ---------------------------------------------------------------------------
# DeepEval context builders
# ---------------------------------------------------------------------------

class TestBuildExpectedOutput:
    def test_joins_answer_points_with_space(self):
        case = _case(expected_answer_points=["Point A.", "Point B."])
        assert build_expected_output(case) == "Point A. Point B."

    def test_fallback_when_no_points(self):
        case = _case()
        assert build_expected_output(case) == "Provide a corpus-grounded answer only."


class TestBuildGoldContext:
    def test_returns_answer_points_when_present(self):
        case = _case(expected_answer_points=["A", "B"])
        assert build_gold_context(case) == ["A", "B"]

    def test_falls_back_to_expected_output_plus_sources(self):
        case = _case(expected_sources=["a.pdf", "b.pdf"])
        out = build_gold_context(case)
        assert out[0] == "Provide a corpus-grounded answer only."
        assert "Relevant source: a.pdf" in out
        assert "Relevant source: b.pdf" in out

    def test_falls_back_with_no_sources(self):
        # Just the expected_output sentence; no source lines appended.
        assert build_gold_context(_case()) == ["Provide a corpus-grounded answer only."]


class TestBuildRetrievalContext:
    def test_empty_results_returns_placeholder(self):
        # Empty input should return a non-empty list so DeepEval doesn't error.
        assert build_retrieval_context([]) == ["No context retrieved."]

    def test_formats_each_result(self):
        results = [{"source": "docs/a.pdf", "chunk_id": "c1", "backend": "fts", "score": 0.9, "page_content": "hello world"}]
        out = build_retrieval_context(results)
        assert len(out) == 1
        # Source path is reduced to filename, all metadata fields are present.
        assert "source=a.pdf" in out[0]
        assert "chunk_id=c1" in out[0]
        assert "backend=fts" in out[0]
        assert "score=0.9" in out[0]
        assert "hello world" in out[0]

    def test_normalizes_newlines_in_page_content(self):
        # Newlines in page_content are flattened to spaces so the formatted
        # context stays one entry per result.
        results = [{"source": "a.pdf", "chunk_id": "c1", "backend": "fts", "score": 1.0, "page_content": "line1\nline2"}]
        out = build_retrieval_context(results)
        assert "line1 line2" in out[0]
        assert "\n" not in out[0]

    def test_handles_missing_fields(self):
        # Defaults kick in for missing source / chunk_id / backend.
        out = build_retrieval_context([{"page_content": "x"}])
        assert "source=unknown" in out[0]
        assert "chunk_id=?" in out[0]
        assert "backend=unknown" in out[0]


# ---------------------------------------------------------------------------
# preview_results
# ---------------------------------------------------------------------------

class TestPreviewResults:
    def test_caps_at_default_limit(self):
        results = [{"source": f"a{i}.pdf", "page_content": "x"} for i in range(10)]
        assert len(preview_results(results)) == 5

    def test_respects_explicit_limit(self):
        results = [{"source": f"a{i}.pdf", "page_content": "x"} for i in range(10)]
        assert len(preview_results(results, limit=2)) == 2

    def test_truncates_long_snippets(self):
        # page_content is trimmed to 300 chars in the snippet field.
        long_text = "a" * 500
        out = preview_results([{"source": "a.pdf", "page_content": long_text}])
        assert len(out[0]["snippet"]) == 300

    def test_strips_source_path_to_filename(self):
        out = preview_results([{"source": "docs/folder/A.pdf", "page_content": "x"}])
        assert out[0]["source"] == "A.pdf"

    def test_handles_missing_source(self):
        out = preview_results([{"page_content": "x"}])
        assert out[0]["source"] == "unknown"

    def test_empty_input(self):
        assert preview_results([]) == []
