"""
Unit tests for the 4 evaluation pillars.

Run with coverage:
  pytest tests/ -v \
    --cov=solution --cov=_chunks --cov=_preprocess \
    --cov-branch --cov-report=term-missing

Pillars:
  1. Retrieval      — chunking, tokenisation, BM25, boosts, adjacent-chunk logic
  2. Temporal       — snapshot date selection (_chunks_as_of)
  3. Conflict       — temporal + inline conflict detection
  4. Answer quality — confident flag, URL aliases, #rates append (LLM mocked)
"""
from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

# ---------------------------------------------------------------------------
# Path setup — harness must be importable before solution imports contract
# ---------------------------------------------------------------------------
_REPO = Path(__file__).resolve().parents[1]
_HARNESS = _REPO.parent / "hackathon-knowledge-kit-participant" / "harness"
if _HARNESS.exists() and str(_HARNESS) not in sys.path:
    sys.path.insert(0, str(_HARNESS))

from contract import Answer, Conflict, Page, Snapshot, Source  # noqa: E402

import _chunks as ch               # noqa: E402
import _preprocess as pp           # noqa: E402
from _chunks import (              # noqa: E402
    _Chunk, _build_context, _chunk_para, _detect_inline_conflict,
    _expand_tokens, _join_short_paras, _match_conflicts,
    _merge_schedule_blocks, _split_para, _tokenize,
)
from _preprocess import (          # noqa: E402
    _expand_charter_area, _expand_model_year_ranges, _expand_month_abbrevs,
    _normalize_loan_query, _tag_recreational_rows, _url_authority,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _chunk(url: str, text: str, snapshot_id: str = "s1",
           taken_on: str = "2026-01-01", title: str = "") -> _Chunk:
    return _Chunk(url=url, title=title, snapshot_id=snapshot_id,
                  taken_on=taken_on, text=text)


def _mock_llm_resp(text: str):
    block = MagicMock()
    block.type = "text"
    block.text = text
    resp = MagicMock()
    resp.content = [block]
    return resp


def _make_snapshot(sid: str, taken_on: str, url: str, markdown: str) -> Snapshot:
    page = Page(url=url, title="Test Page", markdown=markdown,
                fetched_at=f"{taken_on}T00:00:00Z")
    return Snapshot(snapshot_id=sid, taken_on=taken_on, pages=[page])


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def ks():
    with patch("solution.anthropic.Anthropic"):
        from solution import KnowledgeSystem
        return KnowledgeSystem()


@pytest.fixture
def ks_with_data(ks):
    snap = _make_snapshot(
        "s1", "2026-06-01",
        "https://www.dupaco.com/personal/savings-account#rates",
        "Share Savings Account\n\n| Tier | APY |\n|---|---|\n| $1,000+ | 0.35% |",
    )
    ks.ingest(snap)
    return ks


# ===========================================================================
# PILLAR 1 — Retrieval
# ===========================================================================

class TestTokenize:
    def test_basic_split(self):
        assert _tokenize("Hello World") == ["hello", "world"]

    def test_lowercases(self):
        assert _tokenize("APY") == ["apy"]

    def test_keeps_numbers(self):
        tokens = _tokenize("rate 5%")
        assert any("5" in t for t in tokens)

    def test_keeps_percent(self):
        tokens = _tokenize("3.25%")
        assert any("%" in t for t in tokens)

    def test_empty_string(self):
        assert _tokenize("") == []


class TestExpandTokens:
    def test_adds_synonyms_for_rates(self):
        result = _expand_tokens(["rates"])
        assert "rate" in result
        assert "apy" in result
        assert "dividend" in result

    def test_no_duplicate_if_already_present(self):
        result = _expand_tokens(["rates", "rate"])
        assert result.count("rate") == 1

    def test_passthrough_unknown_token(self):
        result = _expand_tokens(["dupaco"])
        assert result == ["dupaco"]

    def test_mortgage_expansion(self):
        result = _expand_tokens(["mortgage"])
        assert "home" in result
        assert "loan" in result


class TestMergeScheduleBlocks:
    def test_merges_consecutive_schedule_lines(self):
        paras = [
            "Lobby Hours",
            "Monday 8:00 a.m. – 5:00 p.m.",
            "Saturday noon – 1:00 p.m.",
        ]
        result = _merge_schedule_blocks(paras)
        assert len(result) == 1
        assert "Monday" in result[0]
        assert "Saturday" in result[0]

    def test_attaches_schedule_to_preceding_header(self):
        paras = [
            "Branch Hours",
            "Monday 9:00 a.m. – 5:00 p.m.",
        ]
        result = _merge_schedule_blocks(paras)
        assert len(result) == 1
        assert "Branch Hours" in result[0]
        assert "Monday" in result[0]

    def test_non_schedule_passthrough(self):
        paras = ["Regular paragraph.", "Another paragraph."]
        result = _merge_schedule_blocks(paras)
        assert result == paras

    def test_mixed_schedule_and_text(self):
        paras = [
            "Monday 9:00 a.m. closed",
            "Contact us for more information.",
        ]
        result = _merge_schedule_blocks(paras)
        assert len(result) == 2


class TestJoinShortParas:
    def test_merges_short_into_next(self):
        paras = ["Rates", "0.35% APY on Share Savings accounts at Dupaco."]
        result = _join_short_paras(paras)
        assert len(result) == 1
        assert "Rates" in result[0]

    def test_keeps_long_para_separate(self):
        long = "A" * 100
        paras = [long, "another paragraph"]
        result = _join_short_paras(paras)
        assert len(result) == 2

    def test_last_short_para_not_lost(self):
        paras = ["Short"]
        result = _join_short_paras(paras)
        assert result == ["Short"]


class TestChunkPara:
    def test_table_produces_header_plus_row(self):
        table = "| Term | APY |\n|---|---|\n| 12 mo | 1.00% |\n| 24 mo | 1.25% |"
        chunks = _chunk_para(table, 700)
        assert len(chunks) == 2
        for c in chunks:
            assert "Term" in c   # header repeated

    def test_table_single_data_row(self):
        table = "| Term | APY |\n|---|---|\n| 12 mo | 1.00% |"
        chunks = _chunk_para(table, 700)
        assert len(chunks) == 1
        assert "12 mo" in chunks[0]

    def test_short_non_table_passthrough(self):
        text = "Dupaco was founded in 1948."
        chunks = _chunk_para(text, 700)
        assert chunks == [text]

    def test_long_non_table_splits(self):
        sentences = ["This is sentence number %d." % i for i in range(30)]
        text = " ".join(sentences)
        chunks = _chunk_para(text, 100)
        assert len(chunks) > 1


class TestSplitPara:
    def test_short_text_no_split(self):
        text = "Short sentence."
        assert _split_para(text, 700) == [text]

    def test_splits_at_sentence_boundary(self):
        text = "This is the first long sentence here. This is the second long sentence there."
        chunks = _split_para(text, 40)
        assert len(chunks) >= 2
        assert all(len(c) >= ch._MIN_CHUNK for c in chunks)

    def test_respects_max_len(self):
        sentences = ["Word " * 10 + "." for _ in range(10)]
        text = " ".join(sentences)
        chunks = _split_para(text, 60)
        assert all(len(c) <= 60 + 20 for c in chunks)  # slight overshoot on sentence boundary OK


class TestBuildContext:
    def test_includes_url_and_snapshot(self):
        c = _chunk("https://www.dupaco.com/personal/savings-account#rates",
                   "0.35% APY")
        result = _build_context([c])
        assert "dupaco.com" in result
        assert "2026-01-01" in result
        assert "0.35% APY" in result

    def test_multiple_chunks_separated_by_divider(self):
        c1 = _chunk("https://www.dupaco.com/personal/savings-account#rates", "text one")
        c2 = _chunk("https://www.dupaco.com/about", "text two")
        result = _build_context([c1, c2])
        assert "---" in result


class TestRetrieveBM25:
    def test_relevant_chunk_ranked_first(self, ks_with_data):
        result = ks_with_data._retrieve_bm25(
            "What is the Share Savings APY?",
            ks_with_data._chunks_as_of(None),
        )
        assert len(result) > 0
        assert any("APY" in c.text or "apy" in c.text.lower() or "0.35" in c.text
                   for c in result)

    def test_hsa_filtered_for_generic_savings_query(self, ks):
        pool = [
            _chunk("https://www.dupaco.com/personal/savings-account#rates",
                   "Share Savings 0.25% APY"),
            _chunk("https://www.dupaco.com/personal/savings-account#rates",
                   "Health Savings Account (HSA) 0.10% APY"),
        ]
        result = ks._retrieve_bm25("What is the savings rate?", pool)
        texts = " ".join(c.text for c in result)
        assert "Health Savings" not in texts

    def test_hsa_included_when_query_asks_hsa(self, ks):
        pool = [
            _chunk("https://www.dupaco.com/personal/savings-account#rates",
                   "Health Savings Account (HSA) 0.10% APY"),
        ]
        result = ks._retrieve_bm25("What is the HSA rate?", pool)
        assert len(result) > 0


class TestApplyBoostsAndCap:
    def test_rate_query_boosts_pct_chunk(self, ks):
        pct_chunk = _chunk("https://www.dupaco.com/rates", "0.35% APY certificate")
        plain_chunk = _chunk("https://www.dupaco.com/about", "About Dupaco Credit Union")
        pool = [pct_chunk, plain_chunk]
        scores = [1.0, 1.0]
        result = ks._apply_boosts_and_cap("What is the certificate rate?", pool, scores)
        assert result[0].text == pct_chunk.text

    def test_per_url_cap_enforced(self, ks):
        pool = [
            _chunk("https://www.dupaco.com/rates", f"chunk {i}") for i in range(10)
        ]
        scores = [1.0] * 10
        from solution import _MAX_PER_URL
        result = ks._apply_boosts_and_cap("savings rate", pool, scores)
        url_count = sum(1 for c in result if c.url == "https://www.dupaco.com/rates")
        assert url_count <= _MAX_PER_URL

    def test_history_query_boosts_year_chunk(self, ks):
        year_chunk = _chunk("https://www.dupaco.com/about", "Founded in 1948 by employees.")
        plain_chunk = _chunk("https://www.dupaco.com/personal/savings-account#rates",
                             "Current savings rates available.")
        pool = [plain_chunk, year_chunk]
        scores = [1.0, 1.0]
        result = ks._apply_boosts_and_cap("When was Dupaco founded?", pool, scores)
        assert result[0].text == year_chunk.text

    def test_deduplicates_exact_text(self, ks):
        same_text = "0.35% APY"
        pool = [
            _chunk("https://www.dupaco.com/rates", same_text, snapshot_id="s1"),
            _chunk("https://www.dupaco.com/rates", same_text, snapshot_id="s2"),
        ]
        scores = [1.0, 1.0]
        result = ks._apply_boosts_and_cap("savings rate", pool, scores)
        assert len(result) == 1


class TestAddAdjacent:
    def test_adds_neighbors_for_top3(self, ks):
        pool = [
            _chunk("https://www.dupaco.com/rates", "chunk 0"),
            _chunk("https://www.dupaco.com/rates", "chunk 1"),
            _chunk("https://www.dupaco.com/rates", "chunk 2"),
        ]
        result_core = pool[:1]
        result = ks._add_adjacent(result_core, pool)
        assert len(result) > 1
        assert any(c.text == "chunk 1" for c in result)

    def test_no_cross_url_neighbors(self, ks):
        pool = [
            _chunk("https://www.dupaco.com/rates", "rates chunk"),
            _chunk("https://www.dupaco.com/about", "about chunk"),
        ]
        result = ks._add_adjacent(pool[:1], pool)
        texts = [c.text for c in result]
        assert "about chunk" not in texts

    def test_branch_url_uses_depth2(self, ks):
        pool = [
            _chunk("https://www.dupaco.com/locations/main", "chunk 0"),
            _chunk("https://www.dupaco.com/locations/main", "chunk 1"),
            _chunk("https://www.dupaco.com/locations/main", "chunk 2"),
        ]
        result = ks._add_adjacent(pool[:1], pool)
        assert any(c.text == "chunk 2" for c in result)


# ===========================================================================
# PILLAR 2 — Temporal filtering
# ===========================================================================

class TestChunksAsOf:
    def test_none_returns_latest_snapshot(self, ks):
        ks.ingest(_make_snapshot("s1", "2026-01-01",
                                 "https://www.dupaco.com/rates", "old rate 1.00%"))
        ks.ingest(_make_snapshot("s2", "2026-06-01",
                                 "https://www.dupaco.com/rates", "new rate 1.25%"))
        pool = ks._chunks_as_of(None)
        assert all(c.snapshot_id == "s2" for c in pool)

    def test_date_filter_returns_eligible_snapshot(self, ks):
        ks.ingest(_make_snapshot("s1", "2026-01-01",
                                 "https://www.dupaco.com/rates", "jan rate 1.00%"))
        ks.ingest(_make_snapshot("s2", "2026-06-01",
                                 "https://www.dupaco.com/rates", "jun rate 1.25%"))
        pool = ks._chunks_as_of("2026-03-01")
        assert all(c.snapshot_id == "s1" for c in pool)

    def test_no_eligible_snapshots_returns_empty(self, ks):
        ks.ingest(_make_snapshot("s1", "2026-06-01",
                                 "https://www.dupaco.com/rates", "rate 1.00%"))
        pool = ks._chunks_as_of("2025-01-01")
        assert pool == []

    def test_exact_date_match_is_eligible(self, ks):
        ks.ingest(_make_snapshot("s1", "2026-06-01",
                                 "https://www.dupaco.com/rates",
                                 "Share Savings rate is currently 1.00% APY effective now"))
        pool = ks._chunks_as_of("2026-06-01")
        assert len(pool) > 0

    def test_empty_system_returns_empty(self, ks):
        assert ks._chunks_as_of(None) == []
        assert ks._chunks_as_of("2026-01-01") == []


# ===========================================================================
# PILLAR 3 — Conflict detection
# ===========================================================================

class TestDetectTemporalConflicts:
    def test_creates_conflict_when_pct_changes(self, ks):
        ks.ingest(_make_snapshot("s1", "2026-01-01",
                                 "https://www.dupaco.com/rates", "Certificate rate: 1.00%"))
        ks.ingest(_make_snapshot("s2", "2026-06-01",
                                 "https://www.dupaco.com/rates", "Certificate rate: 1.25%"))
        conflicts = ks.conflicts()
        assert len(conflicts) > 0
        values = conflicts[0].values
        assert "1.00%" in values or "1.25%" in values

    def test_no_conflict_when_pct_unchanged(self, ks):
        md = "Certificate rate: 1.00%"
        ks.ingest(_make_snapshot("s1", "2026-01-01",
                                 "https://www.dupaco.com/rates", md))
        ks.ingest(_make_snapshot("s2", "2026-06-01",
                                 "https://www.dupaco.com/rates", md))
        assert ks.conflicts() == []

    def test_no_conflict_for_first_snapshot(self, ks):
        ks.ingest(_make_snapshot("s1", "2026-01-01",
                                 "https://www.dupaco.com/rates", "rate 1.00%"))
        assert ks.conflicts() == []

    def test_conflict_has_resolution_pointing_to_new_value(self, ks):
        ks.ingest(_make_snapshot("s1", "2026-01-01",
                                 "https://www.dupaco.com/rates", "rate 1.00%"))
        ks.ingest(_make_snapshot("s2", "2026-06-01",
                                 "https://www.dupaco.com/rates", "rate 1.25%"))
        c = ks.conflicts()[0]
        assert c.resolution == "1.25%"


class TestMatchConflicts:
    def _conflict(self, old: str, new: str) -> Conflict:
        return Conflict(
            topic="rate change",
            values=[old, new],
            sources=[],
            resolution=new,
            reason="newer snapshot",
        )

    def test_history_query_returns_all_conflicts(self):
        conflicts = [self._conflict("1.00%", "1.25%")]
        matched = _match_conflicts(conflicts, "What was the previous rate?")
        assert len(matched) == 1

    def test_non_history_no_match(self):
        conflicts = [self._conflict("1.00%", "1.25%")]
        matched = _match_conflicts(conflicts, "What is the current savings rate?")
        assert matched == []

    def test_value_in_question_matches(self):
        conflicts = [self._conflict("1.00%", "1.25%")]
        matched = _match_conflicts(conflicts, "Do you offer 1.00%?")
        assert len(matched) == 1

    def test_capped_at_four(self):
        conflicts = [self._conflict(f"{i}.00%", f"{i}.25%") for i in range(10)]
        matched = _match_conflicts(conflicts, "What changed before?")
        assert len(matched) <= 4


class TestDetectInlineConflict:
    def test_high_authority_vs_low_returns_note(self):
        official = _chunk(
            "https://www.dupaco.com/personal/savings-account#rates",
            "Share Savings: 0.35%",
        )
        blog = _chunk(
            "https://www.dupaco.com/newsroom/savings-tips",
            "Many savers earn 1.50% on their deposits.",
        )
        result = _detect_inline_conflict([official, blog], "savings rate")
        assert result is not None
        assert "Conflicting" in result

    def test_same_values_no_conflict(self):
        c1 = _chunk("https://www.dupaco.com/personal/savings-account#rates", "rate: 0.35%")
        c2 = _chunk("https://www.dupaco.com/rates", "rate: 0.35%")
        assert _detect_inline_conflict([c1, c2], "savings rate") is None

    def test_single_source_no_conflict(self):
        c = _chunk("https://www.dupaco.com/personal/savings-account#rates", "0.35%")
        assert _detect_inline_conflict([c], "savings rate") is None

    def test_no_pct_in_chunks_no_conflict(self):
        c1 = _chunk("https://www.dupaco.com/about", "Founded in 1948")
        c2 = _chunk("https://www.dupaco.com/newsroom/story", "Award for community service")
        assert _detect_inline_conflict([c1, c2], "history") is None


# ===========================================================================
# PILLAR 4 — Answer quality  (LLM mocked)
# ===========================================================================

class TestAnswerConfidentFlag:
    def test_dont_know_phrase_sets_not_confident(self, ks_with_data):
        ks_with_data._client.messages.create.return_value = _mock_llm_resp(
            "I don't know. Please contact Dupaco Credit Union directly."
        )
        result = ks_with_data.answer("What is the mortgage rate?")
        assert result.confident is False

    def test_direct_answer_sets_confident(self, ks_with_data):
        ks_with_data._client.messages.create.return_value = _mock_llm_resp(
            "5.25% APR for a 60-month auto loan."
        )
        result = ks_with_data.answer("What is the auto loan rate?")
        assert result.confident is True

    def test_please_contact_sets_not_confident(self, ks_with_data):
        ks_with_data._client.messages.create.return_value = _mock_llm_resp(
            "Please contact Dupaco for details."
        )
        result = ks_with_data.answer("What is the CD rate?")
        assert result.confident is False

    def test_not_available_sets_not_confident(self, ks_with_data):
        ks_with_data._client.messages.create.return_value = _mock_llm_resp(
            "That information is not available."
        )
        result = ks_with_data.answer("What is the HELOC rate?")
        assert result.confident is False


class TestAnswerURLAlias:
    def test_alias_url_substituted_to_canonical(self, ks_with_data):
        ks_with_data._client.messages.create.return_value = _mock_llm_resp(
            "https://www.dupaco.com/rates#rates"
        )
        result = ks_with_data.answer("Where are savings rates?")
        assert result.text == "https://www.dupaco.com/personal/savings-account#rates"

    def test_non_alias_url_unchanged(self, ks_with_data):
        url = "https://www.dupaco.com/personal/savings-account#rates"
        ks_with_data._client.messages.create.return_value = _mock_llm_resp(url)
        result = ks_with_data.answer("Where are savings rates?")
        assert result.text == url


class TestAnswerRatesHashAppend:
    def test_bare_product_url_gets_hash_rates_for_rate_question(self, ks_with_data):
        ks_with_data._client.messages.create.return_value = _mock_llm_resp(
            "https://www.dupaco.com/personal/savings-account"
        )
        result = ks_with_data.answer("What is the savings rate?")
        assert result.text.endswith("#rates")

    def test_already_has_fragment_no_double_append(self, ks_with_data):
        ks_with_data._client.messages.create.return_value = _mock_llm_resp(
            "https://www.dupaco.com/personal/savings-account#rates"
        )
        result = ks_with_data.answer("What is the savings rate?")
        assert result.text.count("#") == 1

    def test_non_rate_question_no_hash_append(self, ks_with_data):
        ks_with_data._client.messages.create.return_value = _mock_llm_resp(
            "https://www.dupaco.com/about"
        )
        result = ks_with_data.answer("Where is the main branch located?")
        assert "#rates" not in result.text


class TestAnswerCaching:
    def test_confident_answer_cached(self, ks_with_data):
        ks_with_data._client.messages.create.return_value = _mock_llm_resp(
            "0.35% APY"
        )
        ks_with_data.answer("What is the savings rate?")
        call_count_after_first = ks_with_data._client.messages.create.call_count

        ks_with_data.answer("What is the savings rate?")
        assert ks_with_data._client.messages.create.call_count == call_count_after_first

    def test_not_confident_answer_not_cached(self, ks_with_data):
        ks_with_data._client.messages.create.return_value = _mock_llm_resp(
            "I don't know. Please contact Dupaco Credit Union directly."
        )
        ks_with_data.answer("What is the HELOC rate?")
        first_count = ks_with_data._client.messages.create.call_count

        ks_with_data.answer("What is the HELOC rate?")
        assert ks_with_data._client.messages.create.call_count == first_count + 1


class TestAnswerNoPool:
    def test_empty_system_returns_idontknow(self, ks):
        result = ks.answer("What is the rate?")
        assert result.confident is False
        assert "don't know" in result.text.lower() or "no snapshot" in result.text.lower()


# ===========================================================================
# PILLAR 1b — Preprocess helpers
# ===========================================================================

class TestPreprocess:
    def test_normalize_loan_query_converts_years_to_months(self):
        assert "60 months" in _normalize_loan_query("5 year loan")
        assert "36 months" in _normalize_loan_query("3-year term")

    def test_url_authority_rates_page(self):
        assert _url_authority("https://www.dupaco.com/personal/savings-account#rates") == 5
        assert _url_authority("https://www.dupaco.com/rates") == 5

    def test_url_authority_blog_is_low(self):
        assert _url_authority("https://www.dupaco.com/newsroom/story") == 1

    def test_url_authority_calculator_is_mid(self):
        assert _url_authority("https://www.dupaco.com/calculator/auto") == 2

    def test_url_authority_product_page(self):
        assert _url_authority("https://www.dupaco.com/personal/checking") == 5

    def test_url_authority_about_page(self):
        assert _url_authority("https://www.dupaco.com/about/history") == 4

    def test_expand_model_year_ranges(self):
        result = _expand_model_year_ranges("2024-27 Model Year")
        assert "2025" in result
        assert "2026" in result
        assert "2027" in result

    def test_expand_charter_area_adds_states(self):
        result = _expand_charter_area("Members in our charter area may apply.")
        assert "Iowa" in result
        assert "Wisconsin" in result

    def test_expand_charter_area_noop_when_states_present(self):
        text = "Members in our charter area in Iowa and Wisconsin."
        assert _expand_charter_area(text) == text

    def test_expand_charter_area_noop_when_no_match(self):
        text = "No matching text here."
        assert _expand_charter_area(text) == text

    def test_expand_month_abbrevs(self):
        result = _expand_month_abbrevs("Branch closes Oct. 1, 2026.")
        assert "October" in result
        assert "Oct." not in result

    def test_tag_recreational_rows_appends_tag(self):
        text = "Model Year 2020-19 | 60 Months | 8.94%"
        result = _tag_recreational_rows(text)
        assert "snowmobile" in result
        assert "not auto" in result

    def test_tag_recreational_rows_noop_for_regular(self):
        text = "60 Months | 5.00% APR"
        assert _tag_recreational_rows(text) == text
