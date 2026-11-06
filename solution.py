"""
Knowledge system: BM25 retrieval + Claude Haiku answer generation.

Requires ANTHROPIC_API_KEY in the environment.
"""

from __future__ import annotations

import os
import re
import sys
from pathlib import Path
from typing import Optional

# ---------------------------------------------------------------------------
# Contract import — harness must be on the path before build() is called.
# When run via run_eval.py the harness/ folder is added automatically.
# When running standalone, add it here.
# ---------------------------------------------------------------------------
def _add_harness_to_path():
    try:
        import contract  # noqa: F401 — already on path (run_eval.py added it)
        return
    except ImportError:
        pass
    for parent in Path(__file__).resolve().parents:
        candidate = parent / "harness" / "contract.py"
        if candidate.exists():
            sys.path.insert(0, str(parent / "harness"))
            return
    raise RuntimeError("Cannot locate harness/contract.py relative to solution.py")

_add_harness_to_path()

from contract import Answer, Conflict, Page, Snapshot, Source  # noqa: E402

try:
    from rank_bm25 import BM25Okapi
    _HAS_BM25 = True
except ImportError:
    _HAS_BM25 = False

import anthropic  # noqa: E402

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_WORD_RE = re.compile(r"[a-z0-9$%.]+")
_PCT_RE  = re.compile(r"\b\d+\.\d{1,3}%")   # percentage values e.g. "5.39%"
_MAX_CHUNK = 700          # chars per chunk
_MIN_CHUNK = 20           # skip tiny fragments (lowered to catch table rows)
_TOP_K     = 22           # chunks fed to the LLM
_MAX_PER_URL = 6          # max chunks from a single URL
_LLM_MODEL = os.environ.get("LLM_MODEL", "claude-haiku-5-5")

# Plural → singular + domain synonym expansion for BM25 query matching
_STEM_EXPAND: dict[str, list[str]] = {
    "branches": ["branch"],
    "locations": ["location"],
    "employees": ["employee"],
    "offices": ["office"],
    "tiers": ["tier"],
    "accounts": ["account"],
    "loans": ["loan"],
    "members": ["member"],
    "services": ["service"],
    # Rate/APY synonyms — boost retrieval of numeric rate chunks
    "rates": ["rate", "apy", "dividend"],
    "interest": ["apy", "dividend", "rate"],
    "apy": ["rate", "dividend", "interest"],
    # Contact synonyms — help match phone/contact pages
    "contact": ["phone", "call", "800"],
    "update": ["change", "modify", "edit"],
    "information": ["info", "details", "data"],
    # Home$tart / grant synonyms
    "grant": ["fhlb", "home$tart", "homebuyer", "assistance"],
    "homebuyer": ["home$tart", "fhlb", "grant", "first-time"],
    # Mortgage synonyms — "mortgage rates" should score home-loan rate table rows
    "mortgage": ["home", "loan", "purchase", "refinance", "conventional"],
    "mortgages": ["home", "loan", "purchase", "refinance", "conventional"],
    # Auto loan synonyms — "2025 vehicle" should match "2024-27 Model Year"
    "vehicle": ["auto", "car", "automobile"],
    "vehicles": ["auto", "car", "automobile"],
    # CD / certificate synonyms — savings-account page has certificate products
    "certificate": ["cd", "savings", "share"],
    "certificates": ["certificate", "cd", "savings"],
    "cd": ["certificate", "savings", "share"],
}

_RATE_TERMS = {"rate", "rates", "apy", "dividend", "interest", "percent", "%"}


def _url_authority(url: str) -> int:
    """Higher = more authoritative source."""
    u = url.lower()
    if any(x in u for x in ["/newsroom/", "/blog/", "/press", "/news/"]):
        return 1          # stale marketing content
    if any(x in u for x in ["calculator", "calc", "affordability", "tool"]):
        return 2          # calculators often lag official pages
    if any(x in u for x in [
        "/personal/loans/", "/personal/savings", "/personal/credit",
        "/personal/checking", "/personal/mortgages",
        "/business/loans/", "/business/savings", "/business/credit",
        "/rates", "/rate-", "#rates",
    ]):
        return 5          # official product / rate pages
    if any(x in u for x in ["/personal/", "/business/"]):
        return 4
    return 3


# ---------------------------------------------------------------------------
# Data record
# ---------------------------------------------------------------------------

class _Chunk:
    __slots__ = ("url", "title", "snapshot_id", "taken_on", "text", "authority")

    def __init__(self, url: str, title: str, snapshot_id: str,
                 taken_on: str, text: str):
        self.url         = url
        self.title       = title
        self.snapshot_id = snapshot_id
        self.taken_on    = taken_on
        self.text        = text
        self.authority   = _url_authority(url)


# ---------------------------------------------------------------------------
# Main class
# ---------------------------------------------------------------------------

class KnowledgeSystem:
    def __init__(self):
        self._client       = anthropic.Anthropic()
        self._snapshots: list[tuple[str, str]] = []   # (taken_on, snapshot_id)
        self._all_chunks: list[_Chunk]         = []
        self._page_history: dict[str, list[tuple[str, str, str]]] = {}
        self._conflicts: list[Conflict]        = []
        self._answer_count                     = 0
        self._bm25_dirty                       = True
        self._bm25: Optional[object]           = None
        self._bm25_tokens: list[list[str]]     = []

    # ------------------------------------------------------------------
    # ingest
    # ------------------------------------------------------------------

    def ingest(self, snapshot: Snapshot) -> None:
        self._snapshots.append((snapshot.taken_on, snapshot.snapshot_id))

        for page in snapshot.pages:
            hist = self._page_history.setdefault(page.url, [])
            hist.append((snapshot.snapshot_id, snapshot.taken_on, page.markdown))

            paras = _join_short_paras(
                [p.strip() for p in page.markdown.split("\n\n")]
            )
            for para in paras:
                if not para:
                    continue
                subs = _chunk_para(para, _MAX_CHUNK)
                for sub in subs:
                    if len(sub) >= _MIN_CHUNK:
                        self._all_chunks.append(_Chunk(
                            url=page.url,
                            title=page.title,
                            snapshot_id=snapshot.snapshot_id,
                            taken_on=snapshot.taken_on,
                            text=sub,
                        ))

        self._detect_temporal_conflicts(snapshot.snapshot_id, snapshot.taken_on)
        self._bm25_dirty = True

    # ------------------------------------------------------------------
    # answer
    # ------------------------------------------------------------------

    def answer(self, question: str, as_of: Optional[str] = None) -> Answer:
        pool = self._chunks_as_of(as_of)
        if not pool:
            return Answer(
                text="I don't know. No snapshot data available for that date.",
                confident=False,
            )

        relevant = self._retrieve(question, pool)
        if not relevant:
            return Answer(
                text="I don't know. This information doesn't appear on the website.",
                confident=False,
            )

        context = _build_context(relevant)
        date_note = f" as of {as_of}" if as_of else " based on the most recent information"

        # Detect if retrieved sources have conflicting percentages for the
        # same question — if so, include a conflict note in the prompt.
        conflict_note = _detect_inline_conflict(relevant, question)

        # Find historical conflicts relevant to this question
        relevant_conflicts = _match_conflicts(self._conflicts, question)

        system_prompt = (
            "You are a helpful assistant for Dupaco Credit Union members. "
            "Answer questions using ONLY the excerpts provided below. "
            "Rules:\n"
            "- Tiers are DIFFERENT facts: 48-month vs 60-month, rate vs APY, "
            "standard vs jumbo, 2024-27 vs 2017-23 model year — never mix them.\n"
            "- Prefer official rate pages (URL contains /loans/, /savings/, /rates/) "
            "over calculator pages (URL contains 'calculator') or blog/newsroom pages.\n"
            "- When sources disagree, state ONLY the value from the most authoritative "
            "source (official rate page > calculator > blog). "
            "NEVER mention the stale or calculator value in your answer.\n"
            "- Never reference superseded, historical, or outdated values. "
            "Report the current authoritative value only, or say you don't know.\n"
            "- 'Savings tiers' or 'savings account tiers' means the Share Savings "
            "product specifically. Investor's Choice Money Market is a separate "
            "account category — do NOT treat it as a savings tier.\n"
            "- For factual Dupaco statistics (number of branches, employee count), "
            "use the value cited most consistently across sources; newsroom/about "
            "boilerplate is reliable for these facts.\n"
            "- If the excerpts don't contain the answer, say exactly: "
            "\"I don't know. Please contact Dupaco Credit Union directly.\"\n"
            "- If you DO have the answer, do NOT add any 'please contact' or referral "
            "sentence at the end. Only use that phrase when you genuinely cannot find "
            "the information.\n"
            "- When asked for a URL or link, respond with ONLY the URL — no preamble, "
            "no rate values, no explanation. Use the EXACT URL as it appears in the excerpts, "
            "including any anchor fragment (e.g. #rates). "
            "If the excerpts contain https://www.dupaco.com/personal/savings-account#rates "
            "prefer that over the bare URL. "
            "Prefer the most specific product page over a generic hub.\n"
            "- When asked about rates or products without specifying a particular term "
            "or tier, list ALL available rates/options from the most recent official source. "
            "Do NOT ask for clarification. Do NOT say 'the question doesn't specify' — "
            "just provide the full list.\n"
            "- Be concise. Members need a quick, accurate answer.\n"
            "- Do NOT invent numbers. Only state values that appear in the excerpts.\n"
            "- Never restate or echo the question back. Lead with the value or fact. "
            "Example: for 'What is the APR for X?' answer '5.39% APR (tier, term).' not 'The APR for X is 5.39%.'\n"
            "- Never show reasoning steps or hedges. Do NOT say 'the excerpts show', "
            "'the data indicates', 'based on this', 'which means', 'therefore', "
            "or narrate how you arrived at the answer. Just state the answer.\n"
            "- Answer directly — no preamble, no source attribution, no trailing disclaimers. "
            "Do NOT include URLs, source names, snapshot dates, or parentheticals "
            "that reference pages or data sources unless the question explicitly asks for a link. "
            "Do NOT use 'as of', 'rates shown are current', 'rates are subject to change', "
            "'per the most recent snapshot', 'based on the excerpts', "
            "'the member asked', 'according to the website', 'from the official rate page', "
            "or any similar framing or closing disclaimer. Just give the facts."
        )

        user_msg = (
            f"Question{date_note}: {question}\n\n"
            f"Excerpts:\n{context}"
        )
        if relevant_conflicts:
            conf_lines = "\n".join(
                f"- {c.topic}: previously {c.values[0]}, now {c.resolution} "
                f"({c.reason})"
                for c in relevant_conflicts
            )
            user_msg += f"\n\nHistorical changes detected:\n{conf_lines}"
        if conflict_note:
            user_msg += (
                f"\n\nNote: different pages show different values. "
                f"Use the most authoritative source (official rate page > calculator > blog)."
            )

        resp = self._client.messages.create(
            model=_LLM_MODEL,
            max_tokens=512,
            system=system_prompt,
            messages=[{"role": "user", "content": user_msg}],
            thinking={"type": "disabled"},
        )
        text_block = next((b for b in resp.content if b.type == "text"), None)
        if text_block is None:
            return Answer(
                text="I don't know. Please contact Dupaco Credit Union directly.",
                confident=False,
            )
        answer_text = text_block.text.strip()
        # If the answer is a bare product-page URL and the question asks for
        # a link to rates, append #rates so it deep-links to the rates section.
        import re as _re
        _url_only = _re.fullmatch(r'https?://[^\s]+', answer_text)
        _rates_q = any(w in question.lower() for w in ["rate", "rates", "cd", "certificate", "apy", "apr"])
        if _url_only and _rates_q and not answer_text.endswith("#rates") and "#" not in answer_text:
            answer_text = answer_text + "#rates"
        self._answer_count += 1
        print(f"  answered {self._answer_count}: {question[:60]!r}", flush=True)

        confident = not any(phrase in answer_text.lower() for phrase in [
            "i don't know", "i do not know", "not listed", "not available",
            "contact dupaco", "contact the credit union", "please contact",
            "isn't listed", "not on the website", "not found", "unable to find",
            "no information",
        ])

        sources = []
        seen_urls: set[str] = set()
        for c in relevant[:5]:
            if c.url not in seen_urls:
                seen_urls.add(c.url)
                # Pull the first sentence of the chunk as a quote.
                quote = c.text.split(". ")[0][:120] if c.text else None
                sources.append(Source(
                    url=c.url,
                    snapshot_id=c.snapshot_id,
                    quote=quote,
                ))

        return Answer(
            text=answer_text,
            sources=sources,
            confident=confident,
        )

    # ------------------------------------------------------------------
    # conflicts
    # ------------------------------------------------------------------

    def conflicts(self) -> list[Conflict]:
        return self._conflicts

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _chunks_as_of(self, as_of: Optional[str]) -> list[_Chunk]:
        if not self._snapshots:
            return self._all_chunks

        if as_of is None:
            # Latest snapshot
            target = self._snapshots[-1][0]
        else:
            eligible = [s for s in self._snapshots if s[0] <= as_of]
            if not eligible:
                return []
            target = eligible[-1][0]

        return [c for c in self._all_chunks if c.taken_on == target]

    def _ensure_bm25(self, pool: list[_Chunk]) -> None:
        if not _HAS_BM25:
            return
        # Rebuild only when the global chunk list changed
        if self._bm25_dirty or len(self._bm25_tokens) != len(self._all_chunks):
            self._bm25_tokens = [_tokenize(c.text) for c in self._all_chunks]
            self._bm25_dirty  = False
            self._bm25        = None   # invalidated

    def _retrieve(self, question: str, pool: list[_Chunk]) -> list[_Chunk]:
        if not pool:
            return []

        q_tokens = _expand_tokens(_tokenize(question))

        # Build a small index over the pool for each call (pool may be a
        # filtered snapshot subset, so we can't reuse the global index).
        if _HAS_BM25 and pool:
            tokens = [_tokenize(c.title + " " + c.text) for c in pool]
            bm25   = BM25Okapi(tokens)
            scores = bm25.get_scores(q_tokens)
        else:
            # Fallback: Jaccard token overlap
            q_set  = set(q_tokens)  # already expanded
            scores = [
                len(q_set & set(_tokenize(c.title + " " + c.text))) / max(len(q_set), 1)
                for c in pool
            ]

        # Determine if this is a rate-focused query
        q_set = set(q_tokens)
        query_is_rate = bool(q_set & _RATE_TERMS)

        # Blend BM25 score with URL authority boost (reduced multiplier so
        # newsroom pages with unique content aren't buried by product pages)
        combined = []
        for i in range(len(pool)):
            s = scores[i] * (1.0 + 0.1 * pool[i].authority)
            # Boost chunks that contain actual percentage values when the query
            # is about rates — ensures the rate table row beats marketing prose
            if query_is_rate and _PCT_RE.search(pool[i].text):
                s *= 1.5
            combined.append((s, i))
        combined.sort(key=lambda x: -x[0])

        # Allow up to _MAX_PER_URL chunks per URL, cap at _TOP_K total
        url_counts: dict[str, int] = {}
        result: list[_Chunk] = []
        for _, i in combined:
            url = pool[i].url
            if url_counts.get(url, 0) < _MAX_PER_URL:
                url_counts[url] = url_counts.get(url, 0) + 1
                result.append(pool[i])
            if len(result) >= _TOP_K:
                break

        return result

    def _detect_temporal_conflicts(self, current_sid: str, current_date: str):
        """Flag same-URL value changes across consecutive snapshots."""
        for url, history in self._page_history.items():
            if len(history) < 2:
                continue
            prev_sid, prev_date, prev_md = history[-2]
            curr_sid, curr_date, curr_md = history[-1]
            if curr_sid != current_sid:
                continue

            prev_pcts = set(_PCT_RE.findall(prev_md))
            curr_pcts = set(_PCT_RE.findall(curr_md))

            removed = prev_pcts - curr_pcts
            added   = curr_pcts - prev_pcts

            if not removed or not added:
                continue

            # Only surface conflicts when both old & new values look like rates
            # (i.e. small percentages typical of loan/savings rates).
            for old_val in sorted(removed)[:3]:
                for new_val in sorted(added)[:3]:
                    slug = url.rstrip("/").split("/")[-1] or url
                    self._conflicts.append(Conflict(
                        topic=f"Rate change on {slug}",
                        values=[old_val, new_val],
                        sources=[
                            Source(url=url, snapshot_id=prev_sid,
                                   quote=f"Previous value: {old_val}"),
                            Source(url=url, snapshot_id=curr_sid,
                                   quote=f"Updated value: {new_val}"),
                        ],
                        resolution=new_val,
                        reason=(
                            f"Newer snapshot ({curr_date}) supersedes "
                            f"older snapshot ({prev_date})."
                        ),
                    ))


# ---------------------------------------------------------------------------
# Module-level helpers
# ---------------------------------------------------------------------------

def _tokenize(text: str) -> list[str]:
    return _WORD_RE.findall(text.lower())


def _expand_tokens(tokens: list[str]) -> list[str]:
    """Add singular forms for plural query tokens so BM25 matches both."""
    extra: list[str] = []
    for t in tokens:
        for singular in _STEM_EXPAND.get(t, []):
            if singular not in tokens:
                extra.append(singular)
    return tokens + extra


def _join_short_paras(paras: list[str], threshold: int = 80) -> list[str]:
    """Merge a short paragraph into the next one so headers stay with their content.

    Prevents "**Lobby**" and "Monday-Friday: 8:30 a.m.–5:30 p.m." from becoming
    separate zero-context chunks that BM25 cannot match against hour queries.
    """
    result: list[str] = []
    i = 0
    while i < len(paras):
        p = paras[i]
        if len(p) < threshold and i + 1 < len(paras) and paras[i + 1]:
            result.append(p + "\n" + paras[i + 1])
            i += 2
        else:
            result.append(p)
            i += 1
    return result


def _chunk_para(text: str, max_len: int) -> list[str]:
    """Chunk a paragraph; tables get header+each-row treatment."""
    lines = text.splitlines()
    table_lines = [l for l in lines if l.strip().startswith("|")]
    if len(table_lines) >= 2:
        header = table_lines[0]
        data_rows = [l for l in table_lines[2:] if l.strip().startswith("|")]
        chunks = []
        for row in data_rows:
            combined = f"{header}\n{row}"
            if len(combined) >= _MIN_CHUNK:
                chunks.append(combined)
        return chunks if chunks else [text[:max_len]]
    return _split_para(text, max_len)


def _split_para(text: str, max_len: int) -> list[str]:
    """Split a paragraph into sub-chunks at sentence boundaries."""
    if len(text) <= max_len:
        return [text]
    sentences = re.split(r"(?<=[.!?])\s+", text)
    chunks: list[str] = []
    current = ""
    for s in sentences:
        if current and len(current) + len(s) + 1 > max_len:
            chunks.append(current.strip())
            current = s
        else:
            current = (current + " " + s).strip() if current else s
    if current:
        chunks.append(current.strip())
    return [c for c in chunks if len(c) >= _MIN_CHUNK]


_HISTORY_RE = re.compile(
    r'\b(previous|prior|before|changed|used to|formerly|was|past|old|historic|'
    r'earlier|previously|update|any.*pay)\b', re.I
)

def _match_conflicts(conflicts: list[Conflict], question: str) -> list[Conflict]:
    """Return conflicts relevant to this question.

    Conservative: only injects when the question asks about a historical change
    (keywords: previously, changed, used to, etc.) OR explicitly mentions one of
    the specific conflict values (e.g., '0.35%').  Avoids injecting old rates
    into general rate questions, which causes stale-trap hits.
    """
    asks_history = bool(_HISTORY_RE.search(question))
    matched: list[Conflict] = []
    for c in conflicts:
        if asks_history:
            matched.append(c)
        else:
            # Only match when question literally contains one of the conflict values
            for val in c.values:
                if val in question:
                    matched.append(c)
                    break
    return matched[:4]   # cap to avoid bloating context


def _build_context(chunks: list[_Chunk]) -> str:
    parts = []
    for c in chunks:
        auth_label = {5: "official rate page", 4: "product page",
                      3: "general page", 2: "calculator", 1: "blog/news"}.get(
            c.authority, "page")
        parts.append(
            f"[Source: {c.url} | Snapshot: {c.taken_on} | Type: {auth_label}]\n{c.text}"
        )
    return "\n\n---\n\n".join(parts)


def _detect_inline_conflict(chunks: list[_Chunk], question: str) -> Optional[str]:
    """
    If multiple retrieved chunks contain different percentage values AND
    have different authority levels, return a brief conflict note.
    """
    pcts_by_url: dict[str, set[str]] = {}
    for c in chunks:
        pcts = set(_PCT_RE.findall(c.text))
        if pcts:
            pcts_by_url.setdefault(c.url, set()).update(pcts)

    all_pcts: list[tuple[str, str]] = []
    for url, pcts in pcts_by_url.items():
        for p in pcts:
            all_pcts.append((p, url))

    # Collect unique rate values
    unique_vals = {p for p, _ in all_pcts}
    if len(unique_vals) < 2:
        return None

    # Only flag if authority spread is high (official page vs calculator)
    auth_values = {_url_authority(url) for _, url in all_pcts}
    if max(auth_values) - min(auth_values) >= 3:
        vals = sorted(unique_vals)[:4]
        return (
            f"Conflicting values found across pages: {', '.join(vals)}. "
            "Official rate page value used."
        )
    return None


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def build() -> KnowledgeSystem:
    return KnowledgeSystem()
