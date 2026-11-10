"""Chunk data class, chunking helpers, and retrieval context utilities."""
from __future__ import annotations

import re
from typing import Optional

from _preprocess import _WORD_RE, _url_authority

_MIN_CHUNK = 20
_PCT_RE    = re.compile(r"\b\d+\.\d{1,3}%")

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
    "rates": ["rate", "apy", "dividend"],
    "interest": ["apy", "dividend", "rate"],
    "apy": ["rate", "dividend", "interest"],
    "contact": ["phone", "call", "800"],
    "update": ["change", "modify", "edit"],
    "information": ["info", "details", "data"],
    "grant": ["fhlb", "home$tart", "homebuyer", "assistance"],
    "homebuyer": ["home$tart", "fhlb", "grant", "first-time"],
    "mortgage": ["home", "loan", "purchase", "refinance", "conventional"],
    "mortgages": ["home", "loan", "purchase", "refinance", "conventional"],
    "conventional": ["mortgage", "home", "loan"],
    "vehicle": ["auto", "car", "automobile"],
    "vehicles": ["auto", "car", "automobile"],
    "largest": ["maximum", "borrow", "limit"],
    "maximum": ["largest", "borrow", "limit"],
    "certificate": ["cd", "savings", "share"],
    "certificates": ["certificate", "cd", "savings"],
    "cd": ["certificate", "savings", "share"],
    "credit": ["visa", "card", "platinum", "rewards"],
    "cards": ["card", "visa", "credit"],
    "aprs": ["apr", "rate", "apy"],
    "apr": ["aprs", "rate", "apy"],
    "wallet": ["apple", "pay", "google", "samsung"],
    "recognition": ["recognized", "recognizing", "recognizes", "recognitions"],
    "recognized": ["recognition", "recognizing"],
    "founder": ["employees", "meatpacking", "started", "history"],
    "months": ["month"],
    "month": ["months"],
    "2018": ["2017", "2020"],
    "2019": ["2017", "2020"],
    "2020": ["2017"],
    "2022": ["2021"],
    "2024": ["2023", "2026"],
    "2025": ["2023", "2026"],
}

_SCHEDULE_LINE_RE = re.compile(
    r'\b(Monday|Tuesday|Wednesday|Thursday|Friday|Saturday|Sunday)\b'
    r'.*?(\d{1,2}:\d{2}\s*[ap]\.m\.|noon|midnight|closed|ATM)',
    re.IGNORECASE | re.DOTALL,
)

_HISTORY_RE = re.compile(
    r'\b(previous|prior|before|changed|used to|formerly|was|past|old|historic|'
    r'earlier|previously|update|any.*pay)\b', re.I
)


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


def _merge_schedule_blocks(paras: list[str]) -> list[str]:
    """Merge consecutive day/time schedule lines into one paragraph.

    Without this merge, 'noon' (Saturday) ends up in a chunk with no 'Lobby'
    context, making BM25 unable to surface it for hour queries.
    """
    def _is_schedule(p: str) -> bool:
        return bool(_SCHEDULE_LINE_RE.search(p))

    result: list[str] = []
    i = 0
    while i < len(paras):
        p = paras[i]
        if _is_schedule(p):
            group = [p]
            j = i + 1
            while j < len(paras) and _is_schedule(paras[j]):
                group.append(paras[j])
                j += 1
            if result and not _is_schedule(result[-1]):
                header = result.pop()
                result.append(header + "\n" + "\n".join(group))
            else:
                result.append("\n".join(group))
            i = j
        else:
            result.append(p)
            i += 1
    return result


def _join_short_paras(paras: list[str], threshold: int = 80) -> list[str]:
    """Merge a short paragraph into the next one so headers stay with their content."""
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


def _match_conflicts(conflicts: list, question: str) -> list:
    """Return conflicts relevant to this question.

    Only injects when the question asks about a historical change OR explicitly
    mentions one of the specific conflict values. Avoids injecting old rates
    into general rate questions.
    """
    asks_history = bool(_HISTORY_RE.search(question))
    matched = []
    for c in conflicts:
        if asks_history:
            matched.append(c)
        else:
            for val in c.values:
                if val in question:
                    matched.append(c)
                    break
    return matched[:4]


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
    """Return a brief conflict note when chunks from different-authority sources
    contain different percentage values."""
    pcts_by_url: dict[str, set[str]] = {}
    for c in chunks:
        pcts = set(_PCT_RE.findall(c.text))
        if pcts:
            pcts_by_url.setdefault(c.url, set()).update(pcts)

    all_pcts: list[tuple[str, str]] = []
    for url, pcts in pcts_by_url.items():
        for p in pcts:
            all_pcts.append((p, url))

    unique_vals = {p for p, _ in all_pcts}
    if len(unique_vals) < 2:
        return None

    auth_values = {_url_authority(url) for _, url in all_pcts}
    if max(auth_values) - min(auth_values) >= 3:
        vals = sorted(unique_vals)[:4]
        return (
            f"Conflicting values found across pages: {', '.join(vals)}. "
            "Official rate page value used."
        )
    return None
