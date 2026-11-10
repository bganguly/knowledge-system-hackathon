"""Text normalisation helpers applied at ingest time."""
from __future__ import annotations

import re

_WORD_RE = re.compile(r"[a-z0-9$%.]+")
_LOAN_TERM_RE = re.compile(r'\b(\d+)[- ]?years?\b', re.IGNORECASE)


def _normalize_loan_query(text: str) -> str:
    """Convert 'N year(s)' loan terms to equivalent months so BM25 hits rate-table chunks."""
    return _LOAN_TERM_RE.sub(lambda m: f"{int(m.group(1)) * 12} months", text)


_YEAR_RANGE_EXPAND_RE = re.compile(r'\b(\d{4})-(\d{2})\s+Model\s+Year\b', re.I)


def _expand_model_year_ranges(text: str) -> str:
    """Append individual years to abbreviated year-range tokens in chunk text.

    "2024-27 Model Year" → "2024-27 Model Year [2025 2026 2027]"
    """
    def _expand(m: re.Match) -> str:
        start = int(m.group(1))
        suffix = int(m.group(2))
        end = (start // 100) * 100 + suffix
        if end <= start:
            end += 100
        present = set(re.findall(r'\b\d{4}\b', m.group(0)))
        to_add = [str(y) for y in range(start, end + 1) if str(y) not in present]
        return (m.group(0) + " [" + " ".join(to_add) + "]") if to_add else m.group(0)
    return _YEAR_RANGE_EXPAND_RE.sub(_expand, text)


_CHARTER_AREA_RE = re.compile(r'\bcharter\s+area\b', re.I)


def _expand_charter_area(text: str) -> str:
    """Append state names to 'charter area' when not already present in the chunk."""
    if not _CHARTER_AREA_RE.search(text):
        return text
    if re.search(r'\b(Iowa|Illinois|Wisconsin)\b', text, re.I):
        return text
    return _CHARTER_AREA_RE.sub('charter area [Iowa, Illinois and Wisconsin]', text)


_MONTH_ABBREV: dict[str, str] = {
    "Jan": "January", "Feb": "February", "Mar": "March", "Apr": "April",
    "Jun": "June",    "Jul": "July",     "Aug": "August", "Sep": "September",
    "Oct": "October", "Nov": "November", "Dec": "December",
}
_MONTH_ABBREV_RE = re.compile(
    r'\b(' + '|'.join(_MONTH_ABBREV) + r')\.\s+(\d{1,2}),\s+(\d{4})\b'
)


def _expand_month_abbrevs(text: str) -> str:
    """Replace abbreviated month dates with the full month name.

    "Oct. 1, 2024" → "October 1, 2024"
    """
    return _MONTH_ABBREV_RE.sub(
        lambda m: f"{_MONTH_ABBREV[m.group(1)]} {m.group(2)}, {m.group(3)}", text
    )


_RECREATIONAL_YEAR_RE = re.compile(r'\bModel\s+Year\s+\d{4}-\d{2}\b', re.I)


def _tag_recreational_rows(text: str) -> str:
    """Append a category tag to ATV/snowmobile/jet-ski rate rows.

    Rows like "Model Year 2020-19 | 60 Months | 8.94%" use reversed year
    notation exclusively for recreational loans, NOT auto or motorcycle loans.
    """
    if _RECREATIONAL_YEAR_RE.search(text):
        return text + " [ATV snowmobile jet ski recreational loan — not auto not motorcycle]"
    return text


def _url_authority(url: str) -> int:
    """Higher = more authoritative source."""
    u = url.lower()
    if any(x in u for x in ["/newsroom/", "/blog/", "/press", "/news/"]):
        return 1
    if any(x in u for x in ["calculator", "calc", "affordability", "tool"]):
        return 2
    if any(x in u for x in [
        "/personal/loans/", "/personal/savings", "/personal/credit",
        "/personal/checking", "/personal/mortgages",
        "/business/loans/", "/business/savings", "/business/credit",
        "/rates", "/rate-", "#rates",
    ]):
        return 5
    if any(x in u for x in ["/personal/", "/business/"]):
        return 4
    if "/about" in u:
        return 4
    return 3
