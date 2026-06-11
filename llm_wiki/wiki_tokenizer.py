"""
Query tokenization utilities ported from nashsu_llm_wiki's Rust backend
(src-tauri/src/commands/search.rs).

CJK handling: tokens longer than 2 CJK characters generate bigrams, single
characters, and the original token. Non-CJK tokens pass through unchanged.
"""

import re
from typing import List

_CJK_RE = re.compile(r"[\u3400-\u9fff]")

_STOP_WORDS = {
    "的", "是", "了", "什么", "在", "有", "和", "与", "对", "从",
    "the", "is", "a", "an", "what", "how", "are", "was", "were",
    "do", "does", "did", "be", "been", "being", "have", "has", "had",
    "it", "its", "in", "on", "at", "to", "for", "of", "with", "by",
    "this", "that", "these", "those",
}

_SEPARATOR_CHARS = set(
    " \t\r\n，。！？、；：“”‘’（）·～…"
)


def is_query_separator(c: str) -> bool:
    """Return True if char separates query tokens."""
    if c.isspace():
        return True
    # ASCII punctuation
    code = ord(c)
    if code < 128:
        if 33 <= code <= 47 or 58 <= code <= 64 or 91 <= code <= 96 or 123 <= code <= 126:
            return True
    return c in _SEPARATOR_CHARS


def _split_by_separator(text: str):
    """Split text by query separators and CJK/non-CJK boundaries."""
    current = []
    prev_is_cjk = None
    for ch in text:
        if is_query_separator(ch):
            if current:
                yield "".join(current)
                current = []
            prev_is_cjk = None
            continue

        is_cjk = bool(_CJK_RE.match(ch))
        if prev_is_cjk is not None and prev_is_cjk != is_cjk:
            if current:
                yield "".join(current)
                current = []

        current.append(ch)
        prev_is_cjk = is_cjk

    if current:
        yield "".join(current)


def tokenize_query(query: str) -> List[str]:
    """Tokenize a query into a sorted, deduplicated list of tokens."""
    if not query:
        return []

    raw = [
        token
        for token in _split_by_separator(query.lower())
        if len(token) > 1 and token not in _STOP_WORDS
    ]

    seen = set()
    out = []

    for token in raw:
        chars = list(token)
        has_cjk = any(_CJK_RE.match(ch) for ch in chars)
        if has_cjk and len(chars) > 2:
            # CJK bigrams
            for i in range(len(chars) - 1):
                bigram = chars[i] + chars[i + 1]
                if bigram not in seen:
                    seen.add(bigram)
                    out.append(bigram)
            # CJK single chars
            for ch in chars:
                if ch not in _STOP_WORDS and ch not in seen:
                    seen.add(ch)
                    out.append(ch)
            # original token
            if token not in seen:
                seen.add(token)
                out.append(token)
        else:
            if token not in seen:
                seen.add(token)
                out.append(token)

    return out


def trim_query_punctuation(value: str) -> str:
    """Strip query separators from both ends of a string."""
    chars = list(value.lower())
    start = 0
    while start < len(chars) and is_query_separator(chars[start]):
        start += 1
    end = len(chars) - 1
    while end >= start and is_query_separator(chars[end]):
        end -= 1
    return "".join(chars[start : end + 1])
