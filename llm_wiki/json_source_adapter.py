"""
JSON chunk -> source content adapter for LLM Wiki ingest.
Strategy B: metadata-annotated concatenation.
"""


def json_chunks_to_source_content(chunks: list[dict]) -> str:
    """
    Strategy B: Sort by page_idx + unique_id, prepend metadata header to each chunk.
    Hard-truncate at 50,000 chars to match original LLM Wiki behavior.
    """
    valid = [c for c in chunks if c.get("type", "text") == "text" and c.get("text")]
    valid.sort(key=lambda c: (c.get("page_idx", 0), c.get("unique_id", 0)))
    parts = []
    for c in valid:
        header = f"[Page {c.get('page_idx', '?')}, ID {c.get('unique_id', '?')}, Level {c.get('text_level', '')}]"
        parts.append(f"{header}\n{c['text']}")
    content = "\n\n".join(parts)
    if len(content) > 50000:
        content = content[:50000] + "\n\n[...truncated...]"
    return content
