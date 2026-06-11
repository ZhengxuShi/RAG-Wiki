"""
SHA256-based incremental ingest cache.
Derived from src/lib/ingest-cache.ts
"""

import hashlib
import json
import os

from llm_wiki.wiki_utils import file_exists, read_file_utf8, write_file_utf8, normalize_path


def _cache_file_path(project_path: str) -> str:
    return f"{normalize_path(project_path)}/.llm-wiki/ingest-cache.json"


def _sha256(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def check_ingest_cache(project_path: str, source_identity: str, source_content: str) -> list[str] | None:
    """
    Returns cached list of written file paths if source hasn't changed,
    otherwise None to trigger full pipeline.
    """
    cache_path = _cache_file_path(project_path)
    if not file_exists(cache_path):
        return None
    try:
        data = json.loads(read_file_utf8(cache_path))
        entry = data.get(source_identity)
        if entry and entry.get("hash") == _sha256(source_content):
            return entry.get("files", [])
    except Exception:
        pass
    return None


def save_ingest_cache(project_path: str, source_identity: str, source_content: str, written_files: list[str]) -> None:
    cache_path = _cache_file_path(project_path)
    data: dict = {}
    if file_exists(cache_path):
        try:
            data = json.loads(read_file_utf8(cache_path))
        except Exception:
            data = {}
    data[source_identity] = {
        "hash": _sha256(source_content),
        "files": written_files,
    }
    write_file_utf8(cache_path, json.dumps(data, indent=2, ensure_ascii=False))
