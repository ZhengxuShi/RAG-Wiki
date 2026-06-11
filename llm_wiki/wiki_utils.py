"""
Utility functions for LLM Wiki.
Derived from src/lib/path-utils.ts and src/lib/frontmatter.ts
"""

import os
import re
from typing import Any


def normalize_path(path: str) -> str:
    """Normalize backslashes to forward slashes."""
    return path.replace("\\", "/")


def get_file_name(path: str) -> str:
    """Extract filename from path."""
    return os.path.basename(path)


def get_relative_path(path: str, base_path: str) -> str:
    """Get relative path from base."""
    return os.path.relpath(path, base_path).replace("\\", "/")


def read_file_utf8(path: str) -> str:
    """Read file with UTF-8 encoding."""
    with open(path, "r", encoding="utf-8") as f:
        return f.read()


def write_file_utf8(path: str, content: str) -> None:
    """Write file with UTF-8 encoding, creating parent dirs if needed."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.write(content)


def file_exists(path: str) -> bool:
    return os.path.exists(path) and os.path.isfile(path)


def list_directory(path: str) -> list[dict]:
    """List directory entries matching FileNode shape."""
    entries = []
    for entry in os.listdir(path):
        full = os.path.join(path, entry)
        entries.append({
            "name": entry,
            "path": normalize_path(full),
            "is_dir": os.path.isdir(full),
        })
    return entries


def extract_frontmatter(content: str) -> tuple[dict[str, Any], str]:
    """
    Parse YAML frontmatter from markdown content.
    Returns (frontmatter_dict, body_content).
    """
    match = re.match(r"^---\n(.*?)\n---\n?(.*)$", content, re.DOTALL)
    if not match:
        return {}, content
    fm_text = match.group(1)
    body = match.group(2)
    frontmatter: dict[str, Any] = {}
    # Simple YAML-like parsing for scalar and list fields
    current_key = None
    for line in fm_text.split("\n"):
        line = line.rstrip()
        if not line:
            continue
        # List item
        list_match = re.match(r"^\s+-\s+(.*)$", line)
        if list_match and current_key:
            val = list_match.group(1).strip().strip('"').strip("'")
            existing = frontmatter.get(current_key)
            if not isinstance(existing, list):
                frontmatter[current_key] = []
            frontmatter[current_key].append(val)
            continue
        # Key-value
        kv_match = re.match(r"^(\w+):\s*(.*)$", line)
        if kv_match:
            key = kv_match.group(1)
            val = kv_match.group(2).strip()
            # Strip quotes
            val = val.strip('"').strip("'")
            # Try parse inline list: [a, b, c]
            if val.startswith("[") and val.endswith("]"):
                items = [v.strip().strip('"').strip("'") for v in val[1:-1].split(",") if v.strip()]
                frontmatter[key] = items
            else:
                frontmatter[key] = val
            current_key = key
    return frontmatter, body


def build_frontmatter(data: dict[str, Any]) -> str:
    """Build YAML frontmatter string from dict."""
    lines = ["---"]
    for key, val in data.items():
        if isinstance(val, list):
            lines.append(f"{key}:")
            for item in val:
                lines.append(f'  - "{item}"')
        else:
            lines.append(f'{key}: "{val}"')
    lines.append("---")
    return "\n".join(lines)


def slugify(text: str) -> str:
    """Create a URL-safe slug from text."""
    text = text.lower()
    text = re.sub(r"[^\w\s-]", "", text)
    text = re.sub(r"[\s_]+", "-", text)
    return text.strip("-")


def source_identity_for_path(project_path: str, source_path: str) -> str:
    """Create a stable source identity string."""
    rel = get_relative_path(source_path, project_path)
    return normalize_path(rel)


def source_summary_slug_from_identity(identity: str) -> str:
    """Create a slug for the source summary page."""
    base = os.path.basename(identity)
    name, _ = os.path.splitext(base)
    return slugify(name)
