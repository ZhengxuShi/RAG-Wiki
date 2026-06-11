"""
Two-step ingest core for LLM Wiki.
Derived from src/lib/ingest.ts, src/lib/ingest-sanitize.ts,
src/lib/page-merge.ts, src/lib/sources-merge.ts
"""

import datetime
import logging
import re

from llm_wiki.ingest_cache import check_ingest_cache, save_ingest_cache
from llm_wiki.wiki_llm_client import stream_chat
from llm_wiki.wiki_models import LlmConfig, ParseFileBlocksResult, ParsedFileBlock
from llm_wiki.wiki_prompts import (
    build_analysis_prompt,
    build_generation_prompt,
    build_page_merge_prompt,
)
from llm_wiki.wiki_utils import (
    extract_frontmatter,
    get_file_name,
    normalize_path,
    read_file_utf8,
    source_identity_for_path,
    source_summary_slug_from_identity,
    write_file_utf8,
)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Path safety
# ---------------------------------------------------------------------------

def is_safe_ingest_path(p: str) -> bool:
    """Reject FILE block paths that try to escape the project's wiki/ directory."""
    if not isinstance(p, str) or p.strip() == "":
        return False
    if re.search(r"[\x00-\x1f]", p):
        return False
    if p.startswith("/") or p.startswith("\\"):
        return False
    if re.match(r"^[a-zA-Z]:", p):
        return False
    normalized = p.replace("\\", "/")
    segments = normalized.split("/")
    if ".." in segments:
        return False
    if not all(_is_windows_safe_path_segment(seg) for seg in segments):
        return False
    if not normalized.startswith("wiki/"):
        return False
    return True


def _is_windows_safe_path_segment(segment: str) -> bool:
    if not segment:
        return False
    if re.search(r'[<>:"|?*]', segment):
        return False
    if re.search(r"[ .]$", segment):
        return False
    stem = segment.split(".")[0].upper()
    if not stem:
        return False
    if stem in ("CON", "PRN", "AUX", "NUL") or re.match(r"^COM[1-9]$", stem) or re.match(r"^LPT[1-9]$", stem):
        return False
    return True


def _is_log_path(relative_path: str) -> bool:
    return relative_path == "wiki/log.md" or relative_path.endswith("/log.md")


def _is_listing_path(relative_path: str) -> bool:
    return (
        relative_path == "wiki/index.md"
        or relative_path.endswith("/index.md")
        or relative_path == "wiki/overview.md"
        or relative_path.endswith("/overview.md")
    )


# ---------------------------------------------------------------------------
# File block parser
# ---------------------------------------------------------------------------

_OPENER_LINE = re.compile(r"^---\s*FILE:\s*(.+?)\s*---\s*$", re.IGNORECASE)
_CLOSER_LINE = re.compile(r"^---\s*END\s+FILE\s*---\s*$", re.IGNORECASE)
_FENCE_LINE = re.compile(r"^\s{0,3}(```+|~~~+)")


def parse_file_blocks(text: str) -> ParseFileBlocksResult:
    """Parse an LLM stage-2 generation into FILE blocks.

    Handles CRLF, code fences, stream truncation, marker whitespace, and
    empty paths.
    """
    normalized = text.replace("\r\n", "\n")
    lines = normalized.split("\n")
    blocks: list[ParsedFileBlock] = []
    warnings: list[str] = []
    i = 0
    while i < len(lines):
        opener_match = _OPENER_LINE.match(lines[i])
        if not opener_match:
            i += 1
            continue
        path = opener_match.group(1).strip()
        i += 1
        content_lines: list[str] = []
        fence_marker: str | None = None
        fence_len = 0
        closed = False
        while i < len(lines):
            line = lines[i]
            fence_match = _FENCE_LINE.match(line)
            if fence_match:
                run = fence_match.group(1)
                char = run[0]
                length = len(run)
                if fence_marker is None:
                    fence_marker = char
                    fence_len = length
                elif char == fence_marker and length >= fence_len:
                    fence_marker = None
                    fence_len = 0
                content_lines.append(line)
                i += 1
                continue
            if fence_marker is None and _CLOSER_LINE.match(line):
                closed = True
                i += 1
                break
            content_lines.append(line)
            i += 1
        if not closed:
            label = path or "(unnamed)"
            warnings.append(
                f'FILE block "{label}" was not closed before end of stream — likely truncation. Block dropped.'
            )
            continue
        if not path:
            warnings.append("FILE block with empty path skipped.")
            continue
        if not is_safe_ingest_path(path):
            warnings.append(
                f'FILE block with unsafe path "{path}" rejected (must be under wiki/, no ..).'
            )
            continue
        blocks.append(ParsedFileBlock(path=path, content="\n".join(content_lines)))
    return ParseFileBlocksResult(blocks=blocks, warnings=warnings)


# ---------------------------------------------------------------------------
# Content sanitization (ingest-sanitize.ts)
# ---------------------------------------------------------------------------

def sanitize_ingested_file_content(content: str) -> str:
    """Clean up an LLM-generated wiki page body before it hits disk."""
    cleaned = _strip_outer_code_fence(content)
    cleaned = _strip_frontmatter_key_prefix(cleaned)
    cleaned = _repair_wikilink_lists_in_frontmatter(cleaned)
    return cleaned


def _strip_outer_code_fence(content: str) -> str:
    open_match = re.match(r"^[ \t]*```(?:yaml|md|markdown)?[ \t]*\r?\n", content)
    if not open_match:
        return content
    after_open = content[open_match.end() :]
    close_match = re.search(r"\r?\n[ \t]*```[ \t]*\r?\n?\s*$", after_open)
    if not close_match:
        return content
    return after_open[: close_match.start()]


def _strip_frontmatter_key_prefix(content: str) -> str:
    m = re.match(r"^[ \t]*frontmatter\s*:\s*\r?\n(?=[ \t]*---\s*\r?\n)", content)
    if not m:
        return content
    return content[m.end() :]


def _repair_wikilink_lists_in_frontmatter(content: str) -> str:
    fm_re = re.compile(r"^---\s*\r?\n([\s\S]*?)\r?\n---\s*(?:\r?\n|$)")
    m = fm_re.match(content)
    if not m:
        return content
    payload = m.group(1)
    repaired_lines = []
    for line in payload.split("\n"):
        lm = re.match(
            r"^(\s*[A-Za-z_][\w-]*\s*:\s*)(\[\[[^\]]+\]\](?:\s*,\s*\[\[[^\]]+\]\])+)\s*$",
            line,
        )
        if not lm:
            repaired_lines.append(line)
            continue
        items = [s.strip() for s in lm.group(2).split(",") if s.strip()]
        quoted = ", ".join(f'"{s}"' for s in items)
        repaired_lines.append(f"{lm.group(1)}[{quoted}]")
    repaired_payload = "\n".join(repaired_lines)
    prefix_len = len("---\n")
    return content[:prefix_len] + repaired_payload + content[prefix_len + len(payload) :]


# ---------------------------------------------------------------------------
# Frontmatter array helpers (sources-merge.ts)
# ---------------------------------------------------------------------------

def parse_frontmatter_array(content: str, field_name: str) -> list[str]:
    fm_match = re.match(r"^---\n([\s\S]*?)\n---", content)
    if not fm_match:
        return []
    fm = fm_match.group(1)
    escaped = re.escape(field_name)
    block_re = re.compile(rf"^{escaped}:\s*\n((?:[ \t]+-\s+.+\n?)+)", re.MULTILINE)
    block = block_re.search(fm)
    if block:
        out: list[str] = []
        for line in block.group(1).split("\n"):
            m = re.match(r'^\s+-\s+["\']?(.+?)["\']?\s*$', line)
            if m and m.group(1):
                out.append(m.group(1).strip())
        return out
    inline_re = re.compile(rf"^{escaped}:\s*\[([^\]]*)\]", re.MULTILINE)
    inline = inline_re.search(fm)
    if not inline:
        return []
    body = inline.group(1).strip()
    if body == "":
        return []
    return [s.strip().strip('"').strip("'") for s in body.split(",") if s.strip()]


def write_frontmatter_array(content: str, field_name: str, values: list[str]) -> str:
    fm_match = re.match(r"^(---\n)([\s\S]*?)(\n---)", content)
    if not fm_match:
        return content
    open_delim, fm_body, close_delim = fm_match.groups()
    escaped = re.escape(field_name)
    serialized = ", ".join(f'"{s}"' for s in values)
    new_line = f"{field_name}: [{serialized}]"
    inline_re = re.compile(rf"^{escaped}:\s*\[[^\]]*\]", re.MULTILINE)
    if inline_re.search(fm_body):
        rewritten = inline_re.sub(new_line, fm_body)
        return f"{open_delim}{rewritten}{close_delim}{content[fm_match.end():]}"
    block_re = re.compile(rf"^{escaped}:\s*\n((?:[ \t]+-\s+.+\n?)+)", re.MULTILINE)
    if block_re.search(fm_body):
        rewritten = block_re.sub(new_line, fm_body)
        return f"{open_delim}{rewritten}{close_delim}{content[fm_match.end():]}"
    rewritten = f"{fm_body}\n{new_line}"
    return f"{open_delim}{rewritten}{close_delim}{content[fm_match.end():]}"


def _merge_lists(existing: list[str], incoming: list[str]) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for s in existing + incoming:
        key = s.lower()
        if key in seen:
            continue
        seen.add(key)
        out.append(s)
    return out


def merge_array_fields_into_content(new_content: str, existing_content: str | None, fields: list[str]) -> str:
    if not existing_content:
        return new_content
    if not re.match(r"^---\n", existing_content):
        return new_content
    result = new_content
    changed = False
    for field in fields:
        old_values = parse_frontmatter_array(existing_content, field)
        if not old_values:
            continue
        new_values = parse_frontmatter_array(result, field)
        merged = _merge_lists(old_values, new_values)
        if len(merged) == len(new_values) and all(m == n for m, n in zip(merged, new_values)):
            continue
        result = write_frontmatter_array(result, field, merged)
        changed = True
    return result if changed else new_content


def parse_sources(content: str) -> list[str]:
    return parse_frontmatter_array(content, "sources")


def write_sources(content: str, sources: list[str]) -> str:
    return write_frontmatter_array(content, "sources", sources)


# ---------------------------------------------------------------------------
# Scalar frontmatter helper
# ---------------------------------------------------------------------------

def _set_frontmatter_scalar(content: str, field_name: str, value: str) -> str:
    fm_match = re.match(r"^(---\n)([\s\S]*?)(\n---)", content)
    if not fm_match:
        return content
    open_delim, fm_body, close_delim = fm_match.groups()
    escaped = re.escape(field_name)
    new_line = f"{field_name}: {value}"
    line_re = re.compile(rf"^{escaped}:\s*(?!\[)([^\n]*)", re.MULTILINE)
    if line_re.search(fm_body):
        rewritten = line_re.sub(new_line, fm_body)
        return f"{open_delim}{rewritten}{close_delim}{content[fm_match.end():]}"
    rewritten = f"{fm_body}\n{new_line}"
    return f"{open_delim}{rewritten}{close_delim}{content[fm_match.end():]}"


def _inject_domain(content: str, domain: str) -> str:
    if not domain:
        return content
    fm_match = re.match(r"^(---\n)([\s\S]*?)(\n---)", content)
    if not fm_match:
        return content
    open_delim, fm_body, close_delim = fm_match.groups()
    if re.search(r"^domain\s*:", fm_body, re.MULTILINE):
        return content
    new_line = f"domain: {domain}"
    rewritten = f"{fm_body}\n{new_line}"
    return f"{open_delim}{rewritten}{close_delim}{content[fm_match.end():]}"


# ---------------------------------------------------------------------------
# Sources canonicalization
# ---------------------------------------------------------------------------

def _canonicalize_sources_field(content: str, source_identity: str) -> str:
    if not re.match(r"^---\n", content):
        return content
    identity_key = normalize_path(source_identity).lower()
    identity_basename = get_file_name(source_identity).lower()
    source_values = parse_sources(content)
    canonical: list[str] = []
    for source in source_values:
        normalized = normalize_path(source)
        key = normalized.lower()
        if key == identity_key:
            canonical.append(source_identity)
        elif "/" not in normalized and get_file_name(source).lower() == identity_basename:
            canonical.append(source_identity)
        else:
            canonical.append(source)
    if not any(normalize_path(s).lower() == identity_key for s in canonical):
        canonical.append(source_identity)
    seen: set[str] = set()
    deduped: list[str] = []
    for s in canonical:
        key = normalize_path(s).lower()
        if key in seen:
            continue
        seen.add(key)
        deduped.append(s)
    return write_sources(content, deduped)


# ---------------------------------------------------------------------------
# Page merge (page-merge.ts)
# ---------------------------------------------------------------------------

def merge_page_content(
    new_content: str,
    existing_content: str | None,
    llm_config: LlmConfig,
    source_file_name: str,
    page_path: str,
) -> str:
    """Merge a newly generated page with an existing on-disk version."""
    UNION_FIELDS = ["sources", "tags", "related"]
    LOCKED_FIELDS = ["type", "title", "created"]
    BODY_SHRINK_THRESHOLD = 0.7

    if not existing_content:
        return new_content
    if new_content == existing_content:
        return existing_content

    array_merged = merge_array_fields_into_content(new_content, existing_content, list(UNION_FIELDS))

    old_fm, old_body = extract_frontmatter(existing_content)
    array_merged_fm, array_merged_body = extract_frontmatter(array_merged)

    if old_body.strip() == array_merged_body.strip():
        return array_merged

    system_prompt, user_message = build_page_merge_prompt(existing_content, array_merged, source_file_name)
    result = ""
    error: Exception | None = None

    def on_token(token: str) -> None:
        nonlocal result
        result += token

    def on_done() -> None:
        pass

    def on_error(e: Exception) -> None:
        nonlocal error
        error = e

    stream_chat(
        llm_config,
        [{"role": "system", "content": system_prompt}, {"role": "user", "content": user_message}],
        on_token,
        on_done,
        on_error,
        max_tokens=4096,
        temperature=0.1,
    )

    if error:
        logger.warning("LLM merge failed for %s, falling back: %s", page_path, error)
        return array_merged

    llm_fm, llm_body = extract_frontmatter(result)
    if not llm_fm:
        logger.warning("LLM merge output for %s has no frontmatter — falling back", page_path)
        return array_merged

    min_threshold = max(len(old_body), len(array_merged_body)) * BODY_SHRINK_THRESHOLD
    if len(llm_body) < min_threshold:
        logger.warning(
            "LLM merge for %s produced body %d chars, below threshold %d — falling back",
            page_path,
            len(llm_body),
            int(min_threshold),
        )
        return array_merged

    final = result
    for field in LOCKED_FIELDS:
        existing_value = old_fm.get(field)
        if isinstance(existing_value, str) and existing_value != "":
            final = _set_frontmatter_scalar(final, field, existing_value)

    final = merge_array_fields_into_content(final, array_merged, list(UNION_FIELDS))
    final = _set_frontmatter_scalar(final, "updated", datetime.date.today().isoformat())
    return final


# ---------------------------------------------------------------------------
# Write file blocks
# ---------------------------------------------------------------------------

def write_file_blocks(
    project_path: str,
    text: str,
    llm_config: LlmConfig,
    source_file_name: str,
    source_summary_path: str = "",
    domain: str = "",
) -> tuple[list[str], list[str], list[str]]:
    """Parse FILE blocks from generation output and write them to disk.

    Returns (written_paths, warnings, hard_failures).
    """
    parse_result = parse_file_blocks(text)
    warnings = list(parse_result.warnings)
    written_paths: list[str] = []
    hard_failures: list[str] = []

    for block in parse_result.blocks:
        relative_path = block.path
        if source_summary_path and relative_path.startswith("wiki/sources/"):
            relative_path = source_summary_path

        content = sanitize_ingested_file_content(block.content)
        if not _is_log_path(relative_path) and not _is_listing_path(relative_path):
            content = _canonicalize_sources_field(content, source_file_name)
        if domain and not _is_log_path(relative_path):
            content = _inject_domain(content, domain)

        full_path = f"{project_path}/{relative_path}"
        try:
            if _is_log_path(relative_path):
                existing = _try_read_file(full_path)
                appended = f"{existing}\n\n{content.strip()}" if existing else content.strip()
                write_file_utf8(full_path, appended)
            elif _is_listing_path(relative_path):
                write_file_utf8(full_path, content)
            else:
                existing = _try_read_file(full_path)
                if existing:
                    content = merge_page_content(
                        content, existing, llm_config, source_file_name, relative_path
                    )
                write_file_utf8(full_path, content)
            written_paths.append(relative_path)
        except Exception as e:
            msg = f'Failed to write "{relative_path}": {e}'
            logger.error(msg)
            warnings.append(msg)
            hard_failures.append(relative_path)

    return written_paths, warnings, hard_failures


# ---------------------------------------------------------------------------
# Review block parser
# ---------------------------------------------------------------------------

_REVIEW_BLOCK_REGEX = re.compile(
    r"---REVIEW:\s*(\w[\w-]*)\s*\|\s*(.+?)\s*---\n([\s\S]*?)---END REVIEW---"
)


def parse_review_blocks(text: str, source_path: str) -> list[dict]:
    """Parse REVIEW blocks from generation output."""
    items: list[dict] = []
    for match in _REVIEW_BLOCK_REGEX.finditer(text):
        raw_type = match.group(1).strip().lower()
        title = match.group(2).strip()
        body = match.group(3).strip()
        review_type = (
            raw_type
            if raw_type in ("contradiction", "duplicate", "missing-page", "suggestion")
            else "confirm"
        )
        options_match = re.search(r"^OPTIONS:\s*(.+)$", body, re.MULTILINE)
        options: list[dict] = []
        if options_match:
            options = [{"label": o.strip(), "action": o.strip()} for o in options_match.group(1).split("|")]
        else:
            options = [{"label": "Approve", "action": "Approve"}, {"label": "Skip", "action": "Skip"}]
        pages_match = re.search(r"^PAGES:\s*(.+)$", body, re.MULTILINE)
        affected_pages = [p.strip() for p in pages_match.group(1).split(",")] if pages_match else None
        search_match = re.search(r"^SEARCH:\s*(.+)$", body, re.MULTILINE)
        search_queries = [q.strip() for q in search_match.group(1).split("|") if q.strip()] if search_match else None
        description = re.sub(r"^OPTIONS:.*$", "", body, flags=re.MULTILINE)
        description = re.sub(r"^PAGES:.*$", "", description, flags=re.MULTILINE)
        description = re.sub(r"^SEARCH:.*$", "", description, flags=re.MULTILINE).strip()
        items.append(
            {
                "type": review_type,
                "title": title,
                "description": description,
                "source_path": source_path,
                "affected_pages": affected_pages,
                "search_queries": search_queries,
                "options": options,
            }
        )
    return items


# ---------------------------------------------------------------------------
# Auto ingest orchestrator
# ---------------------------------------------------------------------------

def _try_read_file(path: str) -> str:
    try:
        return read_file_utf8(path)
    except Exception:
        return ""


def auto_ingest(
    project_path: str,
    source_path: str,
    llm_config: LlmConfig,
    source_content: str = "",
    domain: str = "",
    folder_context: str = "",
) -> list[str]:
    """Two-step auto-ingest: Analysis -> Generation -> Write files.

    Args:
        project_path: Root path of the Wiki project.
        source_path: Path to the source file (used for identity).
        llm_config: LLM configuration.
        source_content: Optional pre-loaded source content. If empty, reads from disk.
        domain: Domain tag to inject into generated pages.
        folder_context: Optional folder context hint for categorization.

    Returns:
        List of relative paths written.
    """
    pp = normalize_path(project_path)
    sp = normalize_path(source_path)
    file_name = get_file_name(sp)
    source_identity = source_identity_for_path(pp, sp)
    source_summary_slug = source_summary_slug_from_identity(source_identity)
    source_summary_path = f"wiki/sources/{source_summary_slug}.md"

    source_content = source_content or _try_read_file(sp)
    schema = _try_read_file(f"{pp}/schema.md")
    purpose = _try_read_file(f"{pp}/purpose.md")
    index = _try_read_file(f"{pp}/wiki/index.md")
    overview = _try_read_file(f"{pp}/wiki/overview.md")

    # Cache check
    cached_files = check_ingest_cache(pp, source_identity, source_content)
    if cached_files is not None:
        logger.info("Ingest cache hit for %s (%d files)", source_identity, len(cached_files))
        return cached_files

    # Truncate
    truncated = source_content
    if len(truncated) > 50000:
        truncated = truncated[:50000] + "\n\n[...truncated...]"

    # Step 1: Analysis
    analysis = ""
    analysis_error: Exception | None = None

    def on_analysis_token(token: str) -> None:
        nonlocal analysis
        analysis += token

    def on_analysis_done() -> None:
        pass

    def on_analysis_error(e: Exception) -> None:
        nonlocal analysis_error
        analysis_error = e

    folder_ctx_line = f"\n**Folder context:** {folder_context}" if folder_context else ""
    stream_chat(
        llm_config,
        [
            {"role": "system", "content": build_analysis_prompt(purpose, index, truncated)},
            {
                "role": "user",
                "content": (
                    f"Analyze this source document:\n\n**File:** {source_identity}"
                    f"{folder_ctx_line}"
                    f"\n\n---\n\n{truncated}"
                ),
            },
        ],
        on_analysis_token,
        on_analysis_done,
        on_analysis_error,
        max_tokens=4096,
        temperature=0.1,
    )
    if analysis_error:
        raise analysis_error

    # Step 2: Generation
    generation = ""
    generation_error: Exception | None = None

    def on_generation_token(token: str) -> None:
        nonlocal generation
        generation += token

    def on_generation_done() -> None:
        pass

    def on_generation_error(e: Exception) -> None:
        nonlocal generation_error
        generation_error = e

    stream_chat(
        llm_config,
        [
            {
                "role": "system",
                "content": build_generation_prompt(
                    schema, purpose, index, source_identity, overview, truncated, source_summary_path
                ),
            },
            {
                "role": "user",
                "content": (
                    f"Source document to process: **{source_identity}**\n\n"
                    "The Stage 1 analysis below is CONTEXT to inform your output. Do NOT echo "
                    "its tables, bullet points, or prose. Your output must be FILE/REVIEW "
                    "blocks as specified in the system prompt — nothing else.\n\n"
                    "## Stage 1 Analysis (context only — do not repeat)\n\n"
                    f"{analysis}\n\n"
                    "## Original Source Content\n\n"
                    f"{truncated}\n\n"
                    "---\n\n"
                    f"Now emit the FILE blocks for the wiki files derived from **{source_identity}**. "
                    "Your response MUST begin with `---FILE:` as the very first characters. "
                    "No preamble. No analysis prose. Start immediately."
                ),
            },
        ],
        on_generation_token,
        on_generation_done,
        on_generation_error,
        max_tokens=8192,
        temperature=0.1,
    )
    if generation_error:
        raise generation_error

    # Step 3: Write files
    written_paths, write_warnings, hard_failures = write_file_blocks(
        pp, generation, llm_config, source_identity, source_summary_path, domain
    )
    if write_warnings:
        logger.warning("Ingest write warnings: %s", write_warnings)

    # Ensure source summary page exists
    source_summary_full_path = f"{pp}/{source_summary_path}"
    has_source_summary = any(normalize_path(p) == source_summary_path for p in written_paths)
    if not has_source_summary:
        date = datetime.date.today().isoformat()
        fallback = (
            f"---\n"
            f"type: source\n"
            f'title: "Source: {source_identity}"\n'
            f"created: {date}\n"
            f"updated: {date}\n"
            f'sources: ["{source_identity}"]\n'
            f"tags: []\n"
            f"related: []\n"
            f"---\n"
            f"\n"
            f"# Source: {source_identity}\n"
            f"\n"
            f"{analysis[:3000] if analysis else '(Analysis not available)'}\n"
        )
        if domain:
            fallback = _inject_domain(fallback, domain)
        try:
            write_file_utf8(source_summary_full_path, fallback)
            written_paths.append(source_summary_path)
        except Exception:
            pass

    # Save cache
    if written_paths and not hard_failures:
        save_ingest_cache(pp, source_identity, source_content, written_paths)
    elif hard_failures:
        logger.warning(
            "Skipping cache save for %s — %d block(s) failed to write: %s",
            source_identity,
            len(hard_failures),
            ", ".join(hard_failures),
        )

    return written_paths
