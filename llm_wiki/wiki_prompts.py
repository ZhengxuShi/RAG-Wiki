"""
Prompt builders for LLM Wiki two-step ingest.
Derived from src/lib/ingest.ts (buildAnalysisPrompt, buildGenerationPrompt)
"""

from llm_wiki.wiki_page_types import GENERATION_WIKI_TYPES


def _language_rule(source_content: str = "") -> str:
    """Simplified language directive. Defaults to the language of the source content or English."""
    # Simple heuristic: if significant CJK chars present, assume Chinese
    cjk_count = sum(1 for ch in source_content if "\u4e00" <= ch <= "\u9fff")
    if cjk_count > 50:
        return "## Mandatory Output Language: Chinese\nYou MUST write your entire response in Chinese."
    return "## Mandatory Output Language: English\nYou MUST write your entire response in English."


def build_analysis_prompt(purpose: str, index: str, source_content: str = "") -> str:
    parts = [
        "You are an expert research analyst. Read the source document and produce a structured analysis.",
        "Do not output chain-of-thought, hidden reasoning, or a thinking transcript. Reason internally and write only the concise final analysis.",
        "",
        _language_rule(source_content),
        "",
        "Your analysis should cover:",
        "",
        "## Key Entities",
        "List people, organizations, products, datasets, tools mentioned. For each:",
        "- Name and type",
        "- Role in the source (central vs. peripheral)",
        "- Whether it likely already exists in the wiki (check the index)",
        "",
        "## Key Concepts",
        "List theories, methods, techniques, phenomena. For each:",
        "- Name and brief definition",
        "- Why it matters in this source",
        "- Whether it likely already exists in the wiki",
        "",
        "## Main Arguments & Findings",
        "- What are the core claims or results?",
        "- What evidence supports them?",
        "- How strong is the evidence?",
        "",
        "## Connections to Existing Wiki",
        "- What existing pages does this source relate to?",
        "- Does it strengthen, challenge, or extend existing knowledge?",
        "",
        "## Contradictions & Tensions",
        "- Does anything in this source conflict with existing wiki content?",
        "- Are there internal tensions or caveats?",
        "",
        "## Recommendations",
        "- What wiki pages should be created or updated?",
        "- What should be emphasized vs. de-emphasized?",
        "- Any open questions worth flagging for the user?",
        "",
        "Be thorough but concise. Focus on what's genuinely important.",
        "",
        "If a folder context is provided, use it as a hint for categorization.",
    ]
    if purpose:
        parts.append(f"## Wiki Purpose (for context)\n{purpose}")
    if index:
        parts.append(f"## Current Wiki Index (for checking existing content)\n{index}")
    return "\n".join(parts)


def build_generation_prompt(
    schema: str,
    purpose: str,
    index: str,
    source_file_name: str,
    overview: str = "",
    source_content: str = "",
    source_summary_path: str = "",
) -> str:
    source_base_name = source_file_name.rsplit(".", 1)[0] if "." in source_file_name else source_file_name
    summary_path = source_summary_path or f"wiki/sources/{source_base_name}.md"

    parts = [
        "You are a wiki maintainer. Based on the analysis provided, generate wiki files.",
        "Do not output chain-of-thought, hidden reasoning, or explanatory preamble. Reason internally and output only the requested FILE/REVIEW blocks.",
        "",
        _language_rule(source_content),
        "",
        "## IMPORTANT: Source File",
        f"The original source file is: **{source_file_name}**",
        "All wiki pages generated from this source MUST include this filename in their frontmatter `sources` field.",
        "",
        "## What to generate",
        "",
        f"1. A source summary page at **{summary_path}** (MUST use this exact path)",
        "2. Entity pages in wiki/entities/ for key entities identified in the analysis",
        "3. Concept pages in wiki/concepts/ for key concepts identified in the analysis",
        "4. An updated wiki/index.md — add new entries to existing categories, preserve all existing entries",
        "5. A log entry for wiki/log.md (just the new entry to append, format: ## [YYYY-MM-DD] ingest | Title)",
        "6. An updated wiki/overview.md — a high-level summary of what the entire wiki covers, updated to reflect the newly ingested source. This should be a comprehensive 2-5 paragraph overview of ALL topics in the wiki, not just the new source.",
        "",
        "## Frontmatter Rules (CRITICAL — parser is strict)",
        "",
        "Every page begins with a YAML frontmatter block. Format rules, in order of importance:",
        "",
        "1. The VERY FIRST line of the file MUST be exactly `---` (three hyphens, nothing else).",
        "   Do NOT wrap the file in a ```yaml ... ``` code fence.",
        "   Do NOT prefix it with a `frontmatter:` key or any other line.",
        "2. Each frontmatter line is a `key: value` pair on its own line.",
        "3. The frontmatter ends with another `---` line on its own.",
        "4. The next line after the closing `---` is the start of the page body.",
        "5. Arrays use the standard YAML inline form `[a, b, c]` (no outer brackets around each item).",
        "   Wikilinks belong in the BODY only — never write `related: [[a]], [[b]]` (invalid YAML);",
        "   write `related: [a, b]` with bare slugs.",
        "",
        "Required fields and types:",
        f"  • type     — one of: {' | '.join(GENERATION_WIKI_TYPES)}",
        "  • title    — string (quote it if it contains a colon, e.g. `title: \"Foo: Bar\"`)",
        "  • created  — date in YYYY-MM-DD form (no quotes)",
        "  • updated  — same as created",
        "  • tags     — array of bare strings: `tags: [microbiology, ai]`",
        "  • related  — array of bare wiki page slugs: `related: [foo, bar-baz]`. Do NOT include",
        "               `wiki/`, `.md`, or `[[…]]` here — slugs only.",
        f"  • sources  — array of source filenames; MUST include \"{source_file_name}\".",
        "",
        "Concrete example of a complete, parseable page:",
        "",
        "    ---",
        "    type: entity",
        "    title: Example Entity",
        "    created: 2026-04-29",
        "    updated: 2026-04-29",
        "    tags: [example, demo]",
        "    related: [related-slug-1, related-slug-2]",
        f'    sources: ["{source_file_name}"]',
        "    ---",
        "",
        "    # Example Entity",
        "",
        "    Body content goes here. Use [[wikilink]] syntax in the body for cross-references.",
        "",
        "Other rules:",
        "- Use [[wikilink]] syntax in the BODY for cross-references between pages",
        "- Use kebab-case filenames",
        "- Follow the analysis recommendations on what to emphasize",
        "- If the analysis found connections to existing pages, add cross-references",
        "",
        "## Review block types",
        "",
        "After all FILE blocks, optionally emit REVIEW blocks for anything that needs human judgment:",
        "",
        "- contradiction: the analysis found conflicts with existing wiki content",
        "- duplicate: an entity/concept might already exist under a different name in the index",
        "- missing-page: an important concept is referenced but has no dedicated page",
        "- suggestion: ideas for further research, related sources to look for, or connections worth exploring",
        "",
        "Only create reviews for things that genuinely need human input. Don't create trivial reviews.",
        "",
        "## OPTIONS allowed values (only these predefined labels):",
        "",
        "- contradiction: OPTIONS: Create Page | Skip",
        "- duplicate: OPTIONS: Create Page | Skip",
        "- missing-page: OPTIONS: Create Page | Skip",
        "- suggestion: OPTIONS: Create Page | Skip",
        "",
        "The user also has a 'Deep Research' button (auto-added by the system) that triggers web search.",
        "Do NOT invent custom option labels. Only use 'Create Page' and 'Skip'.",
        "",
        "For suggestion and missing-page reviews, the SEARCH field must contain 2-3 web search queries",
        "(keyword-rich, specific, suitable for a search engine — NOT titles or sentences). Example:",
        "  SEARCH: automated technical debt detection AI generated code | software quality metrics LLM code generation | static analysis tools agentic software development",
    ]
    if purpose:
        parts.append(f"## Wiki Purpose\n{purpose}")
    if schema:
        parts.append(f"## Wiki Schema\n{schema}")
    if index:
        parts.append(f"## Current Wiki Index (preserve all existing entries, add new ones)\n{index}")
    if overview:
        parts.append(f"## Current Overview (update this to reflect the new source)\n{overview}")

    parts.extend([
        "",
        "## Output Format (MUST FOLLOW EXACTLY — this is how the parser reads your response)",
        "",
        "Your ENTIRE response consists of FILE blocks followed by optional REVIEW blocks. Nothing else.",
        "",
        "FILE block template:",
        "```",
        "---FILE: wiki/path/to/page.md---",
        "(complete file content with YAML frontmatter)",
        "---END FILE---",
        "```",
        "",
        "REVIEW block template (optional, after all FILE blocks):",
        "```",
        "---REVIEW: type | Title---",
        "Description of what needs the user's attention.",
        "OPTIONS: Create Page | Skip",
        "PAGES: wiki/page1.md, wiki/page2.md",
        "SEARCH: query 1 | query 2 | query 3",
        "---END REVIEW---",
        "```",
        "",
        "## Output Requirements (STRICT — deviations will cause parse failure)",
        "",
        "1. The FIRST character of your response MUST be `-` (the opening of `---FILE:`).",
        "2. DO NOT output any preamble such as \"Here are the files:\", \"Based on the analysis...\", or any introductory prose.",
        "3. DO NOT echo or restate the analysis — that was stage 1's job. Your job is to emit FILE blocks.",
        "4. DO NOT output markdown tables, bullet lists, or headings outside of FILE/REVIEW blocks.",
        "5. DO NOT output any trailing commentary after the last `---END FILE---` or `---END REVIEW---`.",
        "6. Between blocks, use only blank lines — no prose.",
        "7. EVERY FILE block's content (titles, body, descriptions) MUST be in the mandatory output language specified below. No exceptions.",
        "",
        "If you start with anything other than `---FILE:`, the entire response will be discarded.",
        "",
        "---",
        "",
        _language_rule(source_content),
    ])
    return "\n".join(parts)


def build_page_merge_prompt(existing_content: str, incoming_content: str, source_file_name: str) -> str:
    system_prompt = (
        "You are merging two versions of the same wiki page into one coherent document.\n"
        "Both versions describe the same entity / concept; one is already on disk,\n"
        "the other was just generated from a different source document.\n"
        "\n"
        "Output ONE merged version that:\n"
        "- Preserves every factual claim from both versions (do not drop content)\n"
        "- Eliminates redundancy when both versions state the same fact\n"
        "- Reorganizes sections so the structure is logical for the merged topic,\n"
        "  not just a concatenation of the two inputs\n"
        "- Uses consistent markdown structure (headings, tables, lists, callouts)\n"
        "- Keeps [[wikilink]] references intact\n"
        "\n"
        "Output requirements:\n"
        "- The FIRST character of your response MUST be `-` (the opening of `---`)\n"
        "- Output the COMPLETE file: YAML frontmatter + body\n"
        "- No preamble, no analysis prose\n"
        "- The caller will overwrite `sources`/`tags`/`related`/`updated` — your job is the body and any other fields\n"
    )
    user_message = (
        f"## Existing version on disk\n\n{existing_content}\n\n---\n\n"
        f"## Newly generated version (from {source_file_name})\n\n{incoming_content}\n\n---\n\n"
        "Now output the merged file. Start with `---` on the first line."
    )
    return system_prompt, user_message
