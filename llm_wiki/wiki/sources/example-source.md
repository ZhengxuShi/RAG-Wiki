---
type: source
title: "Example Source Summary"
created: "2026-06-10"
tags: [example, demo]
sources: ["example-spec-v1.pdf"]
---

# Example Source Summary

This is a **demonstration source summary page** showing how raw documents are summarized in LLM Wiki.

## Source Information

| Field | Value |
|-------|-------|
| Original File | `example-spec-v1.pdf` |
| Ingest Date | 2026-06-10 |
| Pages | 120 |
| Format | PDF |

## Key Points

Source summary pages distill the most important information from a raw document into a concise, LLM-readable format. During the **two-step ingest pipeline**:

1. **Analysis Step**: The LLM extracts key entities, concepts, claims, and relationships from the source.
2. **Generation Step**: The LLM produces this summary page, along with individual entity and concept pages, and updates the `index.md`.

## Extracted Entities

- [[example-entity]] — A key entity identified in this source

## Related Concepts

- [[example-concept]] — A concept introduced or elaborated in this source

## Note

Replace this example content with real source summaries by placing your documents in the `raw/sources/` directory and running the ingest pipeline.
