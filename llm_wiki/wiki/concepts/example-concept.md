---
type: concept
title: "Example Concept"
created: "2026-06-10"
tags: [example, demo]
sources: ["example-spec-v1.pdf"]
---

# Example Concept

This is a **demonstration concept page** showing the structure of an LLM Wiki concept node.

## Overview

A concept in LLM Wiki represents a theoretical, methodological, or technical idea that may span multiple source documents. Unlike raw document chunks, concept pages are **synthesized by the LLM** during the ingest phase, capturing the essence of the idea with explicit links to related entities and sources.

## Key Characteristics

- **Type**: `concept` (one of: entity, concept, source, query, synthesis, comparison)
- **Sources**: Tracked in the YAML frontmatter, used for the `Source Overlap` signal in the four-signal relevance model
- **Links**: `[[wikilink]]` syntax creates explicit connections to other Wiki pages, forming the knowledge graph

## Related Topics

- [[example-entity]] — Related entity that implements this concept
- [[example-source]] — Source document where this concept was first introduced

## Ingest Note

This page was generated automatically by the two-step ingest pipeline: first the LLM analyzed the source material, then it generated this structured summary with frontmatter metadata.
