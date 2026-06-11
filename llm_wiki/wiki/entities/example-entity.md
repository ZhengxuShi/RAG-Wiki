---
type: entity
title: "Example Entity"
created: "2026-06-10"
tags: [example, demo]
sources: ["example-spec-v1.pdf"]
---

# Example Entity

This is a **demonstration entity page** showing the structure of an LLM Wiki entity node.

## Definition

An entity in LLM Wiki typically represents a concrete object: a product, a chip model, a protocol version, a standard identifier, or a vendor name. Entities are the "nouns" of the knowledge graph.

## Properties

| Property | Value |
|----------|-------|
| Type | Entity |
| Category | Example / Demo |
| First Seen | 2026-06-10 |

## Relationships

- **Implements**: [[example-concept]] — This entity is a concrete instance of the example concept
- **Defined in**: [[example-source]] — Source document providing the authoritative definition

## Graph Relevance

When searching for "Example Entity", the four-signal model will boost this node if:
- The query shares the same `source` (Source Overlap ×4.0)
- Other nodes link to or from this entity (Direct Link ×3.0)
- This entity shares common neighbors with the query-matched node (Adamic-Adar ×1.5)
- The query-matched node is also an `entity` or a `concept` (Type Affinity)
