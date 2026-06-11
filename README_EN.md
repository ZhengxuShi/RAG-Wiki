# LLM Wiki × RAG: Structured Knowledge Retrieval-Augmented Generation

> Based on Andrej Karpathy's LLM Wiki methodology, combined with the engineering implementation of [nashsu/llm_wiki](https://github.com/nashsu/llm_wiki), we ported reusable modules to Python and adapted them to serve as a complementary retrieval path alongside traditional RAG, integrated into a Haystack-based retrieval pipeline.

[![](https://img.shields.io/badge/Python-3.10+-3776AB?logo=python&logoColor=white)](https://www.python.org/)
[![](https://img.shields.io/badge/FastAPI-009688?logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com/)
[![](https://img.shields.io/badge/Milvus-00A1EA?logoColor=white)](https://milvus.io/)
[![](https://img.shields.io/badge/Haystack-2.14+-F7DF1E?logoColor=black)](https://haystack.deepset.ai/)
[![](https://img.shields.io/badge/OpenAI_API-412991?logo=openai&logoColor=white)](https://openai.com/)
[![](https://img.shields.io/badge/PyTorch-EE4C2C?logo=pytorch&logoColor=white)](https://pytorch.org/)
[![](https://img.shields.io/badge/Docker-2496ED?logo=docker&logoColor=white)](https://www.docker.com/)
[![](https://img.shields.io/badge/Cytoscape.js-225C6E?logoColor=white)](https://js.cytoscape.org/)

---

## 1. Project Overview

This project is a **RAG (Retrieval-Augmented Generation)** framework. Its core capability extends beyond traditional dense + sparse vector hybrid retrieval by introducing a complete **LLM Wiki knowledge graph retrieval path**. Results from both paths are fused via **RRF (Reciprocal Rank Fusion)** and then uniformly reranked by an external Reranker before being sent to the LLM for answer generation.

**Core Positioning**:
- **RAG Path**: Responsible for precisely recalling technical details from raw document chunks (parameters, commands, specification clauses)
- **Wiki Path**: Responsible for recalling LLM-pre-structured knowledge, entity relationships, and cross-document concept associations
- **Fusion Mechanism**: Both paths run independently and complement each other, preventing either path's recall blind spots from dominating the final answer

---

## 2. Core Ideas of LLM Wiki

### 2.1 Essential Differences from Traditional RAG

Traditional RAG follows a "re-derive on every query" paradigm: User Query → Embedding → Vector Retrieval → Chunk Concatenation → LLM Generation. Its limitations include:
- Vector similarity cannot discover **cross-document systematic associations** (e.g., Dolby Vision's multi-faceted exposition across firmware specifications, certification standards, and principle documents)
- Returns are **fragmented chunks**, lacking LLM-pre-distilled concept hierarchies and entity relationships
- Insufficient coverage of **synonyms, abbreviations, and model aliases**

The core idea of LLM Wiki originates from Andrej Karpathy's [LLM Wiki methodology](https://gist.github.com/karpathy/442a6bf555914893e9891c11519de94f):

> **Knowledge is compiled once; queries consume already-organized knowledge.**

Specifically:
1. **Knowledge Pre-compilation**: During the offline phase, the LLM transforms raw documents into structured, interrelated Wiki pages (Markdown + YAML Frontmatter)
2. **Explicit Associations**: Through `[[wikilink]]` bidirectional links and source tracing, a traversable knowledge graph is constructed
3. **Direct Read at Query Time**: Instead of repeating Embedding + vector retrieval, relevant Wiki pages are recalled directly via keyword matching + graph traversal

### 2.2 Inheritance and Adaptation from the Original Implementation

This project references the desktop application implementation of [nashsu/llm_wiki](https://github.com/nashsu/llm_wiki) (TypeScript + Tauri + Rust backend) and performed **Python porting and engineering adaptation** on its core retrieval modules:

| Module | Original Implementation | Adaptation in This Project |
|--------|------------------------|---------------------------|
| **Four-Signal Relevance Model** | TypeScript (`graph-relevance.ts`) | Ported to Python, preserving signal definitions and weights |
| **Tokenizer** | Rust backend (`search.rs`) | Ported to Python, adding CJK/non-CJK boundary awareness |
| **Retrieval Pipeline** | Frontend-driven, Rust local HTTP API | Decoupled into independent `WikiSearcher`, running in parallel with Haystack RAG Pipeline |
| **Graph Expansion Scoring** | `base_score + relevance × 0.9` | Adjusted to weighted average to suppress Hub node dominance |
| **Fusion Strategy** | Simple frontend concatenation | Backend RRF fusion + external Reranker unified reranking |

---

## 3. System Architecture

### 3.1 Dual-Path Retrieval + Fusion Reranking Pipeline

```
User Query
    │
    ├──► RAG Pipeline ───────┐
    │   (File Routing → Query Rewrite │
    │    → Embedding → Milvus          │
    │    → Hybrid Retrieval)          │
    │                                 ├──► RRF Fusion ──► External Reranker ──► Retain ──► LLM Generation
    └──► WikiSearcher ───────────────┘
        (Query Tokenization → Keyword Matching
         → 4-Signal Graph Expansion → 2-Hop Expansion)
```

**Phase Descriptions**:

1. **Dual-Path Independent Retrieval**
   - **RAG Path**: Maintains the classic Haystack Pipeline with dense vector (bge-m3/gte) + sparse vector (BM42) hybrid retrieval
   - **Wiki Path**: `WikiSearcher` runs independently, not relying on RAG intermediate results, starting directly from Query tokenization

2. **RRF Fusion**
   - Computes Reciprocal Rank Fusion scores separately for RAG and Wiki results
   - Deduplicates by the first 200 characters of content, preserving original Document metadata

3. **External Reranker**
   - The internal Reranker is removed from the Pipeline when Wiki is enabled, avoiding conflicts
   - The fused full document list is semantically reranked by an external BGE-Reranker / OpenAI Reranker

4. **Retain Count Truncation**
   - Finally truncated by the configured `retain_count` (default 20~30) to control LLM context length

### 3.2 Four-Signal Relevance Model

The core algorithm for graph expansion, computing association strength between two Wiki nodes:

| Signal | Weight | Computation |
|--------|--------|-------------|
| **Source Overlap** | ×4.0 | Intersection count of `sources[]` arrays in YAML frontmatter |
| **Direct Link** | ×3.0 | Bidirectional `[[wikilink]]` link relationship |
| **Adamic-Adar** | ×1.5 | Shared neighbor nodes, weighted by neighbor degree (higher degree = lower contribution) |
| **Type Affinity** | ×1.0 | Predefined page type affinity matrix (entity / concept / source / query / synthesis) |

### 3.3 2-Hop Graph Expansion and Score Propagation

1. **Seed Selection**: Keyword matching on the full graph, scoring nodes, taking Top-K as seeds
2. **1st Hop Expansion**: Using seeds as centers, finding associated nodes via the four-signal model
3. **2nd Hop Expansion**: Using 1st-hop nodes as the new frontier, discovering indirect associations
4. **Score Decay**: Expansion node score = `base_score × 0.5 + relevance × 0.3`, preventing generic index pages (like index.md) from monopolizing high scores due to numerous links

**Key Adaptation**: The original formula `base_score + relevance × 0.9` caused Hub node dominance. After changing to weighted average, professional concept pages (e.g., Dolby Vision) rose from "not in top 10" to rank 1.

---

## 4. Core Features

- **Dual-Path Independent Retrieval**: RAG + Wiki run independently, with Query driving both paths simultaneously and results complementing each other
- **Four-Signal Knowledge Graph**: Source Overlap / Direct Link / Adamic-Adar / Type Affinity
- **2-Hop Graph Expansion**: Supports multi-hop decay, discovering deep implicit associations
- **CJK/Non-CJK Boundary-Aware Tokenizer**: Correctly splits Chinese-English mixed Queries, avoiding noise like `"vi"`, `"is"` from `"vision的自验证"`
- **Frontend Wikipedia-Style Interaction**: Node detail modal, 2-Hop expansion trigger, breadcrumb path navigation, Cytoscape knowledge graph preview
- **Source Classification Statistics**: Automatically distinguishes RAG sources from Wiki sources, displaying `Total N sources (RAG: X, Wiki: Y)` in real time

## Demo

| Retrieval Q&A Interface | Wikipedia-Style Node Expansion |
|:-----------------------:|:------------------------------:|
| ![Demo Search](assets/demo-search.png) | ![Demo Graph](assets/demo-graph.png) |



---

## 5. Quick Start

### 5.1 Requirements

- Python 3.10+
- Milvus vector database (local or remote)
- Optional: OpenAI-compatible API (for Embedding, Rerank, LLM generation)

### 5.1.1 Milvus Setup

This project uses Milvus to store document vectors. The simplest way to start it:

```bash
docker run -d --name milvus-standalone \
  -p 19530:19530 \
  -p 9091:9091 \
  milvusdb/milvus:latest \
  milvus run standalone
```

Before the first run, create the database and collection. See `scripts/init_milvus.py` (creates the `Firmware` database and `bge_m3_wiki_demo100_v1` collection with 1024-dimensional vectors).

> If you use an existing Milvus instance, update `milvus_db_name` and `milvus_collection_name` in `configs/retriever_config_v3.json`.

### 5.1.2 External Service Setup

This project depends on multiple OpenAI-compatible services. We recommend deploying them locally via [Xinference](https://github.com/xorbitsai/inference) or [Ollama](https://ollama.com):

| Service | Recommended Port | Description |
|---------|------------------|-------------|
| Dense Embedding | `8001` | e.g., `bge-m3` |
| Sparse Embedding | `8002` | e.g., `BM42` |
| Reranker | `8003` | e.g., `bge-reranker-v2-m3` |
| Tokenizer | `8004` | Tokenizer for bge-m3 / reranker |
| LLM | `8005` | e.g., `qwen3-max` |

Quick start with Xinference:
```bash
xinference launch --model-name bge-m3 --model-type embedding --port 8001
xinference launch --model-name bge-reranker-v2-m3 --model-type rerank --port 8003
```

If you use cloud APIs such as DashScope or OpenAI, simply fill in the corresponding URLs in `.env` without local deployment.

### 5.2 Install Dependencies

```bash
pip install -r requirements-retriever.txt
```

### 5.3 Configure Environment Variables

Copy `.env.example` to `.env` and fill in as needed:

```bash
cp .env.example .env
# Edit .env, fill in your API keys and service addresses
```

### 5.4 Configure Retrieval Parameters

Edit `configs/retriever_config_v3.json`, replacing placeholders with your actual service addresses:

| Placeholder | Description |
|-------------|-------------|
| `YOUR_MILVUS_HOST:19530` | Milvus vector database address |
| `YOUR_EMBEDDING_API/v1` | Dense Embedding service (OpenAI-compatible) |
| `YOUR_RERANKER_API/v1` | Reranker service (OpenAI-compatible) |
| `YOUR_LLM_API/v1` | LLM generation service (OpenAI-compatible) |

### 5.5 Start Services (Mind the Dependency Order)

```
1. Milvus (docker or remote instance)
   ↓
2. External Embedding / Reranker / LLM services (or cloud APIs)
   ↓
3. Retrieval service api.py (port 17101)
   ↓
4. Aggregation gateway api_gateway.py (port 8000, includes frontend UI)
```

```bash
# 3. Start retrieval service (ensure Milvus and external services are ready)
python api.py

# 4. Start aggregation gateway (ensure api.py is already running)
python api_gateway.py
```

The frontend UI is accessible at `http://localhost:8000/` by default.

> `api_gateway.py` depends on `api.py` for retrieval endpoints (`RAG_URL` / `WIKI_URL`). If api.py is not running, gateway queries will fail.

---

## 6. Wiki Knowledge Base Construction

### 6.1 Two-Step Ingest Pipeline

The core of LLM Wiki is **transforming raw documents into structured Wiki pages during the offline phase**:

1. **Analysis**: The LLM extracts key entities, concepts, core arguments, and identifies associations/contradictions with existing Wiki content
2. **Generation**: Generates source summary pages with YAML frontmatter, entity pages, concept pages, and updates the global index

### 6.2 Directory Structure

```
llm_wiki/wiki/
├── index.md              # Content index, LLM navigation entry
├── concepts/             # Theories, methods, technologies
├── entities/             # People, organizations, products
└── sources/              # Source summaries
```

Each Wiki page must include YAML frontmatter:

```yaml
---
type: concept | entity | source | query | synthesis | comparison
title: "Page Title"
created: "YYYY-MM-DD"
tags: [tag1, tag2]
sources: ["original-file-name.pdf"]
---
```

### 6.3 Quick Example Wiki

The project includes 4 example pages (`index.md`, `example-concept.md`, `example-entity.md`, `example-source.md`) for directly experiencing Wiki retrieval effects.

To build your own Wiki:
1. Place raw documents in the project directory
2. Call the two-step ingest interface in `ingest.py`, or manually write Markdown pages
3. Ensure pages contain `[[wikilink]]` links and correct frontmatter
4. Restart the service or call the graph cache refresh interface

### 6.4 Automatic Build with `ingest.py` (Recommended)

Before running the ingest script, you must set the required environment variables:

```bash
export OPENAI_API_KEY="your-openai-key"
export KB_BUILD_KNOWLEDGE_UUID="your-knowledge-uuid"   # Required, used for Milvus multi-tenancy
```

Then call the ingest interface:

```python
from llm_wiki.ingest import ingest_documents

# Analysis phase
analyze_results = ingest_documents(
    raw_docs_dir="./your_documents",
    phase="analyze"
)

# Generation phase
ingest_documents(
    raw_docs_dir="./your_documents",
    phase="generate"
)
```

After ingestion, trigger a Wiki graph cache refresh (or restart the service) for the new pages to take effect.

> `KB_BUILD_KNOWLEDGE_UUID` is the unique knowledge base identifier in the Milvus collection. Different document sets should use different UUIDs. Make sure it is set in `.env` before the first run.

---

## 7. Configuration

### 7.1 Environment Variables (`.env`)

| Variable | Required | Description |
|----------|----------|-------------|
| `OPENAI_API_KEY` | Yes | OpenAI API Key (for Embedding / Rerank / Wiki Ingest) |
| `DASHSCOPE_API_KEY` | No | DashScope API Key (for LLM generation) |
| `MINICPM_API_KEY` | No | MiniCPM-V API Key (for image understanding) |
| `MILVUS_URI` | Yes | Milvus connection URI, default `http://localhost:19530` |
| `MILVUS_DB_NAME` | No | Milvus database name, default `default` |
| `EMBEDDING_URL` | Yes | Dense embedding service URL |
| `SPARSE_EMBEDDING_URL` | Yes | Sparse embedding service URL (BM42) |
| `RERANK_URL` | Yes | Reranker service URL |
| `LLM_URL` | Yes | LLM generation service URL |
| `LLM_MODEL` | No | Default LLM model name, default `qwen3-max` |
| `RAG_URL` | No | RAG service address for api_gateway, default `http://127.0.0.1:17101` |
| `WIKI_URL` | No | Wiki-enhanced retrieval address for api_gateway, default `http://127.0.0.1:17101` |
| `MINICPM_API_URL` | No | MiniCPM-V service URL, default `http://localhost:8008/v1` |
| `MINICPM_MODEL` | No | MiniCPM-V model name, default `openbmb/MiniCPM-V-4.6` |
| `MINICPM_TIMEOUT` | No | MiniCPM-V timeout (seconds), default `30` |
| `OPENAI_TIMEOUT` | No | OpenAI-compatible API timeout (seconds), default `30.0` |
| `OPENAI_MAX_RETRIES` | No | OpenAI-compatible API max retries, default `5` |
| `KB_BUILD_KNOWLEDGE_UUID` | No | Wiki knowledge base UUID, required only when running `ingest.py` |
| `CUDA_VISIBLE_DEVICES` | No | GPU device index, `-1` means CPU |

### 7.2 Retrieval Config (`retriever_config_v3.json`)

`configs/retriever_config_v3.json` is a Hypster-style hyperparameter config file. Before the first use, replace the following placeholders with your actual addresses:

| Placeholder | Description |
|-------------|-------------|
| `YOUR_MILVUS_HOST:19530` | Milvus vector database address |
| `YOUR_EMBEDDING_API/v1` | Dense embedding service (OpenAI-compatible) |
| `YOUR_SPARSE_EMBEDDING_API/v1` | Sparse embedding service (BM42, OpenAI-compatible) |
| `YOUR_RERANKER_API/v1` | Reranker service (OpenAI-compatible) |
| `YOUR_TOKENIZE_API/v1/tokenize` | Tokenizer service address |
| `YOUR_LLM_API/v1` | LLM generation service (OpenAI-compatible) |

Core configuration nodes:

- `1000.1st_retriever`: First-level retriever parameters (Embedding model, Top-K, hybrid retrieval config)
- `1003.use_reranker`: Reranker parameters (model type, Top-K, API address)
- `1006.use_compressed`: Result compression parameters (retain_count)
- `1007.llm_wiki`: Wiki configuration (expansion hops, per-hop limit, relevance threshold)

> If you use local Xinference / Ollama deployment, all `YOUR_*` placeholders can be replaced with `http://localhost:PORT/v1`.

### 7.3 Wiki Configuration Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `wiki_search_top_k` | 20 | Number of seed nodes for keyword matching |
| `expansion_hops` | 2 | Graph expansion hop count |
| `expansion_limit` | 3 | Maximum neighbors per hop |
| `expansion_min_relevance` | 2.0 | Expansion node relevance threshold; discard below |

---

## 8. Troubleshooting

See [TROUBLESHOOTING.md](TROUBLESHOOTING.md) for common issues and solutions.

---

## 9. Project Structure

```
.
├── README.md
├── LICENSE
├── .env.example
├── .gitignore
├── requirements-retriever.txt
├── api.py                      # FastAPI retrieval service (/search, /search_with_wiki)
├── api_gateway.py              # Aggregation gateway (with frontend UI & streaming LLM generation)
├── main.py                     # RAG Pipeline assembly example
├── vision_core.py              # Multimodal image understanding (MiniCPM-V)
├── static/                     # Frontend assets (HTML / CSS / JS)
│   ├── index.html
│   ├── css/
│   └── js/
├── retriever/                  # Core retrieval module (Haystack Pipeline)
│   ├── retriever.py            # RetrieverModule, Knowledge_Base
│   ├── configs/                # Hypster hyperparameter configs
│   │   ├── reranker.py
│   │   ├── query.py
│   │   ├── indexing.py
│   │   ├── milvus_retrieval.py
│   │   └── files_routing.py
│   └── src/
│       ├── haystack_utils.py   # Custom Haystack Components
│       └── utils.py            # Milvus query wrappers
├── llm/                        # LLM generation module
│   ├── llm.py
│   └── src/
│       └── generator.py
├── llm_wiki/                   # LLM Wiki retrieval module
│   ├── wiki_searcher.py        # Independent Wiki keyword + graph expansion search
│   ├── wiki_tokenizer.py       # Query tokenizer (CJK-adapted)
│   ├── wiki_fusion.py          # RRF fusion
│   ├── graph_relevance.py      # Four-signal relevance model + graph building
│   ├── wiki_models.py          # Data models
│   ├── wiki_utils.py           # Utility functions
│   ├── ingest.py               # Two-step ingest pipeline
│   ├── wiki_prompts.py         # Ingest prompt templates
│   └── wiki/                   # Wiki knowledge base directory
│       ├── index.md
│       ├── concepts/
│       ├── entities/
│       └── sources/
├── configs/                    # Runtime configuration files
│   └── retriever_config_v3.json
├── assets/                     # README image assets

```

---

## 10. Acknowledgements

- **Andrej Karpathy** proposed the original LLM Wiki methodology, providing groundbreaking ideas for structured knowledge management.
- **[nashsu/llm_wiki](https://github.com/nashsu/llm_wiki)** provided a complete desktop application implementation reference, whose two-step ingest, four-signal model, and graph insights designs served as important engineering blueprints for this project.

---

## 11. License

This project is licensed under the [MIT License](LICENSE).
