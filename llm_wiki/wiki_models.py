"""
Pydantic models for LLM Wiki Python port.
Derived from src/types/wiki.ts and src/stores/wiki-store.ts
"""

from typing import Any, Dict, List, Optional, Set
from pydantic import BaseModel, Field


class FileNode(BaseModel):
    name: str
    path: str
    is_dir: bool
    children: Optional[List["FileNode"]] = None


class WikiPage(BaseModel):
    path: str
    content: str
    frontmatter: Dict[str, Any] = Field(default_factory=dict)


class ParsedFileBlock(BaseModel):
    path: str
    content: str


class ParseFileBlocksResult(BaseModel):
    blocks: List[ParsedFileBlock] = Field(default_factory=list)
    warnings: List[str] = Field(default_factory=list)


class LlmConfig(BaseModel):
    provider: str = "openai"  # openai | ollama
    api_key: str = ""
    model: str = "gpt-4o"
    base_url: str = ""
    max_context_size: int = 204800
    temperature: float = 0.1


class WikiProject(BaseModel):
    id: str
    name: str
    path: str


class IngestTask(BaseModel):
    id: str
    project_id: str
    source_path: str
    status: str = "pending"  # pending | processing | done | failed
    added_at: float = 0.0
    error: Optional[str] = None
    retry_count: int = 0


class GraphNode(BaseModel):
    id: str
    label: str
    type: str
    path: str
    link_count: int = 0
    community: int = 0


class GraphEdge(BaseModel):
    source: str
    target: str
    weight: float = 1.0


class CommunityInfo(BaseModel):
    id: int
    node_count: int
    cohesion: float
    top_nodes: List[str] = Field(default_factory=list)


class RetrievalNode(BaseModel):
    id: str
    title: str
    type: str
    path: str
    sources: List[str] = Field(default_factory=list)
    out_links: Set[str] = Field(default_factory=set)
    in_links: Set[str] = Field(default_factory=set)


class RetrievalGraph(BaseModel):
    nodes: Dict[str, RetrievalNode] = Field(default_factory=dict)
    data_version: int = 0


class ContextBudget(BaseModel):
    max_ctx: int
    response_reserve: int
    index_budget: int
    page_budget: int
    max_page_size: int
