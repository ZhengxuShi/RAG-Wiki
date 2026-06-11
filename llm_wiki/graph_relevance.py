"""
Four-signal relevance model for LLM Wiki graph retrieval.
Derived from src/lib/graph-relevance.ts
"""

import math
import os
import re

from llm_wiki.wiki_models import RetrievalGraph, RetrievalNode
from llm_wiki.wiki_utils import extract_frontmatter, normalize_path, read_file_utf8

_WIKILINK_REGEX = re.compile(r'\[\[([^\]|]+?)(?:\|[^\]]+?)?\]\]')

_WEIGHTS = {
    "direct_link": 3.0,
    "source_overlap": 4.0,
    "common_neighbor": 1.5,
    "type_affinity": 1.0,
}

_TYPE_AFFINITY = {
    "entity": {"concept": 1.2, "entity": 0.8, "source": 1.0, "synthesis": 1.0, "query": 0.8},
    "concept": {"entity": 1.2, "concept": 0.8, "source": 1.0, "synthesis": 1.2, "query": 1.0},
    "source": {"entity": 1.0, "concept": 1.0, "source": 0.5, "query": 0.8, "synthesis": 1.0},
    "query": {"concept": 1.0, "entity": 0.8, "synthesis": 1.0, "source": 0.8, "query": 0.5},
    "synthesis": {"concept": 1.2, "entity": 1.0, "source": 1.0, "query": 1.0, "synthesis": 0.8},
}

_cached_graph: RetrievalGraph | None = None


def _extract_wikilinks(content: str) -> list[str]:
    return [m.group(1).strip() for m in _WIKILINK_REGEX.finditer(content)]


def _file_name_to_id(file_name: str) -> str:
    return re.sub(r"\.md$", "", file_name)


def _resolve_target(raw: str, node_ids: set[str]) -> str | None:
    if raw in node_ids:
        return raw
    normalized = raw.lower().replace(" ", "-")
    for nid in node_ids:
        nid_lower = nid.lower()
        if nid_lower == normalized:
            return nid
        if nid_lower == raw.lower():
            return nid
        if nid_lower.replace(" ", "-") == normalized:
            return nid
    return None


def _get_neighbors(node: RetrievalNode) -> set[str]:
    neighbors = set(node.out_links)
    neighbors.update(node.in_links)
    return neighbors


def _get_node_degree(node: RetrievalNode) -> int:
    return len(node.out_links) + len(node.in_links)


def build_retrieval_graph(project_path: str, data_version: int = 0) -> RetrievalGraph:
    """Build or return cached retrieval graph from wiki/ markdown files."""
    global _cached_graph
    if _cached_graph is not None and _cached_graph.data_version == data_version:
        return _cached_graph

    wiki_root = f"{normalize_path(project_path)}/wiki"
    md_files: list[str] = []
    try:
        for root, _dirs, files in os.walk(wiki_root):
            for f in files:
                if f.endswith(".md"):
                    md_files.append(os.path.join(root, f))
    except Exception:
        empty = RetrievalGraph(nodes={}, data_version=data_version)
        _cached_graph = empty
        return empty

    raw_nodes: list[dict] = []
    for file_path in md_files:
        nid = _file_name_to_id(os.path.basename(file_path))
        try:
            content = read_file_utf8(file_path)
        except Exception:
            continue
        fm, _body = extract_frontmatter(content)
        title = str(fm.get("title", ""))
        if not title:
            heading_match = re.search(r"^#\s+(.+)$", content, re.MULTILINE)
            title = heading_match.group(1).strip() if heading_match else nid.replace("-", " ")

        node_type = str(fm.get("type", "other")).lower()
        sources = fm.get("sources", [])
        if isinstance(sources, str):
            sources = [sources]

        raw_nodes.append(
            {
                "id": nid,
                "title": title,
                "type": node_type,
                "path": normalize_path(file_path),
                "sources": sources,
                "raw_links": _extract_wikilinks(content),
            }
        )

    node_ids = {n["id"] for n in raw_nodes}
    out_links_map: dict[str, set[str]] = {nid: set() for nid in node_ids}
    in_links_map: dict[str, set[str]] = {nid: set() for nid in node_ids}

    for raw in raw_nodes:
        for link_target in raw["raw_links"]:
            resolved = _resolve_target(link_target, node_ids)
            if resolved is None or resolved == raw["id"]:
                continue
            out_links_map[raw["id"]].add(resolved)
            in_links_map[resolved].add(raw["id"])

    nodes: dict[str, RetrievalNode] = {}
    for raw in raw_nodes:
        nodes[raw["id"]] = RetrievalNode(
            id=raw["id"],
            title=raw["title"],
            type=raw["type"],
            path=raw["path"],
            sources=raw["sources"],
            out_links=out_links_map[raw["id"]],
            in_links=in_links_map[raw["id"]],
        )

    graph = RetrievalGraph(nodes=nodes, data_version=data_version)
    _cached_graph = graph
    return graph


def calculate_relevance(node_a: RetrievalNode, node_b: RetrievalNode, graph: RetrievalGraph) -> float:
    """Calculate four-signal relevance score between two nodes."""
    if node_a.id == node_b.id:
        return 0.0

    # Signal 1: Direct links
    forward = 1 if node_b.id in node_a.out_links else 0
    backward = 1 if node_a.id in node_b.out_links else 0
    direct_link_score = (forward + backward) * _WEIGHTS["direct_link"]

    # Signal 2: Source overlap
    sources_a = set(node_a.sources)
    shared = sum(1 for src in node_b.sources if src in sources_a)
    source_overlap_score = shared * _WEIGHTS["source_overlap"]

    # Signal 3: Adamic-Adar common neighbors
    neighbors_a = _get_neighbors(node_a)
    neighbors_b = _get_neighbors(node_b)
    adamic_adar = 0.0
    for neighbor_id in neighbors_a:
        if neighbor_id in neighbors_b:
            neighbor = graph.nodes.get(neighbor_id)
            if neighbor:
                degree = _get_node_degree(neighbor)
                adamic_adar += 1.0 / math.log(max(degree, 2))
    common_neighbor_score = adamic_adar * _WEIGHTS["common_neighbor"]

    # Signal 4: Type affinity
    affinity_map = _TYPE_AFFINITY.get(node_a.type, {})
    type_affinity_score = affinity_map.get(node_b.type, 0.5) * _WEIGHTS["type_affinity"]

    return direct_link_score + source_overlap_score + common_neighbor_score + type_affinity_score


def get_related_nodes(node_id: str, graph: RetrievalGraph, limit: int = 5) -> list[dict]:
    """Return top related nodes sorted by relevance."""
    source_node = graph.nodes.get(node_id)
    if not source_node:
        return []

    scored: list[dict] = []
    for nid, node in graph.nodes.items():
        if nid == node_id:
            continue
        relevance = calculate_relevance(source_node, node, graph)
        if relevance > 0:
            scored.append({"node": node, "relevance": relevance})

    scored.sort(key=lambda x: x["relevance"], reverse=True)
    return scored[:limit]


def clear_graph_cache() -> None:
    global _cached_graph
    _cached_graph = None
