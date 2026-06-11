"""
Wiki graph builder with optional Louvain community detection.
Derived from src/lib/wiki-graph.ts
"""

import logging
import os
import re
from typing import Any

from llm_wiki.graph_relevance import build_retrieval_graph, calculate_relevance
from llm_wiki.wiki_models import CommunityInfo, GraphEdge, GraphNode
from llm_wiki.wiki_utils import extract_frontmatter, normalize_path, read_file_utf8

logger = logging.getLogger(__name__)

_WIKILINK_REGEX = re.compile(r'\[\[([^\]|]+?)(?:\|[^\]]+?)?\]\]')

# Optional Louvain support
try:
    import community as community_louvain  # python-louvain

    HAS_LOUVAIN = True
except ImportError:
    try:
        import networkx as nx

        HAS_LOUVAIN = True
    except ImportError:
        HAS_LOUVAIN = False


def _extract_wikilinks(content: str) -> list[str]:
    return [m.group(1).strip() for m in _WIKILINK_REGEX.finditer(content)]


def _file_name_to_id(file_name: str) -> str:
    return re.sub(r"\.md$", "", file_name)


def _extract_title(content: str, file_name: str) -> str:
    fm, _body = extract_frontmatter(content)
    title = str(fm.get("title", ""))
    if title:
        return title
    heading_match = re.search(r"^#\s+(.+)$", content, re.MULTILINE)
    if heading_match:
        return heading_match.group(1).strip()
    return re.sub(r"\.md$", "", file_name).replace("-", " ")


def _extract_type(content: str) -> str:
    fm, _body = extract_frontmatter(content)
    return str(fm.get("type", "other")).lower()


def _resolve_target(raw: str, node_map: dict[str, Any]) -> str | None:
    if raw in node_map:
        return raw
    normalized = raw.lower().replace(" ", "-")
    for nid in node_map.keys():
        nid_lower = nid.lower()
        if nid_lower == normalized:
            return nid
        if nid_lower == raw.lower():
            return nid
        if nid_lower.replace(" ", "-") == normalized:
            return nid
    return None


def _detect_communities(
    nodes: list[dict], edges: list[GraphEdge]
) -> tuple[dict[str, int], list[CommunityInfo]]:
    """Run Louvain community detection and compute cohesion per community."""
    if not nodes or not HAS_LOUVAIN:
        return {}, []

    # Build graph
    try:
        import community as community_louvain  # type: ignore[import-not-found]

        G = _build_louvain_graph(edges)
        partition: dict[str, int] = community_louvain.best_partition(G, weight="weight")
    except Exception:
        try:
            import networkx as nx  # type: ignore[import-not-found]

            G = _build_louvain_graph(edges)
            communities = nx.community.louvain_communities(G, weight="weight", seed=42)
            partition = {}
            for comm_id, comm_nodes in enumerate(communities):
                for node in comm_nodes:
                    partition[node] = comm_id
        except Exception as e:
            logger.warning("Louvain community detection failed: %s", e)
            return {}, []

    # Group nodes by community
    groups: dict[int, list[str]] = {}
    for node_id, comm_id in partition.items():
        groups.setdefault(comm_id, []).append(node_id)

    # Edge lookup for cohesion
    edge_set: set[str] = set()
    for edge in edges:
        edge_set.add(f"{edge.source}:::{edge.target}")
        edge_set.add(f"{edge.target}:::{edge.source}")

    # Node info lookup
    node_info = {n["id"]: {"label": n["label"], "link_count": n["link_count"]} for n in nodes}

    communities_out: list[CommunityInfo] = []
    for comm_id, member_ids in groups.items():
        n = len(member_ids)
        intra_edges = 0
        for i in range(n):
            for j in range(i + 1, n):
                if f"{member_ids[i]}:::{member_ids[j]}" in edge_set:
                    intra_edges += 1
        possible_edges = (n * (n - 1)) // 2 if n > 1 else 1
        cohesion = intra_edges / possible_edges

        sorted_members = sorted(
            member_ids,
            key=lambda nid: node_info.get(nid, {}).get("link_count", 0),
            reverse=True,
        )
        top_nodes = [node_info.get(nid, {}).get("label", nid) for nid in sorted_members[:5]]
        communities_out.append(
            CommunityInfo(id=comm_id, node_count=n, cohesion=cohesion, top_nodes=top_nodes)
        )

    communities_out.sort(key=lambda c: c.node_count, reverse=True)

    # Renumber community IDs sequentially
    id_remap = {old_id: idx for idx, old_id in enumerate(c.id for c in communities_out)}
    for c in communities_out:
        c.id = id_remap[c.id]
    for node_id in list(partition.keys()):
        partition[node_id] = id_remap[partition[node_id]]

    return partition, communities_out


def _build_louvain_graph(edges: list[GraphEdge]):
    """Build an undirected NetworkX graph from edges."""
    import networkx as nx

    G = nx.Graph()
    node_ids = set()
    for edge in edges:
        node_ids.add(edge.source)
        node_ids.add(edge.target)
    for nid in node_ids:
        G.add_node(nid)
    seen: set[str] = set()
    for edge in edges:
        key = f"{edge.source}::{edge.target}"
        rkey = f"{edge.target}::{edge.source}"
        if key not in seen and rkey not in seen:
            G.add_edge(edge.source, edge.target, weight=edge.weight)
            seen.add(key)
    return G


def build_wiki_graph(
    project_path: str, use_louvain: bool = False, use_retrieval_weights: bool = True
) -> dict:
    """Build the full wiki graph with optional Louvain community detection.

    Returns dict with keys: nodes, edges, communities.
    """
    wiki_root = f"{normalize_path(project_path)}/wiki"
    md_files: list[str] = []
    try:
        for root, _dirs, files in os.walk(wiki_root):
            for f in files:
                if f.endswith(".md"):
                    md_files.append(os.path.join(root, f))
    except Exception:
        return {"nodes": [], "edges": [], "communities": []}

    if not md_files:
        return {"nodes": [], "edges": [], "communities": []}

    node_map: dict[str, dict] = {}
    for file_path in md_files:
        nid = _file_name_to_id(os.path.basename(file_path))
        try:
            content = read_file_utf8(file_path)
        except Exception:
            continue
        node_map[nid] = {
            "id": nid,
            "label": _extract_title(content, os.path.basename(file_path)),
            "type": _extract_type(content),
            "path": normalize_path(file_path),
            "links": _extract_wikilinks(content),
        }

    # Filter out query nodes
    HIDDEN_TYPES = {"query"}
    for nid in list(node_map.keys()):
        if node_map[nid]["type"] in HIDDEN_TYPES:
            del node_map[nid]

    # Count links and build raw edges
    link_counts: dict[str, int] = {nid: 0 for nid in node_map}
    raw_edges: list[dict] = []

    for source_id, node_data in node_map.items():
        for target_raw in node_data["links"]:
            target_id = _resolve_target(target_raw, node_map)
            if target_id is None or target_id == source_id:
                continue
            raw_edges.append({"source": source_id, "target": target_id, "weight": 1.0})
            link_counts[source_id] = link_counts.get(source_id, 0) + 1
            link_counts[target_id] = link_counts.get(target_id, 0) + 1

    # Deduplicate edges
    seen_edges: set[str] = set()
    deduped_edges: list[dict] = []
    for edge in raw_edges:
        key = f"{edge['source']}:::{edge['target']}"
        rkey = f"{edge['target']}:::{edge['source']}"
        if key not in seen_edges and rkey not in seen_edges:
            seen_edges.add(key)
            deduped_edges.append(edge)

    # Calculate relevance weights using retrieval graph
    if use_retrieval_weights:
        try:
            retrieval_graph = build_retrieval_graph(normalize_path(project_path), data_version=0)
        except Exception:
            retrieval_graph = None

        weighted_edges: list[GraphEdge] = []
        for e in deduped_edges:
            weight = 1.0
            if retrieval_graph:
                node_a = retrieval_graph.nodes.get(e["source"])
                node_b = retrieval_graph.nodes.get(e["target"])
                if node_a and node_b:
                    weight = calculate_relevance(node_a, node_b, retrieval_graph)
            weighted_edges.append(
                GraphEdge(source=e["source"], target=e["target"], weight=weight)
            )
    else:
        weighted_edges = [
            GraphEdge(source=e["source"], target=e["target"], weight=e["weight"])
            for e in deduped_edges
        ]

    # Louvain community detection
    communities: list[CommunityInfo] = []
    assignments: dict[str, int] = {}
    if use_louvain:
        prelim_nodes = [
            {"id": n["id"], "label": n["label"], "link_count": link_counts.get(n["id"], 0)}
            for n in node_map.values()
        ]
        assignments, communities = _detect_communities(prelim_nodes, weighted_edges)

    nodes = [
        GraphNode(
            id=n["id"],
            label=n["label"],
            type=n["type"],
            path=n["path"],
            link_count=link_counts.get(n["id"], 0),
            community=assignments.get(n["id"], 0),
        )
        for n in node_map.values()
    ]

    return {"nodes": nodes, "edges": weighted_edges, "communities": communities}
