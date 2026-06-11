"""
Independent LLM Wiki searcher.

Performs query tokenization, keyword matching across all wiki/source nodes,
and configurable multi-hop graph expansion. Outputs Haystack Document objects
compatible with the RAG pipeline.
"""

import os
from typing import Dict, List, Optional

from haystack import Document

from llm_wiki.graph_relevance import build_retrieval_graph, get_related_nodes
from llm_wiki.wiki_models import RetrievalGraph, RetrievalNode
from llm_wiki.wiki_tokenizer import tokenize_query, trim_query_punctuation
from llm_wiki.wiki_utils import normalize_path, read_file_utf8


def _find_project_root(start_dir: str = os.getcwd()) -> str:
    """Locate project root by searching upward for main.py."""
    current_dir = os.path.abspath(start_dir)
    while True:
        if "main.py" in os.listdir(current_dir):
            return current_dir
        parent_dir = os.path.dirname(current_dir)
        if parent_dir == current_dir:
            raise FileNotFoundError("Could not find main.py in any parent directory.")
        current_dir = parent_dir

# Weights aligned with nashsu_llm_wiki's Rust backend (search.rs)
FILENAME_EXACT_BONUS = 200.0
PHRASE_IN_TITLE_BONUS = 50.0
PHRASE_IN_CONTENT_PER_OCC = 20.0
MAX_PHRASE_OCC_COUNTED = 10
TITLE_TOKEN_WEIGHT = 5.0
CONTENT_TOKEN_WEIGHT = 1.0


def _count_occurrences(haystack: str, needle: str) -> int:
    if not needle:
        return 0
    return haystack.count(needle)


def _token_match_score(text: str, tokens: List[str], max_chars: int = 3000) -> int:
    """Count how many distinct tokens appear in text (capped at max_chars to avoid long-doc spam)."""
    lower = text.lower()[:max_chars]
    return sum(1 for token in tokens if token in lower)


def _score_node(
    node: RetrievalNode,
    content: str,
    tokens: List[str],
    query_phrase: str,
) -> float:
    """Score a single wiki node for keyword relevance."""
    file_name = os.path.basename(node.path)
    stem = file_name.removesuffix(".md").lower()

    title_text = f"{node.title} {file_name}".lower()
    content_lower = content.lower()

    filename_exact = bool(query_phrase) and stem == query_phrase
    title_has_phrase = bool(query_phrase) and query_phrase in title_text
    content_phrase_occ = min(
        _count_occurrences(content_lower, query_phrase),
        MAX_PHRASE_OCC_COUNTED,
    )
    title_token_score = _token_match_score(title_text, tokens)
    content_token_score = _token_match_score(content, tokens)

    # Must hit at least one signal
    if (
        not filename_exact
        and not title_has_phrase
        and content_phrase_occ == 0
        and title_token_score == 0
        and content_token_score == 0
    ):
        return 0.0

    score = 0.0
    if filename_exact:
        score += FILENAME_EXACT_BONUS
    if title_has_phrase:
        score += PHRASE_IN_TITLE_BONUS
    score += content_phrase_occ * PHRASE_IN_CONTENT_PER_OCC
    score += title_token_score * TITLE_TOKEN_WEIGHT
    score += content_token_score * CONTENT_TOKEN_WEIGHT

    # Bonus for long tokens (whole English words / CJK words) appearing in the title.
    # This helps "Dolby Vision" outrank generic index pages that merely contain substrings.
    for token in tokens:
        if len(token) >= 3 and token in title_text:
            score += 5.0

    return score


class WikiSearcher:
    """Search the LLM Wiki graph using keyword matching + graph expansion."""

    def __init__(
        self,
        project_path: Optional[str] = None,
        top_k: int = 20,
        expansion_hops: int = 2,
        expansion_limit: int = 3,
        expansion_min_relevance: float = 2.0,
        max_content_length: int = 4000,
    ):
        if project_path is None:
            project_path = normalize_path(os.path.join(_find_project_root(), "llm_wiki"))
        self.project_path = project_path
        self.top_k = top_k
        self.expansion_hops = expansion_hops
        self.expansion_limit = expansion_limit
        self.expansion_min_relevance = expansion_min_relevance
        self.max_content_length = max_content_length

        self._graph: Optional[RetrievalGraph] = None
        self._content_cache: Dict[str, str] = {}

    def _load_graph(self) -> RetrievalGraph:
        if self._graph is None:
            self._graph = build_retrieval_graph(self.project_path, data_version=0)
        return self._graph

    def _read_content(self, node: RetrievalNode) -> str:
        if node.path not in self._content_cache:
            try:
                self._content_cache[node.path] = read_file_utf8(node.path)
            except Exception:
                self._content_cache[node.path] = ""
        return self._content_cache[node.path]

    def search(self, query: str) -> List[Document]:
        """Return top-k Documents matching the query via keyword + graph."""
        graph = self._load_graph()
        if not graph.nodes:
            return []

        tokens = tokenize_query(query)
        effective_tokens = tokens if tokens else [query.strip().lower()]
        query_phrase = trim_query_punctuation(query.lower())

        # 1) keyword scoring across all wiki nodes (sources + entities + concepts)
        scored_nodes: List[tuple[float, RetrievalNode]] = []
        for node in graph.nodes.values():
            content = self._read_content(node)
            score = _score_node(node, content, effective_tokens, query_phrase)
            if score > 0:
                scored_nodes.append((score, node))

        scored_nodes.sort(key=lambda x: (-x[0], x[1].id))
        seed_nodes = scored_nodes[: self.top_k]

        # 2) multi-hop graph expansion
        # Maintain the best combined score seen for each expanded node.
        # Frontier propagates the accumulated relevance of the path from seed.
        expanded: Dict[str, float] = {}
        frontier: Dict[str, float] = {
            node.id: score for score, node in seed_nodes
        }

        for _hop in range(self.expansion_hops):
            next_frontier: Dict[str, float] = {}
            for node_id, base_score in frontier.items():
                related = get_related_nodes(
                    node_id, graph, limit=self.expansion_limit
                )
                for item in related:
                    rel_node = item["node"]
                    relevance = float(item["relevance"])
                    if relevance < self.expansion_min_relevance:
                        continue
                    # Combined score: weighted average so hub nodes cannot
                    # outrank seeds simply by accumulating high relevance.
                    combined = base_score * 0.5 + relevance * 0.3
                    if rel_node.id not in expanded or expanded[rel_node.id] < combined:
                        expanded[rel_node.id] = combined
                    if (
                        rel_node.id not in next_frontier
                        or next_frontier[rel_node.id] < combined
                    ):
                        next_frontier[rel_node.id] = combined
            frontier = next_frontier
            if not frontier:
                break

        # Merge seeds and expanded nodes, keep highest score per node
        final_scores: Dict[str, float] = {
            node.id: score for score, node in seed_nodes
        }
        for node_id, score in expanded.items():
            if node_id not in final_scores or final_scores[node_id] < score:
                final_scores[node_id] = score

        # 3) build Documents sorted by final score
        docs: List[Document] = []
        for node_id, score in sorted(
            final_scores.items(), key=lambda x: (-x[1], x[0])
        ):
            node = graph.nodes.get(node_id)
            if not node:
                continue
            content = self._read_content(node)
            if not content:
                continue
            if len(content) > self.max_content_length:
                content = content[: self.max_content_length] + "\n\n[...truncated...]"

            doc = Document(
                content=content,
                meta={
                    "metadata": {
                        "source": "llm_wiki_graph",
                        "wiki_node_id": node_id,
                        "wiki_node_title": node.title,
                        "wiki_node_type": node.type,
                        "wiki_relevance": round(score, 4),
                        "file_name": node_id,
                    }
                },
            )
            docs.append(doc)

        return docs
