"""
Pure budget allocator for chat context assembly.
Derived from src/lib/context-budget.ts
"""

from llm_wiki.wiki_models import ContextBudget

DEFAULT_MAX_CTX = 204_800
RESPONSE_RESERVE_FRAC = 0.15
INDEX_BUDGET_FRAC = 0.05
PAGE_BUDGET_FRAC = 0.5
PER_PAGE_FRAC = 0.3
PER_PAGE_FLOOR = 5_000


def compute_context_budget(max_context_size: int | None) -> ContextBudget:
    max_ctx = max_context_size if isinstance(max_context_size, int) and max_context_size > 0 else DEFAULT_MAX_CTX
    response_reserve = int(max_ctx * RESPONSE_RESERVE_FRAC)
    index_budget = int(max_ctx * INDEX_BUDGET_FRAC)
    page_budget = int(max_ctx * PAGE_BUDGET_FRAC)
    max_page_size = min(
        page_budget,
        max(PER_PAGE_FLOOR, int(page_budget * PER_PAGE_FRAC)),
    )
    return ContextBudget(
        max_ctx=max_ctx,
        response_reserve=response_reserve,
        index_budget=index_budget,
        page_budget=page_budget,
        max_page_size=max_page_size,
    )
