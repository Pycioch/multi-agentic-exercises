from pydantic import BaseModel


class BudgetConfig(BaseModel):
    """Immutable limits for one pipeline run.

    Created once per invocation and passed into the graph via state.
    Each node reads these limits to decide whether to proceed.
    """

    max_tokens_per_run: int = 10_000
    recursion_limit: int = 10


DEFAULT_BUDGET = BudgetConfig()


def check_budget(tokens_used: int, new_tokens: int, budget: BudgetConfig) -> None:
    """Raise RuntimeError if adding new_tokens would exceed the per-run cap."""
    total = tokens_used + new_tokens
    if total > budget.max_tokens_per_run:
        raise RuntimeError(
            f"Token budget exceeded: {total} tokens > {budget.max_tokens_per_run} limit. "
            "Increase BudgetConfig.max_tokens_per_run or simplify the query."
        )
