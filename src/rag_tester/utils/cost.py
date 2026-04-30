"""Cost calculation utilities for embedding models."""

import logging
from typing import Final

logger = logging.getLogger(__name__)

# Pricing table: model identifier -> price per 1M tokens (USD)
MODEL_PRICING: Final[dict[str, float]] = {
    "openai/text-embedding-3-small": 0.02,
    "openai/text-embedding-3-large": 0.13,
    "openai/text-embedding-ada-002": 0.10,
    "voyage-ai/voyage-2": 0.10,
    "cohere/embed-english-v3.0": 0.10,
}


def calculate_cost(model: str, tokens: int) -> float:
    """Calculate embedding cost for a given model and token count.

    Args:
        model: Model identifier (e.g., "openai/text-embedding-3-small")
        tokens: Total tokens consumed

    Returns:
        Total cost in USD, rounded to 6 decimal places

    Business Rules:
        - Uses MODEL_PRICING table for known models
        - For unknown models: returns 0.0 and logs warning
        - Cost = (tokens / 1,000,000) * price_per_million
        - Rounds to 6 decimal places
    """
    if model not in MODEL_PRICING:
        logger.warning(
            "Unknown model pricing for '%s', cost set to 0.0",
            model,
        )
        return 0.0

    price_per_million = MODEL_PRICING[model]
    cost = (tokens / 1_000_000) * price_per_million

    logger.debug(
        "Cost calculated: $%.6f for %d tokens (model: %s)",
        cost,
        tokens,
        model,
    )

    return round(cost, 6)
