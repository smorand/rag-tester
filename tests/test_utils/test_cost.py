"""Tests for cost calculation utilities."""

import logging

import pytest

from rag_tester.utils.cost import MODEL_PRICING, calculate_cost


class TestCalculateCost:
    """Tests for calculate_cost function."""

    def test_calculate_cost_known_model(self) -> None:
        """Test cost calculation for a known model."""
        # OpenAI text-embedding-3-small: $0.02 / 1M tokens
        cost = calculate_cost("openai/text-embedding-3-small", 15000)
        expected = (15000 / 1_000_000) * 0.02
        assert cost == round(expected, 6)
        assert cost == 0.0003

    def test_calculate_cost_large_model(self) -> None:
        """Test cost calculation for a more expensive model."""
        # OpenAI text-embedding-3-large: $0.13 / 1M tokens
        cost = calculate_cost("openai/text-embedding-3-large", 100000)
        expected = (100000 / 1_000_000) * 0.13
        assert cost == round(expected, 6)
        assert cost == 0.013

    def test_calculate_cost_unknown_model(self, caplog: pytest.LogCaptureFixture) -> None:
        """Test cost calculation for an unknown model returns 0.0 and logs warning."""
        with caplog.at_level(logging.WARNING):
            cost = calculate_cost("custom/my-model", 10000)

        assert cost == 0.0
        assert "Unknown model pricing for 'custom/my-model'" in caplog.text

    def test_calculate_cost_zero_tokens(self) -> None:
        """Test cost calculation with zero tokens."""
        cost = calculate_cost("openai/text-embedding-3-small", 0)
        assert cost == 0.0

    def test_calculate_cost_rounding(self) -> None:
        """Test that cost is rounded to 6 decimal places."""
        # Use a token count that would produce more than 6 decimal places
        cost = calculate_cost("openai/text-embedding-3-small", 1)
        expected = (1 / 1_000_000) * 0.02
        assert cost == round(expected, 6)
        assert cost == 0.00000002  # Should be rounded to 6 decimals

    def test_calculate_cost_all_models(self) -> None:
        """Test cost calculation for all models in pricing table."""
        tokens = 50000

        for model, price_per_million in MODEL_PRICING.items():
            cost = calculate_cost(model, tokens)
            expected = (tokens / 1_000_000) * price_per_million
            assert cost == round(expected, 6)
            assert cost > 0.0

    def test_calculate_cost_voyage_ai(self) -> None:
        """Test cost calculation for Voyage AI model."""
        cost = calculate_cost("voyage-ai/voyage-2", 25000)
        expected = (25000 / 1_000_000) * 0.10
        assert cost == round(expected, 6)
        assert cost == 0.0025

    def test_calculate_cost_cohere(self) -> None:
        """Test cost calculation for Cohere model."""
        cost = calculate_cost("cohere/embed-english-v3.0", 30000)
        expected = (30000 / 1_000_000) * 0.10
        assert cost == round(expected, 6)
        assert cost == 0.003

    def test_calculate_cost_ada_002(self) -> None:
        """Test cost calculation for legacy Ada-002 model."""
        cost = calculate_cost("openai/text-embedding-ada-002", 20000)
        expected = (20000 / 1_000_000) * 0.10
        assert cost == round(expected, 6)
        assert cost == 0.002


class TestModelPricing:
    """Tests for MODEL_PRICING constant."""

    def test_pricing_table_exists(self) -> None:
        """Test that pricing table is defined."""
        assert MODEL_PRICING is not None
        assert isinstance(MODEL_PRICING, dict)

    def test_pricing_table_has_openai_models(self) -> None:
        """Test that pricing table includes OpenAI models."""
        assert "openai/text-embedding-3-small" in MODEL_PRICING
        assert "openai/text-embedding-3-large" in MODEL_PRICING
        assert "openai/text-embedding-ada-002" in MODEL_PRICING

    def test_pricing_table_has_other_providers(self) -> None:
        """Test that pricing table includes other providers."""
        assert "voyage-ai/voyage-2" in MODEL_PRICING
        assert "cohere/embed-english-v3.0" in MODEL_PRICING

    def test_pricing_values_are_positive(self) -> None:
        """Test that all pricing values are positive numbers."""
        for _model, price in MODEL_PRICING.items():
            assert isinstance(price, (int, float))
            assert price > 0.0

    def test_pricing_values_are_reasonable(self) -> None:
        """Test that pricing values are in reasonable range (not typos)."""
        for model, price in MODEL_PRICING.items():
            # Prices should be between $0.01 and $1.00 per 1M tokens
            assert 0.01 <= price <= 1.0, f"Price for {model} seems unreasonable: ${price}"
