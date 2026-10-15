"""Tests for Pairwise Combinatorial and Hybrid search strategies."""

from mirror.agent import NaiveRefundAgent, ReconcilingRefundAgent
from mirror.search import (
    HybridSearchEngine,
    PairwiseCombinatorialSearchEngine,
)


def test_pairwise_combinatorial_search():
    agent = NaiveRefundAgent(max_attempts=2)
    engine = PairwiseCombinatorialSearchEngine(agent=agent, seed=42)
    result = engine.run(budget=20)

    assert result.total_experiments == 20
    assert result.strategy == "pairwise_combinatorial"
    assert len(result.raw_experiments) == 20


def test_hybrid_search_learning_and_discovery():
    agent = ReconcilingRefundAgent(max_attempts=2)
    engine = HybridSearchEngine(agent=agent, seed=42)
    result = engine.run(budget=25)

    assert result.total_experiments == 25
    assert result.strategy == "hybrid_adaptive_combinatorial"
    assert len(result.raw_experiments) == 25
    # Should discover multi-fault interaction
    assert result.unique_violations_found > 0
