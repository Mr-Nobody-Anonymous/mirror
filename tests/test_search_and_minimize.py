"""Tests for search engines (Random and Adaptive) and Delta Debugging minimization."""

import pytest
from mirror.models import FaultMode, FaultRule, ScenarioConfig
from mirror.agent import NaiveRefundAgent
from mirror.search import RandomSearchEngine, AdaptiveSearchEngine
from mirror.minimize import ScenarioMinimizer


def test_random_search_finds_violations():
    agent = NaiveRefundAgent(max_attempts=2)
    engine = RandomSearchEngine(agent=agent, seed=42)
    result = engine.run(budget=25)

    assert result.total_experiments == 25
    # Should discover at least one duplicate refund violation
    assert len(result.discovered_scenarios) > 0
    assert result.first_discovery_step is not None
    assert result.first_discovery_step <= 25


def test_adaptive_search_bandit_learning():
    agent = NaiveRefundAgent(max_attempts=2)
    engine = AdaptiveSearchEngine(agent=agent, seed=123)
    result = engine.run(budget=20)

    assert result.total_experiments == 20
    assert len(result.discovered_scenarios) > 0
    # The timeout_after_commit mode should have been prioritized and yielded violations
    assert engine.mode_violations.get(FaultMode.TIMEOUT_AFTER_COMMIT.value, 0) > 0


def test_delta_debugging_minimization():
    """Verify that ddmin reduces a scenario with irrelevant faults to the minimal root-cause fault."""
    agent = NaiveRefundAgent(max_attempts=2)
    minimizer = ScenarioMinimizer(agent=agent, target_violation_type="duplicate_refund")

    # Construct scenario with 1 root cause fault and 2 extraneous faults
    bloated = ScenarioConfig(
        id="bloated-test",
        faults=[
            FaultRule(tool="payment.get_order_status", occurrence=1, mode=FaultMode.STALE_READ),
            FaultRule(tool="payment.refund", occurrence=1, mode=FaultMode.TIMEOUT_AFTER_COMMIT),
            FaultRule(tool="payment.get_order_status", occurrence=2, mode=FaultMode.STALE_READ),
        ],
    )

    minimized, evals = minimizer.minimize(bloated, method="ddmin")

    assert len(minimized.faults) == 1
    assert minimized.faults[0].tool == "payment.refund"
    assert minimized.faults[0].mode == FaultMode.TIMEOUT_AFTER_COMMIT
    assert minimized.metadata["minimized"] is True
    assert evals > 0


def test_minimization_strictly_1_minimal():
    """Verify that removing the single remaining fault stops the invariant violation."""
    agent = NaiveRefundAgent(max_attempts=2)
    minimizer = ScenarioMinimizer(agent=agent, target_violation_type="duplicate_refund")

    single_fault_config = ScenarioConfig(
        id="single-fault",
        faults=[
            FaultRule(tool="payment.refund", occurrence=1, mode=FaultMode.TIMEOUT_AFTER_COMMIT),
        ],
    )

    # Empty fault candidate
    reproduced, report = minimizer.test_fault_subset(single_fault_config, [])
    assert reproduced is False
    assert report.safety_violation is False
