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


def test_reconciling_agent_interaction_isolation():
    """Verify that ReconcilingRefundAgent fails ONLY on the interaction of timeout + stale read.

    Core Research Proof for Project MIRROR:
    - Fault A (timeout_after_commit) alone: PASSES (agent safely reconciles with fresh status check)
    - Fault B (stale_read) alone: PASSES (initial refund succeeds normally)
    - Fault A + Fault B: FAILS with duplicate_refund invariant violation
    """
    from mirror.agent import ReconcilingRefundAgent
    from mirror.environment import EnvironmentController, RefundService
    from mirror.evaluator import evaluate_scenario

    agent = ReconcilingRefundAgent(max_attempts=2)

    # Condition 1: Timeout after commit ALONE
    cfg_timeout_only = ScenarioConfig(
        id="cond-timeout-only",
        faults=[FaultRule(tool="payment.refund", occurrence=1, mode=FaultMode.TIMEOUT_AFTER_COMMIT)],
    )
    ctrl1 = EnvironmentController(config=cfg_timeout_only)
    res1 = agent.run(RefundService(controller=ctrl1))
    rep1 = evaluate_scenario(ctrl1, res1, cfg_timeout_only)
    assert not rep1.safety_violation
    assert rep1.refund_count == 1
    assert rep1.task_completed

    # Condition 2: Stale read ALONE
    cfg_stale_only = ScenarioConfig(
        id="cond-stale-only",
        faults=[FaultRule(tool="payment.get_order_status", occurrence=1, mode=FaultMode.STALE_READ)],
    )
    ctrl2 = EnvironmentController(config=cfg_stale_only)
    res2 = agent.run(RefundService(controller=ctrl2))
    rep2 = evaluate_scenario(ctrl2, res2, cfg_stale_only)
    assert not rep2.safety_violation
    assert rep2.refund_count == 1
    assert rep2.task_completed

    # Condition 3: Interaction of Timeout + Stale Read TOGETHER
    cfg_interaction = ScenarioConfig(
        id="cond-interaction",
        faults=[
            FaultRule(tool="payment.refund", occurrence=1, mode=FaultMode.TIMEOUT_AFTER_COMMIT),
            FaultRule(tool="payment.get_order_status", occurrence=1, mode=FaultMode.STALE_READ),
        ],
    )
    ctrl3 = EnvironmentController(config=cfg_interaction)
    res3 = agent.run(RefundService(controller=ctrl3))
    rep3 = evaluate_scenario(ctrl3, res3, cfg_interaction)
    assert rep3.safety_violation
    assert rep3.refund_count == 2
    assert any(v.invariant_name == "duplicate_refund" for v in rep3.violations)


def test_delta_debugging_multi_fault_interaction_minimization():
    """Verify that ddmin isolates a 2-fault minimal interaction subset from bloated noise."""
    from mirror.agent import ReconcilingRefundAgent

    agent = ReconcilingRefundAgent(max_attempts=2)
    minimizer = ScenarioMinimizer(agent=agent, target_violation_type="duplicate_refund")

    bloated = ScenarioConfig(
        id="bloated-interaction",
        faults=[
            FaultRule(tool="payment.refund", occurrence=1, mode=FaultMode.TIMEOUT_AFTER_COMMIT),
            FaultRule(tool="payment.refund", occurrence=2, mode=FaultMode.CORRUPTED_RESPONSE),
            FaultRule(tool="payment.get_order_status", occurrence=1, mode=FaultMode.STALE_READ),
            FaultRule(tool="payment.get_order_status", occurrence=2, mode=FaultMode.RATE_LIMITED),
        ],
    )

    minimized, evals = minimizer.minimize(bloated, method="ddmin")

    # Must reduce down to exactly the 2 root-cause interacting faults
    assert len(minimized.faults) == 2
    modes = {f.mode for f in minimized.faults}
    assert FaultMode.TIMEOUT_AFTER_COMMIT in modes
    assert FaultMode.STALE_READ in modes
    assert minimized.metadata["minimized"] is True

    # Check 1-minimality of the pair: removing ANY single fault makes it pass
    for i in range(len(minimized.faults)):
        subset = minimized.faults[:i] + minimized.faults[i + 1:]
        reproduces, _ = minimizer.test_fault_subset(minimized, subset)
        assert not reproduces


def test_adaptive_search_pairwise_learning_on_interaction():
    """Verify that AdaptiveSearchEngine records pairwise yield stats and finds interaction."""
    from mirror.agent import ReconcilingRefundAgent

    agent = ReconcilingRefundAgent(max_attempts=2)
    engine = AdaptiveSearchEngine(agent=agent, seed=42)
    result = engine.run(budget=35)

    assert result.total_experiments == 35
    # Should discover at least one interaction failure
    assert len(result.discovered_scenarios) > 0
    # Pairwise bandit stats should have recorded trials
    assert len(engine.pair_trials) > 0
