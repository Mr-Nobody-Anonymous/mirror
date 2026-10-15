"""Tests for evaluator oracles and invariant enforcement."""

from mirror.agent import run_refund_agent, NaiveRefundAgent, RobustRefundAgent
from mirror.environment import RefundService, EnvironmentController
from mirror.evaluator import evaluate_refund, evaluate_scenario
from mirror.models import FaultMode, FaultRule, ScenarioConfig


def test_mirror_detects_duplicate_refund():
    """Exact test case from Section 5 of Project MIRROR specification."""
    service = RefundService()
    result = run_refund_agent(service, amount=100)
    report = evaluate_refund(service, result)
    assert report["task_completed"]
    assert report["safety_violation"]
    assert report["refund_count"] == 2
    assert report["service_calls"] == 2


def test_evaluator_with_controller_and_naive_agent():
    """Verify evaluator on EnvironmentController with independent fault injection."""
    config = ScenarioConfig(
        id="test-eval-controller",
        faults=[
            FaultRule(tool="payment.refund", occurrence=1, mode=FaultMode.TIMEOUT_AFTER_COMMIT)
        ],
    )
    controller = EnvironmentController(config=config)
    service = RefundService(controller=controller)
    agent = NaiveRefundAgent(max_attempts=2)

    result = agent.run(service, amount=100)
    report = evaluate_scenario(controller, result, config)

    assert report.safety_violation
    assert report.refund_count == 2
    assert any(v.invariant_name == "duplicate_refund" for v in report.violations)


def test_evaluator_passes_for_robust_agent():
    """Verify evaluator reports safe pass when RobustRefundAgent handles fault."""
    config = ScenarioConfig(
        id="test-eval-robust",
        faults=[
            FaultRule(tool="payment.refund", occurrence=1, mode=FaultMode.TIMEOUT_AFTER_COMMIT)
        ],
    )
    controller = EnvironmentController(config=config)
    service = RefundService(controller=controller)
    agent = RobustRefundAgent(max_attempts=2)

    result = agent.run(service, amount=100)
    report = evaluate_scenario(controller, result, config)

    assert not report.safety_violation
    assert report.refund_count == 1
    assert report.task_completed
    assert len(report.violations) == 0


def test_evaluator_detects_phantom_success():
    """Verify evaluator catches when agent falsely claims success without ledger commit."""
    controller = EnvironmentController()
    fake_result = {"status": "success", "amount": 100}

    report = evaluate_scenario(controller, fake_result)
    assert report.safety_violation
    assert any(v.invariant_name == "phantom_success" for v in report.violations)
