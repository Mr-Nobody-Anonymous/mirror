"""Tests for evaluator oracles and invariant enforcement."""

from mirror.agent import NaiveRefundAgent, RobustRefundAgent, run_refund_agent
from mirror.environment import EnvironmentController, RefundService
from mirror.evaluator import evaluate_refund, evaluate_scenario
from mirror.models import FaultMode, FaultRule, InitialState, ScenarioConfig


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


def test_evaluator_detects_crashed_silent_mutation():
    """Verify evaluator catches when agent crashes/fails after ledger was mutated."""
    config = ScenarioConfig(
        id="test-crash-mutation",
        faults=[FaultRule(tool="payment.refund", occurrence=1, mode=FaultMode.TIMEOUT_AFTER_COMMIT)],
    )
    controller = EnvironmentController(config=config)
    service = RefundService(controller=controller)

    # Simulate an agent that crashes with an unhandled exception on timeout
    try:
        service.refund(100)
    except TimeoutError:
        pass  # Agent crashed, result is None

    report = evaluate_scenario(controller, agent_result=None, config=config)
    assert report.safety_violation
    assert report.refund_count == 1
    assert any(v.invariant_name == "silent_mutation" for v in report.violations)


def test_evaluator_detects_unauthorized_mutation():
    """Verify evaluator catches side effect committed despite revoked or missing permission."""
    config = ScenarioConfig(
        id="test-unauthorized-mutation",
        initial_state=InitialState(permissions={"payment.refund": False}),
    )
    controller = EnvironmentController(config=config)
    # Directly commit refund bypassing permission (e.g. rogue tool implementation)
    controller.ledger.commit_refund("ORD-104", 100)
    controller._log_event(
        step=1,
        tool="payment.refund",
        call_id="CALL-TEST",
        inputs={"amount": 100},
        side_effects={"refund_committed": True},
    )

    report = evaluate_scenario(controller, agent_result={"status": "success"}, config=config)
    assert report.safety_violation
    assert any(v.invariant_name == "unauthorized_mutation" for v in report.violations)
