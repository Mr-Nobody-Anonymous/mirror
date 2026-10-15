"""Tests for AuthoritativeLedger, EnvironmentController, and RefundService."""

import pytest
from mirror.models import FaultMode, FaultRule, InitialState, ScenarioConfig
from mirror.environment import AuthoritativeLedger, EnvironmentController, RefundService


def test_authoritative_ledger_initial_state():
    ledger = AuthoritativeLedger(InitialState(order_id="ORD-999", order_amount=250))
    state = ledger.get_order_state("ORD-999")
    assert state["order_id"] == "ORD-999"
    assert state["amount"] == 250
    assert state["refund_count"] == 0
    assert state["status"] == "paid"


def test_authoritative_ledger_commit_without_idempotency():
    ledger = AuthoritativeLedger()
    # First refund
    res1 = ledger.commit_refund("ORD-104", 100)
    assert res1["status"] == "success"
    assert not res1["deduplicated"]
    assert ledger.total_refunds == 1

    # Second refund without key increments refund count (demonstrating bug capability)
    res2 = ledger.commit_refund("ORD-104", 100)
    assert res2["status"] == "success"
    assert not res2["deduplicated"]
    assert ledger.total_refunds == 2


def test_authoritative_ledger_idempotent_commit():
    ledger = AuthoritativeLedger()
    res1 = ledger.commit_refund("ORD-104", 100, idempotency_key="K-1")
    assert not res1["deduplicated"]
    assert ledger.total_refunds == 1

    # Retry with same idempotency key returns cached record without re-incrementing
    res2 = ledger.commit_refund("ORD-104", 100, idempotency_key="K-1")
    assert res2["deduplicated"]
    assert res2["refund_id"] == res1["refund_id"]
    assert ledger.total_refunds == 1


def test_environment_controller_interception_and_trace():
    config = ScenarioConfig(id="test-env-01")
    controller = EnvironmentController(config=config)
    service = RefundService(controller=controller)

    res = service.refund(amount=100)
    assert res["status"] == "success"
    assert len(controller.trace) == 1

    event = controller.trace[0]
    assert event.tool == "payment.refund"
    assert event.inputs["amount"] == 100
    assert event.side_effects["refund_committed"] is True
    assert event.error is None


def test_environment_controller_reset():
    config = ScenarioConfig(id="test-env-reset")
    controller = EnvironmentController(config=config)
    service = RefundService(controller=controller)

    service.refund(100)
    assert controller.ledger.total_refunds == 1
    assert len(controller.trace) == 1

    controller.reset()
    assert controller.ledger.total_refunds == 0
    assert len(controller.trace) == 0


def test_refund_service_property_sync_on_exception():
    """Verify service.successful_refunds and calls stay synchronized even when exceptions are raised."""
    config = ScenarioConfig(
        id="test-prop-sync",
        faults=[FaultRule(tool="payment.refund", occurrence=1, mode=FaultMode.TIMEOUT_AFTER_COMMIT)],
    )
    controller = EnvironmentController(config=config)
    service = RefundService(controller=controller)

    with pytest.raises(TimeoutError):
        service.refund(100)

    # Invariant: property dynamically returns authoritative ledger count
    assert service.successful_refunds == 1
    assert service.calls == 1


def test_stale_read_custom_order_id():
    """Verify STALE_READ operates cleanly on custom orders without KeyError."""
    config = ScenarioConfig(
        id="test-custom-stale",
        initial_state=InitialState(order_id="ORD-CUSTOM", order_amount=300),
        faults=[FaultRule(tool="payment.get_order_status", occurrence=1, mode=FaultMode.STALE_READ)],
    )
    controller = EnvironmentController(config=config)
    service = RefundService(controller=controller, order_id="ORD-CUSTOM")

    # Commit refund
    service.refund(300)
    assert controller.ledger.total_refunds == 1

    # Query status with STALE_READ on ORD-CUSTOM
    status = service.get_order_status("ORD-CUSTOM")
    assert status["refund_count"] == 0
    assert status["cached"] is True


def test_authoritative_ledger_initial_refund_count_consistency():
    """Verify AuthoritativeLedger correctly sets status and refunded_amount when initial refund_count > 0."""
    init_state = InitialState(order_id="ORD-PRE", order_amount=200, refund_count=1)
    ledger = AuthoritativeLedger(init_state)
    order = ledger.get_order_state("ORD-PRE")
    assert order["refund_count"] == 1
    assert order["refunded_amount"] == 200
    assert order["status"] == "refunded"
