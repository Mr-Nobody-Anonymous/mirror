"""Tests for FaultInjector and fault simulation modes."""

import pytest

from mirror.environment import EnvironmentController, RefundService
from mirror.faults import FaultInjector
from mirror.models import FaultMode, FaultRule, ScenarioConfig


def test_fault_injector_occurrence_matching():
    injector = FaultInjector([
        FaultRule(tool="payment.refund", occurrence=2, mode=FaultMode.TIMEOUT_BEFORE_COMMIT)
    ])

    occ1, match1 = injector.advance_and_match("payment.refund")
    assert occ1 == 1
    assert match1 is None

    occ2, match2 = injector.advance_and_match("payment.refund")
    assert occ2 == 2
    assert match2 is not None
    assert match2.mode == FaultMode.TIMEOUT_BEFORE_COMMIT

    occ3, match3 = injector.advance_and_match("payment.refund")
    assert occ3 == 3
    assert match3 is None


def test_timeout_after_commit_fault():
    """Verify that timeout_after_commit commits to the ledger BEFORE raising TimeoutError."""
    config = ScenarioConfig(
        id="test-timeout-after-commit",
        faults=[
            FaultRule(tool="payment.refund", occurrence=1, mode=FaultMode.TIMEOUT_AFTER_COMMIT)
        ],
    )
    controller = EnvironmentController(config=config)
    service = RefundService(controller=controller)

    with pytest.raises(TimeoutError, match="Response lost after refund committed"):
        service.refund(100)

    # Crucial assertion: the refund WAS committed in the authoritative ledger!
    assert controller.ledger.total_refunds == 1
    assert len(controller.trace) == 1
    assert controller.trace[0].fault_applied == FaultMode.TIMEOUT_AFTER_COMMIT


def test_timeout_before_commit_fault():
    """Verify that timeout_before_commit aborts BEFORE mutating the ledger."""
    config = ScenarioConfig(
        id="test-timeout-before-commit",
        faults=[
            FaultRule(tool="payment.refund", occurrence=1, mode=FaultMode.TIMEOUT_BEFORE_COMMIT)
        ],
    )
    controller = EnvironmentController(config=config)
    service = RefundService(controller=controller)

    with pytest.raises(TimeoutError, match="Network timeout before request reached server"):
        service.refund(100)

    # Invariant: ledger was NOT mutated
    assert controller.ledger.total_refunds == 0


def test_permission_revocation_fault():
    config = ScenarioConfig(
        id="test-permission-revocation",
        faults=[
            FaultRule(tool="payment.refund", occurrence=1, mode=FaultMode.PERMISSION_REVOKED)
        ],
    )
    controller = EnvironmentController(config=config)
    service = RefundService(controller=controller)

    with pytest.raises(PermissionError, match="Permission denied"):
        service.refund(100)

    assert controller.ledger.total_refunds == 0
    assert not controller.ledger.is_permission_granted("payment.refund")


def test_stale_read_fault():
    config = ScenarioConfig(
        id="test-stale-read",
        faults=[
            FaultRule(tool="payment.get_order_status", occurrence=1, mode=FaultMode.STALE_READ)
        ],
    )
    controller = EnvironmentController(config=config)
    service = RefundService(controller=controller)

    # First mutate ledger
    controller.ledger.commit_refund("ORD-104", 100)
    assert controller.ledger.total_refunds == 1

    # Calling get_order_status with stale_read fault returns pre-refund snapshot!
    status = service.get_order_status("ORD-104")
    assert status["refund_count"] == 0
    assert status.get("cached") is True
