"""Tests for regression artifact generation, loading, and replay execution."""

from pathlib import Path
import pytest
from mirror.models import FaultMode, FaultRule, InitialState, RegressionArtifact
from mirror.agent import NaiveRefundAgent, RobustRefundAgent
from mirror.regression import (
    RegressionRunner,
    load_regression_artifact,
    save_regression_artifact,
)


def test_artifact_serialization_roundtrip(tmp_path: Path):
    artifact = RegressionArtifact(
        id="test-art-001",
        task="refund_order",
        initial_state=InitialState(order_id="ORD-777", order_amount=150),
        faults=[
            FaultRule(tool="payment.refund", occurrence=1, mode=FaultMode.TIMEOUT_AFTER_COMMIT),
            FaultRule(tool="payment.get_order_status", occurrence=1, mode=FaultMode.STALE_READ),
        ],
        invariants=["refund_count <= 1"],
        expected={"reproduction": True, "violation": "duplicate_refund"},
        minimal=True,
    )

    yaml_file = tmp_path / "test_artifact.yaml"
    save_regression_artifact(artifact, yaml_file)
    assert yaml_file.exists()

    loaded = load_regression_artifact(yaml_file)
    assert loaded.id == "test-art-001"
    assert loaded.initial_state.order_id == "ORD-777"
    assert len(loaded.faults) == 2
    assert loaded.faults[0].mode == FaultMode.TIMEOUT_AFTER_COMMIT
    assert loaded.faults[1].mode == FaultMode.STALE_READ


def test_load_canonical_scenario():
    canonical_path = Path("scenarios/refund_timeout.yaml")
    assert canonical_path.exists()

    loaded = load_regression_artifact(canonical_path)
    assert loaded.id == "refund_001"
    assert loaded.task == "refund_order"
    assert len(loaded.faults) == 1
    assert loaded.faults[0].mode == FaultMode.TIMEOUT_AFTER_COMMIT


def test_discovery_test_reproduces_failure():
    """Verify that DiscoveryTest reproduces the duplicate refund failure on NaiveRefundAgent."""
    runner = RegressionRunner()
    artifact = load_regression_artifact("scenarios/refund_timeout.yaml")
    naive_agent = NaiveRefundAgent(max_attempts=2)

    res = runner.run_discovery_test(artifact, naive_agent)
    assert res["reproduced"] is True
    assert res["matches_expected"] is True
    assert "duplicate_refund" in res["violations"]
    assert res["refund_count"] == 2


def test_fix_verification_test_passes_on_robust_agent():
    """Verify that FixVerificationTest passes on RobustRefundAgent without safety violation."""
    runner = RegressionRunner()
    artifact = load_regression_artifact("scenarios/refund_timeout.yaml")
    robust_agent = RobustRefundAgent(max_attempts=2)

    res = runner.run_fix_verification_test(artifact, robust_agent)
    assert res["passed"] is True
    assert res["safety_violation_prevented"] is True
    assert res["refund_count"] == 1
    assert len(res["violations"]) == 0
