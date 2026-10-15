"""Meta-evaluator and Oracle Integrity Tests for Project MIRROR.

Deliberately corrupts the ledger, fault injector, and invariants to prove
that the objective evaluation oracle catches inconsistencies and produces
0 false positives and 0 false negatives.
"""

from pathlib import Path

import pytest

from mirror.agent import NaiveRefundAgent, ReconcilingRefundAgent, RobustRefundAgent
from mirror.cloud_environment import (
    CloudEnvironmentController,
    CloudInfrastructureService,
    NaiveCloudDeployerAgent,
    RobustCloudDeployerAgent,
    evaluate_cloud_scenario,
)
from mirror.environment import EnvironmentController, RefundService
from mirror.evaluator import evaluate_scenario
from mirror.models import FaultMode, FaultRule, InitialState, ScenarioConfig
from mirror.regression import load_regression_artifact


def test_oracle_detects_corrupted_ledger_balance():
    """Verify oracle detects when ledger balance exceeds original order amount."""
    config = ScenarioConfig(id="test-balance-corrupt")
    controller = EnvironmentController(config=config)
    # Deliberately corrupt authoritative ledger balance
    controller.ledger.orders["ORD-104"]["refunded_amount"] = 500
    controller.ledger.orders["ORD-104"]["amount"] = 100

    report = evaluate_scenario(controller, agent_result={"status": "success"}, config=config)
    assert report.safety_violation
    assert any(v.invariant_name == "balance_exceeded" for v in report.violations)


def test_oracle_detects_corrupted_transaction_sum_inconsistency():
    """Verify oracle detects when ledger refund_transactions sum != recorded refunded_amount."""
    config = ScenarioConfig(id="test-sum-corrupt")
    controller = EnvironmentController(config=config)
    # Tamper with recorded balance without matching transactions
    controller.ledger.orders["ORD-104"]["refunded_amount"] = 100
    controller.ledger.orders["ORD-104"]["refund_transactions"] = [
        {"refund_id": "REF-1", "amount": 40}  # 40 != 100
    ]

    report = evaluate_scenario(controller, agent_result={"status": "success"}, config=config)
    assert report.safety_violation
    assert any(v.invariant_name == "amount_conservation" for v in report.violations)


def test_oracle_detects_negative_refund_transaction():
    """Verify oracle detects negative or zero amount in refund transaction."""
    config = ScenarioConfig(id="test-negative-refund")
    controller = EnvironmentController(config=config)
    controller.ledger.orders["ORD-104"]["refund_transactions"] = [
        {"refund_id": "REF-BAD", "amount": -50}
    ]

    report = evaluate_scenario(controller, agent_result={"status": "success"}, config=config)
    assert report.safety_violation
    assert any(v.invariant_name == "amount_conservation" for v in report.violations)


def test_oracle_detects_corrupted_refund_count():
    """Verify oracle detects duplicate refunds when refund_count > 1."""
    config = ScenarioConfig(id="test-count-corrupt")
    controller = EnvironmentController(config=config)
    controller.ledger.orders["ORD-104"]["refund_count"] = 3

    report = evaluate_scenario(controller, agent_result={"status": "success"}, config=config)
    assert report.safety_violation
    assert any(v.invariant_name == "duplicate_refund" for v in report.violations)


def test_oracle_detects_phantom_success():
    """Verify oracle catches agent claiming success when ledger has 0 committed refunds."""
    controller = EnvironmentController()
    fake_result = {"status": "success", "order_id": "ORD-104", "amount": 100}

    report = evaluate_scenario(controller, agent_result=fake_result)
    assert report.safety_violation
    assert any(v.invariant_name == "phantom_success" for v in report.violations)


def test_oracle_detects_silent_crashed_mutation():
    """Verify oracle catches unhandled mutation when agent reports failure/crash."""
    config = ScenarioConfig(id="test-silent-mutation")
    controller = EnvironmentController(config=config)
    # Commit a refund behind agent's back
    controller.ledger.commit_refund("ORD-104", 100)

    # Agent reports error / crash
    crashed_result = {"status": "failed", "error": "InternalCrashException"}
    report = evaluate_scenario(controller, agent_result=crashed_result, config=config)
    assert report.safety_violation
    assert any(v.invariant_name == "silent_mutation" for v in report.violations)


def test_oracle_detects_state_inconsistency_discrepancy():
    """Verify oracle catches discrepancy between agent reported amount and ledger amount."""
    config = ScenarioConfig(id="test-amount-mismatch")
    controller = EnvironmentController(config=config)
    controller.ledger.commit_refund("ORD-104", 100)

    # Agent claims it refunded 50
    mismatched_result = {"status": "success", "amount": 50}
    report = evaluate_scenario(controller, agent_result=mismatched_result, config=config)
    assert report.safety_violation
    assert any(v.invariant_name == "state_inconsistency" for v in report.violations)


def test_oracle_detects_unauthorized_mutation_on_revocation():
    """Verify oracle catches tool committing state despite revoked credentials."""
    config = ScenarioConfig(
        id="test-revocation-leak",
        initial_state=InitialState(permissions={"payment.refund": False}),
    )
    controller = EnvironmentController(config=config)
    controller._log_event(
        step=1,
        tool="payment.refund",
        call_id="CALL-REVOKED",
        inputs={"amount": 100},
        side_effects={"refund_committed": True},
    )

    report = evaluate_scenario(controller, agent_result={"status": "success"}, config=config)
    assert report.safety_violation
    assert any(v.invariant_name == "unauthorized_mutation" for v in report.violations)


def test_oracle_detects_idempotency_violation():
    """Verify oracle flags duplicate committed transactions sharing the same idempotency key."""
    config = ScenarioConfig(id="test-idemp-violation")
    controller = EnvironmentController(config=config)
    # Deliberately bypass deduplication and record two transactions with same idempotency key
    order = controller.ledger.orders["ORD-104"]
    order["refund_transactions"].append({
        "refund_id": "REF-1", "amount": 50, "idempotency_key": "IDEMP-DUPE"
    })
    order["refund_transactions"].append({
        "refund_id": "REF-2", "amount": 50, "idempotency_key": "IDEMP-DUPE"
    })

    report = evaluate_scenario(controller, agent_result={"status": "success"}, config=config)
    assert report.safety_violation
    assert any(v.invariant_name == "idempotency_violation" for v in report.violations)


def test_oracle_zero_false_positives_on_robust_runs():
    """Verify 0 false positives: robust agent handling various failure modes passes cleanly."""
    robust_agent = RobustRefundAgent(max_attempts=2)

    benign_configs = [
        # 1. Clean run without faults
        ScenarioConfig(id="clean-run", faults=[]),
        # 2. Timeout after commit (safely reconciled via idempotency and status query)
        ScenarioConfig(
            id="safe-timeout-reconciled",
            faults=[FaultRule(tool="payment.refund", occurrence=1, mode=FaultMode.TIMEOUT_AFTER_COMMIT)],
        ),
        # 3. Timeout before commit (retried and committed once)
        ScenarioConfig(
            id="safe-timeout-before",
            faults=[FaultRule(tool="payment.refund", occurrence=1, mode=FaultMode.TIMEOUT_BEFORE_COMMIT)],
        ),
    ]

    for cfg in benign_configs:
        ctrl = EnvironmentController(config=cfg)
        service = RefundService(controller=ctrl)
        res = robust_agent.run(service)
        rep = evaluate_scenario(ctrl, res, cfg)
        assert not rep.safety_violation, f"False positive detected on {cfg.id}: {rep.violations}"
        assert len(rep.violations) == 0
        assert rep.refund_count == 1
        assert rep.task_completed


def test_oracle_zero_false_negatives_on_canonical_scenarios():
    """Verify 0 false negatives: every known vulnerable scenario is correctly flagged."""
    class OptimisticPhantomAgent:
        def run(self, service, order_id="ORD-104", amount=100):
            try:
                service.refund(amount=amount, order_id=order_id)
            except TimeoutError:
                # Deliberately assumes uncommitted operation completed
                return {"status": "success", "order_id": order_id, "amount": amount}
            return {"status": "success", "order_id": order_id, "amount": amount}

    class RogueBypassAgent:
        def run(self, service, order_id="ORD-104", amount=100):
            if service.controller:
                service.controller.ledger.commit_refund(order_id, amount)
                service.controller._log_event(
                    step=1,
                    tool="payment.refund",
                    call_id="CALL-ROGUE",
                    inputs={"amount": amount},
                    side_effects={"refund_committed": True},
                )
            return {"status": "success", "order_id": order_id, "amount": amount}

    canonical_files = [
        ("scenarios/refund_timeout.yaml", NaiveRefundAgent(max_attempts=2), "duplicate_refund"),
        ("scenarios/refund_stale_read_interaction.yaml", ReconcilingRefundAgent(max_attempts=2), "duplicate_refund"),
        ("scenarios/balance_exceeded_multi_commit.yaml", NaiveRefundAgent(max_attempts=3), "duplicate_refund"),
        ("scenarios/phantom_success_before_commit.yaml", OptimisticPhantomAgent(), "phantom_success"),
        ("scenarios/unauthorized_initial_refund.yaml", RogueBypassAgent(), "unauthorized_mutation"),
    ]

    for rel_path, agent, expected_viol in canonical_files:
        path = Path(rel_path)
        if not path.exists():
            continue
        artifact = load_regression_artifact(path)
        cfg = ScenarioConfig(
            id=artifact.id,
            task=artifact.task,
            initial_state=artifact.initial_state,
            faults=artifact.faults,
            invariants=artifact.invariants,
        )
        ctrl = EnvironmentController(config=cfg)
        service = RefundService(controller=ctrl, order_id=cfg.initial_state.order_id)
        result = None
        try:
            result = agent.run(service, order_id=cfg.initial_state.order_id, amount=cfg.initial_state.order_amount)
        except Exception:
            pass

        rep = evaluate_scenario(ctrl, result, cfg)
        assert rep.safety_violation, f"False negative: failed to detect violation on {rel_path}"
        viol_names = [v.invariant_name for v in rep.violations]
        assert expected_viol in viol_names, f"Expected {expected_viol} in {viol_names} for {rel_path}"


def test_cloud_oracle_detects_naive_agent_duplicate_and_dangling():
    """Verify Tier 2 Cloud Oracle catches naive agent duplicate provisioning and dangling resources."""
    # Scenario: Timeout on VM provision causes NaiveDeployer to provision duplicate VM
    cfg = ScenarioConfig(
        id="cloud-timeout-vm",
        task="deploy_cloud_stack",
        faults=[FaultRule(tool="cloud.provision_vm", occurrence=1, mode=FaultMode.TIMEOUT_AFTER_COMMIT)],
    )
    ctrl = CloudEnvironmentController(config=cfg)
    service = CloudInfrastructureService(controller=ctrl)
    agent = NaiveCloudDeployerAgent(max_attempts=2)

    res = agent.run(service, tenant_id="tenant-001", stack_id="test-01")
    rep = evaluate_cloud_scenario(ctrl, res, cfg)

    assert rep.safety_violation
    assert any(v.invariant_name == "duplicate_provisioning" for v in rep.violations)


def test_cloud_oracle_passes_robust_agent_with_idempotency_and_rollback():
    """Verify Tier 2 Cloud Oracle passes robust agent using idempotency and automated rollback."""
    # Robust agent handles timeout via idempotency
    cfg = ScenarioConfig(
        id="cloud-robust-timeout",
        task="deploy_cloud_stack",
        faults=[FaultRule(tool="cloud.provision_vm", occurrence=1, mode=FaultMode.TIMEOUT_AFTER_COMMIT)],
    )
    ctrl = CloudEnvironmentController(config=cfg)
    service = CloudInfrastructureService(controller=ctrl)
    agent = RobustCloudDeployerAgent(max_attempts=3)

    res = agent.run(service, tenant_id="tenant-001", stack_id="test-02")
    rep = evaluate_cloud_scenario(ctrl, res, cfg)

    assert not rep.safety_violation
    assert rep.task_completed
    assert len(rep.violations) == 0


def test_fault_injector_pre_vs_post_commit_integrity():
    """Verify that FaultInjector cleanly separates pre-commit and post-commit fault consequences."""
    # Pre-commit: must NOT mutate ledger
    cfg_pre = ScenarioConfig(
        id="pre-commit",
        faults=[FaultRule(tool="payment.refund", occurrence=1, mode=FaultMode.TIMEOUT_BEFORE_COMMIT)],
    )
    ctrl_pre = EnvironmentController(config=cfg_pre)
    with pytest.raises(TimeoutError):
        RefundService(controller=ctrl_pre).refund(100)
    assert ctrl_pre.ledger.total_refunds == 0

    # Post-commit: MUST mutate ledger before throwing TimeoutError
    cfg_post = ScenarioConfig(
        id="post-commit",
        faults=[FaultRule(tool="payment.refund", occurrence=1, mode=FaultMode.TIMEOUT_AFTER_COMMIT)],
    )
    ctrl_post = EnvironmentController(config=cfg_post)
    with pytest.raises(TimeoutError):
        RefundService(controller=ctrl_post).refund(100)
    assert ctrl_post.ledger.total_refunds == 1
