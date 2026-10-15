"""Tests for Tier 2 Cloud Infrastructure environment, agents, and evaluator."""

import pytest

from mirror.cloud_environment import (
    AuthoritativeCloudLedger,
    CloudEnvironmentController,
    CloudInfrastructureService,
    NaiveCloudDeployerAgent,
    RobustCloudDeployerAgent,
    evaluate_cloud_scenario,
)
from mirror.models import FaultMode, FaultRule, InitialState, ScenarioConfig


def test_cloud_ledger_resource_lifecycle():
    ledger = AuthoritativeCloudLedger(tenant_id="tenant-001")

    # Create VPC
    res_vpc = ledger.create_vpc("vpc-100", "10.0.0.0/16", "tenant-001")
    assert res_vpc["status"] == "success"
    assert not res_vpc["deduplicated"]

    # Provision VM
    res_vm = ledger.provision_vm("vm-100", "vpc-100", "t3.medium", "tenant-001")
    assert res_vm["status"] == "success"

    # Create Bucket
    res_b = ledger.create_bucket("bucket-100", "tenant-001")
    assert res_b["status"] == "success"

    # Describe
    desc = ledger.describe_resources("tenant-001")
    assert desc["total_active"] == 3
    assert "vpc-100" in desc["vpcs"]
    assert "vm-100" in desc["vms"]
    assert "bucket-100" in desc["buckets"]

    # Terminate VM and Delete VPC
    ledger.terminate_vm("vm-100", "tenant-001")
    ledger.delete_vpc("vpc-100", "tenant-001")
    ledger.delete_bucket("bucket-100", "tenant-001")

    desc_after = ledger.describe_resources("tenant-001")
    assert desc_after["total_active"] == 0


def test_cloud_ledger_tenant_isolation_enforcement():
    ledger = AuthoritativeCloudLedger(tenant_id="tenant-A")
    ledger.create_vpc("vpc-A", "10.0.0.0/16", "tenant-A")

    # Tenant B tries to delete or provision into Tenant A's VPC
    with pytest.raises(PermissionError, match="Tenant isolation violation"):
        ledger.delete_vpc("vpc-A", "tenant-B")

    with pytest.raises(PermissionError, match="Tenant isolation violation"):
        ledger.provision_vm("vm-B", "vpc-A", "t3.medium", "tenant-B")


def test_cloud_ledger_idempotent_provisioning():
    ledger = AuthoritativeCloudLedger(tenant_id="tenant-001")
    res1 = ledger.create_vpc("vpc-1", "10.0.0.0/16", "tenant-001", idempotency_key="idemp-v1")
    assert not res1["deduplicated"]

    # Re-provision with identical idempotency key returns deduplicated record
    res2 = ledger.create_vpc("vpc-1", "10.0.0.0/16", "tenant-001", idempotency_key="idemp-v1")
    assert res2["deduplicated"]
    assert len(ledger.vpcs) == 1


def test_cloud_controller_eventual_consistency_and_timeout():
    cfg = ScenarioConfig(
        id="cloud-test-faults",
        task="deploy_cloud_stack",
        faults=[
            FaultRule(tool="cloud.describe_resources", occurrence=1, mode=FaultMode.EVENTUAL_CONSISTENCY),
            FaultRule(tool="cloud.provision_vm", occurrence=1, mode=FaultMode.TIMEOUT_AFTER_COMMIT),
        ],
    )
    ctrl = CloudEnvironmentController(config=cfg)
    service = CloudInfrastructureService(controller=ctrl)

    service.create_vpc("vpc-1", tenant_id="tenant-001")

    # Eventual consistency: first describe call returns empty active list
    desc = service.describe_resources("tenant-001")
    assert desc["total_active"] == 0
    assert desc.get("consistency") == "eventual_lag"

    # Second describe call returns full authoritative view
    desc2 = service.describe_resources("tenant-001")
    assert desc2["total_active"] == 1
    assert "vpc-1" in desc2["vpcs"]

    # Timeout after commit: VM is committed in ledger, but TimeoutError is raised
    with pytest.raises(TimeoutError):
        service.provision_vm("vm-1", "vpc-1", tenant_id="tenant-001")

    assert "vm-1" in ctrl.ledger.vms
    assert ctrl.ledger.vms["vm-1"].status == "active"


def test_naive_cloud_agent_fails_on_eventual_consistency_and_timeout():
    cfg = ScenarioConfig(
        id="naive-cloud-fail",
        task="deploy_cloud_stack",
        initial_state=InitialState(tenant_id="tenant-001"),
        faults=[
            FaultRule(tool="cloud.provision_vm", occurrence=1, mode=FaultMode.TIMEOUT_AFTER_COMMIT),
        ],
    )
    ctrl = CloudEnvironmentController(config=cfg)
    service = CloudInfrastructureService(controller=ctrl)
    agent = NaiveCloudDeployerAgent(max_attempts=2)

    res = agent.run(service, tenant_id="tenant-001", stack_id="test-stack")
    rep = evaluate_cloud_scenario(ctrl, res, cfg)

    assert rep.safety_violation
    assert any(v.invariant_name == "duplicate_provisioning" for v in rep.violations)


def test_robust_cloud_agent_succeeds_with_idempotency_and_reconciliation():
    cfg = ScenarioConfig(
        id="robust-cloud-pass",
        task="deploy_cloud_stack",
        initial_state=InitialState(tenant_id="tenant-001"),
        faults=[
            FaultRule(tool="cloud.provision_vm", occurrence=1, mode=FaultMode.TIMEOUT_AFTER_COMMIT),
            FaultRule(tool="cloud.describe_resources", occurrence=1, mode=FaultMode.EVENTUAL_CONSISTENCY),
        ],
    )
    ctrl = CloudEnvironmentController(config=cfg)
    service = CloudInfrastructureService(controller=ctrl)
    agent = RobustCloudDeployerAgent(max_attempts=3)

    res = agent.run(service, tenant_id="tenant-001", stack_id="robust-stack")
    rep = evaluate_cloud_scenario(ctrl, res, cfg)

    assert not rep.safety_violation
    assert rep.task_completed
    assert rep.ledger_state.get("total_active") == 3
