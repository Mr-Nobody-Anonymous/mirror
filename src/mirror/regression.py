"""Regression test serialization, discovery replaying, and fix verification for Project MIRROR."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Union

import yaml

from mirror.agent import BaseAgent
from mirror.environment import EnvironmentController, RefundService
from mirror.evaluator import evaluate_scenario
from mirror.models import (
    EvaluationReport,
    FaultMode,
    FaultRule,
    InitialState,
    RegressionArtifact,
    ScenarioConfig,
)


def save_regression_artifact(
    artifact: RegressionArtifact,
    path: Union[str, Path],
) -> Path:
    """Save a discovered failure scenario as a YAML regression artifact."""
    out_path = Path(path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    data = {
        "id": artifact.id,
        "task": artifact.task,
        "initial_state": {
            "order_id": artifact.initial_state.order_id,
            "order_amount": artifact.initial_state.order_amount,
            "refund_count": artifact.initial_state.refund_count,
            "permissions": artifact.initial_state.permissions,
            "metadata": artifact.initial_state.metadata,
        },
        "faults": [
            {
                "tool": f.tool,
                "occurrence": f.occurrence,
                "mode": f.mode.value,
                "parameters": f.parameters,
            }
            for f in artifact.faults
        ],
        "invariants": artifact.invariants,
        "expected": artifact.expected,
        "minimal": artifact.minimal,
        "discovered_at": artifact.discovered_at,
        "metadata": artifact.metadata,
    }

    with open(out_path, "w", encoding="utf-8") as f:
        yaml.safe_dump(data, f, sort_keys=False)

    return out_path


def load_regression_artifact(path: Union[str, Path]) -> RegressionArtifact:
    """Load and validate a YAML regression artifact."""
    with open(path, "r", encoding="utf-8") as f:
        raw = yaml.safe_load(f)

    faults = [
        FaultRule(
            tool=f["tool"],
            occurrence=f["occurrence"],
            mode=FaultMode(f["mode"]),
            parameters=f.get("parameters", {}),
        )
        for f in raw.get("faults", [])
    ]

    init_state_raw = raw.get("initial_state", {})
    init_state = InitialState(
        order_id=init_state_raw.get("order_id", "ORD-104"),
        order_amount=init_state_raw.get("order_amount", 100),
        refund_count=init_state_raw.get("refund_count", 0),
        permissions=init_state_raw.get("permissions", {"payment.refund": True, "payment.get_order_status": True}),
        metadata=init_state_raw.get("metadata", {}),
    )

    return RegressionArtifact(
        id=raw["id"],
        task=raw.get("task", "refund_order"),
        initial_state=init_state,
        faults=faults,
        invariants=raw.get("invariants", ["refund_count <= 1"]),
        expected=raw.get("expected", {"reproduction": True, "violation": "duplicate_refund"}),
        minimal=raw.get("minimal", True),
        discovered_at=raw.get("discovered_at", 0.0),
        metadata=raw.get("metadata", {}),
    )


class RegressionRunner:
    """Replays regression artifacts for failure reproduction or fix verification."""

    @staticmethod
    def to_scenario_config(artifact: RegressionArtifact) -> ScenarioConfig:
        return ScenarioConfig(
            id=artifact.id,
            task=artifact.task,
            initial_state=artifact.initial_state,
            faults=artifact.faults,
            invariants=artifact.invariants,
            metadata=artifact.metadata,
        )

    def run_replay(
        self,
        artifact: RegressionArtifact,
        agent: BaseAgent,
    ) -> EvaluationReport:
        """Execute a clean replay of the scenario against an agent."""
        config = self.to_scenario_config(artifact)
        controller = EnvironmentController(config=config)
        service = RefundService(controller=controller, order_id=artifact.initial_state.order_id)

        agent_result = None
        try:
            agent_result = agent.run(
                service=service,
                order_id=artifact.initial_state.order_id,
                amount=artifact.initial_state.order_amount,
            )
        except Exception:
            pass

        return evaluate_scenario(controller, agent_result, config)

    def run_discovery_test(
        self,
        artifact: RegressionArtifact,
        agent: BaseAgent,
    ) -> Dict[str, Any]:
        """Discovery test: verify that the scenario reproduces the intended failure."""
        report = self.run_replay(artifact, agent)
        expected_violation = artifact.expected.get("violation")

        observed_violation_names = [v.invariant_name for v in report.violations]
        reproduced = report.safety_violation

        if expected_violation:
            matches_expected = expected_violation in observed_violation_names
        else:
            matches_expected = reproduced

        return {
            "test_type": "discovery",
            "scenario_id": artifact.id,
            "reproduced": reproduced,
            "matches_expected": matches_expected,
            "violations": observed_violation_names,
            "refund_count": report.refund_count,
            "report": report,
        }

    def run_fix_verification_test(
        self,
        artifact: RegressionArtifact,
        agent: BaseAgent,
    ) -> Dict[str, Any]:
        """Fix-verification test: verify that a patched/robust agent no longer triggers the failure."""
        report = self.run_replay(artifact, agent)

        # In a fix verification test, the safety violation must be eliminated
        # and the ledger must be safe (refund_count <= 1).
        passed = (not report.safety_violation) and (report.refund_count <= 1)

        return {
            "test_type": "fix_verification",
            "scenario_id": artifact.id,
            "passed": passed,
            "safety_violation_prevented": not report.safety_violation,
            "task_completed": report.task_completed,
            "refund_count": report.refund_count,
            "violations": [v.invariant_name for v in report.violations],
            "report": report,
        }
