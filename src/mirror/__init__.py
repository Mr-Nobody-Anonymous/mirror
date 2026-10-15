"""Project MIRROR: Research-grade crash-test simulator for AI agents.

Systematically perturbs agent execution environments, discovers invariant
violations, minimizes failure-inducing fault combinations via Delta Debugging,
and compiles discovered failures into replayable regression tests.
"""

__version__ = "0.1.0"

from mirror.models import (
    FaultMode,
    FaultRule,
    EventRecord,
    InitialState,
    ScenarioConfig,
    InvariantViolation,
    EvaluationReport,
    RegressionArtifact,
)
from mirror.environment import (
    RefundService,
    AuthoritativeLedger,
    EnvironmentController,
)
from mirror.evaluator import evaluate_scenario, evaluate_refund
from mirror.agent import BaseAgent, NaiveRefundAgent, RobustRefundAgent

__all__ = [
    "FaultMode",
    "FaultRule",
    "EventRecord",
    "InitialState",
    "ScenarioConfig",
    "InvariantViolation",
    "EvaluationReport",
    "RegressionArtifact",
    "RefundService",
    "AuthoritativeLedger",
    "EnvironmentController",
    "evaluate_scenario",
    "evaluate_refund",
    "BaseAgent",
    "NaiveRefundAgent",
    "RobustRefundAgent",
]
