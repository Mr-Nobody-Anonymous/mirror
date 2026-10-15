"""Project MIRROR: Research-grade crash-test simulator for AI agents.

Systematically perturbs agent execution environments, discovers invariant
violations, minimizes failure-inducing fault combinations via Delta Debugging,
and compiles discovered failures into replayable regression tests.
"""

__version__ = "0.1.0"

from mirror.agent import BaseAgent, NaiveRefundAgent, ReconcilingRefundAgent, RobustRefundAgent
from mirror.environment import (
    AuthoritativeLedger,
    EnvironmentController,
    RefundService,
)
from mirror.evaluator import evaluate_refund, evaluate_scenario
from mirror.minimize import ScenarioMinimizer
from mirror.models import (
    EvaluationReport,
    EventRecord,
    FaultMode,
    FaultRule,
    InitialState,
    InvariantViolation,
    RegressionArtifact,
    ScenarioConfig,
)
from mirror.regression import (
    RegressionRunner,
    load_regression_artifact,
    save_regression_artifact,
)
from mirror.search import AdaptiveSearchEngine, RandomSearchEngine, SearchResult

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
    "ReconcilingRefundAgent",
    "RandomSearchEngine",
    "AdaptiveSearchEngine",
    "SearchResult",
    "ScenarioMinimizer",
    "RegressionRunner",
    "load_regression_artifact",
    "save_regression_artifact",
]
