"""Scenario minimization engine using Delta Debugging (ddmin) for Project MIRROR."""

from __future__ import annotations

import copy
import logging
from typing import Callable, List, Optional, Set, Tuple
from mirror.models import FaultRule, ScenarioConfig, EvaluationReport
from mirror.environment import EnvironmentController, RefundService
from mirror.evaluator import evaluate_scenario
from mirror.agent import BaseAgent

logger = logging.getLogger(__name__)


class ScenarioMinimizer:
    """Minimizes a failure-inducing fault configuration to its 1-minimal subset.

    Implements Andreas Zeller's Delta Debugging (ddmin) algorithm adapted
    for environment fault injections.
    """

    def __init__(
        self,
        agent: BaseAgent,
        target_violation_type: Optional[str] = "duplicate_refund",
    ):
        self.agent = agent
        self.target_violation_type = target_violation_type
        self.evaluation_count = 0

    def test_fault_subset(
        self,
        base_scenario: ScenarioConfig,
        candidate_faults: List[FaultRule],
    ) -> Tuple[bool, EvaluationReport]:
        """Execute agent with candidate faults and test if target invariant violation occurs."""
        self.evaluation_count += 1
        test_cfg = copy.deepcopy(base_scenario)
        test_cfg.faults = candidate_faults

        controller = EnvironmentController(config=test_cfg)
        service = RefundService(controller=controller, order_id=test_cfg.initial_state.order_id)

        agent_result = None
        try:
            agent_result = self.agent.run(
                service=service,
                order_id=test_cfg.initial_state.order_id,
                amount=test_cfg.initial_state.order_amount,
            )
        except Exception:
            # Exceptions may be part of the failure sequence
            pass

        report = evaluate_scenario(controller, agent_result, test_cfg)

        if not report.safety_violation:
            return False, report

        if self.target_violation_type:
            has_target = any(v.invariant_name == self.target_violation_type for v in report.violations)
            return has_target, report

        return True, report

    def minimize(
        self,
        scenario: ScenarioConfig,
        method: str = "ddmin",
    ) -> Tuple[ScenarioConfig, int]:
        """Reduce faults in scenario to minimal subset.

        Returns (minimized_scenario, number_of_evaluations).
        """
        self.evaluation_count = 0
        original_faults = list(scenario.faults)

        # Baseline check: confirm original scenario actually reproduces the violation
        reproduces, initial_report = self.test_fault_subset(scenario, original_faults)
        if not reproduces:
            logger.warning("Scenario does not reproduce target violation before minimization.")
            return scenario, self.evaluation_count

        # Check if 0 faults triggers the violation (unconditionally buggy agent)
        reproduces_empty, _ = self.test_fault_subset(scenario, [])
        if reproduces_empty:
            minimized_cfg = copy.deepcopy(scenario)
            minimized_cfg.faults = []
            minimized_cfg.metadata["minimized"] = True
            minimized_cfg.metadata["evaluations_used"] = self.evaluation_count
            minimized_cfg.metadata["original_fault_count"] = len(original_faults)
            minimized_cfg.metadata["minimal_fault_count"] = 0
            return minimized_cfg, self.evaluation_count

        if len(original_faults) <= 1:
            return scenario, self.evaluation_count

        if method == "ddmin":
            minimal_faults = self._ddmin(scenario, original_faults)
        else:
            minimal_faults = self._linear_minimize(scenario, original_faults)

        minimized_cfg = copy.deepcopy(scenario)
        minimized_cfg.faults = minimal_faults
        minimized_cfg.metadata["minimized"] = True
        minimized_cfg.metadata["evaluations_used"] = self.evaluation_count
        minimized_cfg.metadata["original_fault_count"] = len(original_faults)
        minimized_cfg.metadata["minimal_fault_count"] = len(minimal_faults)

        return minimized_cfg, self.evaluation_count

    def _ddmin(
        self,
        base_scenario: ScenarioConfig,
        faults: List[FaultRule],
    ) -> List[FaultRule]:
        """Classic ddmin partitioning algorithm (Zeller 2002) using index complement slicing."""
        c = list(faults)
        n = 2

        while len(c) >= 2:
            partitions = self._partition_indices(len(c), n)
            reduced = False

            # Test individual subsets
            for start, end in partitions:
                s = c[start:end]
                reproduces, _ = self.test_fault_subset(base_scenario, s)
                if reproduces:
                    c = s
                    n = 2
                    reduced = True
                    break

            if reduced:
                continue

            # Test complements (c \ s) via index slicing
            for start, end in partitions:
                complement = c[:start] + c[end:]
                if not complement:
                    continue
                reproduces, _ = self.test_fault_subset(base_scenario, complement)
                if reproduces:
                    c = complement
                    n = max(n - 1, 2)
                    reduced = True
                    break

            if not reduced:
                if n < len(c):
                    n = min(len(c), 2 * n)
                else:
                    break

        return c

    def _linear_minimize(
        self,
        base_scenario: ScenarioConfig,
        faults: List[FaultRule],
    ) -> List[FaultRule]:
        """Simple greedy 1-minimization pass removing one fault at a time."""
        current = list(faults)
        i = 0
        while i < len(current):
            candidate = current[:i] + current[i + 1:]
            reproduces, _ = self.test_fault_subset(base_scenario, candidate)
            if reproduces:
                current = candidate
            else:
                i += 1
        return current

    @staticmethod
    def _partition_indices(total: int, n: int) -> List[Tuple[int, int]]:
        """Return (start, end) index intervals partitioning `total` items into n chunks."""
        k, m = divmod(total, n)
        intervals = []
        start = 0
        for i in range(n):
            end = start + k + (1 if i < m else 0)
            if start < end:
                intervals.append((start, end))
            start = end
        return intervals

    @classmethod
    def _partition(cls, items: List[FaultRule], n: int) -> List[List[FaultRule]]:
        """Split items into n roughly equal chunks."""
        indices = cls._partition_indices(len(items), n)
        return [items[start:end] for start, end in indices]
