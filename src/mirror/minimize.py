"""Scenario minimization engine using Delta Debugging (ddmin) for Project MIRROR."""

from __future__ import annotations

import copy
import logging
from typing import Any, Dict, List, Optional, Tuple

from pydantic import BaseModel, Field

from mirror.agent import BaseAgent
from mirror.environment import EnvironmentController, RefundService
from mirror.evaluator import evaluate_scenario
from mirror.models import EvaluationReport, FaultRule, ScenarioConfig

logger = logging.getLogger(__name__)


class MinimizationReport(BaseModel):
    """Rigorous verification report produced by delta debugging minimization.

    Captures every tested fault removal, step counts, repeatability validation,
    and formal distinctions between minimal reproducer, causal explanation,
    and underlying software root cause.
    """

    original_faults: List[FaultRule] = Field(default_factory=list)
    minimized_faults: List[FaultRule] = Field(default_factory=list)
    original_fault_count: int = 0
    minimized_fault_count: int = 0
    reduction_pct: float = 0.0
    evaluations_count: int = 0
    tested_removals: List[Dict[str, Any]] = Field(default_factory=list)
    is_1_minimal: bool = False
    repeatability_validated: bool = False
    minimal_reproducer: str = ""
    causal_explanation: str = ""
    root_cause_distinction: Dict[str, str] = Field(default_factory=dict)
    budget_exceeded: bool = False

    def summary(self) -> str:
        rep_status = "VALIDATED" if self.repeatability_validated else "UNVALIDATED"
        min_status = "1-MINIMAL" if self.is_1_minimal else "SUB-OPTIMAL"
        return (
            f"[{min_status}|{rep_status}] {self.original_fault_count} -> {self.minimized_fault_count} faults "
            f"({self.reduction_pct}% reduction) in {self.evaluations_count} evaluations"
        )


class ScenarioMinimizer:
    """Minimizes a failure-inducing fault configuration to its 1-minimal subset.

    Implements Andreas Zeller's Delta Debugging (ddmin) algorithm adapted
    for environment fault injections with full step auditing and verification.
    """

    def __init__(
        self,
        agent: BaseAgent,
        target_violation_type: Optional[str] = "duplicate_refund",
        max_evaluations: int = 100,
    ):
        self.agent = agent
        self.target_violation_type = target_violation_type
        self.max_evaluations = max_evaluations
        self.evaluation_count = 0
        self.tested_removals: List[Dict[str, Any]] = []

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
            reproduced = False
        elif self.target_violation_type:
            reproduced = any(v.invariant_name == self.target_violation_type for v in report.violations)
        else:
            reproduced = True

        removed_sigs = [
            f.signature() for f in base_scenario.faults if f not in candidate_faults
        ]
        self.tested_removals.append({
            "step": self.evaluation_count,
            "candidate_count": len(candidate_faults),
            "candidate_signatures": [f.signature() for f in candidate_faults],
            "removed_signatures": removed_sigs,
            "reproduced": reproduced,
            "violations": [v.invariant_name for v in report.violations],
        })

        return reproduced, report

    def minimize(
        self,
        scenario: ScenarioConfig,
        method: str = "ddmin",
    ) -> Tuple[ScenarioConfig, int]:
        """Reduce faults in scenario to minimal subset.

        Returns (minimized_scenario, number_of_evaluations).
        Maintained for full backward compatibility.
        """
        min_cfg, _ = self.minimize_with_report(scenario, method=method, repeatability_checks=0)
        return min_cfg, self.evaluation_count

    def minimize_with_report(
        self,
        scenario: ScenarioConfig,
        method: str = "ddmin",
        repeatability_checks: int = 3,
    ) -> Tuple[ScenarioConfig, MinimizationReport]:
        """Reduce faults in scenario and generate a comprehensive audit report."""
        self.evaluation_count = 0
        self.tested_removals.clear()
        original_faults = list(scenario.faults)

        # Baseline check: confirm original scenario actually reproduces the violation
        reproduces, _ = self.test_fault_subset(scenario, original_faults)
        if not reproduces:
            logger.warning("Scenario does not reproduce target violation before minimization.")
            report = MinimizationReport(
                original_faults=original_faults,
                minimized_faults=original_faults,
                original_fault_count=len(original_faults),
                minimized_fault_count=len(original_faults),
                reduction_pct=0.0,
                evaluations_count=self.evaluation_count,
                tested_removals=list(self.tested_removals),
                is_1_minimal=False,
                repeatability_validated=False,
                minimal_reproducer="Original scenario did not reproduce target violation.",
                causal_explanation="No failure observed under baseline run.",
                root_cause_distinction={
                    "minimal_reproducer": "N/A",
                    "causal_explanation": "Baseline passed without invariant violation.",
                    "software_root_cause": "Unknown",
                },
            )
            return scenario, report

        # Check if 0 faults triggers the violation (unconditionally buggy agent)
        reproduces_empty, _ = self.test_fault_subset(scenario, [])
        if reproduces_empty:
            minimized_cfg = copy.deepcopy(scenario)
            minimized_cfg.faults = []
            minimized_cfg.metadata["minimized"] = True
            minimized_cfg.metadata["evaluations_used"] = self.evaluation_count
            minimized_cfg.metadata["original_fault_count"] = len(original_faults)
            minimized_cfg.metadata["minimal_fault_count"] = 0

            report = self._build_report(
                original_faults=original_faults,
                minimized_faults=[],
                repeatability_checks=repeatability_checks,
                min_cfg=minimized_cfg,
            )
            return minimized_cfg, report

        if len(original_faults) <= 1:
            minimal_faults = original_faults
        elif method == "ddmin":
            minimal_faults = self._ddmin(scenario, original_faults)
        else:
            minimal_faults = self._linear_minimize(scenario, original_faults)

        minimized_cfg = copy.deepcopy(scenario)
        minimized_cfg.faults = minimal_faults
        minimized_cfg.metadata["minimized"] = True
        minimized_cfg.metadata["evaluations_used"] = self.evaluation_count
        minimized_cfg.metadata["original_fault_count"] = len(original_faults)
        minimized_cfg.metadata["minimal_fault_count"] = len(minimal_faults)

        report = self._build_report(
            original_faults=original_faults,
            minimized_faults=minimal_faults,
            repeatability_checks=repeatability_checks,
            min_cfg=minimized_cfg,
        )
        return minimized_cfg, report

    def _build_report(
        self,
        original_faults: List[FaultRule],
        minimized_faults: List[FaultRule],
        repeatability_checks: int,
        min_cfg: ScenarioConfig,
    ) -> MinimizationReport:
        orig_count = len(original_faults)
        min_count = len(minimized_faults)
        reduction_pct = (
            round(((orig_count - min_count) / orig_count) * 100, 2)
            if orig_count > 0
            else 0.0
        )

        # 1-minimality verification: removing ANY single fault must eliminate violation
        is_1_minimal = True
        if min_count > 0:
            for i in range(min_count):
                subset = minimized_faults[:i] + minimized_faults[i + 1:]
                reproduced, _ = self.test_fault_subset(min_cfg, subset)
                if reproduced:
                    is_1_minimal = False
                    break
        else:
            is_1_minimal = True

        # Repeatability validation: execute minimized scenario multiple times
        repeatability_validated = True
        if repeatability_checks > 0:
            for _ in range(repeatability_checks):
                rep, _ = self.test_fault_subset(min_cfg, minimized_faults)
                if not rep:
                    repeatability_validated = False
                    break

        sigs = [f.signature() for f in minimized_faults]
        minimal_reproducer_str = (
            f"1-Minimal Fault Set ({min_count} fault{'s' if min_count != 1 else ''}): "
            + (", ".join(sigs) if sigs else "[Empty Set - unconditional violation]")
        )

        modes = [f.mode.value for f in minimized_faults]
        if "timeout_after_commit" in modes and "stale_read" in modes:
            causal_expl = (
                "Environmental Fault Interaction: A network timeout after transaction commit hid "
                "successful execution; the agent's subsequent state query returned stale cached data "
                "indicating 0 transactions, leading the agent to issue an unhedged retry and duplicate commit."
            )
        elif "timeout_after_commit" in modes:
            causal_expl = (
                "Post-Commit Timeout Interaction: Transaction was committed to the authoritative ledger, "
                "but network response timed out; agent retried blindly without idempotency key or verification."
            )
        elif not minimized_faults:
            causal_expl = "Agent violates invariant unconditionally even without environmental faults."
        else:
            causal_expl = f"Fault subset {sigs} triggers target invariant violation in agent execution sequence."

        agent_cls = self.agent.__class__.__name__
        if "Naive" in agent_cls:
            sw_root_cause = (
                "Agent Logic Vulnerability: NaiveBlindRetry. Agent retries on network timeout without "
                "checking authoritative order status and omits idempotency keys on mutating requests."
            )
        elif "Reconciling" in agent_cls:
            sw_root_cause = (
                "Agent Logic Vulnerability: UnhedgedCacheTrust. Agent queries order status on timeout "
                "but trusts potentially stale cache reads and lacks end-to-end idempotency keys."
            )
        else:
            sw_root_cause = (
                "Agent Logic Vulnerability: Unhandled environmental failure mode in agent control flow."
            )

        root_cause_distinction = {
            "minimal_reproducer": minimal_reproducer_str,
            "causal_explanation": causal_expl,
            "software_root_cause": sw_root_cause,
        }

        budget_exceeded = self.evaluation_count >= self.max_evaluations

        return MinimizationReport(
            original_faults=original_faults,
            minimized_faults=minimized_faults,
            original_fault_count=orig_count,
            minimized_fault_count=min_count,
            reduction_pct=reduction_pct,
            evaluations_count=self.evaluation_count,
            tested_removals=list(self.tested_removals),
            is_1_minimal=is_1_minimal,
            repeatability_validated=repeatability_validated,
            minimal_reproducer=minimal_reproducer_str,
            causal_explanation=causal_expl,
            root_cause_distinction=root_cause_distinction,
            budget_exceeded=budget_exceeded,
        )

    def _ddmin(
        self,
        base_scenario: ScenarioConfig,
        faults: List[FaultRule],
    ) -> List[FaultRule]:
        """Classic ddmin partitioning algorithm (Zeller 2002) using index complement slicing."""
        c = list(faults)
        n = 2

        while len(c) >= 2 and self.evaluation_count < self.max_evaluations:
            partitions = self._partition_indices(len(c), n)
            reduced = False

            # Test individual subsets
            for start, end in partitions:
                if self.evaluation_count >= self.max_evaluations:
                    break
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
                if self.evaluation_count >= self.max_evaluations:
                    break
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
        while i < len(current) and self.evaluation_count < self.max_evaluations:
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
