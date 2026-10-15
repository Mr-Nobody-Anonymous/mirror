"""Scenario search engine (Random and Adaptive feedback-guided) for Project MIRROR."""

from __future__ import annotations

import copy
import math
import random
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set, Tuple
from mirror.models import FaultMode, FaultRule, InitialState, ScenarioConfig, EvaluationReport
from mirror.faults import FaultCatalog
from mirror.environment import EnvironmentController, RefundService
from mirror.evaluator import evaluate_scenario
from mirror.agent import BaseAgent


@dataclass
class SearchResult:
    """Summary of a search run across multiple candidate scenarios."""

    total_experiments: int
    unique_violations_found: int
    first_discovery_step: Optional[int]
    discovered_scenarios: List[ScenarioConfig] = field(default_factory=list)
    reports: List[EvaluationReport] = field(default_factory=list)
    discovery_history: List[Dict[str, Any]] = field(default_factory=list)
    mode_yield_stats: Dict[str, Dict[str, int]] = field(default_factory=dict)


class RandomSearchEngine:
    """Baseline: samples fault configurations uniformly at random."""

    def __init__(
        self,
        agent: BaseAgent,
        max_faults_per_scenario: int = 3,
        seed: Optional[int] = None,
    ):
        self.agent = agent
        self.max_faults = max_faults_per_scenario
        self.rng = random.Random(seed)

    def generate_candidate(self, scenario_id: str) -> ScenarioConfig:
        """Sample a candidate scenario uniformly."""
        num_faults = self.rng.randint(1, self.max_faults)
        faults: List[FaultRule] = []
        used_slots: Set[Tuple[str, int]] = set()

        for _ in range(num_faults):
            tool = self.rng.choice(FaultCatalog.SUPPORTED_TOOLS)
            occ = self.rng.randint(1, 2)
            slot = (tool, occ)
            if slot in used_slots:
                continue
            used_slots.add(slot)

            valid_modes = FaultCatalog.get_valid_modes_for_tool(tool)
            mode = self.rng.choice(valid_modes)
            faults.append(FaultRule(tool=tool, occurrence=occ, mode=mode))

        return ScenarioConfig(
            id=scenario_id,
            initial_state=InitialState(),
            faults=faults,
        )

    def run(self, budget: int) -> SearchResult:
        """Execute random search over fixed experiment budget."""
        discovered: List[ScenarioConfig] = []
        reports: List[EvaluationReport] = []
        history: List[Dict[str, Any]] = []
        first_discovery: Optional[int] = None
        seen_violations: Set[str] = set()

        for step in range(1, budget + 1):
            cand = self.generate_candidate(f"rand-{step:04d}")
            controller = EnvironmentController(config=cand)
            service = RefundService(controller=controller)

            agent_result = None
            try:
                agent_result = self.agent.run(
                    service=service,
                    order_id=cand.initial_state.order_id,
                    amount=cand.initial_state.order_amount,
                )
            except Exception:
                pass

            rep = evaluate_scenario(controller, agent_result, cand)
            if rep.safety_violation:
                if first_discovery is None:
                    first_discovery = step
                v_sig = "-".join(sorted(v.invariant_name for v in rep.violations))
                seen_violations.add(v_sig)
                discovered.append(cand)
                reports.append(rep)
                history.append({
                    "step": step,
                    "scenario_id": cand.id,
                    "violations": [v.invariant_name for v in rep.violations],
                    "faults": [f.signature() for f in cand.faults],
                })

        return SearchResult(
            total_experiments=budget,
            unique_violations_found=len(seen_violations),
            first_discovery_step=first_discovery,
            discovered_scenarios=discovered,
            reports=reports,
            discovery_history=history,
        )


class AdaptiveSearchEngine:
    """Feedback-guided scenario search using Upper Confidence Bound (UCB).

    Maintains exploration-exploitation statistics for individual fault modes
    and pairwise fault interactions, prioritizing condition combinations
    that correlate with invariant violations.
    """

    def __init__(
        self,
        agent: BaseAgent,
        max_faults_per_scenario: int = 3,
        exploration_constant: float = 1.414,
        seed: Optional[int] = None,
    ):
        self.agent = agent
        self.max_faults = max_faults_per_scenario
        self.c = exploration_constant
        self.rng = random.Random(seed)

        # Multi-armed bandit statistics for fault modes
        self.mode_trials: Dict[str, int] = {m.value: 0 for m in FaultMode if m != FaultMode.NONE}
        self.mode_violations: Dict[str, int] = {m.value: 0 for m in FaultMode if m != FaultMode.NONE}

        # Pairwise interaction stats
        self.pair_trials: Dict[str, int] = {}
        self.pair_violations: Dict[str, int] = {}

    def _ucb_score(self, mode: str, total_steps: int) -> float:
        """Compute UCB1 value for a fault mode."""
        trials = self.mode_trials.get(mode, 0)
        if trials == 0:
            return float("inf")
        success_rate = self.mode_violations.get(mode, 0) / trials
        bonus = self.c * math.sqrt(math.log(max(total_steps, 1)) / trials)
        return success_rate + bonus

    def select_mode_for_tool(self, tool: str, total_steps: int) -> FaultMode:
        """Select fault mode using UCB with fallback to random exploration."""
        valid_modes = [m for m in FaultCatalog.get_valid_modes_for_tool(tool) if m != FaultMode.NONE]

        # Epsilon-greedy exploration probability decaying slightly
        if self.rng.random() < 0.2:
            return self.rng.choice(valid_modes)

        # UCB-guided selection
        scores = [(m, self._ucb_score(m.value, total_steps)) for m in valid_modes]
        scores.sort(key=lambda x: x[1], reverse=True)
        return scores[0][0]

    def generate_candidate(self, scenario_id: str, step: int) -> ScenarioConfig:
        """Generate feedback-guided candidate scenario."""
        num_faults = self.rng.randint(1, self.max_faults)
        faults: List[FaultRule] = []
        used_slots: Set[Tuple[str, int]] = set()

        for _ in range(num_faults):
            tool = self.rng.choice(FaultCatalog.SUPPORTED_TOOLS)
            occ = self.rng.randint(1, 2)
            slot = (tool, occ)
            if slot in used_slots:
                continue
            used_slots.add(slot)

            mode = self.select_mode_for_tool(tool, step)
            faults.append(FaultRule(tool=tool, occurrence=occ, mode=mode))

        return ScenarioConfig(
            id=scenario_id,
            initial_state=InitialState(),
            faults=faults,
        )

    def run(self, budget: int) -> SearchResult:
        """Execute adaptive search over fixed experiment budget."""
        discovered: List[ScenarioConfig] = []
        reports: List[EvaluationReport] = []
        history: List[Dict[str, Any]] = []
        first_discovery: Optional[int] = None
        seen_violations: Set[str] = set()

        for step in range(1, budget + 1):
            cand = self.generate_candidate(f"adapt-{step:04d}", step)
            controller = EnvironmentController(config=cand)
            service = RefundService(controller=controller)

            agent_result = None
            try:
                agent_result = self.agent.run(
                    service=service,
                    order_id=cand.initial_state.order_id,
                    amount=cand.initial_state.order_amount,
                )
            except Exception:
                pass

            rep = evaluate_scenario(controller, agent_result, cand)

            # Update bandit feedback statistics
            violation_occurred = rep.safety_violation
            modes_in_cand = set(f.mode.value for f in cand.faults)

            for m in modes_in_cand:
                self.mode_trials[m] = self.mode_trials.get(m, 0) + 1
                if violation_occurred:
                    self.mode_violations[m] = self.mode_violations.get(m, 0) + 1

            if violation_occurred:
                if first_discovery is None:
                    first_discovery = step
                v_sig = "-".join(sorted(v.invariant_name for v in rep.violations))
                seen_violations.add(v_sig)
                discovered.append(cand)
                reports.append(rep)
                history.append({
                    "step": step,
                    "scenario_id": cand.id,
                    "violations": [v.invariant_name for v in rep.violations],
                    "faults": [f.signature() for f in cand.faults],
                })

        yield_stats = {
            m: {"trials": self.mode_trials[m], "violations": self.mode_violations[m]}
            for m in self.mode_trials
        }

        return SearchResult(
            total_experiments=budget,
            unique_violations_found=len(seen_violations),
            first_discovery_step=first_discovery,
            discovered_scenarios=discovered,
            reports=reports,
            discovery_history=history,
            mode_yield_stats=yield_stats,
        )
