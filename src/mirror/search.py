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
    pair_yield_stats: Dict[str, Dict[str, int]] = field(default_factory=dict)


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

        # Multi-armed bandit statistics for individual fault modes
        self.mode_trials: Dict[str, int] = {m.value: 0 for m in FaultMode if m != FaultMode.NONE}
        self.mode_violations: Dict[str, int] = {m.value: 0 for m in FaultMode if m != FaultMode.NONE}

        # Multi-armed bandit statistics for specific (tool, mode) arms
        self.tool_mode_trials: Dict[str, int] = {}
        self.tool_mode_violations: Dict[str, int] = {}

        # Multi-armed bandit statistics for pairwise fault combinations
        self.pair_trials: Dict[Tuple[str, str], int] = {}
        self.pair_violations: Dict[Tuple[str, str], int] = {}

    def _ucb_score(self, tool: str, mode: str, total_steps: int) -> float:
        """Compute UCB1 value for a specific (tool, mode) fault arm."""
        key = f"{tool}:{mode}"
        trials = self.tool_mode_trials.get(key, 0)
        if trials == 0:
            return float("inf")
        success_rate = self.tool_mode_violations.get(key, 0) / trials
        bonus = self.c * math.sqrt(math.log(max(total_steps, 1)) / trials)
        return success_rate + bonus

    def _ucb_pair_score(self, pair: Tuple[str, str], total_steps: int) -> float:
        """Compute UCB1 value for a pairwise fault combination."""
        trials = self.pair_trials.get(pair, 0)
        if trials == 0:
            return float("inf")
        success_rate = self.pair_violations.get(pair, 0) / trials
        bonus = self.c * math.sqrt(math.log(max(total_steps, 1)) / trials)
        return success_rate + bonus

    def select_mode_for_tool(self, tool: str, total_steps: int) -> FaultMode:
        """Select fault mode using UCB with fallback to random exploration and random tie-breaking."""
        valid_modes = [m for m in FaultCatalog.get_valid_modes_for_tool(tool) if m != FaultMode.NONE]

        # Epsilon-greedy exploration probability decaying slightly
        if self.rng.random() < 0.15:
            return self.rng.choice(valid_modes)

        # UCB-guided selection with randomized tie-breaking
        self.rng.shuffle(valid_modes)
        scores = [(m, self._ucb_score(tool, m.value, total_steps)) for m in valid_modes]
        scores.sort(key=lambda x: x[1], reverse=True)
        return scores[0][0]

    def generate_candidate(self, scenario_id: str, step: int) -> ScenarioConfig:
        """Generate feedback-guided candidate scenario prioritizing high-yield pairs and modes."""
        num_faults = self.rng.randint(1, self.max_faults)
        faults: List[FaultRule] = []
        used_slots: Set[Tuple[str, int]] = set()

        # If multiple faults requested and we have discovered promising pairs, exploit them
        if num_faults >= 2 and self.pair_violations and self.rng.random() < 0.6:
            promising_pairs = [p for p, v in self.pair_violations.items() if v > 0]
            if promising_pairs:
                self.rng.shuffle(promising_pairs)
                promising_pairs.sort(key=lambda p: self._ucb_pair_score(p, step), reverse=True)
                best_pair = promising_pairs[0]
                for sig in best_pair:
                    try:
                        tool_part, mode_part = sig.split(":")
                        tool, occ_str = tool_part.split("#")
                        occ = int(occ_str)
                        slot = (tool, occ)
                        if slot not in used_slots:
                            used_slots.add(slot)
                            faults.append(FaultRule(tool=tool, occurrence=occ, mode=FaultMode(mode_part)))
                    except Exception:
                        pass

        # For multi-fault exploration, prioritize testing interactions between distinct tools
        candidate_tools = list(FaultCatalog.SUPPORTED_TOOLS)
        self.rng.shuffle(candidate_tools)

        tool_idx = 0
        while len(faults) < num_faults and len(used_slots) < 4:
            tool = candidate_tools[tool_idx % len(candidate_tools)]
            tool_idx += 1
            # Primary occurrence is 1 (where state interactions initiate)
            occ = 1 if self.rng.random() < 0.75 else 2
            slot = (tool, occ)
            if slot in used_slots:
                occ = 2 if occ == 1 else 1
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

            # 1. Update tool-mode specific statistics
            for f in cand.faults:
                tm_key = f"{f.tool}:{f.mode.value}"
                self.tool_mode_trials[tm_key] = self.tool_mode_trials.get(tm_key, 0) + 1
                if violation_occurred:
                    self.tool_mode_violations[tm_key] = self.tool_mode_violations.get(tm_key, 0) + 1

            # 2. Update single mode statistics
            modes_in_cand = set(f.mode.value for f in cand.faults)
            for m in modes_in_cand:
                self.mode_trials[m] = self.mode_trials.get(m, 0) + 1
                if violation_occurred:
                    self.mode_violations[m] = self.mode_violations.get(m, 0) + 1

            # 3. Update pairwise combination statistics
            sigs = [f.signature() for f in cand.faults]
            for i in range(len(sigs)):
                for j in range(i + 1, len(sigs)):
                    pair = (min(sigs[i], sigs[j]), max(sigs[i], sigs[j]))
                    self.pair_trials[pair] = self.pair_trials.get(pair, 0) + 1
                    if violation_occurred:
                        self.pair_violations[pair] = self.pair_violations.get(pair, 0) + 1

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
        pair_stats = {
            f"{p[0]} + {p[1]}": {"trials": self.pair_trials[p], "violations": self.pair_violations.get(p, 0)}
            for p in self.pair_trials if self.pair_violations.get(p, 0) > 0
        }

        return SearchResult(
            total_experiments=budget,
            unique_violations_found=len(seen_violations),
            first_discovery_step=first_discovery,
            discovered_scenarios=discovered,
            reports=reports,
            discovery_history=history,
            mode_yield_stats=yield_stats,
            pair_yield_stats=pair_stats,
        )
