"""Fault injection engine and catalog for Project MIRROR."""

from __future__ import annotations

import copy
import random
from typing import Dict, List, Optional, Tuple

from mirror.models import FaultMode, FaultRule


class FaultInjector:
    """Manages and matches active fault injection rules against tool executions."""

    def __init__(self, rules: Optional[List[FaultRule]] = None):
        self.rules: List[FaultRule] = copy.deepcopy(rules) if rules else []
        self._execution_counts: Dict[str, int] = {}
        self._triggered_faults: List[Tuple[FaultRule, int]] = []

    def reset(self) -> None:
        """Reset internal counters for a clean scenario replay."""
        self._execution_counts.clear()
        self._triggered_faults.clear()

    def advance_and_match(self, tool: str) -> Tuple[int, Optional[FaultRule]]:
        """Increment occurrence counter for `tool` and return active FaultRule if matched."""
        current_count = self._execution_counts.get(tool, 0) + 1
        self._execution_counts[tool] = current_count

        matched: Optional[FaultRule] = None
        for rule in self.rules:
            if rule.tool == tool and rule.occurrence == current_count:
                matched = rule
                self._triggered_faults.append((rule, current_count))
                break

        return current_count, matched

    def get_triggered_rules(self) -> List[Tuple[FaultRule, int]]:
        """Return history of triggered rules and the call counts when triggered."""
        return list(self._triggered_faults)


class FaultCatalog:
    """Standard catalog of available fault types and tools for systematic exploration."""

    SUPPORTED_TOOLS: List[str] = [
        "payment.refund",
        "payment.get_order_status",
    ]

    SUPPORTED_MODES: List[FaultMode] = [
        FaultMode.TIMEOUT_AFTER_COMMIT,
        FaultMode.TIMEOUT_BEFORE_COMMIT,
        FaultMode.SERVER_ERROR,
        FaultMode.STALE_READ,
        FaultMode.PERMISSION_REVOKED,
        FaultMode.RATE_LIMITED,
        FaultMode.CORRUPTED_RESPONSE,
    ]

    @classmethod
    def get_valid_modes_for_tool(cls, tool: str) -> List[FaultMode]:
        """Return valid fault modes for a given tool name."""
        if tool == "payment.refund":
            return [
                FaultMode.TIMEOUT_AFTER_COMMIT,
                FaultMode.TIMEOUT_BEFORE_COMMIT,
                FaultMode.SERVER_ERROR,
                FaultMode.PERMISSION_REVOKED,
                FaultMode.RATE_LIMITED,
                FaultMode.CORRUPTED_RESPONSE,
            ]
        elif tool == "payment.get_order_status":
            return [
                FaultMode.STALE_READ,
                FaultMode.TIMEOUT_BEFORE_COMMIT,
                FaultMode.SERVER_ERROR,
                FaultMode.RATE_LIMITED,
                FaultMode.PERMISSION_REVOKED,
                FaultMode.CORRUPTED_RESPONSE,
            ]
        return cls.SUPPORTED_MODES

    @classmethod
    def generate_random_fault(
        cls,
        max_occurrence: int = 3,
        tool: Optional[str] = None,
        mode: Optional[FaultMode] = None,
        rng: Optional[random.Random] = None,
    ) -> FaultRule:
        """Sample a randomized fault rule from the catalog."""
        r = rng or random.Random()
        chosen_tool = tool or r.choice(cls.SUPPORTED_TOOLS)
        valid_modes = cls.get_valid_modes_for_tool(chosen_tool)
        chosen_mode = mode or r.choice(valid_modes)
        chosen_occ = r.randint(1, max_occurrence)

        return FaultRule(
            tool=chosen_tool,
            occurrence=chosen_occ,
            mode=chosen_mode,
            parameters={},
        )
