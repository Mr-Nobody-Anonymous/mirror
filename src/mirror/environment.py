"""Controlled simulation environment and authoritative ledger for Project MIRROR."""

from __future__ import annotations

import copy
import time
import uuid
from typing import Any, Dict, List, Optional
from mirror.models import (
    EventRecord,
    FaultMode,
    FaultRule,
    InitialState,
    ScenarioConfig,
)
from mirror.faults import FaultInjector


class AuthoritativeLedger:
    """The authoritative ground truth for all transactions and state.

    CRITICAL ARCHITECTURAL GUARANTEE:
    The agent cannot rewrite or manipulate this ledger. Only the simulator
    interception layer commits state transitions here.
    """

    def __init__(self, initial_state: Optional[InitialState] = None):
        self.state = initial_state or InitialState()
        init_refunded = self.state.order_amount if self.state.refund_count > 0 else 0
        init_status = "refunded" if self.state.refund_count > 0 else "paid"
        self.orders: Dict[str, Dict[str, Any]] = {
            self.state.order_id: {
                "order_id": self.state.order_id,
                "amount": self.state.order_amount,
                "refunded_amount": init_refunded,
                "refund_count": self.state.refund_count,
                "status": init_status,
                "refund_transactions": [],
            }
        }
        self.permissions: Dict[str, bool] = copy.deepcopy(self.state.permissions)
        self.idempotency_records: Dict[str, Dict[str, Any]] = {}
        self.history: List[Dict[str, Any]] = []

    def is_permission_granted(self, tool: str) -> bool:
        return self.permissions.get(tool, True)

    def set_permission(self, tool: str, granted: bool) -> None:
        self.permissions[tool] = granted

    def commit_refund(
        self,
        order_id: str,
        amount: int,
        idempotency_key: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Atomically commit a refund transaction into the authoritative ledger."""
        if order_id not in self.orders:
            raise ValueError(f"Unknown order {order_id}")

        order = self.orders[order_id]

        # Idempotency check: if key already exists, return previous transaction
        if idempotency_key and idempotency_key in self.idempotency_records:
            record = self.idempotency_records[idempotency_key]
            return {
                "status": "success",
                "order_id": order_id,
                "amount": record["amount"],
                "refund_id": record["refund_id"],
                "deduplicated": True,
            }

        refund_id = f"REF-{uuid.uuid4().hex[:8].upper()}"
        order["refund_count"] += 1
        order["refunded_amount"] += amount
        if order["refunded_amount"] >= order["amount"]:
            order["status"] = "refunded"
        else:
            order["status"] = "partially_refunded"

        txn_record = {
            "refund_id": refund_id,
            "order_id": order_id,
            "amount": amount,
            "timestamp": time.time(),
            "idempotency_key": idempotency_key,
        }
        order["refund_transactions"].append(txn_record)

        if idempotency_key:
            self.idempotency_records[idempotency_key] = txn_record

        self.history.append({
            "action": "commit_refund",
            "order_id": order_id,
            "amount": amount,
            "refund_count": order["refund_count"],
        })

        return {
            "status": "success",
            "order_id": order_id,
            "amount": amount,
            "refund_id": refund_id,
            "deduplicated": False,
        }

    def get_order_state(self, order_id: str) -> Dict[str, Any]:
        """Return authoritative state for an order."""
        if order_id not in self.orders:
            raise ValueError(f"Unknown order {order_id}")
        return copy.deepcopy(self.orders[order_id])

    @property
    def total_refunds(self) -> int:
        order = self.orders.get(self.state.order_id)
        return order["refund_count"] if order else 0


class EnvironmentController:
    """Interception layer and fault orchestrator.

    Wraps the authoritative ledger and proxies tool invocations, injecting
    pre-commit and post-commit faults according to the active ScenarioConfig.
    """

    def __init__(
        self,
        config: Optional[ScenarioConfig] = None,
        ledger: Optional[AuthoritativeLedger] = None,
    ):
        self.config = config or ScenarioConfig(id="default")
        self.ledger = ledger or AuthoritativeLedger(self.config.initial_state)
        self.injector = FaultInjector(self.config.faults)
        self.trace: List[EventRecord] = []
        self._stale_order_snapshots: Dict[str, Dict[str, Any]] = {}
        # Pre-capture initial snapshot for stale reads
        self._stale_order_snapshots[self.config.initial_state.order_id] = self.ledger.get_order_state(
            self.config.initial_state.order_id
        )

    def reset(self) -> None:
        """Reset environment to fresh initial state."""
        self.ledger = AuthoritativeLedger(self.config.initial_state)
        self.injector.reset()
        self.trace.clear()
        self._stale_order_snapshots.clear()
        self._stale_order_snapshots[self.config.initial_state.order_id] = self.ledger.get_order_state(
            self.config.initial_state.order_id
        )

    def execute_tool(
        self,
        tool: str,
        inputs: Dict[str, Any],
    ) -> Any:
        """Execute a tool through the interception and fault injection pipeline."""
        call_id = f"CALL-{uuid.uuid4().hex[:6]}"
        step = len(self.trace) + 1
        occurrence, matched_fault = self.injector.advance_and_match(tool)

        fault_mode = matched_fault.mode if matched_fault else FaultMode.NONE

        # --- Phase 1: Pre-commit Invariant & Fault Checks ---
        if fault_mode == FaultMode.PERMISSION_REVOKED:
            self.ledger.set_permission(tool, False)
            err = PermissionError(f"Permission denied: {tool} is revoked")
            self._log_event(step, tool, call_id, inputs, error=str(err), fault_applied=fault_mode)
            raise err

        if not self.ledger.is_permission_granted(tool):
            err = PermissionError(f"Permission denied: {tool} not granted")
            self._log_event(step, tool, call_id, inputs, error=str(err), fault_applied=fault_mode)
            raise err

        if fault_mode == FaultMode.RATE_LIMITED:
            err = RuntimeError("429 Too Many Requests: Rate limit exceeded")
            self._log_event(step, tool, call_id, inputs, error=str(err), fault_applied=fault_mode)
            raise err

        if fault_mode == FaultMode.TIMEOUT_BEFORE_COMMIT:
            err = TimeoutError("Network timeout before request reached server")
            self._log_event(step, tool, call_id, inputs, error=str(err), fault_applied=fault_mode)
            raise err

        if fault_mode == FaultMode.SERVER_ERROR:
            err = RuntimeError("500 Internal Server Error: Remote service failed")
            self._log_event(step, tool, call_id, inputs, error=str(err), fault_applied=fault_mode)
            raise err

        # --- Phase 2: Tool Execution on Authoritative Ledger ---
        output: Any = None
        side_effects: Dict[str, Any] = {}

        if tool == "payment.refund":
            order_id = inputs.get("order_id", self.config.initial_state.order_id)
            amount = inputs.get("amount", self.config.initial_state.order_amount)
            idempotency_key = inputs.get("idempotency_key")
            output = self.ledger.commit_refund(order_id, amount, idempotency_key=idempotency_key)
            side_effects["refund_committed"] = True
            side_effects["current_refund_count"] = self.ledger.total_refunds

        elif tool == "payment.get_order_status":
            order_id = inputs.get("order_id", self.config.initial_state.order_id)
            if fault_mode == FaultMode.STALE_READ:
                # Return stale snapshot prior to refund commit
                if order_id not in self._stale_order_snapshots:
                    if order_id in self.ledger.orders:
                        self._stale_order_snapshots[order_id] = copy.deepcopy(self.ledger.orders[order_id])
                    else:
                        raise ValueError(f"Unknown order {order_id}")
                output = copy.deepcopy(self._stale_order_snapshots[order_id])
                output["cached"] = True
            else:
                output = self.ledger.get_order_state(order_id)
        else:
            raise ValueError(f"Unknown tool: {tool}")

        # --- Phase 3: Post-commit Fault Injection ---
        if fault_mode == FaultMode.TIMEOUT_AFTER_COMMIT:
            err = TimeoutError("Response lost after refund committed")
            self._log_event(
                step, tool, call_id, inputs,
                output=None, error=str(err),
                fault_applied=fault_mode, side_effects=side_effects
            )
            raise err

        if fault_mode == FaultMode.CORRUPTED_RESPONSE:
            output = {"corrupted_payload": True, "error_code": "MALFORMED_JSON"}

        self._log_event(
            step, tool, call_id, inputs,
            output=output, error=None,
            fault_applied=fault_mode if fault_mode != FaultMode.NONE else None,
            side_effects=side_effects
        )
        return output

    def _log_event(
        self,
        step: int,
        tool: str,
        call_id: str,
        inputs: Dict[str, Any],
        output: Optional[Any] = None,
        error: Optional[str] = None,
        fault_applied: Optional[FaultMode] = None,
        side_effects: Optional[Dict[str, Any]] = None,
    ) -> None:
        self.trace.append(
            EventRecord(
                step=step,
                tool=tool,
                call_id=call_id,
                inputs=copy.deepcopy(inputs),
                output=copy.deepcopy(output),
                error=error,
                fault_applied=fault_applied,
                side_effects=side_effects or {},
            )
        )


class RefundService:
    """Refund service interface exposed to the agent.

    Can operate as a lightweight standalone mock (compatible with initial
    demonstrations in section 5) OR wrap a full EnvironmentController.
    """

    def __init__(
        self,
        controller: Optional[EnvironmentController] = None,
        order_id: str = "ORD-104",
    ):
        self.controller = controller
        self.order_id = order_id
        # Standalone counters (mirrors section 5)
        self._successful_refunds = 0
        self._calls = 0

    @property
    def successful_refunds(self) -> int:
        if self.controller is not None:
            return self.controller.ledger.total_refunds
        return self._successful_refunds

    @successful_refunds.setter
    def successful_refunds(self, value: int) -> None:
        self._successful_refunds = value

    @property
    def calls(self) -> int:
        if self.controller is not None:
            return len(self.controller.trace)
        return self._calls

    @calls.setter
    def calls(self, value: int) -> None:
        self._calls = value

    def refund(
        self,
        amount: int,
        order_id: Optional[str] = None,
        idempotency_key: Optional[str] = None,
        timeout_after_commit: bool = False,
    ) -> Dict[str, Any]:
        """Execute refund."""
        target_order = order_id or self.order_id

        if self.controller is not None:
            return self.controller.execute_tool(
                "payment.refund",
                {
                    "order_id": target_order,
                    "amount": amount,
                    "idempotency_key": idempotency_key,
                },
            )

        # Standalone fallback mode (Section 5 minimal example)
        self._calls += 1
        self._successful_refunds += 1
        if timeout_after_commit:
            raise TimeoutError("Response lost after refund committed")
        return {"status": "success", "amount": amount, "order_id": target_order}

    def get_order_status(self, order_id: Optional[str] = None) -> Dict[str, Any]:
        """Retrieve order status."""
        target_order = order_id or self.order_id

        if self.controller is not None:
            return self.controller.execute_tool(
                "payment.get_order_status",
                {"order_id": target_order},
            )

        self._calls += 1
        return {
            "order_id": target_order,
            "status": "refunded" if self._successful_refunds > 0 else "paid",
            "refund_count": self._successful_refunds,
        }
