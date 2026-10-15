"""Agent implementations and interfaces for Project MIRROR."""

from __future__ import annotations

import abc
import uuid
from typing import Any, Dict, Optional
from mirror.environment import RefundService


class BaseAgent(abc.ABC):
    """Abstract base class for agents under test."""

    @abc.abstractmethod
    def run(
        self,
        service: RefundService,
        order_id: str = "ORD-104",
        amount: int = 100,
    ) -> Dict[str, Any]:
        """Execute the agent task and return the agent's reported outcome."""
        pass


class NaiveRefundAgent(BaseAgent):
    """A deliberately naive agent that retries blindly on failure.

    Failure mode:
    Retries refund requests on network timeouts without querying order status
    or supplying an idempotency key. When a timeout occurs after the payment
    is committed, this agent causes a duplicate refund.
    """

    def __init__(self, max_attempts: int = 2):
        self.max_attempts = max_attempts

    def run(
        self,
        service: RefundService,
        order_id: str = "ORD-104",
        amount: int = 100,
    ) -> Dict[str, Any]:
        last_error = None
        for attempt in range(self.max_attempts):
            try:
                # If running in Section 5 standalone mode (no controller), use timeout_after_commit
                if service.controller is None:
                    res = service.refund(amount, timeout_after_commit=(attempt == 0))
                else:
                    res = service.refund(amount, order_id=order_id)
                return {
                    "status": "success",
                    "order_id": order_id,
                    "amount": amount,
                    "attempts": attempt + 1,
                    "reported_result": res,
                }
            except TimeoutError as e:
                last_error = e
                if attempt == self.max_attempts - 1:
                    raise
            except Exception as e:
                last_error = e
                if attempt == self.max_attempts - 1:
                    raise

        raise RuntimeError(f"Refund failed after {self.max_attempts} attempts: {last_error}")


class RobustRefundAgent(BaseAgent):
    """A defensive, production-grade agent that uses idempotency and verification.

    Safeguards:
    1. Supplies a unique idempotency key for the order.
    2. Verifies order state before issuing requests.
    3. On timeout, checks the authoritative order status before attempting retries.
    4. Handles permission revocations cleanly.
    """

    def __init__(self, max_attempts: int = 2):
        self.max_attempts = max_attempts

    def run(
        self,
        service: RefundService,
        order_id: str = "ORD-104",
        amount: int = 100,
    ) -> Dict[str, Any]:
        idempotency_key = f"IDEMP-{order_id}-{amount}"

        # Step 1: Pre-check status
        try:
            status = service.get_order_status(order_id)
            if status.get("status") in ("refunded", "partially_refunded") and status.get("refund_count", 0) > 0:
                return {
                    "status": "success",
                    "order_id": order_id,
                    "amount": amount,
                    "note": "Already refunded",
                    "attempts": 0,
                }
        except Exception:
            # If status check fails, proceed cautiously
            pass

        # Step 2: Attempt refund with idempotency key
        for attempt in range(self.max_attempts):
            try:
                res = service.refund(
                    amount=amount,
                    order_id=order_id,
                    idempotency_key=idempotency_key,
                )
                return {
                    "status": "success",
                    "order_id": order_id,
                    "amount": amount,
                    "attempts": attempt + 1,
                    "reported_result": res,
                }
            except TimeoutError:
                # Step 3: Handle timeout by reconciling state
                try:
                    reconciled = service.get_order_status(order_id)
                    if reconciled.get("refund_count", 0) > 0:
                        return {
                            "status": "success",
                            "order_id": order_id,
                            "amount": amount,
                            "attempts": attempt + 1,
                            "note": "Reconciled after timeout",
                        }
                except Exception:
                    pass

                if attempt == self.max_attempts - 1:
                    return {
                        "status": "failed",
                        "order_id": order_id,
                        "error": "Timeout and unable to reconcile",
                    }

            except PermissionError as pe:
                return {
                    "status": "failed",
                    "order_id": order_id,
                    "error": f"Permission denied: {pe}",
                }
            except Exception as e:
                if attempt == self.max_attempts - 1:
                    return {
                        "status": "failed",
                        "order_id": order_id,
                        "error": str(e),
                    }

        return {
            "status": "failed",
            "order_id": order_id,
            "error": "Max attempts exceeded",
        }


def run_refund_agent(
    service: RefundService,
    amount: int = 100,
    max_attempts: int = 2,
) -> Dict[str, Any]:
    """Compatibility wrapper for Section 5 reference implementation."""
    agent = NaiveRefundAgent(max_attempts=max_attempts)
    return agent.run(service, amount=amount)
