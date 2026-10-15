# Contributing to Project MIRROR 🪞

Thank you for your interest in contributing to **Project MIRROR** — a research-grade crash-test simulator and automated reliability-testing loop for AI agents.

This guide outlines our development philosophy, architectural abstractions, testing standards, and pull request procedures.

---

## Architectural Principles

MIRROR evaluates agent behavior against reality, not agent narratives. When extending MIRROR, adhere to four foundational principles:

1. **Independent Authoritative Ledger**: The agent's output is never ground truth. State mutations must be recorded only by the interception layer within an authoritative ledger.
2. **Combinatorial Fault Interaction**: Real-world failures stem from interactions (e.g., timeout + stale read), not isolated faults. Multi-fault exploration is central to the framework.
3. **Delta Debugging Minimization (ddmin)**: Discovered failures must be reduced to their 1-minimal reproducible core. Irrelevant faults must be stripped automatically.
4. **Permanent Regression Artifacts**: Every minimized bug becomes a serializable regression test that can detect if a future agent fix actually works.

---

## Environment Tiers

MIRROR organizes simulated execution environments into tiers:

- **Tier 1 — Transactional Systems** (`mirror.environment.RefundService`):
  Simulates financial operations, idempotency tokens, atomic status queries, and balance conservation.
- **Tier 2 — Stateful Cloud Infrastructure** (`mirror.cloud_environment.CloudInfrastructureService`):
  Simulates multi-resource lifecycles (VPC, VM, Storage Bucket), eventual consistency lags, partial deletion locks, and IAM credential revocations.
- **Tier 3 — Autonomous Tool-Using LLM Agents** (Planned / Roadmap):
  Pluggable adapter interface for real language model agents interacting with dynamic tools.

---

## How to Extend MIRROR

### 1. Adding a New AI Agent Under Test
Agents must inherit from `mirror.agent.BaseAgent` (Tier 1) or implement standard service interactions:

```python
from mirror.agent import BaseAgent
from mirror.environment import RefundService
from typing import Dict, Any

class CustomRefundAgent(BaseAgent):
    def __init__(self, max_attempts: int = 2):
        self.max_attempts = max_attempts

    def run(self, service: RefundService, order_id: str = "ORD-104", amount: int = 100) -> Dict[str, Any]:
        # Implement agent execution logic here
        ...
```

### 2. Adding a New Environmental Fault Mode
1. Register the fault mode in `mirror.models.FaultMode`.
2. Define interceptor logic in `mirror.environment.EnvironmentController.execute_tool` or `mirror.cloud_environment.CloudEnvironmentController.execute_tool`.
3. Add valid tool-mode associations in `mirror.faults.FaultCatalog.get_valid_modes_for_tool`.

### 3. Adding a Safety or Liveness Invariant
Formal invariants are verified in `mirror.evaluator.evaluate_scenario`:

```python
# Example: Adding a custom invariant check
if order["refunded_amount"] > order["amount"]:
    violations.append(
        InvariantViolation(
            invariant_name="balance_exceeded",
            description="Total refunded amount exceeds original order price.",
            severity="fatal",
            observed={"refunded": order["refunded_amount"], "original": order["amount"]},
            evidence="Authoritative ledger balance overflow.",
        )
    )
```

### 4. Adding a Benchmark Scenario
- **Development Scenarios**: Placed in `scenarios/*.yaml` (used for algorithm tuning and unit tests).
- **Held-Out Evaluation Scenarios**: Placed in `scenarios/eval/*.yaml` (reserved strictly for out-of-sample benchmark generalization testing).

---

## Local Development & Testing

### Installation
Clone the repository and install in editable mode with development dependencies:

```bash
git clone https://github.com/Mr-Nobody-Anonymous/mirror.git
cd mirror
pip install -e ".[dev]"
```

### Running Tests
All tests must pass with 100% success rate:

```bash
pytest -v
```

### Code Formatting & Linting
We enforce clean code standards via `ruff`:

```bash
ruff check .
```

### Running Benchmark Reproduction
Verify empirical reproduction across all 4 search strategies and 5 seeds:

```bash
python benchmarks/reproduce.py --budget 30
```

### Running the End-to-End CLI Demo
```bash
python -m mirror.cli demo
```

---

## Pull Request Guidelines

1. **Focused Diffs**: Keep pull requests focused on a single capability, bug fix, or scenario.
2. **Test Coverage**: Any new feature or invariant must include unit tests and meta-evaluator integrity checks in `tests/test_oracle_integrity.py`.
3. **No Flaky Tests**: Tests must execute deterministically across all operating systems.
4. **CI Passing**: All GitHub Actions CI checks (linting, test matrix across Python 3.10, 3.11, 3.12 on Linux & Windows) must pass before merging.
