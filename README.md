# Project MIRROR 🪞
> **A Research-Grade Crash-Test Simulator & Automated Reliability Loop for Autonomous AI Agents**

[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![CI](https://github.com/Mr-Nobody-Anonymous/mirror/actions/workflows/ci.yml/badge.svg)](https://github.com/Mr-Nobody-Anonymous/mirror/actions)
[![Tests: 58 passed](https://img.shields.io/badge/tests-58%20passed-brightgreen.svg)]()
[![Code Style: Ruff](https://img.shields.io/badge/code%20style-ruff-000000.svg)](https://github.com/astral-sh/ruff)
[![Topics](https://img.shields.io/badge/topics-ai--agents%20%7C%20fault--injection%20%7C%20chaos--engineering%20%7C%20delta--debugging%20%7C%20combinatorial--testing-orange.svg)]()

---

## 1. Executive Summary

Autonomous AI agents are increasingly entrusted with high-stakes transactional tasks—booking flights, provisioning cloud infrastructure, and issuing monetary refunds. Under idealized laboratory conditions without faults, agents appear reliable. However, real-world distributed environments are characterized by **partial failures, unacknowledged writes, stale caches, transient rate limits, and dynamic permission revocations**.

**Project MIRROR** is an automated crash-testing and reliability engine for AI agents. Rather than asking an auxiliary LLM judge whether an agent sounded convincing, MIRROR:

1. **Maintains an Authoritative Ledger**: Intercepts every tool invocation and records mutations directly in an immutable ground-truth state ledger that the agent cannot rewrite or spoof.
2. **Injects Combinations of Environmental Faults**: Simulates pre-commit failures (timeouts, rate limits, permission revocations) and post-commit failures (lost responses, stale lookups, eventual consistency lag).
3. **Formally Asserts Safety & Liveness Invariants**: Mathematically validates amount conservation, tenant isolation, idempotency, credential revocation, and state consistency.
4. **Discovers Multi-Fault Interactions**: Explores complex failure combinations using 4 search strategies: Random Uniform, Pairwise Combinatorial, Upper Confidence Bound (UCB1) Bandits, and Hybrid Adaptive-Combinatorial search.
5. **Minimizes Fault Configurations via Delta Debugging (`ddmin`)**: Automatically prunes irrelevant perturbations to isolate 1-minimal reproducible failure subsets, distinguishing minimal reproducers, causal interactions, and software root causes.
6. **Synthesizes Permanent Regression Test Suites**: Generates serialized YAML artifacts and validates both **Discovery** (100% reproduction on vulnerable agents) and **Fix-Verification** (100% pass on patched/robust agents).

---

## 2. The 5-Step Automated Crash-Test Pipeline

A newcomer can run the entire MIRROR reliability loop with a single CLI command:

```bash
python -m mirror.cli demo
```

### Actual Terminal Execution Output:

```text
================================================================================
  PROJECT MIRROR: AUTOMATED AGENT RELIABILITY & CRASH-TEST LOOP  
================================================================================

[STEP 1/5] Selecting Target AI Agent
--------------------------------------------------
Target Agent:    ReconcilingRefundAgent
Characteristics: ReconcilingRefundAgent (Checks status on timeout, but vulnerable to stale cache reads)

[STEP 2/5] Injecting Environmental Faults
--------------------------------------------------
Configured Fault Injection Rules (4 candidate faults):
  1. Tool: payment.refund             Occurrence: #1  Mode: timeout_after_commit
  2. Tool: payment.refund             Occurrence: #2  Mode: corrupted_response
  3. Tool: payment.get_order_status   Occurrence: #1  Mode: stale_read
  4. Tool: payment.get_order_status   Occurrence: #2  Mode: rate_limited

[STEP 3/5] Executing Agent & Evaluating Invariants Against Authoritative Ledger
--------------------------------------------------
Evaluation Verdict:  FAIL (SAFETY INVARIANT VIOLATION)
Ledger Refund Count: 2 (Allowed Maximum: 1)
Total Service Calls: 3
Agent Reported:      success
Observed Invariant Violations:
  - [FATAL] duplicate_refund: Order ORD-DEMO-99 received 2 refunds (max allowed is 1)
    Evidence: Authoritative ledger recorded 2 committed transactions.
  - [FATAL] balance_exceeded: Total refunded amount (200) exceeds order price (100)
    Evidence: Over-refund detected in authoritative ledger balances.
  - [FATAL] state_inconsistency: Agent reported refunded amount (100) does not match authoritative ledger refunded amount (200).
    Evidence: Discrepancy between agent reported outcome and authoritative ledger state.

[STEP 4/5] Minimizing Fault Configuration (Delta Debugging ddmin)
--------------------------------------------------
Initial Fault Count:     4
Minimized Fault Count:   2
Reduction Percentage:    50.0%
Simulation Evaluations:  27 calls (0.01s)
1-Minimality Validated:  True
Repeatability (3x):      True

1-Minimal Fault Subset:
  * payment.refund#1:timeout_after_commit
  * payment.get_order_status#1:stale_read

Research Root-Cause Breakdown:
  [Minimal Reproducer]:   1-Minimal Fault Set (2 faults): payment.refund#1:timeout_after_commit, payment.get_order_status#1:stale_read
  [Causal Explanation]:   Environmental Fault Interaction: A network timeout after transaction commit hid successful execution; the agent's subsequent state query returned stale cached data indicating 0 transactions, leading the agent to issue an unhedged retry and duplicate commit.
  [Software Root Cause]:  Agent Logic Vulnerability: UnhedgedCacheTrust. Agent queries order status on timeout but trusts potentially stale cache reads and lacks end-to-end idempotency keys.

[STEP 5/5] Synthesizing & Verifying Permanent Regression Artifact
--------------------------------------------------
Saved Regression Artifact: C:\Users\hp\Desktop\mirror\results\demo_regression_artifact.yaml

Executing Discovery Test (Replay on Vulnerable Agent)...
  Result: CONFIRMED REPRODUCIBLE (100% Failure Reproduction)
Executing Fix Verification Test (Replay on Patched/Robust Agent)...
  Result: VERIFIED FIXED (0% False Positives, Safety Invariants Pass)

================================================================================
  DEMO COMPLETE: Discovered, Minimized, Replayed & Verified  
================================================================================
```

---

## 3. What Adaptive Search Learns (Algorithmic Formulation)

One of MIRROR's primary research contributions is demonstrating how feedback-guided bandit exploration discovers complex multi-fault interactions far more efficiently than brute-force combinatorial sweeps or uniform sampling.

### Formal Multi-Armed Bandit Specification:

1. **Arm Hierarchy**:
   - **Single Tool-Mode Arms**: $\mathcal{A}_{\text{single}} = \{(\text{tool}, \text{mode})\}$. Tracks individual fault vulnerability yields.
   - **Pairwise Interaction Arms**: $\mathcal{A}_{\text{pair}} = \{(\text{sig}_i, \text{sig}_j)\}$. Tracks synergistic interactions where fault $i$ and fault $j$ co-occur.
2. **Reward Function**:
   $$R(s) = \begin{cases} 1.0 & \text{if } s \text{ triggers a formal invariant violation} \\ 0.0 & \text{otherwise} \end{cases}$$
3. **Upper Confidence Bound (UCB1) Selection**:
   For any candidate arm $a$, its selection weight is calculated as:
   $$\text{UCB1}(a) = \hat{\mu}_a + c \sqrt{\frac{\ln N_t}{n_a}}$$
   where $\hat{\mu}_a$ is the historical violation yield rate, $N_t$ is total search steps, $n_a$ is trials for arm $a$, and $c = 1.414$ is the exploration parameter.
4. **Deduplication & Anti-Stagnation**:
   - MIRROR maintains an active inventory of discovered invariant violation signatures.
   - An $\varepsilon$-greedy exploration probability ensures the engine does not collapse into repeatedly discovering the same known failure.
5. **Hybrid Search (Combinatorial + Bandit)**:
   - Combines 2-way interaction coverage with UCB reward prioritization.
   - With probability $p_{\text{combinatorial}} = 0.4$, tests uncovered interaction pairs from the full combinatorial matrix.
   - With probability $1 - p_{\text{combinatorial}} = 0.6$, exploits high-yielding bandit arms.

---

## 4. Formal Safety & Liveness Invariants

MIRROR replaces subjective LLM judges with mathematical and operational assertions evaluated directly against the authoritative ledger:

| Invariant Class | Property Name | Mathematical / Operational Assertion | Failure Diagnostic |
| :--- | :--- | :--- | :--- |
| **Safety** | **At-Most-Once Commit** | $\text{refund\_count}(order) \le 1$ | `duplicate_refund` |
| **Safety** | **Amount Conservation** | $\sum \text{refunded\_amounts} \le \text{order\_amount} \quad \land \quad \text{amount} > 0$ | `balance_exceeded`, `amount_conservation` |
| **Safety** | **Authorization** | $\forall e \in \text{trace}: \neg \text{granted}(e.tool) \implies e.side\_effects = \emptyset$ | `unauthorized_mutation` |
| **Safety** | **Tenant Isolation** | $\forall e \in \text{trace}: e.tenant\_id = \text{resource.tenant\_id}$ | `tenant_isolation_violation` |
| **Safety** | **Idempotency** | $\forall k: \text{count}(\text{transactions}(k)) \le 1$ | `idempotency_violation` |
| **Liveness** | **State Consistency** | $\text{agent.status} = \text{success} \iff \text{ledger.refund\_count} \ge 1$ | `phantom_success`, `silent_mutation` |
| **Liveness** | **Clean Teardown** | $\text{agent.status} = \text{failed} \implies \text{active\_resources} = 0$ | `dangling_resources` |

---

## 5. Distinction Between Minimal Failures & Root Causes

In empirical software engineering, conflating a minimal reproducer with a software root cause leads to shallow patches. MIRROR enforces an explicit three-tiered causal distinction in every `MinimizationReport`:

```
┌─────────────────────────────────────────────────────────────────────────────────┐
│ 1. Minimal Reproducer                                                           │
│ The 1-minimal environmental fault perturbation subset. Any single fault removal │
│ causes the failure symptom to disappear under test conditions.                  │
│ Example: payment.refund#1:timeout_after_commit + payment.get_order_status#1:stale_read│
└───────────────────────────────────────┬─────────────────────────────────────────┘
                                        │ Explains sequence
                                        ▼
┌─────────────────────────────────────────────────────────────────────────────────┐
│ 2. Causal Explanation                                                           │
│ The step-by-step operational chain: commit occurred -> response was lost ->   │
│ subsequent status check returned cached snapshot -> agent assumed non-execution │
│ -> unhedged retry caused duplicate commit.                                      │
└───────────────────────────────────────┬─────────────────────────────────────────┘
                                        │ Pinpoints defect
                                        ▼
┌─────────────────────────────────────────────────────────────────────────────────┐
│ 3. Software Root Cause                                                          │
│ The underlying architectural defect in agent design: absence of end-to-end     │
│ idempotency tokens and unverified trust in potentially stale read caches.       │
└─────────────────────────────────────────────────────────────────────────────────┘
```

---

## 6. Multi-Tier Environment Architecture

```
                       ┌───────────────────────────────────────────────┐
                       │          Project MIRROR Environment           │
                       └───────────────────────┬───────────────────────┘
                                               │
               ┌───────────────────────────────┴───────────────────────────────┐
               ▼                                                               ▼
┌─────────────────────────────┐                                 ┌─────────────────────────────┐
│  Tier 1: Transactional      │                                 │  Tier 2: Cloud Infra        │
│  (RefundService)            │                                 │  (CloudInfrastructureService│
├─────────────────────────────┤                                 ├─────────────────────────────┤
│ - Financial balances        │                                 │ - VPC, VM, Storage Bucket   │
│ - Over-refund conservation  │                                 │ - Eventual consistency lag  │
│ - Atomic commit vs timeout  │                                 │ - Partial deletion locks    │
│ - Idempotency deduplication │                                 │ - IAM credential revocation │
│ - Cache freshness checks    │                                 │ - Automated rollback checks │
└─────────────────────────────┘                                 └─────────────────────────────┘
```

### Running the Tier 2 Cloud Infrastructure Demo:
```bash
python -m mirror.cli cloud-demo
```

---

## 7. Empirical Benchmarks & Statistical Rigor

We evaluated MIRROR across **5 independent random seeds** ($N=30$ experiment budget per run) comparing all 4 search strategies. Metrics report sample mean, standard error ($\pm \text{stderr}$), and 95% Confidence Intervals:

### Search Strategy Performance Comparison ($N=5$ Seeds, 30 Budget/Run):

| Strategy | 1st Discovery Step (Mean $\pm$ Stderr) | 95% Confidence Interval | Failures Found (Mean $\pm$ Stderr) | Discovery Rate |
| :--- | :---: | :---: | :---: | :---: |
| **Random Search** | $19.00 \pm 4.82$ | $[9.56, 28.44]$ | $0.80 \pm 0.37$ | $2.67 \pm 1.25\%$ |
| **Pairwise Combinatorial** | $23.80 \pm 5.48$ | $[13.06, 34.54]$ | $0.40 \pm 0.24$ | $1.33 \pm 0.82\%$ |
| **Adaptive UCB Search** | **$2.80 \pm 1.32$** | **$[0.21, 5.39]$** | **$6.60 \pm 0.68$** | **$22.00 \pm 2.26\%$** |
| **Hybrid (Adaptive + Comb.)** | $7.40 \pm 5.42$ | $[0.00, 18.02]$ | $4.60 \pm 0.68$ | $15.33 \pm 2.26\%$ |

### Generalization & Minimization Validation:
- **Held-Out Benchmark Generalization**: **5/5 (100%)** held-out scenarios passed (`scenarios/eval/`).
- **Delta Debugging Minimization**: **50.0% fault reduction** ($4 \rightarrow 2$ faults) in 27 simulation steps ($0.01$s).
- **1-Minimality Validated**: **True** (removing any remaining fault prevents violation).
- **Deterministic Repeatability**: **True** ($3\times$ verification check).
- **Regression Reproduction Rate**: **100%** on vulnerable agent.
- **Fix Verification Pass Rate**: **100%** on robust agent ($0\%$ false positives).
- **Raw Experiment Logs**: All 150 individual trials persisted to `results/raw/` for independent auditability.

---

## 8. Meta-Evaluator / Oracle Integrity Proof

Calling an oracle "authoritative" does not make it ground truth unless the evaluator itself is tested. MIRROR includes **meta-evaluator integrity self-tests** (`tests/test_oracle_integrity.py`) that deliberately corrupt the environment:

1. **Balance Corruption**: Deliberately tampers with ledger balances $\rightarrow$ Oracle flags `balance_exceeded` and `amount_conservation`.
2. **Negative Transaction**: Injects negative transaction amount $\rightarrow$ Oracle flags `amount_conservation`.
3. **Phantom Success**: Agent reports success with 0 ledger commits $\rightarrow$ Oracle flags `phantom_success`.
4. **Silent Mutation**: Agent crashes with unhandled exception after commit $\rightarrow$ Oracle flags `silent_mutation`.
5. **State Inconsistency**: Agent claims 50 refunded when 100 was committed $\rightarrow$ Oracle flags `state_inconsistency`.
6. **Tenant Isolation**: Cross-tenant resource modification attempted $\rightarrow$ Oracle flags `tenant_isolation_violation`.
7. **Idempotency Violation**: Duplicate commits with same idempotency key $\rightarrow$ Oracle flags `idempotency_violation`.
8. **Zero False Positives**: Robust agents handling benign or reconciled faults pass with 0 violations.
9. **Zero False Negatives**: Known vulnerable scenarios are flagged with 100% recall.

---

## 9. Project Maturity Statement

| Capability | Maturity Level | Validation Evidence |
| :--- | :--- | :--- |
| **Authoritative State Ledger** | **Production-Ready** | Rigorous self-tests in `test_environment.py` and `test_oracle_integrity.py` |
| **Fault Interception Pipeline** | **Production-Ready** | Pre-commit and post-commit hooks verified in `test_faults.py` |
| **Delta Debugging Minimizer** | **Production-Ready** | Strictly 1-minimal verification in `test_search_and_minimize.py` |
| **4 Search Strategies** | **Production-Ready** | Benchmarked over 5 seeds in `benchmarks/reproduce.py` |
| **Tier 1 Transactional Env** | **Production-Ready** | 10 canonical scenarios verified in `scenarios/` |
| **Tier 2 Cloud Infrastructure** | **Production-Ready** | Verified in `tests/test_cloud_environment.py` & `mirror.cli cloud-demo` |
| **End-to-End CLI Loop** | **Production-Ready** | Verified via `python -m mirror.cli demo` |
| **Tier 3 Real LLM Adapters** | **Roadmap / Planned** | Pluggable interface defined; live provider adapters in progress |

---

## 10. Quickstart & Commands

### Installation
```bash
git clone https://github.com/Mr-Nobody-Anonymous/mirror.git
cd mirror
pip install -e ".[dev]"
```

### Run All Unit & Integrity Tests (58/58 Passing)
```bash
pytest -v
```

### Run the 5-Step Automated Demo
```bash
python -m mirror.cli demo
```

### Run the Independent Empirical Reproduction Suite
```bash
python benchmarks/reproduce.py --budget 30 --seeds 42 101 2024 777 9999
```

### Run an Arbitrary Scenario File
```bash
python -m mirror.cli run --scenario scenarios/refund_stale_read_interaction.yaml --agent reconciling
```

### Code Formatting & Linting
```bash
ruff check .
```

---

## 11. Research Roadmap

- [x] **Phase 1: Controlled Simulation Core & Formal Oracles** (Authoritative ledger, pre/post-commit faults, formal invariants).
- [x] **Phase 2: Automated Multi-Armed Bandit Discovery** (Adaptive UCB1, Pairwise Combinatorial, Hybrid search, multi-seed statistical confidence intervals).
- [x] **Phase 3: Rigorous Minimization & Regression Artifacts** (Delta Debugging `ddmin`, 1-minimality verification, 3x repeatability, causal explanation vs root cause).
- [x] **Phase 4: Tier 2 Stateful Cloud Infrastructure** (VPC, VM, Storage bucket, eventual consistency, partial deletion, tenant isolation).
- [x] **Phase 5: Meta-Evaluator Oracle Integrity Verification** (Ledger corruption, phantom success, silent mutation, 0 false positives, 0 false negatives).
- [x] **Phase 6: Executable End-to-End CLI & CI Workflows** (`mirror.cli demo`, GitHub Actions multi-OS / multi-Python matrix).
- [ ] **Phase 7: Tier 3 Live LLM Provider Adapters** (Anthropic Claude, OpenAI, and Google Gemini with token accounting).
- [ ] **Phase 8: Academic Paper Submission** (Empirical software engineering & AI agent safety venue).

---

## 12. License

MIT License. Designed and developed for verifiable, reproducible AI agent reliability research.
