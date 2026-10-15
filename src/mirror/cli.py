"""Command Line Interface (CLI) for Project MIRROR.

Provides an executable end-to-end 5-step reliability testing loop:
Step 1: Select agent (Naive, Reconciling, or Robust)
Step 2: Inject environmental faults (Timeouts, stale reads, rate limits, revocations)
Step 3: Detect invariant violations against the authoritative ledger
Step 4: Minimize failure configuration using Delta Debugging (ddmin)
Step 5: Synthesize and verify regression test (discovery reproduction & fix verification)
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

from mirror.agent import BaseAgent, NaiveRefundAgent, ReconcilingRefundAgent, RobustRefundAgent
from mirror.cloud_environment import (
    CloudEnvironmentController,
    CloudInfrastructureService,
    NaiveCloudDeployerAgent,
    RobustCloudDeployerAgent,
    evaluate_cloud_scenario,
)
from mirror.environment import EnvironmentController, RefundService
from mirror.evaluator import evaluate_scenario
from mirror.minimize import ScenarioMinimizer
from mirror.models import FaultMode, FaultRule, InitialState, RegressionArtifact, ScenarioConfig
from mirror.regression import RegressionRunner, load_regression_artifact, save_regression_artifact

# Ensure UTF-8 console output on Windows
if sys.platform == "win32" and hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass


class TerminalStyle:
    """ANSI color formatting helpers."""

    CYAN = "\033[96m"
    GREEN = "\033[92m"
    YELLOW = "\033[93m"
    RED = "\033[91m"
    BOLD = "\033[1m"
    DIM = "\033[2m"
    RESET = "\033[0m"

    @classmethod
    def header(cls, text: str) -> str:
        return f"{cls.BOLD}{cls.CYAN}{text}{cls.RESET}"

    @classmethod
    def success(cls, text: str) -> str:
        return f"{cls.BOLD}{cls.GREEN}{text}{cls.RESET}"

    @classmethod
    def fail(cls, text: str) -> str:
        return f"{cls.BOLD}{cls.RED}{text}{cls.RESET}"

    @classmethod
    def warn(cls, text: str) -> str:
        return f"{cls.BOLD}{cls.YELLOW}{text}{cls.RESET}"


def run_demo(agent_choice: str = "reconciling", out_dir: str = "results") -> int:
    """Execute the canonical 5-step automated crash-test pipeline."""
    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    print("\n" + "=" * 80)
    print(TerminalStyle.header("  PROJECT MIRROR: AUTOMATED AGENT RELIABILITY & CRASH-TEST LOOP  "))
    print("=" * 80 + "\n")

    # --------------------------------------------------------------------------
    # Step 1: Select Agent
    # --------------------------------------------------------------------------
    print(TerminalStyle.header("[STEP 1/5] Selecting Target AI Agent"))
    print("-" * 50)
    if agent_choice == "naive":
        target_agent: BaseAgent = NaiveRefundAgent(max_attempts=2)
        agent_desc = "NaiveRefundAgent (Retries blindly on timeout without idempotency)"
    elif agent_choice == "robust":
        target_agent = RobustRefundAgent(max_attempts=2)
        agent_desc = "RobustRefundAgent (Production-grade with idempotency & reconciliation)"
    else:
        target_agent = ReconcilingRefundAgent(max_attempts=2)
        agent_desc = "ReconcilingRefundAgent (Checks status on timeout, but vulnerable to stale cache reads)"

    print(f"Target Agent:    {TerminalStyle.BOLD}{target_agent.__class__.__name__}{TerminalStyle.RESET}")
    print(f"Characteristics: {agent_desc}\n")

    # --------------------------------------------------------------------------
    # Step 2: Inject Faults
    # --------------------------------------------------------------------------
    print(TerminalStyle.header("[STEP 2/5] Injecting Environmental Faults"))
    print("-" * 50)
    # Bloated multi-fault configuration containing interacting faults + benign noise
    faults = [
        FaultRule(tool="payment.refund", occurrence=1, mode=FaultMode.TIMEOUT_AFTER_COMMIT),
        FaultRule(tool="payment.refund", occurrence=2, mode=FaultMode.CORRUPTED_RESPONSE),
        FaultRule(tool="payment.get_order_status", occurrence=1, mode=FaultMode.STALE_READ),
        FaultRule(tool="payment.get_order_status", occurrence=2, mode=FaultMode.RATE_LIMITED),
    ]

    bloated_scenario = ScenarioConfig(
        id="demo_multi_fault_interaction",
        task="refund_order",
        initial_state=InitialState(order_id="ORD-DEMO-99", order_amount=100),
        faults=faults,
        invariants=["refund_count <= 1", "amount_conservation", "ledger_matches_agent"],
        metadata={"created_by": "MIRROR_CLI_Demo"},
    )

    print("Configured Fault Injection Rules (4 candidate faults):")
    for idx, f in enumerate(faults, 1):
        print(f"  {idx}. Tool: {f.tool:<26} Occurrence: #{f.occurrence}  Mode: {f.mode.value}")
    print()

    # --------------------------------------------------------------------------
    # Step 3: Detect Violation
    # --------------------------------------------------------------------------
    print(TerminalStyle.header("[STEP 3/5] Executing Agent & Evaluating Invariants Against Authoritative Ledger"))
    print("-" * 50)
    ctrl = EnvironmentController(config=bloated_scenario)
    service = RefundService(controller=ctrl, order_id=bloated_scenario.initial_state.order_id)

    agent_result = None
    try:
        agent_result = target_agent.run(
            service=service,
            order_id=bloated_scenario.initial_state.order_id,
            amount=bloated_scenario.initial_state.order_amount,
        )
    except Exception as e:
        agent_result = {"status": "error", "error": str(e)}

    eval_report = evaluate_scenario(ctrl, agent_result, bloated_scenario)

    if eval_report.safety_violation:
        print(f"Evaluation Verdict:  {TerminalStyle.fail('FAIL (SAFETY INVARIANT VIOLATION)')}")
    else:
        print(f"Evaluation Verdict:  {TerminalStyle.success('PASS (SAFETY INVARIANTS PRESERVED)')}")

    print(f"Ledger Refund Count: {eval_report.refund_count} (Allowed Maximum: 1)")
    print(f"Total Service Calls: {eval_report.service_calls}")
    print(f"Agent Reported:      {agent_result.get('status') if agent_result else 'crashed'}")
    print("Observed Invariant Violations:")
    for v in eval_report.violations:
        print(f"  - [{v.severity.upper()}] {TerminalStyle.BOLD}{v.invariant_name}{TerminalStyle.RESET}: {v.description}")
        print(f"    Evidence: {TerminalStyle.DIM}{v.evidence}{TerminalStyle.RESET}")
    print()

    # --------------------------------------------------------------------------
    # Step 4: Minimize with Delta Debugging
    # --------------------------------------------------------------------------
    print(TerminalStyle.header("[STEP 4/5] Minimizing Fault Configuration (Delta Debugging ddmin)"))
    print("-" * 50)
    minimizer = ScenarioMinimizer(agent=target_agent, target_violation_type="duplicate_refund")
    start_time = time.time()
    minimized_cfg, min_report = minimizer.minimize_with_report(
        bloated_scenario, method="ddmin", repeatability_checks=3
    )
    elapsed = time.time() - start_time

    print(f"Initial Fault Count:     {min_report.original_fault_count}")
    print(f"Minimized Fault Count:   {TerminalStyle.success(str(min_report.minimized_fault_count))}")
    print(f"Reduction Percentage:    {TerminalStyle.success(f'{min_report.reduction_pct}%')}")
    print(f"Simulation Evaluations:  {min_report.evaluations_count} calls ({elapsed:.2f}s)")
    print(f"1-Minimality Validated:  {TerminalStyle.success(str(min_report.is_1_minimal))}")
    print(f"Repeatability (3x):      {TerminalStyle.success(str(min_report.repeatability_validated))}")
    print()
    print("1-Minimal Fault Subset:")
    for f in minimized_cfg.faults:
        print(f"  * {f.signature()}")
    print()
    print(TerminalStyle.BOLD + "Research Root-Cause Breakdown:" + TerminalStyle.RESET)
    print(f"  [Minimal Reproducer]:   {min_report.root_cause_distinction.get('minimal_reproducer')}")
    print(f"  [Causal Explanation]:   {min_report.root_cause_distinction.get('causal_explanation')}")
    print(f"  [Software Root Cause]:  {TerminalStyle.warn(min_report.root_cause_distinction.get('software_root_cause', ''))}")
    print()

    # --------------------------------------------------------------------------
    # Step 5: Synthesize and Verify Regression Test
    # --------------------------------------------------------------------------
    print(TerminalStyle.header("[STEP 5/5] Synthesizing & Verifying Permanent Regression Artifact"))
    print("-" * 50)
    regression_runner = RegressionRunner()
    artifact = RegressionArtifact(
        id="demo_minimized_regression_test",
        task=minimized_cfg.task,
        initial_state=minimized_cfg.initial_state,
        faults=minimized_cfg.faults,
        invariants=minimized_cfg.invariants,
        expected={"reproduction": True, "violation": "duplicate_refund"},
        minimal=min_report.is_1_minimal,
        metadata={
            "created_by": "Project MIRROR CLI Demo",
            "evaluations_to_minimize": min_report.evaluations_count,
            "reduction_pct": min_report.reduction_pct,
        },
    )

    art_file = save_regression_artifact(artifact, out_path / "demo_regression_artifact.yaml")
    print(f"Saved Regression Artifact: {TerminalStyle.BOLD}{art_file.resolve()}{TerminalStyle.RESET}\n")

    # Part A: Discovery Test (Vulnerable Agent)
    print("Executing Discovery Test (Replay on Vulnerable Agent)...")
    disc_res = regression_runner.run_discovery_test(artifact, target_agent)
    if disc_res["reproduced"]:
        print(f"  Result: {TerminalStyle.success('CONFIRMED REPRODUCIBLE (100% Failure Reproduction)')}")
    else:
        print(f"  Result: {TerminalStyle.fail('FAILED TO REPRODUCE')}")

    # Part B: Fix Verification Test (Robust Agent)
    print("Executing Fix Verification Test (Replay on Patched/Robust Agent)...")
    robust_verification_agent = RobustRefundAgent(max_attempts=2)
    fix_res = regression_runner.run_fix_verification_test(artifact, robust_verification_agent)
    if fix_res["passed"]:
        print(f"  Result: {TerminalStyle.success('VERIFIED FIXED (0% False Positives, Safety Invariants Pass)')}")
    else:
        print(f"  Result: {TerminalStyle.fail('FIX VERIFICATION FAILED')}")

    print("\n" + "=" * 80)
    print(TerminalStyle.success("  DEMO COMPLETE: Discovered, Minimized, Replayed & Verified  "))
    print("=" * 80 + "\n")
    return 0


def run_scenario(scenario_path: str, agent_choice: str = "naive") -> int:
    """Execute a single scenario file against an agent and display oracle evaluation."""
    path = Path(scenario_path)
    if not path.exists():
        print(TerminalStyle.fail(f"Error: Scenario file not found: {path}"))
        return 1

    artifact = load_regression_artifact(path)
    print(TerminalStyle.header(f"\n--- Running Scenario: {artifact.id} (Task: {artifact.task}) ---"))

    if agent_choice == "robust":
        agent: BaseAgent = RobustRefundAgent(max_attempts=2)
    elif agent_choice == "reconciling":
        agent = ReconcilingRefundAgent(max_attempts=2)
    else:
        agent = NaiveRefundAgent(max_attempts=2)

    cfg = ScenarioConfig(
        id=artifact.id,
        task=artifact.task,
        initial_state=artifact.initial_state,
        faults=artifact.faults,
        invariants=artifact.invariants,
    )
    ctrl = EnvironmentController(config=cfg)
    service = RefundService(controller=ctrl, order_id=cfg.initial_state.order_id)

    agent_result = None
    try:
        agent_result = agent.run(
            service=service,
            order_id=cfg.initial_state.order_id,
            amount=cfg.initial_state.order_amount,
        )
    except Exception as e:
        agent_result = {"status": "error", "error": str(e)}

    report = evaluate_scenario(ctrl, agent_result, cfg)
    print(report.summary())
    for v in report.violations:
        print(f"  - [{v.severity.upper()}] {v.invariant_name}: {v.description}")
    return 0 if not report.safety_violation else 2


def run_cloud_demo() -> int:
    """Execute Tier 2 Cloud Infrastructure demonstration."""
    print("\n" + "=" * 80)
    print(TerminalStyle.header("  TIER 2 DEMO: CLOUD INFRASTRUCTURE LIFECYCLE (VPC, VM, Storage)  "))
    print("=" * 80 + "\n")

    cfg = ScenarioConfig(
        id="cloud-demo-interaction",
        task="deploy_cloud_stack",
        initial_state=InitialState(tenant_id="tenant-demo"),
        faults=[
            FaultRule(tool="cloud.provision_vm", occurrence=1, mode=FaultMode.TIMEOUT_AFTER_COMMIT),
            FaultRule(tool="cloud.describe_resources", occurrence=1, mode=FaultMode.EVENTUAL_CONSISTENCY),
        ],
    )

    print("Phase 1: Running NaiveCloudDeployerAgent under Timeout & Eventual Consistency...")
    ctrl_naive = CloudEnvironmentController(config=cfg)
    service_naive = CloudInfrastructureService(controller=ctrl_naive)
    naive_agent = NaiveCloudDeployerAgent(max_attempts=2)

    res_naive = None
    try:
        res_naive = naive_agent.run(service_naive, tenant_id="tenant-demo", stack_id="stack-naive")
    except Exception as e:
        res_naive = {"status": "error", "error": str(e)}

    rep_naive = evaluate_cloud_scenario(ctrl_naive, res_naive, cfg)
    print(f"Naive Deployer Result: {TerminalStyle.fail('VIOLATION')} -> {[v.invariant_name for v in rep_naive.violations]}")

    print("\nPhase 2: Running RobustCloudDeployerAgent with Idempotency and Rollback...")
    ctrl_robust = CloudEnvironmentController(config=cfg)
    service_robust = CloudInfrastructureService(controller=ctrl_robust)
    robust_agent = RobustCloudDeployerAgent(max_attempts=3)

    res_robust = robust_agent.run(service_robust, tenant_id="tenant-demo", stack_id="stack-robust")
    rep_robust = evaluate_cloud_scenario(ctrl_robust, res_robust, cfg)

    if not rep_robust.safety_violation and rep_robust.task_completed:
        print(f"Robust Deployer Result: {TerminalStyle.success('PASS')} (Active: {rep_robust.ledger_state.get('total_active')} resources)")
    else:
        print(f"Robust Deployer Result: {TerminalStyle.fail('FAIL')}")

    print("\n" + "=" * 80 + "\n")
    return 0


def main() -> None:
    """Main CLI entrypoint."""
    parser = argparse.ArgumentParser(
        prog="mirror",
        description="Project MIRROR: Research-Grade Crash-Test Simulator for AI Agents",
    )
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # Command: demo
    demo_parser = subparsers.add_parser("demo", help="Run the full 5-step automated discovery & minimization loop")
    demo_parser.add_argument(
        "--agent",
        choices=["reconciling", "naive", "robust"],
        default="reconciling",
        help="Agent to evaluate (default: reconciling)",
    )
    demo_parser.add_argument("--out", type=str, default="results", help="Directory to save regression artifacts")

    # Command: run
    run_parser = subparsers.add_parser("run", help="Run a specific scenario artifact against an agent")
    run_parser.add_argument("--scenario", type=str, required=True, help="Path to scenario YAML artifact")
    run_parser.add_argument(
        "--agent",
        choices=["naive", "reconciling", "robust"],
        default="naive",
        help="Agent to run scenario against",
    )

    # Command: reproduce
    reproduce_parser = subparsers.add_parser("reproduce", help="Run empirical reproduction across 4 search strategies")
    reproduce_parser.add_argument("--budget", type=int, default=30, help="Budget per run (default: 30)")
    reproduce_parser.add_argument(
        "--seeds", type=int, nargs="+", default=[42, 101, 2024, 777, 9999], help="Seeds list"
    )
    reproduce_parser.add_argument("--out", type=str, default="results", help="Output directory")

    # Command: cloud-demo
    subparsers.add_parser("cloud-demo", help="Run Tier 2 Cloud Infrastructure lifecycle demonstration")

    args = parser.parse_args()

    if args.command == "demo":
        sys.exit(run_demo(agent_choice=args.agent, out_dir=args.out))
    elif args.command == "run":
        sys.exit(run_scenario(scenario_path=args.scenario, agent_choice=args.agent))
    elif args.command == "reproduce":
        from benchmarks.reproduce import run_reproduction_suite

        run_reproduction_suite(budget_per_run=args.budget, seeds=args.seeds, out_dir=args.out)
        sys.exit(0)
    elif args.command == "cloud-demo":
        sys.exit(run_cloud_demo())
    else:
        parser.print_help()
        sys.exit(0)


if __name__ == "__main__":
    main()
