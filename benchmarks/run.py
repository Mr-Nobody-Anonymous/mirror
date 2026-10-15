"""Benchmark suite for Project MIRROR.

Compares:
1. Random Fault Generation
2. Manually Specified Fault Combinations
3. Delta-Debugging-based Minimization
4. MIRROR's Adaptive Discovery & Minimization

Measures:
- Unique failure scenarios found
- Experiments to first discovery
- Reproduction rate
- Faults remaining after minimization
- Regression-test validity
- Simulation calls per discovery
- False-positive rate
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List

# Ensure src is on sys.path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from mirror.agent import NaiveRefundAgent, ReconcilingRefundAgent, RobustRefundAgent
from mirror.minimize import ScenarioMinimizer
from mirror.models import FaultMode, FaultRule, RegressionArtifact, ScenarioConfig
from mirror.regression import RegressionRunner, save_regression_artifact
from mirror.search import AdaptiveSearchEngine, RandomSearchEngine


def run_benchmarks(
    budget_per_run: int = 30,
    seeds: List[int] = [42, 101, 2024],
    results_dir: str = "results",
) -> Dict[str, Any]:
    out_dir = Path(results_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    naive_agent = NaiveRefundAgent(max_attempts=2)
    robust_agent = RobustRefundAgent(max_attempts=2)
    reconciling_agent = ReconcilingRefundAgent(max_attempts=2)
    regression_runner = RegressionRunner()

    print("======================================================================")
    print("Project MIRROR: Research-Grade Benchmarking Suite")
    print(f"Seeds: {seeds} | Budget per run: {budget_per_run} experiments")
    print("======================================================================\n")

    benchmark_summary: Dict[str, Any] = {}

    # -------------------------------------------------------------------------
    # 1. Random Fault Generation Baseline
    # -------------------------------------------------------------------------
    print("[1/4] Running Random Fault Generation Baseline...")
    rand_first_discoveries: List[int] = []
    rand_unique_violations: List[int] = []
    rand_total_discoveries: List[int] = []

    for s in seeds:
        engine = RandomSearchEngine(agent=naive_agent, seed=s)
        res = engine.run(budget=budget_per_run)
        if res.first_discovery_step is not None:
            rand_first_discoveries.append(res.first_discovery_step)
        rand_unique_violations.append(res.unique_violations_found)
        rand_total_discoveries.append(len(res.discovered_scenarios))

    avg_rand_first = sum(rand_first_discoveries) / max(len(rand_first_discoveries), 1)
    avg_rand_unique = sum(rand_unique_violations) / len(seeds)
    avg_rand_total = sum(rand_total_discoveries) / len(seeds)

    benchmark_summary["random_search"] = {
        "avg_first_discovery_step": round(avg_rand_first, 2),
        "avg_unique_violations": round(avg_rand_unique, 2),
        "avg_total_failures_found": round(avg_rand_total, 2),
        "discovery_rate_pct": round((avg_rand_total / budget_per_run) * 100, 2),
        "cost_per_discovery_calls": round(budget_per_run / max(avg_rand_total, 0.001), 2),
    }

    # -------------------------------------------------------------------------
    # 2. Manual Fault Baseline
    # -------------------------------------------------------------------------
    print("[2/4] Running Manually Specified Fault Combinations...")
    manual_scenarios = [
        ScenarioConfig(
            id="man-01",
            faults=[FaultRule(tool="payment.refund", occurrence=1, mode=FaultMode.TIMEOUT_AFTER_COMMIT)],
        ),
        ScenarioConfig(
            id="man-02",
            faults=[FaultRule(tool="payment.refund", occurrence=1, mode=FaultMode.TIMEOUT_BEFORE_COMMIT)],
        ),
        ScenarioConfig(
            id="man-03",
            faults=[FaultRule(tool="payment.refund", occurrence=1, mode=FaultMode.PERMISSION_REVOKED)],
        ),
        ScenarioConfig(
            id="man-04",
            faults=[FaultRule(tool="payment.get_order_status", occurrence=1, mode=FaultMode.STALE_READ)],
        ),
    ]

    manual_violations = 0
    for sc in manual_scenarios:
        art = RegressionArtifact(
            id=sc.id,
            task=sc.task,
            initial_state=sc.initial_state,
            faults=sc.faults,
            invariants=sc.invariants,
        )
        res = regression_runner.run_discovery_test(art, naive_agent)
        if res["reproduced"]:
            manual_violations += 1

    benchmark_summary["manual_baseline"] = {
        "scenarios_evaluated": len(manual_scenarios),
        "violations_discovered": manual_violations,
        "discovery_efficiency_pct": round((manual_violations / len(manual_scenarios)) * 100, 2),
    }

    # -------------------------------------------------------------------------
    # 3. Delta Debugging Minimization (ddmin)
    # -------------------------------------------------------------------------
    print("[3/4] Running Delta Debugging Minimization Suite...")
    # Create scenario with red-herring / benign faults + true root-cause fault
    bloated_scenario = ScenarioConfig(
        id="bloated-001",
        faults=[
            FaultRule(tool="payment.get_order_status", occurrence=1, mode=FaultMode.STALE_READ),
            FaultRule(tool="payment.refund", occurrence=1, mode=FaultMode.TIMEOUT_AFTER_COMMIT),
            FaultRule(tool="payment.get_order_status", occurrence=2, mode=FaultMode.STALE_READ),
        ],
    )
    minimizer = ScenarioMinimizer(agent=naive_agent, target_violation_type="duplicate_refund")
    min_sc, evals_used = minimizer.minimize(bloated_scenario, method="ddmin")

    original_count = len(bloated_scenario.faults)
    minimized_count = len(min_sc.faults)
    reduction_pct = round(((original_count - minimized_count) / original_count) * 100, 2)

    # Verify minimality: verify removing remaining fault stops the violation
    is_strictly_1_minimal = True
    for i in range(len(min_sc.faults)):
        sub = min_sc.faults[:i] + min_sc.faults[i + 1:]
        rep, _ = minimizer.test_fault_subset(min_sc, sub)
        if rep:
            is_strictly_1_minimal = False

    # Also evaluate multi-fault interaction minimization on ReconcilingRefundAgent
    bloated_interaction = ScenarioConfig(
        id="bloated-interact-002",
        faults=[
            FaultRule(tool="payment.refund", occurrence=1, mode=FaultMode.TIMEOUT_AFTER_COMMIT),
            FaultRule(tool="payment.refund", occurrence=3, mode=FaultMode.SERVER_ERROR),
            FaultRule(tool="payment.get_order_status", occurrence=1, mode=FaultMode.STALE_READ),
            FaultRule(tool="payment.get_order_status", occurrence=2, mode=FaultMode.RATE_LIMITED),
        ],
    )
    minimizer_interact = ScenarioMinimizer(agent=reconciling_agent, target_violation_type="duplicate_refund")
    min_interact, evals_interact = minimizer_interact.minimize(bloated_interaction, method="ddmin")

    benchmark_summary["delta_debugging"] = {
        "original_fault_count": original_count,
        "minimized_fault_count": minimized_count,
        "reduction_percentage": reduction_pct,
        "evaluations_to_minimize": evals_used,
        "is_strictly_1_minimal": is_strictly_1_minimal,
        "interaction_original_faults": len(bloated_interaction.faults),
        "interaction_minimized_faults": len(min_interact.faults),
        "interaction_reduction_pct": round(
            ((len(bloated_interaction.faults) - len(min_interact.faults)) / len(bloated_interaction.faults)) * 100, 2
        ),
        "interaction_evaluations": evals_interact,
    }

    # -------------------------------------------------------------------------
    # 4. MIRROR Adaptive Discovery & Minimization
    # -------------------------------------------------------------------------
    print("[4/4] Running MIRROR Adaptive Discovery & Minimization Engine...")
    adapt_first_discoveries: List[int] = []
    adapt_unique_violations: List[int] = []
    adapt_total_discoveries: List[int] = []
    saved_artifacts: List[Path] = []
    reproduction_successes = 0
    fix_verification_successes = 0
    false_positives = 0

    for s in seeds:
        engine = AdaptiveSearchEngine(agent=naive_agent, seed=s)
        res = engine.run(budget=budget_per_run)
        if res.first_discovery_step is not None:
            adapt_first_discoveries.append(res.first_discovery_step)
        adapt_unique_violations.append(res.unique_violations_found)
        adapt_total_discoveries.append(len(res.discovered_scenarios))

        # Minimize first discovered scenario per seed and generate permanent regression test
        if res.discovered_scenarios:
            first_fail = res.discovered_scenarios[0]
            min_cand, _ = minimizer.minimize(first_fail)

            artifact = RegressionArtifact(
                id=f"regression_seed_{s}",
                task=min_cand.task,
                initial_state=min_cand.initial_state,
                faults=min_cand.faults,
                invariants=min_cand.invariants,
                expected={"reproduction": True, "violation": "duplicate_refund"},
                minimal=True,
                metadata={"seed": s, "discovered_by": "AdaptiveSearchEngine"},
            )
            art_file = save_regression_artifact(artifact, out_dir / f"regression_seed_{s}.yaml")
            saved_artifacts.append(art_file)

            # Test 1: Discovery Test (reproduction on naive agent)
            disc_res = regression_runner.run_discovery_test(artifact, naive_agent)
            if disc_res["reproduced"]:
                reproduction_successes += 1
            else:
                false_positives += 1

            # Test 2: Fix Verification Test (safeguard verification on robust agent)
            fix_res = regression_runner.run_fix_verification_test(artifact, robust_agent)
            if fix_res["passed"]:
                fix_verification_successes += 1

    avg_adapt_first = sum(adapt_first_discoveries) / max(len(adapt_first_discoveries), 1)
    avg_adapt_unique = sum(adapt_unique_violations) / len(seeds)
    avg_adapt_total = sum(adapt_total_discoveries) / len(seeds)

    benchmark_summary["adaptive_engine"] = {
        "avg_first_discovery_step": round(avg_adapt_first, 2),
        "avg_unique_violations": round(avg_adapt_unique, 2),
        "avg_total_failures_found": round(avg_adapt_total, 2),
        "discovery_rate_pct": round((avg_adapt_total / budget_per_run) * 100, 2),
        "cost_per_discovery_calls": round(budget_per_run / max(avg_adapt_total, 0.001), 2),
        "reproduction_rate_pct": round((reproduction_successes / max(len(saved_artifacts), 1)) * 100, 2),
        "fix_verification_pass_rate_pct": round((fix_verification_successes / max(len(saved_artifacts), 1)) * 100, 2),
        "false_positive_rate_pct": round((false_positives / max(len(saved_artifacts), 1)) * 100, 2),
        "generated_artifacts_count": len(saved_artifacts),
    }

    # -------------------------------------------------------------------------
    # Comparison & Summary Output
    # -------------------------------------------------------------------------
    report_json = out_dir / "benchmark_report.json"
    with open(report_json, "w", encoding="utf-8") as f:
        json.dump(benchmark_summary, f, indent=2)

    csv_path = out_dir / "benchmark_summary.csv"
    with open(csv_path, "w", encoding="utf-8") as f:
        f.write("Method,FirstDiscoveryStep,TotalFailuresFound,DiscoveryEfficiency,CostPerDiscovery\n")
        f.write(
            f"RandomSearch,{benchmark_summary['random_search']['avg_first_discovery_step']},"
            f"{benchmark_summary['random_search']['avg_total_failures_found']},"
            f"{benchmark_summary['random_search']['discovery_rate_pct']}%,{benchmark_summary['random_search']['cost_per_discovery_calls']}\n"
        )
        f.write(
            f"ManualBaseline,N/A,{benchmark_summary['manual_baseline']['violations_discovered']},"
            f"{benchmark_summary['manual_baseline']['discovery_efficiency_pct']}%,N/A\n"
        )
        f.write(
            f"MIRROR_Adaptive,{benchmark_summary['adaptive_engine']['avg_first_discovery_step']},"
            f"{benchmark_summary['adaptive_engine']['avg_total_failures_found']},"
            f"{benchmark_summary['adaptive_engine']['discovery_rate_pct']}%,{benchmark_summary['adaptive_engine']['cost_per_discovery_calls']}\n"
        )

    print("\n======================================================================")
    print("                    BENCHMARK RESULTS SUMMARY                         ")
    print("======================================================================")
    print(f"{'Metric':<35} | {'Random Search':<16} | {'MIRROR Adaptive':<16}")
    print("-" * 72)
    print(f"{'Experiments to 1st Discovery':<35} | {benchmark_summary['random_search']['avg_first_discovery_step']:<16} | {benchmark_summary['adaptive_engine']['avg_first_discovery_step']:<16}")
    print(f"{'Total Failures Discovered (N=30)':<35} | {benchmark_summary['random_search']['avg_total_failures_found']:<16} | {benchmark_summary['adaptive_engine']['avg_total_failures_found']:<16}")
    print(f"{'Discovery Rate (%)':<35} | {str(benchmark_summary['random_search']['discovery_rate_pct']) + '%':<16} | {str(benchmark_summary['adaptive_engine']['discovery_rate_pct']) + '%':<16}")
    print(f"{'Simulated Calls / Discovery':<35} | {benchmark_summary['random_search']['cost_per_discovery_calls']:<16} | {benchmark_summary['adaptive_engine']['cost_per_discovery_calls']:<16}")
    print("-" * 72)
    print(f"Delta Debugging Fault Reduction: {reduction_pct}% (from {original_count} to {minimized_count} faults in {evals_used} evals)")
    print(f"Regression Reproduction Rate:   {benchmark_summary['adaptive_engine']['reproduction_rate_pct']}%")
    print(f"Fix Verification Pass Rate:     {benchmark_summary['adaptive_engine']['fix_verification_pass_rate_pct']}%")
    print(f"False Positive Rate:             {benchmark_summary['adaptive_engine']['false_positive_rate_pct']}%")
    print(f"Saved artifacts and reports to:  {out_dir.resolve()}")
    print("======================================================================\n")

    return benchmark_summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run Project MIRROR Benchmark Suite")
    parser.add_argument("--budget", type=int, default=30, help="Budget per run")
    parser.add_argument("--out", type=str, default="results", help="Output directory")
    args = parser.parse_args()

    run_benchmarks(budget_per_run=args.budget, results_dir=args.out)
