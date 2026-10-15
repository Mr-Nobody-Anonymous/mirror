"""Tests for the executable Project MIRROR CLI."""

from pathlib import Path

from mirror.cli import run_cloud_demo, run_demo, run_scenario


def test_cli_demo_execution(tmp_path: Path):
    code = run_demo(agent_choice="reconciling", out_dir=str(tmp_path))
    assert code == 0
    # Artifact was generated
    artifact_file = tmp_path / "demo_regression_artifact.yaml"
    assert artifact_file.exists()


def test_cli_demo_naive_agent(tmp_path: Path):
    code = run_demo(agent_choice="naive", out_dir=str(tmp_path))
    assert code == 0


def test_cli_run_scenario_detection():
    # Canonical timeout scenario against naive agent produces violation (exit code 2)
    code = run_scenario("scenarios/refund_timeout.yaml", agent_choice="naive")
    assert code == 2

    # Robust agent handles scenario cleanly (exit code 0)
    code_robust = run_scenario("scenarios/refund_timeout.yaml", agent_choice="robust")
    assert code_robust == 0


def test_cli_cloud_demo():
    code = run_cloud_demo()
    assert code == 0
