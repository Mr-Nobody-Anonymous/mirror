# Security Policy

## Supported Versions

| Version | Supported          |
| ------- | ------------------ |
| 0.1.x   | :white_check_mark: |

---

## Reporting a Vulnerability

The Project MIRROR maintainers take security seriously. If you discover a vulnerability or security flaw in MIRROR's simulation or evaluation engine, please disclose it responsibly.

### How to Report
- **Email**: Send vulnerability reports directly to `bmx310712@gmail.com`.
- **Details to Include**:
  - Description of the vulnerability and attack vector.
  - Minimal reproducible example or scenario configuration.
  - Potential impact on simulated environments, ledger integrity, or isolation boundaries.

We will acknowledge receipt within 48 hours and provide a coordinated disclosure timeline.

---

## Security Guarantees & Safe Simulation Guidelines

1. **Simulated vs. Production Credentials**:
   Project MIRROR is designed to run in controlled, simulated environments. **Never commit, supply, or inject live production credentials, real API secrets, or live financial account tokens into scenario configurations.**

2. **Authoritative Ledger Isolation**:
   The authoritative state ledger runs strictly isolated in memory and cannot be directly mutated or rewritten by agents under test. Agent operations must pass through the environment interceptor layer.

3. **Multi-Tenant Boundaries**:
   Tier 2 environments enforce strict tenant isolation. An agent executing under Tenant A's context is rejected from inspecting, creating, or modifying Tenant B's infrastructure resources.

4. **Local Execution**:
   MIRROR executes entirely on local CPU resources with deterministic pseudo-random seeds. It does not perform unconsented external network telemetries.
