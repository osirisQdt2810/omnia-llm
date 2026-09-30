## Context
<!-- Why is this change needed? What problem does it solve? -->
<!-- Link related issues or incidents -->
- Related issue:
- Background / motivation:
- Constraints / assumptions:

---

## Content / Changes
<!-- What exactly changed in this PR -->
-
-
-

<!-- Optional: call out non-obvious changes -->
- Refactors:
- New features:
- Removed / deprecated behavior:

---

## Test Plan
<!-- How was this change validated -->

### Test Details
<!-- Commands, configs, or steps used to test -->
-

### Test Output / Feature Demonstration
<!-- Paste test output, logs, screenshots, benchmarks, or example requests/responses -->
-

---

## omnia-llm checklist
<!-- Delete a line only when it genuinely cannot apply. "N/A — why" is a valid answer. -->

- **Platforms**: which of NVIDIA / AMD / Apple / CPU this touches, and what ran on real hardware.
  <!-- CI runners have no GPU: say what you ran by hand, and where -->
- **HTTP surface**: does anything Omnia calls change (`/v1/models`, `/v1/chat/completions`,
  `/v1/images/generations`, `/health`, `/status`, `/warm`, `/stop`)? Omnia's server contract
  must stay true.
- **Security**: token checks, the lockout, and what an unauthenticated request can reach.
- **Public repository**: nothing that identifies a host, domain, user or token, and nothing from
  `state/`, `.model-cache/`, `bin/` or `deploy/*.env`.
- **Existing deployments**: does an install need a migration step (moved files, new config keys,
  a reinstall)? Say what, and in what order.
