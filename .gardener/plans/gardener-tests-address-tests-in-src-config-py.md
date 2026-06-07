<!-- gardener-maintenance-pr-plan-id: 466dc1d7-0736-45e1-a7e9-f248b2ace415 -->
# Plan 3 tests maintenance opportunities

## Goal

1 tests signal found in src/config.py. | 1 tests signal found in src/evaluate_agent.py. | 1 tests signal found in src/run_training.py.

## Evidence

src/config.py: Repowise did not find a paired test file. | src/evaluate_agent.py: Repowise did not find a paired test file. | src/run_training.py: Repowise did not find a paired test file. Categories checked against constitution allowed fixes: tests. Changed paths checked against protected modules and never-touch paths: src/config.py, src/evaluate_agent.py, src/run_training.py.

## Entropy impact

Expected -3.6 entropy delta across 3 path(s).

## Verification

Required checks: none configured. Risk tier: tier_2_assisted. Minimum opportunity confidence 0.70; threshold 0.50; meets PR creation threshold. Changed paths: src/config.py, src/evaluate_agent.py, src/run_training.py. Rollback: revert the focused PR branch if checks or review fail.

## ROI impact

Estimated 1.0–2.1 engineering hours saved. 3 high-entropy-delta path(s) addressed. Assumptions: 1.0 hrs/file for tests; confidence 0.70, 0.70, 0.70; conservative scale 0.5–1.0×. Estimates are conservative and indicative only.
