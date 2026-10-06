# Captured 100,000-step experiment

Full artifact (weights and complete CSVs):
https://github.com/Eahmed99/aiml-drl-atari/actions/runs/37510403976/artifacts/11434816652

This snapshot includes plots, episode/evaluation logs, configuration, screenshots and gameplay.
`training/losses_sampled.csv` contains every 100th optimizer log row plus the last row. The loss
plot was generated from all 24,751 updates; the complete losses.csv is in the linked artifact.

The sanitized checkpoint is also in that artifact at `results/breakout/checkpoint.pt`. Extract
it to `evidence/experiment_100k/training/checkpoint.pt` before running the README's snapshot
re-evaluation command. This avoids embedding large model binaries in the source repository.

All values are measured, not simulated. See docs/experiment_results.md for interpretation.
