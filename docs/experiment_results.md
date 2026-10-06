# Observed experiment — 6 October 2026

Training source: https://github.com/Eahmed99/aiml-drl-atari/actions/runs/37508726802
Successful recovery/evaluation: https://github.com/Eahmed99/aiml-drl-atari/actions/runs/37510403976

| Setting / outcome | Recorded value |
|---|---|
| Environment | ALE/Breakout-v5 |
| Training seed | 42 |
| Hardware | GitHub-hosted Ubuntu CPU |
| Training agent steps | 100,000 |
| Optimizer updates | 24,751 |
| Completed training episodes | 563 |
| Training wall time | 517.09 seconds |
| Evaluation seeds | 1000–1009, both policies |
| Trained evaluation epsilon | 0.05 |
| Scores | Original rewards, excluding separately recorded reset launch rewards |

| Policy | Episodes | Mean score | Sample standard deviation |
|---|---:|---:|---:|
| Random with shared FIRE helper | 10 | 1.20 | 1.6193 |
| Final trained checkpoint | 10 | 0.40 | 0.6992 |

The trained policy's observed mean was lower than the random baseline. This experiment does not
demonstrate improved gameplay under the chosen budget. It does demonstrate executable observation
processing, replay, gradient updates, checkpoint saving/loading and evaluation. The result is not
an exact reproduction of the published paper. Ten evaluation episodes with one training seed are
not sufficient for a broad conclusion about the algorithm.

Training scores averaged 1.416; the last 100 completed training episodes averaged 1.37. These are
exploratory training-policy scores, not evaluation scores. Loss was finite but its movement is not
a substitute for observed game performance.

Suggested next experiments: increase the budget with a matched exploration schedule; repeat over
several training seeds; inspect gameplay and replay reward coverage. Keep final evaluation seeds
separate from development decisions. Target-network comparison would be a clearly labelled later
extension, not the original algorithm reproduced exactly.

Checkpoint compatibility fix: the first run saved an np.int64 action count. The completed training
was retained, and this scalar was migrated to a native Python int using a scoped allowlist.
All future checkpoints save native metadata. The eight-test suite includes a full training
checkpoint save/load regression test, and the successful recovery workflow evaluated the same
100,000-step weights. See `scripts/recover_experiment.py` for the one-time migration.

`evidence/experiment_100k/` contains the captured settings, sampled loss logs, screenshots, plots and gameplay.
Its README links the full artifact with sanitized weights and complete loss logs. Do not represent this generated experiment as proof of each member's individual
work. Contributors must verify and record their actual tasks and speaking segments.
