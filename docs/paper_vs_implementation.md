# Scope and deviations

Primary source: https://arxiv.org/pdf/1312.5602 (2013), Algorithm 1 and sections 4–5.

| Component | This repository |
|---|---|
| CNN | Retains your 16-filter, 32-filter, 256-hidden-unit architecture |
| Target | Same current network, detached next-state values; no periodic target copy |
| Replay | Uniform sampling without replacement; capacity 5,000 full transitions |
| Loss | MSE; finite-loss check |
| Optimizer | PyTorch RMSprop: lr 0.00025, alpha 0.95, eps 0.01; these exact settings are ours |
| Preprocessing | Gymnasium grayscale, max-pooling, direct resize to 84×84; differs from original 110×84 then crop |
| Exploration | 1 → 0.1 over 80,000 agent steps; shorter schedule |
| Updates | Every four agent steps after warm-up; educational cadence |
| Evaluation | Ten complete episodes per policy with matching seeds and epsilon 0.05 |
| Games | Breakout only; smaller scope |
| FIRE | Explicit launch/reset and post-life-loss helper; game-specific deviation |
| Frame accounting | 100,000 agent actions, not the paper's frame budget |

The paper reports 10 million training frames and one million recent frames of replay.
Our reduced experiment and modern emulator protocol cannot be compared numerically as an exact
reproduction. Later ideas such as target networks, Double DQN and prioritized replay can be
discussed as future work but are not implemented here.

No checkpoint is selected by evaluation performance. Only the last training checkpoint is evaluated.
The random baseline uses the same FIRE helper and preprocessing. Evaluation score excludes reset
launch rewards, which are separately recorded. Matching seeds improves traceability but does not
make different policies traverse identical states.

Dependency version ranges are installation constraints, not a tested lockfile. Each training run
records core package versions. For repeatable reruns, preserve the environment using
`python -m pip freeze > results/breakout/requirements-lock.txt` after installation.
