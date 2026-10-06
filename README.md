# Contributors

| Contributor | BITS ID |
|---|---|
| EJAZ AHMED | 2025AG05320 |
| ARSHAD HUSAIN SIDDIQUI | 2025AG05458 |
| SAHIL FARAZ ANSARI | 2025AG05719 |
| SYED ANAS AHMED | 2025AG05726 |

# Atari DQN assignment

A compact educational implementation of [Playing Atari with Deep Reinforcement Learning (Mnih et al., 2013)](https://arxiv.org/abs/1312.5602), using Breakout. The assignment deliverables are the paper presentation and group recording; this code supplies a small demonstration and measured evidence.

## Code

| File | Purpose |
|---|---|
| `src/dqn.py` | Convolutional Q network |
| `src/replay_buffer.py` | Seeded replay buffer |
| `src/train.py` | Environment, learning loop, settings, screenshots and plots |
| `src/evaluate.py` | Checkpoint versus random-policy evaluation and gameplay clips |

The notebook runs these same commands. Training defaults are inside `train.py`; optional `--config PATH` accepts an external JSON configuration.

## Run

From the repository root, with your virtual environment active:

```bash
python -m pip install -r requirements.txt
python -m src.dqn
python -m src.train --demo
python -m src.train --smoke --out results/my-smoke-01
```

The smoke check performs 2,000 agent steps. Each training/evaluation output folder must be new; change its name when repeating a run. Training saves a checkpoint, CSV logs, summary and plots automatically. Screenshots go into `screenshots/`.

For a 100,000-step experiment and evaluation:

```bash
python -m src.train --out results/breakout
python -m src.evaluate --checkpoint results/breakout/checkpoint.pt --out results/evaluation --episodes 10
```

Use `--steps 200000` to override the training duration, or `--device cpu` to select CPU. Evaluation writes random/trained scores, a comparison plot and two short gameplay clips. Use `--video-seconds 0` to skip clips. To regenerate plots:

```bash
python -m src.train --plot-run results/breakout --evaluation results/evaluation
```

For help checking your run, share the terminal output, training `summary.json`, `episodes.csv`, `losses.csv`, and evaluation `summary.json` when available.

## Method and limits

The network uses 16 filters (8×8, stride 4), 32 filters (4×4, stride 2), a 256-unit hidden layer and one output per action. Inputs are four grayscale 84×84 frames stored as uint8 and normalized once in the network. Learning uses epsilon-greedy actions, replay, clipped training rewards, detached same-network TD targets, MSE and RMSprop. Actual game scores remain unclipped. True termination masks bootstrapping; a time limit only ends the episode.

This is a small demonstration, not a reproduction of published scores: replay holds 5,000 transitions, training lasts 100,000 agent steps, and preprocessing directly resizes to 84×84 with max-pooling. A shared FIRE helper launches/relaunches Breakout; reset-launch rewards are logged separately. No separate target network is used, consistent with the 2013 algorithm.

## Recorded experiment and submission

`results/previous_run/` preserves the completed experiment's CSVs, plots, screenshots and clips. Across 10 evaluation episodes, random scored **1.2** on average and trained scored **0.4**. This run **did not demonstrate improvement**. Training completed 100,000 agent steps and 24,751 optimizer updates.

The [full experiment artifact](https://github.com/Eahmed99/aiml-drl-atari/actions/runs/37510403976/artifacts/11434816652) contains the checkpoint and complete losses CSV. The committed loss CSV is sampled; its existing plot was generated from all updates.

The results-filled PPT is in `submission/DRL_PlayingAtari_GroupPENDING.pptx`. Before submission, fill the group number, names/BITS IDs privately on the first slide and the Drive recording URL on the last slide. Record every member's actual contribution. Keep the video within 7 minutes and both PPT and video below 10 MB. Explain the paper's motivation, algorithm, results and limitations, and distinguish published results from this short demonstration.
