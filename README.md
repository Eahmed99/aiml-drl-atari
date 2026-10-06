# Contributors

| Contributor | BITS ID |
|---|---|
| EJAZ AHMED | 2025AG05320 |
| ARSHAD HUSAIN SIDDIQUI | 2025AG05458 |
| SAHIL FARAZ ANSARI | 2025AG05719 |
| SYED ANAS AHMED | 2025AG05726 |

# Atari DQN assignment

This repository contains a small Breakout experiment based on [Playing Atari with Deep Reinforcement Learning, Mnih et al. (2013)](https://arxiv.org/abs/1312.5602). It covers the main DQN steps: processing game frames, choosing actions, storing experience and updating the Q network.

The code and results support the paper presentation and group recording.

## Files

| File | Purpose |
|---|---|
| `src/dqn.py` | CNN that predicts a Q-value for each action |
| `src/replay_buffer.py` | Stores transitions and samples random batches |
| `src/train.py` | Environment setup, training, screenshots and plots |
| `src/evaluate.py` | Compares the checkpoint with random play and records clips |

The notebook runs the same code. Settings are in `DEFAULT_CONFIG` and `SMOKE_CONFIG` inside `train.py`. Both use a replay capacity of 5,000 transitions.

## Run the code

Run these commands from the repository root with the virtual environment active:

```bash
python -m pip install -r requirements.txt
python -m src.dqn
python -m src.train --demo
python -m src.train --smoke --out results/my-smoke-01
```

The smoke run takes 2,000 agent steps and checks that training executes. It saves the checkpoint, logs, summary and plots. The demo saves the game screen and four-frame input in `screenshots/`.

Use a new output folder for each training or evaluation run. For example, change `my-smoke-01` to `my-smoke-02` when running it again.

For the default 5,000-step run:

```bash
python -m src.train --out results/breakout
python -m src.evaluate --checkpoint results/breakout/checkpoint.pt --out results/evaluation --episodes 10
```

Evaluation saves scores for both policies, a comparison plot and short gameplay clips.

Useful options:

- `--steps 200000`: change the number of training steps.
- `--device cpu`: run on CPU.
- `--video-seconds 0`: skip clips during evaluation.
- `--config PATH`: load settings from an external JSON file.

To redraw the plots:

```bash
python -m src.train --plot-run results/breakout --evaluation results/evaluation
```

## Training setup

The input is four grayscale 84×84 frames. The network has two convolutional layers with 16 and 32 filters, a 256-unit hidden layer and one output per action. Pixels stay as uint8 in replay and are divided by 255 in the network.

Actions follow an epsilon-greedy policy. Epsilon decreases from 1.0 to 0.1 over the first 4,000 steps of the default 5,000-step run. Training uses batches of 32, gamma 0.99, RMSprop and MSE loss. Rewards are clipped for learning; the reported scores use the original rewards.

The TD target uses the same network, with gradients disabled for the next-state prediction. There is no separate target network. A true terminal state removes the future-reward term; a time limit ends the episode without removing that term.

Breakout needs FIRE to launch the ball. The environment helper launches it on reset and the action-selection code relaunches it after a life loss. Rewards from the reset launch are recorded separately.

## Results

The saved experiment is in `results/previous_run/`. It completed 100,000 agent steps and 24,751 network updates.

| Policy | Mean score | Evaluation episodes |
|---|---:|---:|
| Random | 1.2 | 10 |
| Trained | 0.4 | 10 |

The trained policy scored below random play in this run. The experiment confirms that the code executes and updates the network, but it does not show improved gameplay. TD loss alone cannot establish better performance.

This run uses one game, a short training budget and a replay capacity of 5,000. The preprocessing also differs from the original paper: Gymnasium resizes directly to 84×84 and applies max-pooling. These results should be presented separately from the paper's benchmark results.

The [full experiment artifact](https://github.com/Eahmed99/aiml-drl-atari/actions/runs/37510403976/artifacts/11434816652) includes the checkpoint and complete loss log. The committed `losses_sampled.csv` contains every 100th update and the final row; the saved loss plot uses all updates.

## Presentation

The slides are in `submission/DRL_PlayingAtari_GroupPENDING.pptx`.

Before submitting:

- Fill the group number and names/BITS IDs on the first slide.
- Add each member's actual contribution and supporting evidence.
- Put the accessible Drive recording link on the last slide.
- Keep the recording within 7 minutes and both the PPT and video below 10 MB.

Explain the paper's motivation, algorithm, results and limitations. Use the screenshots, logs and clips to explain this experiment.
