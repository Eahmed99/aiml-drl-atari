# Playing Atari with Deep Reinforcement Learning

BITS Pilani DRL Assignment Problem II — educational implementation and paper presentation.
Assigned paper: [Mnih et al. (2013), arXiv:1312.5602](https://arxiv.org/abs/1312.5602).
Group number is not yet confirmed. Problem II is assigned to groups 41–80.

## What is implemented

Breakout image observations → grayscale 84×84 frames → four-frame stack → the existing
two-convolution CNN → epsilon-greedy actions → uniform replay → same-network TD targets
→ MSE → RMSprop updates. No separate periodically synchronized target network or Double DQN.

Training saves a checkpoint, original episode scores, clipped training returns, loss logs,
configuration, versions and timing. Evaluation compares the checkpoint with random actions
on matching environment seeds and records short gameplay clips. A changing TD loss alone
is not evidence that the policy improves.

GitHub Actions runs tests and smoke training on branch pushes. To request a fresh full experiment
in Actions, include `[experiment]` in a commit message. Ordinary documentation/evidence updates
do not need to repeat the 100,000-step experiment.

## Setup

Use Python 3.11 or 3.12 in a virtual environment. CPU is supported; CUDA is selected when available.
These versions are a practical setup choice, not a claim about the paper's environment.

```bash
git clone --branch replay_buffer_branch https://github.com/Eahmed99/aiml-drl-atari.git
cd aiml-drl-atari
python -m venv .venv
# macOS/Linux:
source .venv/bin/activate
# Windows PowerShell instead: .venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

ALE installation help: https://ale.farama.org/getting-started/ . If Breakout cannot find its ROM,
follow the instructions for your installed ALE version. Do not use obsolete ROM commands blindly.

## Run in order from repository root

```bash
python -m unittest discover -s tests -v
python -m src.dqn
python -m src.replay_buffer
python -m src.environment_demo
python -m src.train --config config/smoke.json --out results/smoke
python -m src.train --config config/breakout.json --out results/breakout
python -m src.evaluate --checkpoint results/breakout/checkpoint.pt --out results/evaluation --episodes 10
python -m src.plot_results --run results/breakout --evaluation results/evaluation
python scripts/build_presentation.py --group YOUR_GROUP --video-link YOUR_DRIVE_LINK
```

Use a NEW output directory for each experiment; logs are never silently overwritten.
Training checkpoints contain model weights and experiment settings for evaluation, not resumable
optimizer/replay state. The 2,000-step smoke run validates execution; the 100,000-step run is a
reduced-budget experiment and may not beat random. Increase `total_steps` and `epsilon_decay_steps`
together for longer experiments, recording every setting. Do not tune on evaluation seeds.

## Files

| Path | Purpose |
|---|---|
| `src/dqn.py` | Your original network, retained |
| `src/replay_buffer.py` | Bounded replay, ownership-safe uint8 copies |
| `src/environment.py` | Shared preprocessing, stack and FIRE helper |
| `src/agent.py` | TD target and gradient update |
| `src/train.py` | Interaction, learning, CSV logs and weights |
| `src/evaluate.py` | Random baseline, policy evaluation, MP4 clips |
| `src/plot_results.py` | Plots from actual CSVs |
| `src/environment_demo.py` | Screenshots and input/network evidence |
| `notebooks/dqn_atari_demo.ipynb` | Notebook calling the modules |
| `config/` | Smoke and experiment JSON settings |
| `docs/` | Assignment checklist, deviations, presentation and contributor evidence |
| `scripts/build_presentation.py` | Generate PPTX; embeds results when available |
| `tests/` | Replay, configuration, TD target and parameter-update checks |

## Experiment protocol and limitations

Step units are agent actions after frame skip 4; emulator frames also include reset no-ops and
launch actions. Training clips rewards but logs original rewards separately. `terminated` masks
bootstrapping; `truncated` resets the game without falsely treating time limit as terminal.
Life loss does not terminate replay transitions. FIRE on reset and after life loss is a disclosed
game-specific helper used identically for both evaluated policies. Launch reward is logged separately.

Replay stores two complete uint8 stacks per transition: 5,000 slots use about 282 MB for pixels,
plus overhead; 1,000 slots use 56 MB. It is intentionally simple, not a million-frame replay design.
No claim of paper score reproduction is made. Details: `docs/paper_vs_implementation.md`.

## Contributors

| Name | BITS ID | Actual contribution and evidence |
|---|---|---|
| Member 1 | FILL PRIVATELY | Fill after actual work; see contributor ledger |
| Member 2 | FILL PRIVATELY | Fill after actual work |
| Member 3 | FILL PRIVATELY | Fill after actual work |
| Member 4 | FILL PRIVATELY | Fill after actual work |

Fill contributor names and BITS IDs privately in the final submission; verify before upload. Generated code
and presentation material do not establish individual contribution. Each member must understand,
review and document their actual work; follow your institution's AI assistance policy.

## Submission

See `docs/submission_checklist.md`. Submit PPT and a narrated group presentation recording,
not just gameplay. The presentation draft has pending experiment/contribution/video fields;
it is not ready for submission until these are filled. No fabricated scores or contributor claims.
