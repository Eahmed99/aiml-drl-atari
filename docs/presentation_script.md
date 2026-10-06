# Recording guide — target 6 minutes 25 seconds

The PPTX includes speaker notes. This guide supplies the speaking order and practical evidence.
Group and contributor ownership must be verified before recording. Rotate presenters so each speaks
roughly 90–100 seconds; the allocation below is suggested, not a claim that work occurred.

## Presenter 1: 0:00–1:35, slides 1–3
Introduce the assigned paper and all members. Explain the learning task: our agent sees pixels,
chooses actions and receives rewards. A Q-table would require a separate entry for an enormous
number of screen configurations. A CNN instead shares learned visual features across states.
Show `src/dqn.py`, the captured frame stack and `network_summary.txt`. Explain that a single still
image cannot reliably reveal ball direction. Four observations provide motion cues. The network
outputs one Q-value per action and has no output ReLU because returns can be negative.

## Presenter 2: 1:35–2:50, slides 4–5
Show `src/replay_buffer.py` and `src/agent.py`. Walk through one transition and one sampled batch.
Explain epsilon-greedy exploration and why replay uses older experience. Show the terminal mask:
a genuine terminal transition has target equal to reward; a time-limit truncation still permits
bootstrapping. Work the numerical example on slide 5. Explain `gather`, `no_grad`, `backward`, and
`optimizer.step`. Our target is detached for the update; it comes from the same model. The replay
test also checks eviction and copying, so changing a live observation cannot corrupt old data.

## Presenter 3: 2:50–4:05, slides 6–7
Read the published table as published evidence, not your own experiment. Explain that aggregate
success on some games does not mean success on every game. Show your actual training command,
configuration, hardware, version file and summary. Explain that our step counter counts agent
actions after frame skip, not emulator frames. Show screenshot evidence that preprocessing and
training executed. Describe the reduced replay capacity and the explicit FIRE helper.

## Presenter 4: 4:05–6:25, slides 8–11
Replace the result placeholders with actual evaluation mean and standard deviation. Show scores
for both policies and a short recorded gameplay segment. If the trained policy did not beat random,
say so: our measured result does not demonstrate improved policy performance under this budget.
Do not interpret reduced loss as sufficient proof. Discuss one-game/single-training-seed limitations;
evaluation episodes are not independent training runs. Explain a next experiment: more training,
several training seeds, or comparing replay with sequential updates. Show contributor evidence,
close with the learning pipeline conclusion, and show the recording link and references.

Adjust the timing so all members have equal speaking/contribution opportunity. Slide 10 should
contain actual work rather than this suggested allocation. The recording must demonstrate each
member's understanding and work; gameplay.mp4 alone does not meet the recording requirement.
