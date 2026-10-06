# Assignment completion checklist

Source: supplied DRL Assignment I PDF dated 20261004. This is the DRL paper presentation assignment,
not the separate ACI notebook/Word assignment.

- Confirm group is 41–80 and Problem II is allocated; replace group placeholder.
- Verify all four names and BITS IDs on the first slide.
- Run tests with PyTorch installed: zero skipped learning tests required for full validation.
- Execute smoke run; summary must show nonzero optimizer updates and finite loss.
- Run actual experiment and evaluation; retain configuration, logs and version files.
- Generate plots and screenshots. Record actual CLI output as contributor evidence.
- Generate final deck with group number and accessible Google Drive recording link.
- Replace pending contributor rows with actual work and evidence.
- Check motivation/objective (2 marks), techniques (2), results/limitations/future/conclusion (3).
- Record narrated presentation with proof of each member's work (5 marks).
- Duration <= 7 minutes; aim for 6–6.5 minutes. Gameplay clips are supporting material only.
- Keep submitted PPT and video each below 10 MB to satisfy size wording conservatively.
- Filename: DRL_AssignmentName_GroupNo. Generator uses DRL_PlayingAtari_GroupN.pptx.
- Put video Google Drive URL on last slide; verify access before upload.
- Submit both PPT and recording through the official assignment channel; only one submission allowed.
- Verify official deadline in your course portal; the supplied PDF does not specify a date.

Example video compression with ffmpeg (after recording; inspect quality):
```bash
ffmpeg -i recording.mp4 -vf scale=960:-2 -c:v libx264 -b:v 140k -c:a aac -b:a 32k DRL_PlayingAtari_GroupN.mp4
```
At seven minutes this bitrate targets roughly 9 MB before container overhead. Check actual bytes;
if still >=10 MB, reduce bitrate or duration. Check that code/screenshots remain readable.

The deck generator inserts results if logs/plots exist. It intentionally labels absent experiment
results as pending. Do not submit the initial draft with these pending fields.
