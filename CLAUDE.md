# Git rules

- Never run `git commit` (including `--amend`) without Glen's explicit approval for that specific commit. Show the files and the commit message first and wait for a yes.
- Never run `git push` (including force pushes) without Glen's explicit approval for that specific push.
- Approval for one commit or push does not carry over to the next one.
- Never add Claude attribution (Co-Authored-By trailers, "Generated with Claude Code") to commit messages or PR descriptions.

# Frozen code

- The calibration code is finished. Do not edit anything in `src/01_Calibration/` (Calibration.py, detector_config.py, detector_presets.json and the scripts beside it). Running it is fine. If a task seems to need a change there, stop and ask Glen first.
