# Git rules

- Never run `git commit` (including `--amend`) without Glen's explicit approval for that specific commit. Show the files and the commit message first and wait for a yes.
- Never run `git push` (including force pushes) without Glen's explicit approval for that specific push.
- Approval for one commit or push does not carry over to the next one.
- Never add Claude attribution (Co-Authored-By trailers, "Generated with Claude Code") to commit messages or PR descriptions.

# Thesis template (Report/0_main.tex)

- `Report/0_main.tex` is the standard UCT-MARIS-ARU template and is the reference version. Do not change anything in it without Glen's express permission for that specific change. This includes packages, margins, fonts, `\includeonly`, the chapter list and the bibliography setup.
- If a build error or another task seems to need a change to 0_main.tex, stop, explain the change and ask first.
- Changes already approved (2026-10-22): Times New Roman font block (newtxtext/newtxmath on pdfLaTeX, fontspec + Times New Roman on XeLaTeX/LuaLaTeX), `\IEEEcitetrue`, `\usetikzlibrary{arrows.meta, positioning, calc}`, `\usepackage{listings}`, `\bibliography{thesis,FIXED}`, `\include{5b_reconstruction}` (after 5_ch5), `\include{9_appendix}` (after D_ECSA), the `\citeurl` command (after natbib), and margins `margin=2cm` (all sides, permanent).

# Frozen code

- The calibration code is finished. Do not edit anything in `src/01_Calibration/` (Calibration.py, detector_config.py, detector_presets.json and the scripts beside it). Running it is fine. If a task seems to need a change there, stop and ask Glen first.
