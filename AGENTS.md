# Repository Guidelines

## Project Structure & Module Organization

This repository contains Python workflows for 3D kinematic analysis, landing probability, landing latency, secondary-contact summaries, and optogenetic/KIR/ablation figures.

- `analyze3D_kinematics.py`: active main runner for building groups, plotting figures, and running statistics.
- `group_config_new.py`: active configuration for data roots, groups, fly counts, trial counts, FPS, offsets, and keypoints.
- `group_config_old.py`: legacy configuration kept for comparison.
- `kinematic_object.py`: `Group`, `Trial`, and `Point` data model and data-loading logic.
- `kinematic_utilities.py`: geometry, angle, contact-search, and helper routines.
- `KinematicPlot.py` and `plot_*.py`: plotting and figure-generation functions.
- `survival_stats_runner.py`: landing probability, survival, and RMST statistics.
- `SC data/`: small secondary-contact CSV inputs used by analysis scripts.
- `*.ipynb`: figure-building and exploratory notebook workflows.

There is no formal `tests/` directory.

## Build, Test, and Development Commands

Run the active analysis pipeline:

```powershell
python analyze3D_kinematics.py
```

Check Python syntax without running the full analysis:

```powershell
python -m py_compile *.py
```

Install expected dependencies when needed:

```powershell
pip install numpy pandas matplotlib seaborn scipy scikit-learn openpyxl lifelines natsort
```

## Coding Style & Naming Conventions

Use 4-space indentation. Prefer `snake_case` for functions and variables and `PascalCase` for classes. Preserve existing public mixed-case method names such as `Calculate_joint_angle`, because external scripts may call them.

Keep changes scoped by responsibility: configuration in `group_config_new.py`, plotting in `KinematicPlot.py` or `plot_*.py`, statistics in `survival_stats_runner.py`, data-model behavior in `kinematic_object.py`, and shared helper logic in `kinematic_utilities.py`.

## Testing Guidelines

No automated test framework or coverage requirement is configured. For every code change, run:

```powershell
python -m py_compile *.py
```

For behavioral changes, run the smallest relevant block in `analyze3D_kinematics.py` or the relevant notebook/script, then manually inspect generated CSV, PDF, or figure outputs.

## Commit & Pull Request Guidelines

Use short, descriptive commit messages. Existing history includes update-style messages and concise imperative messages, for example:

```text
Fix T2 TiTa parser regression
Add CsChrimson angle trace summary
```

Pull requests should describe the analysis change, list affected groups, mention changed paths or thresholds, and include representative output filenames. For plot changes, include before/after figures when practical.

## Configuration & Data Safety

Treat absolute paths in configuration files as machine-specific. Do not commit raw experimental data, generated figure folders, large CSV outputs, or local cache files unless intentionally publishing them.
