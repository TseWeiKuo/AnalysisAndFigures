# Repository Guidelines

## Project Structure & Module Organization

This repository contains a Python analysis pipeline for 3D kinematic, landing probability, landing latency, and optogenetic experiments.

- `analyze3D_kinematics.py`: main runner that builds groups, creates figures, and runs statistical analyses.
- `group_config.py`: central configuration for absolute data paths, group names, fly counts, FPS, trial counts, offsets, and tracked keypoints.
- `kinematic_object.py`: data model layer for `Group`, `Trial`, and `Point`; reads manual Excel sheets and prepares LP/LL data.
- `kinematic_utilities.py`: geometry, angle, leg-search, secondary-contact, and helper analysis routines.
- `KinematicPlot.py`: plotting and figure-generation functions.
- `survival_stats_runner.py`: LP and survival/RMST statistical comparisons.

There is currently no `tests/` directory or packaged asset folder. Raw data and generated outputs live outside this repository in configured Windows paths.

## Build, Test, and Development Commands

Run the active analysis pipeline:

```powershell
python analyze3D_kinematics.py
```

Check syntax without executing the full analysis:

```powershell
python -m py_compile *.py
```

Install expected dependencies if your environment is missing them:

```powershell
pip install numpy pandas matplotlib seaborn scipy scikit-learn openpyxl lifelines natsort
```

Outputs are written to hard-coded folders such as `C:\Users\agrawal-admin\Desktop\Landing\Figures` and `C:\Users\agrawal-admin\Desktop\Landing\STAT`.

## Coding Style & Naming Conventions

Use 4-space indentation. Prefer `snake_case` for functions and variables, and `PascalCase` for classes. Preserve existing public method names, including mixed-case methods such as `Calculate_joint_angle`, because other scripts may call them.

Keep configuration edits in `group_config.py`, plotting changes in `KinematicPlot.py`, statistical changes in `survival_stats_runner.py`, and data-model changes in `kinematic_object.py`.

## Testing Guidelines

No automated test framework is currently configured. For any code change, at minimum run:

```powershell
python -m py_compile *.py
```

For behavioral changes, run a small active block in `analyze3D_kinematics.py` first and verify the generated CSV or PDF outputs manually.

## Commit & Pull Request Guidelines

No Git history is available in this folder, so no project-specific commit convention can be inferred. Use short, imperative commit messages, for example:

```text
Fix GTACR ON/OFF latency filtering
Add CsChrimson angle trace summary
```

Pull requests should describe the analysis change, list affected groups, mention changed paths or thresholds, and include representative output filenames. For plot changes, attach before/after figures when possible.

## Configuration & Data Safety

Do not commit raw experimental data, generated figures, or large CSV outputs unless intentionally publishing results. Treat absolute paths in `group_config.py` as machine-specific configuration and document any required folder layout changes.
