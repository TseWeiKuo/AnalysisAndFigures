# Analysis Workflow Map

This document maps the current repository workflow at a practical level: where data enter, which functions are called by the notebooks, which files contain the real logic, and what each workflow returns or saves.

## Current Entry Points

The main active workflows are notebook-driven.

- `WT-LP_and_ConGeo.ipynb`: WT landing probability, landing latency, secondary contact, contact-angle traces, TT trajectory geometry, and TT metrics versus landing latency.
- `OptogeneticFigure.ipynb`: CsChrimson, GtACR, and optogenetic landing-probability/latency figures.
- `evaluation.ipynb`: tracking QC inspection, 2D/3D projection diagnostics, QC summaries, and TT QC sensitivity analysis.
- `AblationFigure.ipynb`: currently present but contains little or no active workflow code.

Most figure code should be modified through the notebook call and the corresponding `plot_*.py` function. Avoid adding new analysis logic directly into notebooks unless it is only parameter setup or display.

## Data And Configuration Flow

`group_config_new.py` is the active configuration file. It defines data roots, group names, fly counts, trial counts, trial offsets, FPS, metadata paths, and kinematic keypoints.

The normal data-loading chain is:

```text
notebook
  -> group_config_new.build_groups(...)
  -> kinematic_object.Group
  -> kinematic_object.Trial
  -> kinematic_object.Point
```

`Group` stores metadata tables such as LL, MOC, MOL, and kinematic file paths. `Trial` loads one trial. `Point` stores per-keypoint `x/y/z`, reprojection error, camera count, and optional score/likelihood-like columns.

Important input folders:

- `KinematicData/`: 3D kinematic CSV inputs.
- `Metadata/`: LL/MOC/MOL Excel metadata.
- `SC data/`: manual secondary-contact CSVs.
- `2DprojectionData/`: projected 2D camera-view data for QC inspection.

Important output folder:

- `Figures/`: notebook-generated PDFs and CSV summaries.

## Plotting Architecture

`KinematicPlot.py` defines `PlotCreator`, which is mostly a wrapper layer. Public notebook calls usually look like:

```python
plotter = kp.PlotCreator(platform_offset=0.07, platform_height=0.1, radius=0.03, fps=250)
plotter.some_plot_function(...)
```

Most `PlotCreator` methods immediately delegate to a module-level function:

```text
KinematicPlot.py wrapper
  -> plot_landing.py
  -> plot_geometry.py
  -> plot_angles.py
  -> plot_secondary_contact.py
  -> plot_optogenetics.py
```

When changing analysis behavior, the main logic is usually in the relevant `plot_*.py` file, not in `KinematicPlot.py`.

## WT Landing And Contact Geometry Workflow

Notebook: `WT-LP_and_ConGeo.ipynb`

Setup:

```text
build_groups(...)
PlotCreator(...)
SurvivalStatsRunner(...)
output_folder(...)
```

Main WT figure calls:

- `plot_LP_summary_from_groups`: fly-level landing probability summary.
- `stats_runner.compare_lp_unpaired`: unpaired LP statistics between WT groups.
- `plot_KM_curve_from_groups`: landing latency Kaplan-Meier curves.
- `plot_wt_contact_group_angle_traces`: CT/FT angle traces by contact group.
- `plot_manual_sc_inverted_km_from_csv`: secondary-contact timing curves from manual SC CSVs.
- `compare_manual_sc_rmst_across_contact_groups`: RMST-style SC timing comparisons.
- `plot_flywise_first_sc_probability_by_contact_group`: flywise first-SC probability.
- `plot_TT_MOC_to_SLC_endpoint_projected_combined`: TT trajectory projection from MOC to SLC endpoint.
- `plot_TT_summary_metrics_vs_LL`: currently focused on L-hTT path efficiency versus landing latency.

The notebook also contains behavior-subset workflows for T2 IT/OT and T1 BO/NB comparisons:

- `plot_it_ot_landing_probability_and_latency`
- `plot_left_TT_path_efficiency_grouped_stripplots`
- `plot_valid_sc_count_vs_landing_latency`

## TT Trajectory Projection Workflow

Main function:

```text
KinematicPlot.PlotCreator.plot_TT_MOC_to_SLC_endpoint_projected_combined
  -> plot_geometry.plot_TT_MOC_to_SLC_endpoint_projected_combined
  -> plot_geometry._collect_TT_MOC_to_SLC_projected_data
```

Core idea:

1. Select trials by group and trial type.
2. Read MOC from metadata.
3. Use the manual SC CSV to find the SLC endpoint when available.
4. Use `origin_keypoint` as the projection origin at MOC.
5. Use `plane_axis` as the vector normal to the projected plane.
6. Project TT trajectories into a 2D basis on that plane.
7. Save or return projected points, trajectories, radial displacement data, stats, and skipped-trial records.

Typical return:

```text
fig, axes, point_df, trajectory_df, radial_df, radial_stats_df, skipped_df
```

Important statistics are in `radial_stats_df`, including vector permutation p-values such as `Vector_Permutation_P`.

## Landing Probability And Latency Workflow

Main file: `plot_landing.py`

Common functions:

- `plot_LP_summary_from_groups`: creates LP summary from group metadata.
- `plot_KM_curve_from_groups`: creates landing latency KM curves.
- `plot_it_ot_landing_probability_and_latency`: combines behavior labels, LP, latency KM, angle traces, angular velocity, and permutation statistics.
- `plot_landing_latency_distribution`: trial-level landing latency distribution.

Stats file: `survival_stats_runner.py`

Common stats methods:

- `compare_lp_unpaired`
- `compare_lp_paired`
- `compare_unpaired_groups`
- `compare_paired_opto`

These methods write CSV summaries and return summary/fly-level tables.

## Secondary Contact Workflow

Main file: `plot_secondary_contact.py`

Manual SC CSVs are read from `SC data/`. These files are used for secondary-contact timing, first-SC probability, and valid SC count summaries.

Common functions:

- `plot_manual_sc_inverted_km_from_csv`
- `compare_manual_sc_rmst_across_contact_groups`
- `plot_flywise_first_sc_probability_by_contact_group`
- `plot_valid_sc_count_vs_landing_latency`

Secondary-contact helper logic also exists in `kinematic_utilities.py`, including `AnalyzeSecondaryContact`, `get_or_run_secondary_contact`, and SC metadata validation helpers.

## Angle Trace Workflow

Main file: `plot_angles.py`

Angle calculation is centralized in:

```text
kinematic_utilities.Calculate_joint_angle
```

QC-aware angle plotting uses:

```text
Calculate_joint_angle(..., apply_tracking_qc=True, return_qc=True)
```

Common functions:

- `plot_wt_contact_group_angle_traces`
- `plot_selected_chrimson_angle_traces`
- `plot_angle_traces_by_trial_sets`

When tracking QC is enabled, angle plots may also save:

- `*_angle_qc_summary.csv`
- `*_angle_qc_skipped_trials.csv`

## Optogenetic Workflow

Notebook: `OptogeneticFigure.ipynb`

Main file: `plot_optogenetics.py`

CsChrimson workflow:

- `get_chrimson_metadata_on_ll_data`: converts metadata MOL frame values into ON-trial landing-latency/event rows.
- `plot_chrimson_LP_metadata`: plots CsChrimson LP based only on metadata MOL absolute frame numbers.
- `plot_kmc_and_unpaired_rmst_perm`: selected-group landing latency KM and RMST comparisons.
- `plot_selected_chrimson_angle_traces`: selected CsChrimson wing/leg angle traces.

Older CsChrimson MOL-detection workflow:

- `plot_chrimson_LP`
- `_detect_chrimson_wing_mol`

This detects MOL from wing-angle traces and can write MOL frames back to metadata. Use this carefully because it may overwrite metadata cell values.

GtACR workflow:

- `plot_LP_summary_light_from_group`
- `plot_gtacr_LP_change_summary`
- `plot_KM_curve_from_groups`
- `stats_runner.compare_lp_paired`

## Tracking QC Workflow

Main files:

- `tracking_qc.py`: central invalid-frame rules, interpolation, summaries, and QC diagnostic plotting.
- `kinematic_utilities.py`: applies QC to angle traces and xyz trajectories.
- `kinematic_object.py`: loads optional score/likelihood-like columns into each `Point`.

Current frame-level invalid rules:

- `x/y/z` missing or non-finite.
- `camera_count` missing.
- `camera_count < min_cameras`, usually 2.
- reprojection error missing.
- reprojection error above threshold, default 50 unless keypoint-specific thresholds are supplied.
- score/likelihood missing or below 0.8 only when score data are available, or when `require_score=True`.

Current window-level rules:

- Invalid gaps up to 5 frames are linearly interpolated.
- Any invalid gap longer than 5 frames fails the window.
- A window fails when valid frame fraction is below 0.7.
- Equivalently, invalid frame fraction must be <= 0.3.

Important distinction:

- Main analysis functions default to `require_score=False`.
- `evaluation.ipynb` currently sets `QC_REQUIRE_SCORE = True`, which is stricter and can mark every frame invalid if 3D score columns are absent.

## Evaluation Notebook Workflow

Notebook: `evaluation.ipynb`

Main functions from `tracking_qc.py`:

- `plot_2d_keypoint_xy_traces`: compares selected keypoint traces across six 2D projection cameras, 3D xyz, reprojection error, and camera count.
- `summarize_tracking_qc_by_trial_keypoint`: one row per trial/keypoint with invalid fractions, missing error fractions, high error fractions, camera-count failures, score failures, longest gaps, and pass/fail.
- `plot_tracking_qc_summary`: plots trial-keypoint distributions and threshold/sensitivity summaries.

The notebook also includes TT QC sensitivity analysis:

```text
loop over max invalid fraction values
  -> call plot_TT_MOC_to_SLC_endpoint_projected_combined
  -> collect radial_stats_df
  -> plot Vector_Permutation_P versus max invalid fraction
```

This is intended to show whether TT trajectory group-comparison significance is robust to stricter or looser QC thresholds.

## Where To Modify Things

Use this rule of thumb:

- Change group paths, fly counts, trial counts, FPS, or metadata paths in `group_config_new.py`.
- Change data model or loading behavior in `kinematic_object.py`.
- Change core geometry/math/contact helpers in `kinematic_utilities.py`.
- Change tracking QC rules in `tracking_qc.py` and the QC application functions in `kinematic_utilities.py`.
- Change WT landing/latency plots in `plot_landing.py`.
- Change angle plots in `plot_angles.py`.
- Change TT trajectory or path-efficiency logic in `plot_geometry.py`.
- Change secondary-contact plots in `plot_secondary_contact.py`.
- Change CsChrimson/GtACR logic in `plot_optogenetics.py`.
- Change notebook calls and parameter choices in the relevant `.ipynb`.

## Suggested Maintenance Rules

Keep notebooks as orchestration files: parameters, group selection, function calls, and display only.

For each major plot, prefer returning:

```text
figure object
axis object
plot/stat dataframe
skipped-trial dataframe, when applicable
```

For future changes, add a short Markdown note in the relevant notebook section describing:

- biological question
- groups used
- metadata source
- analysis window
- QC setting
- output files

When asking for code changes, specify the allowed files, for example:

```text
Only modify plot_geometry.py and WT-LP_and_ConGeo.ipynb.
Do not modify analyze3D_kinematics.py.
```

