"""Configuration helpers shared by tracking-QC-aware plotting workflows."""

from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator
import numpy as np
import pandas as pd


@dataclass(frozen=True)
class TrackingQCConfig:
    """Tracking-QC options without coupling them to a plotting workflow."""

    # Master switch: when False, callers can bypass all QC filtering.
    enabled: bool = False
    # Reprojection-error cutoff used to mark a frame invalid.
    error_max: float = 50
    # Minimum accepted score/likelihood when a score column is available.
    score_min: float = 0.8
    # Minimum number of cameras contributing to the reconstructed 3D point.
    min_cameras: int = 2
    # Maximum invalid gap that can be linearly interpolated, expressed in seconds.
    max_interp_gap_s: float = 0.02
    # Minimum fraction of valid frames required within the analysis window.
    min_valid_fraction: float = 0.7
    # If True, a missing score/likelihood column makes every frame score-invalid.
    require_score: bool = False

    @property
    def max_invalid_fraction(self):
        # Convert the valid-frame threshold into the equivalent invalid-frame limit.
        return 1.0 - float(self.min_valid_fraction)

    def output_metadata(self):
        """Return the common QC fields written to result dataframes."""
        # Store active thresholds beside downstream analysis results for provenance.
        return {
            "Apply_Tracking_QC": bool(self.enabled),
            "Min_Cameras": self.min_cameras if self.enabled else np.nan,
            "Error_Max": self.error_max if self.enabled else np.nan,
            "Score_Min": self.score_min if self.enabled else np.nan,
            "Max_Interp_Gap_s": self.max_interp_gap_s if self.enabled else np.nan,
            "Min_Valid_Fraction": (
                self.min_valid_fraction if self.enabled else np.nan
            ),
            "Max_Invalid_Fraction": (
                self.max_invalid_fraction if self.enabled else np.nan
            ),
            "Require_Score": bool(self.require_score) if self.enabled else False,
        }


def build_config(
        apply_tracking_qc=False,
        min_cameras=2,
        max_interp_gap_s=0.02,
        min_valid_fraction=0.7,
        error_max=50,
        score_min=0.8,
        require_score=False,
):
    """Build a QC config from the current fixed-threshold, time-gap rule."""
    # Keep the public plotting arguments separate from the dataclass constructor.
    return TrackingQCConfig(
        enabled=apply_tracking_qc,
        error_max=error_max,
        score_min=score_min,
        min_cameras=min_cameras,
        max_interp_gap_s=max_interp_gap_s,
        min_valid_fraction=min_valid_fraction,
        require_score=require_score,
    )


def interp_gap_frames_from_fps(max_interp_gap_s, fps):
    """Convert the time-based interpolation threshold into trial-local frames."""
    # A finite positive FPS is required because the interpolation rule is time-based.
    if fps is None or pd.isna(fps) or float(fps) <= 0:
        raise ValueError("fps must be finite and > 0 to resolve time-based QC interpolation.")
    # Round to the nearest whole-frame gap and keep at least one frame interpolable.
    return max(1, int(round(float(max_interp_gap_s) * float(fps))))


def _optional_score_array(point):
    # Accept several common confidence-column names used by tracking pipelines.
    for attr in ("score", "likelihood", "confidence", "probability"):
        # Skip names that are not present on the Point object.
        if hasattr(point, attr):
            values = getattr(point, attr)
            # A present-but-empty score attribute is treated as unavailable.
            if values is not None:
                return np.asarray(values, dtype=float), getattr(point, "score_column", attr)
    # Returning None lets the caller decide whether missing scores should fail QC.
    return None, None


def point_invalid_components(
        point,
        keypoint,
        min_cameras=2,
        error_max=50,
        score_min=0.8,
        require_score=False,
):
    """Return frame-wise invalid components for one 3D keypoint trace."""
    # Convert point attributes to numeric arrays so NaN/finite checks are consistent.
    x = np.asarray(point.x_coord, dtype=float)
    y = np.asarray(point.y_coord, dtype=float)
    z = np.asarray(point.z_coord, dtype=float)
    camera_count = np.asarray(point.camera_count, dtype=float)
    error = np.asarray(point.error, dtype=float)
    # Truncate all channels to the shortest shared length to avoid frame mismatch.
    n_frames = min(len(x), len(y), len(z), len(camera_count), len(error))

    # Apply the shared frame count to every required QC channel.
    x = x[:n_frames]
    y = y[:n_frames]
    z = z[:n_frames]
    camera_count = camera_count[:n_frames]
    error = error[:n_frames]

    # The current QC rule uses one fixed reprojection-error cutoff for every keypoint.
    threshold = float(error_max)
    # Missing xyz coordinates invalidate a frame because downstream geometry needs all axes.
    xyz_missing = ~(np.isfinite(x) & np.isfinite(y) & np.isfinite(z))
    # Missing camera count invalidates a frame because camera support cannot be verified.
    camera_missing = ~np.isfinite(camera_count)
    # Low camera count invalidates a frame even if coordinates are finite.
    low_camera = np.isfinite(camera_count) & (camera_count < min_cameras)
    # Missing reprojection error invalidates a frame because reconstruction quality is unknown.
    error_missing = ~np.isfinite(error)
    # High reprojection error invalidates a frame because triangulation quality is poor.
    error_high = np.isfinite(error) & (error > threshold)

    # Score/likelihood is optional unless the caller explicitly requires it.
    score, score_column = _optional_score_array(point)
    score_column_missing = score is None
    if score is not None:
        # Align score length to the same frame count as xyz/error/camera_count.
        score = score[:n_frames]
        # Missing score values are invalid when a score column exists.
        score_missing = ~np.isfinite(score)
        # Scores below the caller threshold are invalid.
        score_low = np.isfinite(score) & (score < score_min)
    elif require_score:
        # If score is mandatory and absent, mark all frames as score-missing.
        score_missing = np.ones(n_frames, dtype=bool)
        score_low = np.zeros(n_frames, dtype=bool)
    else:
        # If score is optional and absent, do not let it affect QC.
        score_missing = np.zeros(n_frames, dtype=bool)
        score_low = np.zeros(n_frames, dtype=bool)

    # A frame is invalid if any required coordinate, camera, error, or score rule fails.
    invalid = (
        xyz_missing
        | camera_missing
        | low_camera
        | error_missing
        | error_high
        | score_missing
        | score_low
    )
    # Return xyz together so callers can mask/interpolate the same frames.
    xyz = np.column_stack([x, y, z])
    # Keep separate masks so QC summaries can report why frames failed.
    components = {
        "xyz_missing": xyz_missing,
        "camera_missing": camera_missing,
        "low_camera": low_camera,
        "error_missing": error_missing,
        "error_high": error_high,
        "score_missing": score_missing,
        "score_low": score_low,
        "invalid": invalid,
    }
    # Metadata records exactly which thresholds and score column were used.
    metadata = {
        "Error_Threshold": threshold,
        "Score_Column": score_column,
        "Score_Column_Missing": bool(score_column_missing),
        "Min_Cameras": min_cameras,
        "Score_Min": score_min,
        "Require_Score": bool(require_score),
    }
    return xyz, components, metadata


def summarize_invalid_mask(
        invalid_mask,
        components=None,
        start_frame=None,
        end_frame=None,
        max_interp_gap_frames=None,
        max_interp_gap_s=0.02,
        fps=None,
        min_valid_fraction=0.7,
        require_start_end_valid=False,
):
    """Summarize one frame-wise invalid mask using the current QC rule."""
    # Resolve the time-based interpolation rule once so all gap statistics use
    # the same effective frame threshold for this trial.
    if max_interp_gap_frames is None:
        max_interp_gap_frames = interp_gap_frames_from_fps(max_interp_gap_s, fps)
    # Normalize the mask type before slicing and counting.
    invalid_mask = np.asarray(invalid_mask, dtype=bool)
    n_frames = len(invalid_mask)
    # Default to the full trace if no window is supplied.
    if start_frame is None:
        start_frame = 0
    if end_frame is None:
        end_frame = n_frames - 1
    # Clamp requested windows so summaries never index outside the available frames.
    start_frame = max(int(start_frame), 0)
    end_frame = min(int(end_frame), n_frames - 1)
    # Extract the analysis-window mask; use an empty mask for invalid windows.
    window_invalid = (
        invalid_mask[start_frame:end_frame + 1]
        if end_frame >= start_frame and n_frames
        else np.array([], dtype=bool)
    )
    # Valid frames are the complement of the combined invalid mask.
    window_valid = ~window_invalid
    # Consecutive invalid-frame runs drive the interpolation and long-gap checks.
    gap_lengths = _true_run_lengths(window_invalid)
    # Count window size and invalid burden in the selected analysis window.
    total_frames = len(window_invalid)
    invalid_frames = _count_true(window_invalid)
    invalid_fraction = _fraction(invalid_frames, total_frames)
    valid_fraction = _fraction(total_frames - invalid_frames, total_frames)
    # Convert valid-fraction requirement into the maximum tolerated invalid fraction.
    max_invalid_fraction = 1.0 - float(min_valid_fraction)
    # Track the longest invalid run for the long-gap exclusion rule.
    max_gap = int(max(gap_lengths)) if gap_lengths else 0
    # Count invalid runs that exceed the interpolation threshold.
    long_gap_count = int(sum(gap > max_interp_gap_frames for gap in gap_lengths))
    # Count invalid frames that are short enough to be candidates for interpolation.
    interpolatable_count = int(sum(
        gap for gap in gap_lengths if gap <= max_interp_gap_frames
    ))
    # Optionally require the first frame of the analysis window to be valid.
    start_valid = (
        bool(not invalid_mask[start_frame])
        if n_frames and 0 <= start_frame < n_frames
        else False
    )
    # Optionally require the last frame of the analysis window to be valid.
    end_valid = (
        bool(not invalid_mask[end_frame])
        if n_frames and 0 <= end_frame < n_frames
        else False
    )

    # Collect explicit exclusion reasons instead of returning only a boolean.
    exclusion_reasons = []
    if total_frames == 0:
        exclusion_reasons.append("empty_qc_window")
    if pd.isna(invalid_fraction) or invalid_fraction > max_invalid_fraction:
        exclusion_reasons.append("invalid_fraction_above_threshold")
    if max_gap > max_interp_gap_frames:
        exclusion_reasons.append("long_invalid_gap")
    if require_start_end_valid and not start_valid:
        exclusion_reasons.append("start_frame_invalid")
    if require_start_end_valid and not end_valid:
        exclusion_reasons.append("end_frame_invalid")

    # Main summary used by analysis functions and QC diagnostic plots.
    summary = {
        "Valid_Frame_Fraction": valid_fraction,
        "Invalid_Frame_Fraction": invalid_fraction,
        "Invalid_Frame_Count": invalid_frames,
        "Max_Invalid_Gap_Frames": max_gap,
        "Long_Gap_Count": long_gap_count,
        "Interpolated_Frame_Count": interpolatable_count,
        "Interpolatable_Invalid_Frame_Count": interpolatable_count,
        "Interpolatable_Invalid_Fraction": _fraction(interpolatable_count, total_frames),
        "Start_Frame_Valid": start_valid,
        "End_Frame_Valid": end_valid,
        "Max_Interp_Gap_s": max_interp_gap_s,
        "Max_Interp_Gap_Frames": max_interp_gap_frames,
        "Min_Valid_Fraction": min_valid_fraction,
        "Max_Invalid_Fraction": max_invalid_fraction,
        "QC_Passed": len(exclusion_reasons) == 0,
        "QC_Exclusion_Reason": ";".join(exclusion_reasons),
    }

    if components is not None:
        # Add reason-specific counts/fractions for every provided component mask.
        for name, mask in components.items():
            # The combined invalid mask is already represented by the main summary.
            if name == "invalid":
                continue
            # Align each component to the same window used for the combined mask.
            mask = np.asarray(mask, dtype=bool)
            window = (
                mask[start_frame:end_frame + 1]
                if end_frame >= start_frame and len(mask)
                else np.array([], dtype=bool)
            )
            # Convert snake_case component names into compact dataframe column prefixes.
            column = "".join(part.capitalize() for part in name.split("_"))
            summary[f"{column}_Frame_Count"] = _count_true(window)
            summary[f"{column}_Fraction"] = _fraction(_count_true(window), total_frames)

    return summary


def interpolate_invalid_xyz_gaps(xyz, invalid_mask, max_gap_frames=5):
    """Set invalid xyz frames to NaN and linearly interpolate short invalid runs."""
    # Work on a copy so callers keep access to the raw coordinates.
    xyz = np.asarray(xyz, dtype=float).copy()
    # Normalize the invalid mask before using it to overwrite coordinates.
    invalid_mask = np.asarray(invalid_mask, dtype=bool)
    # Invalid frames are removed first; only eligible short gaps are filled below.
    xyz[invalid_mask] = np.nan
    interpolated_total = 0
    n_frames = len(xyz)
    # Frame numbers are the x-axis for interpolation.
    x_index = np.arange(n_frames)
    runs = []
    i = 0
    # Identify contiguous invalid-frame runs as half-open [start, stop) intervals.
    while i < n_frames:
        if not invalid_mask[i]:
            i += 1
            continue
        start = i
        while i < n_frames and invalid_mask[i]:
            i += 1
        runs.append((start, i))

    # Only fill invalid runs that are short and bounded by valid finite endpoints.
    for start, stop in runs:
        gap_len = stop - start
        left = start - 1
        right = stop
        # Do not interpolate long gaps or edge gaps without both neighboring frames.
        if gap_len > max_gap_frames or left < 0 or right >= n_frames:
            continue
        # All xyz dimensions must be finite at both endpoints for interpolation.
        if not np.all(np.isfinite(xyz[left])) or not np.all(np.isfinite(xyz[right])):
            continue
        # Interpolate each coordinate dimension independently across the gap.
        for dim in range(xyz.shape[1]):
            xyz[start:stop, dim] = np.interp(
                x_index[start:stop],
                [left, right],
                [xyz[left, dim], xyz[right, dim]],
            )
        # Report how many invalid frames were actually filled.
        interpolated_total += gap_len
    return xyz, interpolated_total


def interpolate_invalid_trace_gaps(values, invalid_mask, max_gap_frames=5):
    """Set invalid scalar frames to NaN and linearly interpolate short invalid runs."""
    # Work on a numeric copy so the original scalar trace is unchanged.
    values = np.asarray(values, dtype=float).copy()
    # Normalize the invalid mask before applying it to the trace.
    invalid_mask = np.asarray(invalid_mask, dtype=bool)
    # Invalid scalar values are blanked before short-gap interpolation.
    values[invalid_mask] = np.nan
    interpolated_total = 0
    n_frames = len(values)
    # Frame numbers provide the interpolation x-axis.
    x_index = np.arange(n_frames)
    i = 0
    # Scan contiguous invalid runs and handle each run immediately.
    while i < n_frames:
        if not invalid_mask[i]:
            i += 1
            continue
        start = i
        while i < n_frames and invalid_mask[i]:
            i += 1
        stop = i
        gap_len = stop - start
        left = start - 1
        right = stop
        # Fill only short internal gaps with finite values on both sides.
        if (
                gap_len <= max_gap_frames
                and left >= 0
                and right < n_frames
                and np.isfinite(values[left])
                and np.isfinite(values[right])
        ):
            # Linear interpolation preserves the trace length and frame alignment.
            values[start:stop] = np.interp(
                x_index[start:stop],
                [left, right],
                [values[left], values[right]],
            )
            interpolated_total += gap_len
    return values, interpolated_total


def _resolve_group(group_name):
    # Import lazily so this helper can be used without loading group config at module import time.
    from group_config_new import GROUP_INFO, build_one_group

    # Prefer exact config keys because notebook calls often use those identifiers.
    if group_name in GROUP_INFO:
        return build_one_group(group_name)

    # Fall back to matching the displayed group name stored inside the config.
    matches = [
        key for key, info in GROUP_INFO.items()
        if info.get("group_name") == group_name
    ]
    # A single display-name match is unambiguous and can be built safely.
    if len(matches) == 1:
        return build_one_group(matches[0])
    # Multiple matches would make the requested group ambiguous.
    if len(matches) > 1:
        raise ValueError(
            f"Group name '{group_name}' matches multiple config keys: {matches}"
        )
    # No key or display-name match means the notebook argument is invalid.
    raise KeyError(f"Could not find group '{group_name}' in group_config_new.GROUP_INFO.")


def _read_trial_timing(group_info, fly, trial):
    # Metadata tables use one row per fly and one column per trial, both 1-based externally.
    fly_idx = int(fly) - 1
    trial_idx = int(trial) - 1

    # MOC and MOL are both required to define the analysis window.
    if group_info.moc_data is None:
        raise ValueError(f"MOC metadata is not available for {group_info.group_name}.")
    if group_info.mol_data is None:
        raise ValueError(f"MOL metadata is not available for {group_info.group_name}.")
    # Validate converted indices before indexing Excel-derived metadata tables.
    if fly_idx < 0 or fly_idx >= group_info.total_fly_number:
        raise IndexError(f"Fly {fly} is outside the configured fly range.")
    if trial_idx < 0 or trial_idx >= group_info.trial_num:
        raise IndexError(f"Trial {trial} is outside the configured trial range.")

    # Pull event frames and FPS for the requested trial/fly pair.
    moc = group_info.moc_data.iloc[fly_idx, trial_idx]
    mol = group_info.mol_data.iloc[fly_idx, trial_idx]
    fps = group_info.fps[fly_idx]

    # Missing event timing prevents a valid MOC-to-MOL window.
    if pd.isna(moc) or pd.isna(mol):
        raise ValueError(
            f"Missing MOC/MOL metadata for {group_info.group_name} F{fly}T{trial}."
        )
    # MOL before MOC is biologically and analytically invalid for this window.
    if mol < moc:
        raise ValueError(
            f"MOL ({mol}) is earlier than MOC ({moc}) for "
            f"{group_info.group_name} F{fly}T{trial}."
        )

    # Return integer frame indices plus numeric FPS for downstream slicing.
    return int(round(moc)), int(round(mol)), float(fps)


def _read_2d_projection_h5(h5_path):
    # Normalize user-provided paths before checking the projection file.
    h5_path = Path(h5_path)
    if not h5_path.exists():
        raise FileNotFoundError(f"2D projection H5 file does not exist: {h5_path}")

    # Anipose/DLC files commonly store projected data under /df_with_missing.
    with pd.HDFStore(h5_path, mode="r") as store:
        keys = store.keys()
        if "/df_with_missing" in keys:
            df = store["/df_with_missing"]
        elif len(keys) == 1:
            # If there is only one dataset, use it rather than requiring a fixed key.
            df = store[keys[0]]
        else:
            raise ValueError(
                f"Could not choose an H5 dataset from {h5_path}. Available keys: {keys}"
            )

    # Projection files should use DLC-style hierarchical columns.
    if not isinstance(df.columns, pd.MultiIndex):
        raise ValueError("Expected DeepLabCut-style MultiIndex columns in the H5 file.")
    # The function needs bodypart and coordinate levels to extract keypoint traces.
    if "bodyparts" not in df.columns.names or "coords" not in df.columns.names:
        raise ValueError(
            "Expected H5 columns with MultiIndex levels named 'bodyparts' and 'coords'."
        )

    return df


def _find_projection_h5_files(projection_path, fly, trial):
    # Accept either the projection root or one of the common per-fly subfolders.
    projection_path = Path(projection_path)
    if not projection_path.exists():
        raise FileNotFoundError(f"2D projection path does not exist: {projection_path}")

    # File names use zero-padded fly/trial tags.
    fly_tag = f"F{int(fly):03d}"
    trial_tag = f"T{int(trial):03d}"
    # Search most-specific locations first to avoid scanning unrelated folders.
    search_roots = [
        projection_path / fly_tag / "pose-2d-proj",
        projection_path / fly_tag,
        projection_path,
    ]

    camera_paths = {}
    for root in search_roots:
        # Skip candidate roots that do not exist for this dataset layout.
        if not root.exists():
            continue
        # Collect one H5 projection file per camera for the requested trial.
        for h5_path in root.rglob(f"{fly_tag}_{trial_tag}_*Cam*.h5"):
            stem = h5_path.stem
            cam_text = stem.rsplit("Cam", 1)[-1]
            # Ignore files where the camera suffix is not a plain integer.
            if not cam_text.isdigit():
                continue
            camera_paths[int(cam_text)] = h5_path
        # Stop after the first root that contains matching camera files.
        if camera_paths:
            break

    # Evaluation plots expect all six camera projections to be present.
    expected = set(range(1, 7))
    missing = sorted(expected - set(camera_paths))
    if missing:
        raise FileNotFoundError(
            f"Missing projection H5 files for cameras {missing} under {projection_path}."
        )
    # Return paths in camera-number order for stable plotting colors.
    return {camera: camera_paths[camera] for camera in sorted(expected)}


def _read_3d_trial_dataframe(group_info, fly, trial):
    # Convert fly/trial into the same trial key used by the Group object.
    key = group_info._trial_key(int(fly), int(trial))
    # Kinematic CSV paths are recorded during group initialization.
    data_paths = getattr(group_info, "fly_kinematic_data_path", {})
    if key not in data_paths:
        raise FileNotFoundError(
            f"No 3D kinematic CSV path configured for {group_info.group_name} F{fly}T{trial}."
        )

    # Read only the requested trial CSV for evaluation plotting and QC summaries.
    csv_path = Path(data_paths[key])
    if not csv_path.exists():
        raise FileNotFoundError(f"3D kinematic CSV file does not exist: {csv_path}")
    return pd.read_csv(csv_path), csv_path


def _slice_series(series, start_frame, stop_frame, value_mode):
    # Convert the trace to numeric values before optional differencing.
    values = series.astype(float)
    if value_mode == "change":
        # Change mode displays frame-to-frame jumps rather than absolute position.
        values = values.diff()
    # Slice after differencing so the x-axis still reports absolute frame numbers.
    return values.iloc[start_frame:stop_frame + 1]


def _extract_2d_coord(df, keypoint, coord):
    try:
        # Select the requested bodypart/coordinate from the DLC MultiIndex columns.
        return df.xs((keypoint, coord), level=("bodyparts", "coords"), axis=1).iloc[:, 0]
    except KeyError as exc:
        raise KeyError(
            f"Could not find coordinate '{coord}' for keypoint '{keypoint}' in projection H5."
        ) from exc


def _extract_2d_coord_optional(df, keypoint, coord, start_frame, stop_frame):
    try:
        # Use the normal extractor when the coordinate exists.
        values = _extract_2d_coord(df, keypoint, coord)
    except KeyError:
        # Optional channels, such as likelihood, become NaN traces if absent.
        return pd.Series(
            np.nan,
            index=np.arange(start_frame, stop_frame + 1),
            dtype=float,
        )
    # Return the same frame window as the required coordinate channels.
    return values.astype(float).iloc[start_frame:stop_frame + 1]


def _resolve_axis_ylim(axis_ylim, values=None, symmetric=False):
    # None means automatic matplotlib scaling unless symmetric limits are requested.
    if axis_ylim is None:
        if values is None or not symmetric:
            return None
        # Symmetric autoscaling is useful for change traces centered around zero.
        finite_values = np.asarray(values, dtype=float)
        finite_values = finite_values[np.isfinite(finite_values)]
        if len(finite_values) == 0:
            return None
        max_abs = float(np.nanmax(np.abs(finite_values)))
        # Avoid a zero-height axis when all values are exactly zero.
        if max_abs == 0:
            max_abs = 1.0
        return -max_abs, max_abs
    # A scalar user limit is interpreted as +/- that absolute value.
    if np.isscalar(axis_ylim):
        limit = abs(float(axis_ylim))
        return -limit, limit
    # A two-value iterable is treated as explicit lower/upper limits.
    if len(axis_ylim) != 2:
        raise ValueError("Axis limits must be None, a number, or a 2-value tuple.")
    return tuple(axis_ylim)


def _initialize_standard_metadata(group_info):
    # Reuse existing parsed metadata when a group has already been initialized.
    if getattr(group_info, "trial_metadata", None):
        return
    # QC summaries need metadata and kinematic paths, but not preloaded kinematic arrays.
    group_info.initialize_manual_data(require_kinematics=False)


def _get_analysis_window(group_info, meta, n_frames, margin_s, window_mode):
    # Only two window modes are supported by the evaluation/QC plotting workflow.
    if window_mode not in {"moc_mol", "full"}:
        raise ValueError("window_mode must be 'moc_mol' or 'full'.")
    if n_frames <= 0:
        raise ValueError("Trial dataframe has no frames.")

    # Metadata values define the selected fly/trial and its native sampling rate.
    fly = int(meta["Fly#"])
    trial = int(meta["Trial#"])
    fps = float(meta["fps"])

    # Full-trial mode ignores MOC/MOL and uses the entire available trace.
    if window_mode == "full":
        return 0, n_frames - 1, np.nan, np.nan, fps

    # MOC-to-MOL mode reads event frames and expands them by the requested margin.
    moc, mol, fps = _read_trial_timing(group_info, fly, trial)
    start_frame = max(0, int(round(moc - margin_s * fps)))
    stop_frame = min(n_frames - 1, int(round(mol + margin_s * fps)))
    # A negative window length indicates inconsistent timing or trace length.
    if stop_frame < start_frame:
        raise ValueError(
            f"Invalid analysis window for {group_info.group_name} F{fly}T{trial}: "
            f"start={start_frame}, stop={stop_frame}."
        )
    return start_frame, stop_frame, moc, mol, fps


def _find_score_column(df, keypoint):
    # Try the common confidence-column suffixes used by different tracking exports.
    candidates = [
        f"{keypoint}_score",
        f"{keypoint}_likelihood",
        f"{keypoint}_confidence",
        f"{keypoint}_probability",
    ]
    for column in candidates:
        # Return the first matching score column for this keypoint.
        if column in df.columns:
            return column
    # A missing score column may or may not fail QC, depending on require_score.
    return None


def _count_true(values):
    # Centralize boolean counting so masks are converted consistently.
    values = np.asarray(values, dtype=bool)
    return int(np.count_nonzero(values))


def _fraction(count, total):
    # Empty windows do not have a meaningful fraction.
    if total == 0:
        return np.nan
    return float(count) / float(total)


def _true_run_lengths(values):
    # Convert any boolean-like sequence into a clean mask.
    values = np.asarray(values, dtype=bool)
    lengths = []
    run_length = 0
    # Walk the mask and count consecutive True runs.
    for value in values:
        if value:
            run_length += 1
        elif run_length:
            # A False value terminates the current invalid run.
            lengths.append(run_length)
            run_length = 0
    # Preserve a run that reaches the last frame.
    if run_length:
        lengths.append(run_length)
    return lengths


def _summarize_one_trial_keypoint(
        group_info,
        meta,
        kine_df,
        keypoint,
        margin_s,
        window_mode,
        error_max,
        score_min,
        min_cameras,
        max_interp_gap_s,
        max_invalid_fraction,
        require_score,
):
    # Extract the 1-based fly/trial identifiers used by metadata and filenames.
    fly = int(meta["Fly#"])
    trial = int(meta["Trial#"])
    # Build the canonical trial key for reporting and skip records.
    trial_key = group_info._trial_key(fly, trial)
    # Resolve the requested analysis window before calculating QC fractions.
    start_frame, stop_frame, moc, mol, fps = _get_analysis_window(
        group_info,
        meta,
        len(kine_df),
        margin_s,
        window_mode,
    )
    # Gap interpolation is specified in seconds and resolved to frames using
    # the native FPS for this trial.
    max_interp_gap_frames = interp_gap_frames_from_fps(max_interp_gap_s, fps)
    # Slice the kinematic dataframe once so every metric uses the same frames.
    window = kine_df.iloc[start_frame:stop_frame + 1]
    total_frames = len(window)

    # Each keypoint summary requires x/y/z columns in the 3D CSV.
    coord_columns = [f"{keypoint}_{axis}" for axis in ("x", "y", "z")]
    # Fail explicitly if the requested keypoint is absent from the CSV.
    missing_coord_columns = [column for column in coord_columns if column not in kine_df.columns]
    if missing_coord_columns:
        raise KeyError(
            f"3D coordinate columns not found for {group_info.group_name} {trial_key} "
            f"{keypoint}: {missing_coord_columns}"
        )

    # Missing in any xyz dimension is enough to make the frame coordinate-invalid.
    xyz = window[coord_columns].astype(float)
    xyz_missing = xyz.isna().any(axis=1).to_numpy()

    # Reprojection error is required by the current QC rule.
    error_column = f"{keypoint}_error"
    if error_column in kine_df.columns:
        # Extract frame-wise error and split missing-error from high-error failures.
        error_values = window[error_column].astype(float)
        error_missing = error_values.isna().to_numpy()
        error_high = (error_values > error_max).fillna(False).to_numpy()
        # Keep descriptive error values for diagnostic plots.
        finite_error_values = error_values[np.isfinite(error_values)]
        median_error = float(finite_error_values.median()) if len(finite_error_values) else np.nan
        p95_error = float(finite_error_values.quantile(0.95)) if len(finite_error_values) else np.nan
    else:
        # If the error column is absent, all frames fail the missing-error rule.
        error_missing = np.ones(total_frames, dtype=bool)
        error_high = np.zeros(total_frames, dtype=bool)
        median_error = np.nan
        p95_error = np.nan

    # Camera count is required to verify that enough views supported the 3D point.
    ncams_column = f"{keypoint}_ncams"
    if ncams_column in kine_df.columns:
        # Split absent camera-count values from low-camera-count values.
        ncams = window[ncams_column].astype(float)
        camera_missing = ncams.isna().to_numpy()
        low_camera = (ncams < min_cameras).fillna(False).to_numpy()
        # Keep descriptive camera-count values for diagnostics.
        finite_ncams = ncams[np.isfinite(ncams)]
        median_ncams = float(finite_ncams.median()) if len(finite_ncams) else np.nan
        min_ncams_value = float(finite_ncams.min()) if len(finite_ncams) else np.nan
    else:
        # If the camera-count column is absent, every frame fails camera QC.
        camera_missing = np.ones(total_frames, dtype=bool)
        low_camera = np.ones(total_frames, dtype=bool)
        median_ncams = np.nan
        min_ncams_value = np.nan

    # Score/likelihood is optional unless require_score=True.
    score_column = _find_score_column(kine_df, keypoint)
    score_column_missing = score_column is None
    if score_column is not None:
        # Score failures are tracked separately from reprojection and camera failures.
        score_values = window[score_column].astype(float)
        score_missing = score_values.isna().to_numpy()
        score_low = (score_values < score_min).fillna(False).to_numpy()
        # Keep descriptive score values for diagnostic plots.
        finite_score_values = score_values[np.isfinite(score_values)]
        median_score = float(finite_score_values.median()) if len(finite_score_values) else np.nan
        p05_score = float(finite_score_values.quantile(0.05)) if len(finite_score_values) else np.nan
    elif require_score:
        # If score is mandatory and absent, all frames fail score QC.
        score_missing = np.ones(total_frames, dtype=bool)
        score_low = np.zeros(total_frames, dtype=bool)
        median_score = np.nan
        p05_score = np.nan
    else:
        # If score is optional and absent, score does not affect invalid-frame calls.
        score_missing = np.zeros(total_frames, dtype=bool)
        score_low = np.zeros(total_frames, dtype=bool)
        median_score = np.nan
        p05_score = np.nan

    # Combine sub-rules into category-level masks.
    error_invalid = error_missing | error_high
    camera_invalid = camera_missing | low_camera
    score_invalid = score_missing | score_low
    # The final invalid mask is the union of all active QC failure modes.
    invalid = xyz_missing | error_invalid | camera_invalid | score_invalid

    # Consecutive invalid-run lengths determine short-gap interpolation eligibility.
    invalid_lengths = _true_run_lengths(invalid)
    # Count invalid frames in runs short enough to be interpolated.
    interpolatable_frames = sum(
        length for length in invalid_lengths
        if length <= max_interp_gap_frames
    )
    # Count invalid runs that are too long to be safely interpolated.
    long_gap_count = sum(
        1 for length in invalid_lengths
        if length > max_interp_gap_frames
    )
    # Store the worst gap length for threshold sensitivity and reporting.
    longest_invalid_gap = max(invalid_lengths, default=0)

    # Calculate the overall invalid burden for this trial-keypoint window.
    invalid_frames = _count_true(invalid)
    invalid_fraction = _fraction(invalid_frames, total_frames)
    # The trial-keypoint passes only if total invalid burden is below threshold.
    pass_invalid_fraction = invalid_fraction <= max_invalid_fraction
    # The trial-keypoint also fails if any invalid run is longer than the interpolation threshold.
    pass_long_gap = longest_invalid_gap <= max_interp_gap_frames

    # Count each component separately so the dominant failure mode is visible.
    reason_counts = {
        "XYZ_Missing": _count_true(xyz_missing),
        "Error_Missing": _count_true(error_missing),
        "Error_High": _count_true(error_high),
        "Camera_Missing": _count_true(camera_missing),
        "Low_Camera_Count": _count_true(low_camera),
        "Score_Missing": _count_true(score_missing),
        "Score_Low": _count_true(score_low),
    }
    primary_failure_reason = "None"
    if invalid_frames:
        # The primary reason is descriptive only; it does not change pass/fail logic.
        primary_failure_reason = max(reason_counts, key=reason_counts.get)

    # Return one row suitable for concatenation into a trial-keypoint summary dataframe.
    return {
        "Group": group_info.group_name,
        "Fly": fly,
        "Trial": trial,
        "Trial_Key": trial_key,
        "Trial_Type": meta.get("TrialType"),
        "Keypoint": keypoint,
        "Window_Mode": window_mode,
        "Start_Frame": start_frame,
        "Stop_Frame": stop_frame,
        "MOC_Frame": moc,
        "MOL_Frame": mol,
        "FPS": fps,
        "Total_Frames": total_frames,
        "QC_Error_Max": error_max,
        "QC_Score_Min": score_min,
        "QC_Min_Cameras": min_cameras,
        "QC_Max_Interp_Gap_s": max_interp_gap_s,
        "QC_Max_Interp_Gap_Frames": max_interp_gap_frames,
        "QC_Max_Invalid_Fraction": max_invalid_fraction,
        "QC_Require_Score": bool(require_score),
        "Score_Column": score_column,
        "Score_Column_Missing": bool(score_column_missing),
        "Error_Median": median_error,
        "Error_P95": p95_error,
        "Score_Median": median_score,
        "Score_P05": p05_score,
        "Ncams_Median": median_ncams,
        "Ncams_Min": min_ncams_value,
        "XYZ_Missing_Frames": reason_counts["XYZ_Missing"],
        "Error_Missing_Frames": reason_counts["Error_Missing"],
        "Error_High_Frames": reason_counts["Error_High"],
        "Camera_Missing_Frames": reason_counts["Camera_Missing"],
        "Low_Camera_Count_Frames": reason_counts["Low_Camera_Count"],
        "Score_Missing_Frames": reason_counts["Score_Missing"],
        "Score_Low_Frames": reason_counts["Score_Low"],
        "Error_Invalid_Frames": _count_true(error_invalid),
        "Camera_Invalid_Frames": _count_true(camera_invalid),
        "Score_Invalid_Frames": _count_true(score_invalid),
        "Invalid_Frames": invalid_frames,
        "XYZ_Missing_Fraction": _fraction(reason_counts["XYZ_Missing"], total_frames),
        "Error_Missing_Fraction": _fraction(reason_counts["Error_Missing"], total_frames),
        "Error_High_Fraction": _fraction(reason_counts["Error_High"], total_frames),
        "Error_Invalid_Fraction": _fraction(_count_true(error_invalid), total_frames),
        "Camera_Missing_Fraction": _fraction(reason_counts["Camera_Missing"], total_frames),
        "Low_Camera_Count_Fraction": _fraction(reason_counts["Low_Camera_Count"], total_frames),
        "Camera_Invalid_Fraction": _fraction(_count_true(camera_invalid), total_frames),
        "Score_Missing_Fraction": _fraction(reason_counts["Score_Missing"], total_frames),
        "Score_Low_Fraction": _fraction(reason_counts["Score_Low"], total_frames),
        "Score_Invalid_Fraction": _fraction(_count_true(score_invalid), total_frames),
        "Invalid_Fraction": invalid_fraction,
        "Interpolatable_Invalid_Frames": interpolatable_frames,
        "Interpolatable_Invalid_Fraction": _fraction(interpolatable_frames, total_frames),
        "Longest_Invalid_Gap_Frames": longest_invalid_gap,
        "Long_Invalid_Gap_Count": long_gap_count,
        "Pass_Invalid_Fraction": bool(pass_invalid_fraction),
        "Pass_Long_Gap": bool(pass_long_gap),
        "QC_Passed": bool(pass_invalid_fraction and pass_long_gap),
        "Primary_Failure_Reason": primary_failure_reason,
    }


def summarize_tracking_qc_by_trial_keypoint(
        group_name,
        keypoints=None,
        trial_types=("Landing",),
        margin_s=0.2,
        window_mode="moc_mol",
        error_max=50,
        score_min=0.8,
        min_cameras=2,
        max_interp_gap_s=0.02,
        max_invalid_fraction=0.3,
        require_score=False,
        include_good_fly_only=True,
):
    """
    Build a trial-keypoint-level tracking QC summary table for one group.

    Each output row is one fly x trial x keypoint trajectory within the selected
    analysis window. Missing reprojection error, high reprojection error, low
    camera count, missing coordinates, and low/missing score are tracked as
    separate fractions, then combined into the overall invalid-frame fraction.
    """
    # Build the group and ensure metadata paths/timing are initialized.
    group_info = _resolve_group(group_name)
    _initialize_standard_metadata(group_info)

    # Default to every configured joint if the caller does not provide keypoints.
    if keypoints is None:
        keypoints = list(group_info.joints)
    # Accept either a single trial type string or a collection of trial types.
    if isinstance(trial_types, str):
        trial_types = [trial_types]
    # Use a set for fast trial-type filtering.
    trial_types = set(trial_types)

    rows = []
    skipped = []
    # Iterate over initialized trial metadata instead of scanning data folders.
    for trial_key, meta in group_info.trial_metadata.items():
        # Restrict the summary to selected trial types, such as Landing.
        if trial_types and meta.get("TrialType") not in trial_types:
            continue
        # Optionally keep only flies marked as good in the group configuration.
        if include_good_fly_only and int(meta["Fly#"]) not in group_info.good_fly_index:
            continue
        try:
            # Read the matching 3D CSV for this one trial.
            kine_df, csv_path = _read_3d_trial_dataframe(
                group_info,
                int(meta["Fly#"]),
                int(meta["Trial#"]),
            )
            # Produce one QC row per requested keypoint within this trial.
            for keypoint in keypoints:
                row = _summarize_one_trial_keypoint(
                    group_info=group_info,
                    meta=meta,
                    kine_df=kine_df,
                    keypoint=keypoint,
                    margin_s=margin_s,
                    window_mode=window_mode,
                    error_max=error_max,
                    score_min=score_min,
                    min_cameras=min_cameras,
                    max_interp_gap_s=max_interp_gap_s,
                    max_invalid_fraction=max_invalid_fraction,
                    require_score=require_score,
                )
                # Store the source CSV path for traceability during manual inspection.
                row["Kinematic_CSV"] = str(csv_path)
                rows.append(row)
        except Exception as exc:
            # Keep skipped-trial information in dataframe attrs instead of aborting the batch.
            skipped.append({
                "Group": group_info.group_name,
                "Trial_Key": trial_key,
                "Fly": meta.get("Fly#"),
                "Trial": meta.get("Trial#"),
                "Trial_Type": meta.get("TrialType"),
                "Skip_Reason": f"{type(exc).__name__}: {exc}",
            })

    # Convert accumulated rows to the public summary dataframe.
    summary_df = pd.DataFrame(rows)
    if skipped:
        # Attach skipped trials as metadata so plotting can proceed with valid rows.
        skipped_df = pd.DataFrame(skipped)
        summary_df.attrs["skipped_trials"] = skipped_df
    return summary_df


def _stripplot_by_keypoint(
        ax,
        df,
        y_column,
        keypoint_order,
        ylabel,
        threshold=None,
        color="0.25",
):
    # Assign one x-position to each keypoint so all trial-level dots align by bodypart.
    positions = np.arange(len(keypoint_order))
    # Fixed RNG makes the jitter reproducible across repeated plotting calls.
    rng = np.random.default_rng(0)
    for pos, keypoint in zip(positions, keypoint_order):
        # Every plotted value is one trial-keypoint row from the QC summary table.
        values = df.loc[df["Keypoint"] == keypoint, y_column].dropna().to_numpy(dtype=float)
        if len(values) == 0:
            continue
        # Jitter separates overlapping trial points without changing their y-values.
        jitter = rng.uniform(-0.18, 0.18, size=len(values))
        ax.scatter(
            np.full(len(values), pos) + jitter,
            values,
            s=22,
            alpha=0.7,
            color=color,
            edgecolors="none",
        )
        # The black horizontal mark shows the median for that keypoint.
        median = float(np.nanmedian(values))
        ax.plot([pos - 0.25, pos + 0.25], [median, median], color="black", linewidth=1.5)
    if threshold is not None:
        # Red dashed line marks the user-selected pass/fail threshold when relevant.
        ax.axhline(threshold, color="red", linestyle="--", linewidth=1)
    # Label keypoints on the x-axis and keep a light y-grid for distribution reading.
    ax.set_xticks(positions)
    ax.set_xticklabels(keypoint_order, rotation=45, ha="right")
    ax.set_ylabel(ylabel)
    ax.grid(True, axis="y", alpha=0.25)


def _save_figure(fig, output_dir, filename):
    # Saving is optional because notebook users may only want inline figures.
    if output_dir is None:
        return
    # Create the target directory lazily only when saving is requested.
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_dir / filename, dpi=200, bbox_inches="tight")


def plot_tracking_qc_summary(
        qc_summary_df,
        output_dir=None,
        show=True,
        keypoint_order=None,
        max_invalid_fraction=None,
        max_interp_gap_s=None,
        threshold_values=None,
        figsize=(11, 5),
):
    """
    Plot trial-keypoint-level QC distributions from `summarize_tracking_qc_by_trial_keypoint`.

    Returns a dictionary of matplotlib figures. Every dot in the distribution
    plots is one fly x trial x keypoint row from the summary table.
    """
    # Refuse to plot empty input because threshold defaults require dataframe values.
    if qc_summary_df.empty:
        raise ValueError("qc_summary_df is empty; no QC summary plots can be made.")
    # Preserve first-seen keypoint order unless the caller specifies a custom order.
    if keypoint_order is None:
        keypoint_order = list(dict.fromkeys(qc_summary_df["Keypoint"]))
    # Default the displayed invalid-fraction threshold from the summary table.
    if max_invalid_fraction is None:
        max_invalid_fraction = float(qc_summary_df["QC_Max_Invalid_Fraction"].dropna().iloc[0])
    # The summary figure reports the interpolation setting in seconds, but the
    # longest-gap panel uses each trial's effective frame cutoff.
    if max_interp_gap_s is None:
        max_interp_gap_s = float(qc_summary_df["QC_Max_Interp_Gap_s"].dropna().iloc[0])
    # Use the median effective frame cutoff when trials have mixed FPS.
    max_interp_gap_frames = int(qc_summary_df["QC_Max_Interp_Gap_Frames"].dropna().median())
    # Default sensitivity scan asks how many trial-keypoint rows survive each cutoff.
    if threshold_values is None:
        threshold_values = np.arange(0.05, 0.51, 0.05)

    # Accumulate figures by semantic name for notebook reuse.
    figures = {}
    # Use the first non-missing group name as the figure title prefix.
    group_name = str(qc_summary_df["Group"].dropna().iloc[0])

    # Define the trial-keypoint distributions produced by the summary figure suite.
    distribution_specs = [
        (
            "invalid_fraction",
            "Invalid_Fraction",
            "Invalid frame fraction",
            max_invalid_fraction,
            "#4c78a8",
        ),
        (
            "missing_reprojection_error_fraction",
            "Error_Missing_Fraction",
            "Missing reprojection error fraction",
            None,
            "#8c564b",
        ),
        (
            "high_reprojection_error_fraction",
            "Error_High_Fraction",
            "High reprojection error fraction",
            None,
            "#ff7f0e",
        ),
        (
            "low_camera_count_fraction",
            "Low_Camera_Count_Fraction",
            "Low camera-count fraction",
            None,
            "#7f7f7f",
        ),
        (
            "score_invalid_fraction",
            "Score_Invalid_Fraction",
            "Score invalid fraction",
            None,
            "#2ca02c",
        ),
        (
            "interpolatable_invalid_fraction",
            "Interpolatable_Invalid_Fraction",
            "Interpolatable invalid fraction",
            None,
            "#9467bd",
        ),
        (
            "longest_invalid_gap",
            "Longest_Invalid_Gap_Frames",
            "Longest invalid gap (frames)",
            max_interp_gap_frames,
            "#d62728",
        ),
        (
            "reprojection_error_p95",
            "Error_P95",
            "95th percentile reprojection error",
            None,
            "#ff7f0e",
        ),
    ]

    for fig_key, column, ylabel, threshold, color in distribution_specs:
        # Skip plots for metrics that are not present in the supplied dataframe.
        if column not in qc_summary_df.columns:
            continue
        # Skip plots with no numeric data after NaN removal.
        if qc_summary_df[column].dropna().empty:
            continue
        # Avoid an uninformative score plot when no score column exists and score is optional.
        if (
                fig_key == "score_invalid_fraction"
                and "Score_Column_Missing" in qc_summary_df.columns
                and qc_summary_df["Score_Column_Missing"].all()
                and qc_summary_df[column].fillna(0).eq(0).all()
        ):
            continue
        # Build one stripplot figure for the selected QC metric.
        fig, ax = plt.subplots(figsize=figsize)
        _stripplot_by_keypoint(
            ax,
            qc_summary_df,
            column,
            keypoint_order,
            ylabel,
            threshold=threshold,
            color=color,
        )
        # Title, layout, optional save, and optional close are handled uniformly.
        ax.set_title(f"{group_name}: {ylabel}")
        fig.tight_layout()
        _save_figure(fig, output_dir, f"{group_name}_{fig_key}.png")
        if not show:
            plt.close(fig)
        figures[fig_key] = fig

    # Plot the score distribution only when score/likelihood information exists.
    if "Score_Column_Missing" in qc_summary_df.columns and not qc_summary_df["Score_Column_Missing"].all():
        fig, ax = plt.subplots(figsize=figsize)
        _stripplot_by_keypoint(
            ax,
            qc_summary_df,
            "Score_P05",
            keypoint_order,
            "5th percentile score",
            threshold=float(qc_summary_df["QC_Score_Min"].dropna().iloc[0]),
            color="#2ca02c",
        )
        ax.set_title(f"{group_name}: Score distribution")
        fig.tight_layout()
        _save_figure(fig, output_dir, f"{group_name}_score_p05.png")
        if not show:
            plt.close(fig)
        figures["score_p05"] = fig

    # Pivot trial-keypoint invalid fractions into a heatmap matrix.
    pivot = qc_summary_df.pivot_table(
        index="Trial_Key",
        columns="Keypoint",
        values="Invalid_Fraction",
        aggfunc="mean",
    )
    # Preserve requested keypoint order in the heatmap columns.
    pivot = pivot.reindex(columns=keypoint_order)
    # Scale figure height with trial count so row labels remain readable.
    fig_height = max(4, 0.25 * len(pivot) + 2)
    fig, ax = plt.subplots(figsize=(max(8, 0.8 * len(keypoint_order) + 4), fig_height))
    # Values are fractions, so a fixed 0-1 color scale is comparable across groups.
    image = ax.imshow(pivot.to_numpy(dtype=float), aspect="auto", vmin=0, vmax=1, cmap="viridis")
    ax.set_xticks(np.arange(len(pivot.columns)))
    ax.set_xticklabels(pivot.columns, rotation=45, ha="right")
    ax.set_yticks(np.arange(len(pivot.index)))
    ax.set_yticklabels(pivot.index)
    ax.set_title(f"{group_name}: Trial-keypoint invalid fraction")
    ax.set_xlabel("Keypoint")
    ax.set_ylabel("Trial")
    cbar = fig.colorbar(image, ax=ax)
    cbar.set_label("Invalid frame fraction")
    fig.tight_layout()
    _save_figure(fig, output_dir, f"{group_name}_invalid_fraction_heatmap.png")
    if not show:
        plt.close(fig)
    figures["invalid_fraction_heatmap"] = fig

    # Sensitivity plot shows how retained trial-keypoint fraction changes with QC cutoff.
    fig, ax = plt.subplots(figsize=figsize)
    for keypoint in keypoint_order:
        # Pull the invalid-fraction distribution for one keypoint.
        values = qc_summary_df.loc[
            qc_summary_df["Keypoint"] == keypoint,
            "Invalid_Fraction",
        ].dropna().to_numpy(dtype=float)
        if len(values) == 0:
            continue
        # Retention at each threshold is the fraction with invalid_fraction <= threshold.
        retained = [np.mean(values <= threshold) for threshold in threshold_values]
        ax.plot(threshold_values, retained, marker="o", linewidth=1.5, label=keypoint)
    # Mark the currently configured invalid-fraction threshold.
    ax.axvline(max_invalid_fraction, color="red", linestyle="--", linewidth=1)
    ax.set_xlabel("Allowed invalid-frame fraction")
    ax.set_ylabel("Retained trial-keypoint fraction")
    ax.set_ylim(-0.02, 1.02)
    ax.set_title(f"{group_name}: QC threshold sensitivity")
    ax.grid(True, alpha=0.25)
    ax.legend(loc="best", fontsize="small")
    fig.tight_layout()
    _save_figure(fig, output_dir, f"{group_name}_threshold_sensitivity.png")
    if not show:
        plt.close(fig)
    figures["threshold_sensitivity"] = fig

    return figures


def plot_2d_keypoint_xy_traces(
        group_name,
        fly,
        trial,
        projection_path=None,
        h5_path=None,
        keypoints=None,
        margin_s=0.2,
        window_mode="moc_mol",
        value_mode="change",
        projection_ylim=None,
        xyz_ylim=None,
        change_ylim=None,
        output_dir=None,
        show=True,
        figsize=(12, 12),
):
    """
    Plot 2D projection traces from all cameras with matching 3D trial data.

    Parameters
    ----------
    group_name : str
        Config key or display group name from group_config_new.
    fly, trial : int
        Metadata fly/trial numbers, using the same 1-based indexing as Excel.
    projection_path : str or Path, optional
        Root folder containing per-fly 2D projection H5 files.
    h5_path : str or Path, optional
        Backward-compatible single-camera path. If projection_path is omitted,
        its parent folder is searched for all six camera files.
    keypoints : sequence[str], optional
        Bodyparts to plot. Defaults to all bodyparts present in the H5 file.
    margin_s : float
        Seconds added before MOC and after MOL when window_mode="moc_mol".
    window_mode : {"moc_mol", "full"}
        Plot either the MOC-to-MOL window with margin, or the whole video.
    value_mode : {"change", "absolute"}
        Plot frame-to-frame coordinate changes or absolute coordinates.
    projection_ylim : None, number, or tuple
        Shared y-axis limit for the 2D x/y panels.
    xyz_ylim : None, number, or tuple
        Shared y-axis limit for the 3D xyz panel.
    change_ylim : None, number, or tuple
        Deprecated alias for projection_ylim.
    output_dir : str or Path, optional
        If provided, save one PNG per keypoint in this directory.
    show : bool
        If True, display figures. If False, close them after optional saving.
    figsize : tuple
        Matplotlib figure size for each keypoint.

    Returns
    -------
    list[matplotlib.figure.Figure]
        One 6x1 figure per plotted keypoint.
    """
    # Validate mode arguments before any file lookup.
    if window_mode not in {"moc_mol", "full"}:
        raise ValueError("window_mode must be 'moc_mol' or 'full'.")
    if value_mode not in {"change", "absolute"}:
        raise ValueError("value_mode must be 'change' or 'absolute'.")
    # Preserve the old argument name as an alias for projection y-limits.
    if projection_ylim is None and change_ylim is not None:
        projection_ylim = change_ylim
    # Backward-compatible mode starts from one H5 path and searches its parent folder.
    if projection_path is None:
        if h5_path is None:
            raise ValueError("Provide projection_path, or h5_path for backward-compatible lookup.")
        projection_path = Path(h5_path).parent

    # Resolve group metadata and event timing for the requested fly/trial.
    group_info = _resolve_group(group_name)
    moc, mol, fps = _read_trial_timing(group_info, fly, trial)
    # Locate the six per-camera 2D projection H5 files.
    camera_paths = _find_projection_h5_files(projection_path, fly, trial)
    # Read each camera file into a dataframe keyed by camera number.
    camera_dfs = {
        camera: _read_2d_projection_h5(path)
        for camera, path in camera_paths.items()
    }
    # Read the matching 3D kinematic CSV for reprojection error and camera-count panels.
    kine_df, _ = _read_3d_trial_dataframe(group_info, fly, trial)

    # Use the first camera file to define available keypoints.
    first_df = next(iter(camera_dfs.values()))
    available_keypoints = list(dict.fromkeys(first_df.columns.get_level_values("bodyparts")))
    if keypoints is None:
        # Plot every keypoint when no subset is requested.
        keypoints = available_keypoints
    else:
        # Fail early if the caller requests a keypoint absent from projection data.
        missing = [kp for kp in keypoints if kp not in available_keypoints]
        if missing:
            raise KeyError(f"Keypoints not found in H5 file: {missing}")

    # The 3D panels require xyz, reprojection error, and camera-count columns.
    missing_3d = []
    for keypoint in keypoints:
        for suffix in ("x", "y", "z", "error", "ncams"):
            column = f"{keypoint}_{suffix}"
            if column not in kine_df.columns:
                missing_3d.append(column)
    if missing_3d:
        raise KeyError(f"3D kinematic columns not found: {missing_3d}")

    # Limit plotting to frames present in both all 2D projections and the 3D data.
    max_projection_len = min(len(df) for df in camera_dfs.values())
    max_len = min(max_projection_len, len(kine_df))
    if window_mode == "moc_mol":
        # MOC/MOL mode expands the event interval by the requested margin in seconds.
        start_frame = max(0, int(round(moc - margin_s * fps)))
        stop_frame = min(max_len - 1, int(round(mol + margin_s * fps)))
    else:
        # Full mode plots every shared frame.
        start_frame = 0
        stop_frame = max_len - 1

    # Guard against inconsistent timing or traces too short for the requested window.
    if stop_frame < start_frame:
        raise ValueError(
            f"Invalid plotting window: start={start_frame}, stop={stop_frame}."
        )

    # X-axis reports absolute frame number rather than time relative to MOC.
    frame_index = np.arange(start_frame, stop_frame + 1)
    # Per-camera shades make camera 1 darkest and camera 6 lightest within each color family.
    red_colors = plt.cm.Reds(np.linspace(0.9, 0.35, 6))
    blue_colors = plt.cm.Blues(np.linspace(0.9, 0.35, 6))
    green_colors = plt.cm.Greens(np.linspace(0.9, 0.35, 6))

    # Create the output directory only if the caller asks to save figures.
    if output_dir is not None:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

    figures = []
    for keypoint in keypoints:
        # Store per-camera 2D traces separately so all cameras can be overlaid.
        x_traces = {}
        y_traces = {}
        confidence_traces = {}
        for camera, df in camera_dfs.items():
            # Extract absolute x/y projection coordinates for this keypoint and camera.
            x = _extract_2d_coord(df, keypoint, "x")
            y = _extract_2d_coord(df, keypoint, "y")
            # Slice either absolute coordinates or frame-to-frame changes.
            x_traces[camera] = _slice_series(x, start_frame, stop_frame, value_mode)
            y_traces[camera] = _slice_series(y, start_frame, stop_frame, value_mode)
            # Likelihood/confidence is optional in projection H5 files.
            confidence_traces[camera] = _extract_2d_coord_optional(
                df,
                keypoint,
                "likelihood",
                start_frame,
                stop_frame,
            )

        # Extract matching 3D coordinates in the same value mode as the 2D panels.
        x3d = _slice_series(kine_df[f"{keypoint}_x"], start_frame, stop_frame, value_mode)
        y3d = _slice_series(kine_df[f"{keypoint}_y"], start_frame, stop_frame, value_mode)
        z3d = _slice_series(kine_df[f"{keypoint}_z"], start_frame, stop_frame, value_mode)
        # Reprojection error and camera count are plotted as absolute QC channels.
        error = kine_df[f"{keypoint}_error"].astype(float).iloc[start_frame:stop_frame + 1]
        ncams = kine_df[f"{keypoint}_ncams"].astype(float).iloc[start_frame:stop_frame + 1]

        # Combine all 2D x/y traces to resolve shared y-limits for the projection panels.
        projection_values = np.concatenate([
            trace.to_numpy(dtype=float)
            for trace in list(x_traces.values()) + list(y_traces.values())
        ])
        # Combine 3D xyz traces to resolve shared y-limits for the 3D panel.
        xyz_values = np.concatenate([
            x3d.to_numpy(dtype=float),
            y3d.to_numpy(dtype=float),
            z3d.to_numpy(dtype=float),
        ])
        # Apply user limits or symmetric autoscaling for coordinate-change plots.
        resolved_projection_ylim = _resolve_axis_ylim(
            projection_ylim,
            projection_values,
            symmetric=value_mode == "change",
        )
        resolved_xyz_ylim = _resolve_axis_ylim(
            xyz_ylim,
            xyz_values,
            symmetric=value_mode == "change",
        )

        # The six panels are: 2D x, 2D y, confidence, 3D xyz, error, and camera count.
        fig, axes = plt.subplots(6, 1, figsize=figsize, sharex=True)
        coordinate_label = "change" if value_mode == "change" else "coordinate"
        fig.suptitle(
            f"{group_info.group_name} F{fly}T{trial} {keypoint} 2D projections and 3D data"
        )

        # Overlay the six camera traces in each 2D/confidence panel.
        for color_idx, camera in enumerate(sorted(camera_dfs)):
            axes[0].plot(
                frame_index,
                x_traces[camera].to_numpy(dtype=float),
                color=red_colors[color_idx],
                linewidth=1,
                label=f"Cam{camera}",
            )
            axes[1].plot(
                frame_index,
                y_traces[camera].to_numpy(dtype=float),
                color=blue_colors[color_idx],
                linewidth=1,
                label=f"Cam{camera}",
            )
            axes[2].plot(
                frame_index,
                confidence_traces[camera].to_numpy(dtype=float),
                color=green_colors[color_idx],
                linewidth=1,
                label=f"Cam{camera}",
            )

        # Label the projection panels according to absolute-vs-change mode.
        axes[0].set_ylabel("2D x" if value_mode == "absolute" else "2D dx")
        axes[1].set_ylabel("2D y" if value_mode == "absolute" else "2D dy")
        axes[2].set_ylabel("confidence")
        # Confidence scores are probabilities/likelihoods and should stay near 0-1.
        axes[2].set_ylim(-0.05, 1.05)

        # Plot 3D x/y/z together for direct comparison with 2D projection jumps.
        axes[3].plot(frame_index, x3d.to_numpy(dtype=float), color="#d62728", linewidth=1, label="x")
        axes[3].plot(frame_index, y3d.to_numpy(dtype=float), color="#1f77b4", linewidth=1, label="y")
        axes[3].plot(frame_index, z3d.to_numpy(dtype=float), color="black", linewidth=1, label="z")
        axes[3].set_ylabel("3D xyz" if value_mode == "absolute" else "3D dxyz")

        # Reprojection error helps identify frames where 3D reconstruction quality is poor.
        axes[4].plot(frame_index, error.to_numpy(dtype=float), color="#ff7f0e", linewidth=1)
        axes[4].set_ylabel("reproj. error")

        # Camera-count panel shows how many views contributed to each 3D point.
        axes[5].plot(
            frame_index,
            ncams.to_numpy(dtype=float),
            color="0.25",
            marker=".",
            markersize=2,
            linewidth=0.8,
        )
        axes[5].set_ylabel("3D ncams")
        axes[5].set_xlabel("Frame")

        # Add MOC/MOL markers and frame-aligned grids to every panel.
        for ax in axes:
            # Dashed line marks MOC; dotted line marks MOL.
            ax.axvline(moc, color="black", linestyle="--", linewidth=1)
            ax.axvline(mol, color="black", linestyle=":", linewidth=1)
            # Tick labels appear every 10 frames, while minor grid lines mark every frame.
            ax.xaxis.set_major_locator(MultipleLocator(10))
            ax.xaxis.set_minor_locator(MultipleLocator(1))
            ax.grid(True, which="major", axis="y", alpha=0.25)
            ax.grid(True, which="both", axis="x", alpha=0.18)
        # Apply shared 2D y-limits to both x and y projection panels.
        if resolved_projection_ylim is not None:
            axes[0].set_ylim(resolved_projection_ylim)
            axes[1].set_ylim(resolved_projection_ylim)
        # Apply shared 3D y-limits to the xyz panel.
        if resolved_xyz_ylim is not None:
            axes[3].set_ylim(resolved_xyz_ylim)
        # Keep camera and 3D-coordinate legends compact.
        axes[0].legend(loc="upper right", ncol=6, fontsize="small")
        axes[3].legend(loc="upper right", ncol=3, fontsize="small")

        # Tight layout reduces overlap between the six stacked panels.
        fig.tight_layout()

        if output_dir is not None:
            # Sanitize keypoint names so they are safe in filenames.
            safe_keypoint = str(keypoint).replace("/", "_").replace("\\", "_")
            fig.savefig(
                output_dir / (
                    f"{group_info.group_name}_F{fly}T{trial}_{safe_keypoint}_2D3D_{coordinate_label}.png"
                ),
                dpi=200,
            )
        if not show:
            # Close figures in batch mode to avoid accumulating open matplotlib windows.
            plt.close(fig)
        figures.append(fig)

    # Return figure handles for notebook display or downstream customization.
    return figures
